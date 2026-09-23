"""vLLM wrapper for auto-j-13b (DECISIONS.md D28) - the purpose-built-judge
generalization arm (RQ7). Deliberately mirrors src/judge.py's structure
(native vLLM, not a custom HTTP server like src/judge_kev.py needed) far
more than it mirrors judge_kev.py, since auto-j-13b IS vLLM-native.

Scope, deliberately narrow, same principle as judge.py's own docstring and
the lesson D27 already paid for once (judge_kev.py's first call_kev()
curated a subset of fields and silently dropped `confidence`): this file
runs the model and saves raw materials ONLY - `raw_output` (the complete
generated text, never truncated or curated) plus every call's full
per-token top-20 logprobs (D4's 100%-coverage policy, reused unchanged via
src.judge.write_logprobs) - and computes no derived signal itself. Verdict
extraction, self-consistency, and the order-swap entropy analog are all
computed later by src/autoj_signals.py (task L4) from this saved data.
`logprobs=20` is captured for every call even though conf_lp is explicitly
deferred this pass (D28) - cheap insurance so a later conf_lp attempt needs
no GPU rerun, the same reasoning D4 already established for the primary
judge and the standing "capture the full output" rule this project adopted
after the judge_kev.py incident.

Key structural differences from judge.py, all because auto-j is NOT
schema-constrained the way the primary judge is:
- No StructuredOutputsParams/VERDICT_SCHEMA - auto-j's own trained format
  is free-text critique + an extractable "final decision is Response 1/2/
  Tie" line (ported verbatim from the authors' own constants_prompt.py/
  example.py, D28), and forcing JSON output would deviate from it.
- No `prompt_variant` axis - auto-j has one fixed prompt contract, not a
  P1/P2/P3 ensemble (D28). AutojCallSpec/checkpoint_key are 3-field-plus-
  sample_idx, not judge.py's 5-field version.
- A real `finish_reason` field is captured (vLLM's own CompletionOutput
  attribute) - judge.py doesn't need this since its small, schema-
  constrained JSON output essentially never truncates at max_tokens, but
  auto-j's free-text critique genuinely can run long and cut off before
  reaching its own decision line. Cheap to capture, directly diagnostic
  for L4b's turn=2 validity check and for triaging parse failures without
  guessing from text length alone.
- `turn` is retained as a real, load-bearing column (it already exists in
  judge.py's own row dict too) - RQ7's whole population is stratified by
  turn=1 vs turn=2 (D28), not by a coverage regime the way RQ6 was.

See TASKS.md task L3, DECISIONS.md D28, PLAN.md §8.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from src.data import item_id
from src.judge import append_checkpoint, git_sha, load_full_items_df, write_logprobs
from src.perturb import verbose_pad
from src.prompts import apply_order


@dataclasses.dataclass(frozen=True)
class AutojPaths:
    results_dir: str
    runs_dir: str
    calls_parquet: str
    items_parquet: str
    items_labels_parquet: str


@dataclasses.dataclass(frozen=True)
class AutojConfig:
    judge_model: str
    dataset: str
    tie_policy: str
    conditions: list[str]
    max_model_len: int
    gpu_memory_utilization: float
    temperature_canonical: float
    temperature_sc: float
    k_sc: int
    max_tokens: int
    top_p: float
    logprobs: int
    seed: int
    n_bins: int
    paths: AutojPaths

    @property
    def model_slug(self) -> str:
        # Same derivation as src/config.py::Config.model_slug (D26) -
        # duplicated rather than imported, same reasoning KevConfig already
        # established: Config's own __post_init__ validation (temperature_sc
        # > 0 would actually pass here, but "P1" in prompt_variants would
        # not - auto-j has no prompt-ensemble axis at all) doesn't apply to
        # this harness and shouldn't be worked around with a placeholder.
        name = self.judge_model.split("/")[-1]
        return name.lower().replace("-", "_")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AutojConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = dict(yaml.safe_load(f))
        raw["paths"] = AutojPaths(**raw["paths"])
        return cls(**raw)


# --- The verbatim auto-j prompt/parsing contract (D28) ---------------------
#
# Pulled via `curl` directly from github.com/GAIR-NLP/auto-j's own
# codes/usage/constants_prompt.py - not paraphrased, not improvised. My own
# first Colab smoke test used an improvised prompt instead of this real
# template, and its output read as self-contradictory - confirming this
# template must be used verbatim for any real data collection, never
# reworded, even lightly.

_PROMPT_INPUT_WO_SYSTEM = "[INST] {input} [/INST]"

_PAIRWISE_TIE_TEMPLATE = """You are assessing two submitted responses on a given user's query and judging which response is better or they are tied. Here is the data:

[BEGIN DATA]
***
[Query]: {prompt}
***
[Response 1]: {response}
***
[Response 2]: {response_another}
***
[END DATA]

Here are the instructions to assess and compare the two responses:

1. Pinpoint the key factors to distinguish these two responses.
2. Conclude your comparison by providing a final decision on which response is better, or they are tied. Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie". Ensure that your decision aligns coherently with the comprehensive evaluation and comparison you've provided."""


def autoj_prompt_hash() -> str:
    """Stable hash of auto-j's fixed instructional text (verbatim from the
    authors' own constants_prompt.py, D28). No variant argument, unlike
    src/prompts.py::prompt_hash(variant) - auto-j has one fixed contract,
    not a P1/P2/P3 ensemble. Changes only if the ported template text
    itself is edited.
    """
    fixed_text = _PROMPT_INPUT_WO_SYSTEM + _PAIRWISE_TIE_TEMPLATE
    return hashlib.sha256(fixed_text.encode("utf-8")).hexdigest()[:16]


def build_autoj_prompt(
    condition: str,
    order: str,
    model_a_conversation: list[dict],
    model_b_conversation: list[dict],
    turn: int,
) -> str:
    """Renders the full auto-j prompt for one call.

    Order via apply_order() (public, generic, no template text - reusing it
    doesn't touch anything frozen in prompts.py), then verbose_pad() if
    condition == "verbose" - same order-then-perturb structure
    judge_kev.py::build_state() already established for the second-judge-
    model case.

    Turn handling (D28's own novel design, flagged explicitly - this is
    what task L4b's parse-rate/spot-check validates, not asserted as
    obviously correct):
    - turn=1: direct fill. `prompt` = the shared user query (identical for
      both sides - same underlying MT-Bench item), `response`/
      `response_another` = each side's own single answer.
    - turn=2: auto-j's template has one flat `{prompt}` field with no
      native multi-turn support, but MT-Bench's turn=2 conversations carry
      [user, assistant, user, assistant] (confirmed against the live
      dataset in prompts.py's own docstring). `prompt` here carries ONLY
      the two shared, model-independent user turns - it never contains a
      model-specific answer, so it can't contaminate one candidate's
      context with the other's. Each side's own turn-1 answer is instead
      folded into ITS OWN `response`/`response_another` field as a
      two-part trajectory, so the judge sees each candidate's full
      two-turn history attributed to that candidate alone.
    """
    displayed_1, displayed_2 = apply_order(order, model_a_conversation, model_b_conversation)
    if condition == "verbose":
        displayed_1, displayed_2 = verbose_pad(displayed_1), verbose_pad(displayed_2)

    if turn == 1:
        prompt_field = displayed_1[0]["content"]
        response_1 = displayed_1[1]["content"]
        response_2 = displayed_2[1]["content"]
    elif turn == 2:
        prompt_field = (
            f"Earlier in the conversation, the user asked: '{displayed_1[0]['content']}'. "
            f"Now the user says: '{displayed_1[2]['content']}'"
        )
        response_1 = f"{displayed_1[1]['content']}\n\n[Continuing:] {displayed_1[3]['content']}"
        response_2 = f"{displayed_2[1]['content']}\n\n[Continuing:] {displayed_2[3]['content']}"
    else:
        raise ValueError(f"turn must be 1 or 2, got {turn!r}")

    user_msg = _PAIRWISE_TIE_TEMPLATE.format(prompt=prompt_field, response=response_1, response_another=response_2)
    return _PROMPT_INPUT_WO_SYSTEM.format(input=user_msg)


# --- Schedule / checkpointing (D28: no prompt_variant axis) -----------------


@dataclasses.dataclass(frozen=True)
class AutojCallSpec:
    condition: str
    order: str  # "AB" | "BA"
    sample_idx: int  # 0 = canonical greedy: temperature_canonical. >0: temperature_sc.


def call_schedule_autoj(config: AutojConfig) -> list[AutojCallSpec]:
    """Every (condition, order, sample_idx) combination one item needs, per
    D28's locked schedule: `clean` reuses D5/D6's own clean/P1 shape (2
    greedy across orders + k_sc sampled AB), `verbose` reuses D19's own
    verbose/P1 shape (2 greedy across orders, no sampling) - the primary
    judge's own already-settled asymmetry, not a new one invented here.
    """
    specs = [
        AutojCallSpec("clean", "AB", 0),
        AutojCallSpec("clean", "BA", 0),
    ]
    specs += [AutojCallSpec("clean", "AB", i) for i in range(1, config.k_sc + 1)]
    if "verbose" in config.conditions:
        specs.append(AutojCallSpec("verbose", "AB", 0))
        specs.append(AutojCallSpec("verbose", "BA", 0))
    return specs


def checkpoint_key(item: str, condition: str, order: str, sample_idx: int) -> str:
    return f"{item}|{condition}|{order}|{sample_idx}"


def load_completed_keys(checkpoint_path: Path) -> set[str]:
    """Same resumability guarantee as judge.py's own version (CLAUDE.md
    invariant 9), against this module's own 4-field key (no
    prompt_variant). Not reusing judge.py::load_completed_keys() directly -
    its key construction is hard-coded to the 5-field shape.
    """
    completed: set[str] = set()
    if not checkpoint_path.exists():
        return completed
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed.add(checkpoint_key(row["item_id"], row["condition"], row["order"], row["sample_idx"]))
    return completed


def pending_calls(
    items_df: "pd.DataFrame", specs: list[AutojCallSpec], completed: set[str]
) -> list[tuple[str, "pd.Series", AutojCallSpec]]:
    """Same shape as judge.py/judge_kev.py's own pending_calls() - a pure
    function, no vllm, so resumability is directly testable without a GPU.
    """
    pending: list[tuple[str, "pd.Series", AutojCallSpec]] = []
    for _, item_row in items_df.iterrows():
        item = item_id(item_row["question_id"], item_row["model_a"], item_row["model_b"], item_row["turn"])
        for spec in specs:
            key = checkpoint_key(item, spec.condition, spec.order, spec.sample_idx)
            if key not in completed:
                pending.append((item, item_row, spec))
    return pending


def logprobs_path_autoj(runs_dir: str, item: str, condition: str, order: str, sample_idx: int) -> Path:
    """Where one call's full per-token logprobs get saved - same
    write_logprobs() payload shape as judge.py (imported unchanged), just
    keyed against this module's own 4-field checkpoint_key.
    """
    key = checkpoint_key(item, condition, order, sample_idx).replace("|", "_")
    return Path(runs_dir) / "logprobs" / f"{key}.jsonl.gz"


def split_by_prompt_length(
    pending: list[tuple[str, "pd.Series", AutojCallSpec]],
    token_counter,
    max_prompt_tokens: int,
) -> tuple[list[tuple[str, "pd.Series", AutojCallSpec, str, int]], list[dict]]:
    """Splits `pending` into (runnable, skipped_rows) by each call's REAL
    rendered-prompt token length, checked BEFORE any generation call - a
    single over-length prompt crashes vLLM's WHOLE batch call, it doesn't
    just fail that one request (confirmed empirically, 23 Sep 2026: the
    first real Colab run hit exactly this, VLLMValidationError, after a
    prompt exceeded the model's 8,192-token context - traced to auto-j's
    own tokenizer plus this module's turn=2 flattening design compounding
    verbose padding across BOTH folded turns; real measured impact:
    4.75% of rows overall, ~17.2% of verbose/turn=2 specifically - full
    numbers in DECISIONS.md D28's 23 Sep amendment).

    `runnable` entries carry the already-rendered prompt and its real token
    count, computed once here, not re-rendered later - `skipped_rows` are
    ready-to-append checkpoint dicts, missing only the provenance fields
    (judge_model/vllm_version/prompt_hash/git_sha/seed) the caller adds,
    same "skipped=True, skip_reason=..." shape judge_kev.py::_run_calls()
    already established for kev-8b's own max_state_tokens check.

    `token_counter` is injected (a plain `Callable[[str], int]`) rather
    than this function loading a tokenizer itself, so the split logic
    stays testable without downloading a real tokenizer (D17's own
    "everything except the one function that touches the external thing
    stays locally testable" principle, applied here to `transformers` the
    same way it's already applied to `vllm`).
    """
    runnable: list[tuple[str, "pd.Series", AutojCallSpec, str, int]] = []
    skipped_rows: list[dict] = []
    for item, item_row, spec in pending:
        prompt = build_autoj_prompt(
            spec.condition, spec.order, item_row["conversation_a"], item_row["conversation_b"], int(item_row["turn"])
        )
        n_prompt_tokens = token_counter(prompt)
        if n_prompt_tokens > max_prompt_tokens:
            skipped_rows.append(
                {
                    "item_id": item,
                    "question_id": int(item_row["question_id"]),
                    "category": item_row.get("category"),
                    "model_a": item_row["model_a"],
                    "model_b": item_row["model_b"],
                    "turn": int(item_row["turn"]),
                    "condition": spec.condition,
                    "order": spec.order,
                    "sample_idx": spec.sample_idx,
                    "skipped": True,
                    "skip_reason": "over_max_prompt_tokens",
                    "n_prompt_tokens": n_prompt_tokens,
                }
            )
            continue
        runnable.append((item, item_row, spec, prompt, n_prompt_tokens))
    return runnable, skipped_rows


def _run_generation(
    config: AutojConfig,
    checkpoint_path: Path,
    items_df: "pd.DataFrame",
    batch_size: int = 32,
) -> None:
    """The only function in this module that touches vLLM (or downloads a
    tokenizer). Kept separate so everything above stays importable/
    testable without vllm/transformers installed (D17), same split
    judge.py/judge_kev.py already use.

    No StructuredOutputsParams here, deliberately (D28) - auto-j's own
    trained format is free text, and forcing JSON output would deviate
    from it the same way it would have for kev-8b.

    `max_prompt_tokens = config.max_model_len - config.max_tokens` -
    reserves the full generation budget for every accepted call rather
    than letting borderline-length items get truncated output near the
    context boundary (the "skip at 7,168, not 8,192" decision, D28).
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(config.judge_model)
    max_prompt_tokens = config.max_model_len - config.max_tokens

    llm = LLM(
        model=config.judge_model,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    completed = load_completed_keys(checkpoint_path)
    sha = git_sha()
    vllm_version = __import__("vllm").__version__

    specs = call_schedule_autoj(config)
    pending = pending_calls(items_df, specs, completed)

    runnable, skipped_rows = split_by_prompt_length(pending, lambda p: len(tok.encode(p)), max_prompt_tokens)
    for row in skipped_rows:
        row.update(
            {
                "seed": config.seed + row["sample_idx"],
                "judge_model": config.judge_model,
                "vllm_version": vllm_version,
                "prompt_hash": autoj_prompt_hash(),
                "git_sha": sha,
            }
        )
        append_checkpoint(checkpoint_path, row)

    for batch_start in range(0, len(runnable), batch_size):
        batch = runnable[batch_start : batch_start + batch_size]
        prompts = [b[3] for b in batch]
        sampling_params = [
            SamplingParams(
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                logprobs=config.logprobs,
                temperature=config.temperature_canonical if spec.sample_idx == 0 else config.temperature_sc,
                seed=config.seed + spec.sample_idx,
            )
            for _, _, spec, _, _ in batch
        ]

        batch_start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        batch_elapsed_ms = (time.time() - batch_start_time) * 1000
        # Batch-averaged, not true per-request latency - same caveat
        # judge.py's own _run_generation documents, same reason (vLLM
        # processes the whole batch concurrently).
        avg_latency_ms = batch_elapsed_ms / len(batch)

        for (item, item_row, spec, _, n_prompt_tokens), output in zip(batch, outputs):
            completion = output.outputs[0]

            row = {
                "item_id": item,
                "question_id": int(item_row["question_id"]),
                "category": item_row.get("category"),
                "model_a": item_row["model_a"],
                "model_b": item_row["model_b"],
                "turn": int(item_row["turn"]),
                "condition": spec.condition,
                "order": spec.order,
                "sample_idx": spec.sample_idx,
                "seed": config.seed + spec.sample_idx,
                "judge_model": config.judge_model,
                "vllm_version": vllm_version,
                "prompt_hash": autoj_prompt_hash(),
                "git_sha": sha,
                "skipped": False,
                "skip_reason": None,
                "raw_output": completion.text,
                "finish_reason": completion.finish_reason,
                "n_prompt_tokens": n_prompt_tokens,
                "n_out_tokens": len(completion.token_ids),
                "latency_ms": avg_latency_ms,
            }
            append_checkpoint(checkpoint_path, row)

            path = logprobs_path_autoj(config.paths.runs_dir, item, spec.condition, spec.order, spec.sample_idx)
            write_logprobs(path, completion.token_ids, completion.logprobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        help="Limit to a seeded random sample of N items - for scoped pilot/smoke-test runs "
        "(mirrors task 1.6's 20-item pilot / K3a's own precedent). Default: no limit, every non-tie item.",
    )
    args = parser.parse_args()

    cfg = AutojConfig.from_yaml(args.config)
    items_df = load_full_items_df(cfg)  # type: ignore[arg-type]
    if args.n_items is not None:
        items_df = items_df.sample(n=args.n_items, random_state=cfg.seed)

    checkpoint_path = Path(cfg.paths.runs_dir) / "autoj.jsonl"
    _run_generation(cfg, checkpoint_path, items_df)
