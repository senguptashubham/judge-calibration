"""RQ7: vLLM wrapper for auto-j-13b (D28). Mirrors judge.py's structure,
with three differences, all because auto-j is not schema-constrained:
- no structured-output schema: auto-j's trained format is a free-text
  critique ending in "So, the final decision is Response 1 / Response 2 /
  Tie", and forcing JSON would move it off that format;
- no prompt_variant axis: one fixed prompt contract, so checkpoint keys are
  (item, condition, order, sample_idx);
- `finish_reason` is saved, because a free-text critique can hit
  max_tokens before reaching its decision line.

Saves raw materials only - the complete `raw_output`, `finish_reason`, and
every call's full top-20 logprobs (kept even though no logprob signal is
computed yet, so one could be added without a GPU re-run). Verdicts and
signals come later, from src/autoj_signals.py.
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
        # Same derivation as Config.model_slug (D26); Config itself would
        # reject this harness for having no "P1" prompt variant.
        name = self.judge_model.split("/")[-1]
        return name.lower().replace("-", "_")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AutojConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = dict(yaml.safe_load(f))
        raw["paths"] = AutojPaths(**raw["paths"])
        return cls(**raw)


# --- The auto-j prompt contract, verbatim from the authors' ---------------
# codes/usage/constants_prompt.py (D28). Never reword it: an improvised
# version produced self-contradictory critiques in the first smoke test.

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
    """Hash of auto-j's fixed template text; changes only if it is edited."""
    fixed_text = _PROMPT_INPUT_WO_SYSTEM + _PAIRWISE_TIE_TEMPLATE
    return hashlib.sha256(fixed_text.encode("utf-8")).hexdigest()[:16]


def build_autoj_prompt(
    condition: str,
    order: str,
    model_a_conversation: list[dict],
    model_b_conversation: list[dict],
    turn: int,
) -> str:
    """The full auto-j prompt for one call: order first (apply_order), then
    verbose_pad() if condition == "verbose".

    auto-j's template has one flat {prompt} field, so turn=2 needs a
    flattening of this project's own design (D28):
    - turn=1: {prompt} = the user query; each response field = that side's
      answer.
    - turn=2: {prompt} carries only the two user turns, which both sides
      share. Each side's own turn-1 answer is folded into its own response
      field ("<turn-1 answer>\\n\\n[Continuing:] <turn-2 answer>"), so no
      model-specific text ever appears in the shared field.
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


# --- Schedule / checkpointing -----------------------------------------------


@dataclasses.dataclass(frozen=True)
class AutojCallSpec:
    condition: str
    order: str  # "AB" | "BA"
    sample_idx: int  # 0 = canonical greedy: temperature_canonical. >0: temperature_sc.


def call_schedule_autoj(config: AutojConfig) -> list[AutojCallSpec]:
    """8 calls per item, the primary judge's P1 shape (D19): clean gets 2
    greedy (both orders) + k_sc sampled AB; verbose gets 2 greedy only.
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
    """Keys already in the checkpoint (invariant 9), using this module's
    4-field key.
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
    """Every (item, spec) pair not already completed - pure, testable without a GPU."""
    pending: list[tuple[str, "pd.Series", AutojCallSpec]] = []
    for _, item_row in items_df.iterrows():
        item = item_id(item_row["question_id"], item_row["model_a"], item_row["model_b"], item_row["turn"])
        for spec in specs:
            key = checkpoint_key(item, spec.condition, spec.order, spec.sample_idx)
            if key not in completed:
                pending.append((item, item_row, spec))
    return pending


def logprobs_path_autoj(runs_dir: str, item: str, condition: str, order: str, sample_idx: int) -> Path:
    """Where one call's logprobs are saved (judge.py's write_logprobs() format)."""
    key = checkpoint_key(item, condition, order, sample_idx).replace("|", "_")
    return Path(runs_dir) / "logprobs" / f"{key}.jsonl.gz"


def split_by_prompt_length(
    pending: list[tuple[str, "pd.Series", AutojCallSpec]],
    token_counter,
    max_prompt_tokens: int,
) -> tuple[list[tuple[str, "pd.Series", AutojCallSpec, str, int]], list[dict]]:
    """Splits pending calls into (runnable, skipped_rows) by real prompt
    length, before any generation: one over-length prompt makes vLLM reject
    the WHOLE batch. In the real run 4.75% of calls were over, almost all
    verbose/turn=2, where padding applies to both folded turns (D28).

    `runnable` entries carry the rendered prompt and its token count.
    `skipped_rows` are checkpoint rows with skipped=True, missing only the
    provenance fields the caller adds. `token_counter` (str -> int) is
    injected so this is testable without downloading a tokenizer.
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
    """The only function that touches vLLM or downloads a tokenizer.

    The prompt ceiling is max_model_len - max_tokens (7,168), reserving the
    full generation budget for every accepted call instead of letting
    near-limit prompts truncate their output.
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
        avg_latency_ms = batch_elapsed_ms / len(batch)  # batch average, as in judge.py

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
        help="Limit to a seeded random sample of N items, for scoped smoke-test runs. "
        "Default: every non-tie item.",
    )
    args = parser.parse_args()

    cfg = AutojConfig.from_yaml(args.config)
    items_df = load_full_items_df(cfg)  # type: ignore[arg-type]
    if args.n_items is not None:
        items_df = items_df.sample(n=args.n_items, random_state=cfg.seed)

    checkpoint_path = Path(cfg.paths.runs_dir) / "autoj.jsonl"
    _run_generation(cfg, checkpoint_path, items_df)
