"""vLLM wrapper: batched generation, JSONL append-checkpointing, resumability.
Only runs on Colab (DECISIONS.md D17) - `vllm` is imported lazily, inside
the one function that needs it, so this module stays importable (and its
schedule/checkpoint logic stays locally testable) without vllm installed at
all. See TASKS.md task 1.4, DECISIONS.md D4, D6, D19.

Scope, deliberately narrow: this file runs the model and saves raw
materials - `raw_output` text plus every call's full per-token logprobs
(D4, amended 4 Sep 2026: 100% coverage, not a 10% sample) - and nothing
else. It computes no derived signal itself. `verdict`, `verbalized_conf`,
`verdict_token_logprob`, `p_a`, and the CoT logprob aggregates are all
computed later by `src/parse.py` (task 1.5) from the saved data, keeping
every bit of "turn raw model output into a signal" logic in the one file
CLAUDE.md invariant 7 says it belongs in. This only works because 100% of
calls now keep their full logprobs on disk - under the old 10%-sample
design, the raw data would have vanished for 90% of rows before parse.py
ever ran, which is why an earlier version of this file computed those
fields itself. See DECISIONS.md D4's 4 Sep 2026 amendment for the full
reasoning.
"""

import argparse
import gzip
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import Config
from src.data import item_id
from src.prompts import prompt_hash, render_prompt

# The per-item call schedule (DECISIONS.md D19) - 12 calls/item. Encoded as
# logic, not config, because which (condition, variant) combos exist at all
# is a frozen design decision (D19/D20/D21: P2/P3 and self-consistency
# sampling only exist for clean/P1), not a tunable number. k_sc and the two
# temperatures *are* tunable and come from config.


@dataclass(frozen=True)
class CallSpec:
    condition: str
    prompt_variant: str
    order: str
    sample_idx: int  # 0 = canonical greedy: temperature_canonical. >0: temperature_sc.


def call_schedule(config: Config) -> list[CallSpec]:
    """Every (condition, prompt_variant, order, sample_idx) combination one
    item needs, per D19's schedule table. Independent of any specific item -
    the same schedule applies to every row in items_labels.parquet.
    """
    specs = [
        CallSpec("clean", "P1", "AB", 0),
        CallSpec("clean", "P1", "BA", 0),
    ]
    specs += [CallSpec("clean", "P1", "AB", i) for i in range(1, config.k_sc + 1)]
    for variant in ("P2", "P3"):
        specs.append(CallSpec("clean", variant, "AB", 0))
        specs.append(CallSpec("clean", variant, "BA", 0))
    if "verbose" in config.conditions:
        specs.append(CallSpec("verbose", "P1", "AB", 0))
        specs.append(CallSpec("verbose", "P1", "BA", 0))
    return specs


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def checkpoint_key(item: str, condition: str, prompt_variant: str, order: str, sample_idx: int) -> str:
    return f"{item}|{condition}|{prompt_variant}|{order}|{sample_idx}"


def load_completed_keys(checkpoint_path: Path) -> set[str]:
    """Reads a JSONL checkpoint and returns the set of already-completed
    call keys, so a re-run can skip them (CLAUDE.md invariant 9). Missing
    file means nothing is completed yet, not an error.
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
            completed.add(
                checkpoint_key(
                    row["item_id"], row["condition"], row["prompt_variant"], row["order"], row["sample_idx"]
                )
            )
    return completed


def append_checkpoint(checkpoint_path: Path, row: dict) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def pending_calls(items_df, specs: list[CallSpec], completed: set[str]) -> list[tuple[str, "pd.Series", CallSpec]]:
    """Cross-joins every item in `items_df` with every `CallSpec` in
    `specs`, minus whatever's already in `completed` (from
    `load_completed_keys`). Pulled out as its own pure function - no vllm,
    no I/O - specifically so the resumability guarantee (CLAUDE.md
    invariant 9: "running twice does not duplicate rows") is directly
    testable without a GPU or vllm installed.
    """
    pending: list[tuple[str, "pd.Series", CallSpec]] = []
    for _, item_row in items_df.iterrows():
        item = item_id(item_row["question_id"], item_row["model_a"], item_row["model_b"], item_row["turn"])
        for spec in specs:
            key = checkpoint_key(item, spec.condition, spec.prompt_variant, spec.order, spec.sample_idx)
            if key not in completed:
                pending.append((item, item_row, spec))
    return pending


def filter_schedule(specs: list[CallSpec], prompt_variants: list[str] | None) -> list[CallSpec]:
    """Restricts a call schedule to a subset of prompt variants - what lets
    a pilot/smoke-test run (task 1.6: 20 items, clean/P1 only) stay scoped
    down instead of accidentally launching the full D19 schedule against
    every item. `None` means no restriction (every configured variant runs,
    the normal case for the real W2 run).
    """
    if prompt_variants is None:
        return specs
    return [s for s in specs if s.prompt_variant in prompt_variants]


def logprobs_path(runs_dir: str, item: str, condition: str, prompt_variant: str, order: str, sample_idx: int) -> Path:
    """Where one call's full per-token logprobs get saved (D4, amended
    4 Sep 2026: every call, not a 10% sample - directory renamed from
    `logprobs_sample/` to `logprobs/` to match). Keyed identically to
    `checkpoint_key()` so `parse.py` can look up the matching file for any
    `calls.parquet` row.
    """
    key = checkpoint_key(item, condition, prompt_variant, order, sample_idx).replace("|", "_")
    return Path(runs_dir) / "logprobs" / f"{key}.jsonl.gz"


def write_logprobs(path: Path, token_ids: list[int], per_token_logprobs: list[dict]) -> None:
    """Full per-token top-K logprobs for one call, gzipped, PLUS which token
    was actually generated at each position (`token_ids`/`token_texts`) AND
    the decoded text of every top-K *candidate*, not just the chosen one.

    Both are necessary, for different reasons:
    - `token_ids`/`token_texts` (the chosen sequence) is what
      split_cot_and_verdict_tokens() needs to find the CoT/verdict boundary
      in `raw_output` - without it parse.py would have no way to reconstruct
      which generated token corresponds to which piece of text.
    - Per-candidate decoded text (inside `token_logprobs` itself) is what
      p_a needs: to renormalize P(A) vs P(B) at the verdict position,
      parse.py must find *which* of the ~20 candidate token ids there
      decode to the literal text "A" and "B" - the chosen token's own text
      alone doesn't tell you that for the *other* candidate.

    Saving both here, rather than parse.py re-tokenizing `raw_output` or
    the literal strings "A"/"B" itself, keeps parse.py from needing a
    tokenizer/transformers dependency at all (D17 - parse.py stays a
    light, local-only module).

    Args:
      token_ids: the actually-generated token id at each position, in order.
      per_token_logprobs: vLLM's own per-position dict (token_id -> object
        with `.logprob` and `.decoded_token`), one dict per generated
        token, in generation order - same order/length as token_ids.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        record = {
            "token_ids": list(token_ids),
            "token_texts": [position[tid].decoded_token for tid, position in zip(token_ids, per_token_logprobs)],
            "token_logprobs": [
                {
                    token_id: {"logprob": lp.logprob, "decoded_token": lp.decoded_token}
                    for token_id, lp in position.items()
                }
                for position in per_token_logprobs
            ],
        }
        f.write(json.dumps(record) + "\n")


def _run_generation(
    config: Config,
    condition: str,
    checkpoint_path: Path,
    items_df,
    batch_size: int = 32,
    prompt_variants: list[str] | None = None,
) -> None:
    """The only function in this module that touches vLLM. Kept separate so
    everything above stays importable/testable without vllm installed
    (DECISIONS.md D17 - vllm only exists in Colab's venv).

    `prompt_variants`: passed straight to filter_schedule() - restricts
    which variants actually run, for scoped pilot/smoke-test invocations
    (task 1.6). `None` runs the full D19 schedule, the normal case.

    NOTE: verify `StructuredOutputsParams`'s exact JSON-schema keyword
    against the installed vllm==0.28.0 build before the first real run
    (`from vllm.sampling_params import StructuredOutputsParams; help(...)`)
    - this couldn't be checked locally since vllm isn't installed here.
    """
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    verdict_schema = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "verdict": {"type": "string", "enum": ["A", "B"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["reasoning", "verdict", "confidence"],
    }

    llm = LLM(model=config.judge_model)
    completed = load_completed_keys(checkpoint_path)
    sha = git_sha()
    vllm_version = __import__("vllm").__version__

    specs = [s for s in call_schedule(config) if s.condition == condition]
    specs = filter_schedule(specs, prompt_variants)
    pending = pending_calls(items_df, specs, completed)

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        prompts = [
            render_prompt(
                spec.prompt_variant,
                spec.order,
                item_row["conversation_a"],
                item_row["conversation_b"],
                item_row["turn"],
            )
            for _, item_row, spec in batch
        ]
        sampling_params = [
            SamplingParams(
                max_tokens=config.max_tokens,
                logprobs=config.logprobs,
                temperature=config.temperature_canonical if spec.sample_idx == 0 else config.temperature_sc,
                seed=config.seed + spec.sample_idx,
                structured_outputs=StructuredOutputsParams(json=verdict_schema),
            )
            for _, _, spec in batch
        ]

        batch_start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        batch_elapsed_ms = (time.time() - batch_start_time) * 1000
        # Batch-averaged, not true per-request latency - vLLM processes the
        # whole batch concurrently, so individual request times aren't
        # cleanly separable from one wall-clock measurement around the call.
        # Good enough for GPU-budget extrapolation (task 1.6's DoD), not a
        # latency SLA - documented as an average, not claimed as precise.
        avg_latency_ms = batch_elapsed_ms / len(batch)

        for (item, item_row, spec), output in zip(batch, outputs):
            completion = output.outputs[0]

            row = {
                "item_id": item,
                "question_id": int(item_row["question_id"]),
                "category": item_row.get("category"),
                "model_a": item_row["model_a"],
                "model_b": item_row["model_b"],
                "turn": int(item_row["turn"]),
                "condition": spec.condition,
                "prompt_variant": spec.prompt_variant,
                "order": spec.order,
                "sample_idx": spec.sample_idx,
                "seed": config.seed + spec.sample_idx,
                "judge_model": config.judge_model,
                "vllm_version": vllm_version,
                "prompt_hash": prompt_hash(spec.prompt_variant),
                "git_sha": sha,
                "raw_output": completion.text,
                "n_prompt_tokens": len(output.prompt_token_ids),
                "n_out_tokens": len(completion.token_ids),
                "latency_ms": avg_latency_ms,
            }
            append_checkpoint(checkpoint_path, row)

            path = logprobs_path(
                config.paths.runs_dir, item, spec.condition, spec.prompt_variant, spec.order, spec.sample_idx
            )
            write_logprobs(path, completion.token_ids, completion.logprobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True, choices=["clean", "verbose"])
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        help="Limit to a seeded random sample of N items - for scoped pilot/smoke-test runs "
        "(task 1.6's 20-item pilot). Default: no limit, every non-tie item.",
    )
    parser.add_argument(
        "--prompt-variants",
        type=str,
        default=None,
        help="Comma-separated subset of prompt variants to run, e.g. 'P1' for task 1.6's "
        "clean/P1-only pilot. Default: every variant in the schedule.",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    from src.data import build_items, load_votes

    votes = load_votes(cfg.dataset)
    items_labels = build_items(votes, tie_policy=cfg.tie_policy)
    # build_items() doesn't carry conversation_a/conversation_b - join them
    # back in from votes (one row per (question_id, model_a, model_b, turn)
    # is enough, conversation content is identical across a group's votes).
    conv_cols = votes[
        ["question_id", "model_a", "model_b", "turn", "conversation_a", "conversation_b"]
    ].drop_duplicates(subset=["question_id", "model_a", "model_b", "turn"])
    items_df = items_labels.merge(conv_cols, on=["question_id", "model_a", "model_b", "turn"])
    items_df = items_df[~items_df["is_tie"]]  # D2: no ground truth for tied items

    if args.n_items is not None:
        # Seeded, not the first N - a naive head() risks clustering on a
        # handful of question_ids (each has multiple model_a/model_b
        # pairs), giving a pilot with far less category diversity than a
        # random sample of the same size.
        items_df = items_df.sample(n=args.n_items, random_state=cfg.seed)

    prompt_variants = args.prompt_variants.split(",") if args.prompt_variants else None

    checkpoint_path = Path(cfg.paths.runs_dir) / f"judge_{args.condition}.jsonl"
    _run_generation(cfg, args.condition, checkpoint_path, items_df, prompt_variants=prompt_variants)
