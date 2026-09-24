"""vLLM wrapper for the primary judge: batched generation, JSONL
append-checkpointing, resumability. Colab only (D17) - vllm is imported
inside _run_generation(), so the schedule and checkpoint logic stay
importable and testable without it.

This file saves raw materials only: `raw_output` plus provenance in the
checkpoint, and every call's full per-token logprobs (D4). Every derived
signal is computed later by src/parse.py (invariant 7).
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
from src.perturb import verbose_pad
from src.prompts import prompt_hash, render_prompt

# The structured-output schema every call is constrained to - prompts.py's
# JSON output contract. Shared with ablation_decoding.py and vacuum_test.py.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "verdict": {"type": "string", "enum": ["A", "B"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["reasoning", "verdict", "confidence"],
}


@dataclass(frozen=True)
class CallSpec:
    condition: str
    prompt_variant: str
    order: str
    sample_idx: int  # 0 = canonical greedy: temperature_canonical. >0: temperature_sc.


def call_schedule(config: Config) -> list[CallSpec]:
    """Every call one item needs - D19's 12 calls/item. Which (condition,
    variant) combinations exist is a frozen design decision (D19-D21:
    P2/P3 and sampling are clean/P1-only), so it is encoded here, not in
    config; k_sc and the temperatures do come from config.
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


def load_full_items_df(config: Config) -> "pd.DataFrame":
    """Every non-tie item, with its conversations joined back in from the
    raw votes (build_items() doesn't carry them). Shared by every harness.
    """
    from src.data import build_items, load_votes

    votes = load_votes(config.dataset)
    items_labels = build_items(votes, tie_policy=config.tie_policy)
    conv_cols = votes[
        ["question_id", "model_a", "model_b", "turn", "conversation_a", "conversation_b"]
    ].drop_duplicates(subset=["question_id", "model_a", "model_b", "turn"])
    items_df = items_labels.merge(conv_cols, on=["question_id", "model_a", "model_b", "turn"])
    return items_df[~items_df["is_tie"]]  # D2: no ground truth for tied items


def checkpoint_key(item: str, condition: str, prompt_variant: str, order: str, sample_idx: int) -> str:
    return f"{item}|{condition}|{prompt_variant}|{order}|{sample_idx}"


def load_completed_keys(checkpoint_path: Path) -> set[str]:
    """Keys already in the checkpoint, so a re-run skips them (invariant 9).
    A missing file means nothing is done yet.
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
    """Every (item, spec) pair not already in `completed`. Pure - no vllm,
    no I/O - so "running twice does not duplicate rows" is testable locally.
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
    """Restricts a schedule to some prompt variants, for scoped pilot runs.
    None means no restriction.
    """
    if prompt_variants is None:
        return specs
    return [s for s in specs if s.prompt_variant in prompt_variants]


def _conversations_for_condition(
    condition: str, model_a_conversation: list[dict], model_b_conversation: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Applies the condition's perturbation to both sides, before order is
    applied: "verbose" -> verbose_pad(), anything else unchanged. Separate
    and pure so the condition -> perturbation wiring is unit-tested.
    """
    if condition == "verbose":
        return verbose_pad(model_a_conversation), verbose_pad(model_b_conversation)
    return model_a_conversation, model_b_conversation


def _build_prompts(batch: list[tuple[str, "pd.Series", CallSpec]]) -> list[str]:
    """Full prompt text for a batch: perturbation, then order, then
    template. Pure, so the whole construction path is tested, not just its
    parts.
    """
    return [
        render_prompt(
            spec.prompt_variant,
            spec.order,
            *_conversations_for_condition(spec.condition, item_row["conversation_a"], item_row["conversation_b"]),
            item_row["turn"],
        )
        for _, item_row, spec in batch
    ]


def logprobs_path(runs_dir: str, item: str, condition: str, prompt_variant: str, order: str, sample_idx: int) -> Path:
    """Where one call's per-token logprobs are saved, keyed like checkpoint_key()."""
    key = checkpoint_key(item, condition, prompt_variant, order, sample_idx).replace("|", "_")
    return Path(runs_dir) / "logprobs" / f"{key}.jsonl.gz"


def write_logprobs(path: Path, token_ids: list[int], per_token_logprobs: list[dict]) -> None:
    """One call's full per-token top-K logprobs, gzipped, plus:
    - `token_ids`/`token_texts`: the generated sequence, which parse.py
      needs to locate the CoT/verdict boundary inside raw_output;
    - each candidate's decoded text, which parse.py needs to find which of
      the ~20 candidates at the verdict position are "A" and "B" (for p_a).
    Saving both means parse.py never needs a tokenizer (D17).

    Args:
      token_ids: the generated token id at each position.
      per_token_logprobs: vLLM's per-position dict (token_id -> object with
        `.logprob` and `.decoded_token`), same order and length.
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
    """The only function that touches vLLM. Runs `condition`'s pending
    calls (optionally restricted to `prompt_variants`) in batches,
    checkpointing each row and its logprobs as it completes.
    """
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model=config.judge_model)
    completed = load_completed_keys(checkpoint_path)
    sha = git_sha()
    vllm_version = __import__("vllm").__version__

    specs = [s for s in call_schedule(config) if s.condition == condition]
    specs = filter_schedule(specs, prompt_variants)
    pending = pending_calls(items_df, specs, completed)

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        prompts = _build_prompts(batch)
        sampling_params = [
            SamplingParams(
                max_tokens=config.max_tokens,
                logprobs=config.logprobs,
                temperature=config.temperature_canonical if spec.sample_idx == 0 else config.temperature_sc,
                seed=config.seed + spec.sample_idx,
                structured_outputs=StructuredOutputsParams(json=VERDICT_SCHEMA),
            )
            for _, _, spec in batch
        ]

        batch_start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        batch_elapsed_ms = (time.time() - batch_start_time) * 1000
        # vLLM runs the batch concurrently, so this is a batch average, not
        # per-request latency - fine for budgeting, not for anything finer.
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
        help="Limit to a seeded random sample of N items, for scoped pilot/smoke-test runs. "
        "Default: every non-tie item.",
    )
    parser.add_argument(
        "--prompt-variants",
        type=str,
        default=None,
        help="Comma-separated subset of prompt variants to run, e.g. 'P1'. Default: every variant.",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)

    items_df = load_full_items_df(cfg)

    if args.n_items is not None:
        # Seeded sample, not head(): head() would cluster on a few questions.
        items_df = items_df.sample(n=args.n_items, random_state=cfg.seed)

    prompt_variants = args.prompt_variants.split(",") if args.prompt_variants else None

    checkpoint_path = Path(cfg.paths.runs_dir) / f"judge_{args.condition}.jsonl"
    _run_generation(cfg, args.condition, checkpoint_path, items_df, prompt_variants=prompt_variants)
