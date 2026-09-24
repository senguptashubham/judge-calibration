"""Task 4.5's decoding ablation: constrained (JSON-schema guided decoding)
vs free-form generation on the same N items and the same single call
(clean/P1/AB/greedy). Does constraining the format change the verdict, or
only the absolute logprobs (D25)? Colab only (D17).

`python -m src.ablation_decoding --config configs/run.yaml --n-items 100`

Writes {runs_dir}/ablation_decoding.jsonl, one row per (item,
decoding_mode). Both arms are later scored by the unmodified strict parser
in parse.py - whether free-form text still satisfies it is exactly what
the parse rate measures.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.config import Config
from src.judge import VERDICT_SCHEMA, CallSpec, _build_prompts, git_sha, load_full_items_df
from src.prompts import prompt_hash

DECODING_MODES = ("constrained", "free_form")


def sample_ablation_items(config: Config, n_items: int) -> pd.DataFrame:
    """Seeded random sample of non-tie items - random rather than head(),
    so the sample isn't clustered on a few question_ids.
    """
    items_df = load_full_items_df(config)
    return items_df.sample(n=n_items, random_state=config.seed)


def load_completed_ablation_keys(checkpoint_path: Path) -> set[str]:
    """(item_id, decoding_mode) pairs already written. Its own key: only
    decoding_mode varies here, and it isn't one of judge.py's key fields.
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
            completed.add(f"{row['item_id']}|{row['decoding_mode']}")
    return completed


def run_ablation(config: Config, n_items: int, checkpoint_path: Path, batch_size: int = 64) -> None:
    """Generates both decoding arms and appends each row as it completes
    (resumable: re-run the same command after a disconnect).
    """
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    from src.data import item_id as compute_item_id

    items_df = sample_ablation_items(config, n_items)
    spec = CallSpec("clean", "P1", "AB", 0)
    batch = [
        (compute_item_id(row["question_id"], row["model_a"], row["model_b"], row["turn"]), row, spec)
        for _, row in items_df.iterrows()
    ]

    prompts = _build_prompts(batch)

    llm = LLM(model=config.judge_model)
    completed = load_completed_ablation_keys(checkpoint_path)
    sha = git_sha()
    vllm_version = __import__("vllm").__version__

    for mode in DECODING_MODES:
        pending = [
            (item, item_row, prompt)
            for (item, item_row, _), prompt in zip(batch, prompts)
            if f"{item}|{mode}" not in completed
        ]
        if not pending:
            continue

        for batch_start in range(0, len(pending), batch_size):
            sub_batch = pending[batch_start : batch_start + batch_size]
            sub_prompts = [p for _, _, p in sub_batch]

            structured_outputs = StructuredOutputsParams(json=VERDICT_SCHEMA) if mode == "constrained" else None
            sampling_params = SamplingParams(
                max_tokens=config.max_tokens,
                logprobs=config.logprobs,
                temperature=config.temperature_canonical,
                seed=config.seed,
                structured_outputs=structured_outputs,
            )

            batch_start_time = time.time()
            outputs = llm.generate(sub_prompts, sampling_params)
            batch_elapsed_ms = (time.time() - batch_start_time) * 1000
            avg_latency_ms = batch_elapsed_ms / len(sub_batch)

            for (item, item_row, _), output in zip(sub_batch, outputs):
                completion = output.outputs[0]
                row = {
                    "item_id": item,
                    "question_id": int(item_row["question_id"]),
                    "decoding_mode": mode,
                    "condition": "clean",
                    "prompt_variant": "P1",
                    "order": "AB",
                    "sample_idx": 0,
                    "seed": config.seed,
                    "judge_model": config.judge_model,
                    "vllm_version": vllm_version,
                    "prompt_hash": prompt_hash("P1"),
                    "git_sha": sha,
                    "raw_output": completion.text,
                    "n_prompt_tokens": len(output.prompt_token_ids),
                    "n_out_tokens": len(completion.token_ids),
                    "latency_ms": avg_latency_ms,
                }
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-items", type=int, default=100)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    ckpt_path = Path(cfg.paths.runs_dir) / "ablation_decoding.jsonl"
    run_ablation(cfg, args.n_items, ckpt_path)
    print(f"Wrote ablation checkpoint to {ckpt_path}")
