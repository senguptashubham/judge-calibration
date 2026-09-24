"""Task 1.8's vacuum test: does the judge express a preference when there is
no content difference to judge? 40 pairs of identical responses and 20 pairs
of empty responses, clean/P1/AB/greedy. Colab only (needs vllm, D17).

Writes {runs_dir}/vacuum.jsonl - calls.parquet's provenance fields plus a
`vacuum_type` ("identical"|"empty") - and the matching logprobs files.

Run: python -m src.vacuum_test --config configs/run.yaml
"""

import argparse
from pathlib import Path

from src.config import Config
from src.data import item_id
from src.judge import (
    VERDICT_SCHEMA,
    append_checkpoint,
    checkpoint_key,
    git_sha,
    load_completed_keys,
    load_full_items_df,
    logprobs_path,
    write_logprobs,
)
from src.perturb import vacuum_empty, vacuum_identical
from src.prompts import prompt_hash, render_prompt


def main(cfg: Config) -> None:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    checkpoint_path = Path(cfg.paths.runs_dir) / "vacuum.jsonl"
    items_df = load_full_items_df(cfg)

    # 60 distinct source items: 40 for identical pairs, 20 for empty pairs.
    sample = items_df.sample(n=60, random_state=cfg.seed)
    identical_rows = sample.iloc[:40]
    empty_rows = sample.iloc[40:]

    llm = LLM(model=cfg.judge_model)
    completed = load_completed_keys(checkpoint_path)
    sha = git_sha()
    vllm_version = __import__("vllm").__version__

    # Both sides get the same conversation, so order is irrelevant: AB only.
    calls = []  # (row, vacuum_type, conv_a, conv_b)
    for _, row in identical_rows.iterrows():
        calls.append((row, "identical", vacuum_identical(row["conversation_a"]), vacuum_identical(row["conversation_a"])))
    for _, row in empty_rows.iterrows():
        calls.append((row, "empty", vacuum_empty(row["conversation_a"]), vacuum_empty(row["conversation_a"])))

    pending = []
    for row, vacuum_type, conv_a, conv_b in calls:
        item = item_id(row["question_id"], row["model_a"], row["model_b"], row["turn"])
        key = checkpoint_key(item, "vacuum", "P1", "AB", 0)
        if key not in completed:
            pending.append((item, row, vacuum_type, conv_a, conv_b))

    print(f"{len(pending)} pending vacuum calls out of {len(calls)} total")

    batch_size = 32
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start : batch_start + batch_size]
        prompts = [
            render_prompt("P1", "AB", conv_a, conv_b, row["turn"]) for _, row, _, conv_a, conv_b in batch
        ]
        sampling_params = [
            SamplingParams(
                max_tokens=cfg.max_tokens,
                logprobs=cfg.logprobs,
                temperature=cfg.temperature_canonical,
                seed=cfg.seed,
                structured_outputs=StructuredOutputsParams(json=VERDICT_SCHEMA),
            )
            for _ in batch
        ]
        outputs = llm.generate(prompts, sampling_params)

        for (item, row, vacuum_type, _, _), output in zip(batch, outputs):
            completion = output.outputs[0]
            record = {
                "item_id": item,
                "vacuum_type": vacuum_type,
                "question_id": int(row["question_id"]),
                "model_a": row["model_a"],
                "model_b": row["model_b"],
                "turn": int(row["turn"]),
                "condition": "vacuum",
                "prompt_variant": "P1",
                "order": "AB",
                "sample_idx": 0,
                "seed": cfg.seed,
                "judge_model": cfg.judge_model,
                "vllm_version": vllm_version,
                "prompt_hash": prompt_hash("P1"),
                "git_sha": sha,
                "raw_output": completion.text,
                "n_prompt_tokens": len(output.prompt_token_ids),
                "n_out_tokens": len(completion.token_ids),
            }
            append_checkpoint(checkpoint_path, record)

            path = logprobs_path(cfg.paths.runs_dir, item, "vacuum", "P1", "AB", 0)
            write_logprobs(path, completion.token_ids, completion.logprobs)

    print("Done. Wrote", checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(Config.from_yaml(args.config))
