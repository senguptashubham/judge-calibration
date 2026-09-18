"""Task 4.5's decoding ablation: constrained (JSON-schema guided
decoding) vs. free-form generation on the SAME N items, same single call
(clean/P1/AB/greedy) - does constraining the output format actually
change the judge's verdict, or just its absolute logprobs (D25)?

Only runs on Colab (D17) - `vllm` is imported lazily, inside the one
function that needs it, same pattern as `src/judge.py`. Deliberately its
own module, not folded into judge.py's own schedule/CLI: this is a
one-off diagnostic (2N generations, not part of the resumable 12-calls/
item D19 schedule), reusing judge.py's already-tested
`CallSpec`/`_build_prompts`/`VERDICT_SCHEMA`/`load_full_items_df` rather
than re-deriving any of them.

`python -m src.ablation_decoding --config configs/run.yaml --n-items 100`

Writes `runs/{model_slug}/ablation_decoding.jsonl` - one row per (item,
decoding_mode), `decoding_mode` in {constrained, free_form}. Each row
carries `raw_output` plus the same provenance fields judge.py's own
checkpoint rows carry, so `src/parse.py::parse_verdict_and_confidence()`
can be reused UNMODIFIED to score both arms (CLAUDE.md invariant 7 -
parsing logic lives only in parse.py). Using the exact same strict
parser for both arms is the point: whether free-form generation's raw
text still satisfies it is what "parse rate" is measuring here.
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
    """Seeded random sample of `n_items` non-tie items - same seeding
    convention as judge.py's own `--n-items` (task 1.6's pilot): a random
    sample, not a head(), so the ablation isn't accidentally clustered on
    a handful of question_ids.
    """
    items_df = load_full_items_df(config)
    return items_df.sample(n=n_items, random_state=config.seed)


def load_completed_ablation_keys(checkpoint_path: Path) -> set[str]:
    """(item_id, decoding_mode) pairs already written - this ablation's
    own tiny resumability key, distinct from judge.py's checkpoint_key()
    since decoding_mode isn't one of that key's five fields (condition/
    prompt_variant/order/sample_idx are all fixed here - only
    decoding_mode varies).
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
    """Generates both decoding arms for `n_items` items and appends every
    row to `checkpoint_path` as it completes (resumable - a Colab
    disconnect mid-run just needs the same command re-run).
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
