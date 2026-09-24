"""Raw judge output -> calls.parquet: verdict and confidence extraction, the
failure taxonomy, and every logprob-derived field. The only place in the
primary pipeline that reads `raw_output` (invariant 7).

Two independent paths, since they need different raw materials:
- `parse_verdict_and_confidence()` needs only `raw_output`.
- `compute_logprob_signals()` needs the full per-token logprobs judge.py
  saves for every call (D4), loaded back from runs/logprobs/*.jsonl.gz.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

from src.config import Config
from src.judge import logprobs_path
from src.metrics import truncated_entropy

_CATEGORY_DATASET = "philschmid/mt-bench"


def parse_verdict_and_confidence(raw_output: str) -> dict:
    """`verdict` and `verbalized_conf` from a call's raw JSON text, with
    CLAUDE.md §3's failure taxonomy: none|no_verdict|no_confidence|
    malformed_json|truncated.

    `no_verdict` is checked before `no_confidence`: a row with no verdict is
    unusable everywhere, while a row with a verdict but no confidence still
    carries information, so the more severe failure is the one reported.
    The verdict is kept even when confidence fails - the two columns are
    independently nullable.
    """
    stripped = raw_output.rstrip()
    if not stripped.endswith("}"):
        # The schema is a flat JSON object, so a completion that doesn't end
        # in "}" hit max_tokens mid-generation - a truncation, not a
        # formatting mistake.
        return {"parse_ok": False, "parse_failure_type": "truncated", "verdict": None, "verbalized_conf": None}

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return {"parse_ok": False, "parse_failure_type": "malformed_json", "verdict": None, "verbalized_conf": None}

    if not isinstance(parsed, dict):
        return {"parse_ok": False, "parse_failure_type": "malformed_json", "verdict": None, "verbalized_conf": None}

    verdict = parsed.get("verdict")
    if verdict not in ("A", "B"):
        return {"parse_ok": False, "parse_failure_type": "no_verdict", "verdict": None, "verbalized_conf": None}

    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not (0 <= confidence <= 1):
        return {"parse_ok": False, "parse_failure_type": "no_confidence", "verdict": verdict, "verbalized_conf": None}

    return {"parse_ok": True, "parse_failure_type": "none", "verdict": verdict, "verbalized_conf": float(confidence)}


def reasoning_length(raw_output: str) -> int | None:
    """Character length of the `"reasoning"` value alone (RQ4's
    judge_output_len) - not len(raw_output), which would include the
    ~40-char JSON wrapper around it.

    Exact when raw_output parses as JSON. Otherwise falls back to a
    substring search between the `"reasoning": "` and `"verdict": "` keys,
    which ignores escaped characters - an approximation used only on the
    rare malformed rows, so they don't drop out of Tier B entirely.
    None only if no reasoning value can be located at all.
    """
    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, dict) and isinstance(parsed.get("reasoning"), str):
            return len(parsed["reasoning"])
    except json.JSONDecodeError:
        pass

    reasoning_key = '"reasoning": "'
    verdict_key = '"verdict": "'

    reasoning_key_start = raw_output.find(reasoning_key)
    if reasoning_key_start == -1:
        return None
    reasoning_value_start = reasoning_key_start + len(reasoning_key)

    verdict_key_start = raw_output.find(verdict_key, reasoning_value_start)
    if verdict_key_start == -1:
        return None

    reasoning_value_end = raw_output.rfind('"', reasoning_value_start, verdict_key_start)
    if reasoning_value_end == -1:
        return None

    return reasoning_value_end - reasoning_value_start


def split_cot_and_verdict_tokens(
    token_texts: list[str], full_text: str
) -> tuple[list[int], int | None]:
    """Which generated tokens fall inside the `"reasoning"` value (the CoT
    tokens), and which single token is the `"verdict"` value.

    Uses substring search on the assembled text rather than JSON parsing,
    so a truncated or malformed generation degrades to "every token is
    CoT, no verdict token" instead of raising. `token_texts` must
    concatenate to `full_text` exactly.
    """
    reasoning_key = '"reasoning": "'
    verdict_key = '"verdict": "'

    reasoning_key_start = full_text.find(reasoning_key)
    if reasoning_key_start == -1:
        return list(range(len(token_texts))), None
    reasoning_value_start = reasoning_key_start + len(reasoning_key)

    verdict_key_start = full_text.find(verdict_key, reasoning_value_start)
    if verdict_key_start == -1:
        return list(range(len(token_texts))), None
    verdict_value_start = verdict_key_start + len(verdict_key)

    offsets = [0]
    for t in token_texts:
        offsets.append(offsets[-1] + len(t))

    def _token_at(char_pos: int) -> int:
        for i in range(len(token_texts)):
            if offsets[i] <= char_pos < offsets[i + 1]:
                return i
        return len(token_texts) - 1

    cot_start = _token_at(reasoning_value_start)
    cot_end = _token_at(verdict_key_start)  # exclusive
    cot_token_indices = list(range(cot_start, max(cot_end, cot_start)))
    verdict_token_index = _token_at(verdict_value_start)

    return cot_token_indices, verdict_token_index


def load_logprobs_record(path: Path) -> dict:
    """One call's saved logprobs file (judge.py's write_logprobs())."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.loads(f.readline())


def compute_logprob_signals(
    raw_output: str,
    token_ids: list[int],
    token_texts: list[str],
    token_logprobs: list[dict],
) -> dict:
    """The CoT aggregates, `verdict_token_logprob`, and `p_a` for one call.

        p_a = P(A) / (P(A) + P(B)), at the verdict token position

    renormalized over just the two allowed answers. None when "A" and "B"
    aren't both among that position's top-K candidates. p_a is only valid
    for sample_idx == 0 (D6) - enforced by callers, not here.

    Inputs are as loaded from JSON, so `token_logprobs` is keyed by
    str(token_id), never the int id.
    """
    cot_indices, verdict_idx = split_cot_and_verdict_tokens(token_texts, raw_output)

    cot_logprobs = [token_logprobs[i][str(token_ids[i])]["logprob"] for i in cot_indices]
    cot_entropies = [
        truncated_entropy([candidate["logprob"] for candidate in token_logprobs[i].values()]) for i in cot_indices
    ]

    verdict_token_logprob = None
    p_a = None
    if verdict_idx is not None:
        position = token_logprobs[verdict_idx]
        verdict_token_logprob = position[str(token_ids[verdict_idx])]["logprob"]
        a_logprob = next((c["logprob"] for c in position.values() if c["decoded_token"] == "A"), None)
        b_logprob = next((c["logprob"] for c in position.values() if c["decoded_token"] == "B"), None)
        if a_logprob is not None and b_logprob is not None:
            a_prob, b_prob = np.exp(a_logprob), np.exp(b_logprob)
            p_a = float(a_prob / (a_prob + b_prob))

    return {
        "verdict_token_logprob": verdict_token_logprob,
        "p_a": p_a,
        "cot_logprob_mean": float(np.mean(cot_logprobs)) if cot_logprobs else None,
        "cot_logprob_min": float(np.min(cot_logprobs)) if cot_logprobs else None,
        "cot_logprob_std": float(np.std(cot_logprobs)) if cot_logprobs else None,
        "cot_logprob_p10": float(np.percentile(cot_logprobs, 10)) if cot_logprobs else None,
        "cot_entropy_mean": float(np.mean(cot_entropies)) if cot_entropies else None,
        "n_cot_tokens": len(cot_indices),
    }


def load_category_lookup() -> dict[int, str]:
    """question_id -> MT-Bench category. The human-judgments dataset has no
    category column (it's a property of the question, not the vote), so it
    comes from philschmid/mt-bench, a mirror of the original 80-question
    set whose question_ids line up with the judgments dataset's.
    """
    ds = load_dataset(_CATEGORY_DATASET, split="train")
    return {int(row["question_id"]): row["category"] for row in ds}


def build_calls_dataframe(
    checkpoint_path: Path, runs_dir: str, category_lookup: dict[int, str]
) -> pd.DataFrame:
    """One condition's checkpoint -> one row per call, with the parsed
    verdict/confidence, reasoning_len, and logprob signals merged in and
    `category` backfilled (judge.py's own category field is always None).
    """
    records = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            record = dict(row)
            record["category"] = category_lookup.get(row["question_id"], row.get("category"))

            record.update(parse_verdict_and_confidence(row["raw_output"]))
            record["reasoning_len"] = reasoning_length(row["raw_output"])

            lp_path = logprobs_path(
                runs_dir, row["item_id"], row["condition"], row["prompt_variant"], row["order"], row["sample_idx"]
            )
            lp_record = load_logprobs_record(lp_path)
            record.update(
                compute_logprob_signals(
                    row["raw_output"], lp_record["token_ids"], lp_record["token_texts"], lp_record["token_logprobs"]
                )
            )

            records.append(record)
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    category_lookup = load_category_lookup()

    frames = []
    for condition in config.conditions:
        checkpoint_path = Path(config.paths.runs_dir) / f"judge_{condition}.jsonl"
        if not checkpoint_path.exists():
            print(f"skipping {condition}: no checkpoint at {checkpoint_path}")
            continue
        print(f"parsing {condition} from {checkpoint_path} ...")
        frames.append(build_calls_dataframe(checkpoint_path, config.paths.runs_dir, category_lookup))

    calls = pd.concat(frames, ignore_index=True)
    Path(config.paths.calls_parquet).parent.mkdir(parents=True, exist_ok=True)
    calls.to_parquet(config.paths.calls_parquet)
    print(f"Wrote {len(calls)} calls to {config.paths.calls_parquet}")
