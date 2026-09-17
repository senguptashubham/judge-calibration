"""Verdict and confidence extraction, failure taxonomy. See TASKS.md task 1.5.

Two independent extraction paths, since they need different raw materials:
- `parse_verdict_and_confidence()` is pure text parsing - only needs
  `raw_output`, works for every row.
- `compute_logprob_signals()` needs the full per-token logprobs judge.py
  saves for every call (D4, amended 4 Sep 2026: 100% coverage). It always
  operates on data already loaded back from `runs/logprobs/*.jsonl.gz`
  (via `load_logprobs_record()`), never on live vLLM objects - judge.py
  never touches this file, and this file never touches vLLM (D17).

`split_cot_and_verdict_tokens()` lives here (relocated from `judge.py` on
4 Sep 2026) because it is itself parsing logic - CLAUDE.md invariant 7
says logic that touches raw model output belongs only in this file.
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
    """Extracts `verdict` and `verbalized_conf` from a judge call's raw JSON
    text alone - no logprobs needed. Classifies failures per CLAUDE.md §3's
    taxonomy: none|no_verdict|no_confidence|malformed_json|truncated.

    Ordering rationale: a verdict-shaped enum failure (`no_verdict`) is
    checked before a confidence-shaped one (`no_confidence`) because a row
    with no verdict is unusable for every RQ, while a row with a verdict
    but no confidence still carries real information (RQ2-RQ4 use it with
    conf_verb null) - the more severe failure gets reported when both
    would technically apply.

    Returns:
      dict with parse_ok, parse_failure_type, verdict, verbalized_conf.
      verdict is populated whenever a valid one was found, even if
      confidence separately failed - CLAUDE.md's schema treats verdict and
      verbalized_conf as independently nullable columns, not all-or-nothing.
    """
    stripped = raw_output.rstrip()
    if not stripped.endswith("}"):
        # The structured-output schema is always a flat JSON object, so a
        # well-formed completion always ends in "}" (module whitespace).
        # Not ending in one is the signature of hitting max_tokens
        # mid-generation, not a formatting mistake - classified separately
        # from malformed_json for exactly that reason.
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


def split_cot_and_verdict_tokens(
    token_texts: list[str], full_text: str
) -> tuple[list[int], int | None]:
    """Locates which generated tokens fall inside the JSON `"reasoning"`
    string value (the CoT tokens) and which single token is the `"verdict"`
    value, using plain substring search on the assembled text rather than
    full JSON parsing - a truncated/malformed generation may not be valid
    JSON at all, and this should degrade gracefully (falls back to "every
    token is CoT, no identifiable verdict token") rather than raise.

    token_texts: each token's own decoded text piece, in generation order -
    concatenating them all must reconstruct full_text exactly.
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
    """Loads one call's saved logprobs file (judge.py's write_logprobs()).
    Thin I/O wrapper, kept separate from compute_logprob_signals() so that
    function stays pure and easy to test against hand-built data.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.loads(f.readline())


def compute_logprob_signals(
    raw_output: str,
    token_ids: list[int],
    token_texts: list[str],
    token_logprobs: list[dict],
) -> dict:
    """Everything derivable from one call's full per-token logprobs (D4,
    amended 4 Sep 2026): the CoT aggregate statistics, `verdict_token_logprob`,
    and `p_a`. Needs the full logprobs, not just raw_output text - this is
    why these fields live here and not in parse_verdict_and_confidence().

    Args (all as loaded from `load_logprobs_record()`, i.e. straight out of
    JSON - NOT vLLM's live objects):
      token_ids: the actually-generated token id at each position (ints).
      token_texts: that same sequence's decoded text, same order/length.
      token_logprobs: one dict per position, {str(token_id): {"logprob":
        float, "decoded_token": str}} - JSON object keys are always
        strings, even though the ids themselves are ints, so lookups here
        use str(token_ids[i]), never token_ids[i] directly.

    Returns:
      dict with verdict_token_logprob, p_a, cot_logprob_{mean,min,std,p10},
      cot_entropy_mean, n_cot_tokens. p_a is None whenever "A"/"B" aren't
      both present among the position's top-K candidates (e.g. an
      extremely confident model whose runner-up fell outside the top 20) -
      D6's sample_idx==0-only validity still applies at the call site,
      this function doesn't know sample_idx and doesn't enforce it.
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
    """question_id -> category (writing|roleplay|reasoning|math|coding|
    extraction|stem|humanities), from the canonical 80-question MT-Bench
    set. NOT available in lmsys/mt_bench_human_judgments itself - verified
    empirically that neither its 'human' nor 'gpt4_pair' split carries a
    category column, since category is a property of the QUESTION, not the
    vote. philschmid/mt-bench is a clean HF mirror of the original benchmark
    question set that does carry it; cross-checked question_id=81 ->
    'writing' against the human-judgments dataset's own question_id=81 (a
    Hawaii travel-blog prompt) to confirm the join key actually lines up
    before trusting it.
    """
    ds = load_dataset(_CATEGORY_DATASET, split="train")
    return {int(row["question_id"]): row["category"] for row in ds}


def build_calls_dataframe(
    checkpoint_path: Path, runs_dir: str, category_lookup: dict[int, str]
) -> pd.DataFrame:
    """One condition's raw checkpoint (runs/judge_{condition}.jsonl) ->
    one row per call, with parse_verdict_and_confidence() and
    compute_logprob_signals() merged in, category backfilled from
    category_lookup (the raw checkpoint's own `category` is always None -
    judge.py copies it straight from items_df, which never had it - see
    load_category_lookup()'s docstring).

    Pure row-by-row transformation, no grouping or aggregation - that's
    items.parquet's job (src/signals.py), not this function's.
    """
    records = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            record = dict(row)
            record["category"] = category_lookup.get(row["question_id"], row.get("category"))

            record.update(parse_verdict_and_confidence(row["raw_output"]))

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
