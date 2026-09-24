"""RQ6: runs/kev_8b/kev.jsonl -> calls_kev_8b.parquet -> items_kev_8b.parquet.

The kev-8b counterpart of parse.py + signals.py, at kev's simpler grain:
(item_id, condition), no prompt_variant axis and no sampling - kev makes
one deterministic forward pass per request (D27). kev's `confidence` field
is never used: it is an exact rescaling of probabilities[choice] (D27).

items_kev_8b.parquet columns: item_id, question_id, category, condition,
turn, the human-label columns, judge_verdict, verdict_bidir, conf_kev,
conf_kev_bpe, conf_kev_bpe_prob, flipped, correct, correct_bidir,
input_tokens, any_skipped.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.judge_kev import KevConfig
from src.signals import _binary_entropy, _sanitize_records, prob_on_verdict


def _kev_response_fields(row: dict) -> tuple[str | None, dict, int | None]:
    """(choice, probabilities, input_tokens) from one checkpoint row, in
    either of the two layouts that exist: the analysed run stores the three
    fields flat; the current call_kev() stores kev's whole response body
    under `raw_response` (answers.verdict.{choice, probabilities},
    usage.input_tokens).
    """
    raw = row.get("raw_response")
    if isinstance(raw, dict):
        verdict = raw.get("answers", {}).get("verdict", {})
        return verdict.get("choice"), verdict.get("probabilities") or {}, raw.get("usage", {}).get("input_tokens")
    return row.get("choice"), row.get("probabilities") or {}, row.get("input_tokens")


def build_calls_kev(checkpoint_path: str | Path) -> pd.DataFrame:
    """kev.jsonl -> one row per call, with the `probabilities` dict
    flattened into `prob_a`/`prob_b`. kev's response is already structured,
    so no text parsing is needed.
    """
    rows = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            choice, probabilities, input_tokens = _kev_response_fields(row)
            rows.append(
                {
                    "item_id": row["item_id"],
                    "question_id": row["question_id"],
                    "category": row.get("category"),
                    "model_a": row["model_a"],
                    "model_b": row["model_b"],
                    "turn": row["turn"],
                    "condition": row["condition"],
                    "order": row["order"],
                    "judge_model": row["judge_model"],
                    "state_tokens": row.get("state_tokens"),
                    "skipped": row.get("skipped", False),
                    "skip_reason": row.get("skip_reason"),
                    "ok": row.get("ok"),
                    "choice": choice,
                    "prob_a": probabilities.get("A"),
                    "prob_b": probabilities.get("B"),
                    "input_tokens": input_tokens,
                }
            )
    return pd.DataFrame.from_records(rows)


def _find_kev_call(rows: list[dict], order: str) -> dict | None:
    """One order's call, or None if it is missing, skipped, or failed
    (`ok` is not True) - all three are treated as no data.
    """
    for row in rows:
        if row["order"] == order and row.get("ok") is True:
            return row
    return None


def _p_model_a_wins_kev(rows: list[dict]) -> float | None:
    """Order-corrected P(model_a wins) = (prob_a(AB) + (1 - prob_a(BA))) / 2,
    the same correction as signals.py::_p_model_a_wins: under BA the model
    displayed as "A" is model_b, so prob_a(BA) must be flipped first.
    """
    call_ab = _find_kev_call(rows, "AB")
    call_ba = _find_kev_call(rows, "BA")
    if call_ab is None or call_ba is None:
        return None
    if call_ab["prob_a"] is None or call_ba["prob_a"] is None:
        return None
    return (call_ab["prob_a"] + (1 - call_ba["prob_a"])) / 2


def judge_verdict_kev(rows: list[dict]) -> str | None:
    """D7 primary: the AB call's own choice (AB is the identity mapping)."""
    call = _find_kev_call(rows, "AB")
    return call["choice"] if call else None


def verdict_bidir_kev(rows: list[dict]) -> str | None:
    """D7 secondary: argmax of the order-corrected P(model_a wins)."""
    p = _p_model_a_wins_kev(rows)
    if p is None:
        return None
    return "A" if p >= 0.5 else "B"


def _canonical_verdict_ba_kev(rows: list[dict]) -> str | None:
    """The BA call's choice in canonical identity: a raw "A" means model_b won."""
    call = _find_kev_call(rows, "BA")
    if call is None:
        return None
    return "B" if call["choice"] == "A" else "A"


def flipped_kev(rows: list[dict]) -> bool | None:
    """True when the canonical AB and canonical BA verdicts disagree."""
    ab = judge_verdict_kev(rows)
    ba = _canonical_verdict_ba_kev(rows)
    if ab is None or ba is None:
        return None
    return ab != ba


def conf_kev(rows: list[dict]) -> float | None:
    """kev's probability on whichever answer the AB call gave, in [0.5, 1]
    - the analog of conf_lp (D27).
    """
    call = _find_kev_call(rows, "AB")
    if call is None:
        return None
    return call["prob_a"] if call["choice"] == "A" else call["prob_b"]


def conf_kev_bpe(rows: list[dict]) -> float | None:
    """1 - H(order-corrected P(model_a wins)), in nats - the analog of conf_bpe."""
    p = _p_model_a_wins_kev(rows)
    if p is None:
        return None
    return 1 - _binary_entropy(p)


def compute_item_signals_kev(rows: list[dict]) -> dict:
    return {
        "judge_verdict": judge_verdict_kev(rows),
        "verdict_bidir": verdict_bidir_kev(rows),
        "conf_kev": conf_kev(rows),
        "conf_kev_bpe": conf_kev_bpe(rows),
        # conf_kev_bpe's calibration form: the same order-corrected p, as a
        # probability on the AB verdict (signals.py::prob_on_verdict).
        "conf_kev_bpe_prob": prob_on_verdict(_p_model_a_wins_kev(rows), judge_verdict_kev(rows)),
        "flipped": flipped_kev(rows),
    }


def build_items_kev_dataframe(calls: pd.DataFrame, items_labels: pd.DataFrame) -> pd.DataFrame:
    """calls_kev_8b.parquet -> items_kev_8b.parquet, one row per
    (item_id, condition), human labels joined on item_id.

    `input_tokens` is the AB call's server-reported count (AB and BA render
    the same content, so their counts are equal). A fully skipped item has
    none, so its `state_tokens` is used instead - it exceeded the cap either
    way, which is all the coverage-regime split needs.
    """
    labels_by_item = items_labels.set_index("item_id")

    records = []
    for (item_id, condition), group in calls.groupby(["item_id", "condition"]):
        rows = _sanitize_records(group.to_dict("records"))
        first_row = rows[0]
        signals = compute_item_signals_kev(rows)

        label_row = (
            _sanitize_records([labels_by_item.loc[str(item_id)].to_dict()])[0]
            if item_id in labels_by_item.index
            else {}
        )
        human_label = label_row.get("majority_label")

        ab_call = _find_kev_call(rows, "AB")
        input_tokens = ab_call["input_tokens"] if ab_call else first_row.get("state_tokens")

        records.append(
            {
                "item_id": item_id,
                "question_id": first_row["question_id"],
                "category": first_row["category"],
                "condition": condition,
                "turn": first_row["turn"],
                "human_label": human_label,
                "n_human_votes": label_row.get("n_human_votes"),
                "frac_prefer_a": label_row.get("frac_prefer_a"),
                "human_unanimous": label_row.get("human_unanimous"),
                "human_agreed": label_row.get("human_agreed"),
                "d_human": label_row.get("d_human"),
                **signals,
                "correct": (
                    signals["judge_verdict"] == human_label
                    if signals["judge_verdict"] is not None and human_label is not None
                    else None
                ),
                "correct_bidir": (
                    signals["verdict_bidir"] == human_label
                    if signals["verdict_bidir"] is not None and human_label is not None
                    else None
                ),
                "input_tokens": input_tokens,
                "any_skipped": any(r.get("skipped") for r in rows),
            }
        )

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = KevConfig.from_yaml(args.config)
    checkpoint_path = Path(config.paths.runs_dir) / "kev.jsonl"

    calls = build_calls_kev(checkpoint_path)
    Path(config.paths.calls_parquet).parent.mkdir(parents=True, exist_ok=True)
    calls.to_parquet(config.paths.calls_parquet)
    print(f"Wrote {len(calls)} calls to {config.paths.calls_parquet}")

    items_labels = pd.read_parquet(config.paths.items_labels_parquet)
    items = build_items_kev_dataframe(calls, items_labels)
    items.to_parquet(config.paths.items_parquet)
    print(f"Wrote {len(items)} items to {config.paths.items_parquet}")
