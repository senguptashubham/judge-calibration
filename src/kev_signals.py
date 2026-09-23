"""runs/kev_8b/kev.jsonl -> calls_kev-8b.parquet -> items_kev-8b.parquet
(TASKS.md K3b, DECISIONS.md D27/RQ6).

Mirrors src/signals.py's role and pattern for the primary judge, at kev's
much simpler grain: (item_id, condition) only, no prompt_variant axis, no
sample_idx sampling (D27 - kev makes one deterministic forward pass per
request). `confidence` is deliberately never computed here - confirmed an
exact deterministic rescaling of probabilities[choice] (D27's 23 Sep
amendment), excluded from analysis, not just unused.

Reuses signals.py's own `_binary_entropy` and `_sanitize_records` rather
than reimplementing them - both already handle real edge cases this
module would otherwise have to rediscover (log(0), and pandas'
NaN-not-None convention at the DataFrame -> dict boundary).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.judge_kev import KevConfig
from src.signals import _binary_entropy, _sanitize_records


def build_calls_kev(checkpoint_path: str | Path) -> pd.DataFrame:
    """runs/kev_8b/kev.jsonl -> calls_kev-8b.parquet's DataFrame. Flattens
    the raw `probabilities` dict into `prob_a`/`prob_b` columns (parquet-
    friendly, and matches how the primary study stores `p_a` as a flat
    column rather than a nested structure) - no other parsing needed,
    kev's own response is already structured, unlike judge.py's raw_output
    (CLAUDE.md invariant 7 has nothing to do here beyond "don't invent
    signals judge_kev.py didn't actually collect").
    """
    rows = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            probabilities = row.get("probabilities") or {}
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
                    "choice": row.get("choice"),
                    "prob_a": probabilities.get("A"),
                    "prob_b": probabilities.get("B"),
                    "input_tokens": row.get("input_tokens"),
                }
            )
    return pd.DataFrame.from_records(rows)


def _find_kev_call(rows: list[dict], order: str) -> dict | None:
    """Locates one order's call among an item-condition's rows - the
    kev-arm analog of signals.py::_find_call, minus sample_idx (D27 - no
    sampling exists here). Returns None if that order's row doesn't exist,
    was skipped (over the token ceiling), or failed (ok is not True) -
    callers treat all three identically, same as D27's call_kev() docstring
    already establishes for the harness layer.
    """
    for row in rows:
        if row["order"] == order and row.get("ok") is True:
            return row
    return None


def _p_model_a_wins_kev(rows: list[dict]) -> float | None:
    """Order-corrected mean P(model_a wins), exactly mirroring
    signals.py::_p_model_a_wins - NOT a naive average of prob_a across
    AB/BA, which would silently blend two different physical questions
    (apply_order() swaps which model is displayed as "A" under BA, so
    prob_a(BA) means P(model_b wins) until flipped). This was flagged as a
    real bug risk before it was built (DECISIONS.md D27) - fixed here by
    construction, not left as a caller responsibility.
    """
    call_ab = _find_kev_call(rows, "AB")
    call_ba = _find_kev_call(rows, "BA")
    if call_ab is None or call_ba is None:
        return None
    if call_ab["prob_a"] is None or call_ba["prob_a"] is None:
        return None
    return (call_ab["prob_a"] + (1 - call_ba["prob_a"])) / 2


def judge_verdict_kev(rows: list[dict]) -> str | None:
    """Canonical verdict: the AB-order call's own choice, already in
    canonical model_a/model_b-relative identity (AB is apply_order()'s
    identity mapping). Mirrors D7's judge_verdict for the primary judge.
    """
    call = _find_kev_call(rows, "AB")
    return call["choice"] if call else None


def verdict_bidir_kev(rows: list[dict]) -> str | None:
    """Order-corrected verdict, in canonical model identity - "A" means
    model_a won, "B" means model_b won (see _p_model_a_wins_kev).
    """
    p = _p_model_a_wins_kev(rows)
    if p is None:
        return None
    return "A" if p >= 0.5 else "B"


def _canonical_verdict_ba_kev(rows: list[dict]) -> str | None:
    """The BA-order call's own choice, translated into canonical
    model_a/model_b identity - mirrors signals.py::_canonical_verdict_ba.
    Under BA, displayed-A = model_b, so a raw "A" choice means model_b won.
    """
    call = _find_kev_call(rows, "BA")
    if call is None:
        return None
    return "B" if call["choice"] == "A" else "A"


def flipped_kev(rows: list[dict]) -> bool | None:
    """Canonical verdict differs between AB and BA order - the direct
    analog of signals.py::flipped(), needed for the position-swap flip-
    rate test (K4). Compares judge_verdict_kev (canonical AB) against the
    BA call's own translated verdict, NOT against verdict_bidir_kev (which
    is p-averaged, a different quantity than "what did BA alone say").
    """
    ab = judge_verdict_kev(rows)
    ba = _canonical_verdict_ba_kev(rows)
    if ab is None or ba is None:
        return None
    return ab != ba


def conf_kev(rows: list[dict]) -> float | None:
    """kev's raw class-probability signal: P(whichever verdict the
    canonical AB-order call actually gave), symmetric in [0.5, 1] - the
    direct analog of conf_lp (D27). `confidence` is deliberately never
    used (confirmed redundant, D27's 23 Sep amendment).
    """
    call = _find_kev_call(rows, "AB")
    if call is None:
        return None
    return call["prob_a"] if call["choice"] == "A" else call["prob_b"]


def conf_kev_bpe(rows: list[dict]) -> float | None:
    """1 - H(p_model_a_wins), on the same raw-nats scale as the primary
    study's conf_bpe (signals.py) - the direct analog, reusing the exact
    same order-correction (_p_model_a_wins_kev) and entropy formula
    (signals.py::_binary_entropy) rather than independently re-deriving
    either.
    """
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
        "flipped": flipped_kev(rows),
    }


def build_items_kev_dataframe(calls: pd.DataFrame, items_labels: pd.DataFrame) -> pd.DataFrame:
    """calls_kev-8b.parquet -> items_kev-8b.parquet. One row per
    (item_id, condition) - no prompt_variant axis (D27). Mirrors
    signals.py::build_items_dataframe's join/NaN-handling pattern exactly.

    `input_tokens` on the item row is the AB-order call's own value when
    available (AB/BA render identical total content - same two responses,
    just swapped labels - so their token counts are equal; this is not an
    approximation). For a fully-skipped item (both orders skipped, the
    only pattern ever observed - see K2/D27), no `input_tokens` exists, so
    `state_tokens` is used instead - always safe as a coverage-regime
    indicator here, since a skipped item exceeded the cap by definition
    either way.
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
