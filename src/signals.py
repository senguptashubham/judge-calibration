"""calls.parquet -> items.parquet: the per-item confidence signals
(conf_verb / conf_lp / conf_sc / conf_bpe), the P1/P2/P3 ensemble's
entropy decomposition (conf_ens, D20), and RQ4's Tier B/C columns.

Every function takes `rows`: the already-parsed calls (src/parse.py's
output) belonging to one (item_id, condition, prompt_variant) group. This
file never touches `raw_output` (invariant 7).
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config


def _find_call(rows: list[dict], order: str, sample_idx: int) -> dict | None:
    for row in rows:
        if row["order"] == order and row["sample_idx"] == sample_idx:
            return row
    return None


def judge_verdict(rows: list[dict]) -> str | None:
    """D7 primary: the canonical greedy verdict in AB order. AB is
    apply_order()'s identity mapping, so the judge's "A" already means
    model_a and needs no translation.
    """
    call = _find_call(rows, "AB", 0)
    return call["verdict"] if call else None


def _p_model_a_wins(rows: list[dict]) -> float | None:
    """Order-corrected P(model_a wins) = (p_a(AB) + (1 - p_a(BA))) / 2.

    `p_a` is P(displayed-A wins). Under AB, displayed-A is model_a; under
    BA it is model_b, so P(model_a wins | BA) = 1 - p_a(BA). A raw average
    of the two p_a values would blend two different questions.
    """
    call_ab = _find_call(rows, "AB", 0)
    call_ba = _find_call(rows, "BA", 0)
    if call_ab is None or call_ba is None:
        return None
    if call_ab["p_a"] is None or call_ba["p_a"] is None:
        return None
    return (call_ab["p_a"] + (1 - call_ba["p_a"])) / 2


def verdict_bidir(rows: list[dict]) -> str | None:
    """D7 secondary: argmax of the order-corrected P(model_a wins)."""
    p = _p_model_a_wins(rows)
    if p is None:
        return None
    return "A" if p >= 0.5 else "B"


def conf_verb(rows: list[dict]) -> float | None:
    """Verbalized confidence, from the canonical greedy AB call only (D6)."""
    call = _find_call(rows, "AB", 0)
    return call["verbalized_conf"] if call else None


def conf_lp(rows: list[dict]) -> float | None:
    """exp(verdict_token_logprob): the model's probability on whichever
    answer it gave, so it sits in [0.5, 1] - unlike p_a, which points at
    "A" specifically.

    Greedy AB call only (D6): the sampled draws run at temperature_sc, and
    temperature measurably shifts the reported logprobs.
    """
    call = _find_call(rows, "AB", 0)
    if call is None or call["verdict_token_logprob"] is None:
        return None
    return math.exp(call["verdict_token_logprob"])


def conf_sc(rows: list[dict], k_sc: int) -> float | None:
    """Fraction of the k_sc sampled AB verdicts that match the canonical
    greedy verdict. Sampling only exists for (clean, P1) (D19, D21), so
    this is None everywhere else rather than a fabricated 0 or 1.
    """
    canonical = _find_call(rows, "AB", 0)
    if canonical is None:
        return None
    sampled = [_find_call(rows, "AB", i) for i in range(1, k_sc + 1)]
    sampled = [s for s in sampled if s is not None]
    if not sampled:
        return None
    matches = sum(1 for s in sampled if s["verdict"] == canonical["verdict"])
    return matches / len(sampled)


def conf_bpe(rows: list[dict]) -> float | None:
    """1 - H(p), p = the order-corrected P(model_a wins), H = binary
    entropy in nats (the same raw scale as conf_ens's decomposition).

    This is the entropy OF the averaged probability, not the average of
    each order's entropy. When the two orders disagree, p washes toward
    0.5 and H rises - so it captures cross-order inconsistency, not just
    within-order sharpness (SCOPE, 2026).
    """
    p = _p_model_a_wins(rows)
    if p is None:
        return None
    return 1 - _binary_entropy(p)


def compute_item_signals(rows: list[dict], k_sc: int) -> dict:
    return {
        "judge_verdict": judge_verdict(rows),
        "verdict_bidir": verdict_bidir(rows),
        "conf_verb": conf_verb(rows),
        "conf_lp": conf_lp(rows),
        "conf_sc": conf_sc(rows, k_sc),
        "conf_bpe": conf_bpe(rows),
    }


def _binary_entropy(p: float) -> float:
    """H(p) = -(p*log(p) + (1-p)*log(1-p)), in nats, not normalized to [0, 1]."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -((p * np.log(p)) + ((1 - p) * np.log(1 - p)))


def conf_ens(rows_by_variant: dict[str, list[dict]]) -> dict:
    """Judge-level entropy decomposition over the P1/P2/P3 prompt ensemble
    (D20), clean items only. With p_i = P(model_a wins) for variant i:

        mean_p     = mean(p_P1, p_P2, p_P3)
        Total      = H(mean_p)
        Aleatoric  = mean(H(p_P1), H(p_P2), H(p_P3))
        Epistemic  = Total - Aleatoric      # >= 0 by Jensen's inequality
        conf_ens   = 1 - Total

    Aleatoric is how unsure each variant is on its own; Epistemic is the
    extra uncertainty that appears only when averaging ACROSS variants,
    i.e. disagreement between prompts (the BALD / mutual-information
    reading). Needs all three variants at once, which is why it can't live
    inside compute_item_signals.

    Returns all four fields as None if any variant's p is missing - the
    uniform-over-three-variants prior doesn't hold with one absent.
    """
    p_p1 = _p_model_a_wins(rows_by_variant["P1"])
    p_p2 = _p_model_a_wins(rows_by_variant["P2"])
    p_p3 = _p_model_a_wins(rows_by_variant["P3"])

    if p_p1 is None or p_p2 is None or p_p3 is None:
        return {
            "ens_entropy_total": None,
            "ens_entropy_aleatoric": None,
            "ens_entropy_epistemic": None,
            "conf_ens": None,
        }

    mean_p = (p_p1 + p_p2 + p_p3) / 3
    total = _binary_entropy(mean_p)
    aleatoric = (_binary_entropy(p_p1) + _binary_entropy(p_p2) + _binary_entropy(p_p3)) / 3
    epistemic = total - aleatoric
    return {
        "ens_entropy_total": total,
        "ens_entropy_aleatoric": aleatoric,
        "ens_entropy_epistemic": epistemic,
        "conf_ens": 1 - total,
    }


def _sanitize_records(records: list[dict]) -> list[dict]:
    """DataFrame.to_dict("records") gives NaN, not None, for missing values
    - even in string columns - and NaN slips past every `is None` check in
    this file. Converts NaN -> None once, at the DataFrame -> dict boundary.
    """
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in record.items()}
        for record in records
    ]


def _canonical_verdict_ba(rows: list[dict]) -> str | None:
    """The BA call's verdict in canonical model identity: under BA,
    displayed-A is model_b, so a raw "A" means model_b won.
    """
    call = _find_call(rows, "BA", 0)
    if call is None or call["verdict"] is None:
        return None
    return "B" if call["verdict"] == "A" else "A"


def flipped(rows: list[dict]) -> bool | None:
    """True when the canonical AB verdict and the canonical BA verdict
    disagree. Compared against the BA call itself, not verdict_bidir, which
    is a different, p_a-averaged quantity.
    """
    ab = judge_verdict(rows)
    ba = _canonical_verdict_ba(rows)
    if ab is None or ba is None:
        return None
    return ab != ba


# --- RQ4 Tier B / Tier C columns ------------------------------------------

_COT_FIELDS = (
    "cot_logprob_mean",
    "cot_logprob_min",
    "cot_logprob_std",
    "cot_logprob_p10",
    "cot_entropy_mean",
    "n_cot_tokens",
)


def judge_output_len(rows: list[dict]) -> int | None:
    """Tier B: length of the judge's own reasoning text (`reasoning_len`)
    on the canonical AB greedy call.
    """
    call = _find_call(rows, "AB", 0)
    return call["reasoning_len"] if call is not None else None


def verdict_margin(rows: list[dict]) -> float | None:
    """Tier C: top-2 margin at the verdict token. The verdict is
    schema-constrained to "A"/"B", so the margin is exactly
    |P(A) - P(B)| = |2*p_a - 1|, on the canonical AB greedy call (D6).
    """
    call = _find_call(rows, "AB", 0)
    if call is None or call["p_a"] is None:
        return None
    return abs(2 * call["p_a"] - 1)


def cot_aggregates_greedy(rows: list[dict]) -> dict:
    """Tier C: the AB greedy call's six CoT aggregates, renamed *_greedy."""
    call = _find_call(rows, "AB", 0)
    if call is None:
        return {f"{field}_greedy": None for field in _COT_FIELDS}
    return {f"{field}_greedy": call[field] for field in _COT_FIELDS}


def cot_aggregates_sampled(rows: list[dict], k_sc: int) -> dict:
    """Tier C: each CoT aggregate averaged over the k_sc sampled AB calls,
    suffixed _sampled_t07 - the temperature is in the name so it is never
    averaged together with the _greedy version (D4). None where no sampled
    calls exist (everything but clean/P1).
    """
    sampled_calls = [_find_call(rows, "AB", i) for i in range(1, k_sc + 1)]
    sampled_calls = [call for call in sampled_calls if call is not None]

    result = {}
    for field in _COT_FIELDS:
        # A sampled call can exist with this field None (no identifiable
        # reasoning span), so filter per field, not just per call.
        values = [call[field] for call in sampled_calls if call[field] is not None]
        result[f"{field}_sampled_t07"] = sum(values) / len(values) if values else None
    return result


def len_ratio(len_a: int | None, len_b: int | None) -> float | None:
    """Tier B: len_a / len_b. None when len_b == 0 (real empty responses
    exist) or either length is missing - never inf or a smoothed value.
    """
    ratio = None if len_b in (0, None) or len_a is None else (len_a / len_b)
    return ratio


def longer_is_chosen(judge_verdict_value: str | None, len_a: int | None, len_b: int | None) -> bool | None:
    """Tier B: did the judge pick the strictly longer response? None on
    equal lengths, where there is no longer side to have chosen.
    """
    if len_a is None or len_b is None or len_a == len_b:
        return None
    else:
        return judge_verdict_value == ('A' if len_a > len_b else 'B')


def build_items_dataframe(calls: pd.DataFrame, items_labels: pd.DataFrame, k_sc: int) -> pd.DataFrame:
    """calls.parquet -> items.parquet, one row per (item_id, condition,
    prompt_variant) - invariant 14's grain.

    Groups by (item_id, condition) first, then splits by prompt_variant, so
    conf_ens (needs all three variants) and compute_item_signals (needs one)
    come out of the same pass. conf_ens's four fields are stored on the
    clean P1 row only and are None on P2/P3 and on every verbose row (D21).
    Human labels and len_a/len_b are joined from items_labels on item_id.
    """
    labels_by_item = items_labels.set_index("item_id")
    empty_ens = {
        "ens_entropy_total": None,
        "ens_entropy_aleatoric": None,
        "ens_entropy_epistemic": None,
        "conf_ens": None,
    }

    records = []
    for (item_id, condition), item_condition_group in calls.groupby(["item_id", "condition"]):
        variant_groups: dict[str, list[dict]] = {
            str(variant): _sanitize_records(sub.to_dict("records"))
            for variant, sub in item_condition_group.groupby("prompt_variant")
        }

        ens_result = None
        if condition == "clean" and all(v in variant_groups for v in ("P1", "P2", "P3")):
            ens_result = conf_ens(variant_groups)

        # .loc[] on an object column also returns NaN for a missing scalar.
        label_row = (
            _sanitize_records([labels_by_item.loc[str(item_id)].to_dict()])[0]
            if item_id in labels_by_item.index
            else {}
        )
        human_label = label_row.get("majority_label")

        for prompt_variant, rows in variant_groups.items():
            first_row = rows[0]
            signals = compute_item_signals(rows, k_sc)

            len_a = label_row.get("len_a")
            len_b = label_row.get("len_b")

            record = {
                "item_id": item_id,
                "question_id": first_row["question_id"],
                "category": first_row["category"],
                "condition": condition,
                "prompt_variant": prompt_variant,
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
                "len_a": len_a,
                "len_b": len_b,
                "len_ratio": len_ratio(len_a, len_b),
                "abs_len_diff": abs(len_a - len_b) if len_a is not None and len_b is not None else None,
                "longer_is_chosen": longer_is_chosen(signals["judge_verdict"], len_a, len_b),
                "judge_output_len": judge_output_len(rows),
                "verdict_margin": verdict_margin(rows),
                **cot_aggregates_greedy(rows),
                **cot_aggregates_sampled(rows, k_sc),
                "flipped": flipped(rows),
                **(ens_result if prompt_variant == "P1" and ens_result is not None else empty_ens),
            }
            records.append(record)

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    calls = pd.read_parquet(config.paths.calls_parquet)
    items_labels = pd.read_parquet(config.paths.items_labels_parquet)

    items = build_items_dataframe(calls, items_labels, config.k_sc)

    Path(config.paths.items_parquet).parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(config.paths.items_parquet)
    print(f"Wrote {len(items)} items to {config.paths.items_parquet}")
