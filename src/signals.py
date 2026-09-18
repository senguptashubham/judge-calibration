"""conf_verb / conf_lp / conf_sc / conf_bpe (task 1.7), plus conf_ens and
its judge-level Total/Aleatoric/Epistemic entropy decomposition over the
P1/P2/P3 prompt ensemble - clean condition only (DECISIONS.md D20, task 2.2b).

Operates on already-parsed calls - one dict per call, carrying `order`,
`sample_idx`, `verdict`, `verbalized_conf`, `p_a`, `verdict_token_logprob`
(src/parse.py's output already merged in, matching calls.parquet's real
schema - CLAUDE.md invariant 7: parsing itself lives only in parse.py,
this file never touches `raw_output`). All functions here take `rows`:
every call belonging to one (item_id, condition, prompt_variant) group.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config


def _find_call(rows: list[dict], order: str, sample_idx: int) -> dict | None:
    """Locates one specific call among an item's rows, by order + sample_idx."""
    for row in rows:
        if row["order"] == order and row["sample_idx"] == sample_idx:
            return row
    return None


def judge_verdict(rows: list[dict]) -> str | None:
    """D7 primary: the canonical greedy verdict, AB order specifically.

    AB is `apply_order()`'s identity mapping (prompts.py) - "Assistant A"
    in that rendering already IS model_a, so this raw verdict needs no
    translation to be read as "which model won." This is exactly why the
    primary reference is AB, not BA: it's the one order where the
    judge's own label and the canonical model_a/model_b identity coincide.
    D7: "what a real deployment running one pass would get."
    """
    call = _find_call(rows, "AB", 0)
    return call["verdict"] if call else None


def _p_model_a_wins(rows: list[dict]) -> float | None:
    """Averages both greedy orders' `p_a` onto a single, consistent
    "P(model_a wins)" scale before averaging - not a raw average of the
    two `p_a` values, which would silently blend two different questions.

    `p_a` is P(displayed-A), and "displayed-A" means a different physical
    model depending on order: under AB, displayed-A = model_a, so
    p_a(AB) IS P(model_a wins) directly. Under BA, apply_order() swaps
    which model is shown as A, so displayed-A = model_b, and therefore
    P(model_a wins | BA) = 1 - p_a(BA). Both terms below are now on the
    same "does the judge favor model_a" scale before being combined.
    """
    call_ab = _find_call(rows, "AB", 0)
    call_ba = _find_call(rows, "BA", 0)
    if call_ab is None or call_ba is None:
        return None
    if call_ab["p_a"] is None or call_ba["p_a"] is None:
        return None
    return (call_ab["p_a"] + (1 - call_ba["p_a"])) / 2


def verdict_bidir(rows: list[dict]) -> str | None:
    """D7 secondary: argmax of the order-corrected mean P(model_a wins)
    (see _p_model_a_wins). "A" here means model_a won, "B" means model_b
    won - already in canonical model identity, comparable to human_label
    the same way judge_verdict is.
    """
    p = _p_model_a_wins(rows)
    if p is None:
        return None
    return "A" if p >= 0.5 else "B"


def conf_verb(rows: list[dict]) -> float | None:
    """Verbalized confidence, from the canonical greedy AB call only (D6)."""
    call = _find_call(rows, "AB", 0)
    return call["verbalized_conf"] if call else None


def conf_lp(rows: list[dict]) -> float | None:
    """exp(verdict_token_logprob): the model's own confidence in whichever
    answer it actually gave - symmetric (always in [0.5, 1] for a binary
    forced choice), unlike p_a which is directional toward "A" specifically.

    From the canonical greedy AB call only (D6) - sampled draws
    (sample_idx > 0) run at temperature_sc, and REPORT.md's temperature-
    probe methods note (8 Sep 2026) confirmed empirically, not just
    theoretically, that temperature measurably shifts the reported
    logprobs even when the verdict itself doesn't change.
    """
    call = _find_call(rows, "AB", 0)
    if call is None or call["verdict_token_logprob"] is None:
        return None
    return math.exp(call["verdict_token_logprob"])


def conf_sc(rows: list[dict], k_sc: int) -> float | None:
    """Fraction of the k_sc sampled verdicts (sample_idx=1..k_sc, AB order
    only) matching the canonical greedy verdict. clean/P1 ONLY - sampling
    doesn't exist for any other (condition, prompt_variant) pair (D19,
    D21), so this returns None everywhere else, not a fabricated 0 or 1.
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
    """1 - H(p), where p is the order-corrected mean P(model_a wins) (see
    _p_model_a_wins) and H is binary entropy in nats (not normalized to
    [0,1] - kept on the same raw-nats scale as conf_ens's entropy
    decomposition, D20, task 2.2b, rather than each signal picking its
    own private normalization).

    This is entropy OF the averaged probability, not the average of each
    order's own entropy - a deliberately different, more informative
    quantity: if both orders agree strongly (both push p toward the same
    extreme), the average stays extreme and H stays low. If the orders
    DISAGREE (one order favors model_a, the other favors model_b after
    translation), the average washes toward 0.5 and H rises toward its
    max - directly capturing cross-order inconsistency, not just
    within-order sharpness. (SCOPE, 2026; LEARNING.md C11.)
    """
    p = _p_model_a_wins(rows)
    if p is None:
        return None
    return 1 - _binary_entropy(p)


def compute_item_signals(rows: list[dict], k_sc: int) -> dict:
    """Combines every signal above into one item-level record. `rows` is
    every call belonging to a single (item_id, condition, prompt_variant)
    group - the caller is responsible for that grouping (task 2.2's
    items.parquet-building CLI groups calls.parquet this way).
    """
    return {
        "judge_verdict": judge_verdict(rows),
        "verdict_bidir": verdict_bidir(rows),
        "conf_verb": conf_verb(rows),
        "conf_lp": conf_lp(rows),
        "conf_sc": conf_sc(rows, k_sc),
        "conf_bpe": conf_bpe(rows),
    }


def _binary_entropy(p: float) -> float:
    """H(p) = -(p*log(p) + (1-p)*log(1-p)), in nats. conf_bpe computes this
    same quantity inline (for a different p) - factored out now that
    conf_ens needs it three more times (once per variant, plus once for
    the ensemble mean). Keep this on the same raw-nats scale conf_bpe
    already uses, not normalized to [0,1] - see conf_bpe's docstring for
    why a shared scale matters once these get compared to each other.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -((p * np.log(p)) + ((1 - p) * np.log(1 - p)))


def conf_ens(rows_by_variant: dict[str, list[dict]]) -> dict:
    """Judge-level entropy decomposition over the P1/P2/P3 prompt ensemble
    (DECISIONS.md D20) - clean items only. `rows_by_variant` maps
    "P1"/"P2"/"P3" -> that variant's own rows for ONE item (same shape
    `compute_item_signals` takes, just three of them at once instead of
    one - this is exactly why conf_ens can't live inside
    compute_item_signals: it needs cross-variant information that a single
    (item_id, condition, prompt_variant) group doesn't have).

    Per variant, get p_i = P(model_a wins) via _p_model_a_wins(rows) (the
    same order-corrected mean conf_bpe and verdict_bidir already use) -
    NOT the raw greedy AB p_a alone, so conf_ens is measuring the same
    "which model does this variant favor" quantity the rest of the file
    already standardizes on.

        mean_p     = mean(p_P1, p_P2, p_P3)
        Total      = H(mean_p)                        # _binary_entropy
        Aleatoric  = mean(H(p_P1), H(p_P2), H(p_P3))
        Epistemic  = Total - Aleatoric                 # >=0, Jensen's inequality
        conf_ens   = 1 - Total

    Total is the ensemble's overall uncertainty; Aleatoric is how unsure
    each variant is on its own, on average; Epistemic is the extra
    uncertainty that only appears once you average ACROSS variants - i.e.
    disagreement between P1/P2/P3, not uncertainty within any one of them
    (the BALD / mutual-information reading - LEARNING.md theory K).

    Args:
        rows_by_variant: {"P1": rows, "P2": rows, "P3": rows} - all three
            required (caller's job to only call this when all three exist
            for the item, i.e. condition == "clean").

    Returns:
        dict with ens_entropy_total, ens_entropy_aleatoric,
        ens_entropy_epistemic, conf_ens - all None if any variant's p is
        unavailable.
    """
    p_p1 = _p_model_a_wins(rows_by_variant["P1"])
    p_p2 = _p_model_a_wins(rows_by_variant["P2"])
    p_p3 = _p_model_a_wins(rows_by_variant["P3"])

    if p_p1 is None or p_p2 is None or p_p3 is None:
        # "Uniform prior over three variants" (D20) doesn't hold with a
        # variant missing - propagate the gap, don't fabricate a 0.0 in
        # its place (that would silently claim "certain model_b wins" for
        # a variant that actually just failed to parse).
        return {
            "ens_entropy_total": None,
            "ens_entropy_aleatoric": None,
            "ens_entropy_epistemic": None,
            "conf_ens": None,
        }

    mean_p = (p_p1 + p_p2 + p_p3) / 3
    total = _binary_entropy(mean_p)
    aleatoric = (_binary_entropy(p_p1) + _binary_entropy(p_p2) + _binary_entropy(p_p3)) / 3
    epistemic = total - aleatoric  # >=0, Jensen's inequality
    return {
        "ens_entropy_total": total,
        "ens_entropy_aleatoric": aleatoric,
        "ens_entropy_epistemic": epistemic,
        "conf_ens": 1 - total,
    }


def _sanitize_records(records: list[dict]) -> list[dict]:
    """DataFrame.to_dict("records") represents every missing value as NaN
    (a float) - even for string columns like `verdict` - never as Python's
    `None`, which is what every is-None check in this file (_find_call,
    _p_model_a_wins, conf_lp, conf_sc, conf_ens, ...) was written against.
    A NaN silently sails past `is None` (nan is None -> False), so without
    this, a parse failure would propagate as NaN through the arithmetic
    instead of being treated as missing - confirmed empirically against
    calls.parquet's own truncated rows before this was added. Converts
    NaN -> None once, at the DataFrame -> dict boundary, so nothing above
    this function needs to learn pandas' missing-value convention.
    """
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in record.items()}
        for record in records
    ]


def _canonical_verdict_ba(rows: list[dict]) -> str | None:
    """The BA-order call's own verdict, translated into canonical
    model_a/model_b identity - same translation _p_model_a_wins() applies
    to p_a: under BA, displayed-A = model_b, so a raw "A" verdict means
    model_b won.
    """
    call = _find_call(rows, "BA", 0)
    if call is None or call["verdict"] is None:
        return None
    return "B" if call["verdict"] == "A" else "A"


def flipped(rows: list[dict]) -> bool | None:
    """CLAUDE.md schema: canonical verdict differs between order AB and
    BA, within this (condition, prompt_variant) pair. Compares the
    canonical AB verdict (judge_verdict) against the canonical BA verdict
    - NOT against verdict_bidir, which is a p_a-averaged, order-corrected
    label, a different quantity than "what did BA alone say."
    """
    ab = judge_verdict(rows)
    ba = _canonical_verdict_ba(rows)
    if ab is None or ba is None:
        return None
    return ab != ba


# --- Tier B / Tier C (task 5.2) ---------------------------------------
#
# Skeleton only - bodies TODO. Each function's docstring states the exact
# formula/source columns/scope already agreed (18 Sep 2026 discussion);
# fill in the body, then wire the call into build_items_dataframe()'s
# per-record dict below (marked with matching TODO comments) in place of
# the current None/leftover placeholders.

_COT_FIELDS = (
    "cot_logprob_mean",
    "cot_logprob_min",
    "cot_logprob_std",
    "cot_logprob_p10",
    "cot_entropy_mean",
    "n_cot_tokens",
)


def judge_output_len(rows: list[dict]) -> int | None:
    """Tier B: character length of just the judge's own "reasoning" text -
    calls.parquet's `reasoning_len` column (parse.py::reasoning_length(),
    exact for well-formed JSON, per-call). Canonical AB-greedy call only -
    same D6 scope conf_lp/conf_verb already use (sample_idx=0, order="AB"
    is "the" single deployed pass this signal describes).

    Returns:
      The AB-greedy call's `reasoning_len`, or None if that call or its
      value is missing.
    """
    raise NotImplementedError("task 5.2: _find_call(rows, 'AB', 0)['reasoning_len']")


def verdict_margin(rows: list[dict]) -> float | None:
    """Tier C: exact top-2 margin at the verdict token position. The
    verdict is schema-constrained to exactly 2 candidate tokens ("A"/"B"),
    so the top-2 margin is exactly abs(P(A) - P(B)) = abs(2*p_a - 1) -
    p_a from the canonical AB-greedy call only (D6: p_a is only valid at
    sample_idx=0, same scope conf_bpe/conf_lp already use).

    Returns:
      abs(2*p_a - 1) for the AB-greedy call, or None if that call or its
      p_a is missing.
    """
    raise NotImplementedError("task 5.2: abs(2 * _find_call(rows, 'AB', 0)['p_a'] - 1)")


def cot_aggregates_greedy(rows: list[dict]) -> dict:
    """Tier C: the AB-greedy call's own CoT aggregate columns (already
    computed by parse.py from saved per-token logprobs), passed through
    unchanged with a "_greedy" suffix on every key - exactly one greedy
    call per item, so no aggregation is needed here, only renaming.

    Returns:
      dict with cot_logprob_mean_greedy, cot_logprob_min_greedy,
      cot_logprob_std_greedy, cot_logprob_p10_greedy,
      cot_entropy_mean_greedy, n_cot_tokens_greedy - all None if the
      AB-greedy call is missing.
    """
    raise NotImplementedError(
        "task 5.2: {f'{field}_greedy': _find_call(rows, 'AB', 0)[field] for field in _COT_FIELDS}, "
        "with a None-call guard"
    )


def cot_aggregates_sampled(rows: list[dict], k_sc: int) -> dict:
    """Tier C: mean-of-means across the k_sc sampled calls (sample_idx=
    1..k_sc, AB order only - the only place sampling happens, D19/D21) of
    each of the same six CoT aggregate columns - "_sampled_t07" suffix,
    matching D4's naming convention (temperature goes in the column name
    so this is never silently averaged together with the _greedy variant,
    which ran at a different temperature).

    Returns:
      dict with the same six field names, suffixed "_sampled_t07" - all
      None if no sampled calls exist for this (condition, prompt_variant)
      pair (verbose/P2/P3 never sample, D19/D21 - the D21 sanity check
      this file's own module docstring already establishes for conf_sc/
      conf_ens applies here too, don't special-case it away).
    """
    raise NotImplementedError(
        "task 5.2: mean each _COT_FIELDS entry across [_find_call(rows, 'AB', i) for i in range(1, k_sc + 1)]"
    )


def len_ratio(len_a: int | None, len_b: int | None) -> float | None:
    """Tier B: len_a / len_b. None whenever len_b == 0 (real data has
    genuine empty responses - confirmed empirically, task 5.2 discussion
    18 Sep 2026) or either length is missing - propagate, don't fabricate
    inf or a smoothed value.
    """
    raise NotImplementedError("task 5.2: len_a / len_b, None if len_b in (0, None) or len_a is None")


def longer_is_chosen(judge_verdict_value: str | None, len_a: int | None, len_b: int | None) -> bool | None:
    """Tier B: does the judge's verdict pick whichever side has the
    strictly longer response? None on a tie (len_a == len_b) - there is
    no "longer" side to have been chosen, so False would misreport a real
    non-answer as a negative finding (task 5.2 discussion, 18 Sep 2026).
    """
    raise NotImplementedError(
        "task 5.2: judge_verdict_value == ('A' if len_a > len_b else 'B'), None if tied or either input missing"
    )


def build_items_dataframe(calls: pd.DataFrame, items_labels: pd.DataFrame, k_sc: int) -> pd.DataFrame:
    """calls.parquet -> items.parquet (task 2.2). One row per
    (item_id, condition, prompt_variant) - CLAUDE.md invariant 14's grain.

    Structure: group by (item_id, condition) FIRST, then split each group
    by prompt_variant - this is what lets conf_ens (needs all three
    variants at once) and compute_item_signals (needs just one variant's
    rows) both fall out of the same pass, rather than building the table
    once and then doing a separate cross-variant pass over it afterward.

    Per (item_id, condition, prompt_variant) group:
      - rows.to_dict("records") to get the list[dict] shape every
        signals.py function expects (they were written against dicts,
        not DataFrame rows - see compute_item_signals's own signature).
      - compute_item_signals(rows, k_sc) for judge_verdict/verdict_bidir/
        conf_verb/conf_lp/conf_sc/conf_bpe.
      - join items_labels (on item_id) for majority_label/frac_prefer_a/
        n_human_votes/human_unanimous/human_agreed/d_human.
      - correct = judge_verdict == majority_label;
        correct_bidir = verdict_bidir == majority_label.
      - question_id/category/model_a/model_b/turn carry straight over
        from the group (every row in a group shares them).

    Per (item_id, condition) group, once all its prompt_variant sub-groups
    are built: if condition == "clean" and P1/P2/P3 are ALL present, call
    conf_ens(...) and merge its four fields onto the P1 record only -
    None on P2/P3 (schema note: item-level quantity, store once). For
    condition == "verbose" (or any group missing a variant), those four
    fields are None on every row (D21 - no ensemble exists there; this is
    the D21 sanity check Gate 2 requires, so don't special-case it away).

    Args:
        calls: calls.parquet, as loaded (e.g. pd.read_parquet(...)).
        items_labels: items_labels.parquet - task 0.8's human-label table,
            now carrying item_id (just added) for the join.
        k_sc: config.k_sc, passed through to compute_item_signals.

    Returns:
        items.parquet's DataFrame, per CLAUDE.md sec 3's full schema.
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

        # .loc[] on an object-dtype column returns NaN, not None, for a
        # missing scalar (same pandas quirk _sanitize_records exists for) -
        # route through it here too rather than writing a second,
        # inconsistent NaN-handling path for this one lookup.
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
                # len_a/len_b: now real (items_labels.parquet populates
                # them, task 5.2 prerequisite, 18 Sep 2026) - everything
                # below this line is task 5.2's own TODO (see the Tier
                # B/Tier C stub functions above build_items_dataframe).
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
