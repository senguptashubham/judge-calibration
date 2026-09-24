"""Tests for src/signals.py.

All hand-computed where the formula is non-trivial - particularly the
order-corrected mean used by verdict_bidir/conf_bpe (D7), since a naive
raw average of p_a(AB) and p_a(BA) would silently blend two different
quantities (see signals.py's own docstrings for why).
"""

import math

import pytest

from src.signals import (
    _canonical_verdict_ba,
    compute_item_signals,
    conf_bpe,
    conf_lp,
    conf_sc,
    conf_verb,
    conf_verb_bidir,
    flipped,
    judge_verdict,
    prob_on_verdict,
    verdict_bidir,
)


def _call(order, sample_idx, **fields):
    row = {"order": order, "sample_idx": sample_idx, "verdict": None, "verbalized_conf": None, "p_a": None, "verdict_token_logprob": None}
    row.update(fields)
    return row


# --- judge_verdict ---------------------------------------------------------


def test_judge_verdict_reads_ab_greedy():
    rows = [_call("AB", 0, verdict="A"), _call("BA", 0, verdict="B")]
    assert judge_verdict(rows) == "A"


def test_judge_verdict_none_when_ab_greedy_missing():
    rows = [_call("BA", 0, verdict="B")]
    assert judge_verdict(rows) is None


# --- verdict_bidir (order-corrected mean p_a) -------------------------------


def test_verdict_bidir_hand_computed_favors_a():
    # p_a(AB)=0.8 -> P(model_a wins|AB)=0.8
    # p_a(BA)=0.3 -> P(model_a wins|BA)=1-0.3=0.7
    # mean = (0.8+0.7)/2 = 0.75 >= 0.5 -> "A"
    rows = [_call("AB", 0, p_a=0.8), _call("BA", 0, p_a=0.3)]
    assert verdict_bidir(rows) == "A"


def test_verdict_bidir_hand_computed_favors_b():
    # p_a(AB)=0.3, p_a(BA)=0.8 -> P(model_a wins|BA)=1-0.8=0.2
    # mean = (0.3+0.2)/2 = 0.25 < 0.5 -> "B"
    rows = [_call("AB", 0, p_a=0.3), _call("BA", 0, p_a=0.8)]
    assert verdict_bidir(rows) == "B"


def test_verdict_bidir_a_naive_raw_average_would_have_gotten_this_wrong():
    # p_a(AB)=0.9 (strongly favors model_a), p_a(BA)=0.9 (strongly favors
    # model_b, since BA's "A" = model_b) - a naive raw average would give
    # 0.9, wrongly reading as "strongly favors A". The correct order-
    # corrected mean is (0.9 + (1-0.9))/2 = 0.5, genuine disagreement.
    rows = [_call("AB", 0, p_a=0.9), _call("BA", 0, p_a=0.9)]
    naive_average = (0.9 + 0.9) / 2
    assert naive_average == pytest.approx(0.9)  # what the wrong approach would give
    # Correct: exactly 0.5 is the boundary - verdict_bidir's >= 0.5 rule
    # resolves it to "A", but the real signal here is conf_bpe collapsing
    # toward its minimum (see below), not this verdict alone.
    assert verdict_bidir(rows) == "A"


def test_verdict_bidir_none_when_an_order_is_missing():
    rows = [_call("AB", 0, p_a=0.9)]
    assert verdict_bidir(rows) is None


def test_verdict_bidir_none_when_p_a_is_none():
    rows = [_call("AB", 0, p_a=None), _call("BA", 0, p_a=0.5)]
    assert verdict_bidir(rows) is None


# --- _canonical_verdict_ba (BA's own raw verdict, translated to canonical
# model identity - the same displayed-A/model_b swap _p_model_a_wins applies
# to p_a, but applied to the discrete verdict instead) -----------------------


def test_canonical_verdict_ba_translates_raw_a_to_canonical_b():
    # Under BA, displayed-A = model_b, so a raw "A" verdict means model_b won.
    rows = [_call("BA", 0, verdict="A")]
    assert _canonical_verdict_ba(rows) == "B"


def test_canonical_verdict_ba_translates_raw_b_to_canonical_a():
    rows = [_call("BA", 0, verdict="B")]
    assert _canonical_verdict_ba(rows) == "A"


def test_canonical_verdict_ba_none_when_ba_call_missing():
    rows = [_call("AB", 0, verdict="A")]
    assert _canonical_verdict_ba(rows) is None


def test_canonical_verdict_ba_none_when_verdict_is_none():
    rows = [_call("BA", 0, verdict=None)]
    assert _canonical_verdict_ba(rows) is None


# --- flipped (canonical AB verdict vs. canonical BA verdict) ----------------


def test_flipped_true_when_orders_disagree_after_translation():
    # AB raw "A" -> canonical A (no translation needed).
    # BA raw "A" -> canonical B (displayed-A under BA = model_b).
    # Canonical A != canonical B -> flipped.
    rows = [_call("AB", 0, verdict="A"), _call("BA", 0, verdict="A")]
    assert flipped(rows) is True


def test_flipped_false_when_orders_agree_after_translation():
    # AB raw "A" -> canonical A.
    # BA raw "B" -> canonical A (displayed-B under BA = model_a).
    # Both canonical A -> not flipped, even though the RAW verdict strings
    # differ ("A" vs "B") - this is exactly the translation flipped() must
    # apply, not a naive string comparison.
    rows = [_call("AB", 0, verdict="A"), _call("BA", 0, verdict="B")]
    assert flipped(rows) is False


def test_flipped_none_when_ab_missing():
    rows = [_call("BA", 0, verdict="A")]
    assert flipped(rows) is None


def test_flipped_none_when_ba_missing():
    rows = [_call("AB", 0, verdict="A")]
    assert flipped(rows) is None


def test_flipped_is_not_the_same_quantity_as_judge_verdict_vs_verdict_bidir():
    # Regression guard for the exact distinction flipped()'s docstring
    # warns about: flipped compares canonical AB vs. canonical BA verdicts
    # directly, NOT judge_verdict vs. verdict_bidir (a p_a-averaged,
    # order-corrected label - a different quantity). verdict and p_a are
    # set independently here (each function only reads the field it needs)
    # specifically to demonstrate the two are not interchangeable, not to
    # model a plausible real generation.
    #
    # AB: verdict="A" (canonical A), p_a=0.9 -> P(model_a wins|AB)=0.9
    # BA: verdict="A" (canonical B, translated), p_a=0.4 -> P(model_a wins|BA)=1-0.4=0.6
    # judge_verdict = "A" (raw AB verdict)
    # verdict_bidir: mean(0.9, 0.6) = 0.75 >= 0.5 -> "A"
    # judge_verdict == verdict_bidir ("A" == "A") - NOT flipped by that measure.
    # But flipped() compares canonical AB ("A") vs. canonical BA ("B") -> True.
    rows = [_call("AB", 0, verdict="A", p_a=0.9), _call("BA", 0, verdict="A", p_a=0.4)]
    assert judge_verdict(rows) == verdict_bidir(rows) == "A"
    assert flipped(rows) is True


# --- conf_verb ---------------------------------------------------------


def test_conf_verb_reads_ab_greedy_only():
    rows = [_call("AB", 0, verbalized_conf=0.85), _call("AB", 1, verbalized_conf=0.5)]
    assert conf_verb(rows) == 0.85


# --- conf_lp: the DoD's explicit requirement --------------------------------


def test_conf_lp_hand_computed():
    rows = [_call("AB", 0, verdict_token_logprob=math.log(0.9))]
    assert conf_lp(rows) == pytest.approx(0.9)


def test_conf_lp_never_reads_from_sample_idx_nonzero():
    # TASKS.md task 1.7's own required assertion, made concrete: a
    # sample_idx=1 row present with a DIFFERENT logprob must not leak in.
    rows = [
        _call("AB", 0, verdict_token_logprob=math.log(0.9)),
        _call("AB", 1, verdict_token_logprob=math.log(0.1)),  # very different - would be an obvious bug if picked up
    ]
    assert conf_lp(rows) == pytest.approx(0.9)


def test_conf_lp_none_when_only_sampled_rows_present():
    # No sample_idx=0 row at all - must not fall back to a sampled one.
    rows = [_call("AB", 1, verdict_token_logprob=math.log(0.9))]
    assert conf_lp(rows) is None


# --- conf_sc -----------------------------------------------------------


def test_conf_sc_hand_computed():
    rows = [
        _call("AB", 0, verdict="A"),
        _call("AB", 1, verdict="A"),
        _call("AB", 2, verdict="A"),
        _call("AB", 3, verdict="B"),
        _call("AB", 4, verdict="A"),
    ]
    assert conf_sc(rows, k_sc=4) == pytest.approx(0.75)


def test_conf_sc_none_when_no_sampled_draws_exist():
    # clean/P1 only (D19, D21) - P2/P3/verbose rows have no sampled calls.
    rows = [_call("AB", 0, verdict="A"), _call("BA", 0, verdict="A")]
    assert conf_sc(rows, k_sc=4) is None


# --- conf_bpe ------------------------------------------------------------


def test_conf_bpe_hand_computed():
    # Same setup as the verdict_bidir "favors A" case: order-corrected
    # mean p = 0.75.
    rows = [_call("AB", 0, p_a=0.8), _call("BA", 0, p_a=0.3)]
    p = 0.75
    expected_entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))  # ~0.5623
    assert conf_bpe(rows) == pytest.approx(1 - expected_entropy)


def test_conf_bpe_high_when_orders_agree_strongly():
    # Both orders push the order-corrected mean toward the same extreme.
    rows = [_call("AB", 0, p_a=0.99), _call("BA", 0, p_a=0.01)]  # mean ~0.99
    assert conf_bpe(rows) > 0.9


def test_conf_bpe_at_minimum_when_orders_flatly_disagree():
    # Order-corrected mean collapses to exactly 0.5 - maximum binary
    # entropy (ln 2), so conf_bpe hits its floor of 1 - ln(2).
    rows = [_call("AB", 0, p_a=0.9), _call("BA", 0, p_a=0.9)]
    assert conf_bpe(rows) == pytest.approx(1 - math.log(2))


def test_conf_bpe_none_when_missing_data():
    rows = [_call("AB", 0, p_a=0.9)]
    assert conf_bpe(rows) is None


# --- calibration forms ------------------------------------------------------


def test_prob_on_verdict_reads_the_verdicts_side():
    assert prob_on_verdict(0.75, "A") == pytest.approx(0.75)
    assert prob_on_verdict(0.75, "B") == pytest.approx(0.25)
    assert prob_on_verdict(None, "A") is None


def test_coin_flip_is_0_5_as_a_probability_but_0_307_as_conf_bpe():
    # The reason the *_prob forms exist: orders that flatly disagree are a
    # coin flip. As a probability that is 0.5; conf_bpe reports 1 - ln 2.
    rows = [_call("AB", 0, verdict="A", p_a=0.9), _call("BA", 0, verdict="A", p_a=0.9)]
    signals = compute_item_signals(rows, k_sc=4)
    assert signals["conf_bpe"] == pytest.approx(1 - math.log(2))
    assert signals["conf_bpe_prob"] == pytest.approx(0.5)
    assert signals["conf_lp_bidir"] == pytest.approx(0.5)


def test_conf_bpe_prob_can_fall_below_half_when_ab_verdict_loses_the_average():
    # AB says A at p_a=0.6; BA (raw "A" = model_b) at p_a=0.9 -> P(model_a)=0.1.
    # mean p = 0.35: verdict_bidir is B, and the AB verdict A gets 0.35.
    rows = [_call("AB", 0, verdict="A", p_a=0.6), _call("BA", 0, verdict="A", p_a=0.9)]
    signals = compute_item_signals(rows, k_sc=4)
    assert signals["conf_bpe_prob"] == pytest.approx(0.35)
    assert signals["conf_lp_bidir"] == pytest.approx(0.65)


def test_conf_verb_bidir_counts_a_disagreeing_order_against_the_verdict():
    # AB: A, p_a 0.95. BA: raw "A" = model_b, p_a 0.6 -> P(model_a) 0.4.
    # mean p 0.675 -> verdict_bidir A. AB agrees (0.9), BA disagrees (1 - 0.8).
    rows = [
        _call("AB", 0, verdict="A", verbalized_conf=0.9, p_a=0.95),
        _call("BA", 0, verdict="A", verbalized_conf=0.8, p_a=0.6),
    ]
    assert verdict_bidir(rows) == "A"
    assert conf_verb_bidir(rows) == pytest.approx((0.9 + 0.2) / 2)


def test_conf_verb_bidir_none_without_ba_confidence():
    rows = [_call("AB", 0, verdict="A", verbalized_conf=0.9, p_a=0.7), _call("BA", 0, verdict="B", p_a=0.1)]
    assert conf_verb_bidir(rows) is None


# --- compute_item_signals: integration ------------------------------------


def test_compute_item_signals_all_populated_for_complete_clean_p1_item():
    rows = [
        _call("AB", 0, verdict="A", verbalized_conf=0.9, p_a=0.85, verdict_token_logprob=math.log(0.85)),
        _call("BA", 0, verdict="B", verbalized_conf=0.8, p_a=0.2, verdict_token_logprob=math.log(0.8)),
        _call("AB", 1, verdict="A"),
        _call("AB", 2, verdict="A"),
        _call("AB", 3, verdict="B"),
        _call("AB", 4, verdict="A"),
    ]
    signals = compute_item_signals(rows, k_sc=4)
    for key, value in signals.items():
        assert value is not None, f"{key} was None"


def test_compute_item_signals_conf_sc_none_for_p2_style_data_but_rest_populated():
    # P2/P3 only ever get greedy, both orders - no sampled draws (D19/D21).
    rows = [
        _call("AB", 0, verdict="A", verbalized_conf=0.9, p_a=0.85, verdict_token_logprob=math.log(0.85)),
        _call("BA", 0, verdict="B", verbalized_conf=0.8, p_a=0.2, verdict_token_logprob=math.log(0.8)),
    ]
    signals = compute_item_signals(rows, k_sc=4)
    assert signals["conf_sc"] is None
    for key, value in signals.items():
        if key != "conf_sc":
            assert value is not None, f"{key} was unexpectedly None"
