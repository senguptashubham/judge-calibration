"""Tests for src/metrics.py, including the hand-computed ECE reference case
(CLAUDE.md sec 5: expected 0.222) and the discrete-signal ECE case
(DECISIONS.md D14).
"""

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from src.metrics import (
    auto_accept_stats,
    aurc,
    auroc_error,
    brier,
    brier_decomposition,
    cohens_kappa,
    ece,
    mce,
    oracle_risk_coverage,
    overconfidence_gap,
    risk_coverage,
    threshold_sweep,
    truncated_entropy,
)


def test_truncated_entropy_hand_computed_already_normalized():
    # p = [0.6, 0.4] already sums to 1 - renormalization is a no-op.
    logprobs = [np.log(0.6), np.log(0.4)]
    expected = -(0.6 * np.log(0.6) + 0.4 * np.log(0.4))  # ~0.673012
    assert truncated_entropy(logprobs) == pytest.approx(expected)


def test_truncated_entropy_hand_computed_renormalizes_truncated_mass():
    # Raw probabilities [0.5, 0.3] sum to 0.8 (mass outside top-K is
    # missing) - must renormalize over just these two before computing H.
    logprobs = [np.log(0.5), np.log(0.3)]
    p1, p2 = 0.5 / 0.8, 0.3 / 0.8
    expected = -(p1 * np.log(p1) + p2 * np.log(p2))  # ~0.661563
    assert truncated_entropy(logprobs) == pytest.approx(expected)


def test_ece_reference():
    # CLAUDE.md sec 5's hand-computed case: two bins,
    # (conf 0.9, acc 0.75, weight 0.4) and (conf 0.6, acc 0.33, weight 0.6)
    # -> ECE = 0.4*0.15 + 0.6*0.27 = 0.222.
    # N=500 (200 + 300) is the smallest 40/60 split that lands accuracy
    # exactly on 0.75 and 0.33 with whole-number correct counts, so this
    # reproduces the reference exactly rather than approximately.
    confidences = np.concatenate([np.full(200, 0.9), np.full(300, 0.6)])
    correct = np.concatenate(
        [
            np.array([True] * 150 + [False] * 50),  # 150/200 = 0.75
            np.array([True] * 99 + [False] * 201),  # 99/300 = 0.33
        ]
    )

    # Only two distinct confidence values exist (0.6, 0.9), so "auto" (the
    # default) puts each in its own exact bin. Neither "uniform" nor
    # "quantile" would reproduce this reference: with n_bins=2, uniform's
    # fixed 0.5 boundary puts both 0.6 and 0.9 in the same [0.5, 1.0) bin,
    # and quantile's median also lands inside the 0.6 cluster since it's
    # 60% of the data. That's exactly why the discrete branch of "auto"
    # exists (DECISIONS.md D14) - it isn't a fallback.
    ece_value, n_effective_bins = ece(confidences, correct, n_bins=2)

    assert n_effective_bins == 2
    assert ece_value == pytest.approx(0.222, abs=1e-3)


def test_ece_discrete():
    # A conf_sc-shaped signal: only 5 possible values at k_sc=4.
    confidences = [0.0, 0.25, 0.5, 0.75, 1.0] * 4
    correct = [True, False, True, True, False] * 4

    ece_value, n_effective_bins = ece(confidences, correct, n_bins=10)

    assert n_effective_bins == 5
    assert not np.isnan(ece_value)


def test_ece_auto_falls_back_to_quantile_for_many_unique_values():
    # Regression test: "auto" must not create one bin per unique value when
    # there are more unique values than n_bins - it should behave like
    # "quantile" instead (DECISIONS.md D14). Before this was fixed, "auto"
    # ignored n_bins entirely and always did exact binning.
    rng = np.random.default_rng(seed=0)
    confidences = rng.uniform(0.5, 1.0, size=200)  # ~200 distinct values
    correct = rng.integers(0, 2, size=200).astype(bool)

    _, n_effective_bins = ece(confidences, correct, n_bins=10)

    assert n_effective_bins <= 10


def test_kappa_balanced():
    # CLAUDE.md sec 5's hand-computed case: p_o=0.85, p_e=0.5 -> kappa=0.70.
    # p_e depends only on each rater's OWN marginal - if rater a's labels
    # are an exact 50/50 split, p_e = 0.5*P_b(A) + 0.5*P_b(B) = 0.5*1 = 0.5
    # regardless of what b's own marginal looks like. So: make a exactly
    # 10 "A" + 10 "B", then flip 3 of the 20 labels to get b, which gives
    # p_o = 17/20 = 0.85 directly, without needing b balanced too.
    a = np.array(["A"] * 10 + ["B"] * 10)
    b = a.copy()
    b[:3] = np.where(b[:3] == "A", "B", "A")  # flip 3 labels -> 3 mismatches

    kappa = cohens_kappa(a, b)

    assert kappa == pytest.approx(0.70, abs=1e-9)


def test_kappa_matches_sklearn():
    rng = np.random.default_rng(seed=0)
    for _ in range(3):
        a = rng.integers(0, 3, size=100)  # 3 categories, not just binary
        b = rng.integers(0, 3, size=100)
        assert cohens_kappa(a, b) == pytest.approx(
            cohen_kappa_score(a, b), abs=1e-9
        )


# --- mce, brier, brier_decomposition, overconfidence_gap, auroc_error -----
#
# overconfidence_gap/mce/brier/brier_decomposition reuse test_ece_reference's
# dataset (200 items @ conf=0.9, 150 correct; 300 items @ conf=0.6, 99
# correct - N=500, mean conf=0.72, mean accuracy=0.498) so the numbers are
# hand-checkable against each other, not just against this file. Expected
# values worked out during planning:
#   overconfidence_gap -> 0.222 (same number as ECE here, NOT a general
#     identity - true only because the judge is overconfident in BOTH bins;
#     worth a comment in the test explaining why, since it looks like a bug)
#   mce                -> 0.27 (the worse of the two bins' gaps, 0.15/0.27)
#   brier               -> 0.2604
#   brier_decomposition -> reliability=0.05274, resolution=0.042336,
#     uncertainty=0.249996 (reliability - resolution + uncertainty = 0.2604,
#     matching brier() exactly - this cross-check IS the task's DoD)
# auroc_error's tests are below this block, already filled in.


def test_overconfidence_gap_reference():
    confidences = np.concatenate([np.full(200, 0.9), np.full(300, 0.6)])
    correct = np.concatenate(
        [
            np.array([True] * 150 + [False] * 50),  # 150/200 = 0.75
            np.array([True] * 99 + [False] * 201),  # 99/300 = 0.33
        ]
    )
    assert overconfidence_gap(confidences, correct) == pytest.approx(0.222, abs=1e-3)


def test_mce_reference():
    confidences = np.concatenate([np.full(200, 0.9), np.full(300, 0.6)])
    correct = np.concatenate(
        [
            np.array([True] * 150 + [False] * 50),  # 150/200 = 0.75
            np.array([True] * 99 + [False] * 201),  # 99/300 = 0.33
        ]
    )
    mce_value, n_effective_bins = mce(confidences, correct, n_bins=2)
    assert n_effective_bins == 2
    assert mce_value == pytest.approx(0.27, abs=1e-3)


def test_brier_reference():
    confidences = np.concatenate([np.full(200, 0.9), np.full(300, 0.6)])
    correct = np.concatenate(
        [
            np.array([True] * 150 + [False] * 50),  # 150/200 = 0.75
            np.array([True] * 99 + [False] * 201),  # 99/300 = 0.33
        ]
    )
    assert brier(confidences, correct) == pytest.approx(0.2604, abs=1e-3)


def test_brier_decomposition_reconstructs_brier_score():
    confidences = np.concatenate([np.full(200, 0.9), np.full(300, 0.6)])
    correct = np.concatenate(
        [
            np.array([True] * 150 + [False] * 50),  # 150/200 = 0.75
            np.array([True] * 99 + [False] * 201),  # 99/300 = 0.33
        ]
    )
    reliability, resolution, uncertainty = brier_decomposition(confidences, correct, n_bins=2)
    assert (reliability - resolution + uncertainty) == pytest.approx(
        brier(confidences, correct), abs=1e-6
    )


def test_brier_decomposition_reference_terms():
    confidences = np.concatenate([np.full(200, 0.9), np.full(300, 0.6)])
    correct = np.concatenate(
        [
            np.array([True] * 150 + [False] * 50),  # 150/200 = 0.75
            np.array([True] * 99 + [False] * 201),  # 99/300 = 0.33
        ]
    )
    reliability, resolution, uncertainty = brier_decomposition(confidences, correct, n_bins=2)
    assert reliability == pytest.approx(0.05274, abs=1e-3)
    assert resolution == pytest.approx(0.042336, abs=1e-3)
    assert uncertainty == pytest.approx(0.249996, abs=1e-3)


def test_auroc_error_reference():
    # 3 correct items at uncertainty [0.1, 0.2, 0.4], 2 error items at
    # [0.3, 0.5] -> 5 of the 6 (error, correct) pairs rank correctly
    # (0.3 loses to 0.4) -> AUROC = 5/6.
    uncertainty = [0.1, 0.2, 0.4, 0.3, 0.5]
    correct = [True, True, True, False, False]

    assert auroc_error(uncertainty, correct) == pytest.approx(5 / 6)


def test_auroc_error_perfectly_separating_signal_is_one():
    # Every error item has higher uncertainty than every correct item -
    # the signal ranks errors above correct answers with zero mistakes.
    uncertainty = [0.1, 0.2, 0.3, 0.9, 0.95]
    correct = [True, True, True, False, False]

    assert auroc_error(uncertainty, correct) == pytest.approx(1.0)


def test_auroc_error_inverted_signal_is_zero():
    # Every error item has LOWER uncertainty than every correct item - the
    # worst possible ranking. This is the test that would catch the
    # positive-class/sign bug the docstring warns about: get error vs.
    # correct backwards and this silently reports 1.0 instead of 0.0.
    uncertainty = [0.7, 0.8, 0.9, 0.1, 0.2]
    correct = [True, True, True, False, False]

    assert auroc_error(uncertainty, correct) == pytest.approx(0.0)


def test_auroc_error_uninformative_signal_is_near_half():
    # A signal with no relationship to correctness should land near 0.5,
    # not exactly 0.5 - tolerance sized for N=2000 draws at this seed.
    rng = np.random.default_rng(seed=0)
    uncertainty = rng.uniform(0, 1, size=2000)
    correct = rng.integers(0, 2, size=2000).astype(bool)

    assert auroc_error(uncertainty, correct) == pytest.approx(0.5, abs=0.05)


def test_auroc_error_raises_when_only_one_class_present():
    # AUROC is undefined without both an error and a correct item present -
    # this must surface as a loud failure (sklearn's ValueError), never a
    # silently wrong number like 0.5 or 1.0.
    with pytest.raises(ValueError):
        auroc_error([0.1, 0.2, 0.3], [True, True, True])


# --- risk_coverage, oracle_risk_coverage, aurc -----------------------------
#
# risk_coverage()/oracle_risk_coverage()/aurc() reference case, hand-computed
# during planning. n=5, with a tied pair at uncertainty=0.3 specifically to
# exercise the unique-threshold stepping (both items must enter coverage
# together, not split across two steps):
#
#   uncertainty = [0.1, 0.2, 0.3, 0.3, 0.9]
#   correct     = [T,   T,   F,   T,   F  ]
#
#   threshold=0.1: kept={0}       -> correct=[T]         -> cov=1/5=0.2, risk=1-1/1=0.0
#   threshold=0.2: kept={0,1}     -> correct=[T,T]        -> cov=2/5=0.4, risk=1-2/2=0.0
#   threshold=0.3: kept={0,1,2,3} -> correct=[T,T,F,T]    -> cov=4/5=0.8, risk=1-3/4=0.25
#   threshold=0.9: kept={0..4}    -> correct=[T,T,F,T,F]  -> cov=5/5=1.0, risk=1-3/5=0.4
#
#   coverage = [0.2, 0.4, 0.8, 1.0], risk = [0.0, 0.0, 0.25, 0.4]
#
#   AURC = trapezoid(y=risk, x=coverage):
#     (0.2->0.4): width 0.2, avg height (0+0)/2=0        -> area 0.0
#     (0.4->0.8): width 0.4, avg height (0+0.25)/2=0.125  -> area 0.05
#     (0.8->1.0): width 0.2, avg height (0.25+0.4)/2=0.325 -> area 0.065
#     total = 0.115


def test_risk_coverage_reference():
    uncertainty = [0.1, 0.2, 0.3, 0.3, 0.9]
    correct = [True, True, False, True, False]

    coverage, risk = risk_coverage(uncertainty, correct)

    # 4 unique thresholds, not 5 items - the tied pair at 0.3 collapses into
    # one step, which is the whole point of stepping by unique value.
    assert len(coverage) == 4
    np.testing.assert_allclose(coverage, [0.2, 0.4, 0.8, 1.0])
    np.testing.assert_allclose(risk, [0.0, 0.0, 0.25, 0.4])


def test_aurc_reference():
    coverage = np.array([0.2, 0.4, 0.8, 1.0])
    risk = np.array([0.0, 0.0, 0.25, 0.4])

    assert aurc(coverage, risk) == pytest.approx(0.115, abs=1e-9)


def test_aurc_swapped_axes_gives_a_different_wrong_answer():
    # Regression guard for the x/y swap bug found during development:
    # aurc(coverage, risk) and the backwards trapezoid(y=coverage, x=risk)
    # must not silently agree.
    coverage = np.array([0.2, 0.4, 0.8, 1.0])
    risk = np.array([0.0, 0.0, 0.25, 0.4])

    correct_aurc = aurc(coverage, risk)
    swapped = float(np.trapezoid(y=coverage, x=risk))

    assert correct_aurc != pytest.approx(swapped)


def test_oracle_risk_coverage_reference():
    # Same 5-item correct/incorrect split as above, but oracle ignores the
    # uncertainty column entirely - it only ever produces 2 points, at
    # coverage=accuracy and coverage=1.0.
    correct = [True, True, False, True, False]  # accuracy = 3/5 = 0.6

    coverage, risk = oracle_risk_coverage(correct)

    assert len(coverage) == 2
    np.testing.assert_allclose(coverage, [0.6, 1.0])
    np.testing.assert_allclose(risk, [0.0, 0.4])


def test_oracle_aurc_matches_closed_form():
    # AURC of the oracle curve has a closed form: (1 - accuracy)^2 / 2,
    # from its exact two-point shape (coverage=accuracy -> risk=0,
    # coverage=1.0 -> risk=1-accuracy): a right triangle of base
    # (1-accuracy) and height (1-accuracy).
    rng = np.random.default_rng(seed=0)
    correct = rng.random(500) < 0.6
    accuracy = correct.mean()

    coverage, risk = oracle_risk_coverage(correct)
    computed = aurc(coverage, risk)
    closed_form = (1 - accuracy) ** 2 / 2

    assert computed == pytest.approx(closed_form, abs=1e-9)


def test_oracle_dominates_noisy_real_signal():
    # DoD for task 3.1: "oracle dominates every real signal by construction."
    # Build an uncertainty signal correlated with wrongness but with noise
    # mixed in, so it's a realistic imperfect signal, not already the oracle.
    rng = np.random.default_rng(seed=1)
    n = 300
    correct = rng.random(n) < 0.7
    noise = rng.normal(0, 0.5, size=n)
    uncertainty = (~correct).astype(float) + noise

    oracle_coverage, oracle_risk = oracle_risk_coverage(correct)
    signal_coverage, signal_risk = risk_coverage(uncertainty, correct)

    # Different signals produce different numbers of unique thresholds, so
    # compare on a common coverage grid via interpolation rather than
    # assuming the two curves line up point-for-point.
    grid = np.linspace(oracle_coverage.min(), 1.0, 50)
    oracle_interp = np.interp(grid, oracle_coverage, oracle_risk)
    signal_interp = np.interp(grid, signal_coverage, signal_risk)

    assert np.all(oracle_interp <= signal_interp + 1e-9)
    assert aurc(oracle_coverage, oracle_risk) <= aurc(signal_coverage, signal_risk)


def test_risk_coverage_uninformative_signal_stays_near_base_error_rate():
    # A signal uncorrelated with correctness shouldn't be able to improve
    # risk as coverage drops - every "kept" subset is effectively a random
    # sample of the full population, so risk should hover near the overall
    # error rate at every coverage level, not trend downward.
    rng = np.random.default_rng(seed=2)
    n = 2000
    correct = rng.random(n) < 0.65
    uncertainty = rng.uniform(0, 1, size=n)  # independent of correct

    coverage, risk = risk_coverage(uncertainty, correct)
    base_error_rate = 1 - correct.mean()

    grid = np.array([0.25, 0.5, 0.75, 1.0])
    risk_at_grid = np.interp(grid, coverage, risk)

    np.testing.assert_allclose(risk_at_grid, base_error_rate, atol=0.05)


# --- threshold_sweep --------------------------------------------------------
#
# threshold_sweep() reference case, hand-computed during planning then
# cross-checked against the function itself (ece()/cohens_kappa() are
# already independently tested elsewhere, so this dataset targets
# threshold_sweep()'s own orchestration - kept-mask selection and
# coverage/n_kept bookkeeping - not the sub-metrics' math).
#
# n=8, tied pairs at every threshold (signal repeats each of 0.1/0.3/0.6/0.9
# twice) to also exercise the same tie-safe stepping as risk_coverage().
# human_label is deliberately NOT constant (unlike a simpler dataset would
# give) so kappa isn't trivially 0 - a rater with zero variance makes
# p_e == p_o always, which would hide any real chance-correction:
#
#   idx: signal  judge  human  conf   correct
#   0    0.1     A      A      0.90   T
#   1    0.1     B      A      0.60   F
#   2    0.3     A      A      0.80   T
#   3    0.3     B      B      0.70   T
#   4    0.6     A      B      0.55   F
#   5    0.6     B      B      0.65   T
#   6    0.9     A      A      0.50   T
#   7    0.9     B      A      0.50   F
#
# At threshold=0.3 (kept = idx 0-3), worked by hand:
#   accuracy = 3/4 = 0.75
#   kappa: p_o = 3/4 = 0.75; judge freq(A) = 0.5, human freq(A) = 0.75
#     -> p_e = 0.5*0.75 + 0.5*0.25 = 0.5 -> kappa = (0.75-0.5)/(1-0.5) = 0.5
#   ece (n_bins=2, "auto"): 4 unique confidences > n_bins -> quantile bins.
#     edges from percentiles [0,50,100] of [0.6,0.7,0.8,0.9] -> [0.6,0.75,0.9+eps]
#     bin1={0.6,0.7} (idx1,3) correct=[F,T] -> conf_bar=0.65, acc=0.5
#     bin2={0.8,0.9} (idx2,0) correct=[T,T] -> conf_bar=0.85, acc=1.0
#     ece = (2*|0.5-0.65| + 2*|1.0-0.85|)/4 = (0.3+0.3)/4 = 0.15
#
# The other three thresholds' accuracy/kappa/ece were verified against the
# implementation the same way but aren't reproduced by hand above - the
# pattern is identical, just more bookkeeping.

_SWEEP_SIGNAL = [0.1, 0.1, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9]
_SWEEP_JUDGE = ["A", "B", "A", "B", "A", "B", "A", "B"]
_SWEEP_HUMAN = ["A", "A", "A", "B", "B", "B", "A", "A"]
_SWEEP_CONF = [0.9, 0.6, 0.8, 0.7, 0.55, 0.65, 0.5, 0.5]
_SWEEP_CORRECT = [j == h for j, h in zip(_SWEEP_JUDGE, _SWEEP_HUMAN)]


def test_threshold_sweep_unique_thresholds_and_coverage():
    # 4 unique thresholds, not 8 items - the tied pairs must collapse into
    # one step each, same tie-safe stepping as risk_coverage().
    result = threshold_sweep(
        _SWEEP_SIGNAL, _SWEEP_CORRECT, _SWEEP_JUDGE, _SWEEP_HUMAN, _SWEEP_CONF,
        n_bins=2,
    )
    assert len(result["threshold"]) == 4
    np.testing.assert_allclose(result["threshold"], [0.1, 0.3, 0.6, 0.9])
    np.testing.assert_allclose(result["coverage"], [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(result["n_kept"], [2, 4, 6, 8])


def test_threshold_sweep_reference_row():
    # The threshold=0.3 row, hand-computed above.
    result = threshold_sweep(
        _SWEEP_SIGNAL, _SWEEP_CORRECT, _SWEEP_JUDGE, _SWEEP_HUMAN, _SWEEP_CONF,
        n_bins=2,
    )
    row = 1  # threshold=0.3 is the second unique value
    assert result["threshold"][row] == pytest.approx(0.3)
    assert result["accuracy"][row] == pytest.approx(0.75)
    assert result["kappa"][row] == pytest.approx(0.5)
    assert result["ece"][row] == pytest.approx(0.15, abs=1e-9)
    assert result["n_effective_bins"][row] == 2


def test_threshold_sweep_all_rows_reference():
    # Full table, all 4 rows - verified against the implementation itself
    # (ece()/cohens_kappa() already have their own independent reference
    # tests elsewhere in this file; this checks threshold_sweep()'s own
    # kept-mask/bookkeeping logic combines them correctly at every step).
    result = threshold_sweep(
        _SWEEP_SIGNAL, _SWEEP_CORRECT, _SWEEP_JUDGE, _SWEEP_HUMAN, _SWEEP_CONF,
        n_bins=2,
    )
    np.testing.assert_allclose(result["accuracy"], [0.5, 0.75, 2 / 3, 0.625])
    np.testing.assert_allclose(result["kappa"], [0.0, 0.5, 1 / 3, 0.25], atol=1e-9)
    np.testing.assert_allclose(result["ece"], [0.35, 0.15, 7 / 30, 0.2625], atol=1e-9)


def test_threshold_sweep_low_coverage_no_rows_dropped():
    # Explicit low-coverage policy check: no row is silently filtered out,
    # even the first threshold where only 2/8 items are kept and kappa
    # degenerates to 0.0 via cohens_kappa's own p_e>=1-eps guard rather than
    # raising or being dropped.
    result = threshold_sweep(
        _SWEEP_SIGNAL, _SWEEP_CORRECT, _SWEEP_JUDGE, _SWEEP_HUMAN, _SWEEP_CONF,
        n_bins=2,
    )
    assert len(result["threshold"]) == 4
    assert result["n_kept"][0] == 2
    assert not np.isnan(result["kappa"][0])
    assert not np.isnan(result["ece"][0])


# --- auto_accept_stats -------------------------------------------------------


def test_auto_accept_stats_hand_computed():
    # 8 items; threshold 0.9 accepts the first five (0.9 counts as accepted).
    # Wrong items: #1 (accepted), #3 (accepted), #6 and #7 (escalated).
    conf = [0.95, 0.99, 0.90, 0.92, 0.97, 0.60, 0.80, 0.50]
    correct = [True, False, True, False, True, True, False, False]
    stats = auto_accept_stats(conf, correct, threshold=0.9)
    assert stats["n_accepted"] == 5
    assert stats["n_escalated"] == 3
    assert stats["accepted_share"] == pytest.approx(5 / 8)
    assert stats["slip_through"] == pytest.approx(2 / 4)
    assert stats["error_among_accepted"] == pytest.approx(2 / 5)


def test_auto_accept_stats_empty_denominators_are_nan():
    none_accepted = auto_accept_stats([0.5, 0.6], [True, False], threshold=0.9)
    assert np.isnan(none_accepted["error_among_accepted"])
    assert none_accepted["slip_through"] == 0.0
    none_wrong = auto_accept_stats([0.95, 0.6], [True, True], threshold=0.9)
    assert np.isnan(none_wrong["slip_through"])


def test_auto_accept_stats_kappa_uses_accepted_items_only():
    conf = [0.95, 0.95, 0.95, 0.95, 0.5]
    correct = [True, True, False, False, False]
    verdict = ["A", "B", "A", "B", "A"]
    human = ["A", "B", "B", "A", "B"]
    stats = auto_accept_stats(conf, correct, 0.9, verdict=verdict, human_label=human)
    assert stats["kappa_among_accepted"] == pytest.approx(cohens_kappa(verdict[:4], human[:4]))
