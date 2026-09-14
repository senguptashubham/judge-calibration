"""Tests for src/metrics.py, including the hand-computed ECE reference case
(CLAUDE.md sec 5: expected 0.222) and the discrete-signal ECE case
(DECISIONS.md D14). See TASKS.md tasks 0.4, 0.6, 2.5, 3.1.
"""

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from src.metrics import (
    auroc_error,
    brier,
    brier_decomposition,
    cohens_kappa,
    ece,
    mce,
    overconfidence_gap,
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


# --- TASKS.md task 2.5 ------------------------------------------------------
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
    pass  # TODO(owner): expected 0.222 - see note above


def test_mce_reference():
    pass  # TODO(owner): expected 0.27 - see note above


def test_brier_reference():
    pass  # TODO(owner): expected 0.2604 - see note above


def test_brier_decomposition_reconstructs_brier_score():
    # CLAUDE.md task 2.5's DoD: reliability - resolution + uncertainty must
    # reconstruct brier() to 1e-6. This is the important test in this
    # block - it catches sign/assignment bugs the individual-term checks
    # below can't.
    pass  # TODO(owner)


def test_brier_decomposition_reference_terms():
    pass  # TODO(owner): expected (0.05274, 0.042336, 0.249996)


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
