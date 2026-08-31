"""Tests for src/metrics.py, including the hand-computed ECE reference case
(CLAUDE.md sec 5: expected 0.222) and the discrete-signal ECE case
(DECISIONS.md D14). See TASKS.md tasks 0.4, 0.6, 2.5, 3.1.
"""

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from src.metrics import cohens_kappa, ece


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
