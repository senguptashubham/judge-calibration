"""Tests for src/predictor.py: the repeated StratifiedGroupKFold
protocol (D8/CLAUDE.md invariant 1), feature encoding, and the
permutation null (invariant 12).
"""

import numpy as np
import pandas as pd
import pytest

from src.predictor import (
    encode_features,
    make_fold_splits,
    make_histgbm,
    make_logreg,
    percentile_of_null,
    permutation_null,
    repeated_stratified_group_kfold,
    run_predictor,
    shuffle_within_groups,
)


def _synthetic_rq4_data(n_groups: int = 30, rows_per_group: int = 3, seed: int = 0):
    """n_groups question_ids, each with rows_per_group items, a 2-feature
    matrix, and a target correlated with one feature (so AUROC has
    something real to detect, not just noise) - enough groups/rows for
    StratifiedGroupKFold(n_splits=5) to have both classes represented per
    fold.
    """
    rng = np.random.default_rng(seed)
    n = n_groups * rows_per_group
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, y, groups


def _synthetic_population(n_groups: int = 20, rows_per_group: int = 3, seed: int = 0) -> pd.DataFrame:
    """Tier A's exact column set, plus `correct`/`question_id` - enough
    to exercise run_predictor()'s full tier-builder -> encode -> CV
    wiring, not just the low-level repeated_stratified_group_kfold()
    function in isolation.
    """
    rng = np.random.default_rng(seed)
    n = n_groups * rows_per_group
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    conf = rng.uniform(0.5, 1.0, size=n)
    correct = (conf + rng.normal(scale=0.2, size=n) > 0.75).astype(bool)
    return pd.DataFrame(
        {
            "conf_verb": conf,
            "conf_lp": conf,
            "conf_sc": conf,
            "conf_bpe": conf,
            "conf_ens": conf,
            "ens_entropy_total": 1 - conf,
            "ens_entropy_aleatoric": (1 - conf) / 2,
            "ens_entropy_epistemic": (1 - conf) / 2,
            "correct": correct,
            "question_id": groups,
        }
    )


# --- make_fold_splits: D8's two mandatory properties -----------------


def test_no_question_id_in_both_train_and_test_of_any_fold():
    X, y, groups = _synthetic_rq4_data()
    splits = make_fold_splits(X, y, groups, n_splits=5, seed=0)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        assert train_groups.isdisjoint(test_groups)


def test_every_row_appears_in_exactly_one_folds_test_set():
    X, y, groups = _synthetic_rq4_data()
    splits = make_fold_splits(X, y, groups, n_splits=5, seed=0)
    all_test_idx = np.concatenate([test_idx for _, test_idx in splits])
    assert sorted(all_test_idx) == list(range(len(X)))


def test_two_different_seeds_produce_different_fold_assignments():
    X, y, groups = _synthetic_rq4_data()
    splits_a = make_fold_splits(X, y, groups, n_splits=5, seed=0)
    splits_b = make_fold_splits(X, y, groups, n_splits=5, seed=1)
    test_sets_a = [frozenset(test_idx) for _, test_idx in splits_a]
    test_sets_b = [frozenset(test_idx) for _, test_idx in splits_b]
    assert test_sets_a != test_sets_b


def test_same_seed_is_deterministic():
    X, y, groups = _synthetic_rq4_data()
    splits_a = make_fold_splits(X, y, groups, n_splits=5, seed=42)
    splits_b = make_fold_splits(X, y, groups, n_splits=5, seed=42)
    for (train_a, test_a), (train_b, test_b) in zip(splits_a, splits_b):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(test_a, test_b)


# --- repeated_stratified_group_kfold ------------------------------------


def test_repeated_cv_returns_one_result_per_repeat_with_no_leftover_nans():
    X, y, groups = _synthetic_rq4_data()
    results = repeated_stratified_group_kfold(X, y, groups, make_logreg, n_splits=5, n_repeats=10, seed=0)
    assert len(results) == 10
    assert [r.seed for r in results] == list(range(10))
    for r in results:
        assert not np.isnan(r.oof_pred).any()
        assert 0.0 <= r.auroc <= 1.0


def test_repeated_cv_uses_a_fresh_seed_per_repeat_so_repeats_differ():
    X, y, groups = _synthetic_rq4_data()
    results = repeated_stratified_group_kfold(X, y, groups, make_logreg, n_splits=5, n_repeats=3, seed=0)
    assert [r.seed for r in results] == [0, 1, 2]
    assert not np.array_equal(results[0].oof_pred, results[1].oof_pred)


def test_repeated_cv_works_with_histgbm_too():
    X, y, groups = _synthetic_rq4_data()
    results = repeated_stratified_group_kfold(X, y, groups, make_histgbm, n_splits=5, n_repeats=2, seed=0)
    assert len(results) == 2
    for r in results:
        assert not np.isnan(r.oof_pred).any()


# --- encode_features ---------------------------------------------------


def test_encode_features_one_hot_encodes_category():
    X = pd.DataFrame({"category": ["writing", "coding", "writing"], "conf_verb": [0.9, 0.8, 0.7]})
    encoded = encode_features(X)
    assert "category" not in encoded.columns
    assert "category_writing" in encoded.columns
    assert "category_coding" in encoded.columns
    assert list(encoded["category_writing"]) == [1, 0, 1]


def test_encode_features_casts_booleans_to_int():
    X = pd.DataFrame({"longer_is_chosen": [True, False], "flipped": [False, True]})
    encoded = encode_features(X)
    assert encoded["longer_is_chosen"].dtype.kind in "iu"
    assert list(encoded["longer_is_chosen"]) == [1, 0]
    assert list(encoded["flipped"]) == [0, 1]


def test_encode_features_is_a_no_op_on_tier_a_style_columns():
    X = pd.DataFrame({"conf_verb": [0.9, 0.8], "conf_lp": [0.7, 0.6]})
    encoded = encode_features(X)
    pd.testing.assert_frame_equal(encoded, X)


# --- run_predictor -----------------------------------------------------


def test_run_predictor_rejects_unknown_model_name():
    population = _synthetic_population(n_groups=5, rows_per_group=2)
    with pytest.raises(ValueError):
        run_predictor(population, lambda df: df[["conf_verb"]], "not_a_real_model", seed=0)


def test_run_predictor_full_wiring_tier_a_logreg():
    population = _synthetic_population()
    results = run_predictor(
        population, lambda df: df[["conf_verb", "conf_lp", "conf_sc", "conf_bpe"]], "logreg", seed=0
    )
    assert len(results) == 10
    for r in results:
        assert not np.isnan(r.oof_pred).any()
        assert 0.0 <= r.auroc <= 1.0


# --- permutation_null / percentile_of_null (invariant 12) --------------


def test_permutation_null_returns_n_values_in_range():
    X, y, groups = _synthetic_rq4_data()
    null = permutation_null(X, y, groups, make_logreg, n=15, seed=0)
    assert null.shape == (15,)
    assert np.all((null >= 0.0) & (null <= 1.0))


def test_permutation_null_centers_near_chance_even_with_a_real_signal():
    # A strong real X-y relationship (the unshuffled fit scores well
    # above 0.5) - shuffling y must still destroy it completely, or the
    # grouping/fold structure is leaking the real relationship back in.
    X, y, groups = _synthetic_rq4_data(n_groups=40, rows_per_group=3, seed=1)
    null = permutation_null(X, y, groups, make_logreg, n=30, seed=0)
    assert 0.3 < null.mean() < 0.7


def test_permutation_null_uses_independent_permutations_not_one_repeated_shuffle():
    X, y, groups = _synthetic_rq4_data()
    null = permutation_null(X, y, groups, make_logreg, n=10, seed=0)
    # A shuffle-that-doesn't-shuffle bug would make every permutation
    # identical - confirm real variation exists across permutations.
    assert len(set(np.round(null, 6))) > 1


def test_shuffle_within_groups_keeps_each_groups_labels_and_moves_them():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 1, 1])
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    shuffled = [shuffle_within_groups(y, groups, np.random.default_rng(s)) for s in range(20)]
    for permuted in shuffled:
        for g in np.unique(groups):
            assert sorted(permuted[groups == g]) == sorted(y[groups == g])
    assert any(not np.array_equal(p, y) for p in shuffled)


def test_percentile_of_null_basic():
    null = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    assert percentile_of_null(0.75, null) == pytest.approx(0.6)  # 3/5 <= 0.75
    assert percentile_of_null(1.0, null) == pytest.approx(1.0)
    assert percentile_of_null(0.0, null) == pytest.approx(0.0)
