"""Tests for src/bayesian.py: a held-out question's random intercept must be
marginalized over the population prior, never its would-be fitted value
(DECISIONS.md D22) - the hierarchical-model analogue of test_predictor.py's
no-leakage assertion. Also convergence-diagnostics presence (R-hat, ESS,
divergence count) for every fold-fit. See TASKS.md task 5.9b.
"""

import numpy as np
import pandas as pd
import pytest

from src.bayesian import (
    build_group_index,
    convergence_diagnostics,
    fit_nuts,
    posterior_predictive_entropy_decomposition,
    predict_held_out,
    predict_in_sample,
    repeated_stratified_group_kfold_bayesian,
)


def _tiny_synthetic_data(seed: int = 0):
    """8 questions x 5 rows, a 3-feature X with a real, planted
    relationship to y - enough for NUTS to have something non-trivial to
    fit. Shared by fit_nuts's and convergence_diagnostics's tests so
    every test below is exercising hierarchical_logit_model too (no
    separate direct test for that function - it's a NumPyro model
    function, meant to be called BY fit_nuts, not directly; testing it
    only through fit_nuts mirrors how predictor.py's make_logreg/
    make_histgbm are only ever tested through the CV functions that use
    them, never in isolation).
    """
    rng = np.random.default_rng(seed)
    n_groups_true = 8
    rows_per_group = 5
    n = n_groups_true * rows_per_group
    question_ids = np.repeat(np.arange(n_groups_true), rows_per_group)
    X = rng.normal(size=(n, 3))
    true_beta = np.array([1.5, -0.5, 0.2])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X @ true_beta)))).astype(int)
    group_idx, n_groups, _ = build_group_index(question_ids)
    return X, y, group_idx, n_groups


# --- build_group_index: the dense-index mapping every fold rebuilds ----


def test_normal_case_dense_aligned_codes():
    # One fixture, every core contract property in one go: n_groups
    # counts distinct ids, group_idx codes are dense 0..n_groups-1 and
    # positionally aligned to the input (same id -> same code wherever
    # it appears, agreeing with `mapping`) - this is what lets group_idx
    # sit as a row-aligned column right next to X/y, exactly like
    # StratifiedGroupKFold's own `groups` argument elsewhere.
    question_ids = np.array([70, 12, 70, 41, 12])
    group_idx, n_groups, mapping = build_group_index(question_ids)
    assert n_groups == 3
    assert set(group_idx) == set(mapping.values()) == set(range(n_groups))
    assert group_idx[0] == group_idx[2] == mapping[70]
    assert group_idx[1] == group_idx[4] == mapping[12]
    assert group_idx[3] == mapping[41]


def test_same_input_is_deterministic():
    question_ids = np.array([70, 12, 70, 41, 12])
    group_idx_a, n_groups_a, mapping_a = build_group_index(question_ids)
    group_idx_b, n_groups_b, mapping_b = build_group_index(question_ids)
    assert np.array_equal(group_idx_a, group_idx_b)
    assert n_groups_a == n_groups_b
    assert mapping_a == mapping_b


def test_held_out_question_id_is_not_a_key():
    # The leak-guard property itself: build the mapping from a
    # TRAINING-fold-only id array (D8's grouping already guarantees a
    # held-out question never appears here) and confirm looking it up
    # fails loudly (KeyError) instead of silently returning something.
    # This is what makes the marginalization invariant hard to violate
    # by accident - see build_group_index()'s own docstring.
    train_question_ids = np.array([70, 70, 12, 41, 41])
    held_out_question_id = 99
    _, _, mapping = build_group_index(train_question_ids)
    assert held_out_question_id not in mapping


# --- fit_nuts (also exercises hierarchical_logit_model) -----------------


def test_fit_nuts_rejects_fewer_than_2_chains():
    # R-hat is undefined with 1 chain - NumPyro doesn't error on this
    # itself, it silently returns NaN, so fit_nuts's own guard is what
    # makes this fail loudly instead (see fit_nuts's own docstring).
    X, y, group_idx, n_groups = _tiny_synthetic_data()
    with pytest.raises(ValueError):
        fit_nuts(X, y, group_idx, n_groups, seed=0, num_chains=1)


def test_fit_nuts_returns_expected_sample_shapes():
    X, y, group_idx, n_groups = _tiny_synthetic_data()
    mcmc = fit_nuts(X, y, group_idx, n_groups, seed=0, num_warmup=20, num_samples=20, num_chains=2)
    samples = mcmc.get_samples()
    n_draws = 20 * 2  # num_samples * num_chains, warmup discarded
    assert samples["alpha"].shape == (n_draws,)
    assert samples["sigma_q"].shape == (n_draws,)
    assert samples["alpha_q"].shape == (n_draws, n_groups)
    assert samples["beta"].shape == (n_draws, X.shape[1])


def test_fit_nuts_seed_behavior():
    # Same seed -> identical posterior draws (reproducibility); different
    # seed -> different draws (confirms the seed is actually being used,
    # not silently ignored) - the JAX/NUTS analogue of test_predictor.py's
    # "two seeds produce different fold assignments" pair.
    X, y, group_idx, n_groups = _tiny_synthetic_data()
    mcmc_a = fit_nuts(X, y, group_idx, n_groups, seed=0, num_warmup=20, num_samples=20, num_chains=2)
    mcmc_b = fit_nuts(X, y, group_idx, n_groups, seed=0, num_warmup=20, num_samples=20, num_chains=2)
    mcmc_c = fit_nuts(X, y, group_idx, n_groups, seed=1, num_warmup=20, num_samples=20, num_chains=2)
    assert np.array_equal(mcmc_a.get_samples()["beta"], mcmc_b.get_samples()["beta"])
    assert not np.array_equal(mcmc_a.get_samples()["beta"], mcmc_c.get_samples()["beta"])


# --- convergence_diagnostics --------------------------------------------


def test_convergence_diagnostics_returns_expected_types_and_keys():
    X, y, group_idx, n_groups = _tiny_synthetic_data()
    mcmc = fit_nuts(X, y, group_idx, n_groups, seed=0, num_warmup=50, num_samples=50, num_chains=2)
    diag = convergence_diagnostics(mcmc)
    assert set(diag) == {"max_rhat", "min_ess", "n_divergences", "flagged"}
    assert isinstance(diag["max_rhat"], float)
    assert isinstance(diag["min_ess"], float)
    assert isinstance(diag["n_divergences"], int)
    assert isinstance(diag["flagged"], bool)
    assert diag["min_ess"] > 0
    assert diag["n_divergences"] >= 0


def test_convergence_diagnostics_flags_a_badly_mixed_fit():
    # 5 warmup/5 sample draws is nowhere near enough for NUTS to mix on
    # this model - deterministic given the fixed seed (both the
    # synthetic data's own rng and the fit's PRNGKey are seeded), not a
    # fit that sometimes happens to converge and sometimes doesn't.
    X, y, group_idx, n_groups = _tiny_synthetic_data()
    mcmc = fit_nuts(X, y, group_idx, n_groups, seed=0, num_warmup=5, num_samples=5, num_chains=2)
    diag = convergence_diagnostics(mcmc)
    assert diag["flagged"] is True


def test_convergence_diagnostics_flags_nan_rhat_even_with_no_divergences(monkeypatch):
    # A narrow edge case found during 5.9b's own review (21 Sep 2026),
    # not something a real tiny fit reliably reproduces on demand:
    # pandas' .max() already tolerates ONE parameter's NaN R-hat
    # gracefully (skipna=True), so this constructs an ALL-NaN r_hat
    # summary directly - the case where max_rhat itself ends up NaN -
    # with zero divergences, so only the explicit `or np.isnan(max_rhat)`
    # guard can catch it. Without that guard, `NaN > 1.01` is False in
    # Python and this fit would silently pass as not-flagged.
    X, y, group_idx, n_groups = _tiny_synthetic_data()
    mcmc = fit_nuts(X, y, group_idx, n_groups, seed=0, num_warmup=10, num_samples=10, num_chains=2)

    # az.from_numpyro() itself calls mcmc.get_extra_fields(group_by_chain=True)
    # internally (a different call signature than convergence_diagnostics's
    # own no-arg call below) - since az.summary is about to be fully mocked
    # anyway (it ignores whatever idata it's given), bypass from_numpyro too
    # rather than making one fake get_extra_fields satisfy two shapes.
    fake_summary = pd.DataFrame({"r_hat": [np.nan, np.nan], "ess_bulk": [50.0, 60.0], "ess_tail": [55.0, 65.0]})
    monkeypatch.setattr("arviz.from_numpyro", lambda mcmc: None)
    monkeypatch.setattr("arviz.summary", lambda idata: fake_summary)
    monkeypatch.setattr(mcmc, "get_extra_fields", lambda: {"diverging": np.zeros(20, dtype=bool)})

    diag = convergence_diagnostics(mcmc)
    assert np.isnan(diag["max_rhat"])
    assert diag["n_divergences"] == 0
    assert diag["flagged"] is True


# --- predict_held_out: the actual D22 marginalization invariant --------


@pytest.fixture(scope="module")
def fitted_mcmc_and_held_out():
    # Fit ONCE, shared read-only across every predict_held_out test below
    # (none of them mutate mcmc) - a real NUTS fit is the expensive part
    # here, no reason to pay for it 4 times over.
    X_train, y_train, train_group_idx, n_groups = _tiny_synthetic_data()
    mcmc = fit_nuts(X_train, y_train, train_group_idx, n_groups, seed=0, num_warmup=50, num_samples=50, num_chains=2)

    # 3 HELD-OUT questions (ids 100/101/102 - never in training), 4 rows
    # each, so within-question vs. across-question correlation is
    # actually checkable below.
    test_question_ids = np.repeat(np.array([100, 101, 102]), 4)
    rng = np.random.default_rng(1)
    X_test = rng.normal(size=(12, 3))
    test_group_idx, n_test_groups, _ = build_group_index(test_question_ids)

    return mcmc, X_test, test_group_idx, n_test_groups


def test_predict_held_out_shape_and_valid_probabilities(fitted_mcmc_and_held_out):
    mcmc, X_test, test_group_idx, n_test_groups = fitted_mcmc_and_held_out
    n_draws = 50 * 2
    probs = np.asarray(predict_held_out(mcmc, X_test, test_group_idx, n_test_groups, seed=0))
    assert probs.shape == (n_draws, X_test.shape[0])
    assert np.all((probs >= 0) & (probs <= 1))


def test_predict_held_out_seed_behavior(fitted_mcmc_and_held_out):
    # Same seed -> identical marginalized draws; different seed -> a
    # genuinely different fresh alpha_q_new draw, not silently ignored -
    # same pair-of-properties pattern as fit_nuts's own seed test.
    mcmc, X_test, test_group_idx, n_test_groups = fitted_mcmc_and_held_out
    probs_a = np.asarray(predict_held_out(mcmc, X_test, test_group_idx, n_test_groups, seed=0))
    probs_b = np.asarray(predict_held_out(mcmc, X_test, test_group_idx, n_test_groups, seed=0))
    probs_c = np.asarray(predict_held_out(mcmc, X_test, test_group_idx, n_test_groups, seed=1))
    assert np.array_equal(probs_a, probs_b)
    assert not np.array_equal(probs_a, probs_c)


def test_predict_held_out_shares_intercept_within_held_out_question(fitted_mcmc_and_held_out):
    # The actual marginalization property: rows belonging to the SAME
    # held-out question must share one alpha_q_new draw per posterior
    # sample, so their predicted probabilities should move together
    # across draws more than rows from DIFFERENT held-out questions do -
    # a per-row-independent (wrong) implementation would show no such
    # gap. Directional (>), not an exact value - exact correlation
    # numbers aren't the contract, the gap's existence is.
    mcmc, X_test, test_group_idx, n_test_groups = fitted_mcmc_and_held_out
    probs = np.asarray(predict_held_out(mcmc, X_test, test_group_idx, n_test_groups, seed=0))
    corr_same_question = np.corrcoef(probs[:, 0], probs[:, 1])[0, 1]  # both question 100
    corr_diff_question = np.corrcoef(probs[:, 0], probs[:, 4])[0, 1]  # question 100 vs 101
    assert corr_same_question > corr_diff_question


def test_predict_held_out_does_not_use_training_alpha_q(fitted_mcmc_and_held_out, monkeypatch):
    # Direct, mechanical proof of D22's own wording ("predictions do not
    # depend on that fold's fitted alpha_q"): strip "alpha_q" out of the
    # posterior samples dict predict_held_out sees, and confirm the
    # output is UNCHANGED. If the function secretly read training
    # alpha_q anywhere, this would raise a KeyError or change the
    # result; getting the identical answer without it proves the
    # function never touched it in the first place.
    mcmc, X_test, test_group_idx, n_test_groups = fitted_mcmc_and_held_out
    probs_with_alpha_q = np.asarray(predict_held_out(mcmc, X_test, test_group_idx, n_test_groups, seed=0))

    original_get_samples = mcmc.get_samples
    samples_without_alpha_q = {k: v for k, v in original_get_samples().items() if k != "alpha_q"}
    monkeypatch.setattr(mcmc, "get_samples", lambda *args, **kwargs: samples_without_alpha_q)

    probs_without_alpha_q = np.asarray(predict_held_out(mcmc, X_test, test_group_idx, n_test_groups, seed=0))
    assert np.array_equal(probs_with_alpha_q, probs_without_alpha_q)


# --- predict_in_sample: the real-alpha_q counterpart (task 5.9f) -------


def test_predict_in_sample_matches_manual_computation_with_real_alpha_q():
    # Direct, mechanical proof this function uses the REAL fitted
    # alpha_q, not a marginalized one - recompute the same quantity by
    # hand from the raw posterior samples and confirm an exact match.
    # This is the opposite property predict_held_out's own
    # does_not_use_training_alpha_q test checks.
    X, y, group_idx, n_groups = _tiny_synthetic_data()
    question_ids = np.repeat(np.arange(8), 5)  # matches _tiny_synthetic_data()'s own recipe
    mcmc = fit_nuts(X, y, group_idx, n_groups, seed=0, num_warmup=20, num_samples=20, num_chains=2)
    _, _, question_id_to_index = build_group_index(question_ids)

    probs = np.asarray(predict_in_sample(mcmc, X, question_ids, question_id_to_index))

    samples = mcmc.get_samples()
    manual_group_idx = np.array([question_id_to_index[q] for q in question_ids])
    manual_logits = (
        np.asarray(samples["alpha"])[:, None]
        + np.asarray(samples["alpha_q"])[:, manual_group_idx]
        + np.asarray(samples["beta"]) @ X.T
    )
    manual_probs = 1 / (1 + np.exp(-manual_logits))

    assert np.allclose(probs, manual_probs, atol=1e-5)


def test_predict_in_sample_raises_for_unmapped_question_id():
    # A row whose question_id isn't in question_id_to_index must fail
    # loudly (KeyError), not silently fabricate a prediction - same
    # "propagate, don't fabricate" convention build_group_index()'s own
    # leak-guard test establishes.
    X, y, group_idx, n_groups = _tiny_synthetic_data()
    question_ids = np.repeat(np.arange(8), 5)
    mcmc = fit_nuts(X, y, group_idx, n_groups, seed=0, num_warmup=20, num_samples=20, num_chains=2)
    _, _, question_id_to_index = build_group_index(question_ids)

    unmapped_question_ids = np.full(len(X), 999)
    with pytest.raises(KeyError):
        predict_in_sample(mcmc, X, unmapped_question_ids, question_id_to_index)


def test_predict_in_sample_is_fully_deterministic():
    # Unlike predict_held_out (which needs a seed for its marginalization
    # draw), this function has no randomness at all - same inputs must
    # give byte-identical output every time, with no seed argument to
    # even ask for reproducibility.
    X, y, group_idx, n_groups = _tiny_synthetic_data()
    question_ids = np.repeat(np.arange(8), 5)
    mcmc = fit_nuts(X, y, group_idx, n_groups, seed=0, num_warmup=20, num_samples=20, num_chains=2)
    _, _, question_id_to_index = build_group_index(question_ids)

    probs_a = np.asarray(predict_in_sample(mcmc, X, question_ids, question_id_to_index))
    probs_b = np.asarray(predict_in_sample(mcmc, X, question_ids, question_id_to_index))
    assert np.array_equal(probs_a, probs_b)


# --- repeated_stratified_group_kfold_bayesian: the D8 CV wrapper -------


def _tiny_cv_data(n_groups: int = 8, rows_per_group: int = 5, seed: int = 0):
    """Same generative recipe as _tiny_synthetic_data(), but shaped for
    repeated_stratified_group_kfold_bayesian()'s own contract: X as a
    DataFrame (matching predictor.py::build_xyg()'s output) and RAW
    question_ids, not build_group_index()'s pre-densified codes - the CV
    wrapper calls build_group_index() itself, per fold, so handing it
    already-dense codes here would exercise something subtly different
    from what its real caller (predictor.py's own (X, y, groups) triple)
    actually passes.
    """
    rng = np.random.default_rng(seed)
    n = n_groups * rows_per_group
    question_ids = np.repeat(np.arange(n_groups), rows_per_group)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f1", "f2", "f3"])
    true_beta = np.array([1.5, -0.5, 0.2])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X.to_numpy() @ true_beta)))).astype(int)
    return X, y, question_ids


def test_cv_wrapper_returns_one_result_per_repeat_with_no_leftover_nans():
    X, y, question_ids = _tiny_cv_data()
    results = repeated_stratified_group_kfold_bayesian(
        X, y, question_ids, n_splits=2, n_repeats=1, seed=0, num_warmup=10, num_samples=10, num_chains=2
    )
    assert len(results) == 1
    result = results[0]
    n_draws = 10 * 2
    assert result.oof_pred.shape == (len(X),)
    assert result.oof_draws.shape == (n_draws, len(X))
    assert not np.isnan(result.oof_pred).any()
    assert not np.isnan(result.oof_draws).any()
    assert 0.0 <= result.auroc <= 1.0
    assert len(result.fold_diagnostics) == 2  # n_splits
    for fold_diag in result.fold_diagnostics:
        assert set(fold_diag) == {"max_rhat", "min_ess", "n_divergences", "flagged"}


def test_cv_wrapper_uses_a_fresh_seed_per_repeat_so_repeats_differ():
    X, y, question_ids = _tiny_cv_data()
    results = repeated_stratified_group_kfold_bayesian(
        X, y, question_ids, n_splits=2, n_repeats=2, seed=0, num_warmup=10, num_samples=10, num_chains=2
    )
    assert [r.seed for r in results] == [0, 1]
    assert not np.array_equal(results[0].oof_pred, results[1].oof_pred)


# --- posterior_predictive_entropy_decomposition -------------------------


def test_matches_hand_computed_values():
    # 2 draws x 2 items, chosen so item 1 is item 0's mirror (p <-> 1-p) -
    # entropy is symmetric under that swap, so both items should land on
    # the SAME total/aleatoric/epistemic, a useful cross-check alongside
    # the raw numbers themselves.
    draws = np.array([[0.9, 0.1], [0.7, 0.3]])
    result = posterior_predictive_entropy_decomposition(draws)
    assert result["total"] == pytest.approx([0.500402, 0.500402], abs=1e-5)
    assert result["aleatoric"] == pytest.approx([0.467974, 0.467974], abs=1e-5)
    assert result["epistemic"] == pytest.approx([0.032429, 0.032429], abs=1e-5)


def test_epistemic_is_always_nonnegative():
    # Jensen's inequality (this function's own docstring) - must hold for
    # ANY draws, not just a hand-picked example.
    rng = np.random.default_rng(0)
    draws = rng.uniform(0.01, 0.99, size=(50, 20))
    result = posterior_predictive_entropy_decomposition(draws)
    assert np.all(result["epistemic"] >= -1e-10)  # float slack around exactly 0


def test_epistemic_is_zero_when_draws_agree_perfectly():
    # No disagreement across draws (every draw gives the same p for an
    # item) means Total == Aleatoric by construction - epistemic should
    # collapse to (near) exactly 0, not just "small".
    draws = np.full((30, 5), 0.37)
    result = posterior_predictive_entropy_decomposition(draws)
    assert result["epistemic"] == pytest.approx(np.zeros(5), abs=1e-9)
