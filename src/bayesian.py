"""RQ4: Bayesian hierarchical logistic regression (NumPyro/NUTS) with
question-level random intercepts, fit under the same repeated
StratifiedGroupKFold protocol as src/predictor.py (D22).

Preregistered fallback ladder: NUTS -> Laplace approximation -> bootstrap
ensemble of logistic regressions. NUTS was sufficient; the other rungs were
never built.

Also the meta-model-level Total/Aleatoric/Epistemic entropy decomposition
from posterior predictive draws - the counterpart of signals.py's
judge-level conf_ens decomposition (D20, D22). Runs locally on CPU (D24).
"""

import argparse
import time
from dataclasses import dataclass

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.features import load_rq4_population
from src.predictor import TIER_BUILDERS, build_xyg, make_fold_splits


def build_group_index(question_ids: np.ndarray) -> tuple[np.ndarray, int, dict[int, int]]:
    """Maps raw question_id values to dense codes 0..n_groups-1, which is
    how the model's `alpha_q` array is indexed.

    Must be rebuilt per fold from that fold's TRAINING question_ids only.
    Then a held-out question is simply not a key - looking it up raises
    KeyError instead of silently reading an intercept it should never have
    (the leak D22 exists to prevent; test_bayesian.py asserts it).

    Returns:
        group_idx: dense per-row codes, same order as question_ids.
        n_groups: distinct questions in this set.
        question_id_to_index: the mapping itself (needed by predict_in_sample).
    """
    df = pd.DataFrame({"question_ids": question_ids})
    group_idx, uniques = df["question_ids"].factorize()
    n_groups = len(uniques)
    question_id_to_index = dict(zip(uniques, range(n_groups)))
    return group_idx, n_groups, question_id_to_index


def hierarchical_logit_model(
    X: np.ndarray, group_idx: np.ndarray, n_groups: int, y: np.ndarray | None = None
) -> None:
    """The generative model (D22):

        correct ~ Bernoulli(sigmoid(alpha + alpha_q[question] + X @ beta))
        alpha   ~ Normal(0, 1)        global intercept
        sigma_q ~ HalfNormal(1)       spread of question intercepts
        alpha_q ~ Normal(0, sigma_q)  one per question - partial pooling
        beta    ~ Normal(0, 1)        one per (standardized) feature

    sigma_q is learned: small sigma_q shrinks every alpha_q toward 0 (a
    plain logit model); large sigma_q lets questions differ. That is the
    partial-pooling mechanism. D22 fixes the priors for sigma_q/alpha_q/beta;
    alpha's Normal(0, 1) matches beta's scale.

    Non-centered parameterization: the sampled site is
    alpha_q_raw ~ Normal(0, 1), with alpha_q = sigma_q * alpha_q_raw as a
    deterministic transform - the same prior, but alpha_q is decoupled from
    sigma_q in the sampled space. The centered form produces Neal's funnel
    (when sigma_q is small the plausible alpha_q region narrows sharply and
    NUTS's step size can't follow); on real data it gave R-hat 1.07 and
    ESS 21, against 1.01 and 791 non-centered (D22 amendment).

    Handed to NUTS (y given, fitting) or Predictive (y=None, sampling); the
    sample sites are the output. Uses jax.numpy throughout, since the
    inputs are JAX tracers during MCMC.
    """
    X = jnp.asarray(X)
    group_idx = jnp.asarray(group_idx)

    alpha = numpyro.sample("alpha", dist.Normal(0, 1))
    sigma_q = numpyro.sample("sigma_q", dist.HalfNormal(1))

    with numpyro.plate("questions", n_groups):
        alpha_q_raw = numpyro.sample("alpha_q_raw", dist.Normal(0, 1))
        alpha_q = numpyro.deterministic("alpha_q", sigma_q * alpha_q_raw)

    with numpyro.plate("features", X.shape[1]):
        beta = numpyro.sample("beta", dist.Normal(0, 1))

    logits = alpha + alpha_q[group_idx] + X @ beta

    with numpyro.plate("data", X.shape[0]):
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


def fit_nuts(
    X: np.ndarray,
    y: np.ndarray,
    group_idx: np.ndarray,
    n_groups: int,
    seed: int,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 2,
) -> MCMC:
    """Fits hierarchical_logit_model() with NUTS on one fold's training
    data. Returns the MCMC object itself - diagnostics need more than the
    samples dict.

    At least 2 chains: R-hat compares between-chain to within-chain
    variance, and with one chain NumPyro silently returns NaN R-hats,
    which would make the convergence check quietly meaningless. Chains run
    sequentially; parallel chains need process-level JAX setup a per-fold
    function can't own.
    """
    if num_chains < 2:
        raise ValueError("Minimum 2 chains required to compare variance")

    kernel = NUTS(hierarchical_logit_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential",
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), X=X, group_idx=group_idx, n_groups=n_groups, y=y)
    return mcmc


def convergence_diagnostics(mcmc: MCMC) -> dict:
    """Convergence diagnostics for one fit (D22, invariant 13) - reported
    for every fold, never optional.

    max_rhat is the max over every scalar parameter (each alpha_q[i],
    beta[j] separately): one badly mixed intercept should flag the fit, not
    be averaged away. min_ess is the minimum over both bulk and tail ESS.
    Flagged when max_rhat > 1.01, max_rhat is NaN, or any divergence
    occurred. This only reports - what to do with a flagged fold is the
    caller's decision.
    """
    idata = az.from_numpyro(mcmc)
    summary = az.summary(idata)
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(min(summary["ess_bulk"].min(), summary["ess_tail"].min()))
    n_divergences = int(mcmc.get_extra_fields()["diverging"].sum())
    # .max() skips NaN, so max_rhat is NaN only when EVERY R-hat is NaN - a
    # degenerate fit - and `NaN > 1.01` is False, hence the explicit isnan.
    # bool(...) wraps the whole expression so a numpy.bool_ never leaks out.
    flagged = bool(max_rhat > 1.01 or np.isnan(max_rhat) or n_divergences > 0)
    return {"max_rhat": max_rhat, "min_ess": min_ess, "n_divergences": n_divergences, "flagged": flagged}


def predict_held_out(
    mcmc: MCMC, X_test: np.ndarray, test_group_idx: np.ndarray, n_test_groups: int, seed: int
) -> jax.Array:
    """P(correct) for questions ABSENT from training, with their intercepts
    marginalized over the population prior (D22, invariant 13).

    For every posterior draw d, a fresh alpha_q_new[d, g] ~ Normal(0,
    sigma_q[d]) is drawn per held-out question g - one per (draw, question),
    not per row, so rows of the same question share it, exactly as training
    rows share alpha_q. alpha, sigma_q and beta come from the fitted
    posterior. test_group_idx comes from a separate build_group_index() on
    the test rows and only groups rows by question - it never indexes the
    training posterior.

    Use only for genuinely held-out questions (cross-validation). For rows
    whose questions WERE in training (the verbose-shift check), use
    predict_in_sample(): marginalizing a real fitted intercept away both
    discards information and inflates spread on any input (D21 amendment).

    Returns (n_draws, n_test_rows) - every draw kept, not averaged.
    """
    samples = mcmc.get_samples()
    alpha = samples["alpha"]  # (n_draws,)
    sigma_q = samples["sigma_q"]  # (n_draws,)
    beta = samples["beta"]  # (n_draws, n_features)
    n_draws = alpha.shape[0]

    X_test = jnp.asarray(X_test)
    test_group_idx = jnp.asarray(test_group_idx)

    key = jax.random.PRNGKey(seed)
    alpha_q_new = sigma_q[:, None] * jax.random.normal(key, shape=(n_draws, n_test_groups))

    logits = alpha[:, None] + alpha_q_new[:, test_group_idx] + beta @ X_test.T
    return jax.nn.sigmoid(logits)


def predict_in_sample(
    mcmc: MCMC, X: np.ndarray, row_question_ids: np.ndarray, question_id_to_index: dict[int, int]
) -> jax.Array:
    """P(correct) using each row's REAL fitted alpha_q - for rows whose
    questions were in training under a different condition (the
    verbose-shift check: clean and verbose share all 80 questions).

    Keeping the same fitted alpha_q for both conditions means only the
    feature values differ between them, which is the variable the test is
    about (D21 amendment). `question_id_to_index` must be the TRAINING
    mapping; a fresh build_group_index() on the evaluation rows would number
    questions differently. An unmapped question raises KeyError.

    Returns (n_draws, n_rows).
    """
    samples = mcmc.get_samples()
    alpha = samples["alpha"]  # (n_draws,)
    alpha_q = samples["alpha_q"]  # (n_draws, n_groups) - fitted values, never marginalized
    beta = samples["beta"]  # (n_draws, n_features)

    X = jnp.asarray(X)
    group_idx = jnp.asarray([question_id_to_index[qid] for qid in row_question_ids])

    logits = alpha[:, None] + alpha_q[:, group_idx] + beta @ X.T
    return jax.nn.sigmoid(logits)


@dataclass
class BayesianRepeatResult:
    """One repeat of repeated_stratified_group_kfold_bayesian(). oof_pred
    and oof_draws are positional against X.iloc, like RepeatResult.
    """

    seed: int
    oof_pred: np.ndarray  # (n_rows,) mean P(correct)
    oof_draws: np.ndarray  # (n_draws, n_rows) every posterior draw's P(correct)
    auroc: float
    fold_diagnostics: list[dict]  # one convergence_diagnostics() dict per fold


def repeated_stratified_group_kfold_bayesian(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 2,
) -> list[BayesianRepeatResult]:
    """D8's repeated-CV protocol with the hierarchical model in place of
    LogReg/HistGBM, reusing predictor.py's make_fold_splits() unchanged.

    Per fold:
    - Features are standardized with the TRAINING fold's mean/std, the same
      leakage guard make_logreg()'s Pipeline gives. beta ~ Normal(0, 1) only
      means the same thing for every feature once they share a scale.
    - build_group_index() runs separately on the train and test question_ids,
      so a held-out question can never reach a fitted intercept.
    - Convergence diagnostics are collected for every fit; flagged fits are
      reported, never dropped here.

    Every fold of a repeat uses that repeat's seed, as in predictor.py.
    Accepts the same (X, y, groups) as predictor.py::build_xyg().
    """
    results = []
    for i in range(n_repeats):
        repeat_seed = seed + i
        oof_pred = np.full(len(X), np.nan)
        oof_draws = np.full((num_samples * num_chains, len(X)), np.nan)
        fold_diagnostics = []

        for train_idx, test_idx in make_fold_splits(X, y, groups, n_splits, repeat_seed):
            scaler = StandardScaler().fit(X.iloc[train_idx])
            X_train, y_train = scaler.transform(X.iloc[train_idx]), y[train_idx]
            X_test = scaler.transform(X.iloc[test_idx])
            train_group_idx, n_groups, _ = build_group_index(groups[train_idx])
            test_group_idx, n_test_groups, _ = build_group_index(groups[test_idx])

            mcmc = fit_nuts(
                X_train,
                y_train,
                train_group_idx,
                n_groups,
                seed=repeat_seed,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
            )
            fold_diagnostics.append(convergence_diagnostics(mcmc))

            per_draw_probs = predict_held_out(mcmc, X_test, test_group_idx, n_test_groups, seed=repeat_seed)
            oof_draws[:, test_idx] = per_draw_probs
            oof_pred[test_idx] = per_draw_probs.mean(axis=0)

        auroc = float(roc_auc_score(y, oof_pred))
        results.append(BayesianRepeatResult(repeat_seed, oof_pred, oof_draws, auroc, fold_diagnostics))
    return results


def _binary_entropy_vec(p: np.ndarray) -> np.ndarray:
    """Vectorized binary entropy in nats. Clips p to [eps, 1 - eps] instead
    of branching per element, which keeps log() finite; the result at the
    boundary is negligibly different from the exact 0.
    """
    eps = 1e-12
    p_clipped = np.clip(p, eps, 1 - eps)
    return -(p_clipped * np.log(p_clipped) + (1 - p_clipped) * np.log(1 - p_clipped))


def posterior_predictive_entropy_decomposition(draws: np.ndarray) -> dict:
    """Meta-model-level entropy decomposition (D22) - conf_ens's formula
    (D20), with posterior draws in place of the three prompt variants:

        mean_p    = mean(draws, axis=0)    per item
        Total     = H(mean_p)
        Aleatoric = mean(H(draws), axis=0)
        Epistemic = Total - Aleatoric       >= 0 by Jensen's inequality

    Epistemic is disagreement between posterior draws - parameter
    uncertainty the model hasn't resolved. Entropy is symmetric in p and
    1 - p, so draws of P(correct) and of P(wrong) give the same result.

    Args:
        draws: (n_draws, n_items).

    Returns:
        dict of total, aleatoric, epistemic - each (n_items,).
    """
    draws_arr = np.asarray(draws, dtype=float)
    mean_p = draws_arr.mean(axis=0)
    total = _binary_entropy_vec(mean_p)
    aleatoric = _binary_entropy_vec(draws_arr).mean(axis=0)
    epistemic = total - aleatoric
    return {"total": total, "aleatoric": aleatoric, "epistemic": epistemic}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", required=True, choices=list(TIER_BUILDERS))
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-chains", type=int, default=2)
    parser.add_argument("--n-repeats", type=int, default=10, help="D8's protocol default; reduce only if the "
        "runtime measurement below shows the full count is impractical (D22) - preregister the reduction if so.")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    population = load_rq4_population(config.paths.items_parquet)
    X, y, groups = build_xyg(population, TIER_BUILDERS[args.tier])

    print(
        f"Tier {args.tier}, Bayesian hierarchical logit (non-centered, D22 amended 21 Sep 2026), "
        f"N={len(population)}, {args.n_repeats} repeats, "
        f"num_warmup={args.num_warmup}, num_samples={args.num_samples}, num_chains={args.num_chains}"
    )

    start = time.perf_counter()
    results = repeated_stratified_group_kfold_bayesian(
        X,
        y,
        groups,
        n_repeats=args.n_repeats,
        seed=config.seed,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
    )
    elapsed = time.perf_counter() - start

    aurocs = np.array([r.auroc for r in results])
    print(f"  AUROC mean={aurocs.mean():.4f}, spread=[{aurocs.min():.4f}, {aurocs.max():.4f}]")
    print("  (D8: the across-repeat spread is the headline uncertainty, not a within-split CI)")

    all_diag = [fold_diag for r in results for fold_diag in r.fold_diagnostics]
    n_flagged = sum(fold_diag["flagged"] for fold_diag in all_diag)
    worst_rhat = max(fold_diag["max_rhat"] for fold_diag in all_diag)
    total_divergences = sum(fold_diag["n_divergences"] for fold_diag in all_diag)
    print(
        f"  Convergence (D22, invariant 13): {n_flagged}/{len(all_diag)} fold-fits flagged "
        f"(worst max_rhat={worst_rhat:.3f}, total divergences={total_divergences})"
    )

    print(f"  Runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min) for {len(all_diag)} fold-fits")
