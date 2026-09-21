"""RQ4: Bayesian hierarchical logistic regression (NumPyro/NUTS), with
question-level random intercepts, fit under the same repeated
StratifiedGroupKFold protocol as src/predictor.py.

Fallback ladder, preregistered: NUTS -> Laplace approximation -> bootstrap
ensemble of logistic regressions (DECISIONS.md D22).

Also home to the meta-model-level Total/Aleatoric/Epistemic entropy
decomposition, computed from posterior predictive draws (DECISIONS.md D20,
D22) - the counterpart to signals.py's judge-level (conf_ens) decomposition.

Runs entirely locally (CPU) - no GPU, no Colab (DECISIONS.md D24).
See TASKS.md tasks 5.9b-5.9f.
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

from src.config import Config
from src.features import load_rq4_population
from src.predictor import TIER_BUILDERS, build_xyg, make_fold_splits


def build_group_index(question_ids: np.ndarray) -> tuple[np.ndarray, int, dict[int, int]]:
    """Maps raw `question_id` values to dense codes in [0, n_groups) -
    what the hierarchical model's per-question random intercept `alpha_q`
    is actually indexed by (D22).

    Why dense codes at all: NumPyro's `numpyro.plate("questions",
    n_groups)` holds `alpha_q` as one fixed-size array of length
    n_groups, addressed by plain integer position (`alpha_q[group_idx]`
    inside the model). It has no notion of "question_id" as a concept -
    question_id is just an arbitrary MT-Bench id, not a contiguous range
    starting at 0, so it can't be used as an index directly.

    Why this must be rebuilt FRESH per fold, from that fold's TRAINING
    question_ids only, rather than once for the whole ~80-question
    population and reused everywhere: a held-out question must never be
    a valid index into the fitted `alpha_q` array, or predicting on it
    would silently read that question's own fitted intercept instead of
    marginalizing over the population prior (D22) - exactly the same
    leak GroupKFold's missing shuffle causes for the frequentist model
    (invariant 1), one level deeper. Building the mapping from
    training-fold ids alone makes this structural rather than a
    discipline someone has to remember: a held-out id is simply never a
    key, so looking it up raises KeyError instead of returning a
    plausible-looking wrong number (test_bayesian.py's
    test_held_out_question_id_is_not_a_key asserts exactly this).

    Args:
        question_ids: one fold's training-only question_id array, one
            entry per row, positionally aligned to that fold's X/y - the
            same role `groups` already plays throughout predictor.py.

    Returns:
        group_idx: dense per-row codes, same order/length as
            question_ids - what gets passed into the NumPyro model.
        n_groups: number of distinct questions in THIS fold's training
            set (not 80 - a fold only trains on ~64 of the 80 questions,
            the rest are held out that fold).
        question_id_to_index: the mapping itself, real question_id ->
            dense code - not needed to fit the model, but what makes the
            leak-guard property above directly testable.
    """
    df = pd.DataFrame({"question_ids": question_ids})
    group_idx, uniques = df["question_ids"].factorize()
    n_groups = len(uniques)
    question_id_to_index = dict(zip(uniques, range(n_groups)))
    return group_idx, n_groups, question_id_to_index


def hierarchical_logit_model(
    X: np.ndarray, group_idx: np.ndarray, n_groups: int, y: np.ndarray | None = None
) -> None:
    """The generative model itself (D22): `correct ~ Bernoulli(sigma(alpha
    + alpha_q[question] + X @ beta))`, question-level random intercepts
    with partial pooling. Not called directly - handed to `NUTS`/`MCMC`
    (fitting, `y` given) or `Predictive` (posterior-predictive sampling,
    `y=None`) - both call this function repeatedly, tracing every
    `numpyro.sample` call to build the probability graph. Nothing about
    its return value matters; the sample sites ARE the output.

    NON-CENTERED parameterization for alpha_q, not the naive direct form
    D22's formula might suggest - `alpha_q_raw ~ Normal(0, 1)` is the
    ACTUAL sample site, and `alpha_q = sigma_q * alpha_q_raw` is a
    `numpyro.deterministic` transform, mathematically the same prior
    (alpha_q is still Normal(0, sigma_q) in distribution) but with
    alpha_q decoupled from sigma_q in the SAMPLED parameters. This is
    the standard fix for Neal's funnel: the centered form (sampling
    alpha_q directly as Normal(0, sigma_q)) makes alpha_q and sigma_q
    highly correlated in the posterior geometry - when sigma_q is small,
    the region of plausible alpha_q values narrows into a funnel NUTS's
    fixed step size struggles to navigate, and more warmup/samples alone
    doesn't reliably fix it (a geometry problem, not a sample-count one).
    MEASURED, not assumed (D22's own "measure, don't guess" principle):
    on this project's real Tier A / fold-0 data at num_warmup=500,
    num_samples=1000, num_chains=2, the centered form gave max_rhat=1.07
    (flagged) with min_ess=21; the non-centered form above, SAME
    settings, gave max_rhat=1.01 (not flagged) with min_ess=791 - roughly
    38x better effective sample size for a real, ~1.6x, not 38x, runtime
    cost (21 Sep 2026 measurement). alpha_q stays a
    `numpyro.deterministic` site (not just an internal Python variable)
    specifically so it's still directly inspectable in
    `mcmc.get_samples()`/az diagnostics under its own name, even though
    it's derived rather than sampled.

    Priors:
      alpha ~ Normal(0, 1)     - global intercept. D22's own formula names
      `alpha` but never states its prior (only alpha_q/sigma_q/beta are
      preregistered) - Normal(0, 1) is used here to match beta's own
      weakly-informative scale, not because the spec mandates this exact
      number.
      sigma_q ~ HalfNormal(1)  - half-normal because it's a standard
      deviation (positive support only).
      alpha_q_raw ~ Normal(0, 1), one per question, inside a `n_groups`-
      sized plate; alpha_q = sigma_q * alpha_q_raw (deterministic, see
      above) - sigma_q is LEARNED from the data (how much do questions
      actually differ), which is the partial-pooling mechanism itself:
      small sigma_q shrinks every alpha_q toward 0 (collapsing toward a
      plain, non-hierarchical logit model); large sigma_q lets them move
      more freely.
      beta ~ Normal(0, 1), one per feature, inside a `n_features`-sized
      plate - fixed scale, NOT tied to sigma_q (no group structure to pool
      across for feature coefficients), playing roughly the same "don't let
      coefficients blow up" role predictor.py's C=1.0 plays for the
      frequentist model.

    The three plates (questions/features/data) are siblings, not nested -
    they're three unrelated conditional-independence axes with different
    sizes (n_groups, n_features, n_rows), not a multi-dimensional
    structure where one varies per-unit-of-another.

    Every array op here uses `jax.numpy`, never plain `numpy`: X/alpha_q/
    beta are JAX tracer objects during MCMC, not ordinary arrays, and a
    bare numpy op on a tracer breaks gradient tracing NUTS depends on.

    Site names (`"alpha"`, `"sigma_q"`, `"alpha_q_raw"`, `"alpha_q"`
    (deterministic), `"beta"`, `"obs"`) match D22's own notation as
    closely as the reparameterization allows - these become the keys in
    `mcmc.get_samples()` and in every downstream R-hat/ESS diagnostic, so
    matching the spec's names keeps that output directly readable.

    Args:
        X: feature matrix, (n_rows, n_features) - already numeric/encoded
            (predictor.py::encode_features()'s output), not a DataFrame.
        group_idx: build_group_index()'s dense per-row codes, (n_rows,),
            values in [0, n_groups) - selects each row's own alpha_q.
        n_groups: build_group_index()'s group count - sizes the
            "questions" plate.
        y: observed target (`correct`, 0/1), (n_rows,), or None. `y`
            given -> conditions the "obs" site (fitting). `y=None` -> the
            same site becomes a sampling target instead of an observation
            (posterior-predictive generation on new data).
    """
    X = jnp.asarray(X)
    group_idx = jnp.asarray(group_idx)

    alpha = numpyro.sample("alpha", dist.Normal(0, 1))
    sigma_q = numpyro.sample("sigma_q", dist.HalfNormal(1))

    with numpyro.plate("questions", n_groups):
        # Non-centered: see this function's own docstring for why (Neal's funnel).
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
    """Fits hierarchical_logit_model() via NUTS on ONE fold's training
    data (D22 - rung 1 of the preregistered fallback ladder, NUTS ->
    Laplace -> bootstrap ensemble; the other two rungs are only built if
    this one proves too slow/non-convergent, per the runtime-measurement
    task).

    `num_chains < 2` is rejected, not just defaulted away from: R-hat is
    mathematically undefined with a single chain (it compares between-
    chain to within-chain variance) - NumPyro doesn't error on
    num_chains=1, it silently returns NaN for every parameter's R-hat,
    which would make a convergence check quietly meaningless rather than
    loudly broken. Rejecting it here means that mistake fails at the fit
    call, not three functions later when a diagnostics report full of
    NaN looks like a passing check to someone not reading closely.

    `chain_method="sequential"` runs chains one after another - the
    simplest option, and deliberately not `"parallel"`/`"vectorized"`,
    both of which need `numpyro.set_host_device_count()` called once at
    PROCESS START before any JAX op runs anywhere - not something a
    per-fold function like this can safely own. Correctness first;
    revisit chain parallelism only once real runtime numbers (this
    file's own runtime-measurement task) show it's needed.

    `progress_bar=False` - this runs inside a ~50-fit repeated-CV loop
    (D8's same 5-fold x 10-seed protocol as the frequentist model); a
    live per-fit progress bar is noise at that call volume, not signal.

    Returns the full `MCMC` object, not just `mcmc.get_samples()` -
    downstream steps need more than the samples dict:
    `mcmc.get_extra_fields()["diverging"]` and `az.from_numpyro(mcmc)`
    (the convergence-diagnostics step) both operate on `mcmc` itself.

    Args:
        X: feature matrix, (n_rows, n_features), this fold's TRAINING
            rows only.
        y: observed `correct` target, (n_rows,), same rows as X.
        group_idx, n_groups: build_group_index()'s output on this SAME
            training-only question_id array - never the full population's.
        seed: converted to `jax.random.PRNGKey(seed)` here - JAX has no
            numpy-style global random state, so this is the one place
            that conversion has to happen (CLAUDE.md sec 5: no bare
            randomness, always an explicit seed).
        num_warmup, num_samples: sampler tuning knobs. Unlike D8's fixed
            5-fold/10-seed protocol, these have no preregistered value
            yet - the defaults here are a starting point for the
            runtime-measurement task, not a settled number.
        num_chains: minimum 2 (enforced above) - more chains give a
            better R-hat estimate at proportionally higher cost.

    Returns:
        The fitted MCMC object.
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
    """Convergence diagnostics for one fold's fit (D22) - computed and
    reported for EVERY fold, never optional, the same status invariant
    12's permutation null has for the frequentist model. A Bayesian
    result without these isn't reportable.

    `az.summary()` returns one row per SCALAR parameter component, not
    per named site - `alpha_q` (shape (n_groups,)) becomes `alpha_q[0]`,
    `alpha_q[1]`, ... as separate rows, same for `beta`. `max_rhat` takes
    the max across ALL of these individually, not a mean across them -
    one badly-mixed question-level intercept should flag the whole fit,
    not get diluted into an average that looks fine.

    Two ESS columns exist (`ess_bulk`, `ess_tail`) - they measure
    different things (bulk: how well the posterior's central tendency is
    estimated; tail: how well its extremes/quantiles are). D22 just says
    "effective sample size" without picking one, so `min_ess` takes the
    minimum across BOTH columns, across all parameters - the single
    worst-mixing number in the whole fit, not a choice that might
    understate a problem the other column would have caught.

    Divergence count and R-hat/ESS all come back as numpy/JAX scalar
    types from arviz/NumPyro, not plain Python ones - explicitly wrapped
    in `float(...)`/`int(...)` so the returned dict holds ordinary Python
    scalars, not types that behave like them until something (a CSV
    writer, a direct equality check) cares about the difference.

    This function only REPORTS `flagged` - it never raises or drops a
    fit itself. Deciding what to do with a flagged fold (retry with more
    warmup, fall back to Laplace, just record it) is the CV loop's job,
    not this one's - keeping this a pure, passive reporter is what makes
    it independently testable and reusable outside that loop too.

    Args:
        mcmc: fit_nuts()'s returned, already-fitted MCMC object.

    Returns:
        dict with max_rhat, min_ess, n_divergences, and flagged (True if
        max_rhat > 1.01, max_rhat is NaN, or n_divergences > 0 - D22's
        threshold plus an explicit NaN guard, see below).
    """
    idata = az.from_numpyro(mcmc)
    summary = az.summary(idata)
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(min(summary["ess_bulk"].min(), summary["ess_tail"].min()))
    n_divergences = int(mcmc.get_extra_fields()["diverging"].sum())
    # pandas' .max() skips NaN by default, so ONE parameter with a NaN
    # R-hat is already handled gracefully (the real worst value among
    # the rest still wins). The gap this guards is narrower: every
    # parameter's R-hat coming back NaN at once (a genuinely degenerate
    # fit) leaves max_rhat itself NaN, and `NaN > 1.01` is False in
    # Python - silently NOT flagging a fit that deserves it more than an
    # ordinary elevated R-hat does. Caught via a real (tiny num_warmup)
    # fit's arviz warnings during 5.9b's own review, 21 Sep 2026 - not
    # a hypothetical.
    # bool(...) wraps the WHOLE expression, not just the isnan() term:
    # `or` returns whichever operand short-circuited it, not a fresh
    # coerced bool - if np.isnan(max_rhat) is what triggers True, an
    # unwrapped expression here would leak a numpy.bool_ into the
    # returned dict instead of the plain Python bool the docstring
    # promises (caught by test_convergence_diagnostics_flags_nan_rhat_
    # even_with_no_divergences's own `is True` identity check).
    flagged = bool(max_rhat > 1.01 or np.isnan(max_rhat) or n_divergences > 0)
    return {"max_rhat": max_rhat, "min_ess": min_ess, "n_divergences": n_divergences, "flagged": flagged}


def predict_held_out(
    mcmc: MCMC, X_test: np.ndarray, test_group_idx: np.ndarray, n_test_groups: int, seed: int
) -> np.ndarray:
    """The marginalized held-out prediction itself (D22) - the whole
    reason this file exists. Uses mcmc's ACTUAL FITTED POSTERIOR (alpha/
    sigma_q/beta, learned from training data via NUTS), never resampled
    from the prior - only `alpha_q_new` (the held-out questions' own
    intercepts, which were never fitted at all, since they don't exist
    in the training population) is freshly drawn, and even that draw
    uses each posterior sample's OWN `sigma_q`, not a fresh prior draw
    of sigma_q itself.

    ⚠️ Use this ONLY when the evaluation rows' questions are genuinely
    ABSENT from training (real cross-validation - 5.9b/5.9c/5.9d's use).
    Do NOT use this for the verbose-shift check (task 5.9f) - `clean`
    and `verbose` are paired on the exact same 80 questions (confirmed
    empirically), so `verbose`'s rows already have a REAL fitted
    `alpha_q`; marginalizing it away here would discard valid
    information AND confound the experiment (marginalization inflates
    spread on ANY input, which would make "epistemic rises on verbose"
    unfalsifiable - see D21's own amendment, 22 Sep 2026). Use
    `predict_in_sample()` instead for that same-questions-different-
    condition case.

    `numpyro.sample`/`numpyro.plate` do NOT appear here - those are
    model-DEFINITION tools, only meaningful inside an active NUTS/
    Predictive trace. This function is ordinary post-hoc computation
    over numbers `mcmc` already produced; the fresh alpha_q_new draw
    uses `jax.random` directly instead.

    Marginalization, concretely: for EVERY posterior draw d (there are
    n_draws = num_samples * num_chains of them), draw a fresh
    alpha_q_new[d, g] ~ Normal(0, sigma_q[d]) for each held-out question
    g - one shared value per (draw, question), not one independent value
    per (draw, row): rows belonging to the same held-out question must
    share the same alpha_q_new draw within a given posterior sample,
    exactly like training rows sharing one question share one
    alpha_q[group_idx]. That's what test_group_idx (a SEPARATE
    build_group_index() call on the test fold's own question_ids, never
    the training one) is for - not a lookup into the training posterior,
    only a way to correlate held-out rows that share a held-out question.

    Args:
        mcmc: fit_nuts()'s already-fitted MCMC object (TRAINING fold).
        X_test: held-out fold's feature matrix, (n_test_rows,
            n_features), same columns/order as training X.
        test_group_idx, n_test_groups: build_group_index()'s output on
            the TEST fold's own question_id array - never the training
            group_idx/n_groups.
        seed: converted to jax.random.PRNGKey(seed) here - controls only
            the fresh alpha_q_new draw; the posterior itself is already
            fixed inside `mcmc`.

    Returns:
        per_draw_probs, (n_draws, n_test_rows) - P(correct) for every
        posterior draw x every held-out row, NOT collapsed to a mean.
        Downstream callers need the full spread: the mean gives the
        point OOF prediction, the draws' own spread gives the 90%
        credible-interval coverage (5.9c) and 5.9d's entropy-quality
        AUROC (both free from this one matrix, expensive to redo
        separately if only the mean were kept) - NOT 5.9f's verbose-
        shift entropy decomposition, which needs predict_in_sample()'s
        output instead (see this function's own warning above).
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
) -> np.ndarray:
    """Predicts using the model's REAL fitted alpha_q (D21's amendment,
    22 Sep 2026) - the counterpart to predict_held_out(), for the
    OPPOSITE situation: evaluation rows whose questions WERE in
    training, just under a different condition (task 5.9f's verbose-
    shift check - `clean` and `verbose` are paired on the exact same 80
    questions, confirmed empirically: identical question_id sets,
    identical item_id sets, same row order).

    Never use this for genuinely held-out questions - that's
    predict_held_out()'s job, and it marginalizes for exactly that
    reason. This function does NO marginalization at all: it looks up
    each row's OWN question's real fitted alpha_q via
    `question_id_to_index` (the TRAINING fold's own build_group_index()
    mapping, its third return value) - never a fresh build_group_index()
    call on the evaluation rows' own question_ids, which would produce a
    DIFFERENT, disconnected numbering scheme unrelated to the fitted
    alpha_q array's actual indices.

    Why this is the right tool for the verbose-shift check specifically:
    using the SAME real alpha_q for both the clean (in-sample) and
    verbose (shifted) evaluations isolates the actual variable of
    interest - verbose's rows carry DIFFERENT FEATURE VALUES (conf_verb/
    conf_lp/conf_bpe computed on the judge's verbose-condition behavior),
    not different questions. The mechanism that should make epistemic
    rise under verbose is beta's OWN posterior spread interacting with
    those shifted feature values - different plausible beta draws
    naturally diverge more when extrapolating away from the region the
    training data actually covered. Using predict_held_out()'s
    marginalization here instead would inflate spread on both sides
    equally and confound that signal with an unrelated artifact (D21's
    own amendment has the full reasoning).

    A row whose question_id is genuinely absent from
    question_id_to_index raises a plain KeyError - propagate the gap,
    don't fabricate a value for a question this function has no basis to
    answer for (the same "propagate, don't fabricate" convention
    build_group_index()'s own docstring already establishes for the
    analogous held-out-key case).

    Args:
        mcmc: fit_nuts()'s already-fitted MCMC object, trained on ALL of
            one condition's data (no CV split - task 5.9f fits ONCE on
            all of `clean`, mirroring task 5.7's own "one offline-
            trained model" transfer-test design, not the 10x5 repeated-
            CV protocol 5.9b/5.9c/5.9d use).
        X: feature matrix to evaluate, (n_rows, n_features) - `clean`
            itself (in-sample) or `verbose` (the shifted condition),
            same columns/order as training X.
        row_question_ids: raw question_id per row of X, (n_rows,).
        question_id_to_index: the TRAINING fold's own build_group_index()
            mapping (question_id -> dense index) - NOT rebuilt on X's
            own question_ids.

    Returns:
        per_draw_probs, (n_draws, n_rows) - P(correct) for every
        posterior draw x every row, same shape/convention as
        predict_held_out()'s own output (mean gives the point
        prediction; the full spread feeds
        posterior_predictive_entropy_decomposition()).
    """
    samples = mcmc.get_samples()
    alpha = samples["alpha"]  # (n_draws,)
    alpha_q = samples["alpha_q"]  # (n_draws, n_groups) - REAL fitted values, never marginalized
    beta = samples["beta"]  # (n_draws, n_features)

    X = jnp.asarray(X)
    group_idx = jnp.asarray([question_id_to_index[qid] for qid in row_question_ids])

    logits = alpha[:, None] + alpha_q[:, group_idx] + beta @ X.T
    return jax.nn.sigmoid(logits)


@dataclass
class BayesianRepeatResult:
    """One repeat's worth of repeated_stratified_group_kfold_bayesian()'s
    output - the Bayesian analogue of predictor.py::RepeatResult, with
    two extra fields (oof_draws, fold_diagnostics) the frequentist model
    has no equivalent of.

    oof_pred/oof_draws are in POSITIONAL order matching X.iloc, same
    contract as RepeatResult.oof_pred - callers zipping this back to
    item_id/question_id must do so positionally against the SAME X
    passed in, not via X's own pandas index.
    """

    seed: int
    oof_pred: np.ndarray  # (n_rows,) mean P(correct), positionally aligned to X
    oof_draws: np.ndarray  # (n_draws, n_rows) EVERY posterior draw's P(correct), same alignment
    auroc: float
    fold_diagnostics: list[dict]  # one convergence_diagnostics() dict per fold (5 of them)


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
    """D8's full repeated-CV protocol (same as predictor.py::
    repeated_stratified_group_kfold), but fitting hierarchical_logit_model
    via NUTS per fold instead of LogReg/HistGBM (D22). Reuses
    make_fold_splits() directly, unmodified - StratifiedGroupKFold
    doesn't know or care what model comes next.

    Per fold: build_group_index() is called TWICE, independently - once
    on the training rows' question_ids (for fit_nuts), once on the test
    rows' question_ids (for predict_held_out) - never shared or reused
    between the two, which is what keeps a held-out question's random
    intercept structurally unreachable during prediction (D22, see
    build_group_index()'s and predict_held_out()'s own docstrings).

    Within one repeat, every fold reuses the SAME `repeat_seed` for both
    fit_nuts and predict_held_out - matching predictor.py's own
    established convention (make_model(repeat_seed) reused across a
    repeat's 5 folds) rather than a new per-fold seed scheme. Different
    folds still produce meaningfully different results despite the
    shared nominal seed, since their training data and held-out
    questions differ.

    fold_diagnostics collects every fold's convergence_diagnostics()
    unconditionally - this function only REPORTS them (D22's "never
    silently include a bad fit" requirement), it never drops or reruns a
    flagged fold itself; deciding what to do with a flagged fold belongs
    to the caller (5.9c), not this loop.

    Args:
        X: feature matrix, matching predictor.py::build_xyg()'s output -
            the SAME (X, y, groups) triple already built for the
            frequentist model can be passed here unchanged.
        y: binary target (`correct`, as 0/1), same row order as X.
        groups: clustering column (`question_id`), same row order as X.
        n_splits, n_repeats: D8's 5-fold / 10-repeat protocol - n_repeats
            is a starting point here, not preregistered yet (see
            fit_nuts's own docstring on num_warmup/num_samples) - the
            runtime-measurement task decides the real value.
        seed: base seed - repeat i uses seed + i (D8).
        num_warmup, num_samples, num_chains: passed straight through to
            every fold's fit_nuts() call, fixed across all folds/repeats
            in one invocation.

    Returns:
        One BayesianRepeatResult per repeat, in seed order.
    """
    results = []
    for i in range(n_repeats):
        repeat_seed = seed + i
        oof_pred = np.full(len(X), np.nan)
        oof_draws = np.full((num_samples * num_chains, len(X)), np.nan)
        fold_diagnostics = []

        for train_idx, test_idx in make_fold_splits(X, y, groups, n_splits, repeat_seed):
            X_train, y_train = X.iloc[train_idx].to_numpy(), y[train_idx]
            X_test = X.iloc[test_idx].to_numpy()
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
    """Vectorized H(p) = -(p*log(p) + (1-p)*log(1-p)), in nats - the array
    counterpart to signals.py::_binary_entropy() (scalar-only there,
    since it's called once per item; here it runs over whole
    (n_draws, n_items) draw matrices, where a Python-level loop would be
    both slow and the wrong shape of code for this file's own
    jax.numpy-first convention). Clipped rather than branching on
    `p <= 0` / `p >= 1` per element (that scalar guard doesn't vectorize
    cleanly) - clip(p, eps, 1-eps) keeps log() finite everywhere, and the
    resulting H(p) at the clip boundary is negligibly different from the
    true 0 the scalar version returns there.
    """
    eps = 1e-12
    p_clipped = np.clip(p, eps, 1 - eps)
    return -(p_clipped * np.log(p_clipped) + (1 - p_clipped) * np.log(1 - p_clipped))


def posterior_predictive_entropy_decomposition(draws: np.ndarray) -> dict:
    """Meta-model-level Total/Aleatoric/Epistemic entropy decomposition
    (D22) - the counterpart to signals.py::conf_ens()'s JUDGE-level
    decomposition (D20), same formula, generalized from averaging over 3
    discrete prompt variants (P1/P2/P3) to averaging over n_draws
    posterior predictive draws - structurally the same "how much
    uncertainty is there overall, how much of it is already present in
    each individual opinion, how much only appears once you average
    across opinions" question, just with many posterior draws standing
    in for the ensemble's three prompt variants.

        mean_p    = mean(draws, axis=0)      # per item
        Total     = H(mean_p)                 # binary entropy
        Aleatoric = mean(H(draws), axis=0)     # per item
        Epistemic = Total - Aleatoric          # >=0, Jensen's inequality

    Total is the model's overall uncertainty about an item; Aleatoric is
    how unsure each individual posterior draw is, on average; Epistemic
    is the extra uncertainty that only appears once you average ACROSS
    draws - i.e. genuine disagreement between draws (parameter
    uncertainty NUTS hasn't resolved), not uncertainty within any one
    draw's own prediction - the same BALD/mutual-information reading
    conf_ens()'s own docstring gives for the judge-level version.

    Args:
        draws: (n_draws, n_items) - P(event) per posterior draw per item
            (predict_held_out()'s own output shape, or
            compute_bayesian_arm()'s pooled_draws in analysis/rq4.py).
            "event" is whatever the draws represent (P(correct) or
            P(wrong)) - the decomposition itself doesn't care which, a
            caller just has to be consistent about which one it passes.

    Returns:
        dict with total, aleatoric, epistemic - each an (n_items,) array.
    """
    draws_arr = np.asarray(draws, dtype=float)
    mean_p = draws_arr.mean(axis=0)
    total = _binary_entropy_vec(mean_p)
    aleatoric = _binary_entropy_vec(draws_arr).mean(axis=0)
    epistemic = total - aleatoric
    return {"total": total, "aleatoric": aleatoric, "epistemic": epistemic}


if __name__ == "__main__":
    # Mirrors predictor.py's own __main__ shape (--config/--tier, D8's
    # across-repeat spread as the headline uncertainty) - no --model flag
    # (predictor.py's MODEL_FACTORIES choice doesn't apply here, there's
    # only the one Bayesian model) but the same tier flexibility, since
    # 5.9c's head-to-head comparison needs to be able to pick a tier to
    # match whatever the frequentist side of that comparison uses.
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

    print(f"  Runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min) for {len(all_diag)} fold-fits, real, not extrapolated")

