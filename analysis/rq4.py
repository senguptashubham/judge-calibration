"""RQ4 tasks 5.5 (tier ablation), 5.6 (H4, the continuous disagreement
interaction), and 5.7 (transfer test 1: train on clean, test on verbose)
- same RQ, same file, mirroring how analysis/rq3.py holds both of RQ3's
sub-tasks (4.3, 4.4) rather than splitting per-task. See TASKS.md tasks
5.5/5.6/5.7, PLAN.md §2.2's reframe ("which feature family carries the
signal"), §2.3 ("the label problem"), and §2.4 ("the transfer test").

`python -m analysis.rq4 --config configs/run.yaml --task {ablation,h4,transfer}`
- each task is its own CLI invocation, not run together, since the
ablation alone is already several hundred model fits; running all three
on every invocation would silently multiply that cost for no reason most
of the time.

--- Task 5.5 (tier ablation) ---

Population: features.py::load_rq4_population() (N=1819, the shared
base every RQ4 tier uses), further restricted to `human_agreed == True`
(D16) - N=556, 79/80 question_id groups still represented. This is a
MUCH bigger cut than it might look (1819 -> 556, 69% of the population
dropped) and it is NOT population-neutral: confirmed empirically
(18 Sep 2026 discussion) that judge accuracy is measurably higher on
this restricted population (75.8% -> 78.4%) than on the full RQ4 base -
human-agreed items are, on average, easier ones.

THIS IS WHY THE BASELINE IS RECOMPUTED HERE, NOT REUSED FROM
results/rq2_table_{model_slug}.csv: RQ2's stored AUROC numbers were
computed on RQ1's own population (N=1836, clean/P1, human_label present -
NO human_agreed restriction). Comparing a tier-ablation score computed
on the 556-item restricted population directly against a baseline
computed on the 1836-item unrestricted one would be comparing across two
different populations with two different difficulty levels - exactly
the kind of silent population mismatch CLAUDE.md invariant 14 warns
about for a different filter, and just as real here.

Writes results/rq4_ablation_{model_slug}.csv (one row per (tier, model)
pair) and results/figures/rq4_ablation_{model_slug}.png
(src/plots.py::plot_rq4_ablation - grouped bar chart, baseline as a
reference line + shaded CI band).

--- Task 5.6 (H4, continuous form, D9) ---

Population: clean/P1, human_label not null, n_human_votes >= 2 - D9's
own population, NOT task 5.5's human_agreed-restricted one and NOT
features.py::load_rq4_population()'s len_ratio-filtered one either. H4
is specifically about whether predictability trades off CONTINUOUSLY
against human consensus strength (d_human), so it deliberately keeps
every contested item 5.5 excluded - that's the whole point of using
every item with >=2 votes rather than only the agreed ones. Confirmed
empirically: N=595, 79/80 question_id groups (larger than D9's own
rough ~350-item estimate, made before real data existed).

"The predictor" (PLAN.md §2.3) is Tier A + logreg specifically, not a
free choice among all 6 tier/model combinations - documented in
compute_h4_oof_score()'s own docstring: task 5.5 found no significant
difference between any tier or model (every paired-progression CI
crossed zero), so Tier A is representative, not arbitrary, and it
avoids a second population restriction (Tier B/C need
len_ratio/longer_is_chosen, undefined for 17 items - task 5.2 - which
would shrink H4's already-small ~600-item population for no reason tied
to H4 itself). logreg over histgbm because H4's own interaction model
is itself a logistic regression - keeping "the predictor" and "the
interaction test" in one coherent model family.

Method (PLAN.md §2.3, D15 - mandatory, not a default choice):
  1. Out-of-fold P(correct) from the fixed 10x5 repeated CV (D8),
     averaged across repeats to one score per item - IN-SAMPLE
     predictions would bias the interaction before the CI method even
     matters.
  2. Fit `correct ~ oof_score * d_human` - a 3-feature logistic
     regression (oof_score, d_human, their product), reading off the
     product term's own coefficient.
  3. Cluster-bootstrap over question_id, B=2000, REFITTING each
     resample, percentile CI on the interaction coefficient - never a
     statsmodels/sklearn default standard error (those assume i.i.d.
     rows; rows cluster inside ~80 questions, invariant 2).

DoD: the interaction coefficient with its cluster-bootstrap CI, and the
aleatoric/epistemic reading in one sentence - no figure required.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

from analysis.rq1 import SIGNALS
from analysis.rq3 import load_rq3b_items
from src.boot import cluster_bootstrap, paired_cluster_bootstrap
from src.config import Config
from src.features import TIER_A_COLUMNS, build_tier_a, load_rq4_population
from src.metrics import auroc_error
from src.plots import (
    plot_h4_interaction,
    plot_rq4_ablation,
    plot_rq4_permutation_nulls,
    plot_rq4_progression,
    plot_rq4_transfer,
)
from src.predictor import (
    MODEL_FACTORIES,
    TIER_BUILDERS,
    build_xyg,
    encode_features,
    permutation_null,
    percentile_of_null,
    repeated_stratified_group_kfold,
    run_predictor,
)


def load_ablation_population(items_parquet: str) -> pd.DataFrame:
    """load_rq4_population()'s output, further restricted to
    human_agreed == True (D16: human_unanimous AND n_human_votes >= 2) -
    task 5.5's own population, per its DoD. See this module's own
    docstring for why this is a substantial, non-neutral cut (N=1819 ->
    N=556) rather than a minor refinement.
    """
    population = load_rq4_population(items_parquet)
    return population[population["human_agreed"]]


def compute_baseline_auroc(population: pd.DataFrame, seed: int) -> dict:
    """The best single confidence signal's AUROC(uncertainty -> error) on
    THIS task's own population (not RQ2's N=1836 population - see module
    docstring). Recomputed here rather than read from
    results/rq2_table_{model_slug}.csv, specifically so the ablation's
    "did tier X beat the baseline" comparison is apples-to-apples on one
    population, the same principle that already governs why all three
    tiers share one population (features.py's own module docstring).

    Reuses RQ2's exact recipe (analysis/rq2.py::compute_signal_rq2_metrics's
    own `_auroc` closure): uncertainty = 1 - signal, auroc_error(uncertainty,
    correct), cluster-bootstrap CI grouped on question_id (invariant 2).
    Loop over analysis.rq1.SIGNALS (conf_verb/conf_lp/conf_sc/conf_bpe -
    NOT conf_ens, which is a separate signal outside RQ1/RQ2/this
    baseline's own scope), take whichever has the highest point AUROC.

    Args:
        population: load_ablation_population()'s output.
        seed: config.seed.

    Returns:
        dict with signal (the winning signal's name), auroc, auroc_ci_low,
        auroc_ci_high.
    """
    def _auroc(df: pd.DataFrame) -> float:
        uncertainty = 1 - df[signal].to_numpy(dtype=float)
        return auroc_error(uncertainty, df["correct"].to_numpy())

    winning_signal = {"signal": None, "auroc": float("-inf"), "auroc_ci_low": None, "auroc_ci_high": None}
    for signal in SIGNALS:
        point, ci_low, ci_high = cluster_bootstrap(population, _auroc, "question_id", seed=seed)
        # Selection is on the point estimate ALONE - the CI is reported
        # for whichever signal wins, never used to decide who wins (a
        # signal with a genuinely higher point estimate but a wider/
        # noisier CI must still win; comparing CI bounds here could
        # silently reject the actual best signal).
        if point > winning_signal["auroc"]:
            winning_signal["signal"] = signal
            winning_signal["auroc"] = point
            winning_signal["auroc_ci_low"] = ci_low
            winning_signal["auroc_ci_high"] = ci_high

    return winning_signal


def compute_tier_model_result(population: pd.DataFrame, tier_name: str, model_name: str, seed: int) -> dict:
    """One (tier, model) cell of the ablation table: runs the full D8
    repeated-CV protocol (predictor.py::run_predictor, 10 repeats) and
    summarizes it down to one point estimate + one spread, matching the
    SAME across-repeat-spread convention tasks 5.3/5.4 already
    established (min/max across the 10 repeats' own whole-population
    AUROCs - NOT a bootstrap CI; D8's own explicit instruction is that
    this spread IS the headline uncertainty here, not a stand-in for one).

    Args:
        population: load_ablation_population()'s output.
        tier_name: "A", "B", or "C" (TIER_BUILDERS's keys).
        model_name: "logreg" or "histgbm" (MODEL_FACTORIES's keys).
        seed: config.seed.

    Returns:
        dict with tier, model, auroc_mean, auroc_low, auroc_high,
        n_repeats.
    """
    results = run_predictor(population, TIER_BUILDERS[tier_name], model_name, seed)
    aurocs = [r.auroc for r in results]
    return {
        "tier": tier_name,
        "model": model_name,
        "auroc_mean": sum(aurocs) / len(aurocs),
        "auroc_low": min(aurocs),
        "auroc_high": max(aurocs),
        "n_repeats": len(results),
    }


def compute_permutation_null_summary(
    population: pd.DataFrame, tier_name: str, model_name: str, seed: int, n: int = 50
) -> dict:
    """Invariant 12 for one (tier, model) cell - reuses
    predictor.py::permutation_null() (task 5.4) directly, but pointed at
    THIS task's own 556-item human_agreed population, not the earlier
    1819-item full-population smoke test's - a different, smaller
    population needs its own null, not a borrowed one.

    n=50, not task 5.4's default 200: every one of these six AUROCs sits
    around 0.80-0.82, far from a 0.5 null (confirmed on the first check,
    Tier A/logreg: null mean 0.4928 at n=200) - this isn't a borderline
    case where finer percentile resolution would change the reading, so
    a smaller n is a deliberate, documented cost-saving here, not a
    silent weakening of the check.

    Args:
        population: load_ablation_population()'s output.
        tier_name, model_name, seed: same as compute_tier_model_result().
        n: permutation count (50, reduced from task 5.4's 200 - see above).

    Returns:
        dict with tier, model, observed_auroc, null_mean, null_std,
        percentile (observed's percentile of the null distribution), and
        null_aurocs (the raw n-length array itself, for
        plot_rq4_permutation_nulls() - NOT written to the summary CSV,
        see main()).
    """
    X, y, groups = build_xyg(population, TIER_BUILDERS[tier_name])
    observed = compute_tier_model_result(population, tier_name, model_name, seed)["auroc_mean"]
    null_aurocs = permutation_null(X, y, groups, MODEL_FACTORIES[model_name], n=n, seed=seed)
    return {
        "tier": tier_name,
        "model": model_name,
        "observed_auroc": observed,
        "null_mean": float(null_aurocs.mean()),
        "null_std": float(null_aurocs.std()),
        "percentile": percentile_of_null(observed, null_aurocs),
        "null_aurocs": null_aurocs,
    }


def compute_tier_oof_uncertainty(population: pd.DataFrame, tier_name: str, model_name: str, seed: int) -> np.ndarray:
    """Averages a (tier, model)'s out-of-fold P(correct) predictions
    across its 10 D8 repeats into one value per item - the SAME recipe
    task 5.6 (H4) already plans to use to turn a repeated-CV protocol
    into one score per item. Converted to "uncertainty" scale
    (1 - mean P(correct)) so it's directly comparable to the raw
    confidence signals through the same auroc_error() convention every
    other AUROC in this project already uses.

    Positionally aligned to `population`'s own row order throughout -
    encode_features()/run_predictor() never reorder rows, so index i
    here is item i of `population`, matching RepeatResult.oof_pred's own
    documented contract (predictor.py).
    """
    results = run_predictor(population, TIER_BUILDERS[tier_name], model_name, seed)
    mean_oof_pred = np.mean([r.oof_pred for r in results], axis=0)
    return 1 - mean_oof_pred


def _predictions_df(population: pd.DataFrame, uncertainty: np.ndarray) -> pd.DataFrame:
    """question_id/correct/uncertainty, positionally aligned to
    `population` - the shared shape paired_cluster_bootstrap's stat_fn
    needs on both sides of a comparison. Using the SAME column name
    ("uncertainty") for a raw signal (the baseline) and a model's
    1-mean(P(correct)) (a tier) is what lets one stat_fn serve every
    pairing - RQ1's own compute_verdict_gap() renames columns to a
    shared name for exactly this reason.
    """
    return pd.DataFrame(
        {
            "question_id": population["question_id"].to_numpy(),
            "correct": population["correct"].to_numpy(),
            "uncertainty": uncertainty,
        }
    )


def _auroc_from_uncertainty(df: pd.DataFrame) -> float:
    return auroc_error(df["uncertainty"].to_numpy(), df["correct"].to_numpy())


def compare_tier_progression(population: pd.DataFrame, baseline_signal: str, seed: int) -> pd.DataFrame:
    """Paired cluster-bootstrap comparison (invariant 2/3) of each step
    in the tier progression - baseline -> A, A -> B, B -> C - per model.
    Answers "is this step's apparent change real" directly, rather than
    eyeballing whether two bars' spread whiskers overlap on the ablation
    chart. Reuses paired_cluster_bootstrap (task 2.4) - the SAME tool
    already used for RQ1's judge_verdict-vs-verdict_bidir gap and RQ3b's
    clean-vs-verbose deltas, not a new technique for this comparison.

    Every comparison is on the SAME 556 items (same question_id universe
    on both sides of every pairing - paired_cluster_bootstrap's own
    requirement), just scored by a different tier/signal's uncertainty -
    exactly the "same items, two conditions" shape that function exists
    for.

    Args:
        population: load_ablation_population()'s output.
        baseline_signal: the winning signal's name from
            compute_baseline_auroc() (e.g. "conf_bpe").
        seed: config.seed.

    Returns:
        DataFrame, one row per (model, comparison) - auroc_diff (hi's
        AUROC minus lo's), ci_low, ci_high. A CI excluding 0 means that
        step's change is real, not spread-bar overlap noise.
    """
    baseline_uncertainty = 1 - population[baseline_signal].to_numpy(dtype=float)

    rows = []
    for model_name in MODEL_FACTORIES:
        uncertainty_by_stage = {"baseline": baseline_uncertainty}
        for tier_name in TIER_BUILDERS:
            uncertainty_by_stage[tier_name] = compute_tier_oof_uncertainty(population, tier_name, model_name, seed)

        for lo, hi in [("baseline", "A"), ("A", "B"), ("B", "C")]:
            df_hi = _predictions_df(population, uncertainty_by_stage[hi])
            df_lo = _predictions_df(population, uncertainty_by_stage[lo])
            diff, ci_low, ci_high = paired_cluster_bootstrap(
                df_hi, df_lo, _auroc_from_uncertainty, "question_id", seed=seed
            )
            rows.append(
                {
                    "model": model_name,
                    "comparison": f"{hi} - {lo}",
                    "auroc_diff": diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    return pd.DataFrame.from_records(rows)


def load_h4_population(items_parquet: str) -> pd.DataFrame:
    """D9's own population for H4: clean/P1, human_label not null,
    n_human_votes >= 2. Deliberately NOT load_ablation_population()'s
    human_agreed-restricted population, and NOT
    features.py::load_rq4_population()'s len_ratio-filtered one either -
    see this module's own docstring for why H4 needs every contested
    item, not just the agreed ones.
    """
    items = pd.read_parquet(items_parquet)
    items = items[(items["condition"] == "clean") & (items["prompt_variant"] == "P1")]
    items = items[items["human_label"].notna()]
    return items[items["n_human_votes"] >= 2]


def compute_h4_oof_score(population: pd.DataFrame, seed: int) -> np.ndarray:
    """Tier A + logreg's averaged out-of-fold P(correct) across the 10
    D8 repeats - "the predictor's output" H4 tests the interaction
    against (PLAN.md §2.3). See this module's own docstring for why
    Tier A + logreg specifically, not a free choice among all six tier/
    model combinations.
    """
    results = run_predictor(population, build_tier_a, "logreg", seed)
    return np.mean([r.oof_pred for r in results], axis=0)


def _h4_design_matrix(oof_score: np.ndarray, d_human: np.ndarray) -> np.ndarray:
    """[oof_score, d_human, oof_score*d_human] - the shared 3-column
    design both fit_interaction_coefficient() (per bootstrap replicate)
    and fit_h4_interaction_model() (once, for the figure) build, kept in
    one place so the two fits can never silently drift apart.
    """
    return np.column_stack([oof_score, d_human, oof_score * d_human])


def build_h4_predictions_df(population: pd.DataFrame, seed: int) -> pd.DataFrame:
    """question_id/correct/d_human/oof_score, built once - shared by
    compute_h4_interaction() (the bootstrap CI) and
    compute_h4_interaction_curves() (the optional figure), so the
    expensive 10x5-fold OOF computation (compute_h4_oof_score) only
    ever runs once per main_h4() invocation, not twice.
    """
    oof_score = compute_h4_oof_score(population, seed)
    return pd.DataFrame(
        {
            "question_id": population["question_id"].to_numpy(),
            "correct": population["correct"].to_numpy(),
            "d_human": population["d_human"].to_numpy(),
            "oof_score": oof_score,
        }
    )


def fit_interaction_coefficient(df: pd.DataFrame) -> float:
    """correct ~ oof_score * d_human (D9/PLAN.md §2.3): a 3-feature
    logistic regression - oof_score, d_human, and their product - with
    the product term's own coefficient read off as the H4 statistic.
    LogisticRegression(C=1.0), matching predictor.py's own established
    hyperparameter choice rather than a special-cased fit for this one
    test. Called once per cluster-bootstrap replicate (D15 - refitting
    each resample is mandatory, not just resampling a precomputed
    coefficient), so this must stay a real fit, not a shortcut.
    """
    X = _h4_design_matrix(df["oof_score"].to_numpy(), df["d_human"].to_numpy())
    y = df["correct"].astype(int).to_numpy()
    model = LogisticRegression(C=1.0)
    model.fit(X, y)
    return float(model.coef_[0][-1])


def compute_h4_interaction(predictions_df: pd.DataFrame, seed: int) -> dict:
    """The bootstrap half of the H4 pipeline: cluster-bootstrap CI on
    the interaction coefficient (D15's mandated method - see this
    module's own docstring for the three-step recipe).

    Args:
        predictions_df: build_h4_predictions_df()'s output.
        seed: config.seed.

    Returns:
        dict with n (population size), interaction_coef, ci_low, ci_high.
    """
    point, ci_low, ci_high = cluster_bootstrap(
        predictions_df, fit_interaction_coefficient, "question_id", n=2000, seed=seed
    )
    return {"n": len(predictions_df), "interaction_coef": point, "ci_low": ci_low, "ci_high": ci_high}


def fit_h4_interaction_model(predictions_df: pd.DataFrame) -> LogisticRegression:
    """Fits correct ~ oof_score * d_human ONCE on the real (non-
    bootstrapped) data - the same design fit_interaction_coefficient()
    uses per bootstrap replicate, but returned whole here for
    compute_h4_interaction_curves()'s predicted-probability curves
    (task 5.6's optional figure, requested 19 Sep 2026 after the numeric
    result). Presentation only - the coefficient's own CI always comes
    from compute_h4_interaction()'s bootstrap, never from this fit's own
    (uncorrected, i.i.d.-assuming) standard errors.
    """
    X = _h4_design_matrix(predictions_df["oof_score"].to_numpy(), predictions_df["d_human"].to_numpy())
    y = predictions_df["correct"].astype(int).to_numpy()
    model = LogisticRegression(C=1.0)
    model.fit(X, y)
    return model


def compute_h4_interaction_curves(predictions_df: pd.DataFrame, n_grid: int = 100) -> dict:
    """Predicted P(correct) vs. oof_score curves at each DISTINCT
    d_human level actually present in the data - NOT a min/median/max
    summary, which collapses under this population's real skew (564/595
    items sit at d_human=0.5, so the median trivially equals the max -
    confirmed empirically, 19 Sep 2026). This dataset has exactly 3
    distinct levels (~0.167, ~0.25, 0.5 - vote-count-driven discreteness,
    not a design choice); using them directly shows the interaction
    faithfully rather than forcing a 3-point summary onto data that has
    no meaningfully continuous median.

    ALSO returns the same curves on the log-odds (linear-predictor)
    scale, not just probability - checked empirically (19 Sep 2026) that
    the probability-space curves alone visually undersell the fitted
    interaction: the model's log-odds slope w.r.t. oof_score genuinely
    increases with d_human (that's what the positive interaction
    coefficient means), but in probability space that gets compressed by
    sigmoid saturation, specifically in the high-oof_score region where
    most of this project's real data actually sits (most judge calls are
    high-confidence) - higher-d_human curves sit closer to the ceiling
    there, where the sigmoid is flattest, visually muting a slope
    difference that's actually large and clear on the log-odds scale
    (where the model is literally linear and the interaction IS the
    slope difference, undistorted).

    Args:
        predictions_df: build_h4_predictions_df()'s output.
        n_grid: number of oof_score grid points per curve (100).

    Returns:
        dict with oof_score (the real, per-item values, for a rug plot),
        oof_score_grid, d_human_values (the distinct levels, ascending),
        predicted_curves (list of P(correct) arrays, one per
        d_human_values entry, same order), log_odds_curves (the same
        curves on the linear-predictor scale).
    """
    model = fit_h4_interaction_model(predictions_df)
    d_human_values = sorted(predictions_df["d_human"].round(4).unique())
    oof_score_grid = np.linspace(0, 1, n_grid)

    predicted_curves = []
    log_odds_curves = []
    for d_human_value in d_human_values:
        X_grid = _h4_design_matrix(oof_score_grid, np.full(n_grid, d_human_value))
        predicted_curves.append(model.predict_proba(X_grid)[:, 1])
        log_odds_curves.append(model.decision_function(X_grid))

    return {
        "oof_score": predictions_df["oof_score"].to_numpy(),
        "oof_score_grid": oof_score_grid,
        "d_human_values": d_human_values,
        "predicted_curves": predicted_curves,
        "log_odds_curves": log_odds_curves,
    }


def main_h4(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    population = load_h4_population(config.paths.items_parquet)
    print(f"H4 population: N={len(population)} (clean/P1, human_label present, n_human_votes >= 2, D9)")

    predictions_df = build_h4_predictions_df(population, config.seed)

    result = compute_h4_interaction(predictions_df, config.seed)
    print(
        f"H4 interaction coefficient (oof_score x d_human): {result['interaction_coef']:.4f} "
        f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
    )
    if result["ci_low"] > 0:
        print("CI excludes 0 (positive): the predictor's edge grows with human consensus - supports H4.")
    elif result["ci_high"] < 0:
        print("CI excludes 0 (negative): the predictor's edge SHRINKS with human consensus - contradicts H4.")
    else:
        print("CI includes 0: no detectable interaction at this sample size - H4 neither supported nor refuted.")

    curves = compute_h4_interaction_curves(predictions_df)
    plot_h4_interaction(
        oof_score=curves["oof_score"],
        oof_score_grid=curves["oof_score_grid"],
        d_human_values=curves["d_human_values"],
        predicted_curves=curves["predicted_curves"],
        log_odds_curves=curves["log_odds_curves"],
        model_slug=config.model_slug,
    )


# --- Task 5.7 (transfer test 1: train on clean, test on verbose) -------
#
# "Does an abstention layer trained on well-behaved data still work when
# the judge is under attack?" (PLAN.md §2.4). Population: analysis/rq3.py's
# load_rq3b_items() - the SAME clean/P1 vs verbose/P1 paired population
# RQ3b already established (N=1836), not re-derived a third time.
#
# Feature-parity fix (D21, mandatory - not a Bayesian-specific carve-out):
# Tier A minus {conf_sc, conf_ens, ens_entropy_total, ens_entropy_aleatoric,
# ens_entropy_epistemic} - verbose has none of these (D19: self-consistency
# sampling and the P2/P3 ensemble are both clean/P1-only). What survives is
# just conf_verb/conf_lp/conf_bpe.
#
# Two numbers, computed with the SAME reduced feature set so the
# comparison isolates the TRANSFER effect, not the feature-drop effect:
#   - in-domain baseline: repeated grouped CV (D8) on clean alone - "how
#     good is this reduced-feature model within its own training
#     distribution."
#   - transfer: fit ONCE on all of clean (frozen, no CV - this is about
#     one offline-trained model's real deployment behavior), evaluate on
#     all of verbose. CI via cluster-bootstrap over verbose's own
#     question_id (invariant 2) - resampling the TEST set only, since the
#     model itself is fixed, not refit per resample.
#
# DoD: ΔAUROC (transfer - in-domain) reported, for both LogReg and
# HistGBM (D21 - not one model only).

TRANSFER_SAFE_COLUMNS = [
    c
    for c in TIER_A_COLUMNS
    if c not in {"conf_sc", "conf_ens", "ens_entropy_total", "ens_entropy_aleatoric", "ens_entropy_epistemic"}
]


def build_transfer_xy(items: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """TRANSFER_SAFE_COLUMNS's features (D21), encoded (encode_features()
    is a no-op on these three - all already numeric - kept for
    consistency/robustness, not because it currently does anything), the
    `correct` target, and `question_id` as the grouping column - the
    (X, y, groups) triple both compute_transfer_baseline() and
    compute_transfer_auroc() need, for either clean or verbose items.
    """
    X = encode_features(items[TRANSFER_SAFE_COLUMNS].copy())
    y = items["correct"].astype(int).to_numpy()
    groups = items["question_id"].to_numpy()
    return X, y, groups


def compute_transfer_baseline(clean_items: pd.DataFrame, model_name: str, seed: int) -> dict:
    """In-domain baseline: repeated grouped CV (D8) on clean_items ALONE,
    using the SAME reduced (transfer-safe) feature set the transfer test
    itself uses. Holding the feature set fixed is what isolates the
    transfer effect: D21's parity fix already costs some AUROC even
    within clean (if conf_sc/conf_ens carried real signal), and without
    this baseline using the identical reduced set, a ΔAUROC against the
    full Tier A's own clean performance would conflate "moving to
    verbose hurt" with "dropping two columns hurt."
    """
    X, y, groups = build_transfer_xy(clean_items)
    results = repeated_stratified_group_kfold(X, y, groups, MODEL_FACTORIES[model_name], seed=seed)
    aurocs = [r.auroc for r in results]
    return {"auroc_mean": sum(aurocs) / len(aurocs), "auroc_low": min(aurocs), "auroc_high": max(aurocs)}


def compute_transfer_auroc(clean_items: pd.DataFrame, verbose_items: pd.DataFrame, model_name: str, seed: int) -> dict:
    """The real transfer number: fit ONCE on all of clean_items, evaluate
    the frozen model on verbose_items. Deliberately not a CV protocol -
    D21's question is about one offline-trained model's behavior under
    attack, not about re-characterizing training variance.

    CI via cluster-bootstrap over verbose_items' question_id (invariant
    2) - the model is fixed/frozen going in, so each bootstrap replicate
    only resamples which verbose items get evaluated, never refits the
    model itself.
    """
    X_train, y_train, _ = build_transfer_xy(clean_items)
    X_test, y_test, groups_test = build_transfer_xy(verbose_items)

    model = MODEL_FACTORIES[model_name](seed)
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]

    eval_df = pd.DataFrame({"question_id": groups_test, "correct": y_test, "pred": pred})

    def _auroc(df: pd.DataFrame) -> float:
        return float(roc_auc_score(df["correct"].to_numpy(), df["pred"].to_numpy()))

    point, ci_low, ci_high = cluster_bootstrap(eval_df, _auroc, "question_id", seed=seed)
    return {"auroc": point, "ci_low": ci_low, "ci_high": ci_high}


def main_transfer(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    clean_items, verbose_items = load_rq3b_items(config.paths.items_parquet)
    print(f"Transfer test population: N={len(clean_items)} paired items (clean/P1 vs verbose/P1)")
    print(f"Feature set (D21 parity fix): {TRANSFER_SAFE_COLUMNS}")

    rows = []
    for model_name in MODEL_FACTORIES:
        baseline = compute_transfer_baseline(clean_items, model_name, config.seed)
        transfer = compute_transfer_auroc(clean_items, verbose_items, model_name, config.seed)
        delta = transfer["auroc"] - baseline["auroc_mean"]

        print(
            f"{model_name}: in-domain (clean) AUROC={baseline['auroc_mean']:.4f} "
            f"[{baseline['auroc_low']:.4f}, {baseline['auroc_high']:.4f}] (D8 across-repeat spread)"
        )
        print(
            f"{model_name}: transfer (clean->verbose) AUROC={transfer['auroc']:.4f} "
            f"[{transfer['ci_low']:.4f}, {transfer['ci_high']:.4f}] (cluster-bootstrap over verbose)"
        )
        print(f"{model_name}: ΔAUROC (transfer - in-domain) = {delta:.4f}")

        rows.append(
            {
                "model": model_name,
                "baseline_auroc_mean": baseline["auroc_mean"],
                "baseline_auroc_low": baseline["auroc_low"],
                "baseline_auroc_high": baseline["auroc_high"],
                "transfer_auroc": transfer["auroc"],
                "transfer_ci_low": transfer["ci_low"],
                "transfer_ci_high": transfer["ci_high"],
                "delta_auroc": delta,
            }
        )

    table = pd.DataFrame.from_records(rows)
    table_path = f"results/rq4_transfer_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    plot_rq4_transfer(
        models=table["model"].tolist(),
        baseline_mean=table["baseline_auroc_mean"].to_numpy(),
        baseline_low=table["baseline_auroc_low"].to_numpy(),
        baseline_high=table["baseline_auroc_high"].to_numpy(),
        transfer_mean=table["transfer_auroc"].to_numpy(),
        transfer_low=table["transfer_ci_low"].to_numpy(),
        transfer_high=table["transfer_ci_high"].to_numpy(),
        model_slug=config.model_slug,
    )


# --- Task 5.8 (transfer test 2: LeaveOneGroupOut over category) --------
#
# Skeleton only - bodies TODO. Full Tier A (no exclusions - unlike 5.7,
# this never leaves `clean`, so conf_sc/conf_ens stay valid on both
# sides of every split). `category` is the GROUPING variable for the
# split (LeaveOneGroupOut), never an input feature.


def compute_category_held_out_auroc(population: pd.DataFrame, model_name: str, seed: int) -> pd.DataFrame:
    """LeaveOneGroupOut over `category` (8 MT-Bench categories, D-none -
    this is a new grouping axis, not question_id): for each category,
    fit `model_name` on the other 7 categories' rows, evaluate on the
    held-out category alone. No shuffle/random_state/repeats - unlike
    StratifiedGroupKFold, LeaveOneGroupOut is fully deterministic (one
    fixed split per unique group value), so there's nothing to average
    over the way D8's 10-seed protocol does.

    Args:
        population: features.py::load_rq4_population()'s output.
        model_name: "logreg" or "histgbm" (MODEL_FACTORIES's keys).
        seed: config.seed - passed to the model factory (make_histgbm
            uses it; make_logreg ignores it, see predictor.py).

    Returns:
        DataFrame, one row per category: category, model, n (held-out
        row count), auroc.
    """
    X = encode_features(build_tier_a(population))
    y = population["correct"].astype(int).to_numpy()
    groups = population["category"].to_numpy()

    rows = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        model = MODEL_FACTORIES[model_name](seed)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict_proba(X.iloc[test_idx])[:, 1]
        auroc = roc_auc_score(y[test_idx], pred)
        rows.append({"category": groups[test_idx][0], "model": model_name, "n": len(test_idx), "auroc": auroc})

    return pd.DataFrame.from_records(rows)


def main_category(config_path: str) -> None:
    """TODO (task 5.8): load_rq4_population() -> compute_category_held_out_auroc()
    for each model in MODEL_FACTORIES -> concat into one table -> print
    each row + the min/max AUROC spread across categories (the DoD's
    "generalizes, or learns 'coding is hard'" reading - a tight spread
    says generalizes, one category cratering says shortcut) -> write
    results/rq4_category_transfer_{model_slug}.csv.
    """
    config = Config.from_yaml(config_path)
    population = load_rq4_population(config.paths.items_parquet)

    tables = []
    for model_name in MODEL_FACTORIES:
        result = compute_category_held_out_auroc(population, model_name, config.seed)
        tables.append(result)
        for _, row in result.iterrows():
            print(f"{model_name}, {row['category']}: n={row['n']}, AUROC={row['auroc']:.4f}")
        print(
            f"{model_name}: spread across categories = "
            f"[{result['auroc'].min():.4f}, {result['auroc'].max():.4f}]"
        )

    table = pd.concat(tables, ignore_index=True)
    table_path = f"results/rq4_category_transfer_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


def main_ablation(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    population = load_ablation_population(config.paths.items_parquet)
    print(f"RQ4 ablation population: N={len(population)} (human_agreed items only, D16)")

    baseline = compute_baseline_auroc(population, config.seed)
    print(
        f"Baseline (best single signal, this population): {baseline['signal']} "
        f"AUROC={baseline['auroc']:.4f} [{baseline['auroc_ci_low']:.4f}, {baseline['auroc_ci_high']:.4f}]"
    )

    rows = []
    for tier_name in TIER_BUILDERS:
        for model_name in MODEL_FACTORIES:
            result = compute_tier_model_result(population, tier_name, model_name, config.seed)
            rows.append(result)
            print(
                f"Tier {tier_name}, {model_name}: AUROC={result['auroc_mean']:.4f} "
                f"[{result['auroc_low']:.4f}, {result['auroc_high']:.4f}]"
            )

    table = pd.DataFrame.from_records(rows)
    table_path = f"results/rq4_ablation_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    plot_rq4_ablation(
        tiers=table["tier"].tolist(),
        models=table["model"].tolist(),
        auroc_mean=table["auroc_mean"].to_numpy(),
        auroc_low=table["auroc_low"].to_numpy(),
        auroc_high=table["auroc_high"].to_numpy(),
        baseline=baseline["auroc"],
        baseline_ci_low=baseline["auroc_ci_low"],
        baseline_ci_high=baseline["auroc_ci_high"],
        baseline_label=baseline["signal"],
        model_slug=config.model_slug,
    )

    # Step 1 (invariant 12): does every cell beat chance on THIS
    # population specifically, not just the earlier full-population check.
    print()
    print("Permutation null (n=50 per cell - see compute_permutation_null_summary's own docstring):")
    null_rows = []
    for tier_name in TIER_BUILDERS:
        for model_name in MODEL_FACTORIES:
            null_summary = compute_permutation_null_summary(population, tier_name, model_name, config.seed)
            null_rows.append(null_summary)
            print(
                f"  Tier {tier_name}, {model_name}: observed={null_summary['observed_auroc']:.4f}, "
                f"null mean={null_summary['null_mean']:.4f} (std={null_summary['null_std']:.4f}), "
                f"percentile={null_summary['percentile']:.1%}"
            )

    # null_aurocs (the raw per-permutation array) is plot-only - excluded
    # from the CSV, which stays one summary row per cell, not one column
    # per permutation.
    null_table = pd.DataFrame.from_records(
        [{k: v for k, v in row.items() if k != "null_aurocs"} for row in null_rows]
    )
    null_table_path = f"results/rq4_ablation_null_{config.model_slug}.csv"
    null_table.to_csv(null_table_path, index=False)
    print(f"Wrote {len(null_table)} rows to {null_table_path}")

    plot_rq4_permutation_nulls(
        tiers=[row["tier"] for row in null_rows],
        models=[row["model"] for row in null_rows],
        observed=np.array([row["observed_auroc"] for row in null_rows]),
        null_distributions=[row["null_aurocs"] for row in null_rows],
        model_slug=config.model_slug,
    )

    # Step 2: is the apparent baseline->A->B->C progression real, or
    # spread-bar overlap noise?
    print()
    print("Paired tier-progression comparison (does each step's apparent change survive a paired CI):")
    progression = compare_tier_progression(population, baseline["signal"], config.seed)
    progression_path = f"results/rq4_ablation_progression_{config.model_slug}.csv"
    progression.to_csv(progression_path, index=False)
    for _, row in progression.iterrows():
        print(
            f"  {row['model']}, {row['comparison']}: "
            f"Δauroc={row['auroc_diff']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        )
    print(f"Wrote {len(progression)} rows to {progression_path}")

    plot_rq4_progression(
        comparisons=[f"{row['model']}: {row['comparison']}" for _, row in progression.iterrows()],
        auroc_diff=progression["auroc_diff"].to_numpy(),
        ci_low=progression["ci_low"].to_numpy(),
        ci_high=progression["ci_high"].to_numpy(),
        model_slug=config.model_slug,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=["ablation", "h4", "transfer", "category"])
    args = parser.parse_args()
    if args.task == "ablation":
        main_ablation(args.config)
    elif args.task == "h4":
        main_h4(args.config)
    elif args.task == "transfer":
        main_transfer(args.config)
    else:
        main_category(args.config)
