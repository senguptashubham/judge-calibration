"""RQ4 tasks 5.5 (tier ablation), 5.6 (H4, the continuous disagreement
interaction), 5.7 (transfer test 1: train on clean, test on verbose),
5.8 (transfer test 2: LeaveOneGroupOut over category), 5.9 (meta-model
calibration: reliability diagram + coefficients), and 5.9c (frequentist
vs Bayesian head-to-head) - same RQ, same file, mirroring how
analysis/rq3.py holds both of RQ3's sub-tasks (4.3, 4.4) rather than
splitting per-task. See TASKS.md tasks 5.5-5.9c, PLAN.md §2.2's reframe
("which feature family carries the signal"), §2.3 ("the label
problem"), and §2.4 ("the transfer test").

`python -m analysis.rq4 --config configs/run.yaml --task {ablation,h4,transfer,category,calibration,bayesian_comparison,verbose_shift}`
- each task is its own CLI invocation, not run together, since the
ablation alone is already several hundred model fits; running all three
on every invocation would silently multiply that cost for no reason most
of the time.

--- Task 5.9c (frequentist vs Bayesian head-to-head) -----------------

Population/tier: load_rq4_population() (N=1819), Tier A - matching
5.9b's own choice, NOT 5.9's Tier C. The comparison needs both arms on
the SAME feature set or it conflates "Bayesian vs frequentist" with
"different features" (same apples-to-apples discipline as 5.7's D21
feature-parity fix) - 5.9's Tier C choice answered a different question
(which feature family carries signal), not this one.

AUROC: D8's own convention for BOTH arms - mean + across-repeat spread
(min/max across the 10 repeats' own AUROCs), never a within-split CI.

ECE/Brier: one P(correct) per item, averaged OOF across the 10 repeats
(frequentist: run_predictor's own oof_pred, averaged; Bayesian:
BayesianRepeatResult.oof_pred, averaged the same way) - reusing D8's
established "average across repeats before scoring a per-item metric"
recipe, not a new one.

NLL/coverage_90 (Bayesian-only, D22 - "the last two exist only for the
Bayesian arm, since the frequentist model has no native posterior"):
  - Per-item posterior predictive draws are POOLED across all 10
    repeats (np.concatenate, not averaged) - each repeat is an
    independent full refit on a different fold partition, so pooling
    combines posterior uncertainty AND partition variability into one
    richer per-item predictive sample, rather than discarding 9 of the
    10 repeats' worth of draws.
  - NLL: the proper posterior-predictive log-likelihood per item -
    average the BERNOULLI LIKELIHOOD across pooled draws FIRST, then
    take -log. Never plug the mean probability into a point-NLL formula
    - that would just be Brier with extra steps and throw away exactly
    what makes this metric "Bayesian" (the draws' own spread).
  - coverage_90: D22 doesn't specify how "credible-interval coverage"
    applies to a BINARY outcome - a single 0/1 draw can't meaningfully
    "fall inside" a probability interval the way a continuous value
    can. Resolved (21 Sep 2026 discussion) via BIN-AGGREGATE coverage,
    reusing ece()'s own quantile binning (get_bin_edges): per bin, does
    the bin's EMPIRICAL accuracy (many real 0/1 outcomes aggregated
    into one meaningful continuous quantity) fall inside that bin's OWN
    pooled 90% credible interval (5th/95th percentile of every draw of
    every item in the bin)? coverage_90 = fraction of bins where it
    does - the same "turn per-item binary noise into a checkable
    per-bin quantity" move ECE itself already makes.

Writes results/rq4_bayesian_comparison_{model_slug}.csv (D26) +
results/figures/reliability_rq4_bayesian_meta_model_{model_slug}.png +
results/figures/rq4_bayesian_convergence_{model_slug}.png (the real
50-fold-fit R-hat picture 5.9b's own results were missing).

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

--- Task 5.9 (meta-model calibration: reliability + coefficients) -----

Population: features.py::load_rq4_population() (N=1819) - the same RQ4
base 5.3-5.5 use, NOT H4's ≥2-votes population or 5.5's human_agreed-
restricted one: this task is about the CORE predictor's own calibration
and feature weights, not a sub-question scoped to a smaller population.

Model: Tier C + logreg, not Tier A (task 5.6's choice) or a free pick
among all 6 tier/model combinations. Tier C specifically because this
task's whole point - the DoD's own words, "the coefficients are the
result, more than the AUROC is" - is best answered by a model that
actually CONTAINS all three feature families at once; Tier A alone
couldn't show whether Tier B/C's surface/CoT features carry any weight.
logreg (not histgbm) because raw coefficients are only directly
interpretable for the linear model - HistGBM has no comparable
per-feature weight.

Reliability diagram: out-of-fold P(correct) from the standard 10x5
repeated CV (D8), averaged across repeats - the SAME recipe
compute_h4_oof_score() already established for H4, pointed at Tier C
instead of Tier A - never in-sample predictions, which would look
artificially well-calibrated. Plotted as P(judge is wrong) =
1 - mean_oof_P(correct) against the actual wrong/right outcome, via
src/plots.py's existing plot_reliability_diagram() (RQ1, task 2.6) -
reused directly rather than a second implementation.

Coefficients: LogisticRegression(C=1.0) fit ONCE on the FULL population
(not averaged across CV folds - a coefficient is a property of one fit
on the data, not a per-repeat quantity the way an AUROC is). CI via a
cluster-bootstrap over question_id (invariant 2, D15's same standard-
error discipline as H4) that resamples ONCE per replicate and refits
the WHOLE coefficient vector together (bootstrap_coefficient_cis) -
not src/boot.py's cluster_bootstrap() called once per feature, which
would (a) refit ~37x more than necessary per replicate for no benefit,
since one refit already yields every feature's replicate value at once,
and (b) incorrectly treat each feature's bootstrap draw as independent
when they share the same resampled rows.

Writes results/rq4_coefficients_{model_slug}.csv and
results/figures/rq4_coefficients_{model_slug}.png (D26/DoD) +
results/figures/reliability_rq4_meta_model_{model_slug}.png.

--- Task 5.9f (verbose-shift validation, D21 amended) -----------------

Population: analysis/rq3.py's own load_rq3b_items() (N=1836), the SAME
paired clean/verbose population task 5.7's transfer test already uses -
not re-derived a third time.

Feature set: TRANSFER_SAFE_COLUMNS (Tier A minus conf_sc/conf_ens/
ens_entropy_*), the SAME D21 feature-parity fix task 5.7 already
established, now applied to the Bayesian model too, exactly as D21
itself requires ("for every model compared... not a Bayesian-specific
carve-out").

Fit ONCE on all of clean (no CV split) - mirrors 5.7's own "one
offline-trained model" transfer-test design, not the 10x5 repeated-CV
protocol tasks 5.9b/5.9c/5.9d use. This is deliberate: 5.9f is asking
whether ONE deployed model's own uncertainty estimate correctly
recognizes distribution shift, not characterizing training variance.

⚠️ Evaluation uses src/bayesian.py::predict_in_sample(), NEVER
predict_held_out(). See DECISIONS.md's D21 amendment (22 Sep 2026) for
the full reasoning - in short: clean and verbose are paired on the
EXACT SAME 80 questions (confirmed empirically), so verbose's rows
already have a real fitted alpha_q; predict_held_out()'s marginalization
would discard that real information AND confound the "does epistemic
rise under shift" test with an unrelated marginalization-noise
artifact. predict_in_sample() uses the training fold's own
question_id_to_index mapping (build_group_index()'s third return value)
for BOTH the clean (in-sample) and verbose (shifted) evaluations, so
the only thing that differs between them is the feature values
themselves - exactly the variable this test is about.

The preregistered prediction (professor feedback "consequences",
D21/D23): epistemic uncertainty rises under this distribution shift
while aleatoric stays flat. Tested via a PAIRED cluster-bootstrap
(invariant 3 - same items, two conditions) on mean(verbose) -
mean(clean), separately for aleatoric and epistemic - "rose" means the
epistemic gap's CI is entirely above 0; "stayed flat" means the
aleatoric gap's CI includes 0.

Lives in THIS file (rq4.py), not rq5.py, because it reuses 5.7's own
transfer-test machinery (TRANSFER_SAFE_COLUMNS, build_transfer_xy) -
same principle as compute_bayesian_arm() living here and being
imported BY rq5.py's own 5.9d, rather than duplicated there. Output
filenames still use the "rq5_" prefix, matching where this task's own
DoD/REPORT.md section actually sits (TASKS.md 5.9f, 5.11's own "RQ5
section: ... the verbose-shift check (5.9f)") - code location and
result-file naming answer two different questions (which machinery does
this reuse vs. which RQ does this result belong to) and are allowed to
disagree.

Writes results/rq5_verbose_shift_{model_slug}.csv (D26, one wide row:
both conditions' means, both gaps with CIs, both verdict booleans) +
results/figures/rq5_verbose_shift_{model_slug}.png.
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
from src.features import (
    TIER_A_COLUMNS,
    TIER_C_EXTRA_COLUMNS,
    build_tier_a,
    build_tier_c,
    load_rq4_population,
)
from src.bayesian import (
    build_group_index,
    fit_nuts,
    posterior_predictive_entropy_decomposition,
    predict_in_sample,
    repeated_stratified_group_kfold_bayesian,
)
from src.metrics import auroc_error, brier, ece, get_bin_edges
from src.plots import (
    plot_bayesian_convergence,
    plot_h4_interaction,
    plot_rq4_ablation,
    plot_rq4_category_transfer,
    plot_rq4_coefficients,
    plot_rq4_coefficients_full,
    plot_rq4_permutation_nulls,
    plot_rq4_progression,
    plot_rq4_transfer,
    plot_reliability_diagram,
    plot_rq5_verbose_shift,
)
from src.predictor import (
    MODEL_FACTORIES,
    TIER_BUILDERS,
    build_xyg,
    encode_features,
    make_logreg,
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
# Full Tier A (no exclusions - unlike 5.7,
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
    """load_rq4_population() -> compute_category_held_out_auroc()
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

    plot_rq4_category_transfer(
        categories=table["category"].tolist(),
        models=table["model"].tolist(),
        auroc=table["auroc"].to_numpy(),
        model_slug=config.model_slug,
    )


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


# --- Task 5.9 (meta-model calibration: reliability + coefficients) ----
#
# See this module's own docstring for the full population/model/CI
# rationale.


def compute_meta_oof_score(population: pd.DataFrame, seed: int) -> np.ndarray:
    """Tier C + logreg's averaged out-of-fold P(correct) across the 10
    D8 repeats - the reliability diagram's input. Same recipe as
    compute_h4_oof_score(), pointed at Tier C instead of Tier A: this
    task wants the calibration of the actual full-feature predictor,
    not H4's specifically-Tier-A interaction predictor.
    """
    results = run_predictor(population, build_tier_c, "logreg", seed)
    return np.mean([r.oof_pred for r in results], axis=0)


def compute_meta_calibration(population: pd.DataFrame, seed: int, n_bins: int) -> dict:
    """P(judge is wrong) = 1 - mean_oof_P(correct), and its ECE against
    the actual wrong/right outcome (CLAUDE.md invariant 4 - ece() picks
    its own binning strategy). Also reports the OOF AUROC alongside it
    (invariant 6 - ECE never ships alone).

    Args:
        population: load_rq4_population()'s output.
        seed: config.seed.
        n_bins: config.n_bins.

    Returns:
        dict with p_wrong, is_wrong (both arrays, positionally aligned
        to `population`), ece, n_effective_bins, auroc.
    """
    mean_oof_correct = compute_meta_oof_score(population, seed)
    correct = population["correct"].astype(int).to_numpy()
    p_wrong = 1 - mean_oof_correct
    is_wrong = 1 - correct

    ece_value, n_effective_bins = ece(p_wrong, is_wrong, n_bins)
    auroc = float(roc_auc_score(correct, mean_oof_correct))

    return {
        "p_wrong": p_wrong,
        "is_wrong": is_wrong,
        "ece": ece_value,
        "n_effective_bins": n_effective_bins,
        "auroc": auroc,
    }


def _feature_family(feature_name: str) -> str:
    """Tier-family label ("A"/"B"/"C") for one Tier C ENCODED feature
    name - plot_rq4_coefficients()'s color grouping. Category's one-hot
    dummies (pd.get_dummies's "category_<value>" naming, predictor.py::
    encode_features()) fall through to Tier B, matching `category`'s own
    (unencoded) tier membership rather than becoming a fourth family.
    """
    if feature_name in TIER_A_COLUMNS:
        return "A"
    if feature_name in TIER_C_EXTRA_COLUMNS:
        return "C"
    return "B"


def fit_full_logreg_coefficients(population: pd.DataFrame) -> tuple[LogisticRegression, list[str]]:
    """Fits make_logreg() ONCE on Tier C over the FULL population (no CV
    split) - coefficients describe one model's read of the whole
    dataset, not a per-fold quantity the way an AUROC is. Returns the
    fitted classifier step (coef_ is on the STANDARDIZED scale, so
    magnitudes are directly comparable across features - StandardScaler
    lives inside make_logreg()'s own Pipeline) and the encoded feature
    names in the SAME column order as coef_, so callers never re-derive
    that alignment themselves.
    """
    X = encode_features(build_tier_c(population))
    y = population["correct"].astype(int).to_numpy()
    pipeline = make_logreg()
    pipeline.fit(X, y)
    return pipeline.named_steps["clf"], list(X.columns)


def bootstrap_coefficient_cis(
    population: pd.DataFrame, feature_names: list[str], n: int = 2000, ci: float = 0.95, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster-bootstrap CI (question_id, invariant 2) on EVERY Tier C
    coefficient AT ONCE - see this module's own docstring for why this
    is its own loop rather than n calls to src/boot.py::cluster_bootstrap
    (one refit per replicate yields every feature's draw together, and
    keeps each replicate's features correlated the way the real fit's
    features are, instead of bootstrapping each one independently).

    One replicate: resample question_id values with replacement (same
    mechanics as cluster_bootstrap's own docstring), refit make_logreg()
    on Tier C over the resampled rows, record the whole coefficient
    vector. `.reindex(columns=feature_names, fill_value=0)` guards
    against the rare resample where some category dummy's rows didn't
    get drawn at all - encode_features() would then simply omit that
    column, so it's added back as an all-zero column rather than
    silently misaligning every later column against feature_names.

    Args:
        population: load_rq4_population()'s output.
        feature_names: fit_full_logreg_coefficients()'s own column order
            - what boot_coefs' columns must align to.
        n: bootstrap replicates (2000, D15's own convention).
        ci: confidence level.
        seed: required and explicit (CLAUDE.md sec 5).

    Returns:
        (ci_low, ci_high) - one value per feature_names entry, same
        order.
    """
    rng = np.random.default_rng(seed)
    groups = population["question_id"].unique()
    n_groups = len(groups)
    group_indices = {
        group: population.index[population["question_id"] == group].to_numpy() for group in groups
    }

    boot_coefs = np.empty((n, len(feature_names)))
    for i in range(n):
        sampled_groups = rng.choice(groups, size=n_groups, replace=True)
        sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        resampled = population.loc[sampled_indices]

        X = encode_features(build_tier_c(resampled)).reindex(columns=feature_names, fill_value=0)
        y = resampled["correct"].astype(int).to_numpy()
        pipeline = make_logreg()
        pipeline.fit(X, y)
        boot_coefs[i] = pipeline.named_steps["clf"].coef_[0]

    alpha = 1 - ci
    ci_low = np.quantile(boot_coefs, alpha / 2, axis=0)
    ci_high = np.quantile(boot_coefs, 1 - alpha / 2, axis=0)
    return ci_low, ci_high


def compute_meta_coefficients(population: pd.DataFrame, seed: int, n_boot: int = 2000) -> pd.DataFrame:
    """The full task 5.9 coefficient table: point estimate from ONE fit
    on the real data (fit_full_logreg_coefficients), CI from
    bootstrap_coefficient_cis - matching cluster_bootstrap()'s own
    convention that the point estimate is the real fit, never the mean
    of the bootstrap replicates (the replicates characterize spread,
    they don't re-estimate the center).

    Returns:
        DataFrame: feature, family (A/B/C), coef, ci_low, ci_high - one
        row per Tier C encoded feature.
    """
    clf, feature_names = fit_full_logreg_coefficients(population)
    ci_low, ci_high = bootstrap_coefficient_cis(population, feature_names, n=n_boot, seed=seed)

    return pd.DataFrame(
        {
            "feature": feature_names,
            "family": [_feature_family(name) for name in feature_names],
            "coef": clf.coef_[0],
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )


def main_calibration(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    population = load_rq4_population(config.paths.items_parquet)
    print(f"Meta-model calibration population: N={len(population)} (RQ4 base, Tier C + logreg)")

    calibration = compute_meta_calibration(population, config.seed, config.n_bins)
    print(f"OOF AUROC (D8, 10x5 repeated CV) = {calibration['auroc']:.4f}")
    print(
        f"ECE of P(judge is wrong) = {calibration['ece']:.4f} "
        f"(n_effective_bins={calibration['n_effective_bins']})"
    )

    plot_reliability_diagram(
        confidences=calibration["p_wrong"],
        correct=calibration["is_wrong"],
        signal_name="rq4_meta_model",
        n_bins=config.n_bins,
        model_slug=config.model_slug,
    )

    print()
    print("Fitting Tier C + logreg coefficients on the full population, cluster-bootstrap CI (B=2000)...")
    coef_table = compute_meta_coefficients(population, config.seed)
    coef_path = f"results/rq4_coefficients_{config.model_slug}.csv"
    coef_table.to_csv(coef_path, index=False)
    print(f"Wrote {len(coef_table)} rows to {coef_path}")

    n_significant = int(((coef_table["ci_low"] > 0) | (coef_table["ci_high"] < 0)).sum())
    print(f"{n_significant}/{len(coef_table)} coefficients have a CI excluding 0")
    for family in ["A", "B", "C"]:
        family_rows = coef_table[coef_table["family"] == family]
        family_significant = int(((family_rows["ci_low"] > 0) | (family_rows["ci_high"] < 0)).sum())
        print(f"  Tier {family}: {family_significant}/{len(family_rows)} coefficients have a CI excluding 0")

    # Headline figure: significant coefficients only (plot_rq4_coefficients
    # itself does the CI-excludes-0 filtering, so "significant" has one
    # definition shared by both figures) - the full 37-row table is
    # unusably tall for a slide/PDF at the annotated forest-plot spacing
    # (20 Sep 2026 discussion). Full breakdown: the CSV above, plus a
    # dense (unannotated) appendix figure with the same row ordering.
    plot_rq4_coefficients(
        features=coef_table["feature"].tolist(),
        coef=coef_table["coef"].to_numpy(),
        ci_low=coef_table["ci_low"].to_numpy(),
        ci_high=coef_table["ci_high"].to_numpy(),
        family=coef_table["family"].tolist(),
        model_slug=config.model_slug,
    )
    plot_rq4_coefficients_full(
        features=coef_table["feature"].tolist(),
        coef=coef_table["coef"].to_numpy(),
        ci_low=coef_table["ci_low"].to_numpy(),
        ci_high=coef_table["ci_high"].to_numpy(),
        family=coef_table["family"].tolist(),
        model_slug=config.model_slug,
    )


# --- Task 5.9c (frequentist vs Bayesian head-to-head) -------------------
#
# See this module's own docstring for the full population/tier/metric
# rationale (especially the NLL and bin-aggregate coverage_90 methods,
# neither of which D22 fully specifies).


def compute_frequentist_arm(population: pd.DataFrame, seed: int) -> dict:
    """Tier A + logreg's D8 repeated-CV result - the frequentist side of
    5.9c's head-to-head. Calls run_predictor() directly rather than
    reusing compute_h4_oof_score() (which only exposes the averaged OOF
    array) since this needs BOTH the per-repeat AUROCs (for the D8
    mean/spread) and the averaged OOF prediction (for ECE/Brier).
    """
    results = run_predictor(population, build_tier_a, "logreg", seed)
    return {
        "aurocs": np.array([r.auroc for r in results]),
        "mean_oof_correct": np.mean([r.oof_pred for r in results], axis=0),
    }


def compute_bayesian_arm(
    population: pd.DataFrame, seed: int, num_warmup: int = 500, num_samples: int = 1000, num_chains: int = 2
) -> dict:
    """Tier A Bayesian hierarchical model's D8 repeated-CV result (5.9b)
    on the SAME population/tier as compute_frequentist_arm() - the
    Bayesian side of the head-to-head. Settings default to 5.9b's own
    real, measured (not guessed) protocol - TASKS.md 5.9b: 7.0 minutes
    for the full 50-fold-fit run at these exact numbers.
    """
    X, y, groups = build_xyg(population, build_tier_a)
    results = repeated_stratified_group_kfold_bayesian(
        X, y, groups, seed=seed, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
    )
    return {
        "aurocs": np.array([r.auroc for r in results]),
        "mean_oof_correct": np.mean([r.oof_pred for r in results], axis=0),
        # Pooled across all 10 repeats by concatenation, NOT averaging -
        # see this module's own docstring for why: each repeat is an
        # independent full refit on a different fold partition, so
        # pooling combines posterior uncertainty AND partition
        # variability into one richer per-item predictive sample.
        # float32 - a real memory consideration at this scale (10
        # repeats x 2000 draws x 1819 items), not just a style choice.
        "pooled_draws": np.concatenate([r.oof_draws for r in results], axis=0).astype(np.float32),
        "fold_diagnostics": [fold_diag for r in results for fold_diag in r.fold_diagnostics],
    }


def compute_bayesian_nll(pooled_draws: np.ndarray, is_wrong: np.ndarray) -> float:
    """Proper posterior-predictive NLL, per item: average the BERNOULLI
    LIKELIHOOD across pooled draws FIRST, then -log - never plug the
    mean probability into a point-NLL formula, which would discard the
    draws' own spread and just be Brier with extra steps (see this
    module's own docstring).

    Args:
        pooled_draws: (n_pooled_draws, n_rows) - P(wrong) per draw per
            item, compute_bayesian_arm()'s own pooled_draws framed as
            "wrong" (1 - P(correct)).
        is_wrong: (n_rows,) - 1 if the judge was wrong, 0 if correct.

    Returns:
        Mean NLL across items (lower is better).
    """
    is_wrong_arr = is_wrong.astype(np.float32)
    likelihood_per_draw = is_wrong_arr[None, :] * pooled_draws + (1 - is_wrong_arr[None, :]) * (1 - pooled_draws)
    mean_likelihood_per_item = likelihood_per_draw.mean(axis=0)
    # An item every pooled draw got confidently wrong about would give
    # mean_likelihood_per_item exactly 0 - log(0) = -inf. Clipped, not
    # silently producing inf/nan in a headline number.
    mean_likelihood_per_item = np.clip(mean_likelihood_per_item, 1e-12, 1.0)
    return float(-np.mean(np.log(mean_likelihood_per_item)))


def compute_bayesian_coverage(pooled_draws: np.ndarray, p_wrong: np.ndarray, is_wrong: np.ndarray, n_bins: int) -> float:
    """Bin-aggregate 90% credible-interval coverage (21 Sep 2026
    discussion - see this module's own docstring for why per-item
    coverage isn't well-defined for a binary outcome, and why this
    bin-aggregate form is the resolution).

    Bins by p_wrong - the SAME quantity ece()/plot_reliability_diagram()
    already bin by elsewhere in this project, via get_bin_edges() so the
    binning strategy (quantile vs. discrete-unique-value) is identical
    to every other reliability check here, not a bespoke one.

    Args:
        pooled_draws: (n_pooled_draws, n_rows) - P(wrong) per draw per
            item.
        p_wrong: (n_rows,) - mean P(wrong) per item, the binning
            variable.
        is_wrong: (n_rows,) - 1 if the judge was wrong, 0 if correct.
        n_bins: config.n_bins.

    Returns:
        Fraction of bins whose empirical wrong-rate falls inside that
        bin's own pooled 90% credible interval.
    """
    edges = get_bin_edges(p_wrong, n_bins, "auto")
    bin_labels = np.digitize(p_wrong, edges)

    covered = []
    for bin_label in np.unique(bin_labels):
        mask = bin_labels == bin_label
        empirical_wrong_rate = is_wrong[mask].mean()
        bin_draws = pooled_draws[:, mask]
        lo, hi = np.percentile(bin_draws, [5, 95])
        covered.append(lo <= empirical_wrong_rate <= hi)

    return float(np.mean(covered))


def main_bayesian_comparison(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    population = load_rq4_population(config.paths.items_parquet)
    is_wrong = 1 - population["correct"].astype(int).to_numpy()
    print(f"Bayesian head-to-head population: N={len(population)} (RQ4 base, Tier A)")

    print("Fitting frequentist arm (Tier A + logreg, D8 10x5 repeated CV)...")
    freq = compute_frequentist_arm(population, config.seed)

    print("Fitting Bayesian arm (Tier A hierarchical logit, D8 10x5 repeated CV - ~7 min, see TASKS.md 5.9b)...")
    bayes = compute_bayesian_arm(population, config.seed)

    freq_p_wrong = 1 - freq["mean_oof_correct"]
    bayes_p_wrong = 1 - bayes["mean_oof_correct"]

    freq_ece, _ = ece(freq_p_wrong, is_wrong, config.n_bins)
    freq_brier = brier(freq_p_wrong, is_wrong)
    bayes_ece, _ = ece(bayes_p_wrong, is_wrong, config.n_bins)
    bayes_brier = brier(bayes_p_wrong, is_wrong)

    bayes_pooled_wrong_draws = 1 - bayes["pooled_draws"]
    bayes_nll = compute_bayesian_nll(bayes_pooled_wrong_draws, is_wrong)
    bayes_coverage = compute_bayesian_coverage(bayes_pooled_wrong_draws, bayes_p_wrong, is_wrong, config.n_bins)

    table = pd.DataFrame.from_records(
        [
            {
                "model": "frequentist_logreg",
                "auroc_mean": freq["aurocs"].mean(),
                "auroc_low": freq["aurocs"].min(),
                "auroc_high": freq["aurocs"].max(),
                "ece": freq_ece,
                "brier": freq_brier,
                "nll": np.nan,
                "coverage_90": np.nan,
            },
            {
                "model": "bayesian_hierarchical",
                "auroc_mean": bayes["aurocs"].mean(),
                "auroc_low": bayes["aurocs"].min(),
                "auroc_high": bayes["aurocs"].max(),
                "ece": bayes_ece,
                "brier": bayes_brier,
                "nll": bayes_nll,
                "coverage_90": bayes_coverage,
            },
        ]
    )
    table_path = f"results/rq4_bayesian_comparison_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))
    print(f"Wrote {len(table)} rows to {table_path}")

    plot_reliability_diagram(
        confidences=bayes_p_wrong,
        correct=is_wrong,
        signal_name="rq4_bayesian_meta_model",
        n_bins=config.n_bins,
        model_slug=config.model_slug,
    )

    all_diag = bayes["fold_diagnostics"]
    n_flagged = sum(fold_diag["flagged"] for fold_diag in all_diag)
    print(f"Bayesian convergence: {n_flagged}/{len(all_diag)} fold-fits flagged")

    plot_bayesian_convergence(
        max_rhat=np.array([fold_diag["max_rhat"] for fold_diag in all_diag]),
        flagged=np.array([fold_diag["flagged"] for fold_diag in all_diag]),
        model_slug=config.model_slug,
    )


# --- Task 5.9f (verbose-shift validation, D21 amended) ------------------
#
# See this module's own docstring for the full population/method
# rationale - ESPECIALLY why predict_in_sample() is used here, never
# predict_held_out() (DECISIONS.md's D21 amendment, 22 Sep 2026, has the
# complete reasoning).


def fit_verbose_shift_model(
    clean_items: pd.DataFrame, seed: int, num_warmup: int = 500, num_samples: int = 1000, num_chains: int = 2
) -> tuple[object, dict[int, int]]:
    """Fits the Bayesian model ONCE on ALL of clean (no CV split) -
    mirrors 5.7's own compute_transfer_auroc() "fit once, frozen"
    design, not the 10x5 repeated-CV protocol tasks 5.9b/5.9c/5.9d use.

    Returns the fitted mcmc AND the training fold's own dense-index
    mapping (question_id_to_index) - the second return value is not
    optional bookkeeping here the way it was for 5.9b's CV wrapper: both
    the clean (in-sample) and verbose (shifted) evaluations below need
    THIS SAME mapping passed to predict_in_sample(), never a fresh
    build_group_index() call on either evaluation set's own
    question_ids (see predict_in_sample()'s own docstring for why that
    would silently misalign every index).
    """
    X_train, y_train, train_groups = build_transfer_xy(clean_items)
    train_group_idx, n_groups, question_id_to_index = build_group_index(train_groups)
    mcmc = fit_nuts(
        X_train.to_numpy(),
        y_train,
        train_group_idx,
        n_groups,
        seed=seed,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    return mcmc, question_id_to_index


def compute_condition_entropy(mcmc: object, items: pd.DataFrame, question_id_to_index: dict[int, int]) -> pd.DataFrame:
    """predict_in_sample() + posterior_predictive_entropy_decomposition()
    for one condition's items (clean or verbose) against the SAME fitted
    mcmc and the SAME (training/clean's own) question_id_to_index
    mapping - the shared step both the clean and verbose sides of the
    comparison below call identically.
    """
    X, y, groups = build_transfer_xy(items)
    draws = np.asarray(predict_in_sample(mcmc, X.to_numpy(), groups, question_id_to_index))
    decomposition = posterior_predictive_entropy_decomposition(draws)
    return pd.DataFrame(
        {"question_id": groups, "aleatoric": decomposition["aleatoric"], "epistemic": decomposition["epistemic"]}
    )


def compute_verbose_shift_gap(
    clean_entropy: pd.DataFrame, verbose_entropy: pd.DataFrame, column: str, seed: int
) -> dict:
    """Paired cluster-bootstrap CI on mean(verbose[column]) -
    mean(clean[column]) - invariant 3's paired-comparison logic, same
    items (question_id) on both sides, never two independent samples.
    """
    def _mean(df: pd.DataFrame) -> float:
        return float(df[column].mean())

    diff, ci_low, ci_high = paired_cluster_bootstrap(
        verbose_entropy, clean_entropy, _mean, "question_id", n=2000, seed=seed
    )
    return {"gap": diff, "ci_low": ci_low, "ci_high": ci_high}


def main_verbose_shift(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    clean_items, verbose_items = load_rq3b_items(config.paths.items_parquet)
    print(f"Verbose-shift population: N={len(clean_items)} paired items (clean/P1 vs verbose/P1)")
    print(f"Feature set (D21 parity fix): {TRANSFER_SAFE_COLUMNS}")

    print("Fitting Bayesian model ONCE on all of clean (no CV, mirrors 5.7's transfer-test design)...")
    mcmc, question_id_to_index = fit_verbose_shift_model(clean_items, config.seed)

    clean_entropy = compute_condition_entropy(mcmc, clean_items, question_id_to_index)
    verbose_entropy = compute_condition_entropy(mcmc, verbose_items, question_id_to_index)

    aleatoric_clean = float(clean_entropy["aleatoric"].mean())
    aleatoric_verbose = float(verbose_entropy["aleatoric"].mean())
    epistemic_clean = float(clean_entropy["epistemic"].mean())
    epistemic_verbose = float(verbose_entropy["epistemic"].mean())
    print(f"Aleatoric: clean={aleatoric_clean:.4f}, verbose={aleatoric_verbose:.4f}")
    print(f"Epistemic: clean={epistemic_clean:.4f}, verbose={epistemic_verbose:.4f}")

    aleatoric_gap = compute_verbose_shift_gap(clean_entropy, verbose_entropy, "aleatoric", config.seed)
    epistemic_gap = compute_verbose_shift_gap(clean_entropy, verbose_entropy, "epistemic", config.seed)
    print(
        f"Aleatoric gap (verbose - clean): {aleatoric_gap['gap']:.4f} "
        f"[{aleatoric_gap['ci_low']:.4f}, {aleatoric_gap['ci_high']:.4f}]"
    )
    print(
        f"Epistemic gap (verbose - clean): {epistemic_gap['gap']:.4f} "
        f"[{epistemic_gap['ci_low']:.4f}, {epistemic_gap['ci_high']:.4f}]"
    )

    epistemic_rose = epistemic_gap["ci_low"] > 0
    aleatoric_flat = aleatoric_gap["ci_low"] <= 0 <= aleatoric_gap["ci_high"]
    prediction_held = epistemic_rose and aleatoric_flat

    table = pd.DataFrame.from_records(
        [
            {
                "aleatoric_clean_mean": aleatoric_clean,
                "aleatoric_verbose_mean": aleatoric_verbose,
                "aleatoric_gap": aleatoric_gap["gap"],
                "aleatoric_gap_ci_low": aleatoric_gap["ci_low"],
                "aleatoric_gap_ci_high": aleatoric_gap["ci_high"],
                "epistemic_clean_mean": epistemic_clean,
                "epistemic_verbose_mean": epistemic_verbose,
                "epistemic_gap": epistemic_gap["gap"],
                "epistemic_gap_ci_low": epistemic_gap["ci_low"],
                "epistemic_gap_ci_high": epistemic_gap["ci_high"],
                "epistemic_rose": epistemic_rose,
                "aleatoric_flat": aleatoric_flat,
                "prediction_held": prediction_held,
            }
        ]
    )
    table_path = f"results/rq5_verbose_shift_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    verdict = "HELD" if prediction_held else "DID NOT HOLD"
    print(f"Preregistered prediction (epistemic rises, aleatoric stays flat, D21/D23): {verdict}")

    plot_rq5_verbose_shift(
        aleatoric_clean=aleatoric_clean,
        aleatoric_verbose=aleatoric_verbose,
        aleatoric_gap_ci=(aleatoric_gap["ci_low"], aleatoric_gap["ci_high"]),
        epistemic_clean=epistemic_clean,
        epistemic_verbose=epistemic_verbose,
        epistemic_gap_ci=(epistemic_gap["ci_low"], epistemic_gap["ci_high"]),
        model_slug=config.model_slug,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--task",
        required=True,
        choices=["ablation", "h4", "transfer", "category", "calibration", "bayesian_comparison", "verbose_shift"],
    )
    args = parser.parse_args()
    if args.task == "ablation":
        main_ablation(args.config)
    elif args.task == "h4":
        main_h4(args.config)
    elif args.task == "transfer":
        main_transfer(args.config)
    elif args.task == "category":
        main_category(args.config)
    elif args.task == "calibration":
        main_calibration(args.config)
    elif args.task == "bayesian_comparison":
        main_bayesian_comparison(args.config)
    else:
        main_verbose_shift(args.config)
