"""RQ4 task 5.5: tier ablation A -> B -> C, both frequentist models,
against the best-single-signal baseline. See TASKS.md task 5.5,
PLAN.md §2.2's reframe ("which feature family carries the signal").

`python -m analysis.rq4 --config configs/run.yaml`.

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
"""

import argparse

import numpy as np
import pandas as pd

from analysis.rq1 import SIGNALS
from src.boot import cluster_bootstrap, paired_cluster_bootstrap
from src.config import Config
from src.features import load_rq4_population
from src.metrics import auroc_error
from src.plots import plot_rq4_ablation
from src.predictor import MODEL_FACTORIES, TIER_BUILDERS, build_xyg, permutation_null, percentile_of_null, run_predictor


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
        percentile (observed's percentile of the null distribution).
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


def main(config_path: str) -> None:
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
    null_table = pd.DataFrame.from_records(null_rows)
    null_table_path = f"results/rq4_ablation_null_{config.model_slug}.csv"
    null_table.to_csv(null_table_path, index=False)
    print(f"Wrote {len(null_table)} rows to {null_table_path}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
