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
from src.boot import cluster_bootstrap
from src.config import Config
from src.features import load_rq4_population
from src.metrics import auroc_error
from src.plots import plot_rq4_ablation
from src.predictor import MODEL_FACTORIES, TIER_BUILDERS, run_predictor


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
    raise NotImplementedError(
        "task 5.5: for each signal in SIGNALS, cluster_bootstrap a stat_fn computing "
        "auroc_error(1 - df[signal], df['correct']) over `population`, grouped on question_id; "
        "return the signal with the highest point estimate"
    )


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
    raise NotImplementedError(
        "task 5.5: results = run_predictor(population, TIER_BUILDERS[tier_name], model_name, seed); "
        "aurocs = [r.auroc for r in results]; report mean/min/max"
    )


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
