"""RQ6 - stress-testing the industry's calibration counterclaim (kev-8b).
DECISIONS.md D27, PLAN.md SS7, TASKS.md's K1-K5/GATE K addendum.

Population/signal scope, all settled in D27 and confirmed against real
data (not assumed):
  - Two signals only: `conf_kev` (probabilities[choice], direct analog of
    conf_lp) and `conf_kev_bpe` (order-corrected bidirectional entropy,
    direct analog of conf_bpe). `confidence` is never used anywhere in
    this module - confirmed an exact deterministic rescaling of
    conf_kev, not an independent signal (D27's 23 Sep amendment).
  - Two coverage regimes: in-coverage (clean-side input_tokens <= 1024,
    kev's own disclosed training range) and out-of-coverage (> 1024).
    Regime is a property of the ITEM (its stable, unpadded clean length),
    applied identically to both its clean and verbose rows - not each
    condition's own inflated length, which would make the same item's
    regime membership different depending on which condition you're
    looking at and break the paired verbosity comparison.
  - The verbosity-attack population is 1,852 items (1,904 minus the 52
    kev-8b skipped for exceeding the 8,160-token serving ceiling on
    `verbose`), not 1,904 - see K3b's closeout note.

Mirrors analysis/rq1.py (calibration) and analysis/rq3.py (position-swap,
verbosity attack) exactly wherever the recipe transfers unchanged; only
the population/signal-set plumbing is new.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.rq3 import compute_confidence_gap, compute_flip_rate, compute_signal_rq3b_metrics
from src.bayesian import repeated_stratified_group_kfold_bayesian
from src.boot import cluster_bootstrap, paired_cluster_bootstrap
from src.judge_kev import KevConfig
from src.metrics import auroc_error, brier, ece, overconfidence_gap
from src.plots import FIGURES_DIR, _draw_forest, plot_bayesian_convergence, plot_reliability_diagram
from src.predictor import build_xyg

# plot_rq3a_confidence_gap()/plot_rq3b_deltas() both size their figure as
# 0.9 * len(signals) + 1.5 inches tall - fine at the primary judge's own
# 3-4 signal count, but too short at kev's 2 (D27), clipping their own
# fixed title text on save. _forest_plot() is shared with
# plot_d_human_correlations elsewhere, so its formula isn't touched here -
# same "widen the figure, don't touch a shared primitive" principle
# already established in this project (REPORT.md/TASKS.md's earlier
# title-clipping fixes). These two small RQ6-specific wrappers reuse
# _draw_forest() (the genuinely shared, title-free drawing primitive)
# directly, with enough height for their own titles at 2 labels.


def _plot_rq6_confidence_gap(signals: list[str], gap: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray, model_slug: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _draw_forest(ax, signals, gap, ci_low, ci_high, "mean confidence: flipped - unflipped")
    ax.set_title("Confidence gap on flipped vs. unflipped items (RQ6)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"rq6_position_swap_gap_{model_slug}.png"
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def _plot_rq6_verbosity_deltas(
    signals: list[str],
    delta_ece: np.ndarray, ece_ci_low: np.ndarray, ece_ci_high: np.ndarray,
    delta_auroc: np.ndarray, auroc_ci_low: np.ndarray, auroc_ci_high: np.ndarray,
    model_slug: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _draw_forest(axes[0], signals, delta_ece, ece_ci_low, ece_ci_high, xlabel="delta ECE (verbose - clean)")
    axes[0].set_title("Calibration")
    _draw_forest(axes[1], signals, delta_auroc, auroc_ci_low, auroc_ci_high, xlabel="delta AUROC (verbose - clean)")
    axes[1].set_title("Error-detection")
    axes[1].set_ylabel("")
    fig.suptitle("Verbosity's effect on calibration and error-detection (RQ6)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"rq6_verbosity_deltas_{model_slug}.png"
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)

KEV_SIGNALS = ["conf_kev", "conf_kev_bpe"]
IN_COVERAGE_THRESHOLD = 1024


def _load_all_kev_items(items_parquet: str) -> pd.DataFrame:
    return pd.read_parquet(items_parquet)


def _regime_by_item(clean_items: pd.DataFrame) -> pd.Series:
    """item_id -> "in_coverage" | "out_of_coverage", from the CLEAN row's
    own input_tokens (D27 - a stable property of the item's real,
    unpadded content, not of whichever condition happens to be in front
    of you). clean_items must already be the full clean population (every
    item has a valid clean row - 0 clean-side skips, confirmed in K3b),
    so this mapping is always fully defined for every item.
    """
    return clean_items.set_index("item_id")["input_tokens"].apply(
        lambda t: "in_coverage" if t <= IN_COVERAGE_THRESHOLD else "out_of_coverage"
    )


def load_rq6_clean_items(items_parquet: str) -> pd.DataFrame:
    """results/items_kev_8b.parquet -> the clean/human-labeled population
    RQ6's calibration check and position-swap test use - mirrors
    analysis/rq1.py::load_rq1_items()'s own clean-only, human-label-
    present scope exactly (RQ1/RQ3a are both clean-only for the primary
    judge; kev has no prompt_variant axis to also filter on).

    Adds a `regime` column (in_coverage/out_of_coverage), derived from
    this same clean population's own input_tokens - see _regime_by_item.
    """
    items = _load_all_kev_items(items_parquet)
    clean_items = items[(items["condition"] == "clean") & items["human_label"].notna()].copy()
    regime = _regime_by_item(items[items["condition"] == "clean"])
    clean_items["regime"] = clean_items["item_id"].map(regime)
    return clean_items


def load_rq6_paired_items(items_parquet: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """results/items_kev_8b.parquet -> (clean_items, verbose_items), the
    paired population RQ6's verbosity attack is allowed to touch.

    Unlike analysis/rq3.py::load_rq3b_items() (which asserts the two
    item_id sets are IDENTICAL and raises if not), this filters to the
    INTERSECTION - kev-8b's own 52-item verbose exclusion (over the
    8,160-token ceiling, D27) is a known, documented, expected gap, not a
    data-integrity failure to fail loudly about. Both sides carry the
    same `regime` column, from the clean-side mapping (see
    load_rq6_clean_items's own docstring for why).
    """
    items = _load_all_kev_items(items_parquet)
    items = items[items["human_label"].notna()]
    clean_all = items[items["condition"] == "clean"]
    verbose_all = items[items["condition"] == "verbose"]

    regime = _regime_by_item(clean_all)

    valid_clean_ids = set(clean_all.loc[clean_all["judge_verdict"].notna(), "item_id"])
    valid_verbose_ids = set(verbose_all.loc[verbose_all["judge_verdict"].notna(), "item_id"])
    paired_ids = valid_clean_ids & valid_verbose_ids

    clean_items = clean_all[clean_all["item_id"].isin(paired_ids)].copy()
    verbose_items = verbose_all[verbose_all["item_id"].isin(paired_ids)].copy()
    clean_items["regime"] = clean_items["item_id"].map(regime)
    verbose_items["regime"] = verbose_items["item_id"].map(regime)

    return clean_items, verbose_items


def compute_signal_calibration(items: pd.DataFrame, signal: str, n_bins: int, seed: int) -> dict:
    """ECE, Brier, and the signed overconfidence gap for one kev signal,
    on one already-regime-filtered population - mirrors
    analysis/rq1.py::compute_signal_metrics's exact recipe (ece/brier/
    overconfidence_gap; MCE and the Brier decomposition's
    reliability/resolution terms are RQ1-table-specific extras this
    module doesn't need for RQ6's DoD). `conf_kev_bpe`'s raw-nats range
    ([1-ln(2), 1] ~= [0.307, 1]) needs no rescaling before these - the
    same reasoning already confirmed for conf_bpe (D27, checked against
    the real formula before this was built, not assumed).

    Every metric ships with a cluster-bootstrap CI over question_id
    (invariant 2), same as every other calibration check in this project.
    """
    def _ece(df: pd.DataFrame) -> float:
        value, _ = ece(df[signal].to_numpy(), df["correct"].to_numpy(), n_bins)
        return value

    def _brier(df: pd.DataFrame) -> float:
        return brier(df[signal].to_numpy(), df["correct"].to_numpy())

    def _overconfidence_gap(df: pd.DataFrame) -> float:
        return overconfidence_gap(df[signal].to_numpy(), df["correct"].to_numpy())

    def _auroc(df: pd.DataFrame) -> float:
        uncertainty = 1 - df[signal].to_numpy(dtype=float)
        return auroc_error(uncertainty, df["correct"].to_numpy())

    result = {}
    for name, fn in [("ece", _ece), ("brier", _brier), ("overconfidence_gap", _overconfidence_gap), ("auroc", _auroc)]:
        point, ci_low, ci_high = cluster_bootstrap(items, fn, "question_id", n=2000, seed=seed)
        result[name] = point
        result[f"{name}_ci_low"] = ci_low
        result[f"{name}_ci_high"] = ci_high
    return result


def main_calibration(config_path: str) -> None:
    config = KevConfig.from_yaml(config_path)
    clean_items = load_rq6_clean_items(config.paths.items_parquet)

    print(f"RQ6 calibration population: N={len(clean_items)} (clean, human_label present)")
    print(f"  in_coverage: {(clean_items['regime'] == 'in_coverage').sum()}, "
          f"out_of_coverage: {(clean_items['regime'] == 'out_of_coverage').sum()}")

    rows = []
    for regime in ["in_coverage", "out_of_coverage"]:
        regime_items = clean_items[clean_items["regime"] == regime]
        for signal in KEV_SIGNALS:
            metrics = compute_signal_calibration(regime_items, signal, config.n_bins, config.seed)
            rows.append({"regime": regime, "signal": signal, "n": len(regime_items), **metrics})
            print(
                f"  {regime}/{signal}: ECE={metrics['ece']:.4f} "
                f"[{metrics['ece_ci_low']:.4f}, {metrics['ece_ci_high']:.4f}], "
                f"overconfidence_gap={metrics['overconfidence_gap']:.4f} "
                f"[{metrics['overconfidence_gap_ci_low']:.4f}, {metrics['overconfidence_gap_ci_high']:.4f}], "
                f"AUROC={metrics['auroc']:.4f} [{metrics['auroc_ci_low']:.4f}, {metrics['auroc_ci_high']:.4f}]"
            )
            plot_reliability_diagram(
                confidences=regime_items[signal].to_numpy(),
                correct=regime_items["correct"].to_numpy(),
                signal_name=f"{signal}_{regime}",
                n_bins=config.n_bins,
                model_slug=config.model_slug,
            )

    table = pd.DataFrame.from_records(rows)
    table_path = f"results/rq6_calibration_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


def main_position_swap(config_path: str) -> None:
    """RQ6's position-swap attack (mirrors analysis/rq3.py's RQ3a exactly:
    flip rate + mean-confidence-gap on flipped vs unflipped items),
    clean-only, split by coverage regime. Reuses compute_flip_rate/
    compute_confidence_gap unchanged - both are already generic over any
    DataFrame carrying `flipped`/`question_id`/a named signal column,
    which items_kev_8b.parquet does under the same column names.
    """
    config = KevConfig.from_yaml(config_path)
    clean_items = load_rq6_clean_items(config.paths.items_parquet)
    clean_items = clean_items[clean_items["flipped"].notna()].copy()
    clean_items["flipped"] = clean_items["flipped"].astype(bool)

    print(f"RQ6 position-swap population: N={len(clean_items)} (clean, human_label + flipped present)")

    rows = []
    for regime in ["in_coverage", "out_of_coverage"]:
        regime_items = clean_items[clean_items["regime"] == regime]
        flip_result = compute_flip_rate(regime_items, config.seed)
        print(
            f"  {regime}: flip_rate={flip_result['flip_rate']:.4f} "
            f"[{flip_result['flip_rate_ci_low']:.4f}, {flip_result['flip_rate_ci_high']:.4f}]"
        )

        gap_results = []
        for signal in KEV_SIGNALS:
            gap_result = compute_confidence_gap(regime_items, signal, config.seed)
            gap_results.append(gap_result)
            rows.append({"regime": regime, "signal": signal, "n": len(regime_items), **flip_result, **gap_result})
            print(
                f"    {signal}: gap={gap_result['gap_flipped_minus_unflipped']:.4f} "
                f"[{gap_result['gap_ci_low']:.4f}, {gap_result['gap_ci_high']:.4f}]"
            )

        _plot_rq6_confidence_gap(
            signals=KEV_SIGNALS,
            gap=np.array([g["gap_flipped_minus_unflipped"] for g in gap_results]),
            ci_low=np.array([g["gap_ci_low"] for g in gap_results]),
            ci_high=np.array([g["gap_ci_high"] for g in gap_results]),
            model_slug=f"{config.model_slug}_{regime}",
        )

    table = pd.DataFrame.from_records(rows)
    table_path = f"results/rq6_position_swap_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


def main_verbosity(config_path: str) -> None:
    """RQ6's verbosity attack (mirrors analysis/rq3.py's RQ3b exactly:
    paired ECE/accuracy/AUROC deltas, clean -> verbose), split by
    coverage regime. Population is load_rq6_paired_items()'s 1,784-item
    intersection (D27's pairing requirement), not the full 1,904/1,836 -
    an item missing on either side is dropped from BOTH sides for this
    specific test, not just the side that's actually missing.
    """
    config = KevConfig.from_yaml(config_path)
    clean_items, verbose_items = load_rq6_paired_items(config.paths.items_parquet)
    print(f"RQ6 verbosity-attack population: N={len(clean_items)} paired items (clean vs verbose, both succeeded)")

    rows = []
    for regime in ["in_coverage", "out_of_coverage"]:
        clean_regime = clean_items[clean_items["regime"] == regime]
        verbose_regime = verbose_items[verbose_items["regime"] == regime]
        print(f"  {regime}: N={len(clean_regime)}")

        ece_deltas, auroc_deltas = [], []
        for signal in KEV_SIGNALS:
            metrics = compute_signal_rq3b_metrics(
                clean_regime, verbose_regime, signal, "correct", config.n_bins, config.seed
            )
            rows.append({"regime": regime, "signal": signal, "n": len(clean_regime), **metrics})
            ece_deltas.append(metrics)
            auroc_deltas.append(metrics)
            print(
                f"    {signal}: delta_ECE={metrics['delta_ece_verbose_minus_clean']:.4f} "
                f"[{metrics['delta_ece_ci_low']:.4f}, {metrics['delta_ece_ci_high']:.4f}], "
                f"delta_AUROC={metrics['delta_auroc_verbose_minus_clean']:.4f} "
                f"[{metrics['delta_auroc_ci_low']:.4f}, {metrics['delta_auroc_ci_high']:.4f}]"
            )

        _plot_rq6_verbosity_deltas(
            signals=KEV_SIGNALS,
            delta_ece=np.array([m["delta_ece_verbose_minus_clean"] for m in ece_deltas]),
            ece_ci_low=np.array([m["delta_ece_ci_low"] for m in ece_deltas]),
            ece_ci_high=np.array([m["delta_ece_ci_high"] for m in ece_deltas]),
            delta_auroc=np.array([m["delta_auroc_verbose_minus_clean"] for m in auroc_deltas]),
            auroc_ci_low=np.array([m["delta_auroc_ci_low"] for m in auroc_deltas]),
            auroc_ci_high=np.array([m["delta_auroc_ci_high"] for m in auroc_deltas]),
            model_slug=f"{config.model_slug}_{regime}",
        )

    table = pd.DataFrame.from_records(rows)
    table_path = f"results/rq6_verbosity_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


def build_kev_tier(population: pd.DataFrame) -> pd.DataFrame:
    """kev's whole feature set - both signals, nothing else. `confidence`
    is never a column here (D27 - confirmed redundant with conf_kev).
    Passed straight through build_xyg()'s encode_features() call, which
    is a no-op on two pure-float columns (no categorical/boolean columns
    present), same as it already is on the primary study's own Tier A.
    """
    return population[KEV_SIGNALS]


def main_bayesian_recalibration(config_path: str) -> None:
    """RQ6's D22 recalibration check: does a Bayesian hierarchical
    meta-model over BOTH kev signals together beat kev's own best single
    raw signal at predicting kev's own errors? Full D8 protocol (5-fold x
    10-repeat NUTS), per the owner's explicit confirmation - same rigor as
    the primary judge's own RQ4 Bayesian arm, not a lighter version.
    Clean-only population (mirrors 5.9b/5.9c's own scope), split by
    coverage regime.
    """
    config = KevConfig.from_yaml(config_path)
    clean_items = load_rq6_clean_items(config.paths.items_parquet)

    rows = []
    for regime in ["in_coverage", "out_of_coverage"]:
        population = clean_items[clean_items["regime"] == regime].reset_index(drop=True)
        print(f"RQ6 Bayesian recalibration ({regime}): N={len(population)}, fitting D8 protocol (5x10 NUTS)...")

        X, y, groups = build_xyg(population, build_kev_tier)
        results = repeated_stratified_group_kfold_bayesian(X, y, groups, seed=config.seed)

        aurocs = np.array([r.auroc for r in results])
        fold_diagnostics = [fd for r in results for fd in r.fold_diagnostics]
        n_flagged = sum(fd["flagged"] for fd in fold_diagnostics)

        best_single_auroc = max(
            compute_signal_calibration(population, signal, config.n_bins, config.seed)["auroc"]
            for signal in KEV_SIGNALS
        )

        print(
            f"  meta-model AUROC={aurocs.mean():.4f} [{aurocs.min():.4f}, {aurocs.max():.4f}] "
            f"(D8 across-repeat spread) vs. best single signal AUROC={best_single_auroc:.4f}"
        )
        print(f"  convergence: {n_flagged}/{len(fold_diagnostics)} fold-fits flagged")

        rows.append(
            {
                "regime": regime,
                "n": len(population),
                "meta_model_auroc_mean": aurocs.mean(),
                "meta_model_auroc_min": aurocs.min(),
                "meta_model_auroc_max": aurocs.max(),
                "best_single_signal_auroc": best_single_auroc,
                "n_folds_flagged": n_flagged,
                "n_folds_total": len(fold_diagnostics),
            }
        )

        plot_bayesian_convergence(
            max_rhat=np.array([fd["max_rhat"] for fd in fold_diagnostics]),
            flagged=np.array([fd["flagged"] for fd in fold_diagnostics]),
            model_slug=f"{config.model_slug}_{regime}",
        )

    table = pd.DataFrame.from_records(rows)
    table_path = f"results/rq6_bayesian_recalibration_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--task", required=True, choices=["calibration", "position_swap", "verbosity", "bayesian_recalibration"]
    )
    args = parser.parse_args()

    if args.task == "calibration":
        main_calibration(args.config)
    elif args.task == "position_swap":
        main_position_swap(args.config)
    elif args.task == "verbosity":
        main_verbosity(args.config)
    elif args.task == "bayesian_recalibration":
        main_bayesian_recalibration(args.config)
