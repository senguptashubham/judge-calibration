"""RQ6: does kev-8b resist the failure modes found in the primary judge?
(D27, PLAN.md §7)

`python -m analysis.rq6 --config configs/run_kev.yaml --task {calibration,position_swap,verbosity,bayesian_recalibration}`

- Two signals: conf_kev (probabilities[choice], the analog of conf_lp) and
  conf_kev_bpe (order-corrected bidirectional entropy, the analog of
  conf_bpe; calibrated through conf_kev_bpe_prob). kev's `confidence`
  field is never used - it is an exact rescaling of conf_kev (D27).
- Every test is split by coverage regime: in-coverage when the item's
  CLEAN input_tokens <= 1,024 (kev's training range), else out-of-coverage.
  The regime is a property of the item, applied to both its conditions, so
  the paired verbosity comparison keeps the same items in each regime.
- The verbosity attack runs on 1,784 paired items: 52 verbose items were
  skipped over kev's 8,160-token ceiling, and both sides need a verdict and
  a human label.

Reuses analysis/rq3.py's recipes unchanged; only the population and signal
plumbing is new.
"""

import argparse

import numpy as np
import pandas as pd

from analysis.rq3 import (
    compute_confidence_gap,
    compute_flip_rate,
    compute_signal_rq3b_metrics,
    confidence_gap_panel,
    verbosity_panel,
)
from src.bayesian import repeated_stratified_group_kfold_bayesian, summarize_diagnostics
from src.boot import cluster_bootstrap, paired_cluster_bootstrap
from src.judge_kev import KevConfig
from src.metrics import auroc_error, brier, ece, overconfidence_gap
from src.plots import (
    plot_bayesian_convergence,
    plot_confidence_gap,
    plot_reliability_diagram,
    plot_verbosity_deltas,
    signal_label,
)
from src.predictor import build_xyg

KEV_SIGNALS = ["conf_kev", "conf_kev_bpe"]
CALIBRATION_FORM = {"conf_kev": "conf_kev", "conf_kev_bpe": "conf_kev_bpe_prob"}
IN_COVERAGE_THRESHOLD = 1024
REGIME_TITLES = {"in_coverage": "in coverage (≤ 1,024 tokens)", "out_of_coverage": "out of coverage"}


def _load_all_kev_items(items_parquet: str) -> pd.DataFrame:
    return pd.read_parquet(items_parquet)


def _regime_by_item(clean_items: pd.DataFrame) -> pd.Series:
    """item_id -> "in_coverage" | "out_of_coverage", from the CLEAN row's
    input_tokens. Every item has a clean row (no clean call was skipped).
    """
    return clean_items.set_index("item_id")["input_tokens"].apply(
        lambda t: "in_coverage" if t <= IN_COVERAGE_THRESHOLD else "out_of_coverage"
    )


def load_rq6_clean_items(items_parquet: str) -> pd.DataFrame:
    """Clean rows with a human label, plus a `regime` column - the
    population for the calibration, position-swap, and Bayesian tests.
    """
    items = _load_all_kev_items(items_parquet)
    clean_items = items[(items["condition"] == "clean") & items["human_label"].notna()].copy()
    regime = _regime_by_item(items[items["condition"] == "clean"])
    clean_items["regime"] = clean_items["item_id"].map(regime)
    return clean_items


def load_rq6_paired_items(items_parquet: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(clean_items, verbose_items) for the verbosity attack: items with a
    human label and a verdict on BOTH sides. Filters to the intersection
    rather than asserting equal sets (unlike rq3.load_rq3b_items), since
    kev's 52 skipped verbose items are an expected, documented gap.
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


def compute_signal_calibration(
    items: pd.DataFrame, signal: str, n_bins: int, seed: int, calibration_col: str | None = None
) -> dict:
    """ECE, Brier, overconfidence gap, and AUROC for one signal, each with a
    cluster-bootstrap CI - RQ1's recipe. The caller must already have
    filtered `items` to this signal's non-null rows.

    ECE, Brier and the gap use `calibration_col` (default: `signal`), AUROC
    uses `signal`: an entropy signal ranks errors but needs its probability
    form for calibration (signals.py::prob_on_verdict).
    """
    calibration_col = calibration_col or signal

    def _ece(df: pd.DataFrame) -> float:
        value, _ = ece(df[calibration_col].to_numpy(), df["correct"].to_numpy(), n_bins)
        return value

    def _brier(df: pd.DataFrame) -> float:
        return brier(df[calibration_col].to_numpy(), df["correct"].to_numpy())

    def _overconfidence_gap(df: pd.DataFrame) -> float:
        return overconfidence_gap(df[calibration_col].to_numpy(), df["correct"].to_numpy())

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
            metrics = compute_signal_calibration(
                regime_items, signal, config.n_bins, config.seed, CALIBRATION_FORM[signal]
            )
            rows.append({"regime": regime, "signal": signal, "n": len(regime_items), **metrics})
            print(
                f"  {regime}/{signal}: ECE={metrics['ece']:.4f} "
                f"[{metrics['ece_ci_low']:.4f}, {metrics['ece_ci_high']:.4f}], "
                f"overconfidence_gap={metrics['overconfidence_gap']:.4f} "
                f"[{metrics['overconfidence_gap_ci_low']:.4f}, {metrics['overconfidence_gap_ci_high']:.4f}], "
                f"AUROC={metrics['auroc']:.4f} [{metrics['auroc_ci_low']:.4f}, {metrics['auroc_ci_high']:.4f}]"
            )
            plot_reliability_diagram(
                confidences=regime_items[CALIBRATION_FORM[signal]].to_numpy(),
                correct=regime_items["correct"].to_numpy(),
                signal_name=f"{signal}_{regime}",
                n_bins=config.n_bins,
                model_slug=config.model_slug,
                label="AB verdict",
                title=f"{signal_label(signal)} (kev-8b, {REGIME_TITLES[regime]})",
            )

    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq6_calibration_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


def main_position_swap(config_path: str) -> None:
    """RQ3a's recipe per regime: flip rate and each signal's
    flipped-vs-unflipped confidence gap, on clean items.
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

        for signal in KEV_SIGNALS:
            gap_result = compute_confidence_gap(regime_items, signal, config.seed)
            rows.append({"regime": regime, "signal": signal, "n": len(regime_items), **flip_result, **gap_result})
            print(
                f"    {signal}: gap={gap_result['gap_flipped_minus_unflipped']:.4f} "
                f"[{gap_result['gap_ci_low']:.4f}, {gap_result['gap_ci_high']:.4f}]"
            )

    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq6_position_swap_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    panels = []
    for regime, regime_table in table.groupby("regime", sort=False):
        flip_rate = regime_table["flip_rate"].iloc[0]
        panels.append(confidence_gap_panel(regime_table, f"{REGIME_TITLES[regime]}: flip rate {flip_rate:.1%}"))
    plot_confidence_gap(
        panels,
        title="Is kev-8b less confident when answer order changes its verdict?",
        filename=f"rq6_position_swap_gap_{config.model_slug}.png",
    )


def main_verbosity(config_path: str) -> None:
    """RQ3b's recipe per regime: paired ECE/accuracy/AUROC deltas,
    clean -> verbose, on the paired population.
    """
    config = KevConfig.from_yaml(config_path)
    clean_items, verbose_items = load_rq6_paired_items(config.paths.items_parquet)
    print(f"RQ6 verbosity-attack population: N={len(clean_items)} paired items (clean vs verbose, both succeeded)")

    rows = []
    for regime in ["in_coverage", "out_of_coverage"]:
        clean_regime = clean_items[clean_items["regime"] == regime]
        verbose_regime = verbose_items[verbose_items["regime"] == regime]
        print(f"  {regime}: N={len(clean_regime)}")

        for signal in KEV_SIGNALS:
            metrics = compute_signal_rq3b_metrics(
                clean_regime, verbose_regime, signal, "correct", config.n_bins, config.seed, CALIBRATION_FORM[signal]
            )
            rows.append({"regime": regime, "signal": signal, "n": len(clean_regime), **metrics})
            print(
                f"    {signal}: delta_ECE={metrics['delta_ece_verbose_minus_clean']:.4f} "
                f"[{metrics['delta_ece_ci_low']:.4f}, {metrics['delta_ece_ci_high']:.4f}], "
                f"delta_AUROC={metrics['delta_auroc_verbose_minus_clean']:.4f} "
                f"[{metrics['delta_auroc_ci_low']:.4f}, {metrics['delta_auroc_ci_high']:.4f}]"
            )

    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq6_verbosity_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    plot_verbosity_deltas(
        [verbosity_panel(t, REGIME_TITLES[r]) for r, t in table.groupby("regime", sort=False)],
        title="Padding attack on kev-8b: calibration and error detection",
        filename=f"rq6_verbosity_deltas_{config.model_slug}.png",
    )


def build_kev_tier(population: pd.DataFrame) -> pd.DataFrame:
    """The meta-model's features: kev's two signals, nothing else."""
    return population[KEV_SIGNALS]


def main_bayesian_recalibration(config_path: str) -> None:
    """D22's check per regime: does a Bayesian meta-model over both signals
    beat the best single signal at predicting kev's own errors? Full D8
    protocol (5-fold x 10 repeats, NUTS) on clean items.
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
        diagnostics = summarize_diagnostics(fold_diagnostics)

        best_single_auroc = max(
            compute_signal_calibration(population, signal, config.n_bins, config.seed)["auroc"]
            for signal in KEV_SIGNALS
        )

        print(
            f"  meta-model AUROC={aurocs.mean():.4f} [{aurocs.min():.4f}, {aurocs.max():.4f}] "
            f"(D8 across-repeat spread) vs. best single signal AUROC={best_single_auroc:.4f}"
        )
        print(f"  convergence: {diagnostics}")

        rows.append(
            {
                "regime": regime,
                "n": len(population),
                "meta_model_auroc_mean": aurocs.mean(),
                "meta_model_auroc_min": aurocs.min(),
                "meta_model_auroc_max": aurocs.max(),
                "best_single_signal_auroc": best_single_auroc,
                **diagnostics,
            }
        )

        plot_bayesian_convergence(
            max_rhat=np.array([fd["max_rhat"] for fd in fold_diagnostics]),
            flagged=np.array([fd["flagged"] for fd in fold_diagnostics]),
            n_divergences=np.array([fd["n_divergences"] for fd in fold_diagnostics]),
            model_slug=f"{config.model_slug}_{regime}",
            title=f"Bayesian model convergence (kev-8b, {REGIME_TITLES[regime]})",
        )

    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq6_bayesian_recalibration_{config.model_slug}.csv"
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
