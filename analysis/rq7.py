"""RQ7: does purpose-built judge training (auto-j-13b) resist the same
failure modes? (D28, PLAN.md §8)

`python -m analysis.rq7 --config configs/run_autoj.yaml --task {calibration,position_swap,verbosity,bayesian_recalibration}`

- Signals: conf_sc_autoj (self-consistency; clean only, since verbose has
  no sampled draws) and conf_sc_bpe_autoj (order-swap entropy from
  self-consistency proportions). The verbosity attack uses
  conf_sc_bpe_autoj_greedy - the same signal from the two greedy calls -
  so clean and verbose are scored on the same construction.
- Every test is split by turn (1 vs 2), both always reported.
- auto-j's signals have different null patterns from each other and from
  `correct` (ties and parse failures), so each test filters to its own
  signal's non-null rows right before computing - never one shared filter.

Reuses analysis/rq3.py's and rq6.py's recipes; only the population and
signal plumbing is new.
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
from analysis.rq6 import compute_signal_calibration
from src.bayesian import repeated_stratified_group_kfold_bayesian, summarize_diagnostics
from src.judge_autoj import AutojConfig
from src.plots import (
    plot_bayesian_convergence,
    plot_confidence_gap,
    plot_reliability_diagram,
    plot_verbosity_deltas,
    signal_label,
)
from src.predictor import build_xyg

AUTOJ_SIGNALS = ["conf_sc_autoj", "conf_sc_bpe_autoj"]
# Greedy-only on both sides: verbose never has sampled draws, so the pooled
# conf_sc_bpe_autoj would compare a 6-call clean signal against a 2-call
# verbose one. conf_sc_autoj is excluded outright (always null on verbose).
VERBOSITY_SIGNALS = ["conf_sc_bpe_autoj_greedy"]
# The entropy signals are calibrated through their probability forms
# (signals.py::prob_on_verdict); conf_sc_autoj already is a probability.
CALIBRATION_FORM = {
    "conf_sc_autoj": "conf_sc_autoj",
    "conf_sc_bpe_autoj": "conf_sc_bpe_autoj_prob",
    "conf_sc_bpe_autoj_greedy": "conf_sc_bpe_autoj_greedy_prob",
}


# --- Population loaders ---------------------------------------------------


def load_rq7_clean_items(items_parquet: str) -> pd.DataFrame:
    """Clean rows with a human label - the population for the calibration,
    position-swap, and Bayesian tests (each then filters further).
    """
    items = pd.read_parquet(items_parquet)
    return items[(items["condition"] == "clean") & items["human_label"].notna()].copy()


def load_rq7_paired_items(items_parquet: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(clean_items, verbose_items) for the verbosity attack: items with a
    human label where BOTH sides have the verbosity signal AND `correct`.

    The signal being present does not imply `correct` is: conf_sc_bpe_autoj
    needs only some call per order to give a usable verdict, while
    `correct` needs the greedy AB call specifically. Filtering on the signal
    alone once let `correct` be None and made ece() return NaN.
    Filters to the intersection, since auto-j's skips and parse failures are
    an expected, documented gap.
    """
    items = pd.read_parquet(items_parquet)
    items = items[items["human_label"].notna()]
    clean_all = items[items["condition"] == "clean"]
    verbose_all = items[items["condition"] == "verbose"]

    def _valid_ids(df: pd.DataFrame) -> set:
        valid = df[VERBOSITY_SIGNALS].notna().all(axis=1) & df["correct"].notna()
        return set(df.loc[valid, "item_id"])

    paired_ids = _valid_ids(clean_all) & _valid_ids(verbose_all)

    clean_items = clean_all[clean_all["item_id"].isin(paired_ids)].copy()
    verbose_items = verbose_all[verbose_all["item_id"].isin(paired_ids)].copy()
    return clean_items, verbose_items


# --- Calibration check ------------------------------------------------


def main_calibration(config_path: str) -> None:
    config = AutojConfig.from_yaml(config_path)
    clean_items = load_rq7_clean_items(config.paths.items_parquet)
    clean_items = clean_items[clean_items["correct"].notna()]

    print(f"RQ7 calibration population: N={len(clean_items)} (clean, human_label + correct present)")
    print(f"  turn=1: {(clean_items['turn'] == 1).sum()}, turn=2: {(clean_items['turn'] == 2).sum()}")

    rows = []
    for turn in [1, 2]:
        turn_items = clean_items[clean_items["turn"] == turn]
        for signal in AUTOJ_SIGNALS:
            signal_items = turn_items[turn_items[signal].notna()]
            if len(signal_items) == 0:
                continue
            metrics = compute_signal_calibration(
                signal_items, signal, config.n_bins, config.seed, CALIBRATION_FORM[signal]
            )
            rows.append({"turn": turn, "signal": signal, "n": len(signal_items), **metrics})
            print(
                f"  turn={turn}/{signal} (n={len(signal_items)}): ECE={metrics['ece']:.4f} "
                f"[{metrics['ece_ci_low']:.4f}, {metrics['ece_ci_high']:.4f}], "
                f"overconfidence_gap={metrics['overconfidence_gap']:.4f} "
                f"[{metrics['overconfidence_gap_ci_low']:.4f}, {metrics['overconfidence_gap_ci_high']:.4f}], "
                f"AUROC={metrics['auroc']:.4f} [{metrics['auroc_ci_low']:.4f}, {metrics['auroc_ci_high']:.4f}]"
            )
            plot_reliability_diagram(
                confidences=signal_items[CALIBRATION_FORM[signal]].to_numpy(),
                correct=signal_items["correct"].to_numpy(),
                signal_name=f"{signal}_turn{turn}",
                n_bins=config.n_bins,
                model_slug=config.model_slug,
                label="AB verdict",
                title=f"{signal_label(signal)} (auto-j-13b, turn {turn})",
            )

    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq7_calibration_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


# --- Position-swap attack ----------------------------------------------


def main_position_swap(config_path: str) -> None:
    """RQ3a's recipe per turn: flip rate and each signal's
    flipped-vs-unflipped confidence gap, on clean items.
    """
    config = AutojConfig.from_yaml(config_path)
    clean_items = load_rq7_clean_items(config.paths.items_parquet)
    clean_items = clean_items[clean_items["flipped"].notna()].copy()
    clean_items["flipped"] = clean_items["flipped"].astype(bool)

    print(f"RQ7 position-swap population: N={len(clean_items)} (clean, human_label + flipped present)")

    rows = []
    for turn in [1, 2]:
        turn_items = clean_items[clean_items["turn"] == turn]
        flip_result = compute_flip_rate(turn_items, config.seed)
        print(
            f"  turn={turn} (n={len(turn_items)}): flip_rate={flip_result['flip_rate']:.4f} "
            f"[{flip_result['flip_rate_ci_low']:.4f}, {flip_result['flip_rate_ci_high']:.4f}]"
        )

        for signal in AUTOJ_SIGNALS:
            signal_items = turn_items[turn_items[signal].notna()]
            if len(signal_items) == 0:
                continue
            gap_result = compute_confidence_gap(signal_items, signal, config.seed)
            rows.append({"turn": turn, "signal": signal, "n": len(signal_items), **flip_result, **gap_result})
            print(
                f"    {signal} (n={len(signal_items)}): gap={gap_result['gap_flipped_minus_unflipped']:.4f} "
                f"[{gap_result['gap_ci_low']:.4f}, {gap_result['gap_ci_high']:.4f}]"
            )

    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq7_position_swap_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    panels = []
    for turn, turn_table in table.groupby("turn", sort=True):
        flip_rate = turn_table["flip_rate"].iloc[0]
        panels.append(confidence_gap_panel(turn_table, f"turn {turn}: flip rate {flip_rate:.1%}"))
    plot_confidence_gap(
        panels,
        title="Is auto-j-13b less confident when answer order changes its verdict?",
        filename=f"rq7_position_swap_gap_{config.model_slug}.png",
    )


# --- Verbosity attack ----------------------------------------------------


def main_verbosity(config_path: str) -> None:
    """RQ3b's recipe per turn: paired ECE/accuracy/AUROC deltas,
    clean -> verbose, on VERBOSITY_SIGNALS.
    """
    config = AutojConfig.from_yaml(config_path)
    clean_items, verbose_items = load_rq7_paired_items(config.paths.items_parquet)
    print(f"RQ7 verbosity-attack population: N={len(clean_items)} paired items (clean vs verbose, signal present both sides)")

    rows = []
    for turn in [1, 2]:
        clean_turn = clean_items[clean_items["turn"] == turn]
        verbose_turn = verbose_items[verbose_items["turn"] == turn]
        print(f"  turn={turn}: N={len(clean_turn)}")

        for signal in VERBOSITY_SIGNALS:
            metrics = compute_signal_rq3b_metrics(
                clean_turn, verbose_turn, signal, "correct", config.n_bins, config.seed, CALIBRATION_FORM[signal]
            )
            rows.append({"turn": turn, "signal": signal, "n": len(clean_turn), **metrics})
            print(
                f"    {signal}: delta_ECE={metrics['delta_ece_verbose_minus_clean']:.4f} "
                f"[{metrics['delta_ece_ci_low']:.4f}, {metrics['delta_ece_ci_high']:.4f}], "
                f"delta_AUROC={metrics['delta_auroc_verbose_minus_clean']:.4f} "
                f"[{metrics['delta_auroc_ci_low']:.4f}, {metrics['delta_auroc_ci_high']:.4f}]"
            )

    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq7_verbosity_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    plot_verbosity_deltas(
        [verbosity_panel(t, f"turn {turn}") for turn, t in table.groupby("turn", sort=True)],
        title="Padding attack on auto-j-13b, among the calls it answered",
        filename=f"rq7_verbosity_deltas_{config.model_slug}.png",
    )


# --- Bayesian recalibration ------------------------------------------------


def build_autoj_tier(population: pd.DataFrame) -> pd.DataFrame:
    """The meta-model's features: auto-j's two signals, nothing else."""
    return population[AUTOJ_SIGNALS]


def main_bayesian_recalibration(config_path: str) -> None:
    """D22's check per turn: does a Bayesian meta-model over both signals
    beat the best single signal? Full D8 protocol, clean items where both
    signals are present.
    """
    config = AutojConfig.from_yaml(config_path)
    clean_items = load_rq7_clean_items(config.paths.items_parquet)
    clean_items = clean_items[clean_items["correct"].notna()]

    rows = []
    for turn in [1, 2]:
        turn_items = clean_items[clean_items["turn"] == turn]
        both_present = turn_items[turn_items[AUTOJ_SIGNALS].notna().all(axis=1)].reset_index(drop=True)
        print(f"RQ7 Bayesian recalibration (turn={turn}): N={len(both_present)}, fitting D8 protocol (5x10 NUTS)...")

        X, y, groups = build_xyg(both_present, build_autoj_tier)
        results = repeated_stratified_group_kfold_bayesian(X, y, groups, seed=config.seed)

        aurocs = np.array([r.auroc for r in results])
        fold_diagnostics = [fd for r in results for fd in r.fold_diagnostics]
        diagnostics = summarize_diagnostics(fold_diagnostics)

        best_single_auroc = max(
            compute_signal_calibration(
                both_present[both_present[signal].notna()], signal, config.n_bins, config.seed
            )["auroc"]
            for signal in AUTOJ_SIGNALS
        )

        print(
            f"  meta-model AUROC={aurocs.mean():.4f} [{aurocs.min():.4f}, {aurocs.max():.4f}] "
            f"(D8 across-repeat spread) vs. best single signal AUROC={best_single_auroc:.4f}"
        )
        print(f"  convergence: {diagnostics}")

        rows.append(
            {
                "turn": turn,
                "n": len(both_present),
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
            model_slug=f"{config.model_slug}_turn{turn}",
            title=f"Bayesian model convergence (auto-j-13b, turn {turn})",
        )

    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq7_bayesian_recalibration_{config.model_slug}.csv"
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
