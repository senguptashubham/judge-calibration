"""RQ7 - does purpose-built judge training resist the same failure modes?
(auto-j-13b). DECISIONS.md D28, PLAN.md SS8, TASKS.md's L1-L6/GATE L addendum.

Population/signal scope, settled in D28 and confirmed against real data
(not assumed):
  - Two signals: `conf_sc_autoj` (self-consistency, direct D6 port,
    clean-only - never populated on verbose, no sampled draws exist there)
    and `conf_sc_bpe_autoj` (order-swap bidirectional-entropy analog,
    built from self-consistency proportions - computed on BOTH conditions,
    D28's 24 Sep fix, mirroring D21's own conf_sc-excluded/conf_bpe-
    included precedent for the primary judge).
  - No coverage-regime split (unlike RQ6) - auto-j's disclosed context
    covers the population without a ceiling issue. Population is instead
    split by `turn` (1 vs 2) throughout - both turns are always reported
    (D28's 23 Sep reversal), never silently merged or dropped.
  - Unlike RQ6's kev-8b arm (whose two signals happened to share an
    identical null pattern with judge_verdict, since kev's clean
    population had zero skips), auto-j's own null patterns genuinely
    differ per signal, even on `clean` (real Tie/parse-failure rates
    there - confirmed: judge_verdict 2.05% null, conf_sc_bpe_autoj 1.84%
    null on clean, NOT identical sets). Every function below filters its
    OWN population to that specific signal's non-null rows immediately
    before computing on it, never relying on one blanket filter shared
    across signals the way rq6.py could.
  - The verbosity-attack test uses `conf_sc_bpe_autoj` ALONE - conf_sc_autoj
    is unconditionally null on `verbose` (no data exists there at all, not
    a partial gap), so it cannot be part of a paired clean/verbose
    comparison, mirroring D21's own conf_sc exclusion from RQ3b exactly.

Mirrors analysis/rq1.py (calibration) and analysis/rq3.py (position-swap,
verbosity attack) exactly wherever the recipe transfers unchanged; only
the population/signal-set plumbing is new, same as rq6.py's own approach.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.rq3 import compute_confidence_gap, compute_flip_rate, compute_signal_rq3b_metrics
from src.bayesian import repeated_stratified_group_kfold_bayesian
from src.boot import cluster_bootstrap
from src.judge_autoj import AutojConfig
from src.metrics import auroc_error, brier, ece, overconfidence_gap
from src.plots import FIGURES_DIR, _draw_forest, plot_bayesian_convergence, plot_reliability_diagram
from src.predictor import build_xyg

AUTOJ_SIGNALS = ["conf_sc_autoj", "conf_sc_bpe_autoj"]
VERBOSITY_SIGNALS = ["conf_sc_bpe_autoj"]  # conf_sc_autoj excluded - always null on verbose (D21 precedent)


# --- Plotting wrappers -------------------------------------------------
#
# Same reason rq6.py needed its own: plot_rq3a_confidence_gap()/
# plot_rq3b_deltas() size their figure at 0.9*len(signals)+1.5 inches,
# fine at the primary judge's 3-4 signals but too short at 1-2, clipping
# the fixed title text. Reuses _draw_forest() (the shared, title-free
# primitive) directly, same "widen the figure, don't touch a shared
# primitive" principle already established.


def _plot_rq7_confidence_gap(signals: list[str], gap: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray, model_slug: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _draw_forest(ax, signals, gap, ci_low, ci_high, "mean confidence: flipped - unflipped")
    ax.set_title("Confidence gap on flipped vs. unflipped items (RQ7)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"rq7_position_swap_gap_{model_slug}.png"
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


def _plot_rq7_verbosity_deltas(
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
    fig.suptitle("Verbosity's effect on calibration and error-detection (RQ7)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"rq7_verbosity_deltas_{model_slug}.png"
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    plt.close(fig)


# --- Population loaders ---------------------------------------------------


def load_rq7_clean_items(items_parquet: str) -> pd.DataFrame:
    """results/items_autoj_13b_gptq_4bits.parquet -> the clean/human-
    labeled population RQ7's calibration check, position-swap attack, and
    Bayesian recalibration use - mirrors load_rq1_items()/
    load_rq6_clean_items() exactly. `turn` is already a native column
    (auto-j has no prompt_variant axis, and unlike RQ6's synthesized
    `regime`, no extra computation is needed - the checkpoint already
    carries the item's real turn).
    """
    items = pd.read_parquet(items_parquet)
    return items[(items["condition"] == "clean") & items["human_label"].notna()].copy()


def load_rq7_paired_items(items_parquet: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(clean_items, verbose_items), the paired population for RQ7's
    verbosity-attack test - restricted to items where BOTH sides have a
    valid `conf_sc_bpe_autoj` AND a valid `correct` (the only signal usable
    for this test, see module docstring - `compute_signal_rq3b_metrics`
    scores accuracy/ECE/AUROC against `correct_col="correct"` too, not
    just the signal itself). Filters to the INTERSECTION, not an exact-
    match assertion (mirrors load_rq6_paired_items() - auto-j's parse-
    failure-driven exclusions are a known, expected gap, not a data-
    integrity failure to fail loudly about).

    `conf_sc_bpe_autoj.notna()` does NOT imply `correct.notna()`, and this
    was a real bug caught against real data, not assumed: `conf_sc_bpe_autoj`
    only needs ANY one of AB's k_sc+1 draws to produce a valid canonical
    letter, while `correct` needs SPECIFICALLY the greedy (sample_idx=0)
    AB call to succeed (D7's own single-pass definition) - so an item can
    have a populated signal from a later sampled draw while its greedy
    call itself failed/tied, leaving `correct` null. Filtering on the
    signal alone let `correct` be None on the `verbose` side reach
    compute_signal_rq3b_metrics's `_ece`/`_accuracy`/`_auroc` stat_fns,
    which don't guard against it - `ece()` silently returned NaN.
    """
    items = pd.read_parquet(items_parquet)
    items = items[items["human_label"].notna()]
    clean_all = items[items["condition"] == "clean"]
    verbose_all = items[items["condition"] == "verbose"]

    def _valid_ids(df: pd.DataFrame) -> set:
        valid = df["conf_sc_bpe_autoj"].notna() & df["correct"].notna()
        return set(df.loc[valid, "item_id"])

    paired_ids = _valid_ids(clean_all) & _valid_ids(verbose_all)

    clean_items = clean_all[clean_all["item_id"].isin(paired_ids)].copy()
    verbose_items = verbose_all[verbose_all["item_id"].isin(paired_ids)].copy()
    return clean_items, verbose_items


# --- Calibration check ------------------------------------------------


def compute_signal_calibration(items: pd.DataFrame, signal: str, n_bins: int, seed: int) -> dict:
    """ECE, Brier, overconfidence gap, and AUROC for one auto-j signal, on
    one already-turn-filtered population - mirrors
    analysis/rq6.py::compute_signal_calibration's exact recipe. `items`
    must already be filtered to `items[signal].notna()` by the caller -
    this function does not filter, since different signals here need
    different filters (module docstring), unlike rq6.py's kev arm.
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
            metrics = compute_signal_calibration(signal_items, signal, config.n_bins, config.seed)
            rows.append({"turn": turn, "signal": signal, "n": len(signal_items), **metrics})
            print(
                f"  turn={turn}/{signal} (n={len(signal_items)}): ECE={metrics['ece']:.4f} "
                f"[{metrics['ece_ci_low']:.4f}, {metrics['ece_ci_high']:.4f}], "
                f"overconfidence_gap={metrics['overconfidence_gap']:.4f} "
                f"[{metrics['overconfidence_gap_ci_low']:.4f}, {metrics['overconfidence_gap_ci_high']:.4f}], "
                f"AUROC={metrics['auroc']:.4f} [{metrics['auroc_ci_low']:.4f}, {metrics['auroc_ci_high']:.4f}]"
            )
            plot_reliability_diagram(
                confidences=signal_items[signal].to_numpy(),
                correct=signal_items["correct"].to_numpy(),
                signal_name=f"{signal}_turn{turn}",
                n_bins=config.n_bins,
                model_slug=config.model_slug,
            )

    table = pd.DataFrame.from_records(rows)
    table_path = f"results/rq7_calibration_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


# --- Position-swap attack ----------------------------------------------


def main_position_swap(config_path: str) -> None:
    """RQ7's position-swap attack (mirrors analysis/rq3.py's RQ3a exactly:
    flip rate + mean-confidence-gap on flipped vs unflipped items),
    clean-only, split by turn. Reuses compute_flip_rate/
    compute_confidence_gap unchanged.
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

        gap_results = []
        for signal in AUTOJ_SIGNALS:
            signal_items = turn_items[turn_items[signal].notna()]
            if len(signal_items) == 0:
                continue
            gap_result = compute_confidence_gap(signal_items, signal, config.seed)
            gap_results.append((signal, gap_result))
            rows.append({"turn": turn, "signal": signal, "n": len(signal_items), **flip_result, **gap_result})
            print(
                f"    {signal} (n={len(signal_items)}): gap={gap_result['gap_flipped_minus_unflipped']:.4f} "
                f"[{gap_result['gap_ci_low']:.4f}, {gap_result['gap_ci_high']:.4f}]"
            )

        if gap_results:
            _plot_rq7_confidence_gap(
                signals=[s for s, _ in gap_results],
                gap=np.array([g["gap_flipped_minus_unflipped"] for _, g in gap_results]),
                ci_low=np.array([g["gap_ci_low"] for _, g in gap_results]),
                ci_high=np.array([g["gap_ci_high"] for _, g in gap_results]),
                model_slug=f"{config.model_slug}_turn{turn}",
            )

    table = pd.DataFrame.from_records(rows)
    table_path = f"results/rq7_position_swap_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


# --- Verbosity attack ----------------------------------------------------


def main_verbosity(config_path: str) -> None:
    """RQ7's verbosity attack (mirrors analysis/rq3.py's RQ3b exactly:
    paired ECE/accuracy/AUROC deltas, clean -> verbose), split by turn.
    `conf_sc_autoj` is excluded (D21 precedent, module docstring) - only
    `conf_sc_bpe_autoj` runs here.
    """
    config = AutojConfig.from_yaml(config_path)
    clean_items, verbose_items = load_rq7_paired_items(config.paths.items_parquet)
    print(f"RQ7 verbosity-attack population: N={len(clean_items)} paired items (clean vs verbose, conf_sc_bpe_autoj present both sides)")

    rows = []
    for turn in [1, 2]:
        clean_turn = clean_items[clean_items["turn"] == turn]
        verbose_turn = verbose_items[verbose_items["turn"] == turn]
        print(f"  turn={turn}: N={len(clean_turn)}")

        ece_deltas, auroc_deltas = [], []
        for signal in VERBOSITY_SIGNALS:
            metrics = compute_signal_rq3b_metrics(clean_turn, verbose_turn, signal, "correct", config.n_bins, config.seed)
            rows.append({"turn": turn, "signal": signal, "n": len(clean_turn), **metrics})
            ece_deltas.append(metrics)
            auroc_deltas.append(metrics)
            print(
                f"    {signal}: delta_ECE={metrics['delta_ece_verbose_minus_clean']:.4f} "
                f"[{metrics['delta_ece_ci_low']:.4f}, {metrics['delta_ece_ci_high']:.4f}], "
                f"delta_AUROC={metrics['delta_auroc_verbose_minus_clean']:.4f} "
                f"[{metrics['delta_auroc_ci_low']:.4f}, {metrics['delta_auroc_ci_high']:.4f}]"
            )

        _plot_rq7_verbosity_deltas(
            signals=VERBOSITY_SIGNALS,
            delta_ece=np.array([m["delta_ece_verbose_minus_clean"] for m in ece_deltas]),
            ece_ci_low=np.array([m["delta_ece_ci_low"] for m in ece_deltas]),
            ece_ci_high=np.array([m["delta_ece_ci_high"] for m in ece_deltas]),
            delta_auroc=np.array([m["delta_auroc_verbose_minus_clean"] for m in auroc_deltas]),
            auroc_ci_low=np.array([m["delta_auroc_ci_low"] for m in auroc_deltas]),
            auroc_ci_high=np.array([m["delta_auroc_ci_high"] for m in auroc_deltas]),
            model_slug=f"{config.model_slug}_turn{turn}",
        )

    table = pd.DataFrame.from_records(rows)
    table_path = f"results/rq7_verbosity_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


# --- Bayesian recalibration ------------------------------------------------


def build_autoj_tier(population: pd.DataFrame) -> pd.DataFrame:
    """auto-j's whole feature set - both signals, nothing else. Passed
    straight through build_xyg()'s encode_features() call, a no-op on two
    pure-float columns, same as rq6.py's build_kev_tier and the primary
    study's own Tier A.
    """
    return population[AUTOJ_SIGNALS]


def main_bayesian_recalibration(config_path: str) -> None:
    """RQ7's D22 recalibration check: does a Bayesian hierarchical meta-
    model over both auto-j signals together beat auto-j's own best single
    raw signal at predicting auto-j's own errors? Full D8 protocol
    (5-fold x 10-repeat NUTS), same rigor as the primary judge's and
    kev-8b's own RQ4/RQ6 Bayesian arms. Clean-only population, split by
    turn. Restricted to rows where BOTH signals are non-null (the meta-
    model needs both columns simultaneously, unlike the single-signal
    calibration check above).
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
        n_flagged = sum(fd["flagged"] for fd in fold_diagnostics)

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
        print(f"  convergence: {n_flagged}/{len(fold_diagnostics)} fold-fits flagged")

        rows.append(
            {
                "turn": turn,
                "n": len(both_present),
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
            model_slug=f"{config.model_slug}_turn{turn}",
        )

    table = pd.DataFrame.from_records(rows)
    table_path = f"results/rq7_bayesian_recalibration_{config.model_slug}.csv"
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
