"""Matplotlib figures, one per function, each saved to results/figures/ under
a deterministic, model_slug-suffixed name (CLAUDE.md §5, D26).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.metrics import auroc_error, ece, get_bin_edges

FIGURES_DIR = Path("results/figures")


def plot_ece_auroc_orthogonal() -> Figure:
    """LEARNING.md C4: two hand-built 6-item examples showing ECE and AUROC
    measure different things (invariant 6).

    A: correct items at confidence 0.9, wrong at 0.5 - perfect ranking
       (AUROC 1.0) with miscalibrated numbers (ECE ~0.3).
    B: every item at 0.5 on a 50% base rate - perfectly calibrated (ECE 0)
       and useless for ranking (AUROC 0.5): "a judge that always says 50%".

    The asserts guard the construction; fix the example, never silence them.
    """
    confidences_a = np.array([0.5, 0.5, 0.9, 0.5, 0.9, 0.9])
    correct_a = np.array([0, 0, 1, 0, 1, 1])

    confidences_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    correct_b = np.array([0, 1, 0, 0, 1, 1])

    ece_a, _ = ece(confidences_a, correct_a, n_bins=2)
    auroc_a = auroc_error(1 - confidences_a, correct_a)
    ece_b, _ = ece(confidences_b, correct_b, n_bins=2)
    auroc_b = auroc_error(1 - confidences_b, correct_b)

    assert abs(ece_a - 0.3) < 0.02, f"Example A: ECE={ece_a}, target ~0.3"
    assert auroc_a == 1.0, f"Example A: AUROC={auroc_a}, target 1.0"
    assert ece_b == 0.0, f"Example B: ECE={ece_b}, target 0.0"
    assert auroc_b == 0.5, f"Example B: AUROC={auroc_b}, target 0.5"

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, confidences, correct, ece_val, auroc_val, label in [
        (axes[0], confidences_a, correct_a, ece_a, auroc_a, "A"),
        (axes[1], confidences_b, correct_b, ece_b, auroc_b, "B"),
    ]:
        correct_bool = np.asarray(correct, dtype=bool)
        x = np.arange(len(confidences))
        ax.scatter(x[correct_bool], confidences[correct_bool], c="tab:green", label="correct")
        ax.scatter(x[~correct_bool], confidences[~correct_bool], c="tab:red", label="incorrect")
        ax.set_ylim(0.4, 1.05)
        ax.set_xlabel("item")
        ax.set_title(f"Example {label}\nECE={ece_val:.2f}, AUROC={auroc_val:.2f}")
    axes[0].set_ylabel("confidence")
    axes[0].legend(loc="lower right")
    fig.suptitle("ECE and AUROC are orthogonal (LEARNING.md C4)")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "ece_auroc_orthogonal.png", dpi=150)
    return fig


def _binned_means(
    x: np.ndarray, y: np.ndarray, n_bins: int, strategy: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean x, mean y, weight), binning on x with get_bin_edges() -
    the same bins ece() uses, so a figure and its metric always agree.
    """
    edges = get_bin_edges(x, n_bins, strategy)
    bin_labels = np.digitize(x, edges)
    n = len(x)

    bin_x, bin_y, bin_weight = [], [], []
    for bin_label in np.unique(bin_labels):
        mask = bin_labels == bin_label
        bin_x.append(x[mask].mean())
        bin_y.append(y[mask].mean())
        bin_weight.append(mask.sum() / n)
    return np.array(bin_x), np.array(bin_y), np.array(bin_weight)


def _reliability_points(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int, strategy: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean confidence, accuracy, weight) for a reliability diagram."""
    return _binned_means(confidences, correct, n_bins, strategy)


def _marker_sizes(weights: np.ndarray) -> np.ndarray:
    """Bin weight in [0, 1] -> marker area bounded to [40, 350] points².
    Unbounded, a bin holding most of the data gets a marker that spills
    past the axes (marker size is in screen space); legends use fixed-size
    proxy handles instead.
    """
    return 40 + weights * 310


def plot_reliability_diagram(
    confidences: np.ndarray,
    correct: np.ndarray,
    signal_name: str,
    n_bins: int,
    model_slug: str,
    strategy: str = "auto",
    correct_bidir: np.ndarray | None = None,
) -> Figure:
    """Mean confidence vs accuracy per bin, for one signal. Points below the
    diagonal are overconfident. Marker size is the bin's share of the data.

    With `correct_bidir`, the verdict_bidir definition (D7) is overlaid as a
    second curve - the confidences don't change, only what each bin is
    scored against.

    Saved to results/figures/reliability_{signal_name}_{model_slug}.png.
    """
    confidences_arr = np.asarray(confidences, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    legend_handles = [Line2D([0], [0], linestyle="--", color="gray", label="perfect calibration")]

    conf_pts, acc_pts, weights = _reliability_points(confidences_arr, correct_arr, n_bins, strategy)
    ax.plot(conf_pts, acc_pts, color="tab:blue", alpha=0.5, zorder=1)
    ax.scatter(
        conf_pts,
        acc_pts,
        s=_marker_sizes(weights),
        alpha=0.8,
        color="tab:blue",
        edgecolors="white",
        linewidths=1,
        zorder=2,
    )
    legend_handles.append(
        Line2D([0], [0], marker="o", linestyle="", color="tab:blue", markersize=10, label="judge_verdict")
    )

    if correct_bidir is not None:
        correct_bidir_arr = np.asarray(correct_bidir, dtype=float)
        conf_pts2, acc_pts2, weights2 = _reliability_points(confidences_arr, correct_bidir_arr, n_bins, strategy)
        ax.plot(conf_pts2, acc_pts2, color="tab:orange", alpha=0.5, zorder=1)
        ax.scatter(
            conf_pts2,
            acc_pts2,
            s=_marker_sizes(weights2),
            alpha=0.8,
            color="tab:orange",
            edgecolors="white",
            linewidths=1,
            zorder=2,
        )
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="", color="tab:orange", markersize=10, label="verdict_bidir")
        )

    # A margin beyond [0, 1] so markers at the edge (common near 1.0) aren't clipped.
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_title(f"Reliability diagram: {signal_name}")
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"reliability_{signal_name}_{model_slug}.png", dpi=150)
    return fig


def plot_risk_coverage(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    oracle: tuple[np.ndarray, np.ndarray],
    filename: str = "risk_coverage.png",
    title: str = "Risk-coverage: RQ2",
) -> Figure:
    """Every signal's risk-coverage curve plus the oracle, on one axis -
    RQ2's thesis figure, and RQ5's entropy threshold sweep via
    `filename`/`title`. A flat curve is a finding; the oracle shows how much
    headroom it leaves.

    Args:
        curves: {name: (coverage, risk)}.
        oracle: (coverage, risk) from oracle_risk_coverage().
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Curves can coincide (RQ5's total and aleatoric entropy almost do), so
    # linestyle and marker vary too - color alone can't separate lines drawn
    # on top of each other.
    _LINESTYLES = ["-", "--", "-.", ":"]
    _MARKERS = ["o", "s", "^", "D"]

    for i, (name, (coverage, risk)) in enumerate(curves.items()):
        ax.plot(
            coverage, risk,
            linestyle=_LINESTYLES[i % len(_LINESTYLES)],
            marker=_MARKERS[i % len(_MARKERS)],
            markersize=3,
            markevery=0.05,
            alpha=0.85,
            label=name,
        )

    oracle_coverage, oracle_risk = oracle
    ax.plot(
        oracle_coverage, oracle_risk,
        linestyle="--", color="black", linewidth=1.5, label="oracle",
    )

    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (1 - accuracy)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    return fig


def _ols_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(slope, intercept) for the trend-line overlay. A small copy of
    analysis/human_disagreement.py's formula: src/ must not import analysis/.
    """
    x_mean, y_mean = x.mean(), y.mean()
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2))
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation for the figure's annotation (average ranks for
    ties) - a copy of analysis/human_disagreement.py's, for the same reason.
    """
    x_ranks = pd.Series(x).rank().to_numpy()
    y_ranks = pd.Series(y).rank().to_numpy()
    x_centered = x_ranks - x_ranks.mean()
    y_centered = y_ranks - y_ranks.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    return float(np.sum(x_centered * y_centered) / denom)


def plot_human_disagreement(
    d_human: np.ndarray,
    correct: np.ndarray,
    confidence: np.ndarray,
    n_bins: int,
    model_slug: str,
) -> Figure:
    """Task 3.3 (D9): accuracy (left) and conf_verb (right) against human
    consensus strength d_human. Binned means (marker size = weight) plus the
    OLS trend line in a different style, with Spearman in the title.
    d_human has few distinct values, so the "auto" binning bins it exactly.

    Saved to results/figures/human_disagreement_{model_slug}.png.
    """
    # Rounded to 6 dp: the same true fraction can land on adjacent float64
    # values (|1/3 - 0.5| vs |2/3 - 0.5|), which exact-value binning would
    # split into two bins.
    d_human_arr = np.round(np.asarray(d_human, dtype=float), 6)
    correct_arr = np.asarray(correct, dtype=float)
    confidence_arr = np.asarray(confidence, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    panels = [
        (axes[0], correct_arr, "accuracy", "tab:blue", "o"),
        (axes[1], confidence_arr, "mean conf_verb", "tab:orange", "s"),
    ]
    for ax, y_arr, ylabel, color, marker in panels:
        bin_d, bin_y, weights = _binned_means(d_human_arr, y_arr, n_bins, strategy="auto")
        ax.scatter(
            bin_d, bin_y,
            s=_marker_sizes(weights),
            marker=marker,
            color=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=1,
            zorder=2,
        )

        slope, intercept = _ols_fit(d_human_arr, y_arr)
        line_x = np.array([d_human_arr.min(), d_human_arr.max()])
        ax.plot(
            line_x, slope * line_x + intercept,
            linestyle="--", color="black", linewidth=1.5, zorder=1,
        )

        # Spearman is a rank summary, not a curve, so it goes in the title.
        rho = _spearman_corr(d_human_arr, y_arr)

        legend_handles = [
            Line2D([0], [0], marker=marker, linestyle="", color=color, markersize=8, label="binned mean"),
            Line2D([0], [0], linestyle="--", color="black", label="OLS fit"),
        ]

        x_pad = 0.05 * (d_human_arr.max() - d_human_arr.min())
        y_pad = 0.05 * max(y_arr.max() - y_arr.min(), 1e-6)
        ax.set_xlim(d_human_arr.min() - x_pad, d_human_arr.max() + x_pad)
        ax.set_ylim(y_arr.min() - y_pad, y_arr.max() + y_pad)
        ax.set_xlabel("d_human")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. d_human\nOLS slope={slope:.3f}, Spearman ρ={rho:.3f}", fontsize=10)
        ax.legend(handles=legend_handles, loc="best", fontsize=9)

    fig.suptitle("Human disagreement (D9)")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"human_disagreement_{model_slug}.png", dpi=150)
    return fig


def _draw_forest(
    ax: plt.Axes,
    labels: list[str],
    values: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    xlabel: str,
    colors: list[str] | None = None,
    annotate: bool = True,
) -> None:
    """Draws a forest panel onto `ax`: a point + CI bar per label (first
    label at the top) and a reference line at 0. No title, no save - the
    shared primitive under every forest figure.

    Args:
        colors: optional per-label colors (default: all tab:blue).
        annotate: print "value [lo, hi]" above each point. Turn off for
            dense panels whose exact values live in a CSV.
    """
    labels = list(labels)
    values_arr = np.asarray(values, dtype=float)
    ci_low_arr = np.asarray(ci_low, dtype=float)
    ci_high_arr = np.asarray(ci_high, dtype=float)
    point_colors = list(colors) if colors is not None else ["tab:blue"] * len(labels)

    y_pos = np.arange(len(labels))
    # errorbar takes distances from the point, not absolute CI bounds.
    err_low = values_arr - ci_low_arr
    err_high = ci_high_arr - values_arr

    ax.axvline(0, linestyle="--", color="gray", linewidth=1, zorder=1)
    # One errorbar call per point, since one call takes only one color.
    for x, y, lo_err, hi_err, color in zip(values_arr, y_pos, err_low, err_high, point_colors):
        ax.errorbar(
            [x], [y],
            xerr=[[lo_err], [hi_err]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=7,
            zorder=2,
        )

    # Exact values as text: when one estimate dwarfs the others, small but
    # real ones shrink to a dot with an invisible bar.
    if annotate:
        for x, y, lo, hi in zip(values_arr, y_pos, ci_low_arr, ci_high_arr):
            ax.annotate(
                f"{x:.3f} [{lo:.3f}, {hi:.3f}]",
                xy=(x, y), xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8, color="dimgray",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    # Annotated panels need extra headroom for the label above the top point.
    pad = 0.75 if annotate else 0.5
    ax.set_ylim(len(labels) - 0.5, -pad)
    ax.set_xlabel(xlabel)


def _forest_plot(
    labels: list[str],
    values: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    xlabel: str,
    title: str,
    filename: str,
) -> Figure:
    """A single-panel forest plot (several estimates with CIs against 0),
    saved to results/figures/{filename}. Height scales with the label count.
    """
    fig, ax = plt.subplots(figsize=(6, 0.9 * len(labels) + 1.5))
    _draw_forest(ax, labels, values, ci_low, ci_high, xlabel)
    ax.set_title(title)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    return fig


def plot_d_human_correlations(
    signals: list[str],
    spearman: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    model_slug: str,
    filename_suffix: str = "",
    title: str = "Signal-vs-d_human correlations (task 3.4)",
) -> Figure:
    """Forest plot of signals' Spearman correlation with d_human (task 3.4;
    reused by 5.9e). A second caller with a different signal set must pass
    a `filename_suffix`, or it overwrites the first caller's figure.

    Saved to results/figures/d_human_correlations{filename_suffix}_{model_slug}.png.
    """
    return _forest_plot(
        labels=signals,
        values=spearman,
        ci_low=ci_low,
        ci_high=ci_high,
        xlabel="Spearman ρ (signal vs. d_human)",
        title=title,
        filename=f"d_human_correlations{filename_suffix}_{model_slug}.png",
    )


def plot_rq3a_confidence_gap(
    signals: list[str],
    gap: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    model_slug: str,
) -> Figure:
    """RQ3a: forest plot of each signal's flipped-minus-unflipped mean
    confidence. A negative gap with a CI excluding 0 means the signal drops
    exactly where order changed the verdict.

    Saved to results/figures/rq3a_confidence_gap_{model_slug}.png.
    """
    return _forest_plot(
        labels=signals,
        values=gap,
        ci_low=ci_low,
        ci_high=ci_high,
        xlabel="mean confidence: flipped - unflipped",
        title="Confidence gap on flipped vs. unflipped items (task 4.3, RQ3a)",
        filename=f"rq3a_confidence_gap_{model_slug}.png",
    )


def plot_rq3b_deltas(
    signals: list[str],
    delta_ece: np.ndarray,
    ece_ci_low: np.ndarray,
    ece_ci_high: np.ndarray,
    delta_auroc: np.ndarray,
    auroc_ci_low: np.ndarray,
    auroc_ci_high: np.ndarray,
    model_slug: str,
) -> Figure:
    """RQ3b: ΔECE and ΔAUROC (verbose - clean) side by side, sharing one
    signal order - the finding (AUROC drops everywhere, calibration breaks
    only for conf_bpe) only reads as one result on one figure. Positive
    ΔECE = worse calibration; negative ΔAUROC = less informative signal.

    Saved to results/figures/rq3b_deltas_{model_slug}.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.9 * len(signals) + 1.5))

    _draw_forest(axes[0], signals, delta_ece, ece_ci_low, ece_ci_high, xlabel="Δ ECE (verbose - clean)")
    axes[0].set_title("Calibration")

    _draw_forest(axes[1], signals, delta_auroc, auroc_ci_low, auroc_ci_high, xlabel="Δ AUROC (verbose - clean)")
    axes[1].set_title("Error-detection")
    axes[1].set_ylabel("")  # left panel's labels already identify the rows

    fig.suptitle("Verbosity's effect on calibration and error-detection (task 4.4, RQ3b)")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"rq3b_deltas_{model_slug}.png", dpi=150)
    return fig


def plot_rq4_ablation(
    tiers: list[str],
    models: list[str],
    auroc_mean: np.ndarray,
    auroc_low: np.ndarray,
    auroc_high: np.ndarray,
    baseline: float,
    baseline_ci_low: float,
    baseline_ci_high: float,
    baseline_label: str,
    model_slug: str,
) -> Figure:
    """Task 5.5: one bar per (tier, model), grouped by tier and colored by
    model, next to its own bar for the best single signal (with a thin
    reference line at its height). Whiskers are D8's across-repeat spread
    for the tiers and a cluster-bootstrap CI for the baseline.

    Saved to results/figures/rq4_ablation_{model_slug}.png.
    """
    tier_order = ["A", "B", "C"]
    model_order = sorted(set(models))
    bar_width = 0.8 / max(len(model_order), 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(baseline, linestyle="--", color="tab:gray", linewidth=1, zorder=1)

    # The baseline bar at x=0, in its own color so it never reads as a model.
    baseline_err_low = baseline - baseline_ci_low
    baseline_err_high = baseline_ci_high - baseline
    ax.bar(
        [0], [baseline], width=0.7, color="tab:green", zorder=2,
        label=f"best single signal ({baseline_label})",
    )
    ax.errorbar(
        [0], [baseline], yerr=[[baseline_err_low], [baseline_err_high]],
        fmt="none", ecolor="black", capsize=4, zorder=3,
    )

    for i, model in enumerate(model_order):
        model_mask = [m == model for m in models]
        model_tiers = [t for t, keep in zip(tiers, model_mask) if keep]
        model_means = [v for v, keep in zip(auroc_mean, model_mask) if keep]
        model_low = [v for v, keep in zip(auroc_low, model_mask) if keep]
        model_high = [v for v, keep in zip(auroc_high, model_mask) if keep]

        # Reorder into tier_order; the caller's rows needn't be sorted.
        by_tier = dict(zip(model_tiers, zip(model_means, model_low, model_high)))
        ordered = [by_tier[t] for t in tier_order if t in by_tier]
        means = np.array([v[0] for v in ordered])
        err_low = means - np.array([v[1] for v in ordered])
        err_high = np.array([v[2] for v in ordered]) - means

        # +1 shifts the tier groups right of the baseline bar at x=0.
        x = 1 + np.arange(len(ordered)) + (i - (len(model_order) - 1) / 2) * bar_width
        ax.bar(x, means, width=bar_width, label=model, zorder=2)
        ax.errorbar(
            x, means, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=4, zorder=3
        )

    ax.set_xticks([0] + list(1 + np.arange(len(tier_order))))
    ax.set_xticklabels([baseline_label] + [f"Tier {t}" for t in tier_order])
    ax.set_ylabel("AUROC (uncertainty/features → judge error)")
    ax.set_title("RQ4 tier ablation: A → B → C (task 5.5)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"rq4_ablation_{model_slug}.png", dpi=150)
    return fig


def plot_rq4_progression(
    comparisons: list[str],
    auroc_diff: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    model_slug: str,
) -> Figure:
    """Task 5.5: forest plot of each tier step's paired AUROC change. A CI
    excluding 0 means that step is real.

    Saved to results/figures/rq4_progression_{model_slug}.png.
    """
    return _forest_plot(
        labels=comparisons,
        values=auroc_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        xlabel="Δ AUROC (higher tier - lower tier)",
        title="RQ4 tier-progression paired comparison (task 5.5)",
        filename=f"rq4_progression_{model_slug}.png",
    )


def plot_rq4_permutation_nulls(
    tiers: list[str],
    models: list[str],
    observed: np.ndarray,
    null_distributions: list[np.ndarray],
    model_slug: str,
) -> Figure:
    """Task 5.5: each (tier, model) cell's null AUROC histogram with the
    observed AUROC marked - a visual check that the observed value sits well clear of its null.
    Grid: one row per model, one column per tier, whatever the input order.

    Saved to results/figures/rq4_permutation_nulls_{model_slug}.png.
    """
    tier_order = [t for t in ["A", "B", "C"] if t in tiers]
    model_order = sorted(set(models))
    nrows, ncols = len(model_order), len(tier_order)

    by_cell = {(t, m): i for i, (t, m) in enumerate(zip(tiers, models))}

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for row, model in enumerate(model_order):
        for col, tier in enumerate(tier_order):
            ax = axes[row][col]
            i = by_cell[(tier, model)]

            ax.hist(null_distributions[i], bins=15, color="tab:gray", alpha=0.8, edgecolor="white")
            ax.axvline(observed[i], color="tab:red", linestyle="--", linewidth=2)
            ax.set_title(f"Tier {tier}, {model}", fontsize=10)
            ax.set_xlabel("null AUROC")
            ax.set_xlim(0, 1)

    legend_handles = [
        Line2D([0], [0], color="tab:red", linestyle="--", linewidth=2, label="observed AUROC"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9)
    fig.suptitle("Permutation null distributions vs. observed AUROC (task 5.5)")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"rq4_permutation_nulls_{model_slug}.png", dpi=150)
    return fig


def plot_h4_interaction(
    oof_score: np.ndarray,
    oof_score_grid: np.ndarray,
    d_human_values: list[float],
    predicted_curves: list[np.ndarray],
    log_odds_curves: list[np.ndarray],
    model_slug: str,
) -> Figure:
    """Task 5.6: predicted P(correct) vs the predictor's OOF score, one curve
    per d_human level. Left: probability scale, with a rug of the real
    scores. Right: log-odds, where the model is linear and the interaction
    is simply the slope difference - on the probability scale, sigmoid
    saturation hides it where most real scores sit.

    Saved to results/figures/h4_interaction_{model_slug}.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(d_human_values) - 1, 1)) for i in range(len(d_human_values))]

    ax = axes[0]
    for color, d_human_value, curve in zip(colors, d_human_values, predicted_curves):
        ax.plot(oof_score_grid, curve, color=color, linewidth=2.5, label=f"d_human = {d_human_value:.3f}")
    # Rug below the [0, 1] axis so it never overlaps the curves.
    ax.plot(
        oof_score, np.full_like(oof_score, -0.04), marker="|", linestyle="", color="black", alpha=0.3,
        markersize=8, clip_on=False,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, 1.02)
    ax.set_xlabel("predictor's out-of-fold score (P(correct))")
    ax.set_ylabel("predicted P(correct)")
    ax.set_title("Probability scale")
    ax.legend(loc="upper left", fontsize=9, title="human consensus (d_human)")

    ax = axes[1]
    for color, d_human_value, curve in zip(colors, d_human_values, log_odds_curves):
        ax.plot(oof_score_grid, curve, color=color, linewidth=2.5, label=f"d_human = {d_human_value:.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("predictor's out-of-fold score (P(correct))")
    ax.set_ylabel("log-odds of correct (linear predictor)")
    ax.set_title("Log-odds scale (undistorted slope)")

    fig.suptitle("H4: the predictor's edge grows with human consensus (task 5.6)")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"h4_interaction_{model_slug}.png", dpi=150)
    return fig


def plot_rq4_transfer(
    models: list[str],
    baseline_mean: np.ndarray,
    baseline_low: np.ndarray,
    baseline_high: np.ndarray,
    transfer_mean: np.ndarray,
    transfer_low: np.ndarray,
    transfer_high: np.ndarray,
    model_slug: str,
) -> Figure:
    """Task 5.7: per model, in-domain AUROC (clean CV, D8 spread) next to
    transfer AUROC (fit on clean, evaluated on verbose, bootstrap CI).

    Saved to results/figures/rq4_transfer_{model_slug}.png.
    """
    n = len(models)
    x = np.arange(n)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))

    baseline_mean_arr = np.asarray(baseline_mean, dtype=float)
    transfer_mean_arr = np.asarray(transfer_mean, dtype=float)

    ax.bar(
        x - bar_width / 2, baseline_mean_arr, width=bar_width, label="in-domain (clean)", color="tab:blue", zorder=2
    )
    ax.errorbar(
        x - bar_width / 2, baseline_mean_arr,
        yerr=[baseline_mean_arr - np.asarray(baseline_low), np.asarray(baseline_high) - baseline_mean_arr],
        fmt="none", ecolor="black", capsize=4, zorder=3,
    )

    ax.bar(
        x + bar_width / 2, transfer_mean_arr, width=bar_width, label="transfer (clean → verbose)",
        color="tab:red", zorder=2,
    )
    ax.errorbar(
        x + bar_width / 2, transfer_mean_arr,
        yerr=[transfer_mean_arr - np.asarray(transfer_low), np.asarray(transfer_high) - transfer_mean_arr],
        fmt="none", ecolor="black", capsize=4, zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("AUROC (uncertainty/features → judge error)")
    ax.set_title("RQ4 transfer test: clean → verbose (task 5.7)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"rq4_transfer_{model_slug}.png", dpi=150)
    return fig


def plot_rq4_category_transfer(
    categories: list[str],
    models: list[str],
    auroc: np.ndarray,
    model_slug: str,
) -> Figure:
    """Task 5.8: held-out AUROC per category (LeaveOneGroupOut), one bar per
    model, categories sorted weakest to strongest. No category is singled
    out in advance; the dashed line marks chance.

    Saved to results/figures/rq4_category_transfer_{model_slug}.png.
    """
    df = pd.DataFrame({"category": categories, "model": models, "auroc": auroc})
    category_order = df.groupby("category")["auroc"].mean().sort_values().index.tolist()
    model_order = sorted(df["model"].unique())

    bar_width = 0.8 / max(len(model_order), 1)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, zorder=1, label="chance (0.5)")

    for i, model in enumerate(model_order):
        model_df = df[df["model"] == model].set_index("category").reindex(category_order)
        x = np.arange(len(category_order)) + (i - (len(model_order) - 1) / 2) * bar_width
        ax.bar(x, model_df["auroc"].to_numpy(), width=bar_width, label=model, zorder=2)

    ax.set_xticks(np.arange(len(category_order)))
    ax.set_xticklabels(category_order, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("held-out AUROC (LeaveOneGroupOut)")
    ax.set_title("RQ4 transfer test 2: generalization across category (task 5.8)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"rq4_category_transfer_{model_slug}.png", dpi=150)
    return fig


_FEATURE_FAMILY_COLORS = {"A": "tab:blue", "B": "tab:green", "C": "tab:purple"}


def _sort_coefficient_rows(
    features: list[str], coef: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray, family: list[str]
) -> pd.DataFrame:
    """Row order for both coefficient figures: family A, B, C, and largest
    |coef| first within each. _draw_forest() puts the first row at the top,
    so no reversal is needed.
    """
    df = pd.DataFrame(
        {"feature": features, "coef": coef, "ci_low": ci_low, "ci_high": ci_high, "family": family}
    )
    df["abs_coef"] = df["coef"].abs()
    df["family"] = pd.Categorical(df["family"], categories=["A", "B", "C"], ordered=True)
    return df.sort_values(["family", "abs_coef"], ascending=[True, False]).reset_index(drop=True)


def _family_legend_handles(families_present: list[str]) -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", linestyle="", color=_FEATURE_FAMILY_COLORS[f], label=f"Tier {f}")
        for f in ["A", "B", "C"]
        if f in families_present
    ]


def plot_rq4_coefficients(
    features: list[str],
    coef: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    family: list[str],
    model_slug: str,
) -> Figure:
    """Task 5.9's headline figure: only the coefficients whose CI excludes
    0, colored by feature family (all ~37 rows would be too tall for a slide;
    plot_rq4_coefficients_full() and the CSV have them). Takes the FULL
    table and filters here, so "significant" has one definition. With no
    significant coefficient it still renders, as an empty panel.

    Saved to results/figures/rq4_coefficients_{model_slug}.png.
    """
    df = _sort_coefficient_rows(features, coef, ci_low, ci_high, family)
    df = df[(df["ci_low"] > 0) | (df["ci_high"] < 0)].reset_index(drop=True)

    colors = [_FEATURE_FAMILY_COLORS[f] for f in df["family"]]

    fig, ax = plt.subplots(figsize=(7, 0.9 * max(len(df), 1) + 1.5))
    _draw_forest(
        ax,
        df["feature"].tolist(),
        df["coef"].to_numpy(),
        df["ci_low"].to_numpy(),
        df["ci_high"].to_numpy(),
        "coefficient (standardized scale)",
        colors=colors,
    )

    ax.legend(handles=_family_legend_handles(sorted(df["family"].unique())), loc="lower right", fontsize=9)
    ax.set_title(
        f"RQ4 meta-model: significant coefficients only ({len(df)}/{len(features)} clear 0)\n"
        f"Tier C + logreg (task 5.9, {model_slug})"
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"rq4_coefficients_{model_slug}.png", dpi=150)
    return fig


def plot_rq4_coefficients_full(
    features: list[str],
    coef: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    family: list[str],
    model_slug: str,
) -> Figure:
    """Appendix companion to plot_rq4_coefficients(): every Tier C feature,
    unannotated at a tight row pitch (~10in tall instead of ~35in); exact
    values are in the CSV.

    Saved to results/figures/rq4_coefficients_full_{model_slug}.png.
    """
    df = _sort_coefficient_rows(features, coef, ci_low, ci_high, family)
    colors = [_FEATURE_FAMILY_COLORS[f] for f in df["family"]]

    fig, ax = plt.subplots(figsize=(8, 0.22 * len(df) + 1.2))
    _draw_forest(
        ax,
        df["feature"].tolist(),
        df["coef"].to_numpy(),
        df["ci_low"].to_numpy(),
        df["ci_high"].to_numpy(),
        "coefficient (standardized scale)",
        colors=colors,
        annotate=False,
    )
    ax.tick_params(axis="y", labelsize=7)

    ax.legend(handles=_family_legend_handles(["A", "B", "C"]), loc="lower right", fontsize=9)
    ax.set_title(
        f"RQ4 meta-model coefficients, full Tier C feature set (task 5.9, {model_slug})\n"
        f"exact values: results/rq4_coefficients_{model_slug}.csv",
        fontsize=9,
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"rq4_coefficients_full_{model_slug}.png", dpi=150)
    return fig


def plot_bayesian_convergence(max_rhat: np.ndarray, flagged: np.ndarray, model_slug: str) -> Figure:
    """Max R-hat for every fold-fit against D22's 1.01 threshold, flagged
    fits in red (flagged = R-hat > 1.01, NaN, or any divergence).

    Saved to results/figures/rq4_bayesian_convergence_{model_slug}.png.
    """
    max_rhat_arr = np.asarray(max_rhat, dtype=float)
    flagged_arr = np.asarray(flagged, dtype=bool)
    x = np.arange(len(max_rhat_arr))
    point_colors = np.where(flagged_arr, "tab:red", "tab:blue")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(1.01, linestyle="--", color="gray", linewidth=1, zorder=1)
    ax.scatter(x, max_rhat_arr, c=point_colors, zorder=2)

    n_flagged = int(flagged_arr.sum())
    legend_handles = [
        Line2D([0], [0], linestyle="--", color="gray", label="D22 threshold (1.01)"),
        Line2D(
            [0], [0], marker="o", linestyle="", color="tab:blue",
            label=f"converged ({len(flagged_arr) - n_flagged})",
        ),
        Line2D([0], [0], marker="o", linestyle="", color="tab:red", label=f"flagged ({n_flagged})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    ax.set_xlabel("fold-fit (50 total: 5 folds x 10 repeats, D8)")
    ax.set_ylabel("max R-hat")
    ax.set_title(f"Bayesian model convergence across all fold-fits (task 5.9c, {model_slug})")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"rq4_bayesian_convergence_{model_slug}.png", dpi=150)
    return fig


def plot_rq5_distillation(
    auroc_ensemble: float,
    auroc_ensemble_ci: tuple[float, float],
    auroc_bayesian: float,
    epistemic_auroc_ensemble: float,
    epistemic_auroc_ensemble_ci: tuple[float, float],
    epistemic_auroc_bayesian: float,
    epistemic_auroc_bayesian_ci: tuple[float, float],
    model_slug: str,
) -> Figure:
    """Task 5.9d: the 3-call ensemble vs the 1-call Bayesian model on AUROC
    and on entropy quality (AUROC of each one's epistemic signal).

    The Bayesian AUROC bar has no whisker on purpose: every other bar has a
    cluster-bootstrap CI, and borrowing D8's across-repeat spread for it
    would imply a false equivalence between two kinds of interval.

    Saved to results/figures/rq5_distillation_{model_slug}.png.
    """
    metrics = ["AUROC", "Epistemic AUROC\n(entropy quality)"]
    x = np.arange(len(metrics))
    width = 0.35

    ensemble_vals = np.array([auroc_ensemble, epistemic_auroc_ensemble])
    ensemble_err = np.array(
        [
            [auroc_ensemble - auroc_ensemble_ci[0], epistemic_auroc_ensemble - epistemic_auroc_ensemble_ci[0]],
            [auroc_ensemble_ci[1] - auroc_ensemble, epistemic_auroc_ensemble_ci[1] - epistemic_auroc_ensemble],
        ]
    )
    bayesian_vals = np.array([auroc_bayesian, epistemic_auroc_bayesian])
    bayesian_err = np.array(
        [
            [0.0, epistemic_auroc_bayesian - epistemic_auroc_bayesian_ci[0]],
            [0.0, epistemic_auroc_bayesian_ci[1] - epistemic_auroc_bayesian],
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, zorder=1)

    ax.bar(
        x - width / 2, ensemble_vals, width, yerr=ensemble_err, capsize=4,
        label="ensemble (3-call)", color="tab:orange", zorder=2,
    )
    ax.bar(
        x + width / 2, bayesian_vals, width, yerr=bayesian_err, capsize=4,
        label="Bayesian (1-call)", color="tab:blue", zorder=2,
    )
    ax.annotate(
        "point only - see caption",
        xy=(x[0] + width / 2, auroc_bayesian), xytext=(0, 8), textcoords="offset points",
        ha="center", fontsize=7, color="dimgray",
    )

    legend_handles = [
        Line2D([0], [0], linestyle="--", color="gray", label="chance (0.5)"),
        *[
            plt.Rectangle((0, 0), 1, 1, color=c, label=lbl)
            for c, lbl in [("tab:orange", "ensemble (3-call)"), ("tab:blue", "Bayesian (1-call)")]
        ],
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC")
    ax.set_title(f"RQ5 distillation: ensemble vs single-call Bayesian (task 5.9d, {model_slug})")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"rq5_distillation_{model_slug}.png", dpi=150)
    return fig


def plot_rq5_verbose_shift(
    aleatoric_clean: float,
    aleatoric_verbose: float,
    aleatoric_gap_ci: tuple[float, float],
    epistemic_clean: float,
    epistemic_verbose: float,
    epistemic_gap_ci: tuple[float, float],
    model_slug: str,
) -> Figure:
    """Task 5.9f: mean aleatoric and epistemic entropy on clean vs verbose,
    from the one Bayesian model fit on clean. Two panels with separate
    y-axes - epistemic is ~100x smaller than aleatoric and would look like
    zero on a shared axis. Each panel is annotated with its paired-bootstrap
    gap CI, the statistic the preregistered verdict rests on.

    Saved to results/figures/rq5_verbose_shift_{model_slug}.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    panels = [
        (axes[0], "Aleatoric", aleatoric_clean, aleatoric_verbose, aleatoric_gap_ci),
        (axes[1], "Epistemic", epistemic_clean, epistemic_verbose, epistemic_gap_ci),
    ]
    for ax, label, clean_val, verbose_val, (gap_lo, gap_hi) in panels:
        x = np.arange(1)
        width = 0.35
        ax.bar(x - width / 2, [clean_val], width, label="clean", color="tab:blue", zorder=2)
        ax.bar(x + width / 2, [verbose_val], width, label="verbose", color="tab:red", zorder=2)
        ax.annotate(
            f"gap [{gap_lo:.4f}, {gap_hi:.4f}]",
            xy=(0, max(clean_val, verbose_val)), xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=8, color="dimgray",
        )
        ax.set_xticks([])
        ax.set_ylim(0, max(clean_val, verbose_val) * 1.3)
        ax.set_ylabel("mean entropy (nats)")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"RQ5 verbose-shift validation: clean vs verbose (task 5.9f, {model_slug})")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"rq5_verbose_shift_{model_slug}.png", dpi=150)
    return fig


if __name__ == "__main__":
    plot_ece_auroc_orthogonal()
