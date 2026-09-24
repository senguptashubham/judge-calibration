"""Matplotlib figures, one per function, each saved to results/figures/ under
a deterministic, model_slug-suffixed name (CLAUDE.md §5, D26).

Figures use plain-language names for signals and judges (SIGNAL_LABELS,
JUDGE_NAMES) so they read without the code; the CSVs keep the column names.
Intervals are always labelled with their kind: a cluster-bootstrap CI and
D8's across-repeat spread are different quantities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from src.metrics import aurc, auroc_error, ece, get_bin_edges

FIGURES_DIR = Path("results/figures")

SIGNAL_LABELS = {
    "conf_verb": "stated confidence",
    "conf_lp": "verdict-token probability",
    "conf_sc": "self-consistency",
    "conf_bpe": "order-swap agreement",
    "conf_bpe_prob": "order-swap agreement",
    "conf_verb_bidir": "stated confidence, both orders",
    "conf_lp_bidir": "verdict probability, both orders",
    "conf_ens": "3-prompt ensemble",
    "ens_entropy_total": "ensemble entropy: total",
    "ens_entropy_aleatoric": "ensemble entropy: aleatoric",
    "ens_entropy_epistemic": "ensemble entropy: epistemic",
    "conf_kev": "class probability",
    "conf_kev_bpe": "order-swap agreement",
    "conf_sc_autoj": "self-consistency",
    "conf_sc_bpe_autoj": "order-swap agreement",
    "conf_sc_bpe_autoj_greedy": "order-swap agreement (greedy calls)",
}

# Signals built from the same AB/BA pair that defines `flipped`: their
# flipped-vs-unflipped confidence gap is large by construction.
BY_CONSTRUCTION = {"conf_bpe", "conf_kev_bpe", "conf_sc_bpe_autoj"}

JUDGE_NAMES = {
    "qwen2.5_7b_instruct": "Qwen2.5-7B",
    "kev_8b": "kev-8b",
    "autoj_13b_gptq_4bits": "auto-j-13b",
}
JUDGE_COLORS = {"Qwen2.5-7B": "tab:blue", "kev-8b": "tab:orange", "auto-j-13b": "tab:green"}

_MUTED = "silver"


def signal_label(name: str) -> str:
    return SIGNAL_LABELS.get(name, name)


def judge_name(model_slug: str) -> str:
    """Display name for a model_slug, including suffixed ones such as
    'kev_8b_in_coverage'."""
    for slug, name in JUDGE_NAMES.items():
        if model_slug.startswith(slug):
            return name
    return model_slug


def _save(fig: Figure, filename: str) -> Figure:
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    return fig


def _significance_colors(ci_low: np.ndarray, ci_high: np.ndarray, color: str = "tab:blue") -> list[str]:
    """`color` where the CI excludes 0, muted where it doesn't."""
    return [color if (lo > 0 or hi < 0) else _MUTED for lo, hi in zip(ci_low, ci_high)]


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
        (axes[0], confidences_a, correct_a, ece_a, auroc_a, "A: ranks perfectly, miscalibrated"),
        (axes[1], confidences_b, correct_b, ece_b, auroc_b, "B: calibrated, ranks nothing"),
    ]:
        correct_bool = np.asarray(correct, dtype=bool)
        x = np.arange(len(confidences))
        ax.scatter(x[correct_bool], confidences[correct_bool], c="tab:green", label="correct")
        ax.scatter(x[~correct_bool], confidences[~correct_bool], c="tab:red", label="incorrect")
        ax.set_ylim(0.4, 1.05)
        ax.set_xlabel("item")
        ax.set_title(f"{label}\nECE={ece_val:.2f}, AUROC={auroc_val:.2f}")
    axes[0].set_ylabel("confidence")
    axes[0].legend(loc="lower right")
    fig.suptitle("ECE and AUROC measure different things")
    return _save(fig, "ece_auroc_orthogonal.png")


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
    label: str = "judge_verdict",
    overlay: tuple[np.ndarray, np.ndarray, str] | None = None,
    title: str | None = None,
    xlabel: str = "confidence",
    ylabel: str = "accuracy",
    note: str = "below the diagonal = overconfident",
) -> Figure:
    """Mean confidence vs accuracy per bin, on the same bins ece() uses,
    with each curve's ECE and effective bin count in the legend
    (invariant 4) and a histogram of the confidences underneath - most
    signals pile up near 1, which the bins alone hide. Points below the
    diagonal are overconfident. Marker size is the bin's share of the data.

    `overlay` = (confidences, correct, label) draws a second curve, e.g. the
    verdict_bidir definition (D7). It carries its own confidences: a
    confidence is only meaningful against the verdict it belongs to.

    The axes start at the lowest confidence or accuracy shown, never above
    0.5, so the region where the points actually sit fills the panel.

    Saved to results/figures/reliability_{signal_name}_{model_slug}.png.
    """
    curves = [(np.asarray(confidences, dtype=float), np.asarray(correct, dtype=float), label, "tab:blue")]
    if overlay is not None:
        overlay_conf, overlay_correct, overlay_label = overlay
        curves.append(
            (np.asarray(overlay_conf, dtype=float), np.asarray(overlay_correct, dtype=float), overlay_label, "tab:orange")
        )

    fig, (ax, hist_ax) = plt.subplots(
        2, 1, figsize=(5.5, 6.5), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )

    lowest = 0.5
    legend_handles = []
    for conf, corr, curve_label, color in curves:
        conf_pts, acc_pts, weights = _binned_means(conf, corr, n_bins, strategy)
        ece_value, n_effective_bins = ece(conf, corr, n_bins, strategy)
        lowest = min(lowest, conf_pts.min(), acc_pts.min())

        ax.plot(conf_pts, acc_pts, color=color, alpha=0.5, zorder=1)
        ax.scatter(
            conf_pts, acc_pts, s=_marker_sizes(weights), alpha=0.8, color=color,
            edgecolors="white", linewidths=1, zorder=2,
        )
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="", color=color, markersize=9,
                label=f"{curve_label}: ECE {ece_value:.3f} ({n_effective_bins} bins)",
            )
        )
        hist_ax.hist(conf, bins=40, range=(0, 1), color=color, alpha=0.5)

    lo = np.floor(lowest * 10) / 10 - 0.02
    ax.plot([lo, 1], [lo, 1], linestyle="--", color="gray", zorder=0)
    legend_handles.insert(0, Line2D([0], [0], linestyle="--", color="gray", label="perfect calibration"))

    ax.set_xlim(lo, 1.02)
    ax.set_ylim(lo, 1.02)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{signal_label(signal_name)} ({judge_name(model_slug)})")
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    ax.text(0.98, 0.02, note, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="dimgray")

    hist_ax.set_xlabel(xlabel)
    hist_ax.set_ylabel("items")
    return _save(fig, f"reliability_{signal_name}_{model_slug}.png")


def plot_risk_coverage(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    oracle: tuple[np.ndarray, np.ndarray],
    filename: str = "risk_coverage.png",
    title: str = "Abstaining on the least confident items",
) -> Figure:
    """Every signal's risk-coverage curve plus the oracle, on one axis -
    RQ2's thesis figure, and RQ5's entropy threshold sweep via
    `filename`/`title`. Each signal's AURC is in the legend. The dotted
    line is the error rate with no abstention, which random abstention
    also keeps at every coverage; the oracle shows the headroom.

    Below 5% coverage a point rests on fewer than ~90 items, so that strip
    is shaded as noisy. A discrete signal's curve stops early: tied items
    can't be split, so it has no point below its largest tie group.

    Args:
        curves: {name: (coverage, risk)}.
        oracle: (coverage, risk) from oracle_risk_coverage().
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Curves can coincide (RQ5's total and aleatoric entropy almost do), so
    # linestyle and marker vary too - color alone can't separate lines drawn
    # on top of each other.
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]

    base_risk = None
    for i, (name, (coverage, risk)) in enumerate(curves.items()):
        coverage, risk = np.asarray(coverage, dtype=float), np.asarray(risk, dtype=float)
        base_risk = float(risk[np.argmax(coverage)])
        ax.plot(
            coverage, risk,
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            markersize=3,
            markevery=0.05,
            alpha=0.85,
            label=f"{signal_label(name)} (AURC {aurc(coverage, risk):.3f})",
        )

    oracle_coverage, oracle_risk = oracle
    ax.plot(
        oracle_coverage, oracle_risk, linestyle="--", color="black", linewidth=1.5,
        label=f"oracle (AURC {aurc(oracle_coverage, oracle_risk):.3f})",
    )
    if base_risk is not None:
        ax.axhline(base_risk, linestyle=":", color="gray", linewidth=1.2,
                   label=f"no abstention / random ({base_risk:.3f})")

    ax.axvspan(0, 0.05, color="0.93", zorder=0)
    ax.text(0.025, 0.98, "noisy", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=7, color="dimgray", rotation=90)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("coverage (share of items the judge keeps)")
    ax.set_ylabel("risk (error rate on kept items)")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(0.06, 1), fontsize=8)
    return _save(fig, filename)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation for the figure's annotation (average ranks for
    ties) - a copy of analysis/human_disagreement.py's: src/ must not import
    analysis/.
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
    """Task 3.3 (D9): accuracy (left) and stated confidence (right) at each
    level of human consensus d_human, labelled with how many items each
    level holds. No fitted line: d_human takes only three values here, one
    of them with a handful of items, so a line would overstate the shape.
    The Spearman correlation (the test REPORT.md uses) is in each title.

    Saved to results/figures/human_disagreement_{model_slug}.png.
    """
    # Rounded to 6 dp: the same true fraction can land on adjacent float64
    # values (|1/3 - 0.5| vs |2/3 - 0.5|), which exact-value binning would
    # split into two bins.
    d_human_arr = np.round(np.asarray(d_human, dtype=float), 6)
    correct_arr = np.asarray(correct, dtype=float)
    confidence_arr = np.asarray(confidence, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    panels = [
        (axes[0], correct_arr, "accuracy", "tab:blue", "o"),
        (axes[1], confidence_arr, "mean stated confidence", "tab:orange", "s"),
    ]
    for ax, y_arr, ylabel, color, marker in panels:
        bin_d, bin_y, weights = _binned_means(d_human_arr, y_arr, n_bins, strategy="auto")
        ax.scatter(bin_d, bin_y, s=_marker_sizes(weights), marker=marker, color=color,
                   alpha=0.85, edgecolors="white", linewidths=1, zorder=2)
        for x, y, w in zip(bin_d, bin_y, weights):
            ax.annotate(f"n={round(w * len(d_human_arr))}", xy=(x, y), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=8, color="dimgray")

        rho = _spearman_corr(d_human_arr, y_arr)
        x_pad = 0.08 * (d_human_arr.max() - d_human_arr.min())
        y_pad = 0.15 * max(bin_y.max() - bin_y.min(), 1e-6)
        ax.set_xlim(d_human_arr.min() - x_pad, d_human_arr.max() + x_pad)
        ax.set_ylim(bin_y.min() - y_pad, bin_y.max() + y_pad)
        ax.set_xlabel("human consensus d_human (0.5 = unanimous)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel}\nSpearman ρ = {rho:.3f} over all items", fontsize=10)

    fig.suptitle(f"Judge behaviour vs. human consensus ({judge_name(model_slug)})")
    return _save(fig, f"human_disagreement_{model_slug}.png")


def _draw_forest(
    ax: plt.Axes,
    labels: list[str],
    values: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    xlabel: str,
    colors: list[str] | None = None,
    annotate: bool = True,
    reference: float | None = 0.0,
    decimals: int = 3,
) -> None:
    """Draws a forest panel onto `ax`: a point + interval bar per label
    (first label at the top) and a dashed reference line (default 0). No
    title, no save - the shared primitive under every forest figure.

    Args:
        colors: optional per-label colors (default: all tab:blue).
        annotate: print "value [lo, hi]" above each point. Turn off for
            dense panels whose exact values live in a CSV.
        reference: x of the dashed line, or None for no line.
        decimals: decimal places in the annotations.
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

    if reference is not None:
        ax.axvline(reference, linestyle="--", color="gray", linewidth=1, zorder=1)
    # One errorbar call per point, since one call takes only one color.
    for x, y, lo_err, hi_err, color in zip(values_arr, y_pos, err_low, err_high, point_colors):
        ax.errorbar([x], [y], xerr=[[lo_err], [hi_err]], fmt="o", color=color, ecolor=color,
                    capsize=4, markersize=7, zorder=2)

    # Exact values as text: when one estimate dwarfs the others, small but
    # real ones shrink to a dot with an invisible bar.
    if annotate:
        for x, y, lo, hi in zip(values_arr, y_pos, ci_low_arr, ci_high_arr):
            text = f"{x:.{decimals}f}" if np.isnan(lo) else f"{x:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"
            ax.annotate(text, xy=(x, y), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8, color="dimgray")

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
    colors: list[str] | None = None,
) -> Figure:
    """A single-panel forest plot (several estimates with CIs against 0),
    saved to results/figures/{filename}. Height scales with the label count.
    """
    fig, ax = plt.subplots(figsize=(7, 0.8 * len(labels) + 1.6))
    _draw_forest(ax, labels, values, ci_low, ci_high, xlabel, colors=colors)
    ax.set_title(title)
    return _save(fig, filename)


def plot_d_human_correlations(
    signals: list[str],
    spearman: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    model_slug: str,
    filename_suffix: str = "",
    title: str = "Does confidence track human consensus?",
) -> Figure:
    """Forest plot of signals' Spearman correlation with d_human (task 3.4;
    reused by 5.9e). A second caller with a different signal set must pass
    a `filename_suffix`, or it overwrites the first caller's figure.

    Saved to results/figures/d_human_correlations{filename_suffix}_{model_slug}.png.
    """
    return _forest_plot(
        labels=[signal_label(s) for s in signals],
        values=spearman,
        ci_low=ci_low,
        ci_high=ci_high,
        xlabel="Spearman ρ with d_human, 95% cluster-bootstrap CI",
        title=f"{title} ({judge_name(model_slug)})",
        filename=f"d_human_correlations{filename_suffix}_{model_slug}.png",
        colors=_significance_colors(ci_low, ci_high),
    )


def plot_confidence_gap(panels: list[dict], title: str, filename: str) -> Figure:
    """RQ3a/RQ6/RQ7: each signal's mean confidence on items whose verdict
    flipped between AB and BA, minus on items that didn't - one panel per
    population (e.g. per coverage regime or turn). A negative gap with a CI
    excluding 0 means the signal drops where order changed the verdict.

    Signals in BY_CONSTRUCTION are drawn muted and marked †: they are built
    from the same two orders that define a flip, so their large gap is
    expected and says nothing new.

    Each panel: {"title", "signals", "gap", "ci_low", "ci_high"}.
    Saved to results/figures/{filename}.
    """
    n_rows = max(len(p["signals"]) for p in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(max(6 * len(panels), 9), 0.8 * n_rows + 2), sharex=True,
                             squeeze=False)
    for ax, panel in zip(axes[0], panels):
        labels = [signal_label(s) + (" †" if s in BY_CONSTRUCTION else "") for s in panel["signals"]]
        colors = [_MUTED if s in BY_CONSTRUCTION else "tab:blue" for s in panel["signals"]]
        _draw_forest(ax, labels, panel["gap"], panel["ci_low"], panel["ci_high"],
                     "confidence when flipped − when stable\n(95% cluster-bootstrap CI)", colors=colors)
        ax.set_title(panel["title"])
    fig.suptitle(title)
    fig.text(0.01, 0.01, "† built from the same two orders that define a flip, so a large gap is expected by construction",
             fontsize=7, color="dimgray")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=150)
    return fig


def plot_verbosity_deltas(panels: list[dict], title: str, filename: str) -> Figure:
    """RQ3b/RQ6/RQ7: ΔECE and ΔAUROC (verbose − clean) side by side, one
    row per population. Positive ΔECE = worse calibration; negative ΔAUROC
    = the signal detects errors less well. Estimates whose paired CI
    excludes 0 are coloured; the rest are muted.

    Each panel: {"title", "signals", "delta_ece", "ece_ci_low", "ece_ci_high",
    "delta_auroc", "auroc_ci_low", "auroc_ci_high"}.
    Saved to results/figures/{filename}.
    """
    n_signals = max(len(p["signals"]) for p in panels)
    fig, axes = plt.subplots(len(panels), 2, figsize=(12, (0.8 * n_signals + 1.4) * len(panels) + 0.6), squeeze=False)
    for row, panel in zip(axes, panels):
        labels = [signal_label(s) for s in panel["signals"]]
        _draw_forest(row[0], labels, panel["delta_ece"], panel["ece_ci_low"], panel["ece_ci_high"],
                     "Δ ECE, verbose − clean (> 0 = worse calibrated)",
                     colors=_significance_colors(panel["ece_ci_low"], panel["ece_ci_high"]))
        _draw_forest(row[1], labels, panel["delta_auroc"], panel["auroc_ci_low"], panel["auroc_ci_high"],
                     "Δ AUROC, verbose − clean (< 0 = detects errors less well)",
                     colors=_significance_colors(panel["auroc_ci_low"], panel["auroc_ci_high"]))
        row[0].set_title(f"{panel['title']}: calibration" if panel["title"] else "Calibration")
        row[1].set_title(f"{panel['title']}: error detection" if panel["title"] else "Error detection")
        row[1].set_yticklabels([])
    fig.suptitle(f"{title}\n(paired 95% cluster-bootstrap CIs; muted = CI includes 0)")
    return _save(fig, filename)


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
    null_means: np.ndarray | None = None,
) -> Figure:
    """Task 5.5: AUROC of each (tier, model) meta-model next to the best
    single signal, on a zoomed axis - the differences are hundredths.
    Tier intervals are D8's across-repeat spread; the baseline's is a 95%
    cluster-bootstrap CI. `null_means`, if given, shades the range of the
    permutation-null means (invariant 12).

    Saved to results/figures/rq4_ablation_{model_slug}.png.
    """
    by_cell = {(t, m): (v, lo, hi) for t, m, v, lo, hi in zip(tiers, models, auroc_mean, auroc_low, auroc_high)}
    tier_names = {"A": "A: judge's own signals", "B": "B: + surface features", "C": "C: + token statistics"}
    model_colors = {"logreg": "tab:blue", "histgbm": "tab:purple"}

    labels, values, lows, highs, colors = [f"best single signal ({signal_label(baseline_label)})"], [baseline], \
        [baseline_ci_low], [baseline_ci_high], ["tab:green"]
    for tier in ["A", "B", "C"]:
        for model in sorted(set(models)):
            if (tier, model) in by_cell:
                v, lo, hi = by_cell[(tier, model)]
                labels.append(f"Tier {tier_names[tier]}, {model}")
                values.append(v)
                lows.append(lo)
                highs.append(hi)
                colors.append(model_colors.get(model, "tab:gray"))

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(labels) + 2))
    if null_means is not None and len(null_means):
        ax.axvspan(min(null_means), max(null_means), color="0.9", zorder=0)
        ax.text(np.mean(null_means), len(labels) - 0.6, "permutation null", ha="center", va="bottom",
                fontsize=8, color="dimgray")
    _draw_forest(ax, labels, values, lows, highs, "AUROC predicting judge error", colors=colors, reference=baseline)
    ax.set_title("Do more features help predict judge error?\n"
                 "(tiers: range over 10 CV repeats; baseline: 95% cluster-bootstrap CI)", fontsize=10)
    return _save(fig, f"rq4_ablation_{model_slug}.png")


def plot_rq4_progression(
    comparisons: list[str],
    auroc_diff: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    model_slug: str,
) -> Figure:
    """Task 5.5: forest plot of each tier step's paired AUROC change. A CI
    excluding 0 means that step is real; muted points are not.

    Saved to results/figures/rq4_progression_{model_slug}.png.
    """
    return _forest_plot(
        labels=comparisons,
        values=auroc_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        xlabel="Δ AUROC, higher tier − lower tier (paired 95% cluster-bootstrap CI)",
        title="Does each added feature tier change AUROC?",
        filename=f"rq4_progression_{model_slug}.png",
        colors=_significance_colors(ci_low, ci_high),
    )


def plot_rq4_permutation_nulls(
    tiers: list[str],
    models: list[str],
    observed: np.ndarray,
    null_distributions: list[np.ndarray],
    model_slug: str,
) -> Figure:
    """Task 5.5: for each (tier, model) cell, every permutation-null AUROC
    (grey dots, labels shuffled within each question) and the observed
    AUROC (red diamond), on one shared axis.

    Saved to results/figures/rq4_permutation_nulls_{model_slug}.png.
    """
    rng = np.random.default_rng(0)  # jitter only; no statistic depends on it
    order = sorted(range(len(tiers)), key=lambda i: (tiers[i], models[i]))

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(order) + 1.8))
    for row, i in enumerate(order):
        null = np.asarray(null_distributions[i], dtype=float)
        ax.scatter(null, row + rng.uniform(-0.15, 0.15, len(null)), s=10, color="gray", alpha=0.5, zorder=2)
        ax.scatter([observed[i]], [row], marker="D", s=60, color="tab:red", zorder=3)
    ax.axvline(0.5, linestyle="--", color="gray", linewidth=1, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"Tier {tiers[i]}, {models[i]}" for i in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("AUROC predicting judge error")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="", color="gray", label="null (labels shuffled within question)"),
        Line2D([0], [0], marker="D", linestyle="", color="tab:red", label="observed"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    ax.set_title("Observed AUROC vs. permutation null")
    return _save(fig, f"rq4_permutation_nulls_{model_slug}.png")


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
    ax.plot(oof_score, np.full_like(oof_score, -0.04), marker="|", linestyle="", color="black", alpha=0.3,
            markersize=8, clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, 1.02)
    ax.set_xlabel("meta-model's out-of-fold P(correct)")
    ax.set_ylabel("fitted P(judge correct)")
    ax.set_title("Probability scale")
    ax.legend(loc="upper left", fontsize=9, title="human consensus (0.5 = unanimous)")

    ax = axes[1]
    for color, d_human_value, curve in zip(colors, d_human_values, log_odds_curves):
        ax.plot(oof_score_grid, curve, color=color, linewidth=2.5, label=f"d_human = {d_human_value:.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("meta-model's out-of-fold P(correct)")
    ax.set_ylabel("log-odds of correct")
    ax.set_title("Log-odds scale (slopes undistorted)")

    fig.suptitle("The meta-model's edge grows with human consensus (H4)")
    return _save(fig, f"h4_interaction_{model_slug}.png")


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
    """Task 5.7: per model, in-domain AUROC (clean CV, D8 across-repeat
    spread) next to transfer AUROC (fit on clean, evaluated on verbose, 95%
    cluster-bootstrap CI), on a zoomed axis.

    Saved to results/figures/rq4_transfer_{model_slug}.png.
    """
    labels, values, lows, highs, colors = [], [], [], [], []
    for i, model in enumerate(models):
        labels += [f"{model}: in-domain (clean)", f"{model}: transfer (clean → verbose)"]
        values += [baseline_mean[i], transfer_mean[i]]
        lows += [baseline_low[i], transfer_low[i]]
        highs += [baseline_high[i], transfer_high[i]]
        colors += ["tab:blue", "tab:red"]

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(labels) + 2))
    _draw_forest(ax, labels, values, lows, highs, "AUROC predicting judge error", colors=colors, reference=None)
    ax.set_title("Does a meta-model trained on clean inputs survive the padding attack?\n"
                 "(in-domain: range over 10 CV repeats; transfer: 95% cluster-bootstrap CI)", fontsize=10)
    return _save(fig, f"rq4_transfer_{model_slug}.png")


def plot_rq4_category_transfer(
    categories: list[str],
    models: list[str],
    auroc: np.ndarray,
    model_slug: str,
) -> Figure:
    """Task 5.8: held-out AUROC per category (LeaveOneGroupOut), one marker
    per model, categories sorted weakest to strongest. No category is
    singled out in advance; the dashed line marks chance. Each value is one
    deterministic split, so there is no interval.

    Saved to results/figures/rq4_category_transfer_{model_slug}.png.
    """
    df = pd.DataFrame({"category": categories, "model": models, "auroc": auroc})
    category_order = df.groupby("category")["auroc"].mean().sort_values(ascending=False).index.tolist()
    model_styles = {"logreg": ("tab:blue", "o"), "histgbm": ("tab:purple", "s")}

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvline(0.5, linestyle="--", color="gray", linewidth=1, zorder=1)
    for model in sorted(df["model"].unique()):
        model_df = df[df["model"] == model].set_index("category").reindex(category_order)
        color, marker = model_styles.get(model, ("tab:gray", "o"))
        ax.scatter(model_df["auroc"], np.arange(len(category_order)), color=color, marker=marker, s=60,
                   label=model, zorder=2)
    ax.set_yticks(np.arange(len(category_order)))
    ax.set_yticklabels(category_order)
    ax.set_ylim(len(category_order) - 0.5, -0.5)
    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("AUROC on the held-out category (dashed = chance)")
    ax.set_title("Does the meta-model generalize to an unseen category?")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, f"rq4_category_transfer_{model_slug}.png")


_FEATURE_FAMILY_COLORS = {"A": "tab:blue", "B": "tab:green", "C": "tab:purple"}
_FEATURE_FAMILY_NAMES = {"A": "Tier A: judge's own signals", "B": "Tier B: surface features", "C": "Tier C: token statistics"}


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
        Line2D([0], [0], marker="o", linestyle="", color=_FEATURE_FAMILY_COLORS[f], label=_FEATURE_FAMILY_NAMES[f])
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

    fig, ax = plt.subplots(figsize=(7.5, 0.8 * max(len(df), 1) + 1.8))
    _draw_forest(ax, df["feature"].tolist(), df["coef"].to_numpy(), df["ci_low"].to_numpy(),
                 df["ci_high"].to_numpy(), "coefficient on standardized feature (> 0 = judge more likely right)",
                 colors=colors)
    ax.legend(handles=_family_legend_handles(sorted(df["family"].unique())), loc="lower right", fontsize=8)
    ax.set_title(f"Which features predict judge error?\n{len(df)} of {len(features)} coefficients clear zero "
                 f"(logistic regression, all tiers)", fontsize=10)
    return _save(fig, f"rq4_coefficients_{model_slug}.png")


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
    _draw_forest(ax, df["feature"].tolist(), df["coef"].to_numpy(), df["ci_low"].to_numpy(),
                 df["ci_high"].to_numpy(), "coefficient on standardized feature", colors=colors, annotate=False)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(handles=_family_legend_handles(["A", "B", "C"]), loc="lower right", fontsize=8)
    ax.set_title(f"All meta-model coefficients (exact values: rq4_coefficients_{model_slug}.csv)", fontsize=9)
    return _save(fig, f"rq4_coefficients_full_{model_slug}.png")


def plot_bayesian_convergence(
    max_rhat: np.ndarray,
    flagged: np.ndarray,
    model_slug: str,
    n_divergences: np.ndarray | None = None,
    title: str | None = None,
) -> Figure:
    """Max R-hat for every fold-fit against D22's 1.01 threshold. A fit is
    flagged for R-hat > 1.01 (red) or for any divergence (orange ×) - two
    different problems, so they are drawn differently.

    Saved to results/figures/rq4_bayesian_convergence_{model_slug}.png.
    """
    max_rhat_arr = np.asarray(max_rhat, dtype=float)
    flagged_arr = np.asarray(flagged, dtype=bool)
    divergences = np.zeros(len(max_rhat_arr), dtype=int) if n_divergences is None else np.asarray(n_divergences)
    x = np.arange(len(max_rhat_arr))
    high_rhat = max_rhat_arr > 1.01
    diverged = divergences > 0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(1.01, linestyle="--", color="gray", linewidth=1, zorder=1)
    ok = ~flagged_arr
    ax.scatter(x[ok], max_rhat_arr[ok], color="tab:blue", zorder=2)
    ax.scatter(x[high_rhat], max_rhat_arr[high_rhat], color="tab:red", zorder=3)
    ax.scatter(x[diverged], max_rhat_arr[diverged], marker="x", s=70, color="tab:orange", zorder=4)

    ax.legend(handles=[
        Line2D([0], [0], linestyle="--", color="gray", label="threshold (R-hat 1.01)"),
        Line2D([0], [0], marker="o", linestyle="", color="tab:blue", label=f"converged ({int(ok.sum())})"),
        Line2D([0], [0], marker="o", linestyle="", color="tab:red", label=f"R-hat > 1.01 ({int(high_rhat.sum())})"),
        Line2D([0], [0], marker="x", linestyle="", color="tab:orange",
               label=f"divergences ({int(diverged.sum())} fits, {int(divergences.sum())} total)"),
    ], loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_ylim(min(max_rhat_arr.min(), 1.0) - 0.001, max(max_rhat_arr.max(), 1.01) + 0.003)
    ax.set_xlabel("fold-fit (5 folds × 10 repeats)")
    ax.set_ylabel("worst R-hat in the fit")
    ax.set_title(title or f"Bayesian model convergence ({judge_name(model_slug)})")
    return _save(fig, f"rq4_bayesian_convergence_{model_slug}.png")


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
    """Task 5.9d: the 3-call ensemble vs the 1-call Bayesian model on error
    detection (AUROC) and on how well each one's own epistemic signal
    detects errors.

    The Bayesian AUROC has no interval on purpose: every other point has a
    cluster-bootstrap CI, and borrowing D8's across-repeat spread for it
    would imply a false equivalence between two kinds of interval.

    Saved to results/figures/rq5_distillation_{model_slug}.png.
    """
    labels = [
        "error detection: ensemble (3 calls)",
        "error detection: Bayesian (1 call)",
        "its epistemic signal: ensemble",
        "its epistemic signal: Bayesian",
    ]
    values = [auroc_ensemble, auroc_bayesian, epistemic_auroc_ensemble, epistemic_auroc_bayesian]
    lows = [auroc_ensemble_ci[0], np.nan, epistemic_auroc_ensemble_ci[0], epistemic_auroc_bayesian_ci[0]]
    highs = [auroc_ensemble_ci[1], np.nan, epistemic_auroc_ensemble_ci[1], epistemic_auroc_bayesian_ci[1]]
    colors = ["tab:orange", "tab:blue", "tab:orange", "tab:blue"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # errorbar can't take NaN widths, so the point-only row gets zero width.
    safe_lows = [v if np.isnan(lo) else lo for v, lo in zip(values, lows)]
    safe_highs = [v if np.isnan(hi) else hi for v, hi in zip(values, highs)]
    _draw_forest(ax, labels, values, safe_lows, safe_highs, "AUROC predicting judge error (dashed = chance)",
                 colors=colors, annotate=False, reference=0.5)
    for y, (v, lo, hi) in enumerate(zip(values, lows, highs)):
        text = f"{v:.3f} (point only)" if np.isnan(lo) else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
        ax.annotate(text, xy=(v, y), xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=8, color="dimgray")
    ax.set_title("A single-call Bayesian model vs. a 3-prompt ensemble\n(95% cluster-bootstrap CIs)", fontsize=10)
    return _save(fig, f"rq5_distillation_{model_slug}.png")


def plot_rq5_verbose_shift(
    aleatoric_clean: float,
    aleatoric_verbose: float,
    aleatoric_gap_ci: tuple[float, float],
    epistemic_clean: float,
    epistemic_verbose: float,
    epistemic_gap_ci: tuple[float, float],
    model_slug: str,
) -> Figure:
    """Task 5.9f: the change in mean aleatoric and epistemic entropy from
    clean to verbose, for the one Bayesian model fit on clean, with the
    paired-bootstrap CI the preregistered verdict rests on. Two panels on
    separate axes - epistemic is ~100x smaller than aleatoric. The
    prediction was: epistemic rises (> 0), aleatoric stays at 0.

    Saved to results/figures/rq5_verbose_shift_{model_slug}.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    panels = [
        (axes[0], "Aleatoric (predicted: no change)", aleatoric_clean, aleatoric_verbose, aleatoric_gap_ci),
        (axes[1], "Epistemic (predicted: rises)", epistemic_clean, epistemic_verbose, epistemic_gap_ci),
    ]
    for ax, label, clean_val, verbose_val, (gap_lo, gap_hi) in panels:
        gap = verbose_val - clean_val
        _draw_forest(ax, ["verbose − clean"], [gap], [gap_lo], [gap_hi],
                     f"change in mean entropy, nats\n(clean {clean_val:.4f} → verbose {verbose_val:.4f})",
                     colors=["tab:blue"], decimals=4)
        ax.set_title(label)
    fig.suptitle("Does the meta-model's uncertainty rise under the padding attack?\n(paired 95% cluster-bootstrap CIs)")
    return _save(fig, f"rq5_verbose_shift_{model_slug}.png")


def plot_judge_comparison(panels: list[dict], filename: str = "judge_comparison.png") -> Figure:
    """The three judges side by side, one panel per measure, one row per
    judge population (split by coverage regime or turn where the RQ split
    it). Rows are coloured by judge.

    Each panel: {"title", "xlabel", "reference" (x of the dashed line, or
    None), "rows": [(judge, label, value, ci_low, ci_high), ...]}.
    Saved to results/figures/{filename}.
    """
    n_rows = max(len(p["rows"]) for p in panels)
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, (0.6 * n_rows + 1.6) * nrows), squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        rows = panel["rows"]
        _draw_forest(
            ax,
            [label for _, label, _, _, _ in rows],
            [v for _, _, v, _, _ in rows],
            [lo for _, _, _, lo, _ in rows],
            [hi for _, _, _, _, hi in rows],
            panel["xlabel"],
            colors=[JUDGE_COLORS.get(judge, "tab:gray") for judge, _, _, _, _ in rows],
            reference=panel.get("reference"),
        )
        ax.set_title(panel["title"], fontsize=10)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle("Three judges, same items, same tests (95% cluster-bootstrap CIs)")
    return _save(fig, filename)


if __name__ == "__main__":
    plot_ece_auroc_orthogonal()
