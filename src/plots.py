"""Matplotlib figures, one per function, each saved to results/figures/ under
a deterministic, model_slug-suffixed name (CONTRIBUTING.md §5, D26).

Figures use plain-language names for signals and judges (SIGNAL_LABELS,
JUDGE_NAMES) so they read without the code; the CSVs keep the column names.
Intervals are always labelled with their kind: a cluster-bootstrap CI and
D8's across-repeat spread are different quantities.
"""

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

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
    "conf_ens_prob": "3-prompt ensemble",
    "conf_kev_bpe_prob": "order-swap agreement",
    "conf_sc_bpe_autoj_prob": "order-swap agreement",
    "conf_sc_bpe_autoj_greedy_prob": "order-swap agreement (greedy calls)",
    "p_correct_bayesian": "Bayesian meta-model",
    "p_correct_logreg": "logistic-regression meta-model",
}

# Each judge's naive confidence - what it reports by default, the signal that
# lets errors through - drawn red in single-judge figures. Reliability
# diagrams name a split by suffix (kev's coverage regime, auto-j's turn).
NAIVE_SIGNALS = {
    "conf_verb",
    "conf_kev", "conf_kev_in_coverage", "conf_kev_out_of_coverage",
    "conf_sc_autoj", "conf_sc_autoj_turn1", "conf_sc_autoj_turn2",
}

# A signal's colour role in single-judge comparisons, the same in every figure:
# red the naive one, blue the fix, then signal-2, signal-3, ink. Unlisted
# signals are grey.
SIGNAL_ROLES = {
    "conf_verb": "accent",
    "conf_bpe": "works", "p_correct_bayesian": "works", "ens_entropy_total": "works",
    "conf_ens": "signal2", "conf_ens_prob": "signal2",
    "conf_lp": "signal3", "ens_entropy_aleatoric": "signal3",
    "conf_sc": "ink",
    "ens_entropy_epistemic": "accent",
}


def _signal_color(signal: str) -> str:
    return getattr(_theme, SIGNAL_ROLES.get(signal, "muted"))


# Signals built from the same AB/BA pair that defines `flipped`: their
# flipped-vs-unflipped confidence gap is large by construction.
BY_CONSTRUCTION = {"conf_bpe", "conf_kev_bpe", "conf_sc_bpe_autoj"}

JUDGE_NAMES = {
    "qwen2.5_7b_instruct": "Qwen2.5-7B",
    "kev_8b": "kev-8b",
    "autoj_13b_gptq_4bits": "auto-j-13b",
}
# Colour roles shared with the site (design/tokens.json), so the report, README,
# site and slides agree. Two rules decide a series' colour:
# - a figure with more than one judge colours each by its judge; red marks only
#   the naive signal that lets errors through;
# - a single-judge figure (the judge named in its title) colours by role: red =
#   naive or harmful, blue ("works") = the fix or a significant improvement,
#   grey = not significant or in between. Where it compares several signals,
#   two more colours (signal-2 teal-blue, signal-3 magenta) keep them apart,
#   each signal always in the same one (SIGNAL_ROLES).
# An effect with no good or bad direction is ink, hollow grey if not significant.
# Qwen's purple and works' blue merge for red-green colour-blind readers, which
# is why they never share a figure. Amber and teal sit under 3:1 on the light
# paper, so every figure that uses them also labels the judge in text.
_TOKENS = json.loads((Path(__file__).resolve().parents[1] / "design" / "tokens.json").read_text(encoding="utf-8"))

# Line styles, as on the site: dashed = an attacked (padded or swapped) series;
# dotted = a reference line (chance, zero, a threshold, perfect calibration).
# Dash lengths scale with line width, and round caps turn the dotted pattern
# into dots, so the two never look alike.
DASHED = (0, (6, 4))
DOTTED = (0, (1, 3))
REFERENCE_LINE = {"linestyle": DOTTED, "dash_capstyle": "round", "linewidth": 1.5}


@dataclass(frozen=True)
class _Theme:
    """The colours a figure uses, by role, for one theme of design/tokens.json.
    A dark variant differs only in colour and in its `_dark` filename suffix."""
    rc: dict
    ink: str            # text; reference curves, e.g. the oracle
    grey: str           # reference lines
    note: str           # small annotation text: full ink, since grey at 7-8 pt is hard to read
    shade: str          # background bands
    edge: str           # the paper: marker edges between overlapping points, hollow markers' fill
    muted: str          # estimates whose CI includes 0; in-between signals
    accent: str         # red: the naive signal, a harmful effect
    works: str          # blue: the fix, a helpful effect (single-judge figures only)
    signal2: str        # teal-blue and magenta: further signals in a single-judge comparison
    signal3: str
    ramp_low: str       # the light end of a sequential grey ramp (its dark end is ink)
    judge_colors: dict
    suffix: str


def _theme_from(t: dict, suffix: str) -> _Theme:
    rc = {
        "figure.facecolor": t["paper"], "axes.facecolor": t["paper"], "savefig.facecolor": t["paper"],
        "axes.edgecolor": t["dot-ok"], "axes.labelcolor": t["ink"], "text.color": t["ink"],
        "xtick.color": t["dot-ok"], "ytick.color": t["dot-ok"],
        "xtick.labelcolor": t["ink"], "ytick.labelcolor": t["ink"],
        "legend.edgecolor": t["rule"], "grid.color": t["rule"],
        # Long enough for a legend line to show at least two dashes or several dots.
        "legend.handlelength": 4.0,
        # An axis spanning less than 10^-3 gets its labels as multiples of 1e-n:
        # "-0.0006" tick labels run into each other on a narrow panel.
        "axes.formatter.limits": (-3, 6),
    }
    return _Theme(
        rc=rc, ink=t["ink"], grey=t["ink-3"], note=t["ink"], shade=t["paper-2"], edge=t["paper"],
        muted=t["ink-3"], accent=t["accent"], works=t["works"], signal2=t["signal-2"], signal3=t["signal-3"],
        ramp_low=t["dot-ok"],
        judge_colors={"Qwen2.5-7B": t["qwen"], "kev-8b": t["kev"], "auto-j-13b": t["autoj"]},
        suffix=suffix,
    )


LIGHT = _theme_from(_TOKENS["light"], suffix="")
DARK = _theme_from(_TOKENS["dark"], suffix="_dark")
_theme = LIGHT


def _dark_variant(plot_fn):
    """Adds a keyword `dark: bool = False` to a plot function. dark=True
    draws the same figure in the site's dark theme and saves it as
    `{name}_dark.png`, for slides and dark pages; dark=False draws the
    light theme."""
    @functools.wraps(plot_fn)
    def wrapper(*args, dark: bool = False, **kwargs):
        global _theme
        previous, _theme = _theme, (DARK if dark else LIGHT)
        try:
            with plt.rc_context(_theme.rc):
                return plot_fn(*args, **kwargs)
        finally:
            _theme = previous
    return wrapper


def _themed(filename: str) -> str:
    """`filename` with the active theme's suffix before the extension."""
    path = Path(filename)
    return f"{path.stem}{_theme.suffix}{path.suffix}"


def signal_label(name: str) -> str:
    return SIGNAL_LABELS.get(name, name)


def judge_name(model_slug: str) -> str:
    """Display name for a model_slug, including suffixed ones such as
    'kev_8b_in_coverage'."""
    for slug, name in JUDGE_NAMES.items():
        if model_slug.startswith(slug):
            return name
    return model_slug


def _fit_titles(fig: Figure, renderer) -> None:
    """Keeps every axes title inside the figure. A title is centred over its
    axes, so over an axes pushed right by long tick labels a long title can run
    off the right edge while the left margin is empty: slide it just far enough
    to fit, and shrink its font first if it is wider than the figure."""
    pad = 0.01 * fig.bbox.width
    for ax in fig.axes:
        title = ax.title
        if not title.get_text():
            continue
        while title.get_window_extent(renderer).width > fig.bbox.width - 2 * pad and title.get_fontsize() > 7:
            title.set_fontsize(title.get_fontsize() - 0.5)
        box = title.get_window_extent(renderer)
        shift = min(0.0, fig.bbox.width - pad - box.x1) or max(0.0, pad - box.x0)
        if shift:
            title.set_x(title.get_position()[0] + shift / ax.bbox.width)


def _thin_crowded_ticks(fig: Figure, renderer) -> None:
    """Halves the x ticks of any axis whose tick labels touch, so a narrow
    panel never prints its numbers into each other."""
    for ax in fig.axes:
        boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
        boxes = sorted((b for b in boxes if b.width), key=lambda b: b.x0)
        if any(a.x1 + 2 > b.x0 for a, b in zip(boxes, boxes[1:])):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=max(len(boxes) // 2, 2)))


def _save(fig: Figure, filename: str, rect: tuple[float, float, float, float] = (0, 0, 1, 1)) -> Figure:
    """Lays out and saves a figure. Every figure goes through here, so the
    layout guards (titles inside the figure, readable tick labels) cover new
    figures too."""
    fig.tight_layout(rect=rect)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    _thin_crowded_ticks(fig, renderer)
    _fit_titles(fig, renderer)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / _themed(filename), dpi=150)
    return fig


def _includes(reference: float | None, ci_low, ci_high) -> list[bool]:
    """Per estimate, whether its interval includes `reference` (never, if None)."""
    return [reference is not None and lo <= reference <= hi for lo, hi in zip(ci_low, ci_high)]


def _direction_styles(ci_low, ci_high, worse_if_positive: bool) -> tuple[list[str], list[bool]]:
    """Single-judge forests: (colors, hollow). Red where the interval sits wholly
    on the harmful side of 0, blue where it sits wholly on the helpful side,
    hollow grey where it includes 0 - colour for direction, fill for
    significance."""
    colors = []
    for lo, hi in zip(ci_low, ci_high):
        if lo <= 0 <= hi:
            colors.append(_theme.muted)
        else:
            colors.append(_theme.accent if (lo > 0) == worse_if_positive else _theme.works)
    return colors, _includes(0.0, ci_low, ci_high)


def _neutral_styles(ci_low, ci_high) -> tuple[list[str], list[bool]]:
    """Forests whose effects have no good or bad direction: (colors, hollow).
    Ink where the interval excludes 0, hollow grey where it includes it."""
    hollow = _includes(0.0, ci_low, ci_high)
    return [_theme.muted if empty else _theme.ink for empty in hollow], hollow


@_dark_variant
def plot_ece_auroc_orthogonal() -> Figure:
    """Two hand-built 6-item examples showing ECE and AUROC
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
        ax.scatter(x[correct_bool], confidences[correct_bool], color=_theme.ink, label="correct")
        ax.scatter(x[~correct_bool], confidences[~correct_bool], color=_theme.accent, label="incorrect")
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


@_dark_variant
def plot_reliability_diagram(
    confidences: np.ndarray,
    correct: np.ndarray,
    signal_name: str,
    n_bins: int,
    model_slug: str,
    strategy: str = "auto",
    label: str = "judge_verdict",
    overlay: tuple[np.ndarray, np.ndarray, str] | None = None,
    overlay_role: str = "works",
    main_role: str | None = None,
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

    Colours by role (a single-judge figure): the main curve is red for a
    judge's naive confidence (NAIVE_SIGNALS) and ink otherwise; the overlay
    takes `overlay_role` - "works" (blue) for a fix such as the order-averaged
    verdict, "ink" for an alternative that is neither. `main_role` overrides
    the main curve's colour, e.g. "works" for the Bayesian model. The overlay
    is drawn with smaller squares on top, so where the two curves coincide
    the main curve's circle still shows around the overlay's square.

    The axes start at the lowest confidence or accuracy shown, never above
    0.5, so the region where the points actually sit fills the panel.

    Saved to results/figures/reliability_{signal_name}_{model_slug}.png.
    """
    if main_role is not None:
        main_color = getattr(_theme, main_role)
    else:
        main_color = _theme.accent if signal_name in NAIVE_SIGNALS else _theme.ink
    curves = [(np.asarray(confidences, dtype=float), np.asarray(correct, dtype=float), label, main_color, "o", 1.0)]
    if overlay is not None:
        overlay_conf, overlay_correct, overlay_label = overlay
        curves.append((np.asarray(overlay_conf, dtype=float), np.asarray(overlay_correct, dtype=float),
                       overlay_label, getattr(_theme, overlay_role), "s", 0.35))

    fig, (ax, hist_ax) = plt.subplots(
        2, 1, figsize=(5.5, 6.5), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )

    lowest = 0.5
    legend_handles = []
    for conf, corr, curve_label, color, marker, size in curves:
        conf_pts, acc_pts, weights = _binned_means(conf, corr, n_bins, strategy)
        ece_value, n_effective_bins = ece(conf, corr, n_bins, strategy)
        lowest = min(lowest, conf_pts.min(), acc_pts.min())

        ax.plot(conf_pts, acc_pts, color=color, alpha=0.5, zorder=1)
        ax.scatter(
            conf_pts, acc_pts, s=size * _marker_sizes(weights), alpha=0.8, color=color, marker=marker,
            edgecolors=_theme.edge, linewidths=1, zorder=2 if size == 1.0 else 3,
        )
        legend_handles.append(
            Line2D(
                [0], [0], marker=marker, linestyle="", color=color, markersize=9 if size == 1.0 else 6,
                label=f"{curve_label}: ECE {ece_value:.3f} ({n_effective_bins} bins)",
            )
        )
        # Outlines, not translucent fills: overlapping red and blue fills would blend
        # into Qwen's purple.
        hist_ax.hist(conf, bins=40, range=(0, 1), color=color, histtype="step", linewidth=1.5)

    lo = np.floor(lowest * 10) / 10 - 0.02
    ax.plot([lo, 1], [lo, 1], color=_theme.grey, zorder=0, **REFERENCE_LINE)
    legend_handles.insert(0, Line2D([0], [0], color=_theme.grey, label="perfect calibration", **REFERENCE_LINE))

    ax.set_xlim(lo, 1.02)
    ax.set_ylim(lo, 1.02)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{signal_label(signal_name)} ({judge_name(model_slug)})")
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    ax.text(0.98, 0.02, note, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=_theme.note)

    hist_ax.set_xlabel(xlabel)
    hist_ax.set_ylabel("items")
    return _save(fig, f"reliability_{signal_name}_{model_slug}.png")


@_dark_variant
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
    is shaded as noisy. Inside a tied group the curve is the expected risk
    under a random tie-break (risk_coverage()), so a discrete signal's
    curve is flat across its most-confident group rather than missing.

    Args:
        curves: {name: (coverage, risk)}.
        oracle: (coverage, risk) from oracle_risk_coverage().
    """
    fig, ax = plt.subplots(figsize=(6.5, 6.4))

    # Curves can coincide (RQ5's total and aleatoric entropy almost do), so
    # the marker varies too - colour alone can't separate lines drawn on top
    # of each other. Every curve is solid: dashes mean an attacked series.
    markers = ["o", "s", "^", "D"]

    base_risk = None
    for i, (name, (coverage, risk)) in enumerate(curves.items()):
        coverage, risk = np.asarray(coverage, dtype=float), np.asarray(risk, dtype=float)
        base_risk = float(risk[np.argmax(coverage)])
        ax.plot(
            coverage, risk,
            color=_signal_color(name),
            marker=markers[i % len(markers)],
            markersize=3,
            markevery=0.05,
            alpha=0.85,
            label=f"{signal_label(name)} (AURC {aurc(coverage, risk):.3f})",
        )

    oracle_coverage, oracle_risk = oracle
    ax.plot(
        oracle_coverage, oracle_risk, color=_theme.ink, **REFERENCE_LINE,
        label=f"oracle (AURC {aurc(oracle_coverage, oracle_risk):.3f})",
    )
    if base_risk is not None:
        ax.axhline(base_risk, color=_theme.grey, **REFERENCE_LINE,
                   label=f"no abstention / random ({base_risk:.3f})")

    ax.axvspan(0, 0.05, color=_theme.shade, zorder=0)
    ax.text(0.025, 0.98, "noisy", transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=7, color=_theme.note, rotation=90)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("coverage (share of items the judge keeps)")
    ax.set_ylabel("risk (error rate on kept items)")
    ax.set_title(title)
    # Below the axes: the curves and the reference lines fill every corner.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)
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


@_dark_variant
def plot_human_disagreement(
    d_human: np.ndarray,
    correct: np.ndarray,
    confidence: np.ndarray,
    n_bins: int,
    model_slug: str,
) -> Figure:
    """Human disagreement (D9): accuracy (left) and stated confidence (right) at each
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
        (axes[0], correct_arr, "accuracy", _theme.ink, "o"),
        (axes[1], confidence_arr, "mean stated confidence", _theme.accent, "s"),
    ]
    for ax, y_arr, ylabel, color, marker in panels:
        bin_d, bin_y, weights = _binned_means(d_human_arr, y_arr, n_bins, strategy="auto")
        ax.scatter(bin_d, bin_y, s=_marker_sizes(weights), marker=marker, color=color,
                   alpha=0.85, edgecolors=_theme.edge, linewidths=1, zorder=2)
        for x, y, w in zip(bin_d, bin_y, weights):
            ax.annotate(f"n={round(w * len(d_human_arr))}", xy=(x, y), xytext=(0, 12),
                        textcoords="offset points", ha="center", fontsize=8, color=_theme.note)

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
    muted: str | None = None,
    reference_color: str | None = None,
    hollow: list[bool] | None = None,
    markers: list[str] | None = None,
    dashed: list[bool] | None = None,
) -> None:
    """Draws a forest panel onto `ax`: a point + interval bar per label
    (first label at the top) and a dotted reference line (default 0). No
    title, no save - the shared primitive under every forest figure.

    Args:
        colors: optional per-label colors (default: all ink).
        hollow: per label, draw the point hollow - by convention, where the
            interval includes the reference (not significant).
        markers: per label, the point's marker (default: all circles).
        dashed: per label, draw the interval dashed - an attacked condition.
        annotate: print "value [lo, hi]" above each point. Turn off for
            dense panels whose exact values live in a CSV.
        reference: x of the dotted line, or None for no line.
        decimals: decimal places in the annotations.
        muted, reference_color: annotation and reference-line colours;
            default to the active theme's.
    """
    muted = muted or _theme.note
    reference_color = reference_color or _theme.grey
    labels = list(labels)
    values_arr = np.asarray(values, dtype=float)
    ci_low_arr = np.asarray(ci_low, dtype=float)
    ci_high_arr = np.asarray(ci_high, dtype=float)
    point_colors = list(colors) if colors is not None else [_theme.ink] * len(labels)

    y_pos = np.arange(len(labels))
    # errorbar takes distances from the point, not absolute CI bounds.
    err_low = values_arr - ci_low_arr
    err_high = ci_high_arr - values_arr

    hollow_flags = list(hollow) if hollow is not None else [False] * len(labels)
    point_markers = list(markers) if markers is not None else ["o"] * len(labels)
    dashed_flags = list(dashed) if dashed is not None else [False] * len(labels)
    if reference is not None:
        ax.axvline(reference, color=reference_color, zorder=1, **REFERENCE_LINE)
    # One errorbar call per point, since one call takes only one color.
    for x, y, lo_err, hi_err, color, empty, marker, attacked in zip(
            values_arr, y_pos, err_low, err_high, point_colors, hollow_flags, point_markers, dashed_flags):
        bars = ax.errorbar([x], [y], xerr=[[lo_err], [hi_err]], fmt=marker, color=color, ecolor=color,
                           markerfacecolor=_theme.edge if empty else color, markeredgewidth=1.5,
                           capsize=4, markersize=7, zorder=2)
        if attacked:
            bars.lines[2][0].set_linestyle(DASHED)

    # Exact values as text: when one estimate dwarfs the others, small but
    # real ones shrink to a dot with an invisible bar.
    if annotate:
        for x, y, lo, hi in zip(values_arr, y_pos, ci_low_arr, ci_high_arr):
            text = f"{x:.{decimals}f}" if np.isnan(lo) else f"{x:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"
            ax.annotate(text, xy=(x, y), xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8, color=muted)

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
    hollow: list[bool] | None = None,
) -> Figure:
    """A single-panel forest plot (several estimates with CIs against 0),
    saved to results/figures/{filename}. Height scales with the label count.
    """
    fig, ax = plt.subplots(figsize=(7, 0.8 * len(labels) + 1.6))
    _draw_forest(ax, labels, values, ci_low, ci_high, xlabel, colors=colors, hollow=hollow)
    ax.set_title(title)
    return _save(fig, filename)


@_dark_variant
def plot_d_human_correlations(
    signals: list[str],
    spearman: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    model_slug: str,
    filename_suffix: str = "",
    title: str = "Does confidence track human consensus?",
) -> Figure:
    """Forest plot of signals' Spearman correlation with d_human (also used
    by RQ5). A second caller with a different signal set must pass
    a `filename_suffix`, or it overwrites the first caller's figure.

    A correlation here has no good or bad direction: ink where the CI
    excludes 0, hollow grey where it doesn't.

    Saved to results/figures/d_human_correlations{filename_suffix}_{model_slug}.png.
    """
    colors, hollow = _neutral_styles(ci_low, ci_high)
    return _forest_plot(
        labels=[signal_label(s) for s in signals],
        values=spearman,
        ci_low=ci_low,
        ci_high=ci_high,
        xlabel="Spearman ρ with d_human, 95% cluster-bootstrap CI",
        title=f"{title} ({judge_name(model_slug)})",
        filename=f"d_human_correlations{filename_suffix}_{model_slug}.png",
        colors=colors,
        hollow=hollow,
    )


@_dark_variant
def plot_confidence_gap(panels: list[dict], title: str, filename: str) -> Figure:
    """RQ3a/RQ6/RQ7: each signal's mean confidence on items whose verdict
    flipped between AB and BA, minus on items that didn't - one panel per
    population (e.g. per coverage regime or turn). A negative gap with a CI
    excluding 0 means the signal drops where order changed the verdict: it
    flags those errors (blue). A significant rise is red; a CI including 0
    is hollow grey.

    Signals in BY_CONSTRUCTION are drawn filled grey and marked †: they are built
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
        colors, hollow = _direction_styles(panel["ci_low"], panel["ci_high"], worse_if_positive=True)
        for i, signal in enumerate(panel["signals"]):
            if signal in BY_CONSTRUCTION:
                colors[i], hollow[i] = _theme.muted, False
        _draw_forest(ax, labels, panel["gap"], panel["ci_low"], panel["ci_high"],
                     "confidence when flipped − when stable\n(95% cluster-bootstrap CI)", colors=colors, hollow=hollow)
        ax.set_title(panel["title"])
    fig.suptitle(title)
    fig.text(0.01, 0.01, "† built from the same two orders that define a flip, so a large gap is expected by construction",
             fontsize=7, color=_theme.note)
    return _save(fig, filename, rect=(0, 0.04, 1, 1))


@_dark_variant
def plot_verbosity_deltas(panels: list[dict], title: str, filename: str) -> Figure:
    """RQ3b/RQ6/RQ7: ΔECE and ΔAUROC (verbose − clean) side by side, one
    row per population. Positive ΔECE = worse calibration; negative ΔAUROC
    = the signal detects errors less well. Red where padding significantly
    hurt, blue where it significantly helped, hollow grey where the paired
    CI includes 0.

    Each panel: {"title", "signals", "delta_ece", "ece_ci_low", "ece_ci_high",
    "delta_auroc", "auroc_ci_low", "auroc_ci_high"}.
    Saved to results/figures/{filename}.
    """
    n_signals = max(len(p["signals"]) for p in panels)
    fig, axes = plt.subplots(len(panels), 2, figsize=(12, (0.8 * n_signals + 1.4) * len(panels) + 0.6), squeeze=False)
    for row, panel in zip(axes, panels):
        labels = [signal_label(s) for s in panel["signals"]]
        ece_colors, ece_hollow = _direction_styles(panel["ece_ci_low"], panel["ece_ci_high"], worse_if_positive=True)
        auroc_colors, auroc_hollow = _direction_styles(panel["auroc_ci_low"], panel["auroc_ci_high"],
                                                       worse_if_positive=False)
        _draw_forest(row[0], labels, panel["delta_ece"], panel["ece_ci_low"], panel["ece_ci_high"],
                     "Δ ECE, verbose − clean (> 0 = worse calibrated)", colors=ece_colors, hollow=ece_hollow)
        _draw_forest(row[1], labels, panel["delta_auroc"], panel["auroc_ci_low"], panel["auroc_ci_high"],
                     "Δ AUROC, verbose − clean (< 0 = detects errors less well)",
                     colors=auroc_colors, hollow=auroc_hollow)
        row[0].set_title(f"{panel['title']}: calibration" if panel["title"] else "Calibration")
        row[1].set_title(f"{panel['title']}: error detection" if panel["title"] else "Error detection")
        row[1].set_yticklabels([])
    fig.suptitle(f"{title}\n(paired 95% cluster-bootstrap CIs; red = worse under padding, blue = better, "
                 "hollow = CI includes 0)")
    return _save(fig, filename)


# RQ4's meta-models are the proposed fix, so they are blue and the marker tells them
# apart. Where the Bayesian model sits next to logistic regression, the Bayesian
# model takes the blue and logistic regression is ink.
MODEL_MARKERS = {"logreg": "o", "histgbm": "s"}


@_dark_variant
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
    """RQ4 tier ablation: AUROC of each (tier, model) meta-model next to the best
    single signal, on a zoomed axis - the differences are hundredths.
    Tier intervals are D8's across-repeat spread; the baseline's is a 95%
    cluster-bootstrap CI. `null_means`, if given, shades the range of the
    permutation-null means (invariant 12).

    Saved to results/figures/rq4_ablation_{model_slug}.png.
    """
    by_cell = {(t, m): (v, lo, hi) for t, m, v, lo, hi in zip(tiers, models, auroc_mean, auroc_low, auroc_high)}
    tier_names = {"A": "A: judge's own signals", "B": "B: + surface features", "C": "C: + token statistics"}
    labels, values, lows, highs = [f"best single signal ({signal_label(baseline_label)})"], [baseline], \
        [baseline_ci_low], [baseline_ci_high]
    colors, markers = [_theme.ink], ["D"]
    for tier in ["A", "B", "C"]:
        for model in sorted(set(models)):
            if (tier, model) in by_cell:
                v, lo, hi = by_cell[(tier, model)]
                labels.append(f"Tier {tier_names[tier]}, {model}")
                values.append(v)
                lows.append(lo)
                highs.append(hi)
                colors.append(_theme.works)
                markers.append(MODEL_MARKERS.get(model, "o"))

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(labels) + 2))
    if null_means is not None and len(null_means):
        ax.axvspan(min(null_means), max(null_means), color=_theme.shade, zorder=0)
        ax.text(np.mean(null_means), len(labels) - 0.6, "permutation null", ha="center", va="bottom",
                fontsize=8, color=_theme.note)
    _draw_forest(ax, labels, values, lows, highs, "AUROC predicting judge error", colors=colors, markers=markers,
                 reference=baseline)
    ax.set_title("Do more features help predict judge error?\n"
                 "(tiers: range over 10 CV repeats; baseline: 95% cluster-bootstrap CI)", fontsize=10)
    return _save(fig, f"rq4_ablation_{model_slug}.png")


@_dark_variant
def plot_rq4_progression(
    comparisons: list[str],
    auroc_diff: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    model_slug: str,
) -> Figure:
    """RQ4 tier ablation: forest plot of each tier step's paired AUROC change. A CI
    excluding 0 means that step is real: blue if it raised AUROC, red if it
    lowered it; hollow grey points are not real.

    Saved to results/figures/rq4_progression_{model_slug}.png.
    """
    colors, hollow = _direction_styles(ci_low, ci_high, worse_if_positive=False)
    return _forest_plot(
        labels=comparisons,
        values=auroc_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        xlabel="Δ AUROC, higher tier − lower tier (paired 95% cluster-bootstrap CI)",
        title="Does each added feature tier change AUROC?",
        filename=f"rq4_progression_{model_slug}.png",
        colors=colors,
        hollow=hollow,
    )


@_dark_variant
def plot_rq4_permutation_nulls(
    tiers: list[str],
    models: list[str],
    observed: np.ndarray,
    null_distributions: list[np.ndarray],
    model_slug: str,
) -> Figure:
    """RQ4 tier ablation: for each (tier, model) cell, every permutation-null AUROC
    (grey dots, labels shuffled within each question) and the observed
    AUROC (ink diamond), on one shared axis.

    Saved to results/figures/rq4_permutation_nulls_{model_slug}.png.
    """
    rng = np.random.default_rng(0)  # jitter only; no statistic depends on it
    order = sorted(range(len(tiers)), key=lambda i: (tiers[i], models[i]))

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(order) + 1.8))
    for row, i in enumerate(order):
        null = np.asarray(null_distributions[i], dtype=float)
        ax.scatter(null, row + rng.uniform(-0.15, 0.15, len(null)), s=10, color=_theme.grey, alpha=0.5, zorder=2)
        ax.scatter([observed[i]], [row], marker="D", s=60, color=_theme.ink, zorder=3)
    ax.axvline(0.5, color=_theme.grey, zorder=1, **REFERENCE_LINE)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"Tier {tiers[i]}, {models[i]}" for i in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel("AUROC predicting judge error")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", linestyle="", color=_theme.grey, label="null (labels shuffled within question)"),
        Line2D([0], [0], marker="D", linestyle="", color=_theme.ink, label="observed"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8)
    ax.set_title("Observed AUROC vs. permutation null")
    return _save(fig, f"rq4_permutation_nulls_{model_slug}.png")


@_dark_variant
def plot_h4_interaction(
    oof_score: np.ndarray,
    oof_score_grid: np.ndarray,
    d_human_values: list[float],
    predicted_curves: list[np.ndarray],
    log_odds_curves: list[np.ndarray],
    model_slug: str,
) -> Figure:
    """H4: predicted P(correct) vs the predictor's OOF score, one curve
    per d_human level. Left: probability scale, with a rug of the real
    scores. Right: log-odds, where the model is linear and the interaction
    is simply the slope difference - on the probability scale, sigmoid
    saturation hides it where most real scores sit.

    Saved to results/figures/h4_interaction_{model_slug}.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # A grey ramp, light to dark as consensus grows: a sequential magnitude in
    # one hue, so no curve reads as a judge. Markers repeat the order.
    ramp = LinearSegmentedColormap.from_list("ramp", [_theme.ramp_low, _theme.ink])
    colors = [ramp(i / max(len(d_human_values) - 1, 1)) for i in range(len(d_human_values))]
    markers = ["o", "s", "^", "D"]

    ax = axes[0]
    for i, (color, d_human_value, curve) in enumerate(zip(colors, d_human_values, predicted_curves)):
        ax.plot(oof_score_grid, curve, color=color, linewidth=2.5, marker=markers[i % len(markers)], markevery=0.1,
                label=f"d_human = {d_human_value:.3f}")
    # Rug below the [0, 1] axis so it never overlaps the curves.
    ax.plot(oof_score, np.full_like(oof_score, -0.04), marker="|", linestyle="", color=_theme.ink, alpha=0.3,
            markersize=8, clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.08, 1.02)
    ax.set_xlabel("meta-model's out-of-fold P(correct)")
    ax.set_ylabel("fitted P(judge correct)")
    ax.set_title("Probability scale")
    ax.legend(loc="upper left", fontsize=9, title="human consensus (0.5 = unanimous)")

    ax = axes[1]
    for i, (color, d_human_value, curve) in enumerate(zip(colors, d_human_values, log_odds_curves)):
        ax.plot(oof_score_grid, curve, color=color, linewidth=2.5, marker=markers[i % len(markers)], markevery=0.1,
                label=f"d_human = {d_human_value:.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("meta-model's out-of-fold P(correct)")
    ax.set_ylabel("log-odds of correct")
    ax.set_title("Log-odds scale (slopes undistorted)")

    fig.suptitle("The meta-model's edge grows with human consensus (H4)")
    return _save(fig, f"h4_interaction_{model_slug}.png")


@_dark_variant
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
    """Transfer test (clean -> verbose): per model, in-domain AUROC (clean CV, D8 across-repeat
    spread) next to transfer AUROC (fit on clean, evaluated on verbose, 95%
    cluster-bootstrap CI), on a zoomed axis.

    Saved to results/figures/rq4_transfer_{model_slug}.png.
    """
    labels, values, lows, highs, markers, dashed = [], [], [], [], [], []
    for i, model in enumerate(models):
        labels += [f"{model}: in-domain (clean)", f"{model}: transfer (clean → verbose)"]
        values += [baseline_mean[i], transfer_mean[i]]
        lows += [baseline_low[i], transfer_low[i]]
        highs += [baseline_high[i], transfer_high[i]]
        markers += [MODEL_MARKERS.get(model, "o")] * 2
        dashed += [False, True]

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(labels) + 2))
    _draw_forest(ax, labels, values, lows, highs, "AUROC predicting judge error", colors=[_theme.works] * len(labels),
                 markers=markers, dashed=dashed, reference=None)
    ax.set_title("Does a meta-model trained on clean inputs survive the padding attack?\n"
                 "(in-domain: range over 10 CV repeats; transfer, dashed: 95% cluster-bootstrap CI)", fontsize=10)
    return _save(fig, f"rq4_transfer_{model_slug}.png")


@_dark_variant
def plot_rq4_category_transfer(
    categories: list[str],
    models: list[str],
    auroc: np.ndarray,
    model_slug: str,
) -> Figure:
    """Category transfer: held-out AUROC per category (LeaveOneGroupOut), one marker
    per model, categories sorted weakest to strongest. No category is
    singled out in advance; the dotted line marks chance. Each value is one
    deterministic split, so there is no interval.

    Saved to results/figures/rq4_category_transfer_{model_slug}.png.
    """
    df = pd.DataFrame({"category": categories, "model": models, "auroc": auroc})
    category_order = df.groupby("category")["auroc"].mean().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvline(0.5, color=_theme.grey, zorder=1, **REFERENCE_LINE)
    for model in sorted(df["model"].unique()):
        model_df = df[df["model"] == model].set_index("category").reindex(category_order)
        ax.scatter(model_df["auroc"], np.arange(len(category_order)), color=_theme.works,
                   marker=MODEL_MARKERS.get(model, "o"), s=60, label=model, zorder=2)
    ax.set_yticks(np.arange(len(category_order)))
    ax.set_yticklabels(category_order)
    ax.set_ylim(len(category_order) - 0.5, -0.5)
    ax.set_xlim(0.45, 0.95)
    ax.set_xlabel("AUROC on the held-out category (dotted = chance)")
    ax.set_title("Does the meta-model generalize to an unseen category?")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, f"rq4_category_transfer_{model_slug}.png")


# Feature families are categories, told apart at a glance by colour: blue,
# magenta and ink are the trio that stays apart under colour blindness (blue
# and teal-blue do not). Each also has its own marker shape.
_FEATURE_FAMILY_STYLES = {"A": ("works", "o"), "B": ("signal3", "s"), "C": ("ink", "^")}


def _family_color(family: str) -> str:
    return getattr(_theme, _FEATURE_FAMILY_STYLES[family][0])
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
        Line2D([0], [0], marker=_FEATURE_FAMILY_STYLES[f][1], linestyle="", color=_family_color(f),
               label=_FEATURE_FAMILY_NAMES[f])
        for f in ["A", "B", "C"]
        if f in families_present
    ]


@_dark_variant
def plot_rq4_coefficients(
    features: list[str],
    coef: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    family: list[str],
    model_slug: str,
) -> Figure:
    """RQ4's coefficient figure: only the coefficients whose CI excludes
    0, coloured and shaped by feature family (all ~37 rows would be too tall for a slide;
    plot_rq4_coefficients_full() and the CSV have them). Takes the FULL
    table and filters here, so "significant" has one definition. With no
    significant coefficient it still renders, as an empty panel.

    Saved to results/figures/rq4_coefficients_{model_slug}.png.
    """
    df = _sort_coefficient_rows(features, coef, ci_low, ci_high, family)
    df = df[(df["ci_low"] > 0) | (df["ci_high"] < 0)].reset_index(drop=True)
    colors = [_family_color(f) for f in df["family"]]
    markers = [_FEATURE_FAMILY_STYLES[f][1] for f in df["family"]]

    fig, ax = plt.subplots(figsize=(7.5, 0.8 * max(len(df), 1) + 1.8))
    _draw_forest(ax, df["feature"].tolist(), df["coef"].to_numpy(), df["ci_low"].to_numpy(),
                 df["ci_high"].to_numpy(), "coefficient on standardized feature (> 0 = judge more likely right)",
                 colors=colors, markers=markers)
    ax.legend(handles=_family_legend_handles(sorted(df["family"].unique())), loc="lower right", fontsize=8)
    ax.set_title(f"Which features predict judge error?\n{len(df)} of {len(features)} coefficients clear zero "
                 f"(logistic regression, all tiers)", fontsize=10)
    return _save(fig, f"rq4_coefficients_{model_slug}.png")


@_dark_variant
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
    values are in the CSV. Hollow points have a CI that includes 0.

    Saved to results/figures/rq4_coefficients_full_{model_slug}.png.
    """
    df = _sort_coefficient_rows(features, coef, ci_low, ci_high, family)
    colors = [_family_color(f) for f in df["family"]]
    markers = [_FEATURE_FAMILY_STYLES[f][1] for f in df["family"]]

    fig, ax = plt.subplots(figsize=(8, 0.22 * len(df) + 1.2))
    _draw_forest(ax, df["feature"].tolist(), df["coef"].to_numpy(), df["ci_low"].to_numpy(),
                 df["ci_high"].to_numpy(), "coefficient on standardized feature", colors=colors, markers=markers,
                 hollow=_includes(0.0, df["ci_low"], df["ci_high"]), annotate=False)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(handles=_family_legend_handles(["A", "B", "C"]), loc="lower right", fontsize=8)
    ax.set_title(f"All meta-model coefficients (exact values: rq4_coefficients_{model_slug}.csv)", fontsize=9)
    return _save(fig, f"rq4_coefficients_full_{model_slug}.png")


@_dark_variant
def plot_bayesian_convergence(
    max_rhat: np.ndarray,
    flagged: np.ndarray,
    model_slug: str,
    n_divergences: np.ndarray | None = None,
    title: str | None = None,
) -> Figure:
    """Max R-hat for every fold-fit against D22's 1.01 threshold. A fit is
    flagged for R-hat > 1.01 (red dot) or for any divergence (red ×) - two
    different problems, so they are drawn differently. Converged fits are ink.

    Saved to results/figures/rq4_bayesian_convergence_{model_slug}.png.
    """
    max_rhat_arr = np.asarray(max_rhat, dtype=float)
    flagged_arr = np.asarray(flagged, dtype=bool)
    divergences = np.zeros(len(max_rhat_arr), dtype=int) if n_divergences is None else np.asarray(n_divergences)
    x = np.arange(len(max_rhat_arr))
    high_rhat = max_rhat_arr > 1.01
    diverged = divergences > 0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(1.01, color=_theme.grey, zorder=1, **REFERENCE_LINE)
    ok = ~flagged_arr
    ax.scatter(x[ok], max_rhat_arr[ok], color=_theme.ink, zorder=2)
    ax.scatter(x[high_rhat], max_rhat_arr[high_rhat], color=_theme.accent, zorder=3)
    ax.scatter(x[diverged], max_rhat_arr[diverged], marker="x", s=70, color=_theme.accent, zorder=4)

    ax.legend(handles=[
        Line2D([0], [0], color=_theme.grey, label="threshold (R-hat 1.01)", **REFERENCE_LINE),
        Line2D([0], [0], marker="o", linestyle="", color=_theme.ink, label=f"converged ({int(ok.sum())})"),
        Line2D([0], [0], marker="o", linestyle="", color=_theme.accent, label=f"R-hat > 1.01 ({int(high_rhat.sum())})"),
        Line2D([0], [0], marker="x", linestyle="", color=_theme.accent,
               label=f"divergences ({int(diverged.sum())} fits, {int(divergences.sum())} total)"),
    ], loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_ylim(min(max_rhat_arr.min(), 1.0) - 0.001, max(max_rhat_arr.max(), 1.01) + 0.003)
    ax.set_xlabel("fold-fit (5 folds × 10 repeats)")
    ax.set_ylabel("worst R-hat in the fit")
    ax.set_title(title or f"Bayesian model convergence ({judge_name(model_slug)})")
    return _save(fig, f"rq4_bayesian_convergence_{model_slug}.png")


@_dark_variant
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
    """RQ5 distillation: the 3-call ensemble vs the 1-call Bayesian model on error
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
    # Signal colours: the ensemble in signal-2, the one-call Bayesian model in blue (the fix).
    colors = [_signal_color("conf_ens"), _signal_color("p_correct_bayesian")] * 2

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # errorbar can't take NaN widths, so the point-only row gets zero width.
    safe_lows = [v if np.isnan(lo) else lo for v, lo in zip(values, lows)]
    safe_highs = [v if np.isnan(hi) else hi for v, hi in zip(values, highs)]
    _draw_forest(ax, labels, values, safe_lows, safe_highs, "AUROC predicting judge error (dotted = chance)",
                 colors=colors, markers=["s", "o", "s", "o"], annotate=False, reference=0.5)
    for y, (v, lo, hi) in enumerate(zip(values, lows, highs)):
        text = f"{v:.3f} (point only)" if np.isnan(lo) else f"{v:.3f} [{lo:.3f}, {hi:.3f}]"
        ax.annotate(text, xy=(v, y), xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=8, color=_theme.note)
    ax.set_title("A single-call Bayesian model vs. a 3-prompt ensemble\n(95% cluster-bootstrap CIs)", fontsize=10)
    return _save(fig, f"rq5_distillation_{model_slug}.png")


@_dark_variant
def plot_rq5_verbose_shift(
    aleatoric_clean: float,
    aleatoric_verbose: float,
    aleatoric_gap_ci: tuple[float, float],
    epistemic_clean: float,
    epistemic_verbose: float,
    epistemic_gap_ci: tuple[float, float],
    model_slug: str,
) -> Figure:
    """Verbose-shift check: the change in mean aleatoric and epistemic entropy from
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
        colors, hollow = _neutral_styles([gap_lo], [gap_hi])
        _draw_forest(ax, ["verbose − clean"], [gap], [gap_lo], [gap_hi],
                     f"change in mean entropy, nats\n(clean {clean_val:.4f} → verbose {verbose_val:.4f})",
                     colors=colors, hollow=hollow, decimals=4)
        ax.set_title(label)
    fig.suptitle("Does the meta-model's uncertainty rise under the padding attack?\n(paired 95% cluster-bootstrap CIs)")
    return _save(fig, f"rq5_verbose_shift_{model_slug}.png")


@_dark_variant
def plot_judge_comparison(panels: list[dict], filename: str = "judge_comparison.png") -> Figure:
    """The three judges side by side, one panel per measure, one row per
    judge population (split by coverage regime or turn where the RQ split
    it). Rows are coloured by judge; a point is hollow where its interval
    includes the reference.

    Each panel: {"title", "xlabel", "reference" (x of the dotted line, or
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
            colors=[_theme.judge_colors.get(judge, _theme.muted) for judge, _, _, _, _ in rows],
            reference=panel.get("reference"),
            hollow=_includes(panel.get("reference"), [r[3] for r in rows], [r[4] for r in rows]),
        )
        ax.set_title(panel["title"], fontsize=10)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle("Three judges, same items, same tests "
                 "(95% cluster-bootstrap CIs; hollow = interval includes the dotted reference)")
    return _save(fig, filename)


# The signals drawn per judge in plot_auto_accept(); the CSV has them all.
AUTO_ACCEPT_FIGURE_SIGNALS = {
    "Qwen2.5-7B": ["conf_verb", "conf_bpe_prob", "conf_ens_prob", "p_correct_bayesian"],
    "kev-8b": ["conf_kev", "conf_kev_bpe_prob"],
    "auto-j-13b": ["conf_sc_autoj", "conf_sc_bpe_autoj_greedy_prob"],
}


@_dark_variant
def plot_auto_accept(curves: pd.DataFrame, threshold_marker: float = 0.9) -> Figure:
    """One column per judge. Top: the share of the judge's wrong verdicts an
    auto-accept pipeline lets through, against the confidence threshold it
    accepts at. Bottom: the share of all verdicts it accepts at that
    threshold - the price of a lower slip-through, since any signal reaches
    zero slip-through by accepting nothing. Solid = clean inputs, dashed =
    padded. A multi-judge figure, so colours follow the judges: red is each
    judge's naive confidence (the signal that lets errors through), the
    judge's own colour its best signal, grey and ink the ones in between.

    `curves` is analysis/auto_accept.py's curve table (point estimates).
    Saved to results/figures/auto_accept.png.
    """
    judges = [j for j in AUTO_ACCEPT_FIGURE_SIGNALS if j in set(curves["judge"])]
    fig, axes = plt.subplots(2, len(judges), figsize=(5.2 * len(judges), 9.4), sharex=True, sharey="row", squeeze=False)
    for col, judge in enumerate(judges):
        signals = AUTO_ACCEPT_FIGURE_SIGNALS[judge]
        middle = [_theme.muted, _theme.ink][: len(signals) - 2]
        colors = [_theme.accent, *middle, _theme.judge_colors[judge]]
        for color, signal in zip(colors, signals):
            for condition, linestyle in [("clean", "-"), ("verbose", DASHED)]:
                df = curves[(curves["judge"] == judge) & (curves["signal"] == signal) & (curves["condition"] == condition)]
                if df.empty:
                    continue
                label = signal_label(signal) + ("" if condition == "clean" else " (padded)")
                axes[0][col].plot(df["threshold"], df["slip_through"], color=color, linestyle=linestyle, label=label)
                axes[1][col].plot(df["threshold"], df["accepted_share"], color=color, linestyle=linestyle)
        for row in (0, 1):
            axes[row][col].axvline(threshold_marker, color=_theme.grey, **REFERENCE_LINE)
            axes[row][col].set_xlim(0.5, 1.0)
            axes[row][col].set_ylim(0, 1.02)
        axes[0][col].set_title(judge)
        # Under the bottom panel: the curves leave no empty corner in either.
        handles, labels = axes[0][col].get_legend_handles_labels()
        axes[1][col].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=7)
        axes[1][col].set_xlabel("auto-accept when confidence ≥")
    axes[0][0].set_ylabel("share of WRONG verdicts accepted unreviewed")
    axes[1][0].set_ylabel("share of ALL verdicts accepted")
    fig.suptitle("If you auto-accept confident verdicts, how many errors slip through - and how much do you accept?")
    return _save(fig, "auto_accept.png")


if __name__ == "__main__":
    for dark in (False, True):
        plot_ece_auroc_orthogonal(dark=dark)
