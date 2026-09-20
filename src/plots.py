"""Matplotlib figure-producing functions, one per plot, always saved to
results/figures/ with a deterministic name (CLAUDE.md sec 5).
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
    """LEARNING.md block C4: two hand-built 6-point (confidence, correct)
    examples proving ECE and AUROC measure genuinely different properties -
    a signal can ace one while failing the other completely.

    Example A: ECE ~= 0.3, AUROC = 1.0. Badly miscalibrated confidence
    numbers can still coexist with perfect discrimination - the signal
    ranks every correct item above every incorrect one, it just reports the
    wrong absolute numbers while doing it. Calibration and ranking ability
    are independent properties; this is why CLAUDE.md invariant 6 requires
    ECE to always ship alongside accuracy/AUROC, never alone.

    Example B: ECE = 0, AUROC = 0.5. A signal that reports the IDENTICAL
    confidence for every single item is perfectly calibrated in aggregate
    (its one bin's mean confidence trivially equals its own accuracy) while
    being completely uninformative at telling any individual item apart
    from another. This is CLAUDE.md invariant 6's own example made
    concrete: "a judge that always says 50% has perfect ECE and zero
    usefulness."

    Example A: correct items all at confidence 0.9, incorrect items all at
    0.5 - clean separation (AUROC=1.0) with a big enough confidence gap
    (0.4 vs the forced-choice floor) to land ECE at 0.3. Example B: every
    item at the forced-choice floor (0.5) regardless of correctness - no
    ranking information at all (AUROC=0.5), but the single confidence
    value happens to equal the dataset's own base rate exactly (ECE=0).
    The asserts below catch any future edit that breaks either example -
    don't silence them, fix the construction instead.

    Returns:
        The Figure (also saved to results/figures/ece_auroc_orthogonal.png).
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
    """Per-bin (mean x, mean y, weight), binning on x - the same binning
    ece() uses internally (reuses get_bin_edges() so a diagram and its
    scalar metric are always computed on identical bins). Generic in x/y:
    a reliability diagram bins on confidence and averages correct; the
    human-disagreement figure bins on d_human and averages correct (or
    confidence) - same computation either way, just which column plays
    which role changes.
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
    """Per-bin (mean confidence, accuracy, weight) for a reliability
    diagram. Thin wrapper over _binned_means() - kept as its own name at
    reliability-diagram call sites since "confidence"/"accuracy" reads
    better there than the generic x/y.
    """
    return _binned_means(confidences, correct, n_bins, strategy)


def _marker_sizes(weights: np.ndarray) -> np.ndarray:
    """Bin weight (share of the data, in [0, 1]) -> scatter `s` (marker
    area in points^2), bounded to [40, 350] rather than a raw
    `weight * constant` scale.

    An unbounded scale breaks in two ways once a real (non-uniform) weight
    distribution shows up: a bin holding most of the data gets a marker
    whose radius, in screen points, is large enough to visually extend
    past the axes and get clipped at the boundary (matplotlib draws
    marker size in screen space, not data space, so this isn't self-
    correcting); and matplotlib's default legend handle reuses the
    scatter's own size, so the legend key becomes enormous too. Bounding
    the range keeps every marker readable and clipping-free regardless of
    how concentrated the weight distribution is - the legend uses its own
    fixed-size proxy handles instead (see plot_reliability_diagram), so
    this bound doesn't need to account for the legend at all anymore.
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
    """RQ1 (task 2.6): mean confidence vs. accuracy, per quantile bin, for
    one confidence signal - the standard reliability diagram. Points on
    the y=x diagonal are perfectly calibrated; points below it are
    overconfident (CLAUDE.md invariant 6's signed gap, visualized).

    If `correct_bidir` is given, both verdict definitions (D7) are plotted
    as two overlaid curves on the SAME figure rather than two separate
    files - `confidences` doesn't change between them (conf_verb/conf_lp/
    conf_sc/conf_bpe don't depend on which verdict definition scores
    them), only which correctness label each bin's accuracy is computed
    against - so this is the natural way to make "what does debiasing-by-
    averaging buy you" (task 2.6's framing) visible as an actual picture,
    not just two rows in a table, while still producing exactly the 4
    files (one per signal) the task's DoD asks for.

    Marker size is proportional to each bin's share of the data (its
    weight in ece()'s own weighted average) - a bin with few items sits
    on the diagonal-vs-not question just as validly, but visually it
    should read as less evidence than a bin holding most of the data.

    Args:
        confidences: stated confidence per item, in [0, 1].
        correct: whether the judge was actually right, per item (aligned
            with `confidences` - typically the `judge_verdict` definition).
        signal_name: e.g. "conf_verb" - used in the title and the saved
            filename (results/figures/reliability_{signal_name}_{model_slug}.png).
        n_bins: requested number of bins - see get_bin_edges.
        model_slug: Config.model_slug - namespaces the saved filename so a
            second judge model never overwrites the first's figure.
        strategy: one of "uniform", "quantile", "auto" (default).
        correct_bidir: optional second correctness array (the
            `verdict_bidir` definition) to overlay as a second curve.

    Returns:
        The Figure (also saved to
        results/figures/reliability_{signal_name}_{model_slug}.png).
    """
    confidences_arr = np.asarray(confidences, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    # Fixed-size proxy handles for the legend, built separately from the
    # actual (variable, weight-scaled) scatter markers - see
    # _marker_sizes()'s docstring for why reusing the real markers there
    # breaks.
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

    # A small margin beyond the data's true [0, 1] range - without it, a
    # bin whose mean confidence sits right at the edge (common; confidence
    # piles up near 1.0) gets its marker clipped by the axes border, since
    # marker size is drawn in screen points, not data units, and a point
    # exactly at the boundary has no room to render outward.
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
    """RQ2's thesis figure (task 3.2): every confidence signal's risk-
    coverage curve, overlaid with the oracle upper bound, on one axis.

    A flat curve is a finding (the signal isn't informative - abstaining
    doesn't lower risk), and the oracle overlay is what makes that legible:
    without it, there's no visual reference for how much headroom a flat
    or shallow curve is actually leaving on the table.

    Also reused as-is for task 3.2b's entropy_threshold_sweep.png (RQ5) -
    the curve shape (coverage, risk) and the oracle-overlay framing are
    identical there, just with the three ens_entropy_* signals in place of
    the four original confidence signals, hence the `filename`/`title`
    overrides rather than a second, near-duplicate plotting function.

    Args:
        curves: {signal_name: (coverage, risk)} - typically the output of
            risk_coverage() per signal, real (non-bootstrapped) data only,
            one line per signal.
        oracle: (coverage, risk) from oracle_risk_coverage() - the one
            curve every real signal must fall on or above at every
            coverage level.
        filename: saved under results/figures/{filename}.
        title: figure title.

    Returns:
        The Figure (also saved to results/figures/{filename}).
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Curves can be near-identical or exactly overlapping in places (e.g.
    # RQ5's ens_entropy_total vs ens_entropy_aleatoric, which coincide
    # almost everywhere by construction whenever epistemic ~ 0 - see
    # analysis/rq5.py's module docstring). Color alone can't disambiguate
    # two lines drawn on top of each other, so linestyle/marker are cycled
    # independently of color - a dashed line traced directly over a solid
    # one of a different color stays visually distinguishable at every
    # point, which relying on color (or opacity, which only controls how
    # much of the UNDER line shows through, not whether the OVER line
    # reads as distinct) does not guarantee.
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
    """(slope, intercept) of the OLS fit of y on x, for drawing the trend
    line overlay only - NOT imported from analysis/human_disagreement.py's
    _ols_slope(): src/ must not depend on analysis/ (analysis/ builds on
    src/, never the reverse - same layering every other module here
    follows). Small enough to duplicate the two-line slope formula rather
    than restructure the layering for it; this version additionally
    returns the intercept, which the bootstrap-focused _ols_slope() has no
    use for and deliberately doesn't compute.
    """
    x_mean, y_mean = x.mean(), y.mean()
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2))
    intercept = float(y_mean - slope * x_mean)
    return slope, intercept


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, for the plot's annotation only - NOT
    imported from analysis/human_disagreement.py's own _spearman_corr()
    for the same layering reason _ols_fit() doesn't import _ols_slope()
    (src/ must not depend on analysis/). Duplicated rather than shared
    since it's two small, self-contained formulas, not worth restructuring
    the module layering for.

    pd.Series.rank() (average method) handles d_human's heavy ties
    correctly - see analysis/human_disagreement.py's fuller explanation of
    why a plain double-argsort would be wrong here.
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
    """RQ2/task 3.3's figure (D9): does judge accuracy, and separately the
    judge's stated confidence, track human consensus strength (d_human)?

    Two panels sharing a d_human x-axis: accuracy on the left, confidence
    on the right. Each panel shows the binned real data (mean of the
    y-quantity within each d_human bin, marker size ~ bin weight) as
    scatter points, plus the OLS trend line the regression itself tested,
    as a separate, deliberately differently-styled element - dashed and a
    third color, not just relying on the scatter/line distinction, per the
    lesson from RQ2/RQ5's risk-coverage figures: two elements that could
    visually coincide (here, if the binned means happen to fall right on
    the fit line) need more than color to stay legible, so linestyle and
    marker carry the distinction too, not opacity alone.

    d_human is naturally a low-cardinality signal (a handful of distinct
    values arise from typical small per-item vote counts), so get_bin_edges'
    "auto" strategy (reused via _binned_means) is likely to bin it by exact
    value here rather than falling back to quantile bins - the same
    "exact, not a fallback" property ece() already relies on for conf_sc.

    Args:
        d_human: |frac_prefer_a - 0.5| per item, in [0, 0.5].
        correct: whether the judge was actually right, per item.
        confidence: conf_verb per item (the one signal this figure uses -
            see analysis/human_disagreement.py's module docstring for why).
        n_bins: requested number of bins for the binned view (see
            get_bin_edges - may bin exactly, not just approximately).
        model_slug: Config.model_slug - namespaces the saved filename so a
            second judge model never overwrites the first's figure.

    Returns:
        The Figure (also saved to
        results/figures/human_disagreement_{model_slug}.png).
    """
    # Rounded to 6dp before binning: d_human = |frac_prefer_a - 0.5| computed
    # from small vote-count fractions (e.g. 1/3 vs 2/3) can land on adjacent
    # float64 values for the SAME true fraction (verified on real data:
    # abs(1/3-0.5) and abs(2/3-0.5) differ by ~6e-17, both "really" 1/6) -
    # get_bin_edges' exact-value branch uses np.unique()'s bitwise equality,
    # so without rounding this silently doubles a bin that should be one,
    # splitting its accuracy across two near-identical x positions instead
    # of averaging it. 6dp is far below the real spacing between distinct
    # d_human values (>= 0.05 apart) and far above float64 noise (~1e-16).
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

        # Spearman rho isn't a line in (d_human, y) space - it's a unitless
        # rank-correlation summary, so it's reported as text, not a second
        # plotted curve that would imply a shape it doesn't have.
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
) -> None:
    """Draws one forest/coefficient panel - point estimate + CI error bar
    per label, plus a reference line at 0 - onto an existing `ax`. No
    title, no savefig: this is the shared drawing primitive both
    _forest_plot() (one panel, own figure) and plot_rq3b_deltas() (two
    panels, one figure) build on, so the panel layout exists in exactly
    one place.

    Args:
        ax: the Axes to draw on.
        labels: category names, in display order (top to bottom).
        values: point estimate per label, same order.
        ci_low, ci_high: CI bounds per label, same order.
        xlabel: x-axis label (what the point estimates measure).
        colors: optional per-label color (e.g. one color per feature
            family, task 5.9's plot_rq4_coefficients). Defaults to a
            single "tab:blue" for every point - every existing caller
            omits this and is unaffected.
    """
    labels = list(labels)
    values_arr = np.asarray(values, dtype=float)
    ci_low_arr = np.asarray(ci_low, dtype=float)
    ci_high_arr = np.asarray(ci_high, dtype=float)
    point_colors = list(colors) if colors is not None else ["tab:blue"] * len(labels)

    y_pos = np.arange(len(labels))
    # errorbar wants the half-widths from the point estimate, not the
    # absolute CI bounds themselves.
    err_low = values_arr - ci_low_arr
    err_high = ci_high_arr - values_arr

    ax.axvline(0, linestyle="--", color="gray", linewidth=1, zorder=1)
    # Drawn one point at a time (rather than one vectorized ax.errorbar
    # call) SPECIFICALLY so each point can take its own color - a single
    # errorbar() call only accepts one color for the whole series.
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

    # Exact-value text labels, not just the visual point+whisker: when one
    # label's magnitude dwarfs the others (e.g. task 4.3's conf_bpe, whose
    # gap is ~30x conf_verb's), the small-but-real estimates collapse to a
    # dot with an invisible error bar at this axis scale - the number
    # stays legible even where the geometry doesn't. Placed above each
    # point, not to the side, so the label never competes with the CI
    # whiskers or gets clipped at the axis edge for an extreme value.
    for x, y, lo, hi in zip(values_arr, y_pos, ci_low_arr, ci_high_arr):
        ax.annotate(
            f"{x:.3f} [{lo:.3f}, {hi:.3f}]",
            xy=(x, y), xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=8, color="dimgray",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    # Extra headroom above the top point and below the bottom one -
    # without it, the topmost label's value annotation (offset 10 points
    # above its marker) gets clipped by the axes border itself.
    ax.set_ylim(len(labels) - 0.5, -0.75)
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
    """Generic forest/coefficient plot: one point estimate + CI error bar
    per label, plus a reference line at 0, labels on the y-axis in reading
    order top-to-bottom. The standard visualization for "several point
    estimates with CIs, compared against a null value" - simple, no new
    dependencies, no randomness (unlike the raw-point jitter idea
    considered for the human-disagreement scatter figure and deliberately
    skipped there for adding complexity without adding information). Here
    the plot adds real legibility a markdown table doesn't: which CIs
    cross the zero reference line is immediate, not something a reader
    has to check bracket-by-bracket.

    Shared by task 3.4's signal-vs-d_human correlations
    (plot_d_human_correlations) and task 4.3's flipped-vs-unflipped
    confidence gap (plot_rq3a_confidence_gap) - same shape (N signals,
    each one point estimate + CI against zero), different data and axis
    labels, not worth two near-duplicate implementations. The actual panel
    drawing lives in _draw_forest(), shared again with task 4.4's two-panel
    plot_rq3b_deltas().

    Args:
        labels: category names, in display order (top to bottom).
        values: point estimate per label, same order.
        ci_low, ci_high: CI bounds per label, same order.
        xlabel: x-axis label (what the point estimates measure).
        title: figure title.
        filename: saved under results/figures/{filename}.

    Returns:
        The Figure (also saved to results/figures/{filename}).
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
) -> Figure:
    """Task 3.4's figure: a forest plot of the four signals' Spearman rho
    against d_human. Thin wrapper over _forest_plot() - see that
    docstring for the shared rationale/mechanics.

    Args:
        signals: signal names, in display order (top to bottom).
        spearman: point estimate per signal, same order.
        ci_low, ci_high: CI bounds per signal, same order.
        model_slug: Config.model_slug - namespaces the saved filename so a
            second judge model never overwrites the first's figure.

    Returns:
        The Figure (also saved to
        results/figures/d_human_correlations_{model_slug}.png).
    """
    return _forest_plot(
        labels=signals,
        values=spearman,
        ci_low=ci_low,
        ci_high=ci_high,
        xlabel="Spearman ρ (signal vs. d_human)",
        title="Signal-vs-d_human correlations (task 3.4)",
        filename=f"d_human_correlations_{model_slug}.png",
    )


def plot_rq3a_confidence_gap(
    signals: list[str],
    gap: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    model_slug: str,
) -> Figure:
    """Task 4.3's figure: a forest plot of the four signals'
    flipped-minus-unflipped confidence gap (mean confidence on items where
    the AB/BA order changed the verdict, minus mean confidence on items
    where it didn't). Thin wrapper over _forest_plot() - see that
    docstring for the shared rationale/mechanics.

    A negative gap with a CI excluding 0 means the signal IS picking up on
    its own position-bias-induced errors (lower confidence exactly when
    the order flipped the verdict); a CI crossing 0 means it isn't.

    Args:
        signals: signal names, in display order (top to bottom).
        gap: point estimate (mean_flipped - mean_unflipped) per signal,
            same order.
        ci_low, ci_high: CI bounds per signal, same order.
        model_slug: Config.model_slug - namespaces the saved filename so a
            second judge model never overwrites the first's figure.

    Returns:
        The Figure (also saved to
        results/figures/rq3a_confidence_gap_{model_slug}.png).
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
    """Task 4.4's figure: two forest panels side by side - ΔECE and
    ΔAUROC, both verbose-minus-clean, for the three signals RQ3b scores
    (`conf_sc` dropped, D21). Two panels rather than two separate files:
    the headline finding is a joint one (AUROC drops for all three,
    ECE only breaks for conf_bpe specifically), which only reads as one
    finding when both panels share a figure and a signal ordering, not as
    two tables a reader has to cross-reference by eye.

    A positive ΔECE means the judge is MORE miscalibrated under verbose;
    a negative ΔAUROC means its uncertainty signal is LESS informative
    about its own errors under verbose. Both panels share _draw_forest()
    with plot_rq3a_confidence_gap()/plot_d_human_correlations() - same
    point+CI-vs-zero shape, just two of them on one figure instead of one.

    Args:
        signals: signal names, in display order (top to bottom), shared
            by both panels.
        delta_ece, ece_ci_low, ece_ci_high: ECE panel's point estimate and
            CI bounds per signal, same order as `signals`.
        delta_auroc, auroc_ci_low, auroc_ci_high: AUROC panel's point
            estimate and CI bounds per signal, same order.
        model_slug: Config.model_slug - namespaces the saved filename so a
            second judge model never overwrites the first's figure.

    Returns:
        The Figure (also saved to
        results/figures/rq3b_deltas_{model_slug}.png).
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
    """Task 5.5's figure: grouped bar chart, one bar per (tier, model)
    combination, PLUS a standalone bar for the best-single-signal
    baseline (recomputed on the SAME human_agreed population the
    ablation itself uses - see analysis/rq4.py's own module docstring
    for why reusing RQ2's stored number would compare across two
    different populations) in its own distinct color, so "does combining
    signals into a tier actually beat the single best signal alone" is a
    direct bar-to-bar height comparison, not a bar-vs-line one. A thin
    reference line at the same height is kept too, so that comparison
    stays easy even for the tiers sitting furthest from the baseline bar.

    Bars are grouped by tier (A/B/C) on the x-axis, colored by model -
    this reading order puts "does the next tier beat the last one" (the
    RQ4 headline question) directly adjacent on the page, with "does
    either model beat the cheap single-signal baseline" answered by
    whether a bar clears the baseline bar's own height.

    Args:
        tiers: tier label per bar, e.g. ["A","B","C","A","B","C"].
        models: model label per bar, same order/length as `tiers`, e.g.
            ["logreg"]*3 + ["histgbm"]*3.
        auroc_mean: point estimate per bar (mean across the 10 D8
            repeats), same order.
        auroc_low, auroc_high: the across-repeat spread bounds per bar
            (D8's own headline-uncertainty convention - NOT a bootstrap
            CI), same order.
        baseline: the best single signal's AUROC on this task's own
            population.
        baseline_ci_low, baseline_ci_high: that baseline's own CI
            (cluster-bootstrap, matching RQ2's convention).
        baseline_label: which signal the baseline is (e.g. "conf_bpe"),
            used as that bar's own x-axis label and legend entry.
        model_slug: Config.model_slug - namespaces the saved filename so
            a second judge model never overwrites the first's figure.

    Returns:
        The Figure (also saved to results/figures/rq4_ablation_{model_slug}.png).
    """
    tier_order = ["A", "B", "C"]
    model_order = sorted(set(models))
    bar_width = 0.8 / max(len(model_order), 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Thin reference line at the baseline's height, threaded across the
    # whole plot - a cheap way to judge clearance for the tiers sitting
    # far from the baseline's own bar, without competing visually with it
    # (thinner + no shaded band, since the bar itself now carries the CI).
    ax.axhline(baseline, linestyle="--", color="tab:gray", linewidth=1, zorder=1)

    # The baseline bar itself sits at x=0, in its own distinct color (not
    # reused from the model color cycle below) so it reads as "a single
    # signal alone", never mistaken for a third model.
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

        # Reorders this model's own (tier, value) rows into tier_order -
        # the caller's row order isn't assumed to already be tier-sorted.
        by_tier = dict(zip(model_tiers, zip(model_means, model_low, model_high)))
        ordered = [by_tier[t] for t in tier_order if t in by_tier]
        means = np.array([v[0] for v in ordered])
        err_low = means - np.array([v[1] for v in ordered])
        err_high = np.array([v[2] for v in ordered]) - means

        # +1 shifts every tier group one slot right of the baseline bar
        # at x=0.
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
    """Task 5.5's paired-comparison figure: forest plot of each tier-
    progression step's AUROC delta (baseline->A, A->B, B->C, per model -
    analysis/rq4.py::compare_tier_progression()), against a zero
    reference line. Thin wrapper over _forest_plot() - same point+CI-vs-
    zero shape as plot_rq3a_confidence_gap()/plot_d_human_correlations().

    A CI excluding 0 means that step's change is real. Every CI crossing
    0 (the actual result, 18 Sep 2026) means the ablation bar chart's
    apparent A > B > C decline does not survive a paired test - this
    figure is what makes that visible at a glance, instead of reading it
    off a 6-row CSV.

    Args:
        comparisons: label per row, e.g. "logreg: A - baseline", in
            display order (top to bottom).
        auroc_diff: point estimate per row (higher tier's AUROC minus
            lower tier's), same order.
        ci_low, ci_high: CI bounds per row, same order.
        model_slug: Config.model_slug - namespaces the saved filename so
            a second judge model never overwrites the first's figure.

    Returns:
        The Figure (also saved to
        results/figures/rq4_progression_{model_slug}.png).
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
    """Task 5.5's permutation-null sanity-check figure: one small
    histogram per (tier, model) cell's null AUROC distribution
    (analysis/rq4.py::compute_permutation_null_summary()), with the real,
    observed AUROC marked as a vertical line. Less essential than
    plot_rq4_progression() - the result here is unambiguous (every cell
    at the 100th percentile) - but a visual confirmation that each null
    genuinely centers near 0.5 (not just a printed mean) is cheap and
    catches a leaking pipeline at a glance, the same role the numeric
    check already serves in text.

    Args:
        tiers: tier label per cell, e.g. ["A","A","B","B","C","C"] - any
            input order accepted, the grid below re-sorts into columns
            by tier regardless (so a caller's tier-major or model-major
            list order never changes the figure).
        models: model label per cell, same order/length as `tiers`.
        observed: this cell's real (unshuffled) AUROC, same order.
        null_distributions: this cell's null_aurocs array (n=50 values
            each, compute_permutation_null_summary()'s own `n`), same
            order.
        model_slug: Config.model_slug - namespaces the saved filename so
            a second judge model never overwrites the first's figure.

    Returns:
        The Figure (also saved to
        results/figures/rq4_permutation_nulls_{model_slug}.png).
    """
    # Grid is (model) rows x (tier) columns - every model's panels read
    # left-to-right as A->B->C on one row, and every tier's two models
    # stack in one column, rather than a flat sequential fill (which
    # mixed tiers and models across a row with no visual grouping).
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
    """Task 5.6's optional figure (requested 19 Sep 2026, after the
    numeric interaction result): predicted P(correct) as a function of
    the predictor's out-of-fold score, one curve per distinct d_human
    level actually present in the data (analysis/rq4.py's
    compute_h4_interaction_curves() - NOT a min/median/max summary,
    which collapses under this population's real skew).

    TWO panels, not one - checked empirically (19 Sep 2026) that the
    probability panel alone undersells the fitted interaction (+2.3752
    [1.6849, 3.0739]): the model's log-odds slope w.r.t. oof_score
    genuinely increases with d_human (that's what the positive
    interaction coefficient means, directly), but in probability space
    that gets compressed by sigmoid saturation specifically in the
    high-oof_score region where most of this project's real data sits
    (most judge calls are high-confidence) - higher-d_human curves sit
    closer to the probability ceiling there, where the sigmoid is
    flattest, visually muting a slope difference that reads as large and
    unambiguous on the log-odds scale, where the model is literally
    linear and the interaction coefficient IS the slope difference,
    undistorted.

    Left panel: probability space (intuitive - P(correct) is directly
    meaningful) with a rug plot of the real oof_score values along the
    bottom, so the curves aren't read as equally well-supported across
    their full domain. Right panel: log-odds (the linear predictor) -
    same three lines, undistorted, visibly diverging in slope as
    d_human rises. Together: "here's what it means" and "here's why the
    number says it's real," not two redundant views of one thing.

    Args:
        oof_score: the real, per-item averaged OOF scores (for the left
            panel's rug plot only, not the curves themselves).
        oof_score_grid: shared x-axis grid every curve is evaluated on.
        d_human_values: the distinct d_human levels, ascending - one
            legend entry each, shared across both panels.
        predicted_curves: one P(correct) array per d_human_values entry,
            same order, each the same length as oof_score_grid.
        log_odds_curves: the same curves on the linear-predictor scale,
            same order/length.
        model_slug: Config.model_slug - namespaces the saved filename so
            a second judge model never overwrites the first's figure.

    Returns:
        The Figure (also saved to
        results/figures/h4_interaction_{model_slug}.png).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(d_human_values) - 1, 1)) for i in range(len(d_human_values))]

    ax = axes[0]
    for color, d_human_value, curve in zip(colors, d_human_values, predicted_curves):
        ax.plot(oof_score_grid, curve, color=color, linewidth=2.5, label=f"d_human = {d_human_value:.3f}")
    # Rug: real oof_score values along the bottom, outside the [0,1]
    # probability axis so it never overlaps the curves themselves.
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
    """Task 5.7's figure: grouped bar chart, one group per model
    (logreg/histgbm), each group showing two bars - in-domain (trained
    AND tested on clean, D8's repeated-CV spread) vs. transfer (trained
    on clean, frozen, evaluated on verbose, cluster-bootstrap CI) - same
    grouped-bars-plus-whiskers shape as plot_rq4_ablation, just grouped
    by model instead of by tier.

    Makes the actual 20 Sep 2026 result legible at a glance: logreg's
    two bars land at essentially the same height (ΔAUROC +0.0001) while
    histgbm's transfer bar sits visibly, though not dramatically, below
    its in-domain bar (ΔAUROC -0.0254) - refuting, not confirming, "the
    safety net degrades under attack" as a clean, dramatic story.

    Args:
        models: model name per group, e.g. ["logreg", "histgbm"].
        baseline_mean, baseline_low, baseline_high: in-domain AUROC and
            its D8 across-repeat spread, same order as `models`.
        transfer_mean, transfer_low, transfer_high: transfer AUROC and
            its cluster-bootstrap CI, same order.
        model_slug: Config.model_slug - namespaces the saved filename so
            a second judge model never overwrites the first's figure.

    Returns:
        The Figure (also saved to
        results/figures/rq4_transfer_{model_slug}.png).
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
    """Task 5.8's figure: grouped bar chart, one group per MT-Bench
    category, one bar per model within each group - the held-out AUROC
    from analysis/rq4.py::compute_category_held_out_auroc()'s
    LeaveOneGroupOut protocol.

    Categories are sorted WEAKEST TO STRONGEST (by mean AUROC across
    models), not alphabetically - so "which categories does the
    predictor struggle on" reads directly off the x-axis without
    needing to scan a table. No category is singled out for highlight -
    an earlier version flagged "coding" (PLAN.md §2.4's own illustrative
    phrasing, "is it learning 'coding questions are hard'"), but that
    turned out to be an unexamined example carried over into the plan
    text, not a reasoned hypothesis about this specific judge/dataset
    (20 Sep 2026 discussion) - singling it out visually would have kept
    presenting it as the headline question when the plot itself is a
    more honest, unprejudiced answer without picking a side beforehand.

    A dashed reference line at 0.5 (chance) gives a visual floor - the
    real story here is that every category clears it comfortably, no
    category collapses to chance the way a "some categories are
    unlearnable" failure would look.

    Args:
        categories: category label per row, e.g. 8 values repeated
            once per model (16 rows for 2 models).
        models: model label per row, same order/length as `categories`.
        auroc: held-out AUROC per row, same order.
        model_slug: Config.model_slug - namespaces the saved filename so
            a second judge model never overwrites the first's figure.

    Returns:
        The Figure (also saved to
        results/figures/rq4_category_transfer_{model_slug}.png).
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


def plot_rq4_coefficients(
    features: list[str],
    coef: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    family: list[str],
    model_slug: str,
) -> Figure:
    """Task 5.9's headline figure - DoD's own words: "the coefficients
    are the result, more than the AUROC is." One row per Tier C
    (encoded) feature - the fitted LogisticRegression(C=1.0) coefficient
    on the standardized scale, with its cluster-bootstrap CI
    (analysis/rq4.py::bootstrap_coefficient_cis, question_id, B=2000,
    REFITTING each resample - never a statsmodels/sklearn default SE,
    invariant 2) - plus a reference line at 0.

    Colored by feature FAMILY (A/B/C, analysis/rq4.py::_feature_family)
    rather than left uniform: this is the direct, per-feature answer to
    "which feature family carries the signal" (PLAN.md §2.2's reframe) -
    5.5 already found the tiers statistically indistinguishable at the
    whole-model AUROC level (every paired-progression CI crossed 0), so
    seeing whether Tier B/C's own coefficients individually clear 0 (or
    sit near it while Tier A's don't) is the sharper, complementary
    reading this coefficient-level view can give that the tier-level
    ablation couldn't.

    Rows are grouped by family (A, then B, then C) and sorted by
    |coefficient| descending WITHIN each family block - so the largest,
    most legible effects in each family read first, rather than an
    arbitrary or alphabetical feature order.

    Reuses _draw_forest() (task 4.3/5.5's shared forest-panel drawing
    primitive) with its new optional per-point `colors` - the drawing
    logic itself doesn't change, only that this caller passes a color.

    Args:
        features: encoded feature name per row (predictor.py::
            encode_features()'s column names - one-hot category dummies
            included).
        coef: fitted coefficient per row, same order.
        ci_low, ci_high: cluster-bootstrap CI bounds per row, same order.
        family: "A"/"B"/"C" per row, same order.
        model_slug: Config.model_slug - namespaces the saved filename so
            a second judge model never overwrites the first's figure.

    Returns:
        The Figure (also saved to
        results/figures/rq4_coefficients_{model_slug}.png).
    """
    df = pd.DataFrame(
        {"feature": features, "coef": coef, "ci_low": ci_low, "ci_high": ci_high, "family": family}
    )
    df["abs_coef"] = df["coef"].abs()
    df["family"] = pd.Categorical(df["family"], categories=["A", "B", "C"], ordered=True)
    # _draw_forest() renders list index 0 at the TOP of the figure (its
    # own set_ylim puts y=0 nearest the top) - so this sort order
    # (family ascending, |coef| descending within family) is already
    # "Tier A first, largest-magnitude-first within each block, reading
    # top to bottom" with no further reversal needed.
    df = df.sort_values(["family", "abs_coef"], ascending=[True, False]).reset_index(drop=True)

    colors = [_FEATURE_FAMILY_COLORS[f] for f in df["family"]]

    # 0.9in/row, matching _forest_plot()'s own established spacing - with
    # ~37 Tier C features this makes for a tall figure, but the per-point
    # value annotation (_draw_forest's own) needs that much room to stay
    # legible and non-overlapping at this row count.
    fig, ax = plt.subplots(figsize=(7, 0.9 * len(df) + 1.5))
    _draw_forest(ax, df["feature"].tolist(), df["coef"].to_numpy(), df["ci_low"].to_numpy(), df["ci_high"].to_numpy(), "coefficient (standardized scale)", colors=colors)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=_FEATURE_FAMILY_COLORS[f], label=f"Tier {f}")
        for f in ["A", "B", "C"]
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    ax.set_title(f"RQ4 meta-model coefficients, Tier C + logreg (task 5.9, {model_slug})")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"rq4_coefficients_{model_slug}.png", dpi=150)
    return fig


if __name__ == "__main__":
    plot_ece_auroc_orthogonal()
