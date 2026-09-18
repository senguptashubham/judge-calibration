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
    labels, not worth two near-duplicate implementations.

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
    labels = list(labels)
    values_arr = np.asarray(values, dtype=float)
    ci_low_arr = np.asarray(ci_low, dtype=float)
    ci_high_arr = np.asarray(ci_high, dtype=float)

    y_pos = np.arange(len(labels))
    # errorbar wants the half-widths from the point estimate, not the
    # absolute CI bounds themselves.
    err_low = values_arr - ci_low_arr
    err_high = ci_high_arr - values_arr

    fig, ax = plt.subplots(figsize=(6, 0.9 * len(labels) + 1.5))

    ax.axvline(0, linestyle="--", color="gray", linewidth=1, zorder=1)
    ax.errorbar(
        values_arr, y_pos,
        xerr=[err_low, err_high],
        fmt="o",
        color="tab:blue",
        ecolor="tab:blue",
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


if __name__ == "__main__":
    plot_ece_auroc_orthogonal()
