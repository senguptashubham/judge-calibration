"""Matplotlib figure-producing functions, one per plot, always saved to
results/figures/ with a deterministic name (CLAUDE.md sec 5).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

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


def _reliability_points(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int, strategy: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (mean confidence, accuracy, weight) for a reliability
    diagram - the same binning ece() uses internally (reuses
    get_bin_edges() so a diagram and its ECE number are always computed
    on identical bins), just returned per-bin instead of collapsed to one
    scalar.
    """
    edges = get_bin_edges(confidences, n_bins, strategy)
    bin_labels = np.digitize(confidences, edges)
    n = len(confidences)

    bin_conf, bin_acc, bin_weight = [], [], []
    for bin_label in np.unique(bin_labels):
        mask = bin_labels == bin_label
        bin_conf.append(confidences[mask].mean())
        bin_acc.append(correct[mask].mean())
        bin_weight.append(mask.sum() / n)
    return np.array(bin_conf), np.array(bin_acc), np.array(bin_weight)


def plot_reliability_diagram(
    confidences: np.ndarray,
    correct: np.ndarray,
    signal_name: str,
    n_bins: int,
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
            filename (results/figures/reliability_{signal_name}.png).
        n_bins: requested number of bins - see get_bin_edges.
        strategy: one of "uniform", "quantile", "auto" (default).
        correct_bidir: optional second correctness array (the
            `verdict_bidir` definition) to overlay as a second curve.

    Returns:
        The Figure (also saved to
        results/figures/reliability_{signal_name}.png).
    """
    confidences_arr = np.asarray(confidences, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")

    conf_pts, acc_pts, weights = _reliability_points(confidences_arr, correct_arr, n_bins, strategy)
    ax.plot(conf_pts, acc_pts, color="tab:blue", alpha=0.5, zorder=1)
    ax.scatter(
        conf_pts, acc_pts, s=weights * 2000, alpha=0.8, color="tab:blue", label="judge_verdict", zorder=2
    )

    if correct_bidir is not None:
        correct_bidir_arr = np.asarray(correct_bidir, dtype=float)
        conf_pts2, acc_pts2, weights2 = _reliability_points(confidences_arr, correct_bidir_arr, n_bins, strategy)
        ax.plot(conf_pts2, acc_pts2, color="tab:orange", alpha=0.5, zorder=1)
        ax.scatter(
            conf_pts2, acc_pts2, s=weights2 * 2000, alpha=0.8, color="tab:orange", label="verdict_bidir", zorder=2
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.set_title(f"Reliability diagram: {signal_name}")
    ax.legend(loc="upper left")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"reliability_{signal_name}.png", dpi=150)
    return fig


if __name__ == "__main__":
    plot_ece_auroc_orthogonal()
