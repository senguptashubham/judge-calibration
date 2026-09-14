"""Matplotlib figure-producing functions, one per plot, always saved to
results/figures/ with a deterministic name (CLAUDE.md sec 5).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src.metrics import auroc_error, ece

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

    TODO(owner): construct both 6-point datasets yourself below - this is
    TASKS.md task 2.3 / LEARNING.md block C4, and the point is to build the
    intuition, not just produce the figure. Hints, not answers:
      - Example A: what pair of confidence VALUES gives perfect separation
        (every correct item's confidence beats every incorrect item's)
        while landing far from each group's own actual accuracy? (Reminder:
        confidence is bounded below at 0.5 in a forced binary choice.)
      - Example B: what happens to both metrics if EVERY item - correct and
        incorrect alike - gets the exact same confidence value?
    The asserts below will tell you immediately if your numbers don't hit
    the targets - don't silence them, fix the construction instead.

    Returns:
        The Figure (also saved to results/figures/ece_auroc_orthogonal.png).
    """
    # TODO(owner): replace with your own 6-point construction (3 correct,
    # 3 incorrect, chosen so AUROC = 1.0 and ECE ~= 0.3).
    confidences_a = np.array([0.5, 0.5, 0.9, 0.5, 0.9, 0.9])
    correct_a = np.array([0, 0, 1, 0, 1, 1])

    # TODO(owner): replace with your own 6-point construction (3 correct,
    # 3 incorrect, chosen so ECE = 0 and AUROC = 0.5).
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


if __name__ == "__main__":
    plot_ece_auroc_orthogonal()
