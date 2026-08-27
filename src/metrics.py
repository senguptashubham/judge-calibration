"""Calibration and agreement metrics. See TASKS.md tasks 0.4, 0.6, 2.5, 3.1."""

from collections.abc import Sequence

import numpy as np


def ece(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int,
    strategy: str = "auto",
) -> tuple[float, int]:
    r"""Expected Calibration Error: a weighted average of the gap between
    stated confidence and actual accuracy, computed within bins.

    $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

    Confidence is partitioned into bins (see `get_bin_edges` for the three
    strategies), and within each bin `m`: `acc(B_m)` is the mean of `correct`
    for items in that bin, `conf(B_m)` is the mean confidence in that bin,
    and `|B_m|/n` is that bin's share of all `n` items.

    Why quantile beats uniform here: verbalized confidence (and the other
    signals in this project) piles up near 0.8/0.9/0.95/1.0 rather than
    spreading evenly across [0, 1]. Equal-width ("uniform") bins would leave
    most bins nearly empty and cram almost all the data into one or two bins
    near 1.0, making the "average" gap reported for those crowded bins an
    unstable estimate. Equal-mass ("quantile") bins guarantee every bin
    holds a comparable number of items, wherever the data is concentrated.

    Why exact unique-value binning is not a fallback for a discrete signal:
    binning only exists to approximate a continuous distribution, by
    grouping similar-but-not-identical values together. When a signal
    already has few distinct values (e.g. `conf_sc` at `k_sc=4` can only be
    0, 0.25, 0.5, 0.75, or 1.0), there is nothing left to approximate -
    grouping by the exact value gives the true accuracy at that exact
    confidence level, with zero binning error. That is strictly better
    information than any coarser quantile or uniform binning could produce,
    not a compromise made because binning "didn't work."

    Args:
        confidences: stated confidence per item, in [0, 1].
        correct: whether the judge was actually right, per item.
        n_bins: requested number of bins. May not match n_effective_bins -
            see get_bin_edges.
        strategy: one of "uniform", "quantile", "auto" (default).

    Returns:
        (ece, n_effective_bins) - the actual number of bins used, which can
        differ from n_bins (duplicate quantile edges get dropped, and the
        discrete branch uses exactly as many bins as there are unique
        values).
    """
    confidences_arr = np.asarray(confidences, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)
    n = len(confidences_arr)

    edges = get_bin_edges(confidences_arr, n_bins, strategy)
    bin_labels = np.digitize(confidences_arr, edges)

    weighted_gaps = []
    for bin_label in np.unique(bin_labels):
        mask = bin_labels == bin_label
        count = mask.sum()
        mean_confidence = confidences_arr[mask].mean()
        accuracy = correct_arr[mask].mean()
        weighted_gaps.append(count * abs(accuracy - mean_confidence))

    ece_value = float(sum(weighted_gaps) / n)
    n_effective_bins = len(weighted_gaps)
    return ece_value, n_effective_bins


def get_bin_edges(confidences: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Compute bin edges for ece(), per DECISIONS.md D14.

    - "uniform": n_bins equal-width bins spanning [0, 1].
    - "quantile": n_bins equal-mass bins, edges taken from the confidence
      distribution's own percentiles. Duplicate edges (from ties in the
      data) are dropped deterministically, which can yield fewer than
      n_bins bins - see _quantile_edges.
    - "auto" (ece()'s default): if there are at most n_bins distinct
      confidence values, bin by exact value (see ece()'s docstring for why
      this is exact, not a fallback); otherwise, falls back to the same
      quantile logic as "quantile".

    Every branch nudges its top edge up by a tiny epsilon, because
    np.digitize is half-open on the right ([edges[i-1], edges[i])) - without
    this, an item with confidence exactly equal to the maximum edge would
    overflow past the last bin instead of landing inside it.
    """
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_edges[-1] += 1e-9
    elif strategy == "quantile":
        bin_edges = _quantile_edges(confidences, n_bins)
    elif strategy == "auto":
        unique_vals = np.sort(np.unique(confidences))
        if len(unique_vals) <= n_bins:
            # k unique values need k+1 edges to become k separate bins via
            # digitize, so this appends one edge past the max - it does not
            # just nudge the existing last edge (that would merge the top
            # two values into one bin instead).
            bin_edges = np.append(unique_vals, unique_vals[-1] + 1e-5)
        else:
            bin_edges = _quantile_edges(confidences, n_bins)
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")
    return bin_edges


def _quantile_edges(confidences: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-mass bin edges with duplicate edges dropped deterministically.

    Ties in the data make np.percentile return repeated edge values.
    np.unique collapses them - never randomly jittered, per CLAUDE.md sec 5's
    ban on bare np.random.* calls, and because a metric used in REPORT.md
    must be deterministic.
    """
    raw_edges = np.percentile(confidences, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(raw_edges)
    edges[-1] += 1e-9
    return edges
