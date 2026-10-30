"""Calibration, discrimination, agreement, and selective-prediction metrics."""

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score


def truncated_entropy(logprobs: npt.ArrayLike) -> float:
    """Entropy of one token position's top-K distribution, renormalized
    over just those K tokens (K=20, D4):

        p_i = exp(logprob_i) / sum_j exp(logprob_j)
        H   = -sum_i p_i * log(p_i)

    NOT the full-vocabulary entropy: vLLM reports only the K most likely
    tokens, and discarding the mass outside them systematically
    *underestimates* the true entropy. Hence "truncated" in every name that
    uses it - never call it predictive entropy.
    """
    logprobs_arr = np.asarray(logprobs, dtype=float)
    probs = np.exp(logprobs_arr)
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)))


def ece(
    confidences: npt.ArrayLike,
    correct: npt.ArrayLike,
    n_bins: int,
    strategy: str = "auto",
) -> tuple[float, int]:
    r"""Expected Calibration Error: the bin-weighted gap between stated
    confidence and actual accuracy.

    $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

    Why quantile bins beat uniform ones here: confidence piles up near
    0.8/0.9/0.95/1.0, so equal-width bins leave most bins nearly empty and
    cram the data into one or two, making the estimate unstable. Equal-mass
    bins give every bin a comparable number of items.

    Why unique-value binning is exact, not a fallback: binning exists to
    approximate a continuous distribution. A signal with few distinct
    values (conf_sc at k_sc=4 takes only 0, 0.25, 0.5, 0.75, 1.0) needs no
    approximation - grouping by exact value gives the true accuracy at each
    confidence level, with zero binning error.

    Args:
        confidences: stated confidence per item, in [0, 1].
        correct: whether the judge was right, per item.
        n_bins: requested bin count (see get_bin_edges).
        strategy: "uniform", "quantile", or "auto" (default, D14).

    Returns:
        (ece, n_effective_bins) - the bins actually used, which can differ
        from n_bins: duplicate quantile edges are dropped, and the discrete
        branch uses one bin per unique value. Always report it (invariant 4).
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
    """Bin edges for ece()/mce()/brier_decomposition() (D14).

    - "uniform": n_bins equal-width bins over [0, 1].
    - "quantile": n_bins equal-mass bins from the data's percentiles, with
      duplicate edges dropped (can yield fewer than n_bins).
    - "auto": one bin per unique value if there are at most n_bins of them,
      otherwise "quantile".

    The top edge is always nudged up by a small epsilon: np.digitize bins
    are half-open on the right, so a value equal to the maximum edge would
    otherwise fall outside the last bin.
    """
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_edges[-1] += 1e-9
    elif strategy == "quantile":
        bin_edges = _quantile_edges(confidences, n_bins)
    elif strategy == "auto":
        unique_vals = np.sort(np.unique(confidences))
        if len(unique_vals) <= n_bins:
            # k unique values need k+1 edges to become k bins, so append one
            # edge past the max rather than nudging the last one (which
            # would merge the top two values).
            bin_edges = np.append(unique_vals, unique_vals[-1] + 1e-5)
        else:
            bin_edges = _quantile_edges(confidences, n_bins)
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")
    return bin_edges


def _quantile_edges(confidences: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-mass edges, duplicates dropped deterministically (never
    jittered - a reported metric must be deterministic).
    """
    raw_edges = np.percentile(confidences, np.linspace(0, 100, n_bins + 1))
    edges = np.unique(raw_edges)
    edges[-1] += 1e-9
    return edges


def cohens_kappa(
    a: npt.ArrayLike, b: npt.ArrayLike, weights: npt.ArrayLike | None = None
) -> float:
    """Cohen's kappa: chance-corrected agreement between two raters.

        kappa = (p_o - p_e) / (1 - p_e)
        p_o   = fraction of items where a and b agree
        p_e   = sum over labels c of P_a(c) * P_b(c)

    With `weights`, every fraction is a weighted share (item i counts
    w_i). coverage_weights() uses this to score the expected kept set
    when a coverage cut falls inside a tied group: kappa of the expected
    confusion matrix.

    p_e is the agreement expected from each rater's own label frequencies
    alone. When both raters share a bias (both mostly say "A"), p_e is high
    and kappa deflates the raw agreement accordingly - the reason raw
    agreement overstates judge ability on MT-Bench (invariant 5).

    Returns kappa in [-1, 1]: 1 = perfect, 0 = chance, negative = worse
    than chance. Returns 0.0 in the degenerate case where both raters use
    one single label throughout (p_e = 1, so the formula is 0/0).
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    w = None if weights is None else np.asarray(weights, dtype=float)

    def share(mask: np.ndarray) -> float:
        return float(np.mean(mask)) if w is None else float(np.sum(w * mask) / w.sum())

    p_o = share(a_arr == b_arr)

    labels = np.union1d(np.unique(a_arr), np.unique(b_arr))
    p_e = 0.0
    for label in labels:
        p_e += share(a_arr == label) * share(b_arr == label)

    if p_e >= 1.0 - 1e-12:
        return 0.0

    return (p_o - p_e) / (1 - p_e)


def overconfidence_gap(confidences: npt.ArrayLike, correct: npt.ArrayLike) -> float:
    """Signed gap = mean(confidence) - mean(correct). Positive = overconfident.

    ECE takes |acc - conf| per bin, so a judge overconfident in one bin and
    underconfident in another can post a small ECE. This signed, unbinned
    number can't cancel that way (invariant 6).
    """
    confidences_arr = np.asarray(confidences, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)
    return float(np.mean(confidences_arr) - np.mean(correct_arr))


def mce(
    confidences: npt.ArrayLike,
    correct: npt.ArrayLike,
    n_bins: int,
    strategy: str = "auto",
) -> tuple[float, int]:
    """Maximum Calibration Error: MCE = max_m |acc(B_m) - conf(B_m)|, on the
    same bins as ece(). ECE says how bad the typical bin is; MCE says how
    bad the worst one is, which ECE can average away.

    Returns (mce, n_effective_bins), mirroring ece().
    """
    confidences_arr = np.asarray(confidences, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)

    edges = get_bin_edges(confidences_arr, n_bins, strategy)
    bin_labels = np.digitize(confidences_arr, edges)

    bin_gaps = []
    for bin_label in np.unique(bin_labels):
        mask = bin_labels == bin_label
        mean_confidence = confidences_arr[mask].mean()
        accuracy = correct_arr[mask].mean()
        bin_gaps.append(abs(accuracy - mean_confidence))

    mce_value = float(max(bin_gaps))
    n_effective_bins = len(bin_gaps)

    return mce_value, n_effective_bins


def brier(confidences: npt.ArrayLike, correct: npt.ArrayLike) -> float:
    """Brier score: BS = mean((confidence_i - correct_i)^2), lower is better.

    A strictly proper scoring rule - a judge minimizes it only by reporting
    its true probability of being correct, which accuracy doesn't reward.
    """
    confidences_arr = np.asarray(confidences, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)
    return float(np.mean((confidences_arr - correct_arr) ** 2))


def brier_decomposition(
    confidences: npt.ArrayLike,
    correct: npt.ArrayLike,
    n_bins: int,
    strategy: str = "auto",
) -> tuple[float, float, float]:
    """Murphy (1973)'s decomposition, on the same bins as ece():

        BS = Reliability - Resolution + Uncertainty

    - Reliability = sum_m (n_m/N) * (conf_m - acc_m)^2 - squared
      calibration gap, the squared cousin of ECE. Low is good.
    - Resolution  = sum_m (n_m/N) * (acc_m - obar)^2 - how far bin
      accuracies spread from the base rate obar; the judge only "resolves"
      anything if different confidence levels really mean different
      accuracy. High is good.
    - Uncertainty = obar * (1 - obar) - depends on the base rate alone.

    The reconstruction equals brier() exactly only for discrete signals;
    for quantile-binned continuous ones it is close but not exact
    (within-bin variation is lost to binning).

    Returns (reliability, resolution, uncertainty).
    """
    confidences_arr = np.asarray(confidences, dtype=float)
    correct_arr = np.asarray(correct, dtype=float)
    n = len(confidences_arr)
    obar = float(np.mean(correct_arr))

    edges = get_bin_edges(confidences_arr, n_bins, strategy)
    bin_labels = np.digitize(confidences_arr, edges)

    reliability_terms = []
    resolution_terms = []
    for bin_label in np.unique(bin_labels):
        mask = bin_labels == bin_label
        count = mask.sum()
        mean_confidence = confidences_arr[mask].mean()
        accuracy = correct_arr[mask].mean()
        reliability_terms.append(count * (accuracy - mean_confidence) ** 2)
        resolution_terms.append(count * (accuracy - obar) ** 2)

    reliability = float(sum(reliability_terms) / n)
    resolution = float(sum(resolution_terms) / n)
    uncertainty = obar * (1 - obar)

    return reliability, resolution, uncertainty


def auroc_error(uncertainty: npt.ArrayLike, correct: npt.ArrayLike) -> float:
    """AUROC for predicting judge ERROR from an uncertainty signal - RQ2's
    primary metric: P(a random wrong item scores higher uncertainty than a
    random correct one). 0.5 = uninformative.

    A thin wrapper around roc_auc_score; what it adds is the semantics.
    The positive class is error, not correct, and `uncertainty` must point
    toward error (pass 1 - conf, not conf). Getting either backwards
    silently returns 1 - AUROC. `correct` is cast to bool before negating:
    `~` on an int array flips bits (~1 == -2) instead of negating.

    Raises ValueError when `correct` is all one class - checked here rather
    than left to sklearn, whose behavior on that case varies by version.
    """
    uncertainty_arr = np.asarray(uncertainty, dtype=float)
    error_arr = ~np.asarray(correct, dtype=bool)
    if error_arr.all() or not error_arr.any():
        raise ValueError(
            "auroc_error is undefined: `correct` must contain both "
            "correct and incorrect items, got all one class."
        )
    return float(roc_auc_score(error_arr, uncertainty_arr))


def risk_coverage(
    uncertainty: npt.ArrayLike,
    correct: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Risk-coverage curve: if the judge abstains on its least-confident
    items, how much does accuracy improve on the rest? One point per
    coverage level, keeping the k least-uncertain items, k = 1..n:

        coverage_k = k / n
        risk_k     = E[errors among the k kept] / k

    The expectation is over random tie-breaks. Tied items are
    indistinguishable to the signal, so when the cut falls inside a tied
    group every subset of it is equally likely, and the expected errors
    taken from it are proportional to how many are taken:

        E[errors among the k kept] = E_before + j * e_group / m_group

    E_before counts the errors in all strictly less uncertain groups, j is
    how many items the cut takes from the boundary group, and that group
    has m_group items and e_group errors. Breaking ties by row order instead
    would make the curve depend on how the file happens to be sorted. A
    signal without ties gives the plain sorted curve.

    Every signal gets the same n-point grid starting at 1/n, so aurc()
    integrates them all over the same range - a signal with few distinct
    values (conf_sc has 5) is not let off its low-coverage region.

    Takes an uncertainty-typed signal (higher = abstain first), like
    auroc_error(). Returns (coverage, risk), ascending in coverage.
    """
    uncertainty_arr = np.asarray(uncertainty, dtype=float)
    wrong = ~np.asarray(correct, dtype=bool)
    n = len(uncertainty_arr)

    _, group, group_size = np.unique(uncertainty_arr, return_inverse=True, return_counts=True)
    group_errors = np.bincount(group, weights=wrong.astype(float), minlength=len(group_size))
    errors_before = np.concatenate([[0.0], np.cumsum(group_errors)[:-1]])
    kept_before = np.concatenate([[0], np.cumsum(group_size)[:-1]])

    k = np.arange(1, n + 1)
    group_of_k = np.repeat(np.arange(len(group_size)), group_size)
    taken_from_group = k - kept_before[group_of_k]
    expected_errors = (
        errors_before[group_of_k]
        + taken_from_group * group_errors[group_of_k] / group_size[group_of_k]
    )
    return k / n, expected_errors / k


def oracle_risk_coverage(correct: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """The best possible risk-coverage curve: every correct item is kept
    before any wrong one. Built by passing wrongness itself as the
    uncertainty signal through risk_coverage(), so the oracle and a real
    signal go through identical accounting and any gap between them is a
    property of the signal.

    Risk is 0 up to coverage = accuracy, then (k - n_correct) / k.
    """
    correct_arr = np.asarray(correct, dtype=bool)
    return risk_coverage(uncertainty=(~correct_arr).astype(float), correct=correct_arr)


def aurc(coverage: npt.ArrayLike, risk: npt.ArrayLike) -> float:
    """Area under the risk-coverage curve, integrated from coverage 0:

        AURC = sum_i risk_i * (coverage_i - coverage_{i-1}),  coverage_0 = 0

    On risk_coverage()'s grid (coverage_k = k/n) this is the mean of risk_k
    over k = 1..n, the definition of Geifman and El-Yaniv. Each point's
    risk covers the coverage step that ends at it, so the area always starts
    at 0: a curve that began late and was integrated only from its first
    point would skip its low-coverage region and look better than it is.
    Lower is better. For the oracle it is
    (1/n) * sum_{k > n_correct} (k - n_correct) / k, which tends to
    (1 - accuracy) + accuracy * ln(accuracy) as n grows.
    """
    coverage_arr = np.asarray(coverage, dtype=float)
    risk_arr = np.asarray(risk, dtype=float)
    widths = np.diff(coverage_arr, prepend=0.0)
    return float(np.sum(risk_arr * widths))


def coverage_weights(confidence: npt.ArrayLike, coverage: float) -> np.ndarray:
    """Each item's probability of being kept when the judge keeps its
    `coverage` share of most-confident items, k = round(coverage * n).

    Items strictly above the cut get 1, items below it get 0, and the tied
    group the cut falls inside shares what is left equally:

        w_boundary = (k - n_above) / m_boundary

    That is the inclusion probability under a random tie-break, so any
    weighted mean over these weights (accuracy, the shares inside kappa) is
    the expected value at exactly `coverage`, independent of row order. The
    weights sum to k.
    """
    confidence_arr = np.asarray(confidence, dtype=float)
    n = len(confidence_arr)
    k = max(1, round(coverage * n))
    boundary_value = np.sort(confidence_arr)[::-1][k - 1]
    above = confidence_arr > boundary_value
    at_boundary = confidence_arr == boundary_value
    weights = above.astype(float)
    weights[at_boundary] = (k - above.sum()) / at_boundary.sum()
    return weights


def auto_accept_stats(
    confidences: npt.ArrayLike,
    correct: npt.ArrayLike,
    threshold: float,
    verdict: npt.ArrayLike | None = None,
    human_label: npt.ArrayLike | None = None,
) -> dict:
    """What an auto-accept pipeline would do: accept every verdict whose
    confidence is >= `threshold`, escalate the rest to a human.

        accepted_share      = |accepted| / n
        slip_through        = |accepted AND wrong| / |wrong|
        error_among_accepted = |accepted AND wrong| / |accepted|

    slip_through is the pipeline's cost: the share of the judge's wrong
    verdicts that nobody reviews. Unlike risk_coverage(), which ranks items
    and ignores what the numbers say, a fixed threshold takes the stated
    confidence at face value - so it only means something for a
    probability-scaled signal (invariant 6), and it exposes miscalibration
    directly: a judge that says ">= 0.9" but is right 76% of the time
    accepts its errors along with everything else.

    With `verdict` and `human_label`, also returns Cohen's kappa among the
    accepted items (invariant 5: accepted-item accuracy alone is inflated
    by the base rate). Ratios with an empty denominator are NaN.
    """
    conf_arr = np.asarray(confidences, dtype=float)
    correct_arr = np.asarray(correct, dtype=bool)
    accepted = conf_arr >= threshold
    wrong = ~correct_arr
    n_accepted = int(accepted.sum())
    n_wrong = int(wrong.sum())
    n_accepted_wrong = int((accepted & wrong).sum())

    result = {
        "n": len(conf_arr),
        "n_accepted": n_accepted,
        "n_escalated": len(conf_arr) - n_accepted,
        "accepted_share": n_accepted / len(conf_arr) if len(conf_arr) else float("nan"),
        "slip_through": n_accepted_wrong / n_wrong if n_wrong else float("nan"),
        "error_among_accepted": n_accepted_wrong / n_accepted if n_accepted else float("nan"),
    }
    if verdict is not None and human_label is not None:
        verdict_arr = np.asarray(verdict)[accepted]
        human_arr = np.asarray(human_label)[accepted]
        result["kappa_among_accepted"] = cohens_kappa(verdict_arr, human_arr) if n_accepted else float("nan")
    return result


def threshold_sweep(
    signal: npt.ArrayLike,
    correct: npt.ArrayLike,
    judge_verdict: npt.ArrayLike,
    human_label: npt.ArrayLike,
    confidences: npt.ArrayLike,
    n_bins: int,
    strategy: str = "auto",
) -> dict[str, np.ndarray]:
    """Threshold sweep (D20): what you get if you trust the judge only when
    `signal` is at or below a RAW value t.

    Differs from risk_coverage(), whose x-axis is "keep the top C% most
    confident" - relative to this sample's distribution. A raw threshold is
    a deployable cutoff that means the same thing in any sample, which is
    what comparing ens_entropy_total with ens_entropy_epistemic (RQ5)
    needs: each swept on its own scale.

    At each unique signal value t (tie-safe, like risk_coverage), on the
    kept items only: coverage, n_kept, accuracy, kappa (needs the actual
    labels, not just `correct` - see cohens_kappa), and ECE recomputed on
    the retained set. No minimum-size floor: n_kept is returned so any
    "too few items" cutoff is applied at analysis time.

    Returns a dict of equal-length arrays: threshold, coverage, n_kept,
    accuracy, kappa, ece, n_effective_bins.
    """
    signal_arr = np.asarray(signal, dtype=float)
    correct_arr = np.asarray(correct, dtype=bool)
    judge_verdict_arr = np.asarray(judge_verdict)
    human_label_arr = np.asarray(human_label)
    confidences_arr = np.asarray(confidences, dtype=float)
    n = len(signal_arr)

    thresholds = np.unique(signal_arr)
    coverages = []
    n_kepts = []
    accuracies = []
    kappas = []
    eces = []
    n_effective_bins_list = []
    for threshold in thresholds:
        kept = signal_arr <= threshold
        coverage = kept.sum() / n
        accuracy = correct_arr[kept].mean()
        kappa = cohens_kappa(judge_verdict_arr[kept], human_label_arr[kept])
        ece_value, n_effective_bins = ece(confidences_arr[kept], correct_arr[kept], n_bins, strategy)

        coverages.append(coverage)
        n_kepts.append(int(kept.sum()))
        accuracies.append(accuracy)
        kappas.append(kappa)
        eces.append(ece_value)
        n_effective_bins_list.append(n_effective_bins)

    return {
        "threshold": thresholds,
        "coverage": np.array(coverages),
        "n_kept": np.array(n_kepts),
        "accuracy": np.array(accuracies),
        "kappa": np.array(kappas),
        "ece": np.array(eces),
        "n_effective_bins": np.array(n_effective_bins_list),
    }
