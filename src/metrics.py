"""Calibration and agreement metrics. See TASKS.md tasks 0.4, 0.6, 2.5, 3.1."""

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score


def truncated_entropy(logprobs: npt.ArrayLike) -> float:
    """Entropy of the top-K logprob distribution at a single generated
    token position, renormalized over just those K tokens (K=20 in this
    project - CLAUDE.md's `logprobs` config key, D4). This is NOT the true
    full-vocabulary entropy: vLLM's logprobs=K only reports the K most
    likely tokens, and renormalizing over that truncated set systematically
    *underestimates* the true entropy, since probability mass sitting
    outside the top-K is discarded rather than merely unobserved. Always
    called "truncated" per CLAUDE.md's schema note so this bias is never
    silently forgotten downstream.

    H = -sum(p_i * log(p_i)), i over the K reported tokens, with
    p_i = exp(logprob_i) / sum_j(exp(logprob_j)) - a softmax renormalization
    over just the K observed logprobs.

    Takes a plain array of logprob floats, deliberately decoupled from any
    particular object/dict shape (vLLM's live objects, a JSON-loaded
    record, whatever) - the caller (src/parse.py) extracts the floats
    itself before calling this, since this function only needs the numbers.

    Args:
      logprobs: one generated position's top-K logprobs, as a flat array
        of floats.
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


def cohens_kappa(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Cohen's kappa: chance-corrected agreement between two raters.

    kappa = (p_o - p_e) / (1 - p_e)

    p_o is raw observed agreement: the fraction of items where a and b give
    the same label. p_e is the agreement expected by chance alone, computed
    from each rater's own marginal label distribution, independently of the
    other rater and independently of the joint agreement pattern:

        p_e = sum over every label c seen in EITHER a or b of P_A(c) * P_B(c)

    This is why kappa "deflates" raw agreement whenever both raters share a
    labeling bias (e.g. both mostly say "A" regardless of the item): p_e
    captures exactly that shared-bias inflation and removes it. This is the
    mechanism behind CLAUDE.md invariant 5 - raw agreement overstates judge
    ability by 33-41pp on MT-Bench, because judges and humans both lean
    toward the same popular answers.

    Args:
        a: labels from rater A.
        b: labels from rater B, same length as a. Any number of distinct
            categories is supported, not just two.

    Returns:
        kappa, typically in [-1, 1]. 1 = perfect agreement beyond chance,
        0 = no better than chance, negative = worse than chance.
    """
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)

    p_o = float(np.mean(a_arr == b_arr))

    labels = np.union1d(np.unique(a_arr), np.unique(b_arr))
    p_e = 0.0
    for label in labels:
        p_a = np.mean(a_arr == label)
        p_b = np.mean(b_arr == label)
        p_e += float(p_a * p_b)

    if p_e >= 1.0 - 1e-12:
        # Every item is the same single label for both raters - there is no
        # room for chance disagreement, so (p_o - p_e) / (1 - p_e) is 0/0.
        # Not exercised by real MT-Bench data; guarded so this never raises.
        return 0.0

    return (p_o - p_e) / (1 - p_e)


def overconfidence_gap(confidences: npt.ArrayLike, correct: npt.ArrayLike) -> float:
    """Signed calibration gap: mean stated confidence minus mean accuracy.

    gap = mean(confidences) - mean(correct)

    CLAUDE.md invariant 6: ECE alone can't distinguish overconfidence from
    underconfidence, because it takes |acc(B_m) - conf(B_m)| per bin before
    averaging - a judge overconfident in one bin and underconfident in
    another can post a small ECE while being badly miscalibrated in both
    directions. This signed, unbinned version never cancels that way.

    Args:
        confidences: stated confidence per item, in [0, 1].
        correct: whether the judge was actually right, per item.

    Returns:
        gap. Positive = overconfident on average, negative = underconfident.
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
    """Maximum Calibration Error: the worst single bin's gap, not the
    weighted average ECE reports.

    MCE = max_m |acc(B_m) - conf(B_m)|

    Same binning as ece() (reuse get_bin_edges()) - MCE answers "how bad is
    the worst bin," ECE answers "how bad is the typical bin." A model can
    have a small ECE and still have one badly-miscalibrated bin that MCE
    would catch and ECE would average away.

    Args:
        confidences: stated confidence per item, in [0, 1].
        correct: whether the judge was actually right, per item.
        n_bins: requested number of bins - see get_bin_edges.
        strategy: one of "uniform", "quantile", "auto" (default).

    Returns:
        (mce, n_effective_bins) - mirrors ece()'s return shape.
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
    """Brier score: mean squared error between stated confidence and the
    binary outcome. A strictly proper scoring rule - unlike accuracy, a
    judge cannot minimize this by reporting anything other than its true
    believed probability of being correct.

    BS = mean((confidence_i - correct_i)^2), correct_i in {0, 1}

    Args:
        confidences: stated confidence per item, in [0, 1].
        correct: whether the judge was actually right, per item.

    Returns:
        Brier score in [0, 1]. Lower is better; 0 = perfect.
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
    """Murphy (1973)'s three-term decomposition of the Brier score, computed
    over the same bins as ece()/mce():

    BS = Reliability - Resolution + Uncertainty

    - Uncertainty = obar * (1 - obar), obar = overall mean of `correct`
      (irreducible - depends only on the base rate, not on the judge at all)
    - Reliability = sum_m (n_m/N) * (conf_bar_m - acc_m)^2
      (a squared, bin-weighted calibration gap - this term IS what ece()
      measures, just squared instead of |.| and before the final average.
      Low is good.)
    - Resolution = sum_m (n_m/N) * (acc_m - obar)^2
      (how far each bin's accuracy departs from the overall base rate - the
      judge is only "resolving" anything if different confidence levels
      really do correspond to different accuracy. High is good.)

    Args:
        confidences: stated confidence per item, in [0, 1].
        correct: whether the judge was actually right, per item.
        n_bins: requested number of bins - see get_bin_edges.
        strategy: one of "uniform", "quantile", "auto" (default).

    Returns:
        (reliability, resolution, uncertainty), the three terms separately -
        RQ1 reports each on its own, not just their sum. Caller reconstructs
        BS = reliability - resolution + uncertainty to check against
        brier() (that reconstruction is TASKS.md task 2.5's DoD).
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
    """AUROC for using an uncertainty signal to predict judge ERROR - RQ2's
    primary metric. Positive class is error (NOT correct), scored by
    `uncertainty` (higher = more uncertain = more likely to be flagged).

    A thin wrapper around sklearn.metrics.roc_auc_score, not a from-scratch
    reimplementation: unlike ece()'s binning strategy (D14) or
    cohens_kappa()'s chance-correction (invariant 5), AUROC has no
    project-specific variant embedded in it - it's the standard "P(a random
    error item ranks above a random correct item)" statistic, with no
    design choice to internalize by re-deriving it. What this function
    actually adds - and what's worth testing - is the semantic adapter:
    which class counts as "positive" (error, not correct) and which
    direction `uncertainty` points. Getting either backwards silently
    produces `1 - true_AUROC` with no crash, which is what the tests below
    target, not sklearn's own math.

    Args:
        uncertainty: an uncertainty score per item (higher = more uncertain).
            Caller is responsible for sign - e.g. pass `1 - conf_verb`, not
            `conf_verb` itself, or a raw entropy signal directly.
        correct: whether the judge was actually right, per item. Cast to
            bool explicitly before negating - if this arrives as int/float
            0/1 (e.g. straight out of a pandas column), `~` on an int array
            is two's-complement bit-flipping (`~1 == -2`), not logical
            negation, and would silently corrupt the error label.

    Returns:
        AUROC in [0, 1]. 0.5 = uninformative, matching CLAUDE.md's note
        that Xiong found ~0.5-0.6 for prompted confidence signals. Raises
        ValueError if `correct` is all-True or all-False - AUROC is
        undefined without both classes present. Checked explicitly here
        rather than left to sklearn: roc_auc_score's own behavior in this
        situation has changed across versions (older releases raised;
        this installed version instead warns and returns NaN), so relying
        on it would make this function's behavior depend on which sklearn
        happens to be installed.
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
    """Risk-coverage curve for selective prediction: if the judge is allowed
    to abstain on its least-confident items, how much does accuracy improve
    on what's left?

    At each threshold t, "coverage" is the fraction of items kept (every
    item with uncertainty <= t), and "risk" is the error rate among the kept
    items:

        coverage(t) = |{i : uncertainty_i <= t}| / n
        risk(t)     = 1 - mean(correct_i for i with uncertainty_i <= t)

    Swept over t = every unique uncertainty value actually observed, ascending.
    This is a post-hoc, item-level filter over already-generated verdicts -
    it has nothing to do with decoding-time truncation (top-k/top-p); the
    judge produced one fixed greedy verdict per item long before this
    function runs, and "abstaining" here just means discarding that verdict
    from the kept set based on its confidence.

    Why step through unique threshold values instead of a plain stable sort
    over individual items: several of this project's signals are heavily
    tied (conf_sc has only 5 possible values; conf_verb piles up at
    0.8/0.9/0.95/1.0). A per-item stable sort would split a tied group
    across several coverage levels in whatever order the array happened to
    list them - an arbitrary artifact of row order, not a real ranking.
    Stepping by unique value instead admits an entire tied group into the
    kept set in one step, so the curve never depends on row order. This is
    the same discrete-vs-continuous reasoning get_bin_edges() already uses
    for ece()'s "auto" strategy.

    Convention: like auroc_error(), this takes an UNCERTAINTY-typed signal
    (higher = more uncertain, abstain-worthy), not a confidence-typed one -
    caller is responsible for sign, e.g. pass `1 - conf_verb` for a
    confidence signal, or an entropy signal (ens_entropy_total, etc.)
    directly.

    Args:
        uncertainty: an uncertainty score per item (higher = more uncertain).
        correct: whether the judge was actually right, per item.

    Returns:
        (coverage, risk), both ascending in coverage. Length equals the
        number of unique uncertainty values, which can be less than n when
        the signal has ties (mirrors ece()'s n_effective_bins - fewer
        distinct points is a real property of the signal, not an error).
    """
    uncertainty_arr = np.asarray(uncertainty, dtype=float)
    correct_arr = np.asarray(correct, dtype=bool)
    n = len(uncertainty_arr)

    un_thresholds = np.unique(uncertainty_arr)
    coverages = []
    risks = []
    for threshold in un_thresholds:
        kept = uncertainty_arr <= threshold
        coverages.append(kept.sum() / n)
        risks.append(1 - correct_arr[kept].mean())
    return np.array(coverages), np.array(risks)


def oracle_risk_coverage(correct: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """The best-possible risk-coverage curve: what you'd get if the
    "uncertainty" signal had perfect knowledge of correctness. Every real
    signal must fall on or above this curve at every coverage level - it's
    the oracle upper bound, not just another curve to compare against.

    Reuses risk_coverage() itself rather than a separate implementation, by
    treating wrongness (`~correct`) as the uncertainty signal: an oracle
    would rank every correct item as maximally trustworthy (uncertainty 0)
    and every incorrect item as maximally untrustworthy (uncertainty 1), so
    all correct items enter the kept set before any incorrect one. Going
    through the same function this way - rather than hand-deriving the
    curve's closed form - is what makes "oracle dominates every real signal
    by construction" an actual testable property instead of an assumption:
    both curves are built by identical coverage/risk accounting, so any gap
    between them is a genuine property of the signal, not an artifact of
    two different implementations.

    At coverage <= accuracy, risk is exactly 0 (only correct items are kept
    yet); above that, risk climbs as incorrect items get forced in.

    Args:
        correct: whether the judge was actually right, per item.

    Returns:
        (coverage, risk) - same shape/contract as risk_coverage(). Always
        exactly 2 points (correct=0.0 and correct=1.0 are the only two
        uncertainty values `~correct` can take), at
        coverage = (accuracy, 1.0) and risk = (0.0, 1 - accuracy).
    """
    correct_arr = np.asarray(correct, dtype=bool)
    return risk_coverage(uncertainty=(~correct_arr).astype(float), correct=correct_arr)


def aurc(coverage: npt.ArrayLike, risk: npt.ArrayLike) -> float:
    """Area Under the Risk-Coverage curve: one scalar summary of a
    risk_coverage() curve, via trapezoidal integration of risk as a
    function of coverage.

        AURC = integral of risk(c) dc, c from the lowest observed coverage
        to 1.0

    Lower is better (less risk retained as coverage grows). `risk` is `y`,
    `coverage` is `x` - integrating the other way around silently computes
    the area under the wrong curve, so get the argument order right here
    specifically.

    The curve is integrated only over the coverage values risk_coverage()
    actually returned (starting at the smallest observed coverage, not 0) -
    risk at coverage=0 is undefined (no items kept, mean() of an empty set),
    so there is no (0, ?) point to anchor the integral at.

    Args:
        coverage: coverage values from risk_coverage() or
            oracle_risk_coverage(), ascending.
        risk: the matching risk values, same length as coverage.

    Returns:
        AURC. For the oracle curve specifically this has a closed form,
        (1 - accuracy)^2 / 2, derived from its exact two-point shape
        (see oracle_risk_coverage()) - useful as an independent check.
    """
    coverage_arr = np.asarray(coverage, dtype=float)
    risk_arr = np.asarray(risk, dtype=float)
    return float(np.trapezoid(y=risk_arr, x=coverage_arr))


def threshold_sweep(
    signal: npt.ArrayLike,
    correct: npt.ArrayLike,
    judge_verdict: npt.ArrayLike,
    human_label: npt.ArrayLike,
    confidences: npt.ArrayLike,
    n_bins: int,
    strategy: str = "auto",
) -> dict[str, np.ndarray]:
    """Threshold-sweep table (DECISIONS.md D20, professor feedback point 5):
    a genuinely different x-axis from risk_coverage()'s percentile-based
    curve. risk_coverage() answers "what risk do I get if I keep my top C%
    most confident items" - a framing relative to this sample's own
    distribution. This answers "what do I get if I only trust the judge
    when this signal is below RAW VALUE t" - a fixed, deployable cutoff that
    means the same thing regardless of how the signal happens to be
    distributed in any particular sample. That's what makes it the right
    tool for RQ5: comparing ens_entropy_total vs ens_entropy_epistemic at
    matched coverage only makes sense if each is swept on its own real
    scale, not forced onto a shared percentile axis.

    Swept over every unique observed value of `signal`, ascending, same
    tie-safe stepping as risk_coverage() (a tied group enters the kept set
    together, never split across steps).

    At each threshold, kept = signal <= t, and four things are computed
    on the kept subset alone:
      - accuracy: correct_arr[kept].mean() - CLAUDE.md invariant 5 says
        never report this without kappa alongside it, which is exactly
        why this function computes both together rather than leaving
        kappa to a separate pass.
      - kappa: needs judge_verdict/human_label, not just `correct` - kappa's
        chance-correction depends on each rater's own label marginals
        (how often each says "A" vs "B"), which `correct` alone discards.
      - ece (+ n_effective_bins): recomputed fresh from confidences[kept]
        each step - calibration on the retained set, not the full sample's
        calibration restricted to an index range.

    Low-coverage policy: no floor, no dropped rows - every unique threshold
    gets a row, even where only a handful of items remain and kappa/ece are
    numerically noisy there (cohens_kappa's own p_e>=1-eps guard already
    protects against a crash, not against noise). `n_kept` is returned
    explicitly alongside `coverage` so any "don't trust below N items" cutoff
    is applied later, at plot/analysis time (3.2b), not silently decided here
    - matches how ece() always surfaces n_effective_bins instead of refusing
    to compute below some bin size.

    Args:
        signal: the raw uncertainty/entropy values to threshold on (higher =
            more uncertain, same convention as risk_coverage()).
        correct: whether the judge was actually right, per item.
        judge_verdict: judge's label per item (e.g. "A"/"B"), for kappa.
        human_label: human majority label per item, for kappa.
        confidences: stated confidence per item, for ECE on the retained set.
        n_bins: requested ECE bin count - see get_bin_edges.
        strategy: ECE binning strategy, one of "uniform", "quantile", "auto".

    Returns:
        dict of equal-length arrays: "threshold", "coverage", "n_kept",
        "accuracy", "kappa", "ece", "n_effective_bins" - one row per unique
        signal value.
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
