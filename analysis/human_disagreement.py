"""Human-disagreement decomposition. See TASKS.md tasks 3.3 and 3.4 (D9).

`python -m analysis.human_disagreement --config configs/run.yaml`.

Population: load_rq1_items() (clean, P1, human_label not null) FURTHER
filtered to n_human_votes >= 2 - D9's continuous test explicitly requires
this ("uses every item with >= 2 votes"), and it's a real narrowing:
1836 -> 595 items. Within that population, n_contested = 31 (checked
empirically against real data) - well under D9's n_contested >= 100 bar,
so the secondary bucketed (unanimous/strong-majority/contested) comparison
is OUT OF SCOPE here. Don't build it; the one-sentence finding should say
why it's absent, not just omit it. This same n_human_votes >= 2 restriction
applies to task 3.4 too, not just 3.3 - d_human is equally undefined-as-a-
disagreement-signal below 2 votes either way, same D9 reasoning either task.

Task 3.3: two regressions of a per-item quantity on
d_human = |frac_prefer_a - 0.5| (continuous consensus strength, D9):
  - correct ~ d_human      (does accuracy trend with how contested the item was?)
  - conf_verb ~ d_human    (does the judge's STATED confidence track it?)
conf_verb is used here as the one representative signal for the figure.
Writes results/figures/human_disagreement_{model_slug}.png (src/plots.py's
plot_human_disagreement) and results/human_disagreement_table_{model_slug}.csv.

Task 3.4: Spearman-only (not also an OLS slope, unlike 3.3) correlation
between EACH of the four original confidence signals and d_human. Spearman
only, for consistency with 3.3's own finding that d_human's near-categorical,
3-value/95%-imbalanced shape favors a monotonic-only assumption over a
linear one - repeating the OLS slope four more times would just repeat that
caveat without adding anything new. Writes
results/d_human_correlations_table_{model_slug}.csv and
results/figures/d_human_correlations_{model_slug}.png.

model_slug = Config.model_slug throughout, so a second judge model never
overwrites the first's output.
"""

import argparse

import numpy as np
import pandas as pd

from analysis.rq1 import SIGNALS, load_rq1_items
from src.boot import cluster_bootstrap
from src.config import Config
from src.plots import plot_d_human_correlations, plot_human_disagreement


def load_disagreement_items(items_parquet: str) -> pd.DataFrame:
    """load_rq1_items() plus D9's n_human_votes >= 2 filter.

    Also prints n_contested on the resulting population so a future data
    refresh can't silently drift back above D9's n_contested >= 100 bar
    (which would put the secondary bucketed comparison back in scope)
    without anyone noticing - this only ever warns, it never blocks the
    primary continuous regression from running.

    Raises:
        AssertionError: if fewer than 100 items survive the n_human_votes
            filter - too few for the regression itself to be meaningful.
    """
    items = load_rq1_items(items_parquet)
    filtered_items = items[items["n_human_votes"] >= 2].copy()
    assert len(filtered_items) > 100, (
        f"only {len(filtered_items)} items with n_human_votes >= 2 - too few "
        "for the continuous regression to be meaningful (D9)."
    )

    # Rounded to 6dp: d_human = |frac_prefer_a - 0.5| computed from small
    # vote-count fractions (e.g. 1/3 vs 2/3, both "really" 1/6) can land on
    # adjacent float64 values (~6e-17 apart) for the mathematically same
    # true fraction - confirmed on real data. Left unrounded, this doesn't
    # visibly break _ols_slope() (a continuous regression is insensitive to
    # 1e-16 noise) but DOES silently corrupt _spearman_corr(), which ranks
    # values and needs exact ties detected as ties, not as two adjacent
    # ranks. Rounded once here so every downstream consumer (both
    # regressions AND the plot) sees the same, already-clean d_human,
    # rather than plot_human_disagreement() being the only place this was
    # fixed.
    filtered_items["d_human"] = filtered_items["d_human"].round(6)

    n_contested = int((~filtered_items["human_unanimous"]).sum())
    if n_contested >= 100:
        print(
            f"NOTE: n_contested={n_contested} >= 100 - D9's secondary bucketed "
            "comparison is now in scope and should be added; this script "
            "currently only runs the primary continuous regression."
        )
    else:
        print(f"n_contested={n_contested} (< 100) - bucketed comparison correctly out of scope per D9.")

    return filtered_items


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Closed-form OLS slope of y on x: sum((x-xbar)(y-ybar)) / sum((x-xbar)^2).

    Not np.cov(x, y) / np.var(x): np.cov returns the full 2x2
    covariance MATRIX (var(x), cov(x,y); cov(x,y), var(y)), not the
    scalar cross-covariance - dividing that matrix by a scalar and
    calling float() on the multi-element result raises, it doesn't
    silently give the wrong number. The centered-sum formula below is
    also ddof-independent (np.cov defaults to ddof=1, np.var to ddof=0 -
    mixing them would scale the slope by a spurious n/(n-1) factor),
    so it's the more explicit choice, not just the one that avoids the
    matrix-indexing footgun.

    Known unguarded edge case: undefined (division by zero) if x has
    zero variance - e.g. a bootstrap replicate that happens to resample
    only question_ids whose items all share one d_human value. Not
    guarded here since it's not yet been observed on real data (n=595,
    ~80 groups) - if cluster_bootstrap ever produces a NaN/inf slope in
    practice, np.quantile's CI would silently be corrupted by it, so
    that's the point to add a guard, not before.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_centered = x_arr - x_arr.mean()
    y_centered = y_arr - y_arr.mean()
    return float(np.sum(x_centered * y_centered) / np.sum(x_centered**2))


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient: cov(x,y) / (std(x) * std(y)), via
    the same centered-sum construction as _ols_slope() (and for the same
    reason - explicit, ddof-independent, no np.cov matrix-indexing trap).
    Not exposed on its own; only used as _spearman_corr()'s building block.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_centered = x_arr - x_arr.mean()
    y_centered = y_arr - y_arr.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    return float(np.sum(x_centered * y_centered) / denom)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation: Pearson correlation of the RANKS, not
    the raw values. Complements _ols_slope() rather than replacing it -
    the OLS slope assumes a linear relationship and reports it in y's own
    units; Spearman only assumes a MONOTONIC relationship and reports
    strength/direction on a unitless -1..1 scale, which is the more
    honest tool here given d_human takes only 3 distinct values in this
    population (see module docstring) - not enough points to trust a
    linear-shape assumption, but monotonicity is still a meaningful,
    testable claim.

    pd.Series.rank() (average method, ties get the mean of their tied
    ranks) rather than a plain double-argsort: d_human is heavily tied
    (only 3 distinct values here), and a double-argsort would assign
    tied items arbitrary DISTINCT ranks based on array order - silently
    wrong for a signal this discrete, not just imprecise.
    """
    x_ranks = pd.Series(x).rank().to_numpy()
    y_ranks = pd.Series(y).rank().to_numpy()
    return _pearson_corr(x_ranks, y_ranks)


def compute_disagreement_regression(items: pd.DataFrame, y_col: str, seed: int) -> dict:
    """Cluster-bootstrap CIs on both the OLS slope AND the Spearman rank
    correlation of `y_col` against d_human, over question_id (invariant 2).

    Two separate cluster_bootstrap calls (one per statistic), same
    one-call-per-metric convention as rq1.py/rq2.py - not batched into a
    single resampling pass, for consistency with the rest of this project
    even though it means resampling twice.

    Args:
        items: load_disagreement_items()'s output.
        y_col: "correct" or "conf_verb".
        seed: config.seed.

    Returns:
        {"slope", "slope_ci_low", "slope_ci_high", "spearman",
        "spearman_ci_low", "spearman_ci_high"}.
    """
    def _slope(df: pd.DataFrame) -> float:
        return _ols_slope(df["d_human"].to_numpy(), df[y_col].to_numpy(dtype=float))

    def _spearman(df: pd.DataFrame) -> float:
        return _spearman_corr(df["d_human"].to_numpy(), df[y_col].to_numpy(dtype=float))

    slope_point, slope_lo, slope_hi = cluster_bootstrap(items, _slope, "question_id", n=2000, seed=seed)
    spearman_point, spearman_lo, spearman_hi = cluster_bootstrap(items, _spearman, "question_id", n=2000, seed=seed)
    return {
        "slope": slope_point,
        "slope_ci_low": slope_lo,
        "slope_ci_high": slope_hi,
        "spearman": spearman_point,
        "spearman_ci_low": spearman_lo,
        "spearman_ci_high": spearman_hi,
    }


def compute_d_human_correlation(items: pd.DataFrame, signal: str, seed: int) -> dict:
    """Task 3.4: cluster-bootstrap CI on the Spearman rank correlation
    between `signal` and d_human. Spearman only - see module docstring for
    why 3.4 doesn't also report an OLS slope the way 3.3 does.

    Args:
        items: load_disagreement_items()'s output.
        signal: one of SIGNALS ("conf_verb", "conf_lp", "conf_sc", "conf_bpe").
        seed: config.seed.

    Returns:
        {"spearman", "spearman_ci_low", "spearman_ci_high"}.
    """
    def _spearman(df: pd.DataFrame) -> float:
        return _spearman_corr(df["d_human"].to_numpy(), df[signal].to_numpy(dtype=float))

    point, ci_low, ci_high = cluster_bootstrap(items, _spearman, "question_id", n=2000, seed=seed)
    return {"spearman": point, "spearman_ci_low": ci_low, "spearman_ci_high": ci_high}


def main(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    items = load_disagreement_items(config.paths.items_parquet)

    correct_result = compute_disagreement_regression(items, "correct", config.seed)
    conf_verb_result = compute_disagreement_regression(items, "conf_verb", config.seed)

    plot_human_disagreement(
        d_human=items["d_human"].to_numpy(),
        correct=items["correct"].to_numpy(),
        confidence=items["conf_verb"].to_numpy(),
        n_bins=config.n_bins,
        model_slug=config.model_slug,
    )

    print(
        f"correct ~ d_human slope: {correct_result['slope']:.4f} "
        f"[{correct_result['slope_ci_low']:.4f}, {correct_result['slope_ci_high']:.4f}]"
    )
    print(
        f"correct ~ d_human spearman: {correct_result['spearman']:.4f} "
        f"[{correct_result['spearman_ci_low']:.4f}, {correct_result['spearman_ci_high']:.4f}]"
    )
    print(
        f"conf_verb ~ d_human slope: {conf_verb_result['slope']:.4f} "
        f"[{conf_verb_result['slope_ci_low']:.4f}, {conf_verb_result['slope_ci_high']:.4f}]"
    )
    print(
        f"conf_verb ~ d_human spearman: {conf_verb_result['spearman']:.4f} "
        f"[{conf_verb_result['spearman_ci_low']:.4f}, {conf_verb_result['spearman_ci_high']:.4f}]"
    )
    print(
        "Secondary bucketed (unanimous/strong-majority/contested) comparison "
        "skipped per D9 - n_contested=31 in this population, below the "
        "n_contested >= 100 threshold."
    )

    regression_table = pd.DataFrame.from_records([
        {"target": "correct", **correct_result},
        {"target": "conf_verb", **conf_verb_result},
    ])
    regression_table_path = f"results/human_disagreement_table_{config.model_slug}.csv"
    regression_table.to_csv(regression_table_path, index=False)
    print(f"Wrote {len(regression_table)} rows to {regression_table_path}")

    print("\n--- task 3.4: signal vs. d_human, Spearman only ---")
    correlation_rows = []
    for signal in SIGNALS:
        result = compute_d_human_correlation(items, signal, config.seed)
        print(
            f"{signal} ~ d_human spearman: {result['spearman']:.4f} "
            f"[{result['spearman_ci_low']:.4f}, {result['spearman_ci_high']:.4f}]"
        )
        correlation_rows.append({"signal": signal, **result})

    correlation_table = pd.DataFrame.from_records(correlation_rows)
    correlation_table_path = f"results/d_human_correlations_table_{config.model_slug}.csv"
    correlation_table.to_csv(correlation_table_path, index=False)
    print(f"Wrote {len(correlation_table)} rows to {correlation_table_path}")

    plot_d_human_correlations(
        signals=correlation_table["signal"].tolist(),
        spearman=correlation_table["spearman"].to_numpy(),
        ci_low=correlation_table["spearman_ci_low"].to_numpy(),
        ci_high=correlation_table["spearman_ci_high"].to_numpy(),
        model_slug=config.model_slug,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
