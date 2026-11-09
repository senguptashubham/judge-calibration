"""Human-disagreement decomposition (D9).

`python -m analysis.human_disagreement --config configs/run.yaml`

Population: RQ1's, restricted to n_human_votes >= 2 (N=595) - below two
votes d_human = |frac_prefer_a - 0.5| says nothing about disagreement.
Only 31 of these items are contested, well under D9's 100, so the secondary
bucketed comparison is out of scope.

Accuracy and confidence: OLS slope and Spearman correlation of `correct` and of
conf_verb against d_human, cluster-bootstrap CIs. Writes
results/human_disagreement_table_{model_slug}.csv and
results/figures/human_disagreement_{model_slug}.png.

Signals: Spearman only, for each of the four confidence signals against
d_human. d_human takes just 3 distinct values here, which supports a
monotonic claim but not a linear one. Writes
results/d_human_correlations_table_{model_slug}.csv and
results/figures/d_human_correlations_{model_slug}.png.

Multi-vote sensitivity (D34, `--task multivote`): RQ1's overconfidence gap and
RQ2's AUROC for the four signals on these 595 items, the ones whose label
rests on more than one vote. The subset is easier as well as cleaner, so it
is read beside the full-population numbers, not as a correction of them.
Writes results/multivote_sensitivity_{model_slug}.csv.
"""

import argparse

import numpy as np
import pandas as pd

from analysis.rq1 import CALIBRATION_FORM, SIGNALS, load_rq1_items
from src.boot import cluster_bootstrap
from src.config import Config
from src.metrics import auroc_error, cohens_kappa, overconfidence_gap
from src.plots import plot_d_human_correlations, plot_human_disagreement


def load_disagreement_items(items_parquet: str) -> pd.DataFrame:
    """load_rq1_items() filtered to n_human_votes >= 2, with d_human rounded.

    Rounded to 6 dp because the same true fraction can land on adjacent
    float64 values (|1/3 - 0.5| and |2/3 - 0.5| differ by ~6e-17); ranks
    would then split real ties and corrupt the Spearman correlation.

    Prints n_contested, so a data change that brings the bucketed
    comparison back into scope is noticed.
    """
    items = load_rq1_items(items_parquet)
    filtered_items = items[items["n_human_votes"] >= 2].copy()
    assert len(filtered_items) > 100, (
        f"only {len(filtered_items)} items with n_human_votes >= 2 - too few "
        "for the continuous regression to be meaningful (D9)."
    )

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
    """OLS slope of y on x: sum((x-xbar)(y-ybar)) / sum((x-xbar)^2).

    Written out rather than np.cov(x, y) / np.var(x): np.cov returns a 2x2
    matrix, and it defaults to ddof=1 while np.var uses ddof=0. Undefined if
    x has zero variance (not observed on real data).
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_centered = x_arr - x_arr.mean()
    y_centered = y_arr - y_arr.mean()
    return float(np.sum(x_centered * y_centered) / np.sum(x_centered**2))


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation: cov(x, y) / (std(x) * std(y)), centered-sum form."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    x_centered = x_arr - x_arr.mean()
    y_centered = y_arr - y_arr.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    return float(np.sum(x_centered * y_centered) / denom)


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation: Pearson on the ranks. Assumes only a monotonic
    relationship, the honest claim when d_human has 3 values. Ties get their
    average rank (pd.Series.rank), not arbitrary distinct ranks by position.
    """
    x_ranks = pd.Series(x).rank().to_numpy()
    y_ranks = pd.Series(y).rank().to_numpy()
    return _pearson_corr(x_ranks, y_ranks)


def compute_disagreement_regression(items: pd.DataFrame, y_col: str, seed: int) -> dict:
    """OLS slope and Spearman of `y_col` against d_human, each
    with a cluster-bootstrap CI (invariant 2).
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
    """Spearman of `signal` against d_human, with a cluster-bootstrap CI."""
    def _spearman(df: pd.DataFrame) -> float:
        return _spearman_corr(df["d_human"].to_numpy(), df[signal].to_numpy(dtype=float))

    point, ci_low, ci_high = cluster_bootstrap(items, _spearman, "question_id", n=2000, seed=seed)
    return {"spearman": point, "spearman_ci_low": ci_low, "spearman_ci_high": ci_high}


def compute_multivote_sensitivity(items: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Per signal, on the multi-vote items: the overconfidence gap of its
    calibration form (D7) and AUROC(uncertainty -> error), plus the verdict's
    accuracy and κ (invariant 5), each with a cluster-bootstrap CI.
    """
    def _ci(fn) -> tuple[float, float, float]:
        return cluster_bootstrap(items, fn, "question_id", n=2000, seed=seed)

    accuracy = _ci(lambda df: df["correct"].astype(float).mean())
    kappa = _ci(lambda df: cohens_kappa(df["judge_verdict"], df["human_label"]))

    rows = []
    for signal in SIGNALS:
        calibration_col = CALIBRATION_FORM[signal]
        gap = _ci(lambda df: overconfidence_gap(df[calibration_col].to_numpy(), df["correct"].to_numpy()))
        auroc = _ci(lambda df: auroc_error(1 - df[signal].to_numpy(dtype=float), df["correct"].to_numpy()))
        rows.append(
            {
                "signal": signal,
                "n": len(items),
                "n_questions": items["question_id"].nunique(),
                "accuracy": accuracy[0], "accuracy_ci_low": accuracy[1], "accuracy_ci_high": accuracy[2],
                "kappa": kappa[0], "kappa_ci_low": kappa[1], "kappa_ci_high": kappa[2],
                "overconfidence_gap": gap[0], "overconfidence_gap_ci_low": gap[1], "overconfidence_gap_ci_high": gap[2],
                "auroc": auroc[0], "auroc_ci_low": auroc[1], "auroc_ci_high": auroc[2],
            }
        )
    return pd.DataFrame.from_records(rows)


def main_multivote(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    items = load_disagreement_items(config.paths.items_parquet)
    table = compute_multivote_sensitivity(items, config.seed)
    print(table.to_string(index=False))
    table_path = f"{config.paths.results_dir}/multivote_sensitivity_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


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

    regression_table = pd.DataFrame.from_records([
        {"target": "correct", **correct_result},
        {"target": "conf_verb", **conf_verb_result},
    ])
    regression_table_path = f"{config.paths.results_dir}/human_disagreement_table_{config.model_slug}.csv"
    regression_table.to_csv(regression_table_path, index=False)
    print(f"Wrote {len(regression_table)} rows to {regression_table_path}")

    print("\n--- signal vs. d_human, Spearman only ---")
    correlation_rows = []
    for signal in SIGNALS:
        result = compute_d_human_correlation(items, signal, config.seed)
        print(
            f"{signal} ~ d_human spearman: {result['spearman']:.4f} "
            f"[{result['spearman_ci_low']:.4f}, {result['spearman_ci_high']:.4f}]"
        )
        correlation_rows.append({"signal": signal, **result})

    correlation_table = pd.DataFrame.from_records(correlation_rows)
    correlation_table_path = f"{config.paths.results_dir}/d_human_correlations_table_{config.model_slug}.csv"
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
    parser.add_argument("--task", choices=["all", "multivote"], default="all")
    args = parser.parse_args()
    if args.task == "all":
        main(args.config)
    main_multivote(args.config)
