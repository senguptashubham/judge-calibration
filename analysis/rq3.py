"""RQ3a: does the judge's verdict flip between presentation orders, and is
it confident when it does? See TASKS.md task 4.3.

`python -m analysis.rq3 --config configs/run.yaml`.

Reads results/items.parquet via load_rq1_items() (analysis/rq1.py) - the
same (clean, P1, human_label not null) population as RQ1/RQ2, N=1836. This
is a deliberate scope choice, not a data requirement: flip rate and
confidence-on-flipped don't need human_label at all (they're pure
judge-behavior signals, nothing to do with correctness), so the maximal
population would be all 1904 clean/P1 items. Reusing RQ1's 1836-item
population instead keeps one canonical N across every RQ1-RQ4 core
analysis, rather than needing to justify a second, 68-item-different
population size in REPORT.md for no analytical reason.

Only RQ3a lands here - RQ3b (task 4.4, verbosity) needs the `verbose` run
and will be added to this module once that data exists, not a new file
(same pattern rq5.py's module docstring used for its own Week 3/Week 5
split).

Computes two things, both cluster-bootstrap CIs (question_id, invariant 2):
  - flip rate: mean(flipped) over the population.
  - for each of the four original signals: mean confidence on flipped vs.
    unflipped items, and the DIFFERENCE between the two groups with its
    own CI - via a single cluster_bootstrap() call with a custom
    group-difference stat_fn, NOT paired_cluster_bootstrap(). The two
    functions solve different problems: paired_cluster_bootstrap needs two
    DataFrames sharing the same question_id universe (e.g. the same items
    scored two ways, like RQ1's judge_verdict vs. verdict_bidir
    comparison, or RQ5's same-items-two-signals comparison). flipped and
    unflipped are two DISJOINT SUBSETS of one item set, not two aligned
    copies - a question_id with zero flipped items simply wouldn't exist
    in the "flipped" half at all, so paired_cluster_bootstrap's shared-
    group-universe requirement isn't even satisfiable here. A plain
    cluster_bootstrap() with a stat_fn that computes both group means
    INSIDE one resampled replicate is the correct tool for a two-disjoint-
    subgroup comparison.

Writes results/rq3a_table_{model_slug}.csv (one row per signal) and
results/figures/rq3a_confidence_gap_{model_slug}.png (src/plots.py's
plot_rq3a_confidence_gap, a forest plot of all four signals' gaps against
a zero reference line - same shape as task 3.4's correlation figure, both
now built on the shared _forest_plot() helper). Prints the flip rate plus
the RQ3a money sentence (conf_verb's gap - the project's headline signal
throughout, same convention as RQ1's headline and task 3.3's figure).
"""

import argparse

import pandas as pd

from analysis.rq1 import SIGNALS, load_rq1_items
from src.boot import cluster_bootstrap
from src.config import Config
from src.plots import plot_rq3a_confidence_gap


def compute_flip_rate(items: pd.DataFrame, seed: int) -> dict:
    """Cluster-bootstrap CI on the flip rate: mean(flipped) over the
    population.
    """
    def _flip_rate(df: pd.DataFrame) -> float:
        return df["flipped"].astype(float).mean()

    point, ci_low, ci_high = cluster_bootstrap(items, _flip_rate, "question_id", n=2000, seed=seed)
    return {"flip_rate": point, "flip_rate_ci_low": ci_low, "flip_rate_ci_high": ci_high}


def compute_confidence_gap(items: pd.DataFrame, signal: str, seed: int) -> dict:
    """Cluster-bootstrap CI on mean(signal | flipped) - mean(signal | not
    flipped): does the judge's stated confidence drop on items where the
    presentation order actually changed its verdict? If the judge's
    uncertainty signal is tracking its own bias-induced errors, this gap
    should be negative (lower confidence when flipped) with a CI that
    excludes 0 - if it doesn't, the signal isn't picking up on position
    bias at all.

    Args:
        items: load_rq1_items()'s output, with `flipped` cast to bool.
        signal: one of SIGNALS - the confidence column name.
        seed: config.seed.

    Returns:
        mean_conf_flipped/mean_conf_unflipped (real-data point values, not
        bootstrapped) plus gap_flipped_minus_unflipped/gap_ci_low/
        gap_ci_high (the bootstrapped difference and its CI).
    """
    def _gap(df: pd.DataFrame) -> float:
        flipped_mean = df.loc[df["flipped"], signal].mean()
        unflipped_mean = df.loc[~df["flipped"], signal].mean()
        return flipped_mean - unflipped_mean

    point, ci_low, ci_high = cluster_bootstrap(items, _gap, "question_id", n=2000, seed=seed)

    return {
        "mean_conf_flipped": items.loc[items["flipped"], signal].mean(),
        "mean_conf_unflipped": items.loc[~items["flipped"], signal].mean(),
        "gap_flipped_minus_unflipped": point,
        "gap_ci_low": ci_low,
        "gap_ci_high": ci_high,
    }


def main(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    items = load_rq1_items(config.paths.items_parquet)
    items = items.copy()
    items["flipped"] = items["flipped"].astype(bool)

    flip_rate_result = compute_flip_rate(items, config.seed)
    print(
        f"Flip rate (AB vs BA canonical verdict, (clean, P1), N={len(items)}): "
        f"{flip_rate_result['flip_rate']:.4f} "
        f"[{flip_rate_result['flip_rate_ci_low']:.4f}, {flip_rate_result['flip_rate_ci_high']:.4f}]"
    )

    rows = []
    for signal in SIGNALS:
        gap = compute_confidence_gap(items, signal, config.seed)
        rows.append({"signal": signal, **gap})

    table = pd.DataFrame.from_records(rows)
    table_path = f"results/rq3a_table_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    plot_rq3a_confidence_gap(
        signals=table["signal"].tolist(),
        gap=table["gap_flipped_minus_unflipped"].to_numpy(),
        ci_low=table["gap_ci_low"].to_numpy(),
        ci_high=table["gap_ci_high"].to_numpy(),
        model_slug=config.model_slug,
    )

    headline = table[table["signal"] == "conf_verb"].iloc[0]
    print(
        "conf_verb gap (flipped - unflipped confidence): "
        f"{headline['gap_flipped_minus_unflipped']:.4f} "
        f"[{headline['gap_ci_low']:.4f}, {headline['gap_ci_high']:.4f}]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
