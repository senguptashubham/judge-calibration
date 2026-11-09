"""Vacuum test ("dark current", after Dark Current, 2026): does the judge
express a preference when there is no content difference to judge?

`python -m analysis.vacuum --config configs/run.yaml`

Reads the parsed vacuum calls (src/vacuum_test.py::load_vacuum_calls). The
schema forces a binary verdict, so "does it pick a winner" is 100% by
construction; what is informative is the A/B split (positional skew, with a
two-sided binomial test against 50/50) and whether stated confidence drops on
content-free pairs. The reference row is the mean conf_verb over RQ1's real
(clean, P1) items, and each vacuum type's difference from it gets a
cluster-bootstrap CI (D34).

Writes results/vacuum_{model_slug}.csv.
"""

import argparse

import pandas as pd
from scipy.stats import binomtest

from analysis.rq1 import load_rq1_items
from src.boot import cluster_bootstrap
from src.config import Config
from src.vacuum_test import load_vacuum_calls


def summarize_vacuum(calls: pd.DataFrame) -> pd.DataFrame:
    """Per vacuum_type: n, parse rate, share of A/B verdicts, binomial p
    against a 50/50 split, and mean verbalized confidence.
    """
    rows = []
    for vacuum_type, group in calls.groupby("vacuum_type", sort=False):
        parsed = group[group["parse_ok"]]
        n_a = int((parsed["verdict"] == "A").sum())
        rows.append(
            {
                "row": vacuum_type,
                "n": len(group),
                "parse_rate": len(parsed) / len(group),
                "pct_a": n_a / len(parsed),
                "pct_b": 1 - n_a / len(parsed),
                "binomial_p": binomtest(n_a, len(parsed), 0.5).pvalue,
                "mean_verbalized_conf": parsed["verbalized_conf"].mean(),
            }
        )
    return pd.DataFrame.from_records(rows)


def confidence_gap_vs_real(
    vacuum_conf: pd.DataFrame, real_conf: pd.DataFrame, n: int, seed: int
) -> tuple[float, float, float]:
    """Mean stated confidence on vacuum pairs minus mean on real items, with
    a cluster-bootstrap CI (invariant 2).

    Both frames carry `question_id` and `conf`. Vacuum pairs are built from
    real questions and repeat them, so the two samples are stacked and
    resampled together by question_id: a drawn question brings its vacuum
    rows and its real rows, keeping any dependence between them.
    """
    stacked = pd.concat(
        [vacuum_conf.assign(is_vacuum=True), real_conf.assign(is_vacuum=False)],
        ignore_index=True,
    )

    def gap(df: pd.DataFrame) -> float:
        return df.loc[df["is_vacuum"], "conf"].mean() - df.loc[~df["is_vacuum"], "conf"].mean()

    return cluster_bootstrap(stacked, gap, "question_id", n=n, seed=seed)


def main(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    calls = load_vacuum_calls(f"{config.paths.runs_dir}/vacuum.jsonl")
    table = summarize_vacuum(calls)

    real_items = load_rq1_items(config.paths.items_parquet)
    real_conf = real_items[["question_id"]].assign(conf=real_items["conf_verb"].to_numpy())

    gaps = []
    for vacuum_type in table["row"]:
        parsed = calls[(calls["vacuum_type"] == vacuum_type) & calls["parse_ok"]]
        vacuum_conf = parsed[["question_id"]].assign(conf=parsed["verbalized_conf"].to_numpy())
        point, low, high = confidence_gap_vs_real(vacuum_conf, real_conf, n=2000, seed=config.seed)
        gaps.append({"conf_gap_vs_real": point, "conf_gap_ci_low": low, "conf_gap_ci_high": high})
    table = pd.concat([table, pd.DataFrame.from_records(gaps)], axis=1)

    reference = {"row": "real clean/P1 items", "n": len(real_items), "mean_verbalized_conf": real_items["conf_verb"].mean()}
    table = pd.concat([table, pd.DataFrame.from_records([reference])], ignore_index=True)

    print(table.to_string(index=False))
    table_path = f"{config.paths.results_dir}/vacuum_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
