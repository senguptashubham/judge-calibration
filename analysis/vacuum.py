"""Task 1.8's vacuum test ("dark current", LEARNING.md A8): does the judge
express a preference when there is no content difference to judge?

`python -m analysis.vacuum --config configs/run.yaml`

Reads {runs_dir}/vacuum.jsonl (src/vacuum_test.py) and scores every row with
the production parser. The schema forces a binary verdict, so "does it pick
a winner" is 100% by construction; what is informative is the A/B split
(positional skew, with a two-sided binomial test against 50/50) and whether
stated confidence drops on content-free pairs. The reference row is the mean
conf_verb over RQ1's real (clean, P1) items.

Writes results/vacuum_{model_slug}.csv.
"""

import argparse
import json

import pandas as pd
from scipy.stats import binomtest

from analysis.rq1 import load_rq1_items
from src.config import Config
from src.parse import parse_verdict_and_confidence


def load_vacuum_calls(checkpoint_path: str) -> pd.DataFrame:
    """vacuum.jsonl -> one row per call, with the parsed verdict and confidence."""
    records = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append({"vacuum_type": row["vacuum_type"], **parse_verdict_and_confidence(row["raw_output"])})
    return pd.DataFrame.from_records(records)


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


def main(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    calls = load_vacuum_calls(f"{config.paths.runs_dir}/vacuum.jsonl")
    table = summarize_vacuum(calls)

    real_items = load_rq1_items(config.paths.items_parquet)
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
