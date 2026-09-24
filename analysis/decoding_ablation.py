"""Task 4.5: does constraining generation to a JSON schema (vs free-form)
change the parse rate or the judge's verdict? (D25)

`python -m analysis.decoding_ablation --config configs/run.yaml`

Reads {runs_dir}/ablation_decoding.jsonl (src/ablation_decoding.py) and
scores every row with the production parser, parse.py::
parse_verdict_and_confidence(), unmodified. Reports the parse rate per
decoding mode and, among items where both modes parsed, the fraction with
identical verdicts. Plain proportions, no CI: a 100-item limitations
check, not a core RQ.

Writes results/ablation_decoding_{model_slug}.csv (parse-rate table).
"""

import argparse
import json

import pandas as pd

from src.config import Config
from src.parse import parse_verdict_and_confidence


def load_ablation_calls(checkpoint_path: str) -> pd.DataFrame:
    """ablation_decoding.jsonl -> one row per (item, decoding_mode), with
    parse_verdict_and_confidence() merged in.
    """
    records = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            record = dict(row)
            record.update(parse_verdict_and_confidence(row["raw_output"]))
            records.append(record)
    return pd.DataFrame.from_records(records)


def compute_parse_rates(calls: pd.DataFrame) -> pd.DataFrame:
    """parse_ok rate per decoding_mode, plus counts of each
    parse_failure_type among that mode's failures (one column per
    failure type actually seen, 0/absent otherwise).
    """
    rows = []
    for mode, group in calls.groupby("decoding_mode"):
        row = {"decoding_mode": mode, "n": len(group), "parse_ok_rate": group["parse_ok"].mean()}
        failure_counts = group.loc[~group["parse_ok"], "parse_failure_type"].value_counts()
        for failure_type, count in failure_counts.items():
            row[f"n_{failure_type}"] = int(count)
        rows.append(row)
    return pd.DataFrame.from_records(rows).fillna(0)


def compute_verdict_agreement(calls: pd.DataFrame) -> dict:
    """Among items where both arms parsed, the fraction with identical verdicts."""
    wide = calls.pivot(index="item_id", columns="decoding_mode", values="verdict")
    both_parsed = wide.dropna(subset=["constrained", "free_form"])
    n_disagree = int((both_parsed["constrained"] != both_parsed["free_form"]).sum()) if len(both_parsed) else 0
    agreement_rate = 1 - n_disagree / len(both_parsed) if len(both_parsed) else None
    return {
        "n_items": wide.shape[0],
        "n_both_parsed": len(both_parsed),
        "n_disagree": n_disagree,
        "agreement_rate": agreement_rate,
    }


def main(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    checkpoint_path = f"{config.paths.runs_dir}/ablation_decoding.jsonl"

    calls = load_ablation_calls(checkpoint_path)
    print(f"Loaded {len(calls)} ablation calls from {checkpoint_path}")

    parse_rates = compute_parse_rates(calls)
    print(parse_rates.to_string(index=False))

    agreement = compute_verdict_agreement(calls)
    print(
        f"Verdict agreement (constrained vs free_form), among {agreement['n_both_parsed']}/"
        f"{agreement['n_items']} items where both parsed: "
        f"{agreement['agreement_rate']:.4f} ({agreement['n_disagree']} disagreements)"
    )

    table_path = f"{config.paths.results_dir}/ablation_decoding_{config.model_slug}.csv"
    parse_rates.to_csv(table_path, index=False)
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
