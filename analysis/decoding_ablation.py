"""Task 4.5: does constraining generation to a JSON schema (vs free-form)
change parse rate or the judge's actual verdict? See TASKS.md task 4.5,
DECISIONS.md D25.

`python -m analysis.decoding_ablation --config configs/run.yaml`.

Reads runs/{model_slug}/ablation_decoding.jsonl directly
(src/ablation_decoding.py, Colab) - NOT calls.parquet/items.parquet,
since this is a standalone diagnostic outside the main D19 call schedule,
not part of the RQ1-RQ4 item table. Parses each row's raw_output via
parse.py::parse_verdict_and_confidence() (CLAUDE.md invariant 7 - the
same strict JSON parser every real call is scored with, unmodified), so
a free-form row's "does this still count as parse_ok" question is
answered by the actual production parser, not a purpose-built lenient
one - that's the whole point of the ablation.

Reports two things:
  - parse rate per decoding_mode (parse_ok fraction, plus a breakdown of
    parse_failure_type among the failures).
  - verdict agreement: among items where BOTH modes parsed successfully,
    the fraction whose verdict is IDENTICAL.

No inferential statistics (no bootstrap CI) - N=100 items is a scoping/
limitations check per its own DoD ("a limitations paragraph... saying
whether constraining moved the verdicts"), not one of the five core RQs,
so a plain proportion is reported as-is rather than manufacturing a CI
this task's own DoD doesn't ask for.

Writes results/ablation_decoding_{model_slug}.csv (parse-rate table).
"""

import argparse
import json

import pandas as pd

from src.config import Config
from src.parse import parse_verdict_and_confidence


def load_ablation_calls(checkpoint_path: str) -> pd.DataFrame:
    """runs/{model_slug}/ablation_decoding.jsonl -> one row per (item,
    decoding_mode) call, with parse_verdict_and_confidence() merged in.
    Mirrors src/parse.py::build_calls_dataframe()'s own shape (raw row +
    parsed fields), scoped to this ablation's own checkpoint instead of
    the main schedule's.
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
    """Among items where BOTH decoding_mode arms parsed successfully, the
    fraction whose verdict is identical - the ablation's actual headline
    question (does constraining move the verdict, not just conf_lp, D25).
    """
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

    table_path = f"results/ablation_decoding_{config.model_slug}.csv"
    parse_rates.to_csv(table_path, index=False)
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
