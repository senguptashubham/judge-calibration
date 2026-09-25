"""The three judges side by side, read from the per-RQ result tables.

`python -m analysis.compare_judges --config configs/run.yaml --kev-config configs/run_kev.yaml --autoj-config configs/run_autoj.yaml`

Every judge is compared on its order-swap agreement signal (conf_bpe,
conf_kev_bpe, conf_sc_bpe_autoj; the greedy-only form for auto-j's
verbosity test) - the one signal all three have, and each judge's best
ranker. kev-8b rows are split by coverage regime and auto-j rows by turn,
as in RQ6 and RQ7. Nothing is recomputed: each value and CI is copied from
the table that reported it.

Writes results/judge_comparison.csv and results/figures/judge_comparison.png (plus
judge_comparison_dark.png, the same figure in the site's dark theme for the README).
"""

import argparse

import pandas as pd

from src.config import Config
from src.judge_autoj import AutojConfig
from src.judge_kev import KevConfig
from src.plots import judge_name, plot_judge_comparison

KEV_REGIMES = {"in_coverage": "in coverage", "out_of_coverage": "out of coverage"}

# (measure key, panel title, x label, reference line)
MEASURES = [
    ("auroc", "Error detection (AUROC)", "AUROC of the order-swap signal (dashed = chance)", 0.5),
    ("overconfidence_gap", "Overconfidence", "mean confidence − accuracy (> 0 = overconfident)", 0.0),
    ("flip_rate", "Position bias", "share of verdicts that flip when answer order swaps", None),
    ("delta_ece", "Padding attack on calibration", "Δ ECE, verbose − clean (> 0 = worse calibrated)", 0.0),
]


def _row(measure: str, judge: str, population: str, df: pd.DataFrame, value_col: str, ci_prefix: str | None = None) -> dict:
    """One comparison row copied from a result table's single matching row.
    The CI columns are `{ci_prefix}_ci_low/high` (default: `value_col`'s).
    """
    if len(df) != 1:
        raise ValueError(f"{measure}/{judge}/{population}: expected one row, got {len(df)}")
    record = df.iloc[0]
    ci_prefix = ci_prefix or value_col
    return {
        "measure": measure,
        "judge": judge,
        "population": population,
        "value": record[value_col],
        "ci_low": record[f"{ci_prefix}_ci_low"],
        "ci_high": record[f"{ci_prefix}_ci_high"],
    }


def _delta_ece_row(judge: str, population: str, df: pd.DataFrame) -> dict:
    return _row("delta_ece", judge, population, df, "delta_ece_verbose_minus_clean", ci_prefix="delta_ece")


def qwen_rows(results_dir: str, slug: str) -> list[dict]:
    judge = judge_name(slug)
    rq1 = pd.read_csv(f"{results_dir}/rq1_table_{slug}.csv")
    rq2 = pd.read_csv(f"{results_dir}/rq2_table_{slug}.csv")
    rq3a = pd.read_csv(f"{results_dir}/rq3a_table_{slug}.csv")
    rq3b = pd.read_csv(f"{results_dir}/rq3_table_{slug}.csv")
    population = "all"
    return [
        _row("auroc", judge, population, rq2[rq2["signal"] == "conf_bpe"], "auroc"),
        _row("overconfidence_gap", judge, population,
             rq1[(rq1["confidence_column"] == "conf_bpe_prob") & (rq1["verdict_definition"] == "judge_verdict")],
             "overconfidence_gap"),
        _row("flip_rate", judge, population, rq3a[rq3a["signal"] == "conf_bpe"], "flip_rate"),
        _delta_ece_row(judge, population, rq3b[rq3b["signal"] == "conf_bpe"]),
    ]


def kev_rows(results_dir: str, slug: str) -> list[dict]:
    judge = judge_name(slug)
    cal = pd.read_csv(f"{results_dir}/rq6_calibration_{slug}.csv")
    swap = pd.read_csv(f"{results_dir}/rq6_position_swap_{slug}.csv")
    verb = pd.read_csv(f"{results_dir}/rq6_verbosity_{slug}.csv")
    rows = []
    for regime, population in KEV_REGIMES.items():
        signal = "conf_kev_bpe"
        rows += [
            _row("auroc", judge, population, cal[(cal["regime"] == regime) & (cal["signal"] == signal)], "auroc"),
            _row("overconfidence_gap", judge, population,
                 cal[(cal["regime"] == regime) & (cal["signal"] == signal)], "overconfidence_gap"),
            _row("flip_rate", judge, population, swap[(swap["regime"] == regime) & (swap["signal"] == signal)],
                 "flip_rate"),
            _delta_ece_row(judge, population, verb[(verb["regime"] == regime) & (verb["signal"] == signal)]),
        ]
    return rows


def autoj_rows(results_dir: str, slug: str) -> list[dict]:
    judge = judge_name(slug)
    cal = pd.read_csv(f"{results_dir}/rq7_calibration_{slug}.csv")
    swap = pd.read_csv(f"{results_dir}/rq7_position_swap_{slug}.csv")
    verb = pd.read_csv(f"{results_dir}/rq7_verbosity_{slug}.csv")
    rows = []
    for turn in [1, 2]:
        population = f"turn {turn}"
        signal = "conf_sc_bpe_autoj"
        rows += [
            _row("auroc", judge, population, cal[(cal["turn"] == turn) & (cal["signal"] == signal)], "auroc"),
            _row("overconfidence_gap", judge, population,
                 cal[(cal["turn"] == turn) & (cal["signal"] == signal)], "overconfidence_gap"),
            _row("flip_rate", judge, population, swap[(swap["turn"] == turn) & (swap["signal"] == signal)],
                 "flip_rate"),
            _delta_ece_row(judge, population,
                           verb[(verb["turn"] == turn) & (verb["signal"] == "conf_sc_bpe_autoj_greedy")]),
        ]
    return rows


def main(config_path: str, kev_config_path: str, autoj_config_path: str) -> None:
    config = Config.from_yaml(config_path)
    kev_config = KevConfig.from_yaml(kev_config_path)
    autoj_config = AutojConfig.from_yaml(autoj_config_path)

    table = pd.DataFrame.from_records(
        qwen_rows(config.paths.results_dir, config.model_slug)
        + kev_rows(kev_config.paths.results_dir, kev_config.model_slug)
        + autoj_rows(autoj_config.paths.results_dir, autoj_config.model_slug)
    )
    table_path = f"{config.paths.results_dir}/judge_comparison.csv"
    table.to_csv(table_path, index=False)
    print(table.round(4).to_string(index=False))
    print(f"Wrote {len(table)} rows to {table_path}")

    panels = []
    for measure, title, xlabel, reference in MEASURES:
        measure_rows = table[table["measure"] == measure]
        panels.append(
            {
                "title": title,
                "xlabel": xlabel,
                "reference": reference,
                "rows": [
                    (r["judge"], r["judge"] if r["population"] == "all" else f"{r['judge']}, {r['population']}",
                     r["value"], r["ci_low"], r["ci_high"])
                    for _, r in measure_rows.iterrows()
                ],
            }
        )
    plot_judge_comparison(panels)
    plot_judge_comparison(panels, "judge_comparison_dark.png", dark=True)   # the README's dark-theme copy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--kev-config", required=True)
    parser.add_argument("--autoj-config", required=True)
    args = parser.parse_args()
    main(args.config, args.kev_config, args.autoj_config)
