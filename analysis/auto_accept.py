"""What would auto-accepting the judge's confident verdicts cost?

`python -m analysis.auto_accept --config configs/run.yaml --kev-config configs/run_kev.yaml --autoj-config configs/run_autoj.yaml`

A pipeline that accepts every verdict with confidence >= a threshold and
escalates the rest: for each judge, signal and condition, how many
verdicts it accepts and what share of the judge's WRONG verdicts slip
through unreviewed (src/metrics.py::auto_accept_stats). A post-hoc,
descriptive analysis - not preregistered, and not a new hypothesis test.

Only probability-scaled signals are used, each scored against the verdict
it is a confidence in (D7 amendment): a threshold is meaningless on 1 - H.
The Bayesian and logistic-regression meta-models enter through their
out-of-fold P(correct) (analysis/rq4.py, RQ4's N=1,819, clean only).
Populations are those of RQ1/RQ3b, RQ6 and RQ7, with every item that has a
verdict and the signal; kev-8b's coverage regimes and auto-j's turns are
pooled, and auto-j's padded calls that produced no verdict at all
(39.8% on turn 2) are outside the population, so its padded numbers
describe only the calls it answered.

Writes results/auto_accept_curves.csv (point estimates, thresholds
0.50-0.99), results/auto_accept.csv (headline thresholds with 95%
cluster-bootstrap CIs over question_id), and results/figures/auto_accept.png.
"""

import argparse

import numpy as np
import pandas as pd

from src.boot import cluster_bootstrap
from src.config import Config
from src.judge_autoj import AutojConfig
from src.judge_kev import KevConfig
from src.metrics import auto_accept_stats
from src.plots import judge_name, plot_auto_accept

THRESHOLDS = np.round(np.arange(0.50, 0.995, 0.01), 2)
HEADLINE_THRESHOLDS = [0.8, 0.9, 0.95]
CI_METRICS = ["accepted_share", "slip_through", "error_among_accepted"]

# (signal column, verdict column, correctness column, conditions)
QWEN_SIGNALS = [
    ("conf_verb", "judge_verdict", "correct", ["clean", "verbose"]),
    ("conf_lp", "judge_verdict", "correct", ["clean", "verbose"]),
    ("conf_sc", "judge_verdict", "correct", ["clean"]),
    ("conf_bpe_prob", "judge_verdict", "correct", ["clean", "verbose"]),
    ("conf_ens_prob", "judge_verdict", "correct", ["clean"]),
    ("conf_verb_bidir", "verdict_bidir", "correct_bidir", ["clean", "verbose"]),
    ("conf_lp_bidir", "verdict_bidir", "correct_bidir", ["clean", "verbose"]),
]
KEV_SIGNALS = [
    ("conf_kev", "judge_verdict", "correct", ["clean", "verbose"]),
    ("conf_kev_bpe_prob", "judge_verdict", "correct", ["clean", "verbose"]),
]
AUTOJ_SIGNALS = [
    ("conf_sc_autoj", "judge_verdict", "correct", ["clean"]),
    ("conf_sc_bpe_autoj_prob", "judge_verdict", "correct", ["clean"]),
    ("conf_sc_bpe_autoj_greedy_prob", "judge_verdict", "correct", ["clean", "verbose"]),
]
META_SIGNALS = ["p_correct_bayesian", "p_correct_logreg"]


def _series(items: pd.DataFrame, judge: str, signal: str, verdict_col: str, correct_col: str,
            condition: str) -> pd.DataFrame:
    """One (judge, signal, condition) population in a shared shape:
    question_id, confidence, correct, verdict, human_label."""
    df = items[items[signal].notna() & items[correct_col].notna() & items[verdict_col].notna()]
    return pd.DataFrame(
        {
            "judge": judge,
            "signal": signal,
            "condition": condition,
            "question_id": df["question_id"].to_numpy(),
            "confidence": df[signal].to_numpy(dtype=float),
            "correct": df[correct_col].astype(bool).to_numpy(),
            "verdict": df[verdict_col].to_numpy(),
            "human_label": df["human_label"].to_numpy(),
        }
    )


def load_series(config: Config, kev_config: KevConfig, autoj_config: AutojConfig) -> list[pd.DataFrame]:
    series = []

    qwen = pd.read_parquet(config.paths.items_parquet)
    qwen = qwen[(qwen["prompt_variant"] == "P1") & qwen["human_label"].notna()]
    judge = judge_name(config.model_slug)
    for signal, verdict_col, correct_col, conditions in QWEN_SIGNALS:
        for condition in conditions:
            series.append(_series(qwen[qwen["condition"] == condition], judge, signal, verdict_col, correct_col, condition))

    oof = pd.read_csv(f"{config.paths.results_dir}/rq4_bayesian_oof_{config.model_slug}.csv")
    for signal in META_SIGNALS:
        series.append(_series(oof, judge, signal, "judge_verdict", "correct", "clean"))

    for cfg, signals in [(kev_config, KEV_SIGNALS), (autoj_config, AUTOJ_SIGNALS)]:
        items = pd.read_parquet(cfg.paths.items_parquet)
        items = items[items["human_label"].notna()]
        judge = judge_name(cfg.model_slug)
        for signal, verdict_col, correct_col, conditions in signals:
            for condition in conditions:
                series.append(_series(items[items["condition"] == condition], judge, signal, verdict_col, correct_col, condition))
    return series


def curve_rows(df: pd.DataFrame) -> list[dict]:
    """Point estimates of auto_accept_stats() over THRESHOLDS for one series."""
    rows = []
    for threshold in THRESHOLDS:
        stats = auto_accept_stats(df["confidence"], df["correct"], threshold, df["verdict"], df["human_label"])
        rows.append({"judge": df["judge"].iloc[0], "signal": df["signal"].iloc[0],
                     "condition": df["condition"].iloc[0], "threshold": threshold, **stats})
    return rows


def headline_rows(df: pd.DataFrame, seed: int) -> list[dict]:
    """auto_accept_stats() at HEADLINE_THRESHOLDS, with a cluster-bootstrap
    CI (resampling question_id, invariant 2) on each ratio in CI_METRICS."""
    df = df.reset_index(drop=True)
    rows = []
    for threshold in HEADLINE_THRESHOLDS:
        stats = auto_accept_stats(df["confidence"], df["correct"], threshold, df["verdict"], df["human_label"])
        row = {"judge": df["judge"].iloc[0], "signal": df["signal"].iloc[0],
               "condition": df["condition"].iloc[0], "threshold": threshold, **stats}
        for metric in CI_METRICS:
            def _stat(sample: pd.DataFrame, metric: str = metric) -> float:
                return auto_accept_stats(sample["confidence"], sample["correct"], threshold)[metric]
            _, ci_low, ci_high = cluster_bootstrap(df, _stat, "question_id", n=2000, seed=seed)
            row[f"{metric}_ci_low"] = ci_low
            row[f"{metric}_ci_high"] = ci_high
        rows.append(row)
    return rows


def main(config_path: str, kev_config_path: str, autoj_config_path: str) -> None:
    config = Config.from_yaml(config_path)
    kev_config = KevConfig.from_yaml(kev_config_path)
    autoj_config = AutojConfig.from_yaml(autoj_config_path)
    series = load_series(config, kev_config, autoj_config)

    curves = pd.DataFrame.from_records([row for df in series for row in curve_rows(df)])
    curves_path = f"{config.paths.results_dir}/auto_accept_curves.csv"
    curves.to_csv(curves_path, index=False)
    print(f"Wrote {len(curves)} rows to {curves_path}")

    headline = pd.DataFrame.from_records([row for df in series for row in headline_rows(df, config.seed)])
    headline_path = f"{config.paths.results_dir}/auto_accept.csv"
    headline.to_csv(headline_path, index=False)
    print(f"Wrote {len(headline)} rows to {headline_path}")

    at_09 = headline[headline["threshold"] == 0.9]
    print(at_09[["judge", "signal", "condition", "n", "accepted_share", "slip_through", "slip_through_ci_low",
                 "slip_through_ci_high", "error_among_accepted"]].round(3).to_string(index=False))

    plot_auto_accept(curves)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--kev-config", required=True)
    parser.add_argument("--autoj-config", required=True)
    args = parser.parse_args()
    main(args.config, args.kev_config, args.autoj_config)
