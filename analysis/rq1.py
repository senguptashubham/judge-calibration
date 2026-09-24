"""RQ1: is the judge's stated confidence calibrated?

`python -m analysis.rq1 --config configs/run.yaml`

On (clean, P1): ECE, MCE, Brier and its decomposition, overconfidence gap,
accuracy, κ, each with a cluster-bootstrap CI (invariant 2), for every row
of CALIBRATION_ROWS. Each row pairs a confidence with the verdict it is a
confidence IN - judge_verdict (the deployed AB pass) or verdict_bidir
(order-averaged, D7). conf_ens is RQ5's.

Writes results/rq1_table_{model_slug}.csv and
results/figures/reliability_{signal}_{model_slug}.png.
"""

import argparse

import pandas as pd

from src.boot import cluster_bootstrap, paired_cluster_bootstrap
from src.config import Config
from src.metrics import auroc_error, brier, brier_decomposition, cohens_kappa, ece, mce, overconfidence_gap
from src.plots import plot_reliability_diagram

# The ranking signals (AUROC, risk-coverage, RQ4's baseline).
SIGNALS = ["conf_verb", "conf_lp", "conf_sc", "conf_bpe"]

# The column each ranking signal's calibration metrics use. conf_bpe is
# 1 - entropy, not a probability, so it is scored through conf_bpe_prob
# (signals.py::prob_on_verdict).
CALIBRATION_FORM = {"conf_verb": "conf_verb", "conf_lp": "conf_lp", "conf_sc": "conf_sc", "conf_bpe": "conf_bpe_prob"}

# (signal, confidence column, verdict column, correctness column). conf_sc
# has no verdict_bidir row: its sampled draws exist in AB order only.
# conf_lp_bidir = max(p, 1 - p) is the verdict_bidir form of both conf_lp
# and conf_bpe, which is why it appears once.
CALIBRATION_ROWS = [
    *[(s, CALIBRATION_FORM[s], "judge_verdict", "correct") for s in SIGNALS],
    ("conf_verb", "conf_verb_bidir", "verdict_bidir", "correct_bidir"),
    ("conf_lp", "conf_lp_bidir", "verdict_bidir", "correct_bidir"),
]


def load_rq1_items(items_parquet: str) -> pd.DataFrame:
    """The RQ1 population (N=1,836): condition == clean AND prompt_variant
    == P1 (invariant 14), and human_label present - the 68 items with an
    exact 50/50 non-tie split have no majority, so `correct` is None there.
    """
    items = pd.read_parquet(items_parquet)
    items = items[(items["condition"] == "clean") & (items["prompt_variant"] == "P1")]
    items = items[items["human_label"].notna()]
    return items


def _bootstrap_battery(items: pd.DataFrame, metric_fns: dict, seed: int) -> dict:
    """cluster_bootstrap per (name, stat_fn), flattened into
    name / name_ci_low / name_ci_high keys.
    """
    result = {}
    for name, fn in metric_fns.items():
        point, ci_low, ci_high = cluster_bootstrap(items, fn, "question_id", n=2000, seed=seed)
        result[name] = point
        result[f"{name}_ci_low"] = ci_low
        result[f"{name}_ci_high"] = ci_high
    return result


def compute_verdict_metrics(items: pd.DataFrame, correct_col: str, verdict_col: str, seed: int) -> dict:
    """The metrics that depend only on the verdict definition, not on any
    signal: accuracy, κ, and Brier's uncertainty term obar*(1 - obar).
    Computed once per verdict definition rather than once per signal, since
    they are identical across signals.
    """
    def _accuracy(df: pd.DataFrame) -> float:
        return df[correct_col].astype(float).mean()

    def _kappa(df: pd.DataFrame) -> float:
        return cohens_kappa(df[verdict_col], df["human_label"])

    def _uncertainty(df: pd.DataFrame) -> float:
        obar = df[correct_col].astype(float).mean()
        return obar * (1 - obar)

    return _bootstrap_battery(
        items, {"accuracy": _accuracy, "kappa": _kappa, "uncertainty": _uncertainty}, seed
    )


def compute_signal_metrics(items: pd.DataFrame, signal: str, correct_col: str, n_bins: int, seed: int) -> dict:
    """The per-signal metrics: ECE, MCE, Brier, the decomposition's
    reliability and resolution terms (each with its own CI), and the
    overconfidence gap.
    """
    def _ece(df: pd.DataFrame) -> float:
        value, _ = ece(df[signal].to_numpy(), df[correct_col].to_numpy(), n_bins)
        return value

    def _mce(df: pd.DataFrame) -> float:
        value, _ = mce(df[signal].to_numpy(), df[correct_col].to_numpy(), n_bins)
        return value

    def _brier(df: pd.DataFrame) -> float:
        return brier(df[signal].to_numpy(), df[correct_col].to_numpy())

    def _reliability(df: pd.DataFrame) -> float:
        reliability, _, _ = brier_decomposition(df[signal].to_numpy(), df[correct_col].to_numpy(), n_bins)
        return reliability

    def _resolution(df: pd.DataFrame) -> float:
        _, resolution, _ = brier_decomposition(df[signal].to_numpy(), df[correct_col].to_numpy(), n_bins)
        return resolution

    def _overconfidence_gap(df: pd.DataFrame) -> float:
        return overconfidence_gap(df[signal].to_numpy(), df[correct_col].to_numpy())

    return _bootstrap_battery(
        items,
        {
            "ece": _ece,
            "mce": _mce,
            "brier": _brier,
            "reliability": _reliability,
            "resolution": _resolution,
            "overconfidence_gap": _overconfidence_gap,
        },
        seed,
    )


def compute_verdict_gap(items: pd.DataFrame, seed: int) -> dict:
    """Paired cluster-bootstrap CI on accuracy(verdict_bidir) -
    accuracy(judge_verdict). The same items scored two ways is a paired
    comparison (invariant 3): renaming each correctness column to a shared
    name lets one stat_fn serve both sides of paired_cluster_bootstrap.
    """
    df_bidir = items.rename(columns={"correct_bidir": "score"})
    df_judge = items.rename(columns={"correct": "score"})

    def _accuracy(df: pd.DataFrame) -> float:
        return df["score"].astype(float).mean()

    diff, ci_low, ci_high = paired_cluster_bootstrap(
        df_bidir, df_judge, _accuracy, "question_id", n=2000, seed=seed
    )
    return {
        "accuracy_gap_bidir_minus_judge": diff,
        "accuracy_gap_ci_low": ci_low,
        "accuracy_gap_ci_high": ci_high,
    }


def main(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    items = load_rq1_items(config.paths.items_parquet)

    verdict_metrics = {
        verdict_col: compute_verdict_metrics(items, correct_col, verdict_col, config.seed)
        for verdict_col, correct_col in [("judge_verdict", "correct"), ("verdict_bidir", "correct_bidir")]
    }
    rows = []
    for signal, confidence_col, verdict_col, correct_col in CALIBRATION_ROWS:
        signal_metrics = compute_signal_metrics(items, confidence_col, correct_col, config.n_bins, config.seed)
        rows.append(
            {
                "signal": signal,
                "confidence_column": confidence_col,
                "verdict_definition": verdict_col,
                **verdict_metrics[verdict_col],
                **signal_metrics,
            }
        )

    bidir_form = {s: c for s, c, v, _ in CALIBRATION_ROWS if v == "verdict_bidir"}
    bidir_form["conf_bpe"] = "conf_lp_bidir"
    for signal in SIGNALS:
        overlay = None
        if signal in bidir_form:
            overlay = (items[bidir_form[signal]].to_numpy(), items["correct_bidir"].to_numpy(), "order-averaged verdict")
        plot_reliability_diagram(
            items[CALIBRATION_FORM[signal]].to_numpy(),
            items["correct"].to_numpy(),
            signal_name=signal,
            n_bins=config.n_bins,
            model_slug=config.model_slug,
            label="AB verdict",
            overlay=overlay,
        )

    gap = compute_verdict_gap(items, config.seed)
    print(
        "Paired accuracy gap (verdict_bidir - judge_verdict): "
        f"{gap['accuracy_gap_bidir_minus_judge']:.4f} "
        f"[{gap['accuracy_gap_ci_low']:.4f}, {gap['accuracy_gap_ci_high']:.4f}]"
    )

    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq1_table_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
