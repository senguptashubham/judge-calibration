"""RQ1: is the judge's stated confidence calibrated?

`python -m analysis.rq1 --config configs/run.yaml`

For each of conf_verb, conf_lp, conf_sc, conf_bpe on (clean, P1): ECE, MCE,
Brier and its decomposition, overconfidence gap, accuracy, κ - scored
against both verdict definitions (D7: judge_verdict, the deployed single
pass, and verdict_bidir, order-averaged), each with a cluster-bootstrap CI
(invariant 2). conf_ens is RQ5's.

Writes results/rq1_table_{model_slug}.csv and
results/figures/reliability_{signal}_{model_slug}.png (both verdict
definitions overlaid on each).
"""

import argparse

import pandas as pd

from src.boot import cluster_bootstrap, paired_cluster_bootstrap
from src.config import Config
from src.metrics import auroc_error, brier, brier_decomposition, cohens_kappa, ece, mce, overconfidence_gap
from src.plots import plot_reliability_diagram

SIGNALS = ["conf_verb", "conf_lp", "conf_sc", "conf_bpe"]
VERDICT_DEFINITIONS = [
    ("judge_verdict", "correct"),
    ("verdict_bidir", "correct_bidir"),
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

    rows = []
    for verdict_col, correct_col in VERDICT_DEFINITIONS:
        verdict_metrics = compute_verdict_metrics(items, correct_col, verdict_col, config.seed)
        for signal in SIGNALS:
            signal_metrics = compute_signal_metrics(items, signal, correct_col, config.n_bins, config.seed)
            rows.append(
                {
                    "signal": signal,
                    "verdict_definition": verdict_col,
                    **verdict_metrics,
                    **signal_metrics,
                }
            )

    for signal in SIGNALS:
        plot_reliability_diagram(
            items[signal].to_numpy(),
            items["correct"].to_numpy(),
            signal_name=signal,
            n_bins=config.n_bins,
            model_slug=config.model_slug,
            correct_bidir=items["correct_bidir"].to_numpy(),
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
