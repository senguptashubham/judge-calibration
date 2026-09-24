"""RQ3: does the judge's uncertainty flag bias-induced errors, or is the
fooled judge confident?

`python -m analysis.rq3 --config configs/run.yaml`

RQ3a - position bias, on RQ1's population (clean, P1, N=1,836): the flip
rate between AB and BA, and each signal's mean confidence on flipped vs
unflipped items. The gap is a difference between two DISJOINT subsets of
one item set, so it uses one cluster_bootstrap with both group means
computed inside each replicate - paired_cluster_bootstrap needs two aligned
copies of the same items, which this is not.
Writes results/rq3a_table_{model_slug}.csv and
results/figures/rq3a_confidence_gap_{model_slug}.png.

RQ3b - verbosity, clean(P1) vs verbose(P1): the same 1,836 items under two
conditions, so a genuinely paired comparison (invariant 3). Paired
cluster-bootstrap deltas (verbose - clean) in ECE, accuracy, and
AUROC(uncertainty -> error), scored against judge_verdict (D7). conf_sc is
excluded - verbose has no sampled draws at all (D21).
Writes results/rq3_table_{model_slug}.csv and
results/figures/rq3b_deltas_{model_slug}.png.
"""

import argparse

import pandas as pd

from analysis.rq1 import SIGNALS, load_rq1_items
from src.boot import cluster_bootstrap, paired_cluster_bootstrap
from src.config import Config
from src.metrics import auroc_error, ece
from src.plots import plot_rq3a_confidence_gap, plot_rq3b_deltas

RQ3B_SIGNALS = ["conf_verb", "conf_lp", "conf_bpe"]  # conf_sc dropped, D21


def compute_flip_rate(items: pd.DataFrame, seed: int) -> dict:
    """mean(flipped), with a cluster-bootstrap CI."""
    def _flip_rate(df: pd.DataFrame) -> float:
        return df["flipped"].astype(float).mean()

    point, ci_low, ci_high = cluster_bootstrap(items, _flip_rate, "question_id", n=2000, seed=seed)
    return {"flip_rate": point, "flip_rate_ci_low": ci_low, "flip_rate_ci_high": ci_high}


def compute_confidence_gap(items: pd.DataFrame, signal: str, seed: int) -> dict:
    """mean(signal | flipped) - mean(signal | unflipped), with a
    cluster-bootstrap CI. A negative gap whose CI excludes 0 means the
    signal drops exactly where order changed the verdict - it notices its
    own position-bias errors. `items["flipped"]` must be bool.

    Returns the real per-group means plus the bootstrapped gap and CI.
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


def load_rq3b_items(items_parquet: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(clean_items, verbose_items): P1 rows with human_label present, per
    condition. Raises if the two item sets differ - a partial verbose
    re-run should fail loudly here rather than silently unpair the bootstrap.
    """
    items = pd.read_parquet(items_parquet)
    items = items[items["prompt_variant"] == "P1"]
    items = items[items["human_label"].notna()]

    clean_items = items[items["condition"] == "clean"]
    verbose_items = items[items["condition"] == "verbose"]

    clean_ids = set(clean_items["item_id"])
    verbose_ids = set(verbose_items["item_id"])
    if clean_ids != verbose_ids:
        raise ValueError(
            "clean/P1 and verbose/P1 item populations differ - RQ3b's paired "
            f"comparison requires the same items on both sides. Only in clean: "
            f"{clean_ids - verbose_ids}; only in verbose: {verbose_ids - clean_ids}"
        )

    return clean_items, verbose_items


def compute_signal_rq3b_metrics(
    clean_items: pd.DataFrame, verbose_items: pd.DataFrame, signal: str, correct_col: str, n_bins: int, seed: int
) -> dict:
    """Paired cluster-bootstrap deltas (verbose - clean) in ECE, accuracy,
    and AUROC for one signal, plus each side's real value. A positive ΔECE
    means worse calibration under verbose; a negative ΔAUROC means the
    signal is less informative about errors.
    """
    def _ece(df: pd.DataFrame) -> float:
        value, _ = ece(df[signal].to_numpy(), df[correct_col].to_numpy(), n_bins)
        return value

    def _accuracy(df: pd.DataFrame) -> float:
        return df[correct_col].astype(float).mean()

    def _auroc(df: pd.DataFrame) -> float:
        uncertainty = 1 - df[signal].to_numpy(dtype=float)
        return auroc_error(uncertainty, df[correct_col].to_numpy())

    result = {}
    for name, fn in [("ece", _ece), ("accuracy", _accuracy), ("auroc", _auroc)]:
        diff, ci_low, ci_high = paired_cluster_bootstrap(
            verbose_items, clean_items, fn, "question_id", n=2000, seed=seed
        )
        result[f"{name}_clean"] = fn(clean_items)
        result[f"{name}_verbose"] = fn(verbose_items)
        result[f"delta_{name}_verbose_minus_clean"] = diff
        result[f"delta_{name}_ci_low"] = ci_low
        result[f"delta_{name}_ci_high"] = ci_high

    return result


def run_rq3b(config: Config) -> pd.DataFrame:
    clean_items, verbose_items = load_rq3b_items(config.paths.items_parquet)
    print(f"RQ3b population: N={len(clean_items)} paired items (clean/P1 vs verbose/P1, human_label present)")

    rows = []
    for signal in RQ3B_SIGNALS:
        metrics = compute_signal_rq3b_metrics(clean_items, verbose_items, signal, "correct", config.n_bins, config.seed)
        rows.append({"signal": signal, **metrics})

    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq3_table_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    plot_rq3b_deltas(
        signals=table["signal"].tolist(),
        delta_ece=table["delta_ece_verbose_minus_clean"].to_numpy(),
        ece_ci_low=table["delta_ece_ci_low"].to_numpy(),
        ece_ci_high=table["delta_ece_ci_high"].to_numpy(),
        delta_auroc=table["delta_auroc_verbose_minus_clean"].to_numpy(),
        auroc_ci_low=table["delta_auroc_ci_low"].to_numpy(),
        auroc_ci_high=table["delta_auroc_ci_high"].to_numpy(),
        model_slug=config.model_slug,
    )

    headline = table[table["signal"] == "conf_verb"].iloc[0]
    print(
        "conf_verb ECE gap (verbose - clean): "
        f"{headline['delta_ece_verbose_minus_clean']:.4f} "
        f"[{headline['delta_ece_ci_low']:.4f}, {headline['delta_ece_ci_high']:.4f}]"
    )

    return table


def run_rq3a(config: Config) -> pd.DataFrame:
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
    table_path = f"{config.paths.results_dir}/rq3a_table_{config.model_slug}.csv"
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

    return table


def main(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    run_rq3a(config)
    print()
    run_rq3b(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
