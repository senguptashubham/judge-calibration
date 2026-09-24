"""RQ2: is any cheap uncertainty signal informative about error?

`python -m analysis.rq2 --config configs/run.yaml`

Population: load_rq1_items() - (clean, P1), human_label present. Scored
against judge_verdict/correct only (D7's deployed single-pass verdict); the
verdict_bidir comparison is RQ1's job.

For each of conf_verb, conf_lp, conf_sc, conf_bpe: the risk-coverage curve,
AURC, accuracy and κ at {90, 75, 50}% coverage (invariant 5), and
AUROC(uncertainty -> error), each with a cluster-bootstrap CI (invariant 2).
conf_ens's entropy decomposition is RQ5 (analysis/rq5.py).

Writes results/rq2_table_{model_slug}.csv and
results/figures/risk_coverage_{model_slug}.png (all four signals + oracle).
"""

import argparse

import numpy as np
import pandas as pd

from analysis.rq1 import SIGNALS, load_rq1_items
from src.boot import cluster_bootstrap
from src.config import Config
from src.metrics import auroc_error, aurc, cohens_kappa, oracle_risk_coverage, risk_coverage
from src.plots import plot_risk_coverage

COVERAGE_TARGETS = [0.90, 0.75, 0.50]


def _bootstrap_battery(items: pd.DataFrame, metric_fns: dict, seed: int) -> dict:
    """cluster_bootstrap per (name, stat_fn), flattened into
    name / name_ci_low / name_ci_high keys. A local copy of rq1.py's private
    helper, so this script doesn't depend on rq1's internals.
    """
    result = {}
    for name, fn in metric_fns.items():
        point, ci_low, ci_high = cluster_bootstrap(items, fn, "question_id", n=2000, seed=seed)
        result[name] = point
        result[f"{name}_ci_low"] = ci_low
        result[f"{name}_ci_high"] = ci_high
    return result


def _top_k_mask(confidences: np.ndarray, coverage: float) -> np.ndarray:
    """Mask for the `coverage` fraction of most-confident items.

    A rank cut, unlike the tie-safe value thresholds risk_coverage() uses
    for the plotted curve: those admit tied groups whole, so the curve's
    coverage levels rarely land exactly on 0.90/0.75/0.50. A rank cut always
    hits the target exactly, at the cost of a deterministic (stable-sort)
    tie-break at the boundary - the right trade-off for a single scalar
    query, and cheap enough to run 2000 times per bootstrap.
    """
    n = len(confidences)
    k = max(1, round(coverage * n))
    order = np.argsort(-confidences, kind="stable")  # descending confidence
    kept = np.zeros(n, dtype=bool)
    kept[order[:k]] = True
    return kept


def compute_signal_rq2_metrics(
    items: pd.DataFrame, signal: str, correct_col: str, verdict_col: str, seed: int
) -> dict:
    """AURC, AUROC(uncertainty -> error), and accuracy/κ at each of
    COVERAGE_TARGETS for one signal, each with a cluster-bootstrap CI.
    Uncertainty = 1 - signal throughout.
    """
    def _aurc(df: pd.DataFrame) -> float:
        uncertainty = 1 - df[signal].to_numpy(dtype=float)
        coverage, risk = risk_coverage(uncertainty, df[correct_col].to_numpy())
        return aurc(coverage, risk)

    def _auroc(df: pd.DataFrame) -> float:
        uncertainty = 1 - df[signal].to_numpy(dtype=float)
        return auroc_error(uncertainty, df[correct_col].to_numpy())

    metric_fns = {"aurc": _aurc, "auroc": _auroc}

    for target in COVERAGE_TARGETS:
        pct = round(target * 100)

        def _accuracy_at(df: pd.DataFrame, target=target) -> float:
            kept = _top_k_mask(df[signal].to_numpy(dtype=float), target)
            return df[correct_col].to_numpy()[kept].astype(float).mean()

        def _kappa_at(df: pd.DataFrame, target=target) -> float:
            kept = _top_k_mask(df[signal].to_numpy(dtype=float), target)
            return cohens_kappa(
                df[verdict_col].to_numpy()[kept], df["human_label"].to_numpy()[kept]
            )

        metric_fns[f"accuracy_at_{pct}"] = _accuracy_at
        metric_fns[f"kappa_at_{pct}"] = _kappa_at

    return _bootstrap_battery(items, metric_fns, seed)


def main(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    items = load_rq1_items(config.paths.items_parquet)

    correct_col = "correct"
    verdict_col = "judge_verdict"

    oracle_curve = oracle_risk_coverage(items[correct_col].to_numpy())
    print(
        f"Oracle AURC (reference upper bound): "
        f"{aurc(*oracle_curve):.4f}"
    )

    rows = []
    curves = {}
    for signal in SIGNALS:
        metrics = compute_signal_rq2_metrics(items, signal, correct_col, verdict_col, config.seed)
        rows.append({"signal": signal, **metrics})

        uncertainty = 1 - items[signal].to_numpy(dtype=float)
        curves[signal] = risk_coverage(uncertainty, items[correct_col].to_numpy())

    plot_risk_coverage(curves, oracle_curve, filename=f"risk_coverage_{config.model_slug}.png")

    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq2_table_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
