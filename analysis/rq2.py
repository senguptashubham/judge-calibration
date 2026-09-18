"""RQ2: is any cheap uncertainty signal informative about error? See
TASKS.md task 3.2.

Plain script, not a notebook - same CLI pattern as analysis/rq1.py:
`python -m analysis.rq2 --config configs/run.yaml`.

Reads results/items.parquet filtered to (clean, P1) via load_rq1_items()
(analysis/rq1.py) - invariant 14's filter is identical for RQ2, so it's
reused rather than re-implemented.

Scope: unlike RQ1, this does NOT pair against verdict_bidir - RQ2 asks
"is signal X informative about the judge's actual deployed error rate,"
which is the single-pass judge_verdict/correct definition (D7). The
verdict-definition comparison itself is RQ1's job (compute_verdict_gap).

For each of the four ORIGINAL confidence signals - conf_verb, conf_lp,
conf_sc, conf_bpe (conf_ens's entropy decomposition is RQ5, task 3.2b, a
separate script reusing threshold_sweep()) - computes:
  - the risk-coverage curve itself (for the figure)
  - AURC
  - accuracy@{90,75,50}% coverage
  - kappa@{90,75,50}% coverage (invariant 5 - never accuracy without kappa)
  - AUROC(uncertainty -> error)
every scalar with a cluster-bootstrap CI (question_id, invariant 2).

Writes results/rq2_table_{model_slug}.csv (one row per signal) and
results/figures/risk_coverage_{model_slug}.png (src/plots.py's
plot_risk_coverage - all four signals + the oracle, one axis - task 3.2's
thesis figure). model_slug = Config.model_slug, so a second judge model
never overwrites the first's output.
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
    """Same pattern as analysis/rq1.py's own helper of the same name - runs
    cluster_bootstrap once per (name, stat_fn) pair and flattens the result
    into name/name_ci_low/name_ci_high keys. Kept as a local copy rather
    than importing rq1's (private, underscore-prefixed) version - this
    script shouldn't depend on rq1.py's internals surviving unchanged.
    """
    result = {}
    for name, fn in metric_fns.items():
        point, ci_low, ci_high = cluster_bootstrap(items, fn, "question_id", n=2000, seed=seed)
        result[name] = point
        result[f"{name}_ci_low"] = ci_low
        result[f"{name}_ci_high"] = ci_high
    return result


def _top_k_mask(confidences: np.ndarray, coverage: float) -> np.ndarray:
    """Boolean mask for "the top `coverage` fraction of items by
    confidence" - the most-confident-first rank cut used for the
    accuracy@coverage / kappa@coverage point queries.

    Deliberately NOT the same tie-safe value-threshold stepping
    risk_coverage()/threshold_sweep() use for the plotted curve: those
    walk unique signal VALUES so a tied group always enters together,
    which is exactly right for drawing an unambiguous curve, but it means
    the curve's own coverage levels are whatever the data's ties happen to
    produce - they don't generally land on exactly 0.90/0.75/0.50. A rank
    cut (argsort, most-confident k items) always hits the target coverage
    exactly, at the cost of an arbitrary (but deterministic, stable-sort)
    tiebreak for whichever items sit right at the cutoff boundary. That
    tradeoff is the right one specifically for a single scalar query like
    "accuracy at 90% coverage" - it's also cheap enough to call 2000x per
    bootstrap replicate, unlike re-running the full threshold sweep for a
    single point every time.

    Args:
        confidences: stated confidence per item, in [0, 1].
        coverage: target fraction of items to keep, in (0, 1].

    Returns:
        Boolean mask, same length as confidences, True for the kept items.
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
    """RQ2's full scalar battery for one confidence signal: AURC,
    AUROC(uncertainty -> error), and accuracy/kappa at each of
    COVERAGE_TARGETS - every value with a cluster-bootstrap CI.

    Args:
        items: load_rq1_items()'s output (already filtered).
        signal: one of SIGNALS - the confidence column name.
        correct_col: "correct" (judge_verdict's correctness column - see
            module docstring on why RQ2 doesn't also score correct_bidir).
        verdict_col: "judge_verdict".
        seed: config.seed.

    Returns:
        aurc/auroc, and accuracy_at_{pct}/kappa_at_{pct} for pct in
        {90, 75, 50}, each with a _ci_low/_ci_high pair.
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
    table_path = f"results/rq2_table_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
