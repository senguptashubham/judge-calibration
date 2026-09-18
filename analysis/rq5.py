"""RQ5, task 3.2b: does the epistemic component of the judge-level entropy
decomposition beat total entropy as a threshold-sweep filter? See
TASKS.md task 3.2b (D20, D23). Only this one piece of RQ5 lands in Week 3 -
the rest (distillation, human-disagreement validation) is Week 5; this
module will grow then rather than being renamed.

`python -m analysis.rq5 --config configs/run.yaml`.

Reads results/items.parquet via load_rq1_items() (clean, P1, human_label
not null) - conf_ens/ens_entropy_* are populated exactly on that slice
(Gate 2: 1904/1904 clean/P1 rows), so this is the same filter RQ1/RQ2 use,
not a coincidence.

For each of ens_entropy_total, ens_entropy_aleatoric, ens_entropy_epistemic,
runs threshold_sweep() (task 3.1b) - these are already uncertainty-typed
(higher entropy = more uncertain), so unlike the confidence signals in
rq2.py, no `1 - x` sign flip is needed. ECE-on-the-retained-set is computed
against conf_ens (1 - ens_entropy_total) throughout, for all three sweeps -
using the same confidence column across all three keeps the three curves
answering a comparable question ("how calibrated is the ensemble's own
confidence on what THIS filter keeps"), rather than three different,
unrelated calibration stories.

Writes results/rq5_threshold_sweep_table_{model_slug}.csv (long format, one
row per (entropy_signal, threshold)) and
results/figures/entropy_threshold_sweep_{model_slug}.png (src/plots.py's
plot_risk_coverage, reused as-is - task 3.2b's DoD figure). model_slug =
Config.model_slug, so a second judge model never overwrites the first's
output.

Prints the preregistered-prediction verdict (D23): epistemic thresholding
should beat total thresholding on AURC, since epistemic is the reducible
part of the uncertainty and total conflates it with aleatoric noise a
threshold can never remove. Tested via a paired cluster-bootstrap on
AURC(epistemic) - AURC(total) over the SAME items (not two independent
samples - invariant 3's paired-comparison logic applies to two signals on
one item set exactly like it applies to two conditions), using the same
rename-to-a-shared-column trick analysis/rq1.py's compute_verdict_gap uses
for its own single-item-set paired comparison.
"""

import argparse

import pandas as pd

from analysis.rq1 import load_rq1_items
from src.boot import paired_cluster_bootstrap
from src.config import Config
from src.metrics import aurc, oracle_risk_coverage, risk_coverage, threshold_sweep
from src.plots import plot_risk_coverage

ENTROPY_SIGNALS = ["ens_entropy_total", "ens_entropy_aleatoric", "ens_entropy_epistemic"]


def load_rq5_items(items_parquet: str) -> pd.DataFrame:
    """load_rq1_items() plus a defensive dropna on the entropy columns.

    Expected to be a no-op on real data (Gate 2 confirmed conf_ens/entropy
    populated for all 1904 clean/P1 rows) - kept explicit rather than
    silently trusting that invariant holds forever, the same spirit as
    load_rq1_items()'s own human_label.notna() filter.
    """
    items = load_rq1_items(items_parquet)
    return items[items[ENTROPY_SIGNALS].notna().all(axis=1)]


def compute_entropy_sweep(items: pd.DataFrame, signal: str, n_bins: int) -> pd.DataFrame:
    """threshold_sweep() for one entropy signal, as a tidy DataFrame with
    the signal name attached - point estimates only (no bootstrap CI; see
    module docstring for why the one number this task actually needs to
    defend statistically, the epistemic-vs-total AURC gap, gets its own
    paired bootstrap in main() instead of bootstrapping all of this table).
    """
    result = threshold_sweep(
        signal=items[signal].to_numpy(dtype=float),
        correct=items["correct"].to_numpy(),
        judge_verdict=items["judge_verdict"].to_numpy(),
        human_label=items["human_label"].to_numpy(),
        confidences=items["conf_ens"].to_numpy(dtype=float),
        n_bins=n_bins,
    )
    table = pd.DataFrame(result)
    table.insert(0, "entropy_signal", signal)
    return table


def compute_epistemic_vs_total_gap(items: pd.DataFrame, seed: int) -> dict:
    """Paired cluster-bootstrap CI on AURC(epistemic) - AURC(total), the
    preregistered prediction's actual statistical test (D23): epistemic
    thresholding beats total thresholding means this difference is
    negative, and the CI excluding 0 is what makes that a defensible claim
    rather than an eyeballed gap between two point estimates.

    Same rows scored by two different signal columns is a paired
    comparison, not two independent samples (invariant 3's logic) - reuses
    the rename-to-"score" trick analysis/rq1.py's compute_verdict_gap uses
    for exactly this same shape of problem (one item set, two columns).
    """
    def _aurc(df: pd.DataFrame) -> float:
        uncertainty = df["score"].to_numpy(dtype=float)
        coverage, risk = risk_coverage(uncertainty, df["correct"].to_numpy())
        return aurc(coverage, risk)

    df_epistemic = items.rename(columns={"ens_entropy_epistemic": "score"})
    df_total = items.rename(columns={"ens_entropy_total": "score"})

    diff, ci_low, ci_high = paired_cluster_bootstrap(
        df_epistemic, df_total, _aurc, "question_id", n=2000, seed=seed
    )
    return {"aurc_gap_epistemic_minus_total": diff, "ci_low": ci_low, "ci_high": ci_high}


def main(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    items = load_rq5_items(config.paths.items_parquet)

    oracle_curve = oracle_risk_coverage(items["correct"].to_numpy())

    tables = []
    curves = {}
    aurc_by_signal = {}
    for signal in ENTROPY_SIGNALS:
        table = compute_entropy_sweep(items, signal, config.n_bins)
        tables.append(table)

        coverage = table["coverage"].to_numpy()
        risk = 1 - table["accuracy"].to_numpy()
        curves[signal] = (coverage, risk)
        aurc_by_signal[signal] = aurc(coverage, risk)

    plot_risk_coverage(
        curves, oracle_curve,
        filename=f"entropy_threshold_sweep_{config.model_slug}.png",
        title="Entropy threshold sweep: RQ5",
    )

    full_table = pd.concat(tables, ignore_index=True)
    table_path = f"results/rq5_threshold_sweep_table_{config.model_slug}.csv"
    full_table.to_csv(table_path, index=False)
    print(f"Wrote {len(full_table)} rows to {table_path}")

    for signal, value in aurc_by_signal.items():
        print(f"AURC({signal}) = {value:.4f}")

    gap = compute_epistemic_vs_total_gap(items, config.seed)
    held = gap["ci_high"] < 0
    verdict = "HELD" if held else "DID NOT HOLD"
    print(
        f"Preregistered prediction (epistemic beats total, D23): {verdict} - "
        f"AURC(epistemic) - AURC(total) = "
        f"{gap['aurc_gap_epistemic_minus_total']:.4f} "
        f"[{gap['ci_low']:.4f}, {gap['ci_high']:.4f}]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
