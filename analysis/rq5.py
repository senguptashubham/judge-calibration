"""RQ5: does marginalizing over the judge prompt improve uncertainty
quality, and does that survive distillation to a single call? (D20, D23)

`python -m analysis.rq5 --config configs/run.yaml --task {threshold_sweep,distillation,human_disagreement}`

threshold_sweep (task 3.2b) - RQ1's population. threshold_sweep() over
each of ens_entropy_{total,aleatoric,epistemic} (already uncertainty-typed,
no 1 - x flip), with ECE on the retained set always scored against
conf_ens so the three curves answer one comparable question. Tests the
preregistered prediction that epistemic thresholding beats total (a
paired cluster-bootstrap on the AURC gap - one item set, two signals).
Writes results/rq5_threshold_sweep_table_{model_slug}.csv and
results/figures/entropy_threshold_sweep_{model_slug}.png.

distillation (task 5.9d) - RQ4's population (N=1,819, a strict subset of
RQ5's 1,836), so both arms are scored on the same items. Teacher: the
3-call ensemble's conf_ens and ens_entropy_epistemic. Student: the 1-call
Bayesian meta-model (analysis/rq4.py::compute_bayesian_arm, Tier A) and
its own posterior epistemic entropy. AUROC gaps get paired
cluster-bootstrap CIs; ECE is reported side by side. "% of the edge
retained" measures AUROC above chance (0.5). Writes
results/rq5_distillation_{model_slug}.csv and a figure.

human_disagreement (task 5.9e) - human_disagreement.py's N=595
population. Spearman of the ensemble's aleatoric and epistemic entropy
against d_human. d_human = |frac_prefer_a - 0.5| is CONSENSUS strength, so
"aleatoric tracks human disagreement" means a NEGATIVE correlation
(TASKS.md's original wording had the direction backwards). Writes
results/rq5_human_disagreement_{model_slug}.csv and a figure.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.human_disagreement import compute_d_human_correlation, load_disagreement_items
from analysis.rq1 import load_rq1_items
from analysis.rq4 import compute_bayesian_arm
from src.bayesian import posterior_predictive_entropy_decomposition
from src.boot import cluster_bootstrap, paired_cluster_bootstrap
from src.config import Config
from src.features import load_rq4_population
from src.metrics import aurc, auroc_error, ece, oracle_risk_coverage, risk_coverage, threshold_sweep
from src.plots import plot_d_human_correlations, plot_risk_coverage, plot_rq5_distillation

ENTROPY_SIGNALS = ["ens_entropy_total", "ens_entropy_aleatoric", "ens_entropy_epistemic"]


def load_rq5_items(items_parquet: str) -> pd.DataFrame:
    """load_rq1_items() plus a defensive dropna on the entropy columns
    (a no-op on the real data, where they are populated for every row).
    """
    items = load_rq1_items(items_parquet)
    return items[items[ENTROPY_SIGNALS].notna().all(axis=1)]


def compute_entropy_sweep(items: pd.DataFrame, signal: str, n_bins: int) -> pd.DataFrame:
    """threshold_sweep() for one entropy signal as a tidy table (point
    estimates; the AURC gap gets its own CI in compute_epistemic_vs_total_gap).
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
    """Paired cluster-bootstrap CI on AURC(epistemic) - AURC(total). The
    D23 prediction holds if this is negative with a CI below 0. Same items,
    two columns, so each is renamed to "score" for one shared stat_fn.
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


def main_threshold_sweep(config_path: str) -> None:
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
    table_path = f"{config.paths.results_dir}/rq5_threshold_sweep_table_{config.model_slug}.csv"
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


# --- Task 5.9d: distillation ----------------------------------------------


def compute_ensemble_metrics(items: pd.DataFrame, n_bins: int, seed: int) -> dict:
    """conf_ens's AUROC(uncertainty -> error), with a cluster-bootstrap CI,
    and its ECE as P(wrong).
    """
    def _auroc(df: pd.DataFrame) -> float:
        uncertainty = 1 - df["conf_ens"].to_numpy(dtype=float)
        return auroc_error(uncertainty, df["correct"].to_numpy())

    auroc_point, auroc_low, auroc_high = cluster_bootstrap(items, _auroc, "question_id", seed=seed)

    p_wrong = 1 - items["conf_ens"].to_numpy(dtype=float)
    is_wrong = 1 - items["correct"].astype(int).to_numpy()
    ece_value, _ = ece(p_wrong, is_wrong, n_bins)

    return {"auroc": auroc_point, "auroc_ci_low": auroc_low, "auroc_ci_high": auroc_high, "ece": ece_value}


def build_distillation_comparison_df(
    population: pd.DataFrame, bayesian_mean_oof_correct: np.ndarray, bayesian_epistemic: np.ndarray
) -> pd.DataFrame:
    """One row per item with both arms' uncertainty and epistemic columns,
    positionally aligned to `population`, so every paired gap scores the
    same rows.
    """
    return pd.DataFrame(
        {
            "question_id": population["question_id"].to_numpy(),
            "correct": population["correct"].to_numpy(),
            "ensemble_uncertainty": 1 - population["conf_ens"].to_numpy(dtype=float),
            "bayesian_uncertainty": 1 - bayesian_mean_oof_correct,
            "ensemble_epistemic": population["ens_entropy_epistemic"].to_numpy(dtype=float),
            "bayesian_epistemic": bayesian_epistemic,
        }
    )


def _auroc_from_column(df: pd.DataFrame, column: str) -> float:
    return auroc_error(df[column].to_numpy(dtype=float), df["correct"].to_numpy())


def compute_auroc_gap(comparison_df: pd.DataFrame, ensemble_col: str, bayesian_col: str, seed: int) -> dict:
    """Paired cluster-bootstrap CI on AUROC(ensemble_col) - AUROC(bayesian_col)."""
    df_ensemble = comparison_df.rename(columns={ensemble_col: "score"})
    df_bayesian = comparison_df.rename(columns={bayesian_col: "score"})

    def _auroc(df: pd.DataFrame) -> float:
        return auroc_error(df["score"].to_numpy(dtype=float), df["correct"].to_numpy())

    diff, ci_low, ci_high = paired_cluster_bootstrap(df_ensemble, df_bayesian, _auroc, "question_id", n=2000, seed=seed)
    return {"gap": diff, "ci_low": ci_low, "ci_high": ci_high}


def main_distillation(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    population = load_rq4_population(config.paths.items_parquet)
    print(f"Distillation population: N={len(population)} (RQ4's own, the common subset of RQ5's N=1836)")

    print("Computing ensemble (3-call) metrics: conf_ens AUROC/ECE...")
    ensemble = compute_ensemble_metrics(population, config.n_bins, config.seed)

    print("Fitting single-call Bayesian model (Tier A, D8 10x5 repeated CV - ~7 min)...")
    bayes = compute_bayesian_arm(population, config.seed)
    bayesian_auroc = float(roc_auc_score(population["correct"].astype(int), bayes["mean_oof_correct"]))
    bayesian_p_wrong = 1 - bayes["mean_oof_correct"]
    is_wrong = 1 - population["correct"].astype(int).to_numpy()
    bayesian_ece, _ = ece(bayesian_p_wrong, is_wrong, config.n_bins)

    bayesian_epistemic = posterior_predictive_entropy_decomposition(bayes["pooled_draws"])["epistemic"]
    comparison_df = build_distillation_comparison_df(population, bayes["mean_oof_correct"], bayesian_epistemic)

    ensemble_epistemic_auroc, ee_low, ee_high = cluster_bootstrap(
        comparison_df, lambda df: _auroc_from_column(df, "ensemble_epistemic"), "question_id", seed=config.seed
    )
    bayesian_epistemic_auroc, be_low, be_high = cluster_bootstrap(
        comparison_df, lambda df: _auroc_from_column(df, "bayesian_epistemic"), "question_id", seed=config.seed
    )

    auroc_gap = compute_auroc_gap(comparison_df, "ensemble_uncertainty", "bayesian_uncertainty", config.seed)
    epistemic_auroc_gap = compute_auroc_gap(comparison_df, "ensemble_epistemic", "bayesian_epistemic", config.seed)

    pct_auroc_retained = 100 * (bayesian_auroc - 0.5) / (ensemble["auroc"] - 0.5)

    table = pd.DataFrame.from_records(
        [
            {
                "model": "ensemble_3call",
                "auroc": ensemble["auroc"],
                "auroc_ci_low": ensemble["auroc_ci_low"],
                "auroc_ci_high": ensemble["auroc_ci_high"],
                "ece": ensemble["ece"],
                "epistemic_auroc": ensemble_epistemic_auroc,
                "epistemic_auroc_ci_low": ee_low,
                "epistemic_auroc_ci_high": ee_high,
            },
            {
                "model": "bayesian_1call",
                "auroc": bayesian_auroc,
                "auroc_ci_low": np.nan,
                "auroc_ci_high": np.nan,
                "ece": bayesian_ece,
                "epistemic_auroc": bayesian_epistemic_auroc,
                "epistemic_auroc_ci_low": be_low,
                "epistemic_auroc_ci_high": be_high,
            },
        ]
    )
    table_path = f"{config.paths.results_dir}/rq5_distillation_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))
    print(f"Wrote {len(table)} rows to {table_path}")

    print()
    print(
        f"AUROC gap (ensemble - bayesian) = {auroc_gap['gap']:.4f} "
        f"[{auroc_gap['ci_low']:.4f}, {auroc_gap['ci_high']:.4f}]"
    )
    print(
        f"Entropy-quality AUROC gap (ensemble epistemic - bayesian epistemic) = "
        f"{epistemic_auroc_gap['gap']:.4f} [{epistemic_auroc_gap['ci_low']:.4f}, {epistemic_auroc_gap['ci_high']:.4f}]"
    )
    print(f"ECE: ensemble={ensemble['ece']:.4f}, bayesian={bayesian_ece:.4f}")
    print(
        f"Headline: the single-call Bayesian model retains {pct_auroc_retained:.1f}% of the "
        f"3-call ensemble's AUROC edge over chance (0.5)."
    )

    gaps = pd.DataFrame.from_records(
        [
            {"comparison": "auroc_ensemble_minus_bayesian", **auroc_gap},
            {"comparison": "epistemic_auroc_ensemble_minus_bayesian", **epistemic_auroc_gap},
            {"comparison": "pct_auroc_edge_retained", "gap": pct_auroc_retained, "ci_low": np.nan, "ci_high": np.nan},
        ]
    )
    gaps_path = f"{config.paths.results_dir}/rq5_distillation_gaps_{config.model_slug}.csv"
    gaps.to_csv(gaps_path, index=False)
    print(f"Wrote {len(gaps)} rows to {gaps_path}")

    plot_rq5_distillation(
        auroc_ensemble=ensemble["auroc"],
        auroc_ensemble_ci=(ensemble["auroc_ci_low"], ensemble["auroc_ci_high"]),
        auroc_bayesian=bayesian_auroc,
        epistemic_auroc_ensemble=ensemble_epistemic_auroc,
        epistemic_auroc_ensemble_ci=(ee_low, ee_high),
        epistemic_auroc_bayesian=bayesian_epistemic_auroc,
        epistemic_auroc_bayesian_ci=(be_low, be_high),
        model_slug=config.model_slug,
    )


# --- Task 5.9e: human-disagreement validation -------------------------------

DISAGREEMENT_VALIDATION_SIGNALS = ["ens_entropy_aleatoric", "ens_entropy_epistemic"]


def main_human_disagreement(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    items = load_disagreement_items(config.paths.items_parquet)
    print(f"Human-disagreement validation population: N={len(items)} (D9's own, clean/P1, n_human_votes >= 2)")

    rows = []
    for signal in DISAGREEMENT_VALIDATION_SIGNALS:
        result = compute_d_human_correlation(items, signal, config.seed)
        result["signal"] = signal
        rows.append(result)
        print(
            f"{signal} vs d_human: Spearman={result['spearman']:.4f} "
            f"[{result['spearman_ci_low']:.4f}, {result['spearman_ci_high']:.4f}]"
        )

    table = pd.DataFrame.from_records(rows)[["signal", "spearman", "spearman_ci_low", "spearman_ci_high"]]
    table_path = f"{config.paths.results_dir}/rq5_human_disagreement_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    plot_d_human_correlations(
        signals=table["signal"].tolist(),
        spearman=table["spearman"].to_numpy(),
        ci_low=table["spearman_ci_low"].to_numpy(),
        ci_high=table["spearman_ci_high"].to_numpy(),
        model_slug=config.model_slug,
        filename_suffix="_ensemble_entropy",
        title="Ensemble entropy vs. d_human (task 5.9e)",
    )

    aleatoric_row = table[table["signal"] == "ens_entropy_aleatoric"].iloc[0]
    epistemic_row = table[table["signal"] == "ens_entropy_epistemic"].iloc[0]
    # "Tracks disagreement" = negative correlation with d_human (consensus
    # strength) whose CI excludes 0.
    aleatoric_tracks_disagreement = aleatoric_row["spearman_ci_high"] < 0
    epistemic_also_tracks = epistemic_row["spearman_ci_high"] < 0

    print()
    if aleatoric_tracks_disagreement and not epistemic_also_tracks:
        print(
            f"Reading: aleatoric DOES track genuine human disagreement (Spearman={aleatoric_row['spearman']:.4f}, "
            f"CI excludes 0 - higher aleatoric where consensus is weak), while epistemic shows no such "
            f"relationship (Spearman={epistemic_row['spearman']:.4f}, CI includes 0) - the aleatoric/epistemic "
            f"vocabulary validates against this independent, model-free signal."
        )
    elif aleatoric_tracks_disagreement and epistemic_also_tracks:
        print(
            "Reading: aleatoric DOES track human disagreement, but epistemic shows a similar relationship too - "
            "the decomposition doesn't cleanly separate the two against this independent signal."
        )
    else:
        print(
            f"Reading: aleatoric's correlation with d_human does NOT clear significance "
            f"(Spearman={aleatoric_row['spearman']:.4f}, CI includes 0) - the aleatoric/epistemic vocabulary "
            f"does not validate against this independent signal."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=["threshold_sweep", "distillation", "human_disagreement"])
    args = parser.parse_args()
    if args.task == "threshold_sweep":
        main_threshold_sweep(args.config)
    elif args.task == "distillation":
        main_distillation(args.config)
    else:
        main_human_disagreement(args.config)
