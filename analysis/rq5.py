"""RQ5: does marginalizing over the judge prompt improve uncertainty
quality, and does that survive distillation to fewer calls? (D20, D23, D31)

`python -m analysis.rq5 --config configs/run.yaml --task {threshold_sweep,distillation,human_disagreement}`

threshold_sweep - RQ1's population. threshold_sweep() over
each of ens_entropy_{total,aleatoric,epistemic} (already uncertainty-typed,
no 1 - x flip), with ECE on the retained set always scored against
conf_ens_prob so the three curves answer one comparable question. Tests the
prediction made in advance (D23) that epistemic thresholding beats total (a
paired cluster-bootstrap on the AURC gap - one item set, two signals). The
figure and the AURCs use risk_coverage() on each entropy, like RQ2, so the
AURC here and in RQ2 is the same quantity.
Writes results/rq5_threshold_sweep_table_{model_slug}.csv and
results/figures/entropy_threshold_sweep_{model_slug}.png.

distillation - RQ4's population (N=1,819, a strict subset of
RQ5's 1,836), so every arm is scored on the same items. Teacher: the
3-prompt ensemble's conf_ens and ens_entropy_epistemic (6 calls: three
prompts, both orders). Two Bayesian meta-models
(analysis/rq4.py::compute_bayesian_arm) are compared with it, each with its
own posterior epistemic entropy: the two-call model reads only the greedy
P1 verdict in both orders, and the all-signals model reads all of Tier A,
the ensemble's own outputs included (D31). Every AUROC and gap is per
fold (D32): each meta-model's fold model and the ensemble are scored on the
same held-out questions, with paired cluster-bootstrap CIs, and each
meta-model is checked against a within-question permutation null
(invariant 12); ECE is reported side by side. "% of the edge retained" measures AUROC above chance (0.5).
Writes
results/rq5_distillation_{model_slug}.csv, the paired gaps and a figure.

human_disagreement - human_disagreement.py's N=595
population. Spearman of the ensemble's aleatoric and epistemic entropy
against d_human. d_human = |frac_prefer_a - 0.5| is CONSENSUS strength, so
"aleatoric tracks human disagreement" means a NEGATIVE correlation. Writes
results/rq5_human_disagreement_{model_slug}.csv and a figure.
"""

import argparse
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis.human_disagreement import compute_d_human_correlation, load_disagreement_items
from analysis.rq1 import load_rq1_items
from analysis.rq4 import compute_bayesian_arm
from analysis.rq6 import summarize_meta_null
from src.bayesian import posterior_predictive_entropy_decomposition, summarize_diagnostics
from src.boot import cluster_bootstrap, paired_cluster_bootstrap, paired_per_fold_bootstrap
from src.config import Config
from src.features import build_tier_a, build_two_call, load_rq4_population
from src.metrics import aurc, auroc_error, ece, oracle_risk_coverage, per_fold_auroc_error, risk_coverage, threshold_sweep
from src.plots import judge_name, plot_d_human_correlations, plot_risk_coverage, plot_rq5_distillation
from src.predictor import build_xyg

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
        confidences=items["conf_ens_prob"].to_numpy(dtype=float),
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
        tables.append(compute_entropy_sweep(items, signal, config.n_bins))

        coverage, risk = risk_coverage(
            items[signal].to_numpy(dtype=float), items["correct"].to_numpy()
        )
        curves[signal] = (coverage, risk)
        aurc_by_signal[signal] = aurc(coverage, risk)

    for dark in (False, True):
        plot_risk_coverage(
            curves, oracle_curve,
            filename=f"entropy_threshold_sweep_{config.model_slug}.png",
            title=f"Abstaining by ensemble entropy ({judge_name(config.model_slug)})",
            dark=dark,
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
        f"Prediction made in advance (epistemic beats total, D23): {verdict} - "
        f"AURC(epistemic) - AURC(total) = "
        f"{gap['aurc_gap_epistemic_minus_total']:.4f} "
        f"[{gap['ci_low']:.4f}, {gap['ci_high']:.4f}]"
    )


# --- Distillation ----------------------------------------------


def compute_ensemble_metrics(items: pd.DataFrame, n_bins: int, seed: int) -> dict:
    """conf_ens's AUROC(uncertainty -> error), with a cluster-bootstrap CI,
    and its ECE as P(wrong) - on conf_ens_prob, since conf_ens is 1 - entropy,
    not a probability (signals.py::prob_on_verdict).
    """
    def _auroc(df: pd.DataFrame) -> float:
        uncertainty = 1 - df["conf_ens"].to_numpy(dtype=float)
        return auroc_error(uncertainty, df["correct"].to_numpy())

    auroc_point, auroc_low, auroc_high = cluster_bootstrap(items, _auroc, "question_id", seed=seed)

    p_wrong = 1 - items["conf_ens_prob"].to_numpy(dtype=float)
    is_wrong = 1 - items["correct"].astype(int).to_numpy()
    ece_value, _ = ece(p_wrong, is_wrong, n_bins)

    return {"auroc": auroc_point, "auroc_ci_low": auroc_low, "auroc_ci_high": auroc_high, "ece": ece_value}


# Calls per item. The ensemble reads three prompts in both orders. The
# all-signals model reads those same 6 calls plus conf_sc's 4 sampled draws;
# the two-call model reads only the greedy P1 verdict in both orders (D31).
ENSEMBLE_CALLS = 6
DISTILLED_MODELS = {
    "bayesian_two_call": (build_two_call, 2),
    "bayesian_all_signals": (build_tier_a, 10),
}


def fit_distilled_model(
    population: pd.DataFrame, feature_builder: Callable[[pd.DataFrame], pd.DataFrame], n_bins: int, seed: int
) -> dict:
    """One Bayesian meta-model under the D8 protocol, scored per fold (D32):
    its per-fold AUROC, the per-fold AUROC of its epistemic entropy (each
    repeat's own posterior draws), the pooled conventions for reference, its ECE, and its convergence diagnostics over all 50
    fold-fits (invariant 13). Keeps each repeat's predictions and folds so
    the caller can pair it with the ensemble fold by fold.
    """
    bayes = compute_bayesian_arm(population, seed, feature_builder=feature_builder)
    correct = population["correct"].astype(int).to_numpy()
    ece_value, _ = ece(1 - bayes["mean_oof_correct"], 1 - correct, n_bins)
    epistemic_aurocs = [
        per_fold_auroc_error(e, correct, f) for e, f in zip(bayes["epistemic_by_repeat"], bayes["fold_ids"])
    ]
    pooled_epistemic = posterior_predictive_entropy_decomposition(bayes["pooled_draws"])["epistemic"]
    return {
        "auroc": float(bayes["aurocs"].mean()),
        "auroc_low": float(bayes["aurocs"].min()),
        "auroc_high": float(bayes["aurocs"].max()),
        "auroc_pooled": float(roc_auc_score(correct, bayes["mean_oof_correct"])),
        "epistemic_auroc": float(np.mean(epistemic_aurocs)),
        "epistemic_auroc_pooled_draws": auroc_error(pooled_epistemic, correct),
        "ece": ece_value,
        "uncertainty_by_repeat": [1 - p for p in bayes["oof_by_repeat"]],
        "epistemic_by_repeat": bayes["epistemic_by_repeat"],
        "fold_ids": bayes["fold_ids"],
        "diagnostics": summarize_diagnostics(bayes["fold_diagnostics"]),
    }


def main_distillation(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    population = load_rq4_population(config.paths.items_parquet)
    print(f"Distillation population: N={len(population)} (RQ4's own, the common subset of RQ5's N=1836)")

    correct = population["correct"].astype(int).to_numpy()
    groups = population["question_id"].to_numpy()
    ensemble_uncertainty = 1 - population["conf_ens"].to_numpy(dtype=float)
    ensemble_epistemic = population["ens_entropy_epistemic"].to_numpy(dtype=float)

    print(f"Computing ensemble ({ENSEMBLE_CALLS} calls) metrics: conf_ens AUROC/ECE...")
    ensemble = compute_ensemble_metrics(population, config.n_bins, config.seed)

    rows, gaps, plotted = [], [], []
    for name, (feature_builder, n_calls) in DISTILLED_MODELS.items():
        print(f"Fitting {name} ({n_calls} calls, D8 10x5 repeated CV - ~7 min)...")
        fit = fit_distilled_model(population, feature_builder, config.n_bins, config.seed)
        folds = fit["fold_ids"]
        n_repeats = len(folds)

        # The ensemble is a raw signal, scored on the student's own folds so
        # each pair compares the two on the same held-out questions (D32).
        ensemble_auroc_per_fold = float(np.mean([per_fold_auroc_error(ensemble_uncertainty, correct, f) for f in folds]))
        ensemble_epistemic_per_fold = float(np.mean([per_fold_auroc_error(ensemble_epistemic, correct, f) for f in folds]))
        auroc_gap = paired_per_fold_bootstrap(
            fit["uncertainty_by_repeat"], [ensemble_uncertainty] * n_repeats, correct, groups, folds, seed=config.seed
        )
        epistemic_gap = paired_per_fold_bootstrap(
            fit["epistemic_by_repeat"], [ensemble_epistemic] * n_repeats, correct, groups, folds, seed=config.seed
        )
        pct_retained = 100 * (fit["auroc"] - 0.5) / (ensemble_auroc_per_fold - 0.5)
        print(f"Permutation null for {name} (n=20, five NUTS fits each - ~15 min)...")
        X, y, groups_xy = build_xyg(population, feature_builder)
        null = summarize_meta_null(X, y, groups_xy, fit["auroc"], config.seed)

        rows.append(
            {
                "model": name,
                "n_calls": n_calls,
                "auroc": fit["auroc"],
                "auroc_low": fit["auroc_low"],
                "auroc_high": fit["auroc_high"],
                "auroc_pooled": fit["auroc_pooled"],
                "ece": fit["ece"],
                "epistemic_auroc": fit["epistemic_auroc"],
                "epistemic_auroc_pooled_draws": fit["epistemic_auroc_pooled_draws"],
                "ensemble_auroc_same_folds": ensemble_auroc_per_fold,
                "ensemble_epistemic_auroc_same_folds": ensemble_epistemic_per_fold,
                **null,
                **fit["diagnostics"],
            }
        )
        gaps += [
            {"model": name, "comparison": "auroc_model_minus_ensemble",
             "gap": auroc_gap[0], "ci_low": auroc_gap[1], "ci_high": auroc_gap[2]},
            {"model": name, "comparison": "epistemic_auroc_model_minus_ensemble",
             "gap": epistemic_gap[0], "ci_low": epistemic_gap[1], "ci_high": epistemic_gap[2]},
            {"model": name, "comparison": "pct_auroc_edge_retained",
             "gap": pct_retained, "ci_low": np.nan, "ci_high": np.nan},
        ]
        plotted.append({"name": name, "n_calls": n_calls, "auroc_gap": auroc_gap, "epistemic_gap": epistemic_gap,
                        "auroc": fit["auroc"], "epistemic_auroc": fit["epistemic_auroc"],
                        "ensemble_auroc": ensemble_auroc_per_fold, "ensemble_epistemic": ensemble_epistemic_per_fold})
        print(
            f"{name}: per-fold AUROC {fit['auroc']:.4f} vs ensemble {ensemble_auroc_per_fold:.4f} on the same folds; "
            f"difference {auroc_gap[0]:+.4f} [{auroc_gap[1]:+.4f}, {auroc_gap[2]:+.4f}]; "
            f"epistemic {fit['epistemic_auroc']:.4f} vs {ensemble_epistemic_per_fold:.4f}, "
            f"difference {epistemic_gap[0]:+.4f} [{epistemic_gap[1]:+.4f}, {epistemic_gap[2]:+.4f}]; "
            f"ECE {fit['ece']:.4f}; keeps {pct_retained:.1f}% of the ensemble's edge over chance"
        )
        print(f"{name} convergence: {fit['diagnostics']}; permutation null: {null}")

    rows.insert(
        0,
        {
            "model": "ensemble_3prompt",
            "n_calls": ENSEMBLE_CALLS,
            "auroc": ensemble["auroc"],
            "auroc_low": ensemble["auroc_ci_low"],
            "auroc_high": ensemble["auroc_ci_high"],
            "ece": ensemble["ece"],
            "epistemic_auroc": auroc_error(ensemble_epistemic, correct),
        },
    )
    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq5_distillation_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))
    print(f"Wrote {len(table)} rows to {table_path}")

    gaps_table = pd.DataFrame.from_records(gaps)
    gaps_path = f"{config.paths.results_dir}/rq5_distillation_gaps_{config.model_slug}.csv"
    gaps_table.to_csv(gaps_path, index=False)
    print(f"Wrote {len(gaps_table)} rows to {gaps_path}")

    for dark in (False, True):
        plot_rq5_distillation(
            ensemble_calls=ENSEMBLE_CALLS,
            models=plotted,
            model_slug=config.model_slug,
            dark=dark,
        )


# --- Human-disagreement validation -------------------------------

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
        title="Does ensemble entropy track human consensus?",
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
