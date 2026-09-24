"""RQ4: can a cheap meta-model beat the best single signal at predicting
judge error? (PLAN.md §2) - plus the Bayesian arm's RQ5 verbose-shift check.

`python -m analysis.rq4 --config configs/run.yaml --task {ablation,h4,transfer,category,calibration,bayesian_comparison,verbose_shift}`

Each task is its own invocation; several run hundreds of model fits.

  ablation             5.5  tier A -> B -> C on human_agreed items (N=556),
                            with permutation nulls and paired tier steps
  h4                   5.6  correct ~ oof_score * d_human, cluster bootstrap
  transfer             5.7  train on clean, evaluate frozen on verbose
  category             5.8  LeaveOneGroupOut over the 8 categories
  calibration          5.9  meta-model reliability + Tier C coefficients
  bayesian_comparison  5.9c LogReg vs Bayesian hierarchical model, Tier A
  verbose_shift        5.9f Bayesian epistemic/aleatoric, clean vs verbose

Populations differ by task on purpose - each function's docstring says
which and why. The "rq5_" output name of verbose_shift reflects the RQ it
answers; it lives here because it reuses 5.7's transfer machinery.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from analysis.rq1 import SIGNALS
from analysis.rq3 import load_rq3b_items
from src.boot import cluster_bootstrap, paired_cluster_bootstrap
from src.config import Config
from src.features import (
    TIER_A_COLUMNS,
    TIER_C_EXTRA_COLUMNS,
    build_tier_a,
    build_tier_c,
    load_rq4_population,
)
from src.bayesian import (
    build_group_index,
    fit_nuts,
    posterior_predictive_entropy_decomposition,
    predict_in_sample,
    repeated_stratified_group_kfold_bayesian,
)
from src.metrics import auroc_error, brier, ece, get_bin_edges
from src.plots import (
    plot_bayesian_convergence,
    plot_h4_interaction,
    plot_rq4_ablation,
    plot_rq4_category_transfer,
    plot_rq4_coefficients,
    plot_rq4_coefficients_full,
    plot_rq4_permutation_nulls,
    plot_rq4_progression,
    plot_rq4_transfer,
    plot_reliability_diagram,
    plot_rq5_verbose_shift,
)
from src.predictor import (
    MODEL_FACTORIES,
    TIER_BUILDERS,
    build_xyg,
    encode_features,
    make_logreg,
    permutation_null,
    percentile_of_null,
    repeated_stratified_group_kfold,
    run_predictor,
)


# --- Task 5.5: tier ablation ----------------------------------------------


def load_ablation_population(items_parquet: str) -> pd.DataFrame:
    """The RQ4 base restricted to human_agreed items (D16): N=1,819 -> 556.
    Not a neutral cut - agreed items are easier (accuracy 75.8% -> 78.4%) -
    so every comparison in this task is computed on this population.
    """
    population = load_rq4_population(items_parquet)
    return population[population["human_agreed"]]


def compute_baseline_auroc(population: pd.DataFrame, seed: int) -> dict:
    """The best single signal's AUROC(uncertainty -> error) on THIS
    population, recomputed rather than read from RQ2's table (which used
    the 1,836-item population) so baseline and tiers share their items.
    RQ2's recipe: uncertainty = 1 - signal, cluster-bootstrap CI. The winner
    is chosen on the point estimate alone; its CI is reported, not used to
    pick.
    """
    def _auroc(df: pd.DataFrame) -> float:
        uncertainty = 1 - df[signal].to_numpy(dtype=float)
        return auroc_error(uncertainty, df["correct"].to_numpy())

    winning_signal = {"signal": None, "auroc": float("-inf"), "auroc_ci_low": None, "auroc_ci_high": None}
    for signal in SIGNALS:
        point, ci_low, ci_high = cluster_bootstrap(population, _auroc, "question_id", seed=seed)
        if point > winning_signal["auroc"]:
            winning_signal["signal"] = signal
            winning_signal["auroc"] = point
            winning_signal["auroc_ci_low"] = ci_low
            winning_signal["auroc_ci_high"] = ci_high

    return winning_signal


def compute_tier_model_result(population: pd.DataFrame, tier_name: str, model_name: str, seed: int) -> dict:
    """One (tier, model) cell: the D8 protocol's mean AUROC and its
    across-repeat min/max - D8's headline uncertainty, not a bootstrap CI.
    """
    results = run_predictor(population, TIER_BUILDERS[tier_name], model_name, seed)
    aurocs = [r.auroc for r in results]
    return {
        "tier": tier_name,
        "model": model_name,
        "auroc_mean": sum(aurocs) / len(aurocs),
        "auroc_low": min(aurocs),
        "auroc_high": max(aurocs),
        "n_repeats": len(results),
    }


def compute_permutation_null_summary(
    population: pd.DataFrame, tier_name: str, model_name: str, seed: int, n: int = 50
) -> dict:
    """Invariant 12 for one cell, on this population. n=50 rather than
    200: every observed AUROC here sits near 0.8 against nulls near 0.5-0.57,
    so finer percentile resolution can't change the reading.

    `null_aurocs` (the raw array) is returned for the figure, not the CSV.
    """
    X, y, groups = build_xyg(population, TIER_BUILDERS[tier_name])
    observed = compute_tier_model_result(population, tier_name, model_name, seed)["auroc_mean"]
    null_aurocs = permutation_null(X, y, groups, MODEL_FACTORIES[model_name], n=n, seed=seed)
    return {
        "tier": tier_name,
        "model": model_name,
        "observed_auroc": observed,
        "null_mean": float(null_aurocs.mean()),
        "null_std": float(null_aurocs.std()),
        "percentile": percentile_of_null(observed, null_aurocs),
        "null_aurocs": null_aurocs,
    }


def compute_tier_oof_uncertainty(population: pd.DataFrame, tier_name: str, model_name: str, seed: int) -> np.ndarray:
    """1 - (out-of-fold P(correct) averaged over the 10 repeats), one value
    per item, positionally aligned with `population` - on the same
    uncertainty scale as a raw signal's 1 - conf.
    """
    results = run_predictor(population, TIER_BUILDERS[tier_name], model_name, seed)
    mean_oof_pred = np.mean([r.oof_pred for r in results], axis=0)
    return 1 - mean_oof_pred


def _predictions_df(population: pd.DataFrame, uncertainty: np.ndarray) -> pd.DataFrame:
    """question_id / correct / uncertainty - one shared column name so one
    stat_fn scores any baseline or tier in a paired comparison.
    """
    return pd.DataFrame(
        {
            "question_id": population["question_id"].to_numpy(),
            "correct": population["correct"].to_numpy(),
            "uncertainty": uncertainty,
        }
    )


def _auroc_from_uncertainty(df: pd.DataFrame) -> float:
    return auroc_error(df["uncertainty"].to_numpy(), df["correct"].to_numpy())


def compare_tier_progression(population: pd.DataFrame, baseline_signal: str, seed: int) -> pd.DataFrame:
    """Paired cluster-bootstrap CI on each step's AUROC change (baseline ->
    A, A -> B, B -> C), per model - the same items scored two ways, so the
    test is whether a step is real, not whether two bars' whiskers overlap.

    Returns one row per (model, comparison): auroc_diff (higher stage minus
    lower), ci_low, ci_high.
    """
    baseline_uncertainty = 1 - population[baseline_signal].to_numpy(dtype=float)

    rows = []
    for model_name in MODEL_FACTORIES:
        uncertainty_by_stage = {"baseline": baseline_uncertainty}
        for tier_name in TIER_BUILDERS:
            uncertainty_by_stage[tier_name] = compute_tier_oof_uncertainty(population, tier_name, model_name, seed)

        for lo, hi in [("baseline", "A"), ("A", "B"), ("B", "C")]:
            df_hi = _predictions_df(population, uncertainty_by_stage[hi])
            df_lo = _predictions_df(population, uncertainty_by_stage[lo])
            diff, ci_low, ci_high = paired_cluster_bootstrap(
                df_hi, df_lo, _auroc_from_uncertainty, "question_id", seed=seed
            )
            rows.append(
                {
                    "model": model_name,
                    "comparison": f"{hi} - {lo}",
                    "auroc_diff": diff,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    return pd.DataFrame.from_records(rows)


# --- Task 5.6: H4 --------------------------------------------------------


def load_h4_population(items_parquet: str) -> pd.DataFrame:
    """D9's population: clean/P1, human_label present, n_human_votes >= 2
    (N=595). It keeps the contested items the ablation drops - H4 is about
    how predictability varies with consensus, so it needs them.
    """
    items = pd.read_parquet(items_parquet)
    items = items[(items["condition"] == "clean") & (items["prompt_variant"] == "P1")]
    items = items[items["human_label"].notna()]
    return items[items["n_human_votes"] >= 2]


def compute_h4_oof_score(population: pd.DataFrame, seed: int) -> np.ndarray:
    """"The predictor" for H4: Tier A + logreg, out-of-fold P(correct)
    averaged over the 10 repeats. Tier A because 5.5 found no tier or model
    better than another and B/C would cut 17 more items; logreg because the
    interaction test is itself a logistic regression.
    """
    results = run_predictor(population, build_tier_a, "logreg", seed)
    return np.mean([r.oof_pred for r in results], axis=0)


def _h4_design_matrix(oof_score: np.ndarray, d_human: np.ndarray) -> np.ndarray:
    """[oof_score, d_human, oof_score * d_human], shared by both H4 fits."""
    return np.column_stack([oof_score, d_human, oof_score * d_human])


def build_h4_predictions_df(population: pd.DataFrame, seed: int) -> pd.DataFrame:
    """question_id / correct / d_human / oof_score, built once per run so
    the 10x5 CV isn't repeated for the figure.
    """
    oof_score = compute_h4_oof_score(population, seed)
    return pd.DataFrame(
        {
            "question_id": population["question_id"].to_numpy(),
            "correct": population["correct"].to_numpy(),
            "d_human": population["d_human"].to_numpy(),
            "oof_score": oof_score,
        }
    )


def fit_interaction_coefficient(df: pd.DataFrame) -> float:
    """The H4 statistic: the product term's coefficient in
    correct ~ oof_score + d_human + oof_score * d_human
    (LogisticRegression(C=1.0), predictor.py's fixed choice). Refit on every
    bootstrap replicate (D15).
    """
    X = _h4_design_matrix(df["oof_score"].to_numpy(), df["d_human"].to_numpy())
    y = df["correct"].astype(int).to_numpy()
    model = LogisticRegression(C=1.0)
    model.fit(X, y)
    return float(model.coef_[0][-1])


def compute_h4_interaction(predictions_df: pd.DataFrame, seed: int) -> dict:
    """Cluster-bootstrap CI (question_id, B=2000, refit per replicate) on
    the interaction coefficient - D15's method; never a default standard
    error, which assumes independent rows.
    """
    point, ci_low, ci_high = cluster_bootstrap(
        predictions_df, fit_interaction_coefficient, "question_id", n=2000, seed=seed
    )
    return {"n": len(predictions_df), "interaction_coef": point, "ci_low": ci_low, "ci_high": ci_high}


def fit_h4_interaction_model(predictions_df: pd.DataFrame) -> LogisticRegression:
    """The same fit once on the real data, for the figure's curves only."""
    X = _h4_design_matrix(predictions_df["oof_score"].to_numpy(), predictions_df["d_human"].to_numpy())
    y = predictions_df["correct"].astype(int).to_numpy()
    model = LogisticRegression(C=1.0)
    model.fit(X, y)
    return model


def compute_h4_interaction_curves(predictions_df: pd.DataFrame, n_grid: int = 100) -> dict:
    """Predicted P(correct) vs oof_score at each distinct d_human level in
    the data (3 of them; 564/595 items sit at 0.5, so min/median/max would
    collapse), in probability and log-odds. The log-odds curves show the
    interaction undistorted; in probability space sigmoid saturation hides
    it where most data sits (high oof_score).
    """
    model = fit_h4_interaction_model(predictions_df)
    d_human_values = sorted(predictions_df["d_human"].round(4).unique())
    oof_score_grid = np.linspace(0, 1, n_grid)

    predicted_curves = []
    log_odds_curves = []
    for d_human_value in d_human_values:
        X_grid = _h4_design_matrix(oof_score_grid, np.full(n_grid, d_human_value))
        predicted_curves.append(model.predict_proba(X_grid)[:, 1])
        log_odds_curves.append(model.decision_function(X_grid))

    return {
        "oof_score": predictions_df["oof_score"].to_numpy(),
        "oof_score_grid": oof_score_grid,
        "d_human_values": d_human_values,
        "predicted_curves": predicted_curves,
        "log_odds_curves": log_odds_curves,
    }


def main_h4(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    population = load_h4_population(config.paths.items_parquet)
    print(f"H4 population: N={len(population)} (clean/P1, human_label present, n_human_votes >= 2, D9)")

    predictions_df = build_h4_predictions_df(population, config.seed)

    result = compute_h4_interaction(predictions_df, config.seed)
    print(
        f"H4 interaction coefficient (oof_score x d_human): {result['interaction_coef']:.4f} "
        f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
    )
    if result["ci_low"] > 0:
        print("CI excludes 0 (positive): the predictor's edge grows with human consensus - supports H4.")
    elif result["ci_high"] < 0:
        print("CI excludes 0 (negative): the predictor's edge SHRINKS with human consensus - contradicts H4.")
    else:
        print("CI includes 0: no detectable interaction at this sample size - H4 neither supported nor refuted.")

    table_path = f"{config.paths.results_dir}/rq4_h4_interaction_{config.model_slug}.csv"
    pd.DataFrame.from_records([result]).to_csv(table_path, index=False)
    print(f"Wrote 1 row to {table_path}")

    curves = compute_h4_interaction_curves(predictions_df)
    plot_h4_interaction(
        oof_score=curves["oof_score"],
        oof_score_grid=curves["oof_score_grid"],
        d_human_values=curves["d_human_values"],
        predicted_curves=curves["predicted_curves"],
        log_odds_curves=curves["log_odds_curves"],
        model_slug=config.model_slug,
    )


# --- Task 5.7: transfer clean -> verbose -------------------------------------
#
# Does an abstention layer trained on clean data still work when the judge
# is attacked? Population: RQ3b's paired clean/verbose items (N=1,836).
# Features: Tier A minus the signals verbose doesn't have (D21) - leaving
# conf_verb, conf_lp, conf_bpe - for every model compared.

TRANSFER_SAFE_COLUMNS = [
    c
    for c in TIER_A_COLUMNS
    if c not in {"conf_sc", "conf_ens", "ens_entropy_total", "ens_entropy_aleatoric", "ens_entropy_epistemic"}
]


def build_transfer_xy(items: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(X, y, groups) on TRANSFER_SAFE_COLUMNS, for clean or verbose items."""
    X = encode_features(items[TRANSFER_SAFE_COLUMNS].copy())
    y = items["correct"].astype(int).to_numpy()
    groups = items["question_id"].to_numpy()
    return X, y, groups


def compute_transfer_baseline(clean_items: pd.DataFrame, model_name: str, seed: int) -> dict:
    """In-domain baseline: repeated grouped CV on clean alone, with the same
    reduced features - so the transfer delta measures the move to verbose,
    not the dropped features.
    """
    X, y, groups = build_transfer_xy(clean_items)
    results = repeated_stratified_group_kfold(X, y, groups, MODEL_FACTORIES[model_name], seed=seed)
    aurocs = [r.auroc for r in results]
    return {"auroc_mean": sum(aurocs) / len(aurocs), "auroc_low": min(aurocs), "auroc_high": max(aurocs)}


def compute_transfer_auroc(clean_items: pd.DataFrame, verbose_items: pd.DataFrame, model_name: str, seed: int) -> dict:
    """Fit once on all of clean, evaluate the frozen model on verbose - one
    offline-trained model's deployment behavior, not training variance.
    The CI resamples verbose's questions only, since the model is fixed.
    """
    X_train, y_train, _ = build_transfer_xy(clean_items)
    X_test, y_test, groups_test = build_transfer_xy(verbose_items)

    model = MODEL_FACTORIES[model_name](seed)
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]

    eval_df = pd.DataFrame({"question_id": groups_test, "correct": y_test, "pred": pred})

    def _auroc(df: pd.DataFrame) -> float:
        return float(roc_auc_score(df["correct"].to_numpy(), df["pred"].to_numpy()))

    point, ci_low, ci_high = cluster_bootstrap(eval_df, _auroc, "question_id", seed=seed)
    return {"auroc": point, "ci_low": ci_low, "ci_high": ci_high}


def main_transfer(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    clean_items, verbose_items = load_rq3b_items(config.paths.items_parquet)
    print(f"Transfer test population: N={len(clean_items)} paired items (clean/P1 vs verbose/P1)")
    print(f"Feature set (D21 parity fix): {TRANSFER_SAFE_COLUMNS}")

    rows = []
    for model_name in MODEL_FACTORIES:
        baseline = compute_transfer_baseline(clean_items, model_name, config.seed)
        transfer = compute_transfer_auroc(clean_items, verbose_items, model_name, config.seed)
        delta = transfer["auroc"] - baseline["auroc_mean"]

        print(
            f"{model_name}: in-domain (clean) AUROC={baseline['auroc_mean']:.4f} "
            f"[{baseline['auroc_low']:.4f}, {baseline['auroc_high']:.4f}] (D8 across-repeat spread)"
        )
        print(
            f"{model_name}: transfer (clean->verbose) AUROC={transfer['auroc']:.4f} "
            f"[{transfer['ci_low']:.4f}, {transfer['ci_high']:.4f}] (cluster-bootstrap over verbose)"
        )
        print(f"{model_name}: delta AUROC (transfer - in-domain) = {delta:.4f}")

        rows.append(
            {
                "model": model_name,
                "baseline_auroc_mean": baseline["auroc_mean"],
                "baseline_auroc_low": baseline["auroc_low"],
                "baseline_auroc_high": baseline["auroc_high"],
                "transfer_auroc": transfer["auroc"],
                "transfer_ci_low": transfer["ci_low"],
                "transfer_ci_high": transfer["ci_high"],
                "delta_auroc": delta,
            }
        )

    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq4_transfer_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    plot_rq4_transfer(
        models=table["model"].tolist(),
        baseline_mean=table["baseline_auroc_mean"].to_numpy(),
        baseline_low=table["baseline_auroc_low"].to_numpy(),
        baseline_high=table["baseline_auroc_high"].to_numpy(),
        transfer_mean=table["transfer_auroc"].to_numpy(),
        transfer_low=table["transfer_ci_low"].to_numpy(),
        transfer_high=table["transfer_ci_high"].to_numpy(),
        model_slug=config.model_slug,
    )


# --- Task 5.8: LeaveOneGroupOut over category ------------------------------


def compute_category_held_out_auroc(population: pd.DataFrame, model_name: str, seed: int) -> pd.DataFrame:
    """For each of the 8 categories: fit on the other 7, evaluate on the
    held-out one. Full Tier A (this never leaves clean). `category` is the
    grouping variable here, never a feature. LeaveOneGroupOut is
    deterministic, so there are no repeats to average.

    Returns one row per category: category, model, n, auroc.
    """
    X = encode_features(build_tier_a(population))
    y = population["correct"].astype(int).to_numpy()
    groups = population["category"].to_numpy()

    rows = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        model = MODEL_FACTORIES[model_name](seed)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict_proba(X.iloc[test_idx])[:, 1]
        auroc = roc_auc_score(y[test_idx], pred)
        rows.append({"category": groups[test_idx][0], "model": model_name, "n": len(test_idx), "auroc": auroc})

    return pd.DataFrame.from_records(rows)


def main_category(config_path: str) -> None:
    """A tight AUROC spread across categories means the predictor
    generalizes; one category collapsing means it learned a shortcut.
    """
    config = Config.from_yaml(config_path)
    population = load_rq4_population(config.paths.items_parquet)

    tables = []
    for model_name in MODEL_FACTORIES:
        result = compute_category_held_out_auroc(population, model_name, config.seed)
        tables.append(result)
        for _, row in result.iterrows():
            print(f"{model_name}, {row['category']}: n={row['n']}, AUROC={row['auroc']:.4f}")
        print(
            f"{model_name}: spread across categories = "
            f"[{result['auroc'].min():.4f}, {result['auroc'].max():.4f}]"
        )

    table = pd.concat(tables, ignore_index=True)
    table_path = f"{config.paths.results_dir}/rq4_category_transfer_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    plot_rq4_category_transfer(
        categories=table["category"].tolist(),
        models=table["model"].tolist(),
        auroc=table["auroc"].to_numpy(),
        model_slug=config.model_slug,
    )


def main_ablation(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    population = load_ablation_population(config.paths.items_parquet)
    print(f"RQ4 ablation population: N={len(population)} (human_agreed items only, D16)")

    baseline = compute_baseline_auroc(population, config.seed)
    print(
        f"Baseline (best single signal, this population): {baseline['signal']} "
        f"AUROC={baseline['auroc']:.4f} [{baseline['auroc_ci_low']:.4f}, {baseline['auroc_ci_high']:.4f}]"
    )

    rows = []
    for tier_name in TIER_BUILDERS:
        for model_name in MODEL_FACTORIES:
            result = compute_tier_model_result(population, tier_name, model_name, config.seed)
            rows.append(result)
            print(
                f"Tier {tier_name}, {model_name}: AUROC={result['auroc_mean']:.4f} "
                f"[{result['auroc_low']:.4f}, {result['auroc_high']:.4f}]"
            )

    table = pd.DataFrame.from_records(rows)
    table_path = f"{config.paths.results_dir}/rq4_ablation_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    plot_rq4_ablation(
        tiers=table["tier"].tolist(),
        models=table["model"].tolist(),
        auroc_mean=table["auroc_mean"].to_numpy(),
        auroc_low=table["auroc_low"].to_numpy(),
        auroc_high=table["auroc_high"].to_numpy(),
        baseline=baseline["auroc"],
        baseline_ci_low=baseline["auroc_ci_low"],
        baseline_ci_high=baseline["auroc_ci_high"],
        baseline_label=baseline["signal"],
        model_slug=config.model_slug,
    )

    # Does every cell beat chance on this population specifically? (invariant 12)
    print()
    print("Permutation null (n=50 per cell - see compute_permutation_null_summary's own docstring):")
    null_rows = []
    for tier_name in TIER_BUILDERS:
        for model_name in MODEL_FACTORIES:
            null_summary = compute_permutation_null_summary(population, tier_name, model_name, config.seed)
            null_rows.append(null_summary)
            print(
                f"  Tier {tier_name}, {model_name}: observed={null_summary['observed_auroc']:.4f}, "
                f"null mean={null_summary['null_mean']:.4f} (std={null_summary['null_std']:.4f}), "
                f"percentile={null_summary['percentile']:.1%}"
            )

    null_table = pd.DataFrame.from_records(
        [{k: v for k, v in row.items() if k != "null_aurocs"} for row in null_rows]
    )
    null_table_path = f"{config.paths.results_dir}/rq4_ablation_null_{config.model_slug}.csv"
    null_table.to_csv(null_table_path, index=False)
    print(f"Wrote {len(null_table)} rows to {null_table_path}")

    plot_rq4_permutation_nulls(
        tiers=[row["tier"] for row in null_rows],
        models=[row["model"] for row in null_rows],
        observed=np.array([row["observed_auroc"] for row in null_rows]),
        null_distributions=[row["null_aurocs"] for row in null_rows],
        model_slug=config.model_slug,
    )

    # Is the apparent baseline -> A -> B -> C progression real?
    print()
    print("Paired tier-progression comparison (does each step's apparent change survive a paired CI):")
    progression = compare_tier_progression(population, baseline["signal"], config.seed)
    progression_path = f"{config.paths.results_dir}/rq4_ablation_progression_{config.model_slug}.csv"
    progression.to_csv(progression_path, index=False)
    for _, row in progression.iterrows():
        print(
            f"  {row['model']}, {row['comparison']}: "
            f"delta_auroc={row['auroc_diff']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        )
    print(f"Wrote {len(progression)} rows to {progression_path}")

    plot_rq4_progression(
        comparisons=[f"{row['model']}: {row['comparison']}" for _, row in progression.iterrows()],
        auroc_diff=progression["auroc_diff"].to_numpy(),
        ci_low=progression["ci_low"].to_numpy(),
        ci_high=progression["ci_high"].to_numpy(),
        model_slug=config.model_slug,
    )


# --- Task 5.9: meta-model calibration and coefficients -----------------------
#
# RQ4 base population (N=1,819), Tier C + logreg: the only tier holding all
# three feature families, and the only model with directly readable
# coefficients.


def compute_meta_oof_score(population: pd.DataFrame, seed: int) -> np.ndarray:
    """Tier C + logreg out-of-fold P(correct), averaged over the 10 repeats
    - never in-sample predictions, which would look artificially calibrated.
    """
    results = run_predictor(population, build_tier_c, "logreg", seed)
    return np.mean([r.oof_pred for r in results], axis=0)


def compute_meta_calibration(population: pd.DataFrame, seed: int, n_bins: int) -> dict:
    """P(judge wrong) = 1 - mean OOF P(correct): its ECE against the real
    outcome, plus the OOF AUROC (invariant 6).

    Returns p_wrong, is_wrong (aligned with `population`), ece,
    n_effective_bins, auroc.
    """
    mean_oof_correct = compute_meta_oof_score(population, seed)
    correct = population["correct"].astype(int).to_numpy()
    p_wrong = 1 - mean_oof_correct
    is_wrong = 1 - correct

    ece_value, n_effective_bins = ece(p_wrong, is_wrong, n_bins)
    auroc = float(roc_auc_score(correct, mean_oof_correct))

    return {
        "p_wrong": p_wrong,
        "is_wrong": is_wrong,
        "ece": ece_value,
        "n_effective_bins": n_effective_bins,
        "auroc": auroc,
    }


def _feature_family(feature_name: str) -> str:
    """"A"/"B"/"C" for an encoded Tier C feature; category's one-hot
    columns count as Tier B, like `category` itself.
    """
    if feature_name in TIER_A_COLUMNS:
        return "A"
    if feature_name in TIER_C_EXTRA_COLUMNS:
        return "C"
    return "B"


def fit_full_logreg_coefficients(population: pd.DataFrame) -> tuple[LogisticRegression, list[str]]:
    """make_logreg() fit once on Tier C over the full population - a
    coefficient describes one fit, not a per-fold quantity. coef_ is on the
    standardized scale, so magnitudes compare across features. Returns the
    classifier and the encoded feature names in coef_ order.
    """
    X = encode_features(build_tier_c(population))
    y = population["correct"].astype(int).to_numpy()
    pipeline = make_logreg()
    pipeline.fit(X, y)
    return pipeline.named_steps["clf"], list(X.columns)


def bootstrap_coefficient_cis(
    population: pd.DataFrame, feature_names: list[str], n: int = 2000, ci: float = 0.95, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster-bootstrap CIs (question_id) on every coefficient at once:
    each replicate resamples questions once and refits the whole vector,
    rather than calling cluster_bootstrap() per feature - one refit yields
    every coefficient, and keeps them from the same resample.

    `.reindex(..., fill_value=0)` restores any category dummy that a
    resample happens not to contain, so columns stay aligned.

    Returns (ci_low, ci_high), in feature_names order.
    """
    rng = np.random.default_rng(seed)
    groups = population["question_id"].unique()
    n_groups = len(groups)
    group_indices = {
        group: population.index[population["question_id"] == group].to_numpy() for group in groups
    }

    boot_coefs = np.empty((n, len(feature_names)))
    for i in range(n):
        sampled_groups = rng.choice(groups, size=n_groups, replace=True)
        sampled_indices = np.concatenate([group_indices[group] for group in sampled_groups])
        resampled = population.loc[sampled_indices]

        X = encode_features(build_tier_c(resampled)).reindex(columns=feature_names, fill_value=0)
        y = resampled["correct"].astype(int).to_numpy()
        pipeline = make_logreg()
        pipeline.fit(X, y)
        boot_coefs[i] = pipeline.named_steps["clf"].coef_[0]

    alpha = 1 - ci
    ci_low = np.quantile(boot_coefs, alpha / 2, axis=0)
    ci_high = np.quantile(boot_coefs, 1 - alpha / 2, axis=0)
    return ci_low, ci_high


def compute_meta_coefficients(population: pd.DataFrame, seed: int, n_boot: int = 2000) -> pd.DataFrame:
    """feature, family, coef (from the real fit), ci_low, ci_high - one row
    per encoded Tier C feature.
    """
    clf, feature_names = fit_full_logreg_coefficients(population)
    ci_low, ci_high = bootstrap_coefficient_cis(population, feature_names, n=n_boot, seed=seed)

    return pd.DataFrame(
        {
            "feature": feature_names,
            "family": [_feature_family(name) for name in feature_names],
            "coef": clf.coef_[0],
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )


def main_calibration(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    population = load_rq4_population(config.paths.items_parquet)
    print(f"Meta-model calibration population: N={len(population)} (RQ4 base, Tier C + logreg)")

    calibration = compute_meta_calibration(population, config.seed, config.n_bins)
    print(f"OOF AUROC (D8, 10x5 repeated CV) = {calibration['auroc']:.4f}")
    print(
        f"ECE of P(judge is wrong) = {calibration['ece']:.4f} "
        f"(n_effective_bins={calibration['n_effective_bins']})"
    )

    plot_reliability_diagram(
        confidences=calibration["p_wrong"],
        correct=calibration["is_wrong"],
        signal_name="rq4_meta_model",
        n_bins=config.n_bins,
        model_slug=config.model_slug,
    )

    print()
    print("Fitting Tier C + logreg coefficients on the full population, cluster-bootstrap CI (B=2000)...")
    coef_table = compute_meta_coefficients(population, config.seed)
    coef_path = f"{config.paths.results_dir}/rq4_coefficients_{config.model_slug}.csv"
    coef_table.to_csv(coef_path, index=False)
    print(f"Wrote {len(coef_table)} rows to {coef_path}")

    n_significant = int(((coef_table["ci_low"] > 0) | (coef_table["ci_high"] < 0)).sum())
    print(f"{n_significant}/{len(coef_table)} coefficients have a CI excluding 0")
    for family in ["A", "B", "C"]:
        family_rows = coef_table[coef_table["family"] == family]
        family_significant = int(((family_rows["ci_low"] > 0) | (family_rows["ci_high"] < 0)).sum())
        print(f"  Tier {family}: {family_significant}/{len(family_rows)} coefficients have a CI excluding 0")

    # Headline figure shows the significant coefficients only; the full table
    # is the CSV plus a dense appendix figure.
    plot_rq4_coefficients(
        features=coef_table["feature"].tolist(),
        coef=coef_table["coef"].to_numpy(),
        ci_low=coef_table["ci_low"].to_numpy(),
        ci_high=coef_table["ci_high"].to_numpy(),
        family=coef_table["family"].tolist(),
        model_slug=config.model_slug,
    )
    plot_rq4_coefficients_full(
        features=coef_table["feature"].tolist(),
        coef=coef_table["coef"].to_numpy(),
        ci_low=coef_table["ci_low"].to_numpy(),
        ci_high=coef_table["ci_high"].to_numpy(),
        family=coef_table["family"].tolist(),
        model_slug=config.model_slug,
    )


# --- Task 5.9c: frequentist vs Bayesian head-to-head -------------------------
#
# RQ4 base population, Tier A for BOTH arms - comparing the two models on
# different features would conflate method with features.


def compute_frequentist_arm(population: pd.DataFrame, seed: int) -> dict:
    """Tier A + logreg: the per-repeat AUROCs (D8 spread) and the OOF
    P(correct) averaged over repeats (for ECE/Brier).
    """
    results = run_predictor(population, build_tier_a, "logreg", seed)
    return {
        "aurocs": np.array([r.auroc for r in results]),
        "mean_oof_correct": np.mean([r.oof_pred for r in results], axis=0),
    }


def compute_bayesian_arm(
    population: pd.DataFrame, seed: int, num_warmup: int = 500, num_samples: int = 1000, num_chains: int = 2
) -> dict:
    """Tier A Bayesian hierarchical model under the same protocol (about 7
    minutes for the 50 fold-fits).

    `pooled_draws` concatenates every repeat's posterior draws rather than
    averaging them: each repeat is an independent refit on a different
    partition, so pooling keeps both posterior and partition variability.
    float32 keeps the (20,000 x 1,819) array small.
    """
    X, y, groups = build_xyg(population, build_tier_a)
    results = repeated_stratified_group_kfold_bayesian(
        X, y, groups, seed=seed, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
    )
    return {
        "aurocs": np.array([r.auroc for r in results]),
        "mean_oof_correct": np.mean([r.oof_pred for r in results], axis=0),
        "pooled_draws": np.concatenate([r.oof_draws for r in results], axis=0).astype(np.float32),
        "fold_diagnostics": [fold_diag for r in results for fold_diag in r.fold_diagnostics],
    }


def compute_bayesian_nll(pooled_draws: np.ndarray, is_wrong: np.ndarray) -> float:
    """Posterior-predictive NLL per item, averaged:

        NLL_i = -log( mean_d  p_d^y_i * (1 - p_d)^(1 - y_i) )

    The Bernoulli likelihood is averaged over draws BEFORE the log. Plugging
    the mean probability into a point NLL would discard the draws' spread -
    Brier with extra steps. Clipped at 1e-12 so one confidently wrong item
    can't make the mean infinite.

    Args:
        pooled_draws: (n_draws, n_rows) P(wrong) per draw per item.
        is_wrong: (n_rows,) 1 if the judge was wrong.
    """
    is_wrong_arr = is_wrong.astype(np.float32)
    likelihood_per_draw = is_wrong_arr[None, :] * pooled_draws + (1 - is_wrong_arr[None, :]) * (1 - pooled_draws)
    mean_likelihood_per_item = likelihood_per_draw.mean(axis=0)
    mean_likelihood_per_item = np.clip(mean_likelihood_per_item, 1e-12, 1.0)
    return float(-np.mean(np.log(mean_likelihood_per_item)))


def compute_bayesian_coverage(pooled_draws: np.ndarray, p_wrong: np.ndarray, is_wrong: np.ndarray, n_bins: int) -> float:
    """Bin-aggregate 90% credible-interval coverage. A single 0/1 outcome
    can't "fall inside" a probability interval, so items are binned by
    p_wrong (ece()'s binning) and each bin asks whether its empirical wrong
    rate lies within the 5th-95th percentile of its pooled draws.

    Returns the fraction of bins covered.
    """
    edges = get_bin_edges(p_wrong, n_bins, "auto")
    bin_labels = np.digitize(p_wrong, edges)

    covered = []
    for bin_label in np.unique(bin_labels):
        mask = bin_labels == bin_label
        empirical_wrong_rate = is_wrong[mask].mean()
        bin_draws = pooled_draws[:, mask]
        lo, hi = np.percentile(bin_draws, [5, 95])
        covered.append(lo <= empirical_wrong_rate <= hi)

    return float(np.mean(covered))


def main_bayesian_comparison(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    population = load_rq4_population(config.paths.items_parquet)
    is_wrong = 1 - population["correct"].astype(int).to_numpy()
    print(f"Bayesian head-to-head population: N={len(population)} (RQ4 base, Tier A)")

    print("Fitting frequentist arm (Tier A + logreg, D8 10x5 repeated CV)...")
    freq = compute_frequentist_arm(population, config.seed)

    print("Fitting Bayesian arm (Tier A hierarchical logit, D8 10x5 repeated CV - ~7 min, see TASKS.md 5.9b)...")
    bayes = compute_bayesian_arm(population, config.seed)

    freq_p_wrong = 1 - freq["mean_oof_correct"]
    bayes_p_wrong = 1 - bayes["mean_oof_correct"]

    freq_ece, _ = ece(freq_p_wrong, is_wrong, config.n_bins)
    freq_brier = brier(freq_p_wrong, is_wrong)
    bayes_ece, _ = ece(bayes_p_wrong, is_wrong, config.n_bins)
    bayes_brier = brier(bayes_p_wrong, is_wrong)

    bayes_pooled_wrong_draws = 1 - bayes["pooled_draws"]
    bayes_nll = compute_bayesian_nll(bayes_pooled_wrong_draws, is_wrong)
    bayes_coverage = compute_bayesian_coverage(bayes_pooled_wrong_draws, bayes_p_wrong, is_wrong, config.n_bins)

    table = pd.DataFrame.from_records(
        [
            {
                "model": "frequentist_logreg",
                "auroc_mean": freq["aurocs"].mean(),
                "auroc_low": freq["aurocs"].min(),
                "auroc_high": freq["aurocs"].max(),
                "ece": freq_ece,
                "brier": freq_brier,
                "nll": np.nan,
                "coverage_90": np.nan,
            },
            {
                "model": "bayesian_hierarchical",
                "auroc_mean": bayes["aurocs"].mean(),
                "auroc_low": bayes["aurocs"].min(),
                "auroc_high": bayes["aurocs"].max(),
                "ece": bayes_ece,
                "brier": bayes_brier,
                "nll": bayes_nll,
                "coverage_90": bayes_coverage,
            },
        ]
    )
    table_path = f"{config.paths.results_dir}/rq4_bayesian_comparison_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))
    print(f"Wrote {len(table)} rows to {table_path}")

    plot_reliability_diagram(
        confidences=bayes_p_wrong,
        correct=is_wrong,
        signal_name="rq4_bayesian_meta_model",
        n_bins=config.n_bins,
        model_slug=config.model_slug,
    )

    all_diag = bayes["fold_diagnostics"]
    n_flagged = sum(fold_diag["flagged"] for fold_diag in all_diag)
    print(f"Bayesian convergence: {n_flagged}/{len(all_diag)} fold-fits flagged")

    plot_bayesian_convergence(
        max_rhat=np.array([fold_diag["max_rhat"] for fold_diag in all_diag]),
        flagged=np.array([fold_diag["flagged"] for fold_diag in all_diag]),
        model_slug=config.model_slug,
    )


# --- Task 5.9f: verbose-shift validation (RQ5, D21 amendment) ----------------
#
# Preregistered prediction (D23): under the clean -> verbose shift, epistemic
# rises while aleatoric stays flat. The model is fit ONCE on all of clean
# with 5.7's transfer-safe features and evaluated with predict_in_sample()
# on both conditions - clean and verbose share all 80 questions, so each
# row keeps its real fitted intercept and only the features differ.


def fit_verbose_shift_model(
    clean_items: pd.DataFrame, seed: int, num_warmup: int = 500, num_samples: int = 1000, num_chains: int = 2
) -> tuple[object, dict[int, int], StandardScaler]:
    """Fits the Bayesian model once on all of clean. Returns the fit, the
    training question-index mapping, and the feature scaler - both
    evaluations below must reuse this same mapping and scaler (re-fitting
    the scaler on verbose would erase the very shift being tested).
    """
    X_train, y_train, train_groups = build_transfer_xy(clean_items)
    train_group_idx, n_groups, question_id_to_index = build_group_index(train_groups)
    scaler = StandardScaler().fit(X_train)
    mcmc = fit_nuts(
        scaler.transform(X_train),
        y_train,
        train_group_idx,
        n_groups,
        seed=seed,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )
    return mcmc, question_id_to_index, scaler


def compute_condition_entropy(
    mcmc: object, items: pd.DataFrame, question_id_to_index: dict[int, int], scaler: StandardScaler
) -> pd.DataFrame:
    """Per-item aleatoric/epistemic for one condition, from the shared fit."""
    X, y, groups = build_transfer_xy(items)
    draws = np.asarray(predict_in_sample(mcmc, scaler.transform(X), groups, question_id_to_index))
    decomposition = posterior_predictive_entropy_decomposition(draws)
    return pd.DataFrame(
        {"question_id": groups, "aleatoric": decomposition["aleatoric"], "epistemic": decomposition["epistemic"]}
    )


def compute_verbose_shift_gap(
    clean_entropy: pd.DataFrame, verbose_entropy: pd.DataFrame, column: str, seed: int
) -> dict:
    """Paired cluster-bootstrap CI on mean(verbose) - mean(clean) (invariant 3)."""
    def _mean(df: pd.DataFrame) -> float:
        return float(df[column].mean())

    diff, ci_low, ci_high = paired_cluster_bootstrap(
        verbose_entropy, clean_entropy, _mean, "question_id", n=2000, seed=seed
    )
    return {"gap": diff, "ci_low": ci_low, "ci_high": ci_high}


def main_verbose_shift(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    clean_items, verbose_items = load_rq3b_items(config.paths.items_parquet)
    print(f"Verbose-shift population: N={len(clean_items)} paired items (clean/P1 vs verbose/P1)")
    print(f"Feature set (D21 parity fix): {TRANSFER_SAFE_COLUMNS}")

    print("Fitting Bayesian model ONCE on all of clean (no CV, mirrors 5.7's transfer-test design)...")
    mcmc, question_id_to_index, scaler = fit_verbose_shift_model(clean_items, config.seed)

    clean_entropy = compute_condition_entropy(mcmc, clean_items, question_id_to_index, scaler)
    verbose_entropy = compute_condition_entropy(mcmc, verbose_items, question_id_to_index, scaler)

    aleatoric_clean = float(clean_entropy["aleatoric"].mean())
    aleatoric_verbose = float(verbose_entropy["aleatoric"].mean())
    epistemic_clean = float(clean_entropy["epistemic"].mean())
    epistemic_verbose = float(verbose_entropy["epistemic"].mean())
    print(f"Aleatoric: clean={aleatoric_clean:.4f}, verbose={aleatoric_verbose:.4f}")
    print(f"Epistemic: clean={epistemic_clean:.4f}, verbose={epistemic_verbose:.4f}")

    aleatoric_gap = compute_verbose_shift_gap(clean_entropy, verbose_entropy, "aleatoric", config.seed)
    epistemic_gap = compute_verbose_shift_gap(clean_entropy, verbose_entropy, "epistemic", config.seed)
    print(
        f"Aleatoric gap (verbose - clean): {aleatoric_gap['gap']:.4f} "
        f"[{aleatoric_gap['ci_low']:.4f}, {aleatoric_gap['ci_high']:.4f}]"
    )
    print(
        f"Epistemic gap (verbose - clean): {epistemic_gap['gap']:.4f} "
        f"[{epistemic_gap['ci_low']:.4f}, {epistemic_gap['ci_high']:.4f}]"
    )

    epistemic_rose = epistemic_gap["ci_low"] > 0
    aleatoric_flat = aleatoric_gap["ci_low"] <= 0 <= aleatoric_gap["ci_high"]
    prediction_held = epistemic_rose and aleatoric_flat

    table = pd.DataFrame.from_records(
        [
            {
                "aleatoric_clean_mean": aleatoric_clean,
                "aleatoric_verbose_mean": aleatoric_verbose,
                "aleatoric_gap": aleatoric_gap["gap"],
                "aleatoric_gap_ci_low": aleatoric_gap["ci_low"],
                "aleatoric_gap_ci_high": aleatoric_gap["ci_high"],
                "epistemic_clean_mean": epistemic_clean,
                "epistemic_verbose_mean": epistemic_verbose,
                "epistemic_gap": epistemic_gap["gap"],
                "epistemic_gap_ci_low": epistemic_gap["ci_low"],
                "epistemic_gap_ci_high": epistemic_gap["ci_high"],
                "epistemic_rose": epistemic_rose,
                "aleatoric_flat": aleatoric_flat,
                "prediction_held": prediction_held,
            }
        ]
    )
    table_path = f"{config.paths.results_dir}/rq5_verbose_shift_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    verdict = "HELD" if prediction_held else "DID NOT HOLD"
    print(f"Preregistered prediction (epistemic rises, aleatoric stays flat, D21/D23): {verdict}")

    plot_rq5_verbose_shift(
        aleatoric_clean=aleatoric_clean,
        aleatoric_verbose=aleatoric_verbose,
        aleatoric_gap_ci=(aleatoric_gap["ci_low"], aleatoric_gap["ci_high"]),
        epistemic_clean=epistemic_clean,
        epistemic_verbose=epistemic_verbose,
        epistemic_gap_ci=(epistemic_gap["ci_low"], epistemic_gap["ci_high"]),
        model_slug=config.model_slug,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--task",
        required=True,
        choices=["ablation", "h4", "transfer", "category", "calibration", "bayesian_comparison", "verbose_shift"],
    )
    args = parser.parse_args()
    if args.task == "ablation":
        main_ablation(args.config)
    elif args.task == "h4":
        main_h4(args.config)
    elif args.task == "transfer":
        main_transfer(args.config)
    elif args.task == "category":
        main_category(args.config)
    elif args.task == "calibration":
        main_calibration(args.config)
    elif args.task == "bayesian_comparison":
        main_bayesian_comparison(args.config)
    else:
        main_verbose_shift(args.config)
