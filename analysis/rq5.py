"""RQ5, task 3.2b: does the epistemic component of the judge-level entropy
decomposition beat total entropy as a threshold-sweep filter? See
TASKS.md task 3.2b (D20, D23). Also tasks 5.9d (distillation comparison)
and 5.9e (human-disagreement validation) - the Week 5 growth this
module's own original docstring promised.

`python -m analysis.rq5 --config configs/run.yaml --task {threshold_sweep,distillation,human_disagreement}`

--- Task 5.9d (distillation comparison, D23) --------------------------

Population: features.py::load_rq4_population() (N=1819), NOT this
module's own wider load_rq5_items() (N=1836) - confirmed empirically
(21 Sep 2026) that RQ4's population is a STRICT SUBSET of RQ5's,
differing by exactly the 17 len_ratio-undefined items features.py
already documents dropping. Using the smaller, common population keeps
the ensemble-vs-single-call comparison apples-to-apples: the ensemble's
own conf_ens/ens_entropy_* signals are defined on the wider 1836, but
scoring them against a Bayesian model trained/evaluated on only 1819 of
those items would conflate "which items" with "which method is better."

Teacher = the 3-call ensemble's conf_ens / ens_entropy_epistemic (D20).
Student = the 1-call Bayesian model (5.9b/5.9c, Tier A, P1-only
features) - reuses analysis/rq4.py::compute_bayesian_arm() directly
rather than re-deriving the CV wiring a third time. Real cost: ~7
minutes (TASKS.md 5.9b/5.9c's own measured runtime) - no caching layer
exists anywhere in this project's analysis/ scripts, so this pays that
cost fresh again.

AUROC/ECE for conf_ens: genuinely NEW computation - analysis/rq1.py's
own SIGNALS list deliberately excludes conf_ens (it's D20's own signal,
outside RQ1/RQ2's scope), so it has never had its own AUROC/ECE
computed anywhere else in this project before this task.

Entropy quality: AUROC(ens_entropy_epistemic -> error) - already
uncertainty-typed, no 1-x flip, same convention this module's own
threshold-sweep task already established - vs AUROC(Bayesian's OWN
meta-model-level epistemic -> error), via bayesian.py::
posterior_predictive_entropy_decomposition() on compute_bayesian_arm()'s
pooled_draws. Entropy is symmetric in p vs 1-p, so it makes no
difference that pooled_draws is P(correct) rather than P(wrong) - the
resulting total/aleatoric/epistemic come out identical either way.

Every gap (AUROC, entropy-quality AUROC) gets a PAIRED cluster-bootstrap
CI - same items, two different scores, invariant 3's paired-comparison
logic, the SAME rename-to-a-shared-column trick this module's own
compute_epistemic_vs_total_gap() already uses for exactly this shape of
problem. ECE gets no such gap/CI - reported as two numbers side by side,
since there's no natural "which one is better by how much" statistic to
bootstrap the way there is for AUROC's rank-based difference.

"% of the ensemble's AUROC edge over chance retained at 1 call" uses 0.5
as the normalizing zero-point - a principled choice specific to AUROC
(chance = 0.5 is what "no information" means for a ranking statistic).
No equivalent framing is forced onto ECE or the entropy-quality AUROC -
those are reported as plain numbers/gaps, not squeezed into the same
percentage framing where it wouldn't have a natural interpretation.

Writes results/rq5_distillation_{model_slug}.csv (D26).

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

--- Task 5.9e (human-disagreement validation, D23) ---------------------

Population: analysis/human_disagreement.py::load_disagreement_items()
(N=595, D9's own population - clean/P1, n_human_votes >= 2) - REUSED
directly, not re-derived: the same population tasks 3.3/3.4/H4 already
use for d_human-based analysis, and it already handles d_human's own
float-tie rounding gotcha (see that function's own docstring - raw
|frac_prefer_a - 0.5| produces float64 values ~1e-17 apart for the "same"
true fraction, silently corrupting rank-based ties otherwise).

Correlation: reuses analysis/human_disagreement.py::
compute_d_human_correlation() directly (task 3.4's own Spearman +
cluster-bootstrap recipe, question_id-grouped) - NOT re-derived - over
["ens_entropy_aleatoric", "ens_entropy_epistemic"].

WARNING, resolved 21 Sep 2026: TASKS.md's own task 5.9e text has an
internal contradiction. It reads "aleatoric high specifically where
humans actually disagreed (high d_human)" - but src/data.py defines
d_human = |frac_prefer_a - 0.5|, so HIGH d_human means STRONG CONSENSUS
(votes clustered near 0% or 100%), not disagreement; LOW d_human is the
50/50-split disagreement zone. CLAUDE.md's own schema calls d_human
"consensus strength," and D23/H4's closeout both describe that same
direction independently - three sources agree against the one that
doesn't, and TASKS.md's parenthetical is a typo, confirmed with the
owner. Tested here as originally intended: aleatoric should correlate
NEGATIVELY with d_human (high aleatoric where consensus is weak, i.e.
low d_human, the genuine disagreement zone), and epistemic should show a
much weaker (or no) such relationship - NOT the literal-but-backwards
reading TASKS.md's own parenthetical would otherwise imply.

Writes results/rq5_human_disagreement_{model_slug}.csv (D26) +
results/figures/d_human_correlations_ensemble_entropy_{model_slug}.png
(src/plots.py's plot_d_human_correlations, reused via its own
filename_suffix/title params - added specifically so this call doesn't
silently overwrite task 3.4's own existing
d_human_correlations_{model_slug}.png, which plots a DIFFERENT signal
set on the same base filename).
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


# --- Task 5.9d (distillation comparison, D23) ---------------------------
#
# See this module's own docstring for the full population/metric
# rationale.


def compute_ensemble_metrics(items: pd.DataFrame, n_bins: int, seed: int) -> dict:
    """conf_ens's own AUROC(uncertainty -> error) and ECE - genuinely new
    computation, see this module's own docstring for why it's never been
    computed anywhere else in this project.
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
    """One row per item, both arms' relevant columns side by side - what
    every paired-gap bootstrap below needs, built once so the ensemble
    and Bayesian arms are never accidentally scored on rows in different
    orders.

    Args:
        population: load_rq4_population()'s output.
        bayesian_mean_oof_correct: compute_bayesian_arm()'s own
            mean_oof_correct, positionally aligned to population.
        bayesian_epistemic: posterior_predictive_entropy_decomposition()'s
            "epistemic" array on compute_bayesian_arm()'s pooled_draws,
            same alignment.
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
    """Paired cluster-bootstrap CI on AUROC(ensemble_col) -
    AUROC(bayesian_col) - same items, two different signals (invariant
    3's paired-comparison logic), the same rename-to-a-shared-column
    trick this module's own compute_epistemic_vs_total_gap() already
    uses for exactly this shape of problem.
    """
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
    table_path = f"results/rq5_distillation_{config.model_slug}.csv"
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


# --- Task 5.9e (human-disagreement validation, D23) ----------------------
#
# See this module's own docstring for the full population/direction
# rationale, INCLUDING the TASKS.md wording contradiction found and
# resolved (21 Sep 2026) - read that before touching the sign of
# anything here.

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
    table_path = f"results/rq5_human_disagreement_{config.model_slug}.csv"
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
    # Aleatoric "tracks disagreement" means its correlation with d_human
    # (consensus strength) is NEGATIVE and the CI excludes 0 - see this
    # module's own docstring for why negative, not positive, is the
    # correct direction to check.
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
