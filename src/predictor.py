"""RQ4: repeated StratifiedGroupKFold cross-validation for the two
frequentist error predictors (D8, invariants 1, 11, 12) - LogisticRegression
and HistGradientBoostingClassifier. The Bayesian model is src/bayesian.py.

Never `GroupKFold`: it has no `shuffle` parameter, so repeating it across
seeds reproduces the identical split every time - a fake stability result.
`StratifiedGroupKFold(shuffle=True, random_state=seed)` stratifies on the
target while keeping every question_id's rows in one fold.

5-fold CV repeated over 10 seeds (D8). With only ~80 questions, effective N
for split-to-split noise is ~80, not ~1,800 rows, so the headline
uncertainty is the ACROSS-REPEAT spread of each repeat's AUROC, never one
split's CI.

`python -m src.predictor --config configs/run.yaml --tier B --model logreg [--permutation-null]`
"""

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.features import build_tier_a, build_tier_b, build_tier_c, load_rq4_population

_CATEGORICAL_COLUMNS = ("category",)
_BOOLEAN_COLUMNS = ("longer_is_chosen", "flipped")


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot `category`, cast boolean features to int. Both transforms
    read no data statistics (the 8 categories are a fixed vocabulary), so
    unlike StandardScaler they are safe to apply once, before CV. A no-op
    on Tier A.
    """
    X = X.copy()
    present_categorical = [c for c in _CATEGORICAL_COLUMNS if c in X.columns]
    if present_categorical:
        X = pd.get_dummies(X, columns=present_categorical)
    for col in _BOOLEAN_COLUMNS:
        if col in X.columns:
            X[col] = X[col].astype(int)
    return X


def make_logreg(seed: int = 0) -> Pipeline:
    """LogisticRegression(C=1.0) on standardized features - the fixed,
    preregistered baseline (invariant 11).

    StandardScaler sits inside the Pipeline, so it is fit on each training
    fold only and never sees held-out data. `max_iter=1000` only lets the
    optimizer converge; it changes nothing about what is optimized. `seed`
    is unused (lbfgs is deterministic) and exists so both factories share a
    signature.
    """
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000))])


def make_histgbm(seed: int = 0) -> HistGradientBoostingClassifier:
    """HistGradientBoostingClassifier(max_depth=3, max_iter=200,
    learning_rate=0.05) - the fixed nonlinearity check (invariant 11).
    Trees are scale-invariant, so no scaling. `random_state` seeds only the
    model's own tie-breaking, not a search.
    """
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=seed)


MODEL_FACTORIES: dict[str, Callable[[int], object]] = {
    "logreg": make_logreg,
    "histgbm": make_histgbm,
}


def make_fold_splits(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One StratifiedGroupKFold split. Kept free of any model fitting so
    invariant 1's two properties - no group on both sides of a fold, and
    different seeds give different splits - are testable directly.

    Returns (train_idx, test_idx) pairs of positional indices.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(X, y, groups))


@dataclass
class RepeatResult:
    """One repeat of repeated_stratified_group_kfold(). `oof_pred[i]` is the
    prediction for X.iloc[i] - positional, not by X's index labels.
    """

    seed: int
    oof_pred: np.ndarray
    auroc: float


def repeated_stratified_group_kfold(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    make_model: Callable[[int], object],
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 0,
) -> list[RepeatResult]:
    """D8's protocol: `n_repeats` independent 5-fold splits, repeat i
    seeded with seed + i. Every row gets exactly one out-of-fold prediction
    per repeat, and a fresh model is built per fold, so nothing leaks
    between folds.

    Args:
        X: encoded features (encode_features()); no target columns.
        y: binary `correct` target, same row order as X.
        groups: question_id per row, same order.
        make_model: seed -> fresh unfitted estimator (make_logreg/make_histgbm).
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    results = []

    for i in range(n_repeats):
        repeat_seed = seed + i
        oof_pred = np.full(len(X), np.nan)

        for train_idx, test_idx in make_fold_splits(X, y, groups, n_splits, repeat_seed):
            model = make_model(repeat_seed)
            model.fit(X.iloc[train_idx], y[train_idx])
            oof_pred[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

        auroc = roc_auc_score(y, oof_pred)
        results.append(RepeatResult(seed=repeat_seed, oof_pred=oof_pred, auroc=auroc))

    return results


def shuffle_within_groups(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A copy of `y` with values permuted within each group only, so every
    group keeps exactly the same multiset of labels.
    """
    y_shuffled = y.copy()
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        y_shuffled[rows] = rng.permutation(y[rows])
    return y_shuffled


def permutation_null(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    make_model: Callable[[int], object],
    n: int = 200,
    n_splits: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Invariant 12: the null AUROC distribution, so an observed AUROC can
    be distinguished from what a feature set with no item-level
    relationship to `correct` would score through this exact pipeline.

    One permutation shuffles `y` WITHIN each question (shuffle_within_groups),
    then runs one grouped 5-fold CV. A global shuffle would also break the
    clustering of `correct` - some questions are simply harder - and so
    understate how high a no-information model can score by chance. Keeping
    each question's error rate intact is the stricter null: features that
    only track question difficulty still score above 0.5 under it.
    Permutation i seeds both the shuffle and the split with seed + i.

    Returns n null AUROCs.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    null_aurocs = np.empty(n)

    for i in range(n):
        permutation_seed = seed + i
        y_shuffled = shuffle_within_groups(y, groups, np.random.default_rng(permutation_seed))

        [result] = repeated_stratified_group_kfold(
            X, y_shuffled, groups, make_model, n_splits=n_splits, n_repeats=1, seed=permutation_seed
        )
        null_aurocs[i] = result.auroc

    return null_aurocs


def percentile_of_null(observed: float, null_distribution: np.ndarray) -> float:
    """Empirical percentile of `observed` within its null - e.g. 0.97 means
    only 3% of shuffled runs scored as high. Empirical rather than a
    parametric p-value, since AUROC's null isn't guaranteed normal.
    """
    return float(np.mean(null_distribution <= observed))


TIER_BUILDERS = {"A": build_tier_a, "B": build_tier_b, "C": build_tier_c}


def run_predictor(
    population: pd.DataFrame,
    tier_builder: Callable[[pd.DataFrame], pd.DataFrame],
    model_name: str,
    seed: int,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> list[RepeatResult]:
    """One tier + one model through the full repeated CV, predicting
    `correct` (D7's single-pass verdict), grouped on question_id.
    Override n_splits/n_repeats only in tests.
    """
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"model_name must be one of {list(MODEL_FACTORIES)}, got {model_name!r}")

    X, y, groups = build_xyg(population, tier_builder)

    return repeated_stratified_group_kfold(
        X, y, groups, MODEL_FACTORIES[model_name], n_splits=n_splits, n_repeats=n_repeats, seed=seed
    )


def build_xyg(
    population: pd.DataFrame, tier_builder: Callable[[pd.DataFrame], pd.DataFrame]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(X, y, groups): encoded tier features, `correct` as 0/1, question_id."""
    X = encode_features(tier_builder(population))
    y = population["correct"].astype(int).to_numpy()
    groups = population["question_id"].to_numpy()
    return X, y, groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", required=True, choices=list(TIER_BUILDERS))
    parser.add_argument("--model", required=True, choices=list(MODEL_FACTORIES))
    parser.add_argument(
        "--permutation-null",
        action="store_true",
        help="Also run the n=200 permutation null (invariant 12) - about 20x the model fits of the real result.",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    population = load_rq4_population(config.paths.items_parquet)

    repeats = run_predictor(population, TIER_BUILDERS[args.tier], args.model, config.seed)
    aurocs = np.array([r.auroc for r in repeats])

    print(f"Tier {args.tier}, {args.model}, N={len(population)}, {len(repeats)} repeats:")
    print(f"  AUROC mean={aurocs.mean():.4f}, spread=[{aurocs.min():.4f}, {aurocs.max():.4f}]")
    print("  (D8: the across-repeat spread is the headline uncertainty, not a within-split CI)")

    if args.permutation_null:
        X, y, groups = build_xyg(population, TIER_BUILDERS[args.tier])
        null_aurocs = permutation_null(X, y, groups, MODEL_FACTORIES[args.model], n=200, seed=config.seed)
        percentile = percentile_of_null(aurocs.mean(), null_aurocs)
        print(
            f"  Permutation null (n=200): mean={null_aurocs.mean():.4f}, "
            f"std={null_aurocs.std():.4f} (invariant 12: near 0.5, a little above if features track question difficulty)"
        )
        print(f"  Observed AUROC ({aurocs.mean():.4f}) is at the {percentile:.1%} percentile of the null")
