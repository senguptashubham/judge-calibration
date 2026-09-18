"""RQ4: repeated StratifiedGroupKFold cross-validation for the two
frequentist error predictors (task 5.3, D8/CLAUDE.md invariant 1). Fits
`LogisticRegression(C=1.0)` (standardized features) and
`HistGradientBoostingClassifier(max_depth=3, max_iter=200,
learning_rate=0.05)` - the two frequentist models invariant 11 permits
(the third, the Bayesian hierarchical model, lives in src/bayesian.py,
D22 - a different tool for a different question, not a competitor to
tune away).

Never `GroupKFold` (invariant 1): sklearn's `GroupKFold` has no
`shuffle` parameter, so repeating it across seeds silently produces
IDENTICAL splits every time - a fake stability result, not a real one.
Always `StratifiedGroupKFold(shuffle=True, random_state=seed)`, which
stratifies on the target while keeping every `question_id`'s rows
together in one fold - no `question_id` ever appears in both a fold's
train and test set.

5-fold CV, repeated over 10 independently-seeded runs (D8): only ~80
question_id groups means fold-to-fold variance is real at that scale
(effective N for split-to-split noise is ~80, not ~1800 rows), so the
headline uncertainty this file's callers should report is the
ACROSS-REPEAT spread of each repeat's whole-population AUROC, never a
single split's own CI.

`python -m src.predictor --config configs/run.yaml --tier B --model logreg`
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

# Tier B/C carry these non-numeric columns (features.py's own module
# docstring defers encoding to this file, task 5.3's job, not 5.2's).
# Tier A has neither - encode_features() is then a no-op on it.
_CATEGORICAL_COLUMNS = ("category",)
_BOOLEAN_COLUMNS = ("longer_is_chosen", "flipped")


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encodes `category` and casts the boolean feature columns
    to int - both target-independent, deterministic transforms (MT-Bench's
    8 categories are a fixed, known vocabulary; a bool -> int cast reads
    no data statistics at all), so - unlike StandardScaler, which must be
    fit inside each CV fold to avoid leaking test-fold statistics into
    train (handled inside make_logreg()'s Pipeline, not here) - this is
    safe to apply exactly ONCE, before cross-validation ever starts.

    A no-op on Tier A (has neither column).
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
    preregistered frequentist baseline (invariant 11). `seed` is accepted
    and ignored (default solver "lbfgs" has no randomness to seed) purely
    so this shares one call signature with make_histgbm() - both are
    usable interchangeably as `make_model(repeat_seed)`.

    `max_iter=1000` (default 100) is a numerical-convergence setting, NOT
    a hyperparameter deviation from the preregistered `C=1.0` - it exists
    only so the optimizer reliably reaches its actual optimum on this
    feature count rather than stopping early and emitting a
    ConvergenceWarning; it does not change what's being optimized or
    introduce any search.

    StandardScaler lives inside the Pipeline, not applied upfront in
    encode_features() - so `.fit()` on a training fold alone determines
    its mean/std, never the held-out test fold's (the standard scaling-
    inside-CV leakage guard).
    """
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000))])


def make_histgbm(seed: int = 0) -> HistGradientBoostingClassifier:
    """HistGradientBoostingClassifier(max_depth=3, max_iter=200,
    learning_rate=0.05) - the fixed, preregistered nonlinearity check
    (invariant 11). No scaling - tree splits are scale-invariant.

    `random_state=seed` is the one exception to "no tuning": it controls
    the model's OWN internal randomness (histogram-binning tie-breaks),
    not a hyperparameter search, and is set to the fold's own repeat seed
    so the whole pipeline stays fully reproducible end to end.
    """
    return HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=seed)


MODEL_FACTORIES: dict[str, Callable[[int], object]] = {
    "logreg": make_logreg,
    "histgbm": make_histgbm,
}


def make_fold_splits(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One 5-fold StratifiedGroupKFold split (D8). `shuffle=True` and a
    per-repeat `random_state=seed` are both mandatory: GroupKFold has no
    shuffle at all (invariant 1's own trap), and a fixed random_state is
    what makes ONE seed's split reproducible while still differing
    between seeds (D8's repeated-CV protocol depends on that).

    Pulled out as its own function - no model fitting inside it at all -
    specifically so invariant 1's two mandatory properties (no group ever
    split across a fold's train/test; different seeds give different
    assignments) are directly testable without fitting any classifier,
    the same reasoning judge.py's pending_calls() is kept separate from
    _run_generation().

    Returns:
        List of (train_idx, test_idx) - positional (iloc-style) index
        arrays into X/y/groups, one pair per fold, in sklearn's own
        split() order.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(X, y, groups))


@dataclass
class RepeatResult:
    """One repeat's worth of repeated_stratified_group_kfold()'s output.

    oof_pred is in POSITIONAL order matching X.iloc - i.e. oof_pred[i]
    is the prediction for X.iloc[i], regardless of what X's own pandas
    index labels happen to be. Callers that need to zip this back to
    item_id/question_id must do so positionally against the SAME X they
    passed in, not via X's index.
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
    """D8's full protocol: `n_repeats` independent 5-fold
    StratifiedGroupKFold CVs, each with its own seed (`seed`, `seed+1`,
    ..., `seed+n_repeats-1`) - shuffling plus a different random_state
    per repeat is what makes the repeats independent draws rather than
    the identical split repeated `n_repeats` times.

    Every row gets exactly one out-of-fold prediction per repeat (each
    row sits in exactly one fold's test set per 5-fold split). A fresh,
    unfitted model is built per FOLD (not reused across folds or
    repeats) via `make_model(repeat_seed)`, so nothing about a previous
    fold's fit can leak into the next.

    Args:
        X: feature matrix, already encoded (encode_features()'s output) -
            `correct`/`human_label`/etc. must NOT be columns of X.
        y: binary target (`correct`, as 0/1), same row order as X.
        groups: clustering column (`question_id`), same row order as X.
        make_model: seed -> a fresh, unfitted estimator/Pipeline
            (make_logreg or make_histgbm).
        n_splits: folds per repeat (5, D8).
        n_repeats: independent repeats (10, D8).
        seed: base seed - repeat i uses seed + i.

    Returns:
        One RepeatResult per repeat, in seed order.
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


TIER_BUILDERS = {"A": build_tier_a, "B": build_tier_b, "C": build_tier_c}


def run_predictor(
    population: pd.DataFrame,
    tier_builder: Callable[[pd.DataFrame], pd.DataFrame],
    model_name: str,
    seed: int,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> list[RepeatResult]:
    """Ties features.py::load_rq4_population()'s output to one tier +
    one model's full repeated-CV run: builds the tier's feature matrix,
    encodes it (encode_features), and cross-validates against `correct`
    (D7's single-pass, deployed-verdict definition - RQ4 predicts THAT
    error, not verdict_bidir's), grouped on `question_id`.

    Args:
        population: features.py::load_rq4_population()'s output.
        tier_builder: build_tier_a, build_tier_b, or build_tier_c
            (TIER_BUILDERS's values).
        model_name: "logreg" or "histgbm" (MODEL_FACTORIES's keys).
        seed: base seed - repeat i uses seed + i (D8).
        n_splits, n_repeats: D8's 5-fold / 10-repeat protocol - override
            only for tests, never for a reported result.

    Returns:
        One RepeatResult per repeat, in seed order.
    """
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"model_name must be one of {list(MODEL_FACTORIES)}, got {model_name!r}")

    X = encode_features(tier_builder(population))
    y = population["correct"].astype(int).to_numpy()
    groups = population["question_id"].to_numpy()

    return repeated_stratified_group_kfold(
        X, y, groups, MODEL_FACTORIES[model_name], n_splits=n_splits, n_repeats=n_repeats, seed=seed
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tier", required=True, choices=list(TIER_BUILDERS))
    parser.add_argument("--model", required=True, choices=list(MODEL_FACTORIES))
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    population = load_rq4_population(config.paths.items_parquet)

    repeats = run_predictor(population, TIER_BUILDERS[args.tier], args.model, config.seed)
    aurocs = np.array([r.auroc for r in repeats])

    print(f"Tier {args.tier}, {args.model}, N={len(population)}, {len(repeats)} repeats:")
    print(f"  AUROC mean={aurocs.mean():.4f}, spread=[{aurocs.min():.4f}, {aurocs.max():.4f}]")
    print("  (D8: the across-repeat spread is the headline uncertainty, not a within-split CI)")
