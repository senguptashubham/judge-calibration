"""Cluster bootstrap, paired cluster bootstrap, and the paired per-fold
AUROC bootstrap, all over question_id (invariants 2, 3; D32)."""

from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def cluster_bootstrap(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    group_col: str,
    n: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile-bootstrap CI for stat_fn(df), resampling whole `group_col`
    values (question_id) rather than rows (invariant 2).

    Why not rows: rows sharing a question are correlated, not independent
    draws. Resampling rows treats them as independent, understates the
    variance, and gives CIs that are too narrow - effective N is ~80
    questions, not ~1000 rows.

    One replicate draws len(unique groups) group values with replacement,
    takes every row of each drawn group (duplicated if drawn twice), and
    evaluates stat_fn. The CI is the (1-ci)/2 and (1+ci)/2 percentiles of
    `n` replicates. The point estimate is stat_fn on the real data, not the
    replicates' mean - the replicates measure spread, not the center.

    Returns (point_estimate, ci_low, ci_high).
    """
    if group_col not in df.columns:
        raise ValueError(f"{group_col!r} not found in DataFrame")

    if not df.index.is_unique:
        raise ValueError("df.index must be unique - rows are resampled by index label")

    if n <= 0:
        raise ValueError("n must be positive")

    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")

    rng = np.random.default_rng(seed)

    point_estimate = stat_fn(df)

    groups = df[group_col].unique()
    n_groups = len(groups)

    if n_groups == 0:
        raise ValueError("DataFrame contains no groups")

    group_indices = {
        group: df.index[df[group_col] == group].to_numpy()
        for group in groups
    }

    bootstrap_stats = np.empty(n)

    for i in range(n):
        sampled_groups = rng.choice(
            groups,
            size=n_groups,
            replace=True,
        )

        sampled_indices = np.concatenate(
            [group_indices[group] for group in sampled_groups]
        )

        resampled_df = df.loc[sampled_indices]

        bootstrap_stats[i] = stat_fn(resampled_df)

    alpha = 1 - ci

    ci_low, ci_high = np.quantile(
        bootstrap_stats,
        [alpha / 2, 1 - alpha / 2],
    )

    return float(point_estimate), float(ci_low), float(ci_high)


def paired_cluster_bootstrap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    group_col: str,
    n: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired cluster-bootstrap CI for stat_fn(df_a) - stat_fn(df_b)
    (invariant 3: clean vs verbose, or two signals on the same items, are
    the same items measured twice, never two independent samples).

    Each replicate resamples group values ONCE and uses that same draw for
    both df_a and df_b before taking the difference. Resampling the two
    independently would throw the pairing away - that is the whole
    difference from calling cluster_bootstrap() twice and subtracting.
    df_a and df_b must contain exactly the same group values.

    Returns (point_diff, ci_low, ci_high); a CI excluding 0 means the
    difference is significant at this level.
    """
    if group_col not in df_a.columns:
        raise ValueError(f"{group_col!r} not found in df_a")

    if group_col not in df_b.columns:
        raise ValueError(f"{group_col!r} not found in df_b")

    if not (df_a.index.is_unique and df_b.index.is_unique):
        raise ValueError("df_a/df_b indexes must be unique - rows are resampled by index label")

    if n <= 0:
        raise ValueError("n must be positive")

    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")

    groups_a = df_a.groupby(group_col).groups
    groups_b = df_b.groupby(group_col).groups

    set_a = set(groups_a.keys())
    set_b = set(groups_b.keys())

    if set_a != set_b:
        only_a = set_a - set_b
        only_b = set_b - set_a

        raise ValueError(
            "df_a and df_b do not contain the same group values. "
            f"Only in df_a: {only_a}; only in df_b: {only_b}"
        )

    # Sorted, so the draw order - and so each replicate, for a fixed seed -
    # doesn't depend on set iteration order. key=str because group values
    # need only be hashable, not comparable.
    groups = np.array(sorted(set_a, key=str))

    rng = np.random.default_rng(seed)

    point_diff = stat_fn(df_a) - stat_fn(df_b)

    bootstrap_diffs = np.empty(n)

    for i in range(n):

        sampled_groups = rng.choice(
            groups,
            size=len(groups),
            replace=True,
        )

        indices_a = np.concatenate(
            [groups_a[group] for group in sampled_groups]
        )

        indices_b = np.concatenate(
            [groups_b[group] for group in sampled_groups]
        )

        resampled_a = df_a.loc[indices_a]
        resampled_b = df_b.loc[indices_b]

        bootstrap_diffs[i] = (
            stat_fn(resampled_a)
            - stat_fn(resampled_b)
        )

    alpha = 1 - ci

    ci_low, ci_high = np.quantile(
        bootstrap_diffs,
        [alpha / 2, 1 - alpha / 2],
    )

    return float(point_diff), float(ci_low), float(ci_high)


def _auroc_error_fast(uncertainty: np.ndarray, is_error: np.ndarray) -> float:
    """metrics.auroc_error() through the Mann-Whitney rank formula, with
    tied values given their average rank (identical to roc_auc_score). The
    per-fold bootstrap below calls it hundreds of thousands of times, so it
    skips roc_auc_score's input checks. NaN when one class is missing.
    """
    n_error = int(is_error.sum())
    n_correct = len(is_error) - n_error
    if n_error == 0 or n_correct == 0:
        return float("nan")
    ranks = rankdata(uncertainty)
    return float((ranks[is_error].sum() - n_error * (n_error + 1) / 2) / (n_error * n_correct))


def paired_per_fold_bootstrap(
    uncertainty_a: list[np.ndarray],
    uncertainty_b: list[np.ndarray],
    correct: np.ndarray,
    groups: np.ndarray,
    fold_ids: list[np.ndarray],
    n: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
    correct_b: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Paired cluster-bootstrap CI for the per-fold AUROC difference
    (metrics.per_fold_auroc_error), averaged over CV repeats (D32):

        diff = mean over repeats r, folds k of
               auroc_error(a_r[fold k]) - auroc_error(b_r[fold k])

    Each list holds one array per repeat, all positionally aligned with
    `correct` and `groups`. A raw signal, which has no folds, is passed as
    the same array for every repeat; it is still scored fold by fold, on
    exactly the rows the meta-model's fold is scored on, so the two sides
    are compared like for like (invariant 3). `correct_b`, when given, is
    side b's own label for the same items: clean and padded copies of an
    item share a question and a fold but can differ in whether the judge
    was right.

    Each replicate resamples question values once (invariant 2), keeps every
    row's fold assignment fixed, and recomputes the difference on the
    resampled rows. Holding the folds fixed measures uncertainty over which
    questions were sampled, given the fitted fold models, which is the
    variation the comparison is about. A fold left with one class in a
    replicate is skipped in that replicate's average.

    Returns (point_diff, ci_low, ci_high).
    """
    if not (len(uncertainty_a) == len(uncertainty_b) == len(fold_ids)):
        raise ValueError("uncertainty_a, uncertainty_b and fold_ids need one array per repeat")
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")

    is_error = ~np.asarray(correct, dtype=bool)
    is_error_b = is_error if correct_b is None else ~np.asarray(correct_b, dtype=bool)
    groups = np.asarray(groups)
    a = [np.asarray(x, dtype=float) for x in uncertainty_a]
    b = [np.asarray(x, dtype=float) for x in uncertainty_b]
    folds = [np.asarray(f) for f in fold_ids]

    def _diff(rows: np.ndarray) -> float:
        per_repeat = []
        for a_r, b_r, f_r in zip(a, b, folds):
            fold_of_row = f_r[rows]
            per_fold = []
            for k in np.unique(fold_of_row):
                in_fold = rows[fold_of_row == k]
                per_fold.append(
                    _auroc_error_fast(a_r[in_fold], is_error[in_fold])
                    - _auroc_error_fast(b_r[in_fold], is_error_b[in_fold])
                )
            per_repeat.append(np.nanmean(per_fold))
        return float(np.mean(per_repeat))

    point_diff = _diff(np.arange(len(groups)))

    unique_groups = np.array(sorted(set(groups.tolist()), key=str))
    rows_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(seed)
    bootstrap_diffs = np.empty(n)
    for i in range(n):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        bootstrap_diffs[i] = _diff(np.concatenate([rows_by_group[g] for g in sampled]))

    alpha = 1 - ci
    ci_low, ci_high = np.quantile(bootstrap_diffs, [alpha / 2, 1 - alpha / 2])
    return float(point_diff), float(ci_low), float(ci_high)
