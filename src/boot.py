"""cluster_bootstrap, paired_cluster_bootstrap. See TASKS.md task 2.4."""

from typing import Callable

import numpy as np
import pandas as pd


def cluster_bootstrap(
    df: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    group_col: str,
    n: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Cluster bootstrap CI for `stat_fn(df)`, resampling whole `group_col`
    values (e.g. question_id) rather than individual rows (CLAUDE.md
    invariant 2).

    Why row-resampling is wrong here: rows sharing a `group_col` value are
    correlated (same underlying question), not independent draws. A naive
    row bootstrap treats them as independent anyway, which understates the
    true sampling variance and produces CIs that are too narrow - findings
    look significant when they're actually noise at the ~80-question level,
    not the ~1000-row level. Resampling `group_col` values and taking every
    row that belongs to each sampled value preserves that correlation
    structure inside every replicate.

    One replicate:
      1. Draw len(unique(df[group_col])) values from the unique group_col
         values, WITH replacement, using `rng` (never bare np.random.* -
         CLAUDE.md sec 5).
      2. Build the resampled DataFrame: every row belonging to each sampled
         group value, duplicated if that value was drawn more than once.
      3. stat_fn(resampled_df) -> one replicate.

    Repeat `n` times; the CI is the (1-ci)/2 and (1+ci)/2 percentiles of the
    `n` replicates (percentile bootstrap - same convention as task 5.6's
    RQ4 interaction CI). The reported point estimate is stat_fn(df) on the
    REAL data, not the mean of the replicates - the replicates characterize
    spread, they don't re-estimate the center.

    Args:
        df: one row per observation; must contain `group_col`.
        stat_fn: takes a DataFrame (same columns as df), returns a float.
        group_col: the clustering column - question_id throughout this
            project.
        n: number of bootstrap replicates.
        ci: confidence level, e.g. 0.95 for a 95% CI.
        seed: required and explicit (CLAUDE.md sec 5 - no bare np.random.*,
            every stochastic function takes an explicit seed or rng).

    Returns:
        (point_estimate, ci_low, ci_high).
    """
    if group_col not in df.columns:
        raise ValueError(f"{group_col!r} not found in DataFrame")

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
    """Paired cluster bootstrap CI for `stat_fn(df_a) - stat_fn(df_b)`
    (CLAUDE.md invariant 3 - condition comparisons like clean vs. verbose,
    or AB vs. BA, are the SAME items measured twice, never two independent
    samples).

    The pairing requirement: each bootstrap replicate must resample
    `group_col` values ONCE, then use that SAME set of sampled values to
    build the resampled DataFrame on BOTH df_a and df_b before computing
    their difference. Resampling df_a and df_b independently would silently
    throw away the pairing and collapse this back into an (invalid) unpaired
    two-sample comparison - this is the one detail that makes this function
    different from calling cluster_bootstrap() twice and subtracting.

    df_a and df_b are expected to share the same universe of `group_col`
    values (e.g. both are `(clean, P1)` and `(verbose, P1)` items on the
    same question_ids) - a sampled group value that exists in one but not
    the other means df_a/df_b weren't actually paired to begin with.

    Args:
        df_a: first condition's rows.
        df_b: second condition's rows, paired with df_a on `group_col`.
        stat_fn: takes a DataFrame, returns a float. Applied separately to
            the resampled df_a and resampled df_b each replicate.
        group_col: the clustering column - question_id throughout this
            project.
        n: number of bootstrap replicates.
        ci: confidence level, e.g. 0.95 for a 95% CI.
        seed: required and explicit - see cluster_bootstrap().

    Returns:
        (point_diff, ci_low, ci_high) for stat_fn(df_a) - stat_fn(df_b).
        If the CI excludes 0, the difference is significant at this `ci`
        level.
    """
    if group_col not in df_a.columns:
        raise ValueError(f"{group_col!r} not found in df_a")

    if group_col not in df_b.columns:
        raise ValueError(f"{group_col!r} not found in df_b")

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

    # sorted(), not list(set_a) - set iteration order is a CPython
    # implementation detail, not a language guarantee. Sorting makes the
    # draw order (and therefore, for a fixed seed, the exact replicates)
    # deterministic by construction rather than by incidental hashing
    # behavior - the same principle cluster_bootstrap() gets for free from
    # df[group_col].unique()'s order-of-first-appearance. key=str rather
    # than a bare sort: group_col values are Hashable in general (pandas'
    # own typing for groupby().groups.keys()), not necessarily comparable -
    # question_id is always int in this project, but str() sorting keeps
    # this correct for any hashable group_col without narrowing the type.
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
