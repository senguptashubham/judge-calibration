"""Tests for src/boot.py: clustered CIs must be strictly wider than naive
row-resampled CIs on the same data.
"""

import numpy as np
import pandas as pd
import pytest

from src.boot import cluster_bootstrap, paired_cluster_bootstrap

_MEAN_STAT = lambda d: d["value"].mean()  # noqa: E731


def _correlated_dataset(seed=42, n_groups=40, rows_per_group=5):
    """Synthetic data shaped like "the same question judged several times":
    large between-group variance (mu_g), small within-group noise. This is
    exactly the correlation structure a naive row bootstrap can't see (it
    never resamples groups, only rows within whatever groups already
    happen to be in the sample) and cluster_bootstrap() exists to recover.
    """
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    mu_g = rng.normal(0, 1.0, size=n_groups)
    noise = rng.normal(0, 0.05, size=n_groups * rows_per_group)
    values = mu_g[group_ids] + noise
    return pd.DataFrame({"question_id": group_ids, "value": values})


def _naive_row_bootstrap(df, stat_fn, n=2000, seed=0):
    """Comparison baseline only - plain row-level bootstrap with no group
    awareness, exactly the approach CLAUDE.md invariant 2 forbids. Exists
    purely so test_cluster_bootstrap_wider_than_naive_row_bootstrap has a
    "too narrow" CI to compare against.
    """
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy()
    stats = np.empty(n)
    for i in range(n):
        sampled = rng.choice(idx, size=len(idx), replace=True)
        stats[i] = stat_fn(df.loc[sampled])
    alpha = 0.05
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return lo, hi


def test_cluster_bootstrap_wider_than_naive_row_bootstrap():
    # The actual DoD: on data with real within-cluster correlation, the
    # clustered CI must be strictly wider than a naive row-level bootstrap
    # CI on the same data - if it isn't, the clustering isn't doing
    # anything and invariant 2's whole justification falls apart.
    df = _correlated_dataset()

    _, lo_c, hi_c = cluster_bootstrap(df, _MEAN_STAT, "question_id", n=2000, seed=0)
    lo_n, hi_n = _naive_row_bootstrap(df, _MEAN_STAT, n=2000, seed=0)

    assert (hi_c - lo_c) > (hi_n - lo_n)


def test_cluster_bootstrap_point_estimate_matches_direct_computation():
    # Regression guard: the reported point estimate must be stat_fn(df) on
    # the real data, not the mean of the bootstrap replicates - an easy
    # mistake to introduce that wouldn't show up as a crash, just a
    # silently different number.
    df = _correlated_dataset()

    point, _, _ = cluster_bootstrap(df, _MEAN_STAT, "question_id", n=500, seed=0)

    assert point == pytest.approx(_MEAN_STAT(df))


def test_cluster_bootstrap_deterministic_with_same_seed():
    # Invariant 9's reproducibility promise rests on this: a fixed seed
    # must reproduce the exact same result, not just a similar one.
    df = _correlated_dataset()

    result_1 = cluster_bootstrap(df, _MEAN_STAT, "question_id", n=500, seed=7)
    result_2 = cluster_bootstrap(df, _MEAN_STAT, "question_id", n=500, seed=7)

    assert result_1 == result_2


def test_cluster_bootstrap_raises_on_missing_group_col():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError):
        cluster_bootstrap(df, _MEAN_STAT, "question_id", n=10, seed=0)


def test_cluster_bootstrap_raises_on_duplicate_index_labels():
    # Rows are resampled by index label, so duplicate labels (e.g. from a
    # concat without ignore_index) would silently pull extra rows per group.
    df = pd.concat([_correlated_dataset(seed=1), _correlated_dataset(seed=2)])

    with pytest.raises(ValueError, match="unique"):
        cluster_bootstrap(df, _MEAN_STAT, "question_id", n=10, seed=0)
    with pytest.raises(ValueError, match="unique"):
        paired_cluster_bootstrap(df, df, _MEAN_STAT, "question_id", n=10, seed=0)


def test_cluster_bootstrap_raises_on_invalid_n():
    df = _correlated_dataset()

    with pytest.raises(ValueError):
        cluster_bootstrap(df, _MEAN_STAT, "question_id", n=0, seed=0)


def test_cluster_bootstrap_raises_on_invalid_ci():
    df = _correlated_dataset()

    with pytest.raises(ValueError):
        cluster_bootstrap(df, _MEAN_STAT, "question_id", n=10, ci=1.5, seed=0)


def test_paired_cluster_bootstrap_identical_data_gives_exact_zero_diff():
    # The test that actually catches a broken pairing implementation: when
    # df_a and df_b are the SAME data, a correctly paired bootstrap uses
    # the identical sampled group set on both sides every replicate, so
    # stat_fn(resampled_a) - stat_fn(resampled_b) is EXACTLY 0 for every
    # single replicate - not just close to it. If pairing were silently
    # broken (e.g. "simplified" into two independent rng.choice() draws),
    # this would start showing nonzero spread even on identical data,
    # because independent resamples of the same data don't cancel by
    # chance.
    df = _correlated_dataset()

    point_diff, lo, hi = paired_cluster_bootstrap(
        df, df, _MEAN_STAT, "question_id", n=500, seed=0
    )

    assert point_diff == 0.0
    assert lo == 0.0
    assert hi == 0.0


def test_paired_cluster_bootstrap_raises_on_mismatched_groups():
    df_a = _correlated_dataset(n_groups=40)
    df_b = df_a[df_a["question_id"] < 39].copy()  # missing question_id 39

    with pytest.raises(ValueError):
        paired_cluster_bootstrap(df_a, df_b, _MEAN_STAT, "question_id", n=10, seed=0)


def test_paired_cluster_bootstrap_deterministic_with_same_seed():
    # Different underlying values, same group_col universe (both default
    # n_groups=40) - a valid, non-degenerate pairing target.
    df_a = _correlated_dataset(seed=1)
    df_b = _correlated_dataset(seed=2)

    result_1 = paired_cluster_bootstrap(df_a, df_b, _MEAN_STAT, "question_id", n=500, seed=7)
    result_2 = paired_cluster_bootstrap(df_a, df_b, _MEAN_STAT, "question_id", n=500, seed=7)

    assert result_1 == result_2
