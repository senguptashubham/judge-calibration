"""Tests for src/features.py: tier column sets, no-leakage, population
filtering. See TASKS.md task 5.2.

Uses a hand-built synthetic items.parquet-shaped DataFrame throughout,
not the real (gitignored) results/items.parquet - tests must be
hermetic and pass on a fresh clone before any run happens (matches the
convention every other tests/test_*.py in this repo already follows,
e.g. test_judge.py's tmp_path checkpoints, test_data.py's hand-built
vote groups).
"""

import pandas as pd
import pytest

from src.features import (
    TIER_A_COLUMNS,
    TIER_B_COLUMNS,
    TIER_C_COLUMNS,
    build_tier_a,
    build_tier_b,
    build_tier_c,
    load_rq4_population,
)

_LEAKAGE_COLUMNS = {"correct", "correct_bidir", "human_label", "frac_prefer_a"}


def _fake_items(n: int = 3, **overrides) -> pd.DataFrame:
    """One synthetic items.parquet row per index 0..n-1, every column the
    real table carries (CLAUDE.md §3's schema) with deterministic, always-
    defined values - including every Tier A/B/C column and the four
    leakage columns task 5.2's DoD explicitly checks are excluded.

    `overrides`: {column_name: list-of-n-values} to replace specific
    columns for one test (e.g. injecting a None to test filtering).
    """
    base = {
        "item_id": [f"item{i}" for i in range(n)],
        "question_id": list(range(n)),
        "category": ["writing"] * n,
        "condition": ["clean"] * n,
        "prompt_variant": ["P1"] * n,
        "turn": [1] * n,
        "human_label": ["A"] * n,
        "n_human_votes": [2] * n,
        "frac_prefer_a": [0.5] * n,
        "human_unanimous": [True] * n,
        "human_agreed": [True] * n,
        "d_human": [0.0] * n,
        "judge_verdict": ["A"] * n,
        "verdict_bidir": ["A"] * n,
        "conf_verb": [0.9] * n,
        "conf_lp": [0.8] * n,
        "conf_sc": [0.75] * n,
        "conf_bpe": [0.7] * n,
        "correct": [True] * n,
        "correct_bidir": [True] * n,
        "len_a": [100] * n,
        "len_b": [80] * n,
        "len_ratio": [1.25] * n,
        "abs_len_diff": [20] * n,
        "longer_is_chosen": [True] * n,
        "judge_output_len": [250] * n,
        "verdict_margin": [0.6] * n,
        "cot_logprob_mean_greedy": [-0.5] * n,
        "cot_logprob_min_greedy": [-1.0] * n,
        "cot_logprob_std_greedy": [0.2] * n,
        "cot_logprob_p10_greedy": [-0.8] * n,
        "cot_entropy_mean_greedy": [0.3] * n,
        "n_cot_tokens_greedy": [40] * n,
        "cot_logprob_mean_sampled_t07": [-0.6] * n,
        "cot_logprob_min_sampled_t07": [-1.1] * n,
        "cot_logprob_std_sampled_t07": [0.25] * n,
        "cot_logprob_p10_sampled_t07": [-0.9] * n,
        "cot_entropy_mean_sampled_t07": [0.35] * n,
        "n_cot_tokens_sampled_t07": [42.0] * n,
        "flipped": [False] * n,
        "ens_entropy_total": [0.2] * n,
        "ens_entropy_aleatoric": [0.15] * n,
        "ens_entropy_epistemic": [0.05] * n,
        "conf_ens": [0.8] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


# --- build_tier_a/b/c: column sets, additivity, no leakage -----------------


def test_build_tier_a_returns_exactly_tier_a_columns():
    items = _fake_items()
    tier_a = build_tier_a(items)
    assert set(tier_a.columns) == set(TIER_A_COLUMNS)
    assert len(tier_a) == len(items)


def test_build_tier_b_returns_exactly_tier_b_columns():
    items = _fake_items()
    tier_b = build_tier_b(items)
    assert set(tier_b.columns) == set(TIER_B_COLUMNS)


def test_build_tier_c_returns_exactly_tier_c_columns():
    items = _fake_items()
    tier_c = build_tier_c(items)
    assert set(tier_c.columns) == set(TIER_C_COLUMNS)


def test_tiers_are_strictly_additive():
    a, b, c = set(TIER_A_COLUMNS), set(TIER_B_COLUMNS), set(TIER_C_COLUMNS)
    assert a < b < c  # strict subset - each tier adds at least one column


def test_no_leakage_columns_in_any_tier():
    for tier_columns in (TIER_A_COLUMNS, TIER_B_COLUMNS, TIER_C_COLUMNS):
        assert _LEAKAGE_COLUMNS.isdisjoint(tier_columns)


def test_no_nans_in_any_tier_on_a_fully_populated_population():
    items = _fake_items(n=5)
    for build_fn in (build_tier_a, build_tier_b, build_tier_c):
        assert build_fn(items).isna().sum().sum() == 0


# --- load_rq4_population: filtering logic -----------------------------


def test_load_rq4_population_keeps_only_clean_p1(tmp_path):
    items = pd.concat(
        [
            _fake_items(n=1, item_id=["a"]),
            _fake_items(n=1, item_id=["b"], condition=["verbose"]),
            _fake_items(n=1, item_id=["c"], prompt_variant=["P2"]),
        ],
        ignore_index=True,
    )
    path = tmp_path / "items.parquet"
    items.to_parquet(path)

    population = load_rq4_population(str(path))

    assert list(population["item_id"]) == ["a"]


def test_load_rq4_population_drops_rows_with_no_human_label(tmp_path):
    items = pd.concat(
        [
            _fake_items(n=1, item_id=["a"]),
            _fake_items(n=1, item_id=["b"], human_label=[None]),
        ],
        ignore_index=True,
    )
    path = tmp_path / "items.parquet"
    items.to_parquet(path)

    population = load_rq4_population(str(path))

    assert list(population["item_id"]) == ["a"]


def test_load_rq4_population_drops_rows_with_undefined_len_ratio_or_longer_is_chosen(tmp_path):
    items = pd.concat(
        [
            _fake_items(n=1, item_id=["a"]),
            _fake_items(n=1, item_id=["b"], len_ratio=[None]),  # e.g. len_b == 0
            _fake_items(n=1, item_id=["c"], longer_is_chosen=[None]),  # e.g. a length tie
        ],
        ignore_index=True,
    )
    path = tmp_path / "items.parquet"
    items.to_parquet(path)

    population = load_rq4_population(str(path))

    assert list(population["item_id"]) == ["a"]


def test_load_rq4_population_then_every_tier_is_still_nan_free(tmp_path):
    # End-to-end: the filtering in load_rq4_population() is specifically
    # what makes task 5.2's "no NaNs" DoD hold - confirm the composition
    # of the two pieces, not just each in isolation.
    items = pd.concat(
        [
            _fake_items(n=2, item_id=["a", "b"]),
            _fake_items(n=1, item_id=["c"], len_ratio=[None]),
            _fake_items(n=1, item_id=["d"], human_label=[None]),
            _fake_items(n=1, item_id=["e"], condition=["verbose"]),
        ],
        ignore_index=True,
    )
    path = tmp_path / "items.parquet"
    items.to_parquet(path)

    population = load_rq4_population(str(path))
    assert list(population["item_id"]) == ["a", "b"]

    for build_fn in (build_tier_a, build_tier_b, build_tier_c):
        tier = build_fn(population)
        assert tier.isna().sum().sum() == 0
        assert len(tier) == 2
