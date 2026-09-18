"""RQ4 feature tiers A/B/C (task 5.2, PLAN.md §2.2). Each tier is
strictly additive, built on ONE shared population so a score change
between tiers is a real signal-content difference, not a population
artifact:

    A: the judge's own uncertainty signals
    B: A + surface properties of the input/output the judge doesn't use
    C: B + token-distribution detail (verdict margin, CoT logprob/entropy
       aggregates)

Population: results/items.parquet filtered to (clean, P1), human_label
not null (RQ1's own 1836-item population - invariant 14 + D7), MINUS 17
items where len_ratio/longer_is_chosen are legitimately undefined (3
zero-length responses, 14 exact-length ties - CLAUDE.md's own "propagate,
don't fabricate" convention, applied when len_a/len_b were added). Those
17 are dropped from ALL THREE tiers uniformly, even though Tier A alone
never touches those columns, specifically so "does tier B beat tier A"
stays a same-population comparison (18 Sep 2026 discussion). Final N =
1819 - confirmed empirically zero remaining NaNs across every tier
column at this population.

Each build_tier_*() function returns a DataFrame of ONLY that tier's
feature columns, index-aligned with load_rq4_population()'s output - NOT
carrying item_id/question_id/correct/human_label itself (that would blur
"feature" with "target"/"grouping key"; predictor.py joins back by the
shared index for the target and StratifiedGroupKFold's group column).

No encoding happens here (one-hot for `category`, bool->int for
`longer_is_chosen`/`flipped`) - that's a modelling choice
(LogisticRegression needs one-hot, HistGradientBoostingClassifier can
take native categoricals), predictor.py's job (task 5.3), not this
file's.
"""

import pandas as pd

TIER_A_COLUMNS = [
    "conf_verb",
    "conf_lp",
    "conf_sc",
    "conf_bpe",
    "conf_ens",
    "ens_entropy_total",
    "ens_entropy_aleatoric",
    "ens_entropy_epistemic",
]

TIER_B_EXTRA_COLUMNS = [
    "len_a",
    "len_b",
    "len_ratio",
    "abs_len_diff",
    "longer_is_chosen",
    "turn",
    "category",
    "judge_output_len",
    "flipped",
]

TIER_C_EXTRA_COLUMNS = [
    "verdict_margin",
    "cot_logprob_mean_greedy",
    "cot_logprob_min_greedy",
    "cot_logprob_std_greedy",
    "cot_logprob_p10_greedy",
    "cot_entropy_mean_greedy",
    "n_cot_tokens_greedy",
    "cot_logprob_mean_sampled_t07",
    "cot_logprob_min_sampled_t07",
    "cot_logprob_std_sampled_t07",
    "cot_logprob_p10_sampled_t07",
    "cot_entropy_mean_sampled_t07",
    "n_cot_tokens_sampled_t07",
]

TIER_B_COLUMNS = TIER_A_COLUMNS + TIER_B_EXTRA_COLUMNS
TIER_C_COLUMNS = TIER_B_COLUMNS + TIER_C_EXTRA_COLUMNS


def load_rq4_population(items_parquet: str) -> pd.DataFrame:
    """results/items.parquet -> the exact population every RQ4 tier is
    built on: (clean, P1) only (invariant 14), human_label not null (D7 -
    correct/correct_bidir are None otherwise; RQ1's own 1836-item
    population), MINUS the 17 items with an undefined len_ratio or
    longer_is_chosen (module docstring - dropped uniformly across all
    tiers, not just B/C, to keep the tier comparison on one population).

    Duplicated from analysis/rq1.py::load_rq1_items()'s first two filters
    rather than imported - src/ must not depend on analysis/ (same
    layering reason src/plots.py's _ols_fit()/_spearman_corr() give for
    duplicating analysis/human_disagreement.py's own versions).

    Args:
        items_parquet: config.paths.items_parquet.

    Returns:
        items.parquet's rows, filtered - full column set still present
        (feature tiers, id columns, target columns all included; callers
        pick what they need by column, e.g. build_tier_a(pop) for
        features or pop["correct"] for the target).
    """
    items = pd.read_parquet(items_parquet)
    items = items[(items["condition"] == "clean") & (items["prompt_variant"] == "P1")]
    items = items[items["human_label"].notna()]
    items = items[items["len_ratio"].notna() & items["longer_is_chosen"].notna()]
    return items


def build_tier_a(items: pd.DataFrame) -> pd.DataFrame:
    """Tier A: the judge's own uncertainty signals - conf_verb/conf_lp/
    conf_sc/conf_bpe/conf_ens + conf_ens's three entropy components
    (D20). Tests "do the five uncertainty signals combine, or are they
    redundant" (PLAN.md §2.2).
    """
    return items[TIER_A_COLUMNS].copy()


def build_tier_b(items: pd.DataFrame) -> pd.DataFrame:
    """Tier B: A + surface properties of the input/output the judge
    itself doesn't use as an uncertainty signal - response lengths,
    turn/category, the judge's own output length, and flipped (order-
    sensitivity). Tests "is judge error predictable from surface
    features the judge ignores" (PLAN.md §2.2).
    """
    return items[TIER_B_COLUMNS].copy()


def build_tier_c(items: pd.DataFrame) -> pd.DataFrame:
    """Tier C: B + token-distribution detail - the exact verdict-position
    top-2 margin, and the CoT logprob/entropy aggregates in both their
    greedy and sampled-at-T=0.7 forms (D4). Tests "is the usable signal
    in the token distribution rather than the verbalization" (PLAN.md
    §2.2).
    """
    return items[TIER_C_COLUMNS].copy()
