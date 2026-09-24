"""RQ4 feature tiers (PLAN.md §2.2). Each tier strictly adds to the last:

    A: the judge's own uncertainty signals
    B: A + surface properties of the input/output the judge doesn't use
    C: B + token-distribution detail (verdict margin, CoT logprob/entropy)

All three are built on ONE population, so a score change between tiers
reflects signal content, not a different set of items. Each build_tier_*()
returns only that tier's feature columns, index-aligned with the population
- never the target or the grouping key. Encoding (one-hot `category`,
bool -> int) is a modelling choice and lives in predictor.py.
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
    """The population every RQ4 tier uses (N=1,819): (clean, P1) only
    (invariant 14), human_label present, minus the 17 items where
    len_ratio/longer_is_chosen are undefined (3 empty responses, 14 exact
    length ties). Those 17 are dropped from all three tiers, even though
    Tier A never reads those columns, so tier comparisons stay
    same-population.

    Duplicates analysis/rq1.py's first two filters rather than importing
    them: src/ must not depend on analysis/.
    """
    items = pd.read_parquet(items_parquet)
    items = items[(items["condition"] == "clean") & (items["prompt_variant"] == "P1")]
    items = items[items["human_label"].notna()]
    items = items[items["len_ratio"].notna() & items["longer_is_chosen"].notna()]
    return items


def build_tier_a(items: pd.DataFrame) -> pd.DataFrame:
    """Tier A: the confidence signals + conf_ens's entropy components (D20).
    Do the signals combine, or are they redundant?
    """
    return items[TIER_A_COLUMNS].copy()


def build_tier_b(items: pd.DataFrame) -> pd.DataFrame:
    """Tier B: A + response lengths, turn, category, the judge's own output
    length, and flipped. Is judge error predictable from surface features
    the judge ignores?
    """
    return items[TIER_B_COLUMNS].copy()


def build_tier_c(items: pd.DataFrame) -> pd.DataFrame:
    """Tier C: B + the verdict-token margin and the CoT logprob/entropy
    aggregates, greedy and sampled (D4). Is the usable signal in the token
    distribution rather than the verbalization?
    """
    return items[TIER_C_COLUMNS].copy()
