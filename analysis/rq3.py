"""RQ3: order-position bias (RQ3a) and verbosity bias (RQ3b) - does the
judge's stated uncertainty flag either bias-induced error, or is the fooled
judge confident? See TASKS.md tasks 4.3 (RQ3a) and 4.4 (RQ3b).

`python -m analysis.rq3 --config configs/run.yaml`.

--- RQ3a (task 4.3) ---

Reads results/items.parquet via load_rq1_items() (analysis/rq1.py) - the
same (clean, P1, human_label not null) population as RQ1/RQ2, N=1836. This
is a deliberate scope choice, not a data requirement: flip rate and
confidence-on-flipped don't need human_label at all (they're pure
judge-behavior signals, nothing to do with correctness), so the maximal
population would be all 1904 clean/P1 items. Reusing RQ1's 1836-item
population instead keeps one canonical N across every RQ1-RQ4 core
analysis, rather than needing to justify a second, 68-item-different
population size in REPORT.md for no analytical reason.

Computes two things, both cluster-bootstrap CIs (question_id, invariant 2):
  - flip rate: mean(flipped) over the population.
  - for each of the four original signals: mean confidence on flipped vs.
    unflipped items, and the DIFFERENCE between the two groups with its
    own CI - via a single cluster_bootstrap() call with a custom
    group-difference stat_fn, NOT paired_cluster_bootstrap(). The two
    functions solve different problems: paired_cluster_bootstrap needs two
    DataFrames sharing the same question_id universe (e.g. the same items
    scored two ways, like RQ1's judge_verdict vs. verdict_bidir
    comparison, or RQ5's same-items-two-signals comparison). flipped and
    unflipped are two DISJOINT SUBSETS of one item set, not two aligned
    copies - a question_id with zero flipped items simply wouldn't exist
    in the "flipped" half at all, so paired_cluster_bootstrap's shared-
    group-universe requirement isn't even satisfiable here. A plain
    cluster_bootstrap() with a stat_fn that computes both group means
    INSIDE one resampled replicate is the correct tool for a two-disjoint-
    subgroup comparison.

Writes results/rq3a_table_{model_slug}.csv (one row per signal) and
results/figures/rq3a_confidence_gap_{model_slug}.png (src/plots.py's
plot_rq3a_confidence_gap, a forest plot of all four signals' gaps against
a zero reference line - same shape as task 3.4's correlation figure, both
now built on the shared _forest_plot() helper). Prints the flip rate plus
the RQ3a money sentence (conf_verb's gap - the project's headline signal
throughout, same convention as RQ1's headline and task 3.3's figure).

--- RQ3b (task 4.4) ---

Reads results/items.parquet filtered to (condition in {clean, verbose},
prompt_variant == P1, human_label not null), via load_rq3b_items(). The
`verbose` run only ever collects P1 (D19), so no prompt_variant filter
ambiguity exists on that side; the human_label filter matches RQ1's
1836-item population exactly, because human_label depends only on
item_id (human votes), and clean/verbose share the identical 1904-item
population 1:1 (task 4.2's own verification) - so the same 68 items are
excluded from both sides, and the two resulting DataFrames are the SAME
1836 items, just scored under a different `condition`. That is what makes
this a genuinely PAIRED comparison, unlike RQ3a's flipped/unflipped split:
paired_cluster_bootstrap's shared-question_id-universe requirement holds
here by construction.

`conf_sc` is dropped from this comparison (D21) - self-consistency
sampling only happens for clean/P1, so `verbose` has no conf_sc values at
all, not a smaller sample of them. `conf_verb`, `conf_lp`, `conf_bpe` are
unaffected (each needs only the greedy call at both orders, which
`verbose` does collect).

For each surviving signal: paired cluster-bootstrap CI (question_id,
invariant 2) on stat_fn(verbose) - stat_fn(clean), for three statistics -
ECE, accuracy, AUROC(uncertainty -> error) - all scored against
judge_verdict/correct (D7's single-pass, deployed definition - the same
choice RQ2 made, not verdict_bidir, which is RQ1's own comparison). A
positive ECE delta means the judge is MORE miscalibrated under verbose; a
negative accuracy or AUROC delta means verbosity makes the judge worse or
its uncertainty signal less informative, respectively.

Writes results/rq3_table_{model_slug}.csv (one row per surviving signal).
Prints the RQ3b money sentence (conf_verb's paired ECE delta).
"""

import argparse

import pandas as pd

from analysis.rq1 import SIGNALS, load_rq1_items
from src.boot import cluster_bootstrap, paired_cluster_bootstrap
from src.config import Config
from src.metrics import auroc_error, ece
from src.plots import plot_rq3a_confidence_gap

RQ3B_SIGNALS = ["conf_verb", "conf_lp", "conf_bpe"]  # conf_sc dropped, D21


def compute_flip_rate(items: pd.DataFrame, seed: int) -> dict:
    """Cluster-bootstrap CI on the flip rate: mean(flipped) over the
    population.
    """
    def _flip_rate(df: pd.DataFrame) -> float:
        return df["flipped"].astype(float).mean()

    point, ci_low, ci_high = cluster_bootstrap(items, _flip_rate, "question_id", n=2000, seed=seed)
    return {"flip_rate": point, "flip_rate_ci_low": ci_low, "flip_rate_ci_high": ci_high}


def compute_confidence_gap(items: pd.DataFrame, signal: str, seed: int) -> dict:
    """Cluster-bootstrap CI on mean(signal | flipped) - mean(signal | not
    flipped): does the judge's stated confidence drop on items where the
    presentation order actually changed its verdict? If the judge's
    uncertainty signal is tracking its own bias-induced errors, this gap
    should be negative (lower confidence when flipped) with a CI that
    excludes 0 - if it doesn't, the signal isn't picking up on position
    bias at all.

    Args:
        items: load_rq1_items()'s output, with `flipped` cast to bool.
        signal: one of SIGNALS - the confidence column name.
        seed: config.seed.

    Returns:
        mean_conf_flipped/mean_conf_unflipped (real-data point values, not
        bootstrapped) plus gap_flipped_minus_unflipped/gap_ci_low/
        gap_ci_high (the bootstrapped difference and its CI).
    """
    def _gap(df: pd.DataFrame) -> float:
        flipped_mean = df.loc[df["flipped"], signal].mean()
        unflipped_mean = df.loc[~df["flipped"], signal].mean()
        return flipped_mean - unflipped_mean

    point, ci_low, ci_high = cluster_bootstrap(items, _gap, "question_id", n=2000, seed=seed)

    return {
        "mean_conf_flipped": items.loc[items["flipped"], signal].mean(),
        "mean_conf_unflipped": items.loc[~items["flipped"], signal].mean(),
        "gap_flipped_minus_unflipped": point,
        "gap_ci_low": ci_low,
        "gap_ci_high": ci_high,
    }


def load_rq3b_items(items_parquet: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """results/items.parquet -> (clean_items, verbose_items), the paired
    population RQ3b is allowed to touch.

    Both sides filtered to prompt_variant == P1 and human_label.notna()
    (same reasoning as load_rq1_items() - correct/correct_bidir are None
    for the ~68 items with no clear human majority). Asserts the two
    resulting item_id sets are identical - if a future run ever breaks
    that (e.g. a partial verbose re-run), this should fail loudly here
    rather than silently unpairing the bootstrap.

    Args:
        items_parquet: config.paths.items_parquet.

    Returns:
        (clean_items, verbose_items), each indexed by item_id-unique rows.
    """
    items = pd.read_parquet(items_parquet)
    items = items[items["prompt_variant"] == "P1"]
    items = items[items["human_label"].notna()]

    clean_items = items[items["condition"] == "clean"]
    verbose_items = items[items["condition"] == "verbose"]

    clean_ids = set(clean_items["item_id"])
    verbose_ids = set(verbose_items["item_id"])
    if clean_ids != verbose_ids:
        raise ValueError(
            "clean/P1 and verbose/P1 item populations differ - RQ3b's paired "
            f"comparison requires the same items on both sides. Only in clean: "
            f"{clean_ids - verbose_ids}; only in verbose: {verbose_ids - clean_ids}"
        )

    return clean_items, verbose_items


def compute_signal_rq3b_metrics(
    clean_items: pd.DataFrame, verbose_items: pd.DataFrame, signal: str, correct_col: str, n_bins: int, seed: int
) -> dict:
    """Paired cluster-bootstrap CIs on ECE, accuracy, and AUROC's change
    from `clean` to `verbose`, for one confidence signal.

    Args:
        clean_items, verbose_items: load_rq3b_items()'s output.
        signal: one of RQ3B_SIGNALS.
        correct_col: "correct" (judge_verdict's correctness column - D7's
            single-pass definition, matching RQ2's own choice).
        n_bins: config.n_bins.
        seed: config.seed.

    Returns:
        For each of ece/accuracy/auroc: the real point value under clean
        and under verbose, plus delta_{name}_verbose_minus_clean and its
        CI (paired_cluster_bootstrap's stat_fn(verbose) - stat_fn(clean)).
    """
    def _ece(df: pd.DataFrame) -> float:
        value, _ = ece(df[signal].to_numpy(), df[correct_col].to_numpy(), n_bins)
        return value

    def _accuracy(df: pd.DataFrame) -> float:
        return df[correct_col].astype(float).mean()

    def _auroc(df: pd.DataFrame) -> float:
        uncertainty = 1 - df[signal].to_numpy(dtype=float)
        return auroc_error(uncertainty, df[correct_col].to_numpy())

    result = {}
    for name, fn in [("ece", _ece), ("accuracy", _accuracy), ("auroc", _auroc)]:
        diff, ci_low, ci_high = paired_cluster_bootstrap(
            verbose_items, clean_items, fn, "question_id", n=2000, seed=seed
        )
        result[f"{name}_clean"] = fn(clean_items)
        result[f"{name}_verbose"] = fn(verbose_items)
        result[f"delta_{name}_verbose_minus_clean"] = diff
        result[f"delta_{name}_ci_low"] = ci_low
        result[f"delta_{name}_ci_high"] = ci_high

    return result


def run_rq3b(config: Config) -> pd.DataFrame:
    """RQ3b: paired ECE/accuracy/AUROC deltas, clean(P1) -> verbose(P1),
    for the three signals that survive D21's conf_sc drop. Writes
    results/rq3_table_{model_slug}.csv and prints the money sentence.
    """
    clean_items, verbose_items = load_rq3b_items(config.paths.items_parquet)
    print(f"RQ3b population: N={len(clean_items)} paired items (clean/P1 vs verbose/P1, human_label present)")

    rows = []
    for signal in RQ3B_SIGNALS:
        metrics = compute_signal_rq3b_metrics(clean_items, verbose_items, signal, "correct", config.n_bins, config.seed)
        rows.append({"signal": signal, **metrics})

    table = pd.DataFrame.from_records(rows)
    table_path = f"results/rq3_table_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    headline = table[table["signal"] == "conf_verb"].iloc[0]
    print(
        "conf_verb ECE gap (verbose - clean): "
        f"{headline['delta_ece_verbose_minus_clean']:.4f} "
        f"[{headline['delta_ece_ci_low']:.4f}, {headline['delta_ece_ci_high']:.4f}]"
    )

    return table


def run_rq3a(config: Config) -> pd.DataFrame:
    """RQ3a: flip rate plus confidence-on-flipped-vs-unflipped, for all
    four original signals. Writes results/rq3a_table_{model_slug}.csv and
    the forest-plot figure, and prints the money sentence.
    """
    items = load_rq1_items(config.paths.items_parquet)
    items = items.copy()
    items["flipped"] = items["flipped"].astype(bool)

    flip_rate_result = compute_flip_rate(items, config.seed)
    print(
        f"Flip rate (AB vs BA canonical verdict, (clean, P1), N={len(items)}): "
        f"{flip_rate_result['flip_rate']:.4f} "
        f"[{flip_rate_result['flip_rate_ci_low']:.4f}, {flip_rate_result['flip_rate_ci_high']:.4f}]"
    )

    rows = []
    for signal in SIGNALS:
        gap = compute_confidence_gap(items, signal, config.seed)
        rows.append({"signal": signal, **gap})

    table = pd.DataFrame.from_records(rows)
    table_path = f"results/rq3a_table_{config.model_slug}.csv"
    table.to_csv(table_path, index=False)
    print(f"Wrote {len(table)} rows to {table_path}")

    plot_rq3a_confidence_gap(
        signals=table["signal"].tolist(),
        gap=table["gap_flipped_minus_unflipped"].to_numpy(),
        ci_low=table["gap_ci_low"].to_numpy(),
        ci_high=table["gap_ci_high"].to_numpy(),
        model_slug=config.model_slug,
    )

    headline = table[table["signal"] == "conf_verb"].iloc[0]
    print(
        "conf_verb gap (flipped - unflipped confidence): "
        f"{headline['gap_flipped_minus_unflipped']:.4f} "
        f"[{headline['gap_ci_low']:.4f}, {headline['gap_ci_high']:.4f}]"
    )

    return table


def main(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    run_rq3a(config)
    print()
    run_rq3b(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
