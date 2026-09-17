"""RQ1: is the judge's stated confidence calibrated? See TASKS.md task 2.6.

Plain script, not a notebook (CLAUDE.md sec 5's "no notebooks in src/"
extends to analysis orchestration too - this runs as
`python -m analysis.rq1 --config configs/run.yaml`, one deterministic
command, matching every src/ module's own CLI pattern and REPRODUCE.md's
eventual "exact commands" requirement, task 7.1).

Reads results/items.parquet, filtered to (clean, P1) only (CLAUDE.md
invariant 14 - the other 3/4 of items.parquet's rows are P2/P3 and would
silently triple the sample size otherwise). For each of the four ORIGINAL
confidence signals - conf_verb, conf_lp, conf_sc, conf_bpe (conf_ens is
separate, task 3.2b, not part of RQ1) - computes the full metric battery
against BOTH verdict definitions (D7): judge_verdict/correct (the
deployed single-pass case) and verdict_bidir/correct_bidir (order-
averaged). Every metric ships with a cluster-bootstrap CI (grouped on
question_id, invariant 2) - never a bare point estimate.

Writes results/rq1_table.csv and results/figures/reliability_{signal}.png
x4 (src/plots.py's plot_reliability_diagram, one figure per signal,
judge_verdict and verdict_bidir overlaid on each).
"""

import argparse

import pandas as pd

from src.boot import cluster_bootstrap, paired_cluster_bootstrap
from src.config import Config
from src.metrics import auroc_error, brier, brier_decomposition, cohens_kappa, ece, mce, overconfidence_gap
from src.plots import plot_reliability_diagram

SIGNALS = ["conf_verb", "conf_lp", "conf_sc", "conf_bpe"]
VERDICT_DEFINITIONS = [
    ("judge_verdict", "correct"),
    ("verdict_bidir", "correct_bidir"),
]


def load_rq1_items(items_parquet: str) -> pd.DataFrame:
    """results/items.parquet -> the exact slice RQ1 is allowed to touch.

    Two filters, both mandatory:
      1. condition == "clean" AND prompt_variant == "P1" (invariant 14) -
         items.parquet's grain is (item_id, condition, prompt_variant);
         RQ1 only ever looks at one of those combinations.
      2. human_label.notna() - correct/correct_bidir are already None for
         the ~68 items with no clear human majority (a genuine 50/50 non-
         tie split, not a data bug - see data.py's build_items()); metrics
         functions expect a clean boolean array, not None mixed in.
    """
    items = pd.read_parquet(items_parquet)
    items = items[(items["condition"] == "clean") & (items["prompt_variant"] == "P1")]
    items = items[items["human_label"].notna()]
    return items


def _bootstrap_battery(items: pd.DataFrame, metric_fns: dict, seed: int) -> dict:
    """Runs cluster_bootstrap once per (name, stat_fn) pair in `metric_fns`
    and flattens the results into name/name_ci_low/name_ci_high keys.
    Shared by compute_verdict_metrics and compute_signal_metrics so the
    "call cluster_bootstrap, unpack, store three keys" loop exists once,
    not twice.
    """
    result = {}
    for name, fn in metric_fns.items():
        point, ci_low, ci_high = cluster_bootstrap(items, fn, "question_id", n=2000, seed=seed)
        result[name] = point
        result[f"{name}_ci_low"] = ci_low
        result[f"{name}_ci_high"] = ci_high
    return result


def compute_verdict_metrics(items: pd.DataFrame, correct_col: str, verdict_col: str, seed: int) -> dict:
    """The part of RQ1's metric battery that depends only on WHICH verdict
    definition (D7) is being scored, not on any particular confidence
    signal: accuracy, kappa, and the Brier decomposition's `uncertainty`
    term.

    `uncertainty = obar*(1-obar)` (obar = mean of `correct_col`) is
    Murphy's base-rate term - by definition it never reads a confidence
    column at all, so computing it here via the plain formula (rather than
    calling the full brier_decomposition(), which would also redo binning
    work just to extract this one signal-independent piece) avoids paying
    for binning twice for no reason.

    Computed ONCE per verdict definition, not once per (signal, verdict)
    pair - confirmed empirically that all three are bit-identical across
    every signal for a fixed verdict definition (they don't read `signal`
    at all), so folding this into the signal loop would silently redo the
    same n=2000 bootstrap 4x over for values guaranteed not to change.

    Args:
        items: load_rq1_items()'s output (already filtered).
        correct_col: "correct" or "correct_bidir".
        verdict_col: "judge_verdict" or "verdict_bidir" - paired with
            correct_col in VERDICT_DEFINITIONS.
        seed: config.seed.

    Returns:
        accuracy/kappa/uncertainty, each with a _ci_low/_ci_high pair.
    """
    def _accuracy(df: pd.DataFrame) -> float:
        return df[correct_col].astype(float).mean()

    def _kappa(df: pd.DataFrame) -> float:
        return cohens_kappa(df[verdict_col], df["human_label"])

    def _uncertainty(df: pd.DataFrame) -> float:
        obar = df[correct_col].astype(float).mean()
        return obar * (1 - obar)

    return _bootstrap_battery(
        items, {"accuracy": _accuracy, "kappa": _kappa, "uncertainty": _uncertainty}, seed
    )


def compute_signal_metrics(items: pd.DataFrame, signal: str, correct_col: str, n_bins: int, seed: int) -> dict:
    """The part of RQ1's metric battery that DOES depend on a particular
    confidence signal: ece, mce, brier, brier_decomposition's
    reliability/resolution terms (its uncertainty term lives in
    compute_verdict_metrics instead - see that function's docstring for
    why), and overconfidence_gap. Every metric ships with a cluster-
    bootstrap CI over question_id (invariant 2).

    reliability and resolution get independent bootstrap CIs (two
    separate cluster_bootstrap calls, not one) rather than only bootstrapping
    their reconstructed sum - each term is reported on its own in RQ1's
    table, so each needs its own CI, not just the total's.

    Note (REPORT.md, 17 Sep 2026): brier_decomposition()'s reconstruction
    (reliability - resolution + uncertainty) only equals brier() exactly
    for discrete-valued signals (conf_verb, conf_sc) - for continuous,
    quantile-binned ones (conf_lp, conf_bpe) it's a close but real
    approximation ("grouping loss"), not a bug.

    Args:
        items: load_rq1_items()'s output (already filtered).
        signal: one of SIGNALS - the confidence column name.
        correct_col: "correct" or "correct_bidir".
        n_bins: config.n_bins.
        seed: config.seed.

    Returns:
        ece/mce/brier/reliability/resolution/overconfidence_gap, each
        with a _ci_low/_ci_high pair.
    """
    def _ece(df: pd.DataFrame) -> float:
        value, _ = ece(df[signal].to_numpy(), df[correct_col].to_numpy(), n_bins)
        return value

    def _mce(df: pd.DataFrame) -> float:
        value, _ = mce(df[signal].to_numpy(), df[correct_col].to_numpy(), n_bins)
        return value

    def _brier(df: pd.DataFrame) -> float:
        return brier(df[signal].to_numpy(), df[correct_col].to_numpy())

    def _reliability(df: pd.DataFrame) -> float:
        reliability, _, _ = brier_decomposition(df[signal].to_numpy(), df[correct_col].to_numpy(), n_bins)
        return reliability

    def _resolution(df: pd.DataFrame) -> float:
        _, resolution, _ = brier_decomposition(df[signal].to_numpy(), df[correct_col].to_numpy(), n_bins)
        return resolution

    def _overconfidence_gap(df: pd.DataFrame) -> float:
        return overconfidence_gap(df[signal].to_numpy(), df[correct_col].to_numpy())

    return _bootstrap_battery(
        items,
        {
            "ece": _ece,
            "mce": _mce,
            "brier": _brier,
            "reliability": _reliability,
            "resolution": _resolution,
            "overconfidence_gap": _overconfidence_gap,
        },
        seed,
    )


def compute_verdict_gap(items: pd.DataFrame, seed: int) -> dict:
    """Paired cluster-bootstrap CI on accuracy's change from
    `judge_verdict` to `verdict_bidir` (D7).

    Invariant 3: the SAME 1836 items scored two ways is a paired
    comparison, not two independent samples - a single item's verdict
    flipping moves both accuracy numbers at once, so eyeballing whether
    compute_verdict_metrics()'s two separately-bootstrapped accuracy CIs
    overlap is the wrong tool for claiming the gap itself is real.

    Reuses paired_cluster_bootstrap (task 2.4) rather than a new bootstrap
    loop: that function compares stat_fn(df_a) vs stat_fn(df_b) for two
    DIFFERENT row-sets sharing one stat_fn (its usual job - e.g. clean vs
    verbose). Here both "sides" are the SAME rows, just reading a
    different column (`correct` vs `correct_bidir`) - renaming each to a
    shared column name first lets one stat_fn serve both sides, so the
    existing, already-tested function applies unmodified.

    Args:
        items: load_rq1_items()'s output (already filtered).
        seed: config.seed.

    Returns:
        dict with accuracy_gap_bidir_minus_judge, _ci_low, _ci_high. If
        the CI excludes 0, debiasing-by-averaging's accuracy improvement
        is real at this confidence level, not just directionally likely.
    """
    df_bidir = items.rename(columns={"correct_bidir": "score"})
    df_judge = items.rename(columns={"correct": "score"})

    def _accuracy(df: pd.DataFrame) -> float:
        return df["score"].astype(float).mean()

    diff, ci_low, ci_high = paired_cluster_bootstrap(
        df_bidir, df_judge, _accuracy, "question_id", n=2000, seed=seed
    )
    return {
        "accuracy_gap_bidir_minus_judge": diff,
        "accuracy_gap_ci_low": ci_low,
        "accuracy_gap_ci_high": ci_high,
    }


def main(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    items = load_rq1_items(config.paths.items_parquet)

    rows = []
    for verdict_col, correct_col in VERDICT_DEFINITIONS:
        verdict_metrics = compute_verdict_metrics(items, correct_col, verdict_col, config.seed)
        for signal in SIGNALS:
            signal_metrics = compute_signal_metrics(items, signal, correct_col, config.n_bins, config.seed)
            rows.append(
                {
                    "signal": signal,
                    "verdict_definition": verdict_col,
                    **verdict_metrics,
                    **signal_metrics,
                }
            )

    for signal in SIGNALS:
        plot_reliability_diagram(
            items[signal].to_numpy(),
            items["correct"].to_numpy(),
            signal_name=signal,
            n_bins=config.n_bins,
            correct_bidir=items["correct_bidir"].to_numpy(),
        )

    gap = compute_verdict_gap(items, config.seed)
    print(
        "Paired accuracy gap (verdict_bidir - judge_verdict): "
        f"{gap['accuracy_gap_bidir_minus_judge']:.4f} "
        f"[{gap['accuracy_gap_ci_low']:.4f}, {gap['accuracy_gap_ci_high']:.4f}]"
    )

    table = pd.DataFrame.from_records(rows)
    table.to_csv("results/rq1_table.csv", index=False)
    print(f"Wrote {len(table)} rows to results/rq1_table.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
