"""MT-Bench votes -> item table: tie policy, vote aggregation, and
human-human Cohen's kappa.
"""
import argparse
import hashlib
from pathlib import Path
from typing import cast

import pandas as pd
from datasets import load_dataset
from src import metrics
from src.config import Config


def item_id(question_id: int, model_a: str, model_b: str, turn: int) -> str:
    """Stable hash of an item's identity, the join key for calls and items
    tables. Defined once, here, so every table hashes identically. 16 hex
    chars (64 bits) - a collision across ~2,400 items is not a concern.
    """
    raw = f"{question_id}|{model_a}|{model_b}|{turn}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_votes(dataset: str) -> pd.DataFrame:
    ds_human = load_dataset(dataset, split="human")
    df_human = cast(pd.DataFrame, ds_human.to_pandas())
    if len(df_human) != 3355:
        raise ValueError(
            f"expected 3355 rows in mt_bench_human_judgments 'human' split, got {len(df_human)} - "
            "the dataset shape changed upstream."
        )
    return df_human


def _response_length(conversation: list[dict], turn: int) -> int:
    """Character length of the assistant response being judged at `turn`
    (index 1 for turn=1, 3 for turn=2) - the same message prompts.py renders
    last, not the whole conversation history.
    """
    index = 1 if turn == 1 else 3
    return len(conversation[index]["content"])


def build_items(votes: pd.DataFrame, tie_policy: str) -> pd.DataFrame:
    """One row per vote -> one row per item, keyed by
    (question_id, model_a, model_b, turn).

    Tie policy (D1, compared against three alternatives in
    PREREGISTRATION.md): "mark_tie_strict". A tie ballot is different
    information from a weak preference, so ties are a categorical property
    of the item (`is_tie`), never smoothed into frac_prefer_a. "Strict"
    means tie ballots must outnumber BOTH sides (n_tie > n_a and
    n_tie > n_b) - the lenient >= version marked e.g. 2 tie / 2 A / 1 B as
    fully tied, discarding A's 2-to-1 win among decisive voters.

    human_unanimous means no disagreement among the votes that count: every
    raw vote for a tied item, the non-tie votes otherwise.

    Returns one row per item with n_human_votes, frac_prefer_a,
    majority_label, human_unanimous, is_tie, d_human, human_agreed (D16),
    and len_a/len_b (the judged responses' character lengths, used by RQ4's
    Tier B; model-independent, so they live here rather than in items.parquet).
    """
    if tie_policy != "mark_tie_strict":
        raise ValueError(
            f"tie_policy={tie_policy!r} is not supported - D1 decided "
            "'mark_tie_strict' after comparing four candidates; see "
            "PREREGISTRATION.md for the comparison."
        )

    records = []
    group_cols = ["question_id", "model_a", "model_b", "turn"]
    for (question_id, model_a, model_b, turn), group in votes.groupby(group_cols):
        n_a = int((group["winner"] == "model_a").sum())
        n_b = int((group["winner"] == "model_b").sum())
        n_tie = int((group["winner"] == "tie").sum())
        n_human_votes = n_a + n_b + n_tie
        n_unique_raw = group["winner"].nunique()

        is_tie = n_tie > n_a and n_tie > n_b
        if is_tie:
            frac_prefer_a = float("nan")
            human_unanimous = n_unique_raw == 1
        else:
            frac_prefer_a = n_a / (n_a + n_b)
            human_unanimous = (n_a == 0 or n_b == 0)

        if frac_prefer_a == frac_prefer_a:  # not NaN
            majority_label = "A" if frac_prefer_a > 0.5 else "B" if frac_prefer_a < 0.5 else None
            d_human = abs(frac_prefer_a - 0.5)
        else:
            majority_label = None
            d_human = float("nan")

        # Every vote in a group shares the same conversation, so the first
        # row's is the item's.
        len_a = _response_length(group["conversation_a"].iloc[0], int(turn))
        len_b = _response_length(group["conversation_b"].iloc[0], int(turn))

        records.append({
            "item_id": item_id(int(question_id), str(model_a), str(model_b), int(turn)),
            "question_id": question_id,
            "model_a": model_a,
            "model_b": model_b,
            "turn": turn,
            "n_human_votes": n_human_votes,
            "frac_prefer_a": frac_prefer_a,
            "majority_label": majority_label,
            "human_unanimous": human_unanimous,
            "is_tie": is_tie,
            "d_human": d_human,
            "human_agreed": bool(human_unanimous) and (n_human_votes >= 2),
            "len_a": len_a,
            "len_b": len_b,
        })

    return pd.DataFrame.from_records(records)


def summarize_items(items: pd.DataFrame) -> dict:
    """Gate 0's summary counts, as a dict."""
    return {
        "n_total": len(items),
        "n_non_tie": int((~items["is_tie"]).sum()),
        "n_ge2_votes": int((items["n_human_votes"] >= 2).sum()),
        "n_ge3_votes": int((items["n_human_votes"] >= 3).sum()),
        "n_unanimous": int(items["human_unanimous"].sum()),
        # Contested (D9): enough votes to show disagreement, and does show it.
        "n_contested": int(((items["n_human_votes"] >= 2) & (~items["human_unanimous"])).sum()),
    }


def human_human_kappa(votes: pd.DataFrame) -> tuple[float, int]:
    """Human-human Cohen's kappa - the ceiling on everything downstream.
    One pair of non-tie votes per item with >= 2 of them (not all pairwise
    combinations), so every pair is an independent observation.

    Votes are sorted by `judge` before taking the first and last, so the
    pairing is deterministic regardless of the source's row order. No judge
    votes twice on one item in this dataset, so a pair is never one judge
    agreeing with themselves.

    Returns (kappa, number of item pairs used).
    """
    non_tie_votes = votes[votes["winner"] != "tie"]
    group_cols = ["question_id", "model_a", "model_b", "turn"]
    rater1, rater2 = [], []
    for (question_id, model_a, model_b, turn), group in non_tie_votes.groupby(group_cols):
        group = group.sort_values("judge")
        winners = group["winner"]
        if len(winners) >= 2:
            rater1.append(winners.iloc[0])
            rater2.append(winners.iloc[-1])
    return metrics.cohens_kappa(rater1, rater2), len(rater1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    votes = load_votes(config.dataset)
    items = build_items(votes, tie_policy=config.tie_policy)
    kappa, N = human_human_kappa(votes)
    print(f"Got {N} human-human comparisons with {kappa} score")

    Path(config.paths.items_labels_parquet).parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(config.paths.items_labels_parquet)

    summary = summarize_items(items)
    print(f"Wrote {len(items)} items to {config.paths.items_labels_parquet}")
    for key, value in summary.items():
        print(f"  {key}: {value}")
