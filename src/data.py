"""MT-Bench votes -> item table: tie policy, vote aggregation, human-human
Cohen's kappa. See TASKS.md tasks 0.7-0.9.
"""
import argparse
from pathlib import Path
from typing import cast

import pandas as pd
from datasets import load_dataset
from src import metrics
from src.config import Config


def load_votes(dataset: str) -> pd.DataFrame:
  ds_human = load_dataset(dataset, split="human")
  df_human = cast(pd.DataFrame, ds_human.to_pandas())
  if len(df_human) != 3355:
    raise ValueError(
      f"expected 3355 rows in mt_bench_human_judgments 'human' split, got {len(df_human)} - "
      "the dataset shape changed upstream."
    )
  return df_human


def build_items(votes: pd.DataFrame, tie_policy: str) -> pd.DataFrame:
  """Aggregate one-row-per-vote to one-row-per-item, keyed by
  (question_id, model_a, model_b, turn).

  tie_policy is decision D1. Four candidates were implemented and compared
  empirically on this exact dataset (drop_ties, count_half, mark_tie_lenient,
  mark_tie_strict) before deciding - the comparison table and full reasoning
  live in PREREGISTRATION.md, not here, since those numbers are a snapshot
  of one run and would drift out of sync with this docstring over time.

  Decided: "mark_tie_strict". A "tie" ballot is different information from
  a weak preference for either side, not a 0.5-strength vote for both - so
  an item's ties are treated as a categorical property of the item
  (is_tie), never smoothed into frac_prefer_a. "Strict" means tie ballots
  must strictly outnumber both sides (n_tie > n_a and n_tie > n_b) to mark
  the item tied - not just tie for the most common outcome - because the
  lenient (>=) version was discarding real signal: e.g. a group of 2 tie /
  2 A / 1 B was being marked fully tied under lenient even though A beat B
  2-to-1 among the voters who actually expressed a preference.

  human_unanimous means "no disagreement among the votes that count" - for
  a tied item that's every raw vote (only true if literally everyone voted
  tie); for a non-tied item it's the non-tie votes only.

  Args:
    votes: load_votes()'s output - one row per vote.
    tie_policy: must be "mark_tie_strict" - the only value D1 decided.
      Still a parameter, not hardcoded, so this stays config-driven
      (CLAUDE.md sec 5) even though only one value is currently valid.

  Returns:
    One row per item, with n_human_votes, frac_prefer_a, majority_label,
    human_unanimous, is_tie, d_human, human_agreed.
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

    records.append({
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
    })

  return pd.DataFrame.from_records(records)


def summarize_items(items: pd.DataFrame) -> dict:
  """The Gate 0 / task 0.8 printed summary, as a dict so it's easy to
  compare across tie_policy runs rather than just eyeballing print output.
  """
  return {
    "n_total": len(items),
    "n_non_tie": int((~items["is_tie"]).sum()),
    "n_ge2_votes": int((items["n_human_votes"] >= 2).sum()),
    "n_ge3_votes": int((items["n_human_votes"] >= 3).sum()),
    "n_unanimous": int(items["human_unanimous"].sum()),
    # Contested (D9): enough votes to show disagreement, and does show it.
    # Independent of is_tie, whose definition varies by tie_policy.
    "n_contested": int(((items["n_human_votes"] >= 2) & (~items["human_unanimous"])).sum()),
  }


def human_human_kappa(votes: pd.DataFrame) -> tuple[float, int]:
  """Human-human Cohen's kappa: pairs up two non-tie votes per qualifying
  item (Option A - one pair per item, not all pairwise combinations, so
  every pair is an independent observation) and feeds the pooled pairs into
  cohens_kappa(). This is the ceiling on everything downstream (task 0.9).

  Items with >2 non-tie votes have more than two judges to choose from,
  so which two matters for reproducibility. Each group is sorted by
  `judge` (a stable identifier, unlike row order from the source data)
  before taking the first and last - deterministic across runs, and
  confirmed empirically that no judge votes twice on the same item in
  this dataset (max group size 5, always 5 distinct judges), so first/last
  after sorting is never degenerate self-agreement.

  Args:
    votes: load_votes()'s output - one row per vote.

  Returns:
    (kappa, N) where N is the number of qualifying items/pairs used, not
    the number of underlying votes.
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