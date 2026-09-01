"""MT-Bench votes -> item table: tie policy, vote aggregation, human-human
Cohen's kappa. See TASKS.md tasks 0.7-0.9.
"""
import argparse
from typing import cast

import pandas as pd
from datasets import load_dataset

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


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", required=True)
  args = parser.parse_args()

  config = Config.from_yaml(args.config)
  votes = load_votes(config.dataset)
  print(votes.head())