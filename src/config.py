"""Config dataclass and YAML loader.

Central, typed config for every entrypoint (`python -m src.data`,
`src.judge`, `src.signals`, `src.predictor`, ...). Nothing else in `src/`
should read `configs/*.yaml` directly or hardcode a model name, path,
threshold, or k value - everything tunable comes through this dataclass
(CLAUDE.md sec 5).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass(frozen=True)
class Paths:
    results_dir: str
    runs_dir: str
    figures_dir: str
    calls_parquet: str
    items_parquet: str


@dataclasses.dataclass(frozen=True)
class Config:
    judge_model: str
    dataset: str
    tie_policy: str
    temperature_canonical: float
    temperature_sc: float
    k_sc: int
    max_tokens: int
    logprobs: int
    conditions: list[str]
    seed: int
    n_bins: int
    paths: Paths

    def __post_init__(self) -> None:
        # A single `temperature` key (or temperature_sc == 0) would make all
        # k_sc draws identical and conf_sc a dead constant - discovered only
        # in the W5 Tier A ablation if left unchecked. See DECISIONS.md D6.
        if self.temperature_sc <= 0:
            raise ValueError(
                "temperature_sc must be > 0 - at 0 every k_sc draw would be "
                "identical and conf_sc would be a dead constant (DECISIONS.md D6)."
            )
        # 'swap' double-named the order axis as a condition. See DECISIONS.md D5.
        if "swap" in self.conditions:
            raise ValueError(
                "'swap' is not a valid condition - order (AB/BA) is an "
                "orthogonal axis collected within every condition, not a "
                "condition itself (DECISIONS.md D5)."
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = dict(yaml.safe_load(f))
        raw["paths"] = Paths(**raw["paths"])
        return cls(**raw)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
