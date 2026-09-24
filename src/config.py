"""Typed config for the primary judge's entrypoints. Nothing else in src/
reads configs/*.yaml directly or hardcodes a model name, path, threshold,
or k value (CLAUDE.md §5). The RQ6/RQ7 harnesses have their own smaller
config classes (judge_kev.KevConfig, judge_autoj.AutojConfig).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass(frozen=True)
class Paths:
    results_dir: str
    runs_dir: str
    calls_parquet: str
    items_parquet: str
    items_labels_parquet: str


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
    prompt_variants: list[str]
    seed: int
    n_bins: int
    paths: Paths

    def __post_init__(self) -> None:
        if self.temperature_sc <= 0:
            raise ValueError(
                "temperature_sc must be > 0 - at 0 every k_sc draw would be "
                "identical and conf_sc would be a dead constant (DECISIONS.md D6)."
            )
        if "swap" in self.conditions:
            raise ValueError(
                "'swap' is not a valid condition - order (AB/BA) is an "
                "orthogonal axis collected within every condition, not a "
                "condition itself (DECISIONS.md D5)."
            )
        if "attribution" in self.conditions:
            raise ValueError(
                "'attribution' was dropped as a condition on 31 Aug 2026 "
                "(DECISIONS.md D18) - it is not coming back via config."
            )
        if "P1" not in self.prompt_variants:
            raise ValueError(
                "prompt_variants must include 'P1' - it is the primary "
                "variant RQ1-RQ4 use alone (DECISIONS.md D19, D20)."
            )

    @property
    def model_slug(self) -> str:
        """Filesystem-safe tag for `judge_model` that suffixes every
        model-dependent output (D26): "Qwen/Qwen2.5-7B-Instruct" ->
        "qwen2.5_7b_instruct". items_labels.parquet is the one exception -
        it depends on human votes only.
        """
        name = self.judge_model.split("/")[-1]
        return name.lower().replace("-", "_")

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
