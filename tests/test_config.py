"""Tests for src/config.py. See TASKS.md task 0.2."""

import pytest

from src.config import Config, Paths


def _paths() -> Paths:
    return Paths(
        results_dir="results",
        runs_dir="runs",
        figures_dir="results/figures",
        calls_parquet="results/calls.parquet",
        items_parquet="results/items.parquet",
        items_labels_parquet="results/items_labels.parquet",
    )


def _config(**overrides) -> Config:
    kwargs = dict(
        judge_model="Qwen/Qwen2.5-7B-Instruct",
        dataset="lmsys/mt_bench_human_judgments",
        tie_policy="drop_ties",
        temperature_canonical=0.0,
        temperature_sc=0.7,
        k_sc=4,
        max_tokens=512,
        logprobs=20,
        conditions=["clean", "verbose"],
        prompt_variants=["P1", "P2", "P3"],
        seed=1234,
        n_bins=10,
        paths=_paths(),
    )
    kwargs.update(overrides)
    return Config(**kwargs)


def test_from_yaml_round_trips(tmp_path):
    original = _config()
    yaml_path = tmp_path / "run.yaml"
    original.to_yaml(yaml_path)
    loaded = Config.from_yaml(yaml_path)
    assert loaded == original


def test_real_config_is_valid():
    # configs/run.yaml itself must load and satisfy the invariants below.
    config = Config.from_yaml("configs/run.yaml")
    assert config.temperature_sc > 0
    assert "swap" not in config.conditions


def test_temperature_sc_must_be_positive():
    with pytest.raises(ValueError):
        _config(temperature_sc=0.0)


def test_swap_condition_is_rejected():
    with pytest.raises(ValueError):
        _config(conditions=["clean", "swap"])


def test_attribution_condition_is_rejected():
    with pytest.raises(ValueError):
        _config(conditions=["clean", "verbose", "attribution"])


def test_prompt_variants_must_include_p1():
    with pytest.raises(ValueError):
        _config(prompt_variants=["P2", "P3"])
