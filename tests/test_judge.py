"""Tests for src/judge.py's locally-testable logic: the call schedule (D19),
checkpoint dedup/resumability (invariant 9), and the per-call logprobs file
path/writer (D4, amended 4 Sep 2026 to 100% coverage). Deliberately does not
import vllm - none of this logic touches it (see judge.py's own module
docstring). See TASKS.md task 1.4.
"""

import gzip
import json

import pandas as pd
import pytest

from src.config import Config
from src.data import item_id
from src.judge import (
    CallSpec,
    append_checkpoint,
    call_schedule,
    checkpoint_key,
    load_completed_keys,
    logprobs_path,
    pending_calls,
    write_logprobs,
)


@pytest.fixture
def config():
    return Config.from_yaml("configs/run.yaml")


def test_call_schedule_total_is_12_per_item(config):
    # DECISIONS.md D19: 12 calls/item (2 + 4 + 4 + 2).
    assert len(call_schedule(config)) == 12


def test_call_schedule_sample_idx_only_nonzero_for_clean_p1(config):
    # TASKS.md task 1.4's own required assertion.
    for spec in call_schedule(config):
        if spec.sample_idx > 0:
            assert (spec.condition, spec.prompt_variant) == ("clean", "P1")


def test_call_schedule_p2_p3_are_clean_only(config):
    for spec in call_schedule(config):
        if spec.prompt_variant in ("P2", "P3"):
            assert spec.condition == "clean"


def test_call_schedule_verbose_is_p1_only_both_orders(config):
    verbose_specs = [s for s in call_schedule(config) if s.condition == "verbose"]
    assert {s.order for s in verbose_specs} == {"AB", "BA"}
    assert all(s.prompt_variant == "P1" and s.sample_idx == 0 for s in verbose_specs)


def test_checkpoint_key_deterministic():
    assert checkpoint_key("abc", "clean", "P1", "AB", 0) == checkpoint_key("abc", "clean", "P1", "AB", 0)


def test_checkpoint_key_differs_on_any_field():
    base = checkpoint_key("abc", "clean", "P1", "AB", 0)
    assert base != checkpoint_key("xyz", "clean", "P1", "AB", 0)
    assert base != checkpoint_key("abc", "verbose", "P1", "AB", 0)
    assert base != checkpoint_key("abc", "clean", "P2", "AB", 0)
    assert base != checkpoint_key("abc", "clean", "P1", "BA", 0)
    assert base != checkpoint_key("abc", "clean", "P1", "AB", 1)


def test_load_completed_keys_empty_when_file_missing(tmp_path):
    assert load_completed_keys(tmp_path / "does_not_exist.jsonl") == set()


def test_append_and_load_completed_keys_round_trip(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    row = {
        "item_id": "abc",
        "condition": "clean",
        "prompt_variant": "P1",
        "order": "AB",
        "sample_idx": 0,
    }
    append_checkpoint(checkpoint_path, row)
    completed = load_completed_keys(checkpoint_path)
    assert completed == {checkpoint_key("abc", "clean", "P1", "AB", 0)}


def test_append_checkpoint_is_additive_not_overwriting(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    append_checkpoint(checkpoint_path, {"item_id": "a", "condition": "clean", "prompt_variant": "P1", "order": "AB", "sample_idx": 0})
    append_checkpoint(checkpoint_path, {"item_id": "b", "condition": "clean", "prompt_variant": "P1", "order": "AB", "sample_idx": 0})
    completed = load_completed_keys(checkpoint_path)
    assert len(completed) == 2


def _items_df():
    return pd.DataFrame(
        [
            {"question_id": 1, "model_a": "m1", "model_b": "m2", "turn": 1},
            {"question_id": 2, "model_a": "m3", "model_b": "m4", "turn": 1},
        ]
    )


def test_pending_calls_is_full_cross_join_when_nothing_completed():
    specs = [CallSpec("clean", "P1", "AB", 0), CallSpec("clean", "P1", "BA", 0)]
    pending = pending_calls(_items_df(), specs, completed=set())
    assert len(pending) == 2 * len(specs)  # 2 items x 2 specs


def test_pending_calls_skips_completed_keys_running_twice_does_not_duplicate():
    # CLAUDE.md invariant 9 / task 1.4's DoD, directly: simulate "run once,
    # then run again with the same checkpoint" and confirm nothing repeats.
    specs = [CallSpec("clean", "P1", "AB", 0), CallSpec("clean", "P1", "BA", 0)]
    items_df = _items_df()

    first_pass = pending_calls(items_df, specs, completed=set())
    assert len(first_pass) == 4

    # Simulate having completed every call from the first pass.
    completed_after_first_run = {
        checkpoint_key(item, spec.condition, spec.prompt_variant, spec.order, spec.sample_idx)
        for item, _, spec in first_pass
    }
    second_pass = pending_calls(items_df, specs, completed_after_first_run)
    assert second_pass == []


def test_pending_calls_resumes_only_the_missing_half_after_a_partial_kill():
    # "Killing mid-run and restarting loses at most one batch" - simulate
    # completing only AB, confirm only BA remains pending.
    specs = [CallSpec("clean", "P1", "AB", 0), CallSpec("clean", "P1", "BA", 0)]
    items_df = _items_df()

    # pending_calls recomputes item ids itself from question_id/model_a/model_b/turn,
    # so build `completed` using the same ids it will actually produce.
    real_item_1 = item_id(1, "m1", "m2", 1)
    completed = {checkpoint_key(real_item_1, "clean", "P1", "AB", 0)}

    remaining = pending_calls(items_df, specs, completed)
    remaining_keys = {
        checkpoint_key(item, spec.condition, spec.prompt_variant, spec.order, spec.sample_idx)
        for item, _, spec in remaining
    }
    assert checkpoint_key(real_item_1, "clean", "P1", "AB", 0) not in remaining_keys
    assert checkpoint_key(real_item_1, "clean", "P1", "BA", 0) in remaining_keys


def test_logprobs_path_is_deterministic_and_unique_per_call():
    p1 = logprobs_path("runs", "item1", "clean", "P1", "AB", 0)
    p2 = logprobs_path("runs", "item1", "clean", "P1", "AB", 0)
    assert p1 == p2
    p3 = logprobs_path("runs", "item1", "clean", "P1", "BA", 0)
    assert p1 != p3


def test_logprobs_path_lives_under_logprobs_dir_not_the_old_sample_name():
    # D4, amended 4 Sep 2026: directory renamed logprobs_sample/ -> logprobs/
    # now that coverage is 100%, not a 10% sample.
    path = logprobs_path("runs", "item1", "clean", "P1", "AB", 0)
    assert path.parent.name == "logprobs"


class _FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


def test_write_logprobs_round_trips_through_gzip(tmp_path):
    path = tmp_path / "logprobs" / "some_call.jsonl.gz"
    per_token_logprobs = [
        {101: _FakeLogprob(-0.1), 202: _FakeLogprob(-2.3)},
        {303: _FakeLogprob(-0.05)},
    ]
    write_logprobs(path, per_token_logprobs)

    with gzip.open(path, "rt", encoding="utf-8") as f:
        record = json.loads(f.readline())

    assert record["token_logprobs"] == [
        {"101": -0.1, "202": -2.3},
        {"303": -0.05},
    ]
