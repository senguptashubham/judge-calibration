"""Tests for src/judge_kev.py's locally-testable logic: the flat (D27)
call schedule, checkpoint dedup/resumability (invariant 9), and state/
payload construction. Deliberately does not import transformers or
requests-against-a-real-server - none of this logic touches either (see
judge_kev.py's own module docstring, same "only touches the external thing
inside one boundary function" split as tests/test_judge.py).
"""

import pandas as pd
import pytest

from src.data import item_id
from src.judge_kev import (
    KevCallSpec,
    KevConfig,
    append_checkpoint,
    build_payload,
    build_state,
    call_schedule,
    checkpoint_key,
    load_completed_keys,
    pending_calls,
)


@pytest.fixture
def config():
    return KevConfig.from_yaml("configs/run_kev.yaml")


def test_call_schedule_is_flat_two_conditions_times_two_orders(config):
    # D27's schedule has no prompt_variant axis and no sampling - just
    # (condition, order), 4 calls/item for run_kev.yaml's [clean, verbose].
    specs = call_schedule(config)
    assert len(specs) == 4
    assert {(s.condition, s.order) for s in specs} == {
        ("clean", "AB"),
        ("clean", "BA"),
        ("verbose", "AB"),
        ("verbose", "BA"),
    }


def test_checkpoint_key_deterministic_and_differs_on_any_field():
    base = checkpoint_key("abc", "clean", "AB")
    assert base == checkpoint_key("abc", "clean", "AB")
    assert base != checkpoint_key("xyz", "clean", "AB")
    assert base != checkpoint_key("abc", "verbose", "AB")
    assert base != checkpoint_key("abc", "clean", "BA")


def test_append_and_load_completed_keys_round_trip(tmp_path):
    checkpoint_path = tmp_path / "kev.jsonl"
    append_checkpoint(checkpoint_path, {"item_id": "abc", "condition": "clean", "order": "AB"})
    assert load_completed_keys(checkpoint_path) == {checkpoint_key("abc", "clean", "AB")}


def _items_df():
    return pd.DataFrame(
        [
            {"question_id": 1, "model_a": "m1", "model_b": "m2", "turn": 1},
            {"question_id": 2, "model_a": "m3", "model_b": "m4", "turn": 1},
        ]
    )


def test_pending_calls_skips_completed_keys_running_twice_does_not_duplicate():
    # CLAUDE.md invariant 9, directly: simulate "run once, then run again
    # with the same checkpoint" and confirm nothing repeats.
    specs = [KevCallSpec("clean", "AB"), KevCallSpec("clean", "BA")]
    items_df = _items_df()

    first_pass = pending_calls(items_df, specs, completed=set())
    assert len(first_pass) == 4  # 2 items x 2 specs

    completed_after_first_run = {
        checkpoint_key(item, spec.condition, spec.order) for item, _, spec in first_pass
    }
    assert pending_calls(items_df, specs, completed_after_first_run) == []


def test_pending_calls_resumes_only_the_missing_half_after_a_partial_kill():
    specs = [KevCallSpec("clean", "AB"), KevCallSpec("clean", "BA")]
    items_df = _items_df()
    real_item_1 = item_id(1, "m1", "m2", 1)
    completed = {checkpoint_key(real_item_1, "clean", "AB")}

    remaining = pending_calls(items_df, specs, completed)
    remaining_keys = {checkpoint_key(item, spec.condition, spec.order) for item, _, spec in remaining}
    assert checkpoint_key(real_item_1, "clean", "AB") not in remaining_keys
    assert checkpoint_key(real_item_1, "clean", "BA") in remaining_keys


# --- build_state --------------------------------------------------------


def _conversations():
    conv_a = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello there. Nice to meet you."}]
    conv_b = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hey. Good day to you."}]
    return conv_a, conv_b


def test_build_state_verbose_is_padded_and_longer_than_clean():
    conv_a, conv_b = _conversations()
    clean_state = build_state("clean", "AB", conv_a, conv_b, turn=1)
    verbose_state = build_state("verbose", "AB", conv_a, conv_b, turn=1)
    assert verbose_state != clean_state
    assert len(verbose_state) > len(clean_state)


def test_build_state_order_swap_changes_which_content_is_labeled_a():
    conv_a, conv_b = _conversations()
    ab_state = build_state("clean", "AB", conv_a, conv_b, turn=1)
    ba_state = build_state("clean", "BA", conv_a, conv_b, turn=1)
    assert ab_state != ba_state

    # The two rendered blocks are separated by exactly one blank line.
    ab_first_block, ab_second_block = ab_state.split("\n\n")
    ba_first_block, ba_second_block = ba_state.split("\n\n")

    # AB (identity mapping): conv_a's content is labeled Assistant A (first block).
    assert "Nice to meet you" in ab_first_block
    assert "Nice to meet you" not in ab_second_block

    # BA (swapped): conv_a's content is now labeled Assistant B (second block).
    assert "Nice to meet you" in ba_second_block
    assert "Nice to meet you" not in ba_first_block


def test_build_payload_shape_is_a_two_option_choice_question():
    payload = build_payload("some state text")
    assert payload["state"] == "some state text"
    assert payload["questions"]["verdict"]["type"] == "choice"
    assert set(payload["questions"]["verdict"]["criteria"]) == {"A", "B"}
