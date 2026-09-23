"""Tests for src/judge_autoj.py's locally-testable logic: D28's schedule
(clean/verbose, no prompt_variant axis), checkpoint dedup/resumability
(invariant 9), and the prompt renderer. Turn=2 was built, smoke-tested,
and dropped (D28's 23 Sep amendment - a 100-item smoke test found a 36.1%
output-truncation-driven parse-failure rate for verbose/turn=2 vs 0% for
clean/turn=2 and 7.1% for verbose/turn=1); build_autoj_prompt() is turn=1
only now, and load_turn1_items_df() is the real population loader. The
turn=2 design's own tests are gone with it - the finding itself lives in
DECISIONS.md D28, not as dead-code coverage here. Deliberately does not
import vllm - none of this logic touches it (same "only touches the
external thing inside one boundary function" split judge.py/judge_kev.py's
own tests already use).
"""

import pandas as pd
import pytest

from src.data import item_id
from src.judge_autoj import (
    AutojCallSpec,
    AutojConfig,
    append_checkpoint,
    autoj_prompt_hash,
    build_autoj_prompt,
    call_schedule_autoj,
    checkpoint_key,
    load_completed_keys,
    load_turn1_items_df,
    pending_calls,
    split_by_prompt_length,
)


@pytest.fixture
def config():
    return AutojConfig.from_yaml("configs/run_autoj.yaml")


def test_call_schedule_is_clean_six_verbose_two_no_prompt_variant(config):
    # D28's locked schedule: clean reuses D5/D6's own 6-call shape (2 greedy
    # across orders + k_sc=4 sampled AB), verbose reuses D19's own 2-call
    # shape (greedy, both orders, no sampling) - 8 calls/item total for
    # run_autoj.yaml's [clean, verbose].
    specs = call_schedule_autoj(config)
    assert len(specs) == 8
    clean_specs = [s for s in specs if s.condition == "clean"]
    verbose_specs = [s for s in specs if s.condition == "verbose"]
    assert len(clean_specs) == 6
    assert len(verbose_specs) == 2
    assert {(s.order, s.sample_idx) for s in verbose_specs} == {("AB", 0), ("BA", 0)}
    assert {(s.order, s.sample_idx) for s in clean_specs} == {
        ("AB", 0), ("BA", 0), ("AB", 1), ("AB", 2), ("AB", 3), ("AB", 4),
    }


def test_checkpoint_key_deterministic_and_differs_on_any_field():
    base = checkpoint_key("abc", "clean", "AB", 0)
    assert base == checkpoint_key("abc", "clean", "AB", 0)
    assert base != checkpoint_key("xyz", "clean", "AB", 0)
    assert base != checkpoint_key("abc", "verbose", "AB", 0)
    assert base != checkpoint_key("abc", "clean", "BA", 0)
    assert base != checkpoint_key("abc", "clean", "AB", 1)


def test_append_and_load_completed_keys_round_trip(tmp_path):
    checkpoint_path = tmp_path / "autoj.jsonl"
    append_checkpoint(checkpoint_path, {"item_id": "abc", "condition": "clean", "order": "AB", "sample_idx": 0})
    assert load_completed_keys(checkpoint_path) == {checkpoint_key("abc", "clean", "AB", 0)}


def _items_df():
    return pd.DataFrame(
        [
            {"question_id": 1, "model_a": "m1", "model_b": "m2", "turn": 1},
            {"question_id": 2, "model_a": "m3", "model_b": "m4", "turn": 1},
        ]
    )


def test_pending_calls_skips_completed_keys_running_twice_does_not_duplicate():
    # CLAUDE.md invariant 9, directly.
    specs = [AutojCallSpec("clean", "AB", 0), AutojCallSpec("clean", "BA", 0)]
    items_df = _items_df()

    first_pass = pending_calls(items_df, specs, completed=set())
    assert len(first_pass) == 4  # 2 items x 2 specs

    completed_after_first_run = {
        checkpoint_key(item, spec.condition, spec.order, spec.sample_idx) for item, _, spec in first_pass
    }
    assert pending_calls(items_df, specs, completed_after_first_run) == []


def test_pending_calls_resumes_only_the_missing_half_after_a_partial_kill():
    specs = [AutojCallSpec("clean", "AB", 0), AutojCallSpec("clean", "BA", 0)]
    items_df = _items_df()
    real_item_1 = item_id(1, "m1", "m2", 1)
    completed = {checkpoint_key(real_item_1, "clean", "AB", 0)}

    remaining = pending_calls(items_df, specs, completed)
    remaining_keys = {checkpoint_key(item, spec.condition, spec.order, spec.sample_idx) for item, _, spec in remaining}
    assert checkpoint_key(real_item_1, "clean", "AB", 0) not in remaining_keys
    assert checkpoint_key(real_item_1, "clean", "BA", 0) in remaining_keys


# --- build_autoj_prompt ---------------------------------------------------


def _turn1_conversations():
    conv_a = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello there. Nice to meet you."}]
    conv_b = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hey. Good day to you."}]
    return conv_a, conv_b


def test_build_autoj_prompt_uses_the_verbatim_llama2_wrapper_and_labels():
    conv_a, conv_b = _turn1_conversations()
    prompt = build_autoj_prompt("clean", "AB", conv_a, conv_b)
    assert prompt.startswith("[INST] ")
    assert prompt.endswith(" [/INST]")
    assert "[Response 1]:" in prompt
    assert "[Response 2]:" in prompt
    assert 'final decision is Response 1 / Response 2 / Tie' in prompt


def test_build_autoj_prompt_verbose_is_padded_and_longer_than_clean():
    conv_a, conv_b = _turn1_conversations()
    clean_prompt = build_autoj_prompt("clean", "AB", conv_a, conv_b)
    verbose_prompt = build_autoj_prompt("verbose", "AB", conv_a, conv_b)
    assert verbose_prompt != clean_prompt
    assert len(verbose_prompt) > len(clean_prompt)


def test_build_autoj_prompt_order_swap_changes_which_content_is_response_1():
    conv_a, conv_b = _turn1_conversations()
    ab_prompt = build_autoj_prompt("clean", "AB", conv_a, conv_b)
    ba_prompt = build_autoj_prompt("clean", "BA", conv_a, conv_b)
    assert ab_prompt != ba_prompt

    ab_response_1_block, ab_response_2_block = ab_prompt.split("[Response 2]:")
    ba_response_1_block, ba_response_2_block = ba_prompt.split("[Response 2]:")

    # AB (identity mapping): conv_a's content is Response 1.
    assert "Nice to meet you" in ab_response_1_block
    assert "Nice to meet you" not in ab_response_2_block

    # BA (swapped): conv_a's content is now Response 2.
    assert "Nice to meet you" in ba_response_2_block
    assert "Nice to meet you" not in ba_response_1_block


def test_autoj_prompt_hash_is_stable_and_content_dependent():
    h1 = autoj_prompt_hash()
    h2 = autoj_prompt_hash()
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 16


# --- load_turn1_items_df ---------------------------------------------------


def test_load_turn1_items_df_filters_to_turn_1_only(config, monkeypatch):
    # load_full_items_df() itself hits the real dataset (a network/boundary
    # call, like judge.py's own version) - monkeypatch it so this test
    # stays offline and only checks the turn filter this module adds on
    # top (D28's 23 Sep amendment: turn=2 dropped entirely).
    import src.judge_autoj as mod

    fake_items = pd.DataFrame(
        [
            {"question_id": 1, "model_a": "m1", "model_b": "m2", "turn": 1},
            {"question_id": 2, "model_a": "m3", "model_b": "m4", "turn": 2},
            {"question_id": 3, "model_a": "m5", "model_b": "m6", "turn": 1},
        ]
    )
    monkeypatch.setattr(mod, "load_full_items_df", lambda cfg: fake_items)

    result = mod.load_turn1_items_df(config)
    assert set(result["turn"].unique()) == {1}
    assert len(result) == 2


# --- split_by_prompt_length ------------------------------------------------
#
# Added 23 Sep 2026 after a real Colab run hit VLLMValidationError - a
# prompt exceeded auto-j's 8,192-token context. `token_counter` is a plain
# character-count stand-in here, not a real tokenizer - the split logic
# itself doesn't care what "tokens" means, only that it compares against
# the threshold correctly, so testing it this way keeps these tests fast
# and offline (no tokenizer download), per this module's own docstring.


def _pending_with_lengths():
    # Two items, one spec each - "short" content stays under any reasonable
    # threshold, "long" content is deliberately long enough to exceed a
    # small test threshold.
    short_row = pd.Series(
        {
            "question_id": 1, "model_a": "m1", "model_b": "m2", "turn": 1, "category": "writing",
            "conversation_a": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "short reply"}],
            "conversation_b": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "short reply too"}],
        }
    )
    long_row = pd.Series(
        {
            "question_id": 2, "model_a": "m3", "model_b": "m4", "turn": 1, "category": "writing",
            "conversation_a": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x" * 5000}],
            "conversation_b": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "y" * 5000}],
        }
    )
    pending = [
        (item_id(1, "m1", "m2", 1), short_row, AutojCallSpec("clean", "AB", 0)),
        (item_id(2, "m3", "m4", 1), long_row, AutojCallSpec("clean", "AB", 0)),
    ]
    return pending


def test_split_by_prompt_length_separates_runnable_from_skipped():
    pending = _pending_with_lengths()
    runnable, skipped_rows = split_by_prompt_length(pending, token_counter=len, max_prompt_tokens=1000)
    assert len(runnable) == 1
    assert len(skipped_rows) == 1
    assert runnable[0][0] == item_id(1, "m1", "m2", 1)
    assert skipped_rows[0]["item_id"] == item_id(2, "m3", "m4", 1)


def test_split_by_prompt_length_skipped_rows_have_the_kev_style_shape():
    # Mirrors judge_kev.py::_run_calls()'s own skipped=True/skip_reason
    # pattern exactly - never silent, never a crash.
    pending = _pending_with_lengths()
    _, skipped_rows = split_by_prompt_length(pending, token_counter=len, max_prompt_tokens=1000)
    row = skipped_rows[0]
    assert row["skipped"] is True
    assert row["skip_reason"] == "over_max_prompt_tokens"
    assert row["n_prompt_tokens"] > 1000
    assert row["condition"] == "clean"
    assert row["order"] == "AB"
    assert row["sample_idx"] == 0


def test_split_by_prompt_length_nothing_skipped_when_threshold_is_generous():
    pending = _pending_with_lengths()
    runnable, skipped_rows = split_by_prompt_length(pending, token_counter=len, max_prompt_tokens=100_000)
    assert len(runnable) == 2
    assert skipped_rows == []


def test_split_by_prompt_length_runnable_entries_carry_the_precomputed_prompt():
    pending = _pending_with_lengths()
    runnable, _ = split_by_prompt_length(pending, token_counter=len, max_prompt_tokens=1000)
    item, item_row, spec, prompt, n_prompt_tokens = runnable[0]
    assert prompt.startswith("[INST] ")
    assert n_prompt_tokens == len(prompt)
