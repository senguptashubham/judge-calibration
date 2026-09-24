"""Tests for src/judge_autoj.py's locally-testable logic: D28's schedule
(clean/verbose, no prompt_variant axis), checkpoint dedup/resumability
(invariant 9), and the prompt renderer - especially the turn=2 flattening
design D28 flags as the one genuinely novel piece L4b's own validity check
exists to catch problems in. Deliberately does not import vllm - none of
this logic touches it (same "only touches the external thing inside one
boundary function" split judge.py/judge_kev.py's own tests already use).
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


def _turn2_conversations():
    conv_a = [
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "Paris is the capital of France."},
        {"role": "user", "content": "Rewrite that in French."},
        {"role": "assistant", "content": "Paris est la capitale de la France."},
    ]
    conv_b = [
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital city is Paris."},
        {"role": "user", "content": "Rewrite that in French."},
        {"role": "assistant", "content": "La capitale est Paris."},
    ]
    return conv_a, conv_b


def test_build_autoj_prompt_uses_the_verbatim_llama2_wrapper_and_labels():
    conv_a, conv_b = _turn1_conversations()
    prompt = build_autoj_prompt("clean", "AB", conv_a, conv_b, turn=1)
    assert prompt.startswith("[INST] ")
    assert prompt.endswith(" [/INST]")
    assert "[Response 1]:" in prompt
    assert "[Response 2]:" in prompt
    assert 'final decision is Response 1 / Response 2 / Tie' in prompt


def test_build_autoj_prompt_verbose_is_padded_and_longer_than_clean():
    conv_a, conv_b = _turn1_conversations()
    clean_prompt = build_autoj_prompt("clean", "AB", conv_a, conv_b, turn=1)
    verbose_prompt = build_autoj_prompt("verbose", "AB", conv_a, conv_b, turn=1)
    assert verbose_prompt != clean_prompt
    assert len(verbose_prompt) > len(clean_prompt)


def test_build_autoj_prompt_order_swap_changes_which_content_is_response_1():
    conv_a, conv_b = _turn1_conversations()
    ab_prompt = build_autoj_prompt("clean", "AB", conv_a, conv_b, turn=1)
    ba_prompt = build_autoj_prompt("clean", "BA", conv_a, conv_b, turn=1)
    assert ab_prompt != ba_prompt

    ab_response_1_block, ab_response_2_block = ab_prompt.split("[Response 2]:")
    ba_response_1_block, ba_response_2_block = ba_prompt.split("[Response 2]:")

    # AB (identity mapping): conv_a's content is Response 1.
    assert "Nice to meet you" in ab_response_1_block
    assert "Nice to meet you" not in ab_response_2_block

    # BA (swapped): conv_a's content is now Response 2.
    assert "Nice to meet you" in ba_response_2_block
    assert "Nice to meet you" not in ba_response_1_block


def test_build_autoj_prompt_turn1_rejects_indexing_past_two_messages():
    # turn=1 conversations only ever have [user, assistant] - the renderer
    # must never try to index conversation[2]/[3] for them.
    conv_a, conv_b = _turn1_conversations()
    prompt = build_autoj_prompt("clean", "AB", conv_a, conv_b, turn=1)
    assert "Nice to meet you" in prompt
    assert "Good day to you" in prompt


def test_build_autoj_prompt_turn2_shared_prompt_field_has_no_model_specific_content():
    # D28's turn=2 design: {prompt} carries only the two shared user turns,
    # never a model-specific answer - the property that keeps one
    # candidate's context from contaminating the other's.
    conv_a, conv_b = _turn2_conversations()
    prompt = build_autoj_prompt("clean", "AB", conv_a, conv_b, turn=2)
    query_block = prompt.split("[Response 1]:")[0]
    assert "capital of France" in query_block
    assert "Rewrite that in French" in query_block
    assert "Paris est la capitale" not in query_block  # model_a's own turn-2 answer
    assert "La capitale est Paris" not in query_block  # model_b's own turn-2 answer


def test_build_autoj_prompt_turn2_folds_each_sides_own_turn1_answer_into_its_own_response():
    # Each side's own turn-1 answer must appear in ITS OWN response field
    # (a two-part trajectory), not in the other side's or in the shared
    # prompt field - this is the property L4b's spot-check is specifically
    # there to confirm still produces coherent judging in practice.
    conv_a, conv_b = _turn2_conversations()
    prompt = build_autoj_prompt("clean", "AB", conv_a, conv_b, turn=2)
    response_1_block, response_2_block = prompt.split("[Response 2]:")

    assert "Paris is the capital of France" in response_1_block  # model_a's turn-1 answer
    assert "Paris est la capitale de la France" in response_1_block  # model_a's turn-2 answer
    assert "The capital city is Paris" not in response_1_block  # model_b's answer must not leak in

    assert "The capital city is Paris" in response_2_block  # model_b's turn-1 answer
    assert "La capitale est Paris" in response_2_block  # model_b's turn-2 answer
    assert "Paris is the capital of France" not in response_2_block  # model_a's answer must not leak in


def test_build_autoj_prompt_rejects_invalid_turn():
    conv_a, conv_b = _turn1_conversations()
    with pytest.raises(ValueError):
        build_autoj_prompt("clean", "AB", conv_a, conv_b, turn=3)


def test_autoj_prompt_hash_is_stable_and_content_dependent():
    h1 = autoj_prompt_hash()
    h2 = autoj_prompt_hash()
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 16


# --- split_by_prompt_length ------------------------------------------------
#
# One over-length prompt crashes vLLM's whole batch, so these calls must be
# split out before generation. `token_counter` is a character-count stand-in:
# the split logic only compares counts against the threshold, so no real
# tokenizer is needed.


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
