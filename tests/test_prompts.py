"""Tests for src/prompts.py, including the order-renderer round-trip
property (rendering AB then BA returns the original assignment) that moved
lives here, not in test_perturb.py: order is a renderer concern (D5).
"""

import pytest

from src.prompts import VARIANTS, apply_order, prompt_hash, render_prompt

_CONV_1TURN_A = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "UNIQUE_A_TURN1"},
]
_CONV_1TURN_B = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "UNIQUE_B_TURN1"},
]
_CONV_2TURN_A = _CONV_1TURN_A + [
    {"role": "user", "content": "Now answer in French."},
    {"role": "assistant", "content": "UNIQUE_A_TURN2"},
]
_CONV_2TURN_B = _CONV_1TURN_B + [
    {"role": "user", "content": "Now answer in French."},
    {"role": "assistant", "content": "UNIQUE_B_TURN2"},
]


def _block(prompt: str, label: str) -> str:
    # Split the rendered prompt into the "Assistant A" and "Assistant B"
    # sections so assertions can check what landed under which label.
    marker_a = "[Conversation with Assistant A]"
    marker_b = "[Conversation with Assistant B]"
    task_marker = "[Your task]"
    a_start = prompt.index(marker_a)
    b_start = prompt.index(marker_b)
    task_start = prompt.index(task_marker)
    if label == "A":
        return prompt[a_start:b_start]
    return prompt[b_start:task_start]


@pytest.mark.parametrize("variant", VARIANTS)
def test_prompt_hash_stable_across_calls(variant):
    assert prompt_hash(variant) == prompt_hash(variant)


def test_prompt_hash_differs_across_variants():
    hashes = {variant: prompt_hash(variant) for variant in VARIANTS}
    assert len(set(hashes.values())) == len(VARIANTS)


def test_prompt_hash_rejects_unknown_variant():
    with pytest.raises(ValueError):
        prompt_hash("P4")


@pytest.mark.parametrize("variant", VARIANTS)
def test_render_prompt_contains_json_contract(variant):
    prompt = render_prompt(variant, "AB", _CONV_1TURN_A, _CONV_1TURN_B, turn=1)
    assert '"reasoning"' in prompt
    assert '"verdict"' in prompt
    assert '"confidence"' in prompt


def test_render_prompt_turn1_omits_second_turn():
    prompt = render_prompt("P1", "AB", _CONV_1TURN_A, _CONV_1TURN_B, turn=1)
    assert "UNIQUE_A_TURN1" in prompt
    assert "UNIQUE_B_TURN1" in prompt
    assert "TURN2" not in prompt


def test_render_prompt_turn2_includes_full_history():
    prompt = render_prompt("P1", "AB", _CONV_2TURN_A, _CONV_2TURN_B, turn=2)
    assert "UNIQUE_A_TURN1" in prompt
    assert "UNIQUE_A_TURN2" in prompt
    assert "UNIQUE_B_TURN1" in prompt
    assert "UNIQUE_B_TURN2" in prompt


def test_render_prompt_rejects_bad_turn():
    with pytest.raises(ValueError):
        render_prompt("P1", "AB", _CONV_1TURN_A, _CONV_1TURN_B, turn=3)


def test_apply_order_ab_is_identity():
    displayed_a, displayed_b = apply_order("AB", _CONV_1TURN_A, _CONV_1TURN_B)
    assert displayed_a == _CONV_1TURN_A
    assert displayed_b == _CONV_1TURN_B


def test_apply_order_ba_swaps():
    displayed_a, displayed_b = apply_order("BA", _CONV_1TURN_A, _CONV_1TURN_B)
    assert displayed_a == _CONV_1TURN_B
    assert displayed_b == _CONV_1TURN_A


def test_apply_order_rejects_unknown_order():
    with pytest.raises(ValueError):
        apply_order("SWAP", _CONV_1TURN_A, _CONV_1TURN_B)


def test_order_round_trip():
    # D5: rendering AB then BA returns the original assignment - the
    # content shown under "Assistant A" and "Assistant B" swaps between
    # the two orders, but the underlying text itself is never altered.
    prompt_ab = render_prompt("P1", "AB", _CONV_1TURN_A, _CONV_1TURN_B, turn=1)
    prompt_ba = render_prompt("P1", "BA", _CONV_1TURN_A, _CONV_1TURN_B, turn=1)

    assert "UNIQUE_A_TURN1" in _block(prompt_ab, "A")
    assert "UNIQUE_B_TURN1" in _block(prompt_ab, "B")

    # Under BA, model_a's content is now displayed as Assistant B and vice
    # versa - the exact swap that makes order a position-bias probe.
    assert "UNIQUE_B_TURN1" in _block(prompt_ba, "A")
    assert "UNIQUE_A_TURN1" in _block(prompt_ba, "B")
