"""Tests for src/kev_signals.py: the raw-checkpoint -> calls -> items
conversion (TASKS.md K3b). Mirrors tests/test_signals.py's hand-computed
style for the order-correction and entropy signals, since kev_signals.py
deliberately reuses the exact same formulas (signals.py::_binary_entropy)
and the exact same order-correction logic as conf_bpe/_p_model_a_wins -
this was flagged as a real bug risk before being built (DECISIONS.md D27:
naively averaging prob_a across AB/BA would blend two different physical
questions), so it gets the most test coverage here.
"""

import json
import math

import pandas as pd
import pytest

from src.kev_signals import (
    build_calls_kev,
    build_items_kev_dataframe,
    conf_kev,
    conf_kev_bpe,
    flipped_kev,
    judge_verdict_kev,
    verdict_bidir_kev,
    _p_model_a_wins_kev,
)


def _call(order: str, choice: str, prob_a: float, ok: bool = True) -> dict:
    prob_b = 1 - prob_a
    return {"order": order, "ok": ok, "choice": choice, "prob_a": prob_a, "prob_b": prob_b, "input_tokens": 500}


# --- build_calls_kev ------------------------------------------------------


def test_build_calls_kev_round_trips_a_jsonl_checkpoint(tmp_path):
    checkpoint = tmp_path / "kev.jsonl"
    row = {
        "item_id": "abc",
        "question_id": 1,
        "category": None,
        "model_a": "m1",
        "model_b": "m2",
        "turn": 1,
        "condition": "clean",
        "order": "AB",
        "judge_model": "jaredpalmer/kev-8b",
        "state_tokens": 400,
        "skipped": False,
        "ok": True,
        "choice": "A",
        "probabilities": {"A": 0.7, "B": 0.3},
        "input_tokens": 420,
    }
    checkpoint.write_text(json.dumps(row) + "\n", encoding="utf-8")

    df = build_calls_kev(checkpoint)
    assert len(df) == 1
    assert df.iloc[0]["prob_a"] == 0.7
    assert df.iloc[0]["prob_b"] == 0.3
    assert df.iloc[0]["choice"] == "A"


def test_build_calls_kev_handles_skipped_rows_with_no_probabilities(tmp_path):
    checkpoint = tmp_path / "kev.jsonl"
    row = {
        "item_id": "abc",
        "question_id": 1,
        "category": None,
        "model_a": "m1",
        "model_b": "m2",
        "turn": 1,
        "condition": "verbose",
        "order": "AB",
        "judge_model": "jaredpalmer/kev-8b",
        "state_tokens": 9000,
        "skipped": True,
        "skip_reason": "over_max_state_tokens",
        "ok": False,
    }
    checkpoint.write_text(json.dumps(row) + "\n", encoding="utf-8")

    df = build_calls_kev(checkpoint)
    assert bool(df.iloc[0]["skipped"]) is True
    assert pd.isna(df.iloc[0]["prob_a"])
    assert pd.isna(df.iloc[0]["choice"])


# --- _p_model_a_wins_kev: the order-correction, most important to get right


def test_p_model_a_wins_kev_hand_computed():
    # Same numbers as test_signals.py's own conf_bpe hand-computed case,
    # for direct comparability: AB's prob_a=0.8, BA's prob_a=0.3 ->
    # order-corrected mean = (0.8 + (1 - 0.3)) / 2 = 0.75.
    rows = [_call("AB", "A", prob_a=0.8), _call("BA", "B", prob_a=0.3)]
    assert _p_model_a_wins_kev(rows) == pytest.approx(0.75)


def test_p_model_a_wins_kev_is_not_a_naive_average():
    # The exact bug this function exists to prevent: a naive average of
    # prob_a across AB/BA would give (0.9+0.9)/2=0.9, not the order-
    # corrected 0.5 this should actually produce when both orders agree
    # the SAME physical model wins (AB says model_a's slot wins at 0.9;
    # BA says model_a's slot - now occupied by model_b's content - wins
    # at 0.9, meaning model_b actually won BA, canceling out model_a's
    # apparent edge from AB).
    rows = [_call("AB", "A", prob_a=0.9), _call("BA", "A", prob_a=0.9)]
    naive_average = 0.9
    assert _p_model_a_wins_kev(rows) == pytest.approx(0.5)
    assert _p_model_a_wins_kev(rows) != pytest.approx(naive_average)


def test_p_model_a_wins_kev_none_when_an_order_is_missing_skipped_or_failed():
    assert _p_model_a_wins_kev([_call("AB", "A", prob_a=0.9)]) is None
    assert _p_model_a_wins_kev([_call("AB", "A", prob_a=0.9, ok=False), _call("BA", "B", prob_a=0.2)]) is None


# --- judge_verdict_kev / verdict_bidir_kev / conf_kev ----------------------


def test_judge_verdict_kev_is_the_ab_order_choice():
    rows = [_call("AB", "A", prob_a=0.8), _call("BA", "B", prob_a=0.3)]
    assert judge_verdict_kev(rows) == "A"


def test_verdict_bidir_kev_uses_order_corrected_mean():
    # Order-corrected mean 0.75 >= 0.5 -> "A", even if AB/BA individually disagreed.
    rows = [_call("AB", "A", prob_a=0.8), _call("BA", "B", prob_a=0.3)]
    assert verdict_bidir_kev(rows) == "A"


def test_conf_kev_is_probability_of_the_chosen_verdict():
    rows = [_call("AB", "A", prob_a=0.8), _call("BA", "B", prob_a=0.3)]
    assert conf_kev(rows) == pytest.approx(0.8)  # prob_a, since AB's choice was "A"


# --- flipped_kev -------------------------------------------------------------


def test_flipped_kev_true_when_canonical_verdicts_disagree():
    # AB says "A" (model_a wins). BA's own choice is "A" too, but under BA
    # displayed-A = model_b, so BA's canonical verdict is "B" (model_b
    # wins) - AB and BA disagree in canonical terms -> flipped.
    rows = [_call("AB", "A", prob_a=0.8), _call("BA", "A", prob_a=0.7)]
    assert flipped_kev(rows) is True


def test_flipped_kev_false_when_canonical_verdicts_agree():
    # AB says "A" (model_a wins). BA's own choice is "B", which under BA's
    # translation also means model_a wins - agree -> not flipped.
    rows = [_call("AB", "A", prob_a=0.8), _call("BA", "B", prob_a=0.7)]
    assert flipped_kev(rows) is False


def test_flipped_kev_none_when_an_order_is_missing():
    assert flipped_kev([_call("AB", "A", prob_a=0.8)]) is None


# --- conf_kev_bpe -----------------------------------------------------------


def test_conf_kev_bpe_hand_computed():
    rows = [_call("AB", "A", prob_a=0.8), _call("BA", "B", prob_a=0.3)]
    p = 0.75
    expected_entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    assert conf_kev_bpe(rows) == pytest.approx(1 - expected_entropy)


def test_conf_kev_bpe_none_when_missing_data():
    assert conf_kev_bpe([_call("AB", "A", prob_a=0.8)]) is None


# --- build_items_kev_dataframe: integration ---------------------------------


def _calls_df():
    return pd.DataFrame.from_records(
        [
            {
                "item_id": "item1", "question_id": 1, "category": None, "turn": 1,
                "condition": "clean", "order": "AB", "skipped": False, "ok": True,
                "choice": "A", "prob_a": 0.8, "prob_b": 0.2, "input_tokens": 500,
            },
            {
                "item_id": "item1", "question_id": 1, "category": None, "turn": 1,
                "condition": "clean", "order": "BA", "skipped": False, "ok": True,
                "choice": "B", "prob_a": 0.3, "prob_b": 0.7, "input_tokens": 500,
            },
            {
                "item_id": "item2", "question_id": 2, "category": None, "turn": 1,
                "condition": "verbose", "order": "AB", "skipped": True, "ok": False,
                "choice": None, "prob_a": None, "prob_b": None, "input_tokens": None,
                "state_tokens": 9000,
            },
            {
                "item_id": "item2", "question_id": 2, "category": None, "turn": 1,
                "condition": "verbose", "order": "BA", "skipped": True, "ok": False,
                "choice": None, "prob_a": None, "prob_b": None, "input_tokens": None,
                "state_tokens": 9000,
            },
        ]
    )


def _items_labels_df():
    return pd.DataFrame.from_records(
        [
            {"item_id": "item1", "majority_label": "A", "n_human_votes": 3, "frac_prefer_a": 0.8,
             "human_unanimous": False, "human_agreed": True, "d_human": 0.3},
            {"item_id": "item2", "majority_label": "B", "n_human_votes": 3, "frac_prefer_a": 0.2,
             "human_unanimous": True, "human_agreed": True, "d_human": 0.3},
        ]
    )


def test_build_items_kev_dataframe_grain_is_item_and_condition():
    items = build_items_kev_dataframe(_calls_df(), _items_labels_df())
    assert len(items) == 2  # one row per (item_id, condition), not per call
    assert set(items["item_id"]) == {"item1", "item2"}


def test_build_items_kev_dataframe_correct_matches_human_label():
    items = build_items_kev_dataframe(_calls_df(), _items_labels_df()).set_index("item_id")
    # item1: judge_verdict (AB choice) = "A", human_label = "A" -> correct.
    assert items.loc["item1", "judge_verdict"] == "A"
    assert items.loc["item1", "correct"] is True


def test_build_items_kev_dataframe_fully_skipped_item_has_null_signals_and_state_tokens_fallback():
    items = build_items_kev_dataframe(_calls_df(), _items_labels_df()).set_index("item_id")
    assert pd.isna(items.loc["item2", "judge_verdict"])
    assert pd.isna(items.loc["item2", "correct"])
    assert bool(items.loc["item2", "any_skipped"]) is True
    # No successful call means no input_tokens - falls back to state_tokens.
    assert items.loc["item2", "input_tokens"] == 9000
