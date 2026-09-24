"""Tests for src/autoj_signals.py: the raw-checkpoint -> calls -> items
conversion. Heaviest coverage on _p_model_a_wins_autoj, whose inputs are
already canonically translated - so it combines them with a plain average,
not signals.py's (1 - x)-flipped formula. The pure-position-bias case below
is the regression test for that.
"""

import json
import math

import pandas as pd
import pytest

from src.autoj_signals import (
    _canonical_letter,
    _p_model_a_wins_autoj,
    _self_consistency_proportion_a,
    build_calls_autoj,
    build_items_autoj_dataframe,
    conf_sc_autoj,
    conf_sc_bpe_autoj,
    conf_sc_bpe_autoj_greedy,
    extract_autoj_verdict,
    flipped_autoj,
    judge_verdict_autoj,
    verdict_bidir_autoj,
)


def _call(order: str, sample_idx: int, pred_label: int | None, skipped: bool = False) -> dict:
    return {"order": order, "sample_idx": sample_idx, "skipped": skipped, "pred_label": pred_label}


# --- extract_autoj_verdict ---------------------------------------------


def test_extract_autoj_verdict_response_1():
    text = "Some critique.\n\nSo, the final decision is Response 1. It was better."
    assert extract_autoj_verdict(text) == 0


def test_extract_autoj_verdict_response_2():
    text = "Some critique.\n\nSo, the final decision is Response 2 because it was clearer."
    assert extract_autoj_verdict(text) == 1


def test_extract_autoj_verdict_tie():
    text = "Some critique.\n\nSo, the final decision is Tie."
    assert extract_autoj_verdict(text) == 2


def test_extract_autoj_verdict_case_insensitive():
    text = "So, the final decision is response 1."
    assert extract_autoj_verdict(text) == 0


def test_extract_autoj_verdict_no_match_returns_minus_one():
    text = "A critique that never reaches a conclusion because it got cut off"
    assert extract_autoj_verdict(text) == -1


def test_extract_autoj_verdict_uses_the_last_occurrence():
    # rfind, not find - mirrors the real extract_pariwise_result() exactly.
    text = "Draft: the final decision is Response 1. Revised: the final decision is Response 2."
    assert extract_autoj_verdict(text) == 1


# --- _canonical_letter ---------------------------------------------------


def test_canonical_letter_ab_order_is_identity_mapping():
    assert _canonical_letter(0, "AB") == "A"
    assert _canonical_letter(1, "AB") == "B"


def test_canonical_letter_ba_order_is_swapped():
    assert _canonical_letter(0, "BA") == "B"
    assert _canonical_letter(1, "BA") == "A"


def test_canonical_letter_tie_and_failure_and_missing_are_none():
    assert _canonical_letter(2, "AB") is None  # Tie
    assert _canonical_letter(-1, "AB") is None  # parse failure
    assert _canonical_letter(None, "AB") is None  # skipped / no call


# --- build_calls_autoj ---------------------------------------------------


def test_build_calls_autoj_computes_pred_label_from_raw_output(tmp_path):
    checkpoint = tmp_path / "autoj.jsonl"
    row = {
        "item_id": "abc", "question_id": 1, "category": "writing", "model_a": "m1", "model_b": "m2",
        "turn": 1, "condition": "clean", "order": "AB", "sample_idx": 0,
        "judge_model": "GAIR/autoj-13b-GPTQ-4bits", "skipped": False, "finish_reason": "stop",
        "raw_output": "Critique. So, the final decision is Response 1.",
        "n_prompt_tokens": 500, "n_out_tokens": 40,
    }
    checkpoint.write_text(json.dumps(row) + "\n", encoding="utf-8")
    df = build_calls_autoj(checkpoint)
    assert len(df) == 1
    assert df.iloc[0]["pred_label"] == 0


def test_build_calls_autoj_skipped_row_has_no_pred_label(tmp_path):
    checkpoint = tmp_path / "autoj.jsonl"
    row = {
        "item_id": "abc", "question_id": 1, "category": "writing", "model_a": "m1", "model_b": "m2",
        "turn": 2, "condition": "verbose", "order": "AB", "sample_idx": 0,
        "judge_model": "GAIR/autoj-13b-GPTQ-4bits", "skipped": True, "skip_reason": "over_max_prompt_tokens",
    }
    checkpoint.write_text(json.dumps(row) + "\n", encoding="utf-8")
    df = build_calls_autoj(checkpoint)
    assert bool(df.iloc[0]["skipped"]) is True
    assert pd.isna(df.iloc[0]["pred_label"])


# --- judge_verdict_autoj / _canonical_verdict_ba_autoj / flipped_autoj ----


def test_judge_verdict_autoj_is_the_ab_canonical_letter():
    rows = [_call("AB", 0, pred_label=0)]  # Response 1 under AB -> model_a
    assert judge_verdict_autoj(rows) == "A"


def test_flipped_autoj_true_when_canonical_verdicts_disagree():
    # AB: Response 1 -> model_a wins ("A"). BA: Response 1 -> model_b wins
    # ("B") under BA's own translation - canonical verdicts disagree.
    rows = [_call("AB", 0, pred_label=0), _call("BA", 0, pred_label=0)]
    assert flipped_autoj(rows) is True


def test_flipped_autoj_false_when_canonical_verdicts_agree():
    # AB: Response 1 -> "A". BA: Response 2 -> model_a wins under BA's own
    # translation -> "A" too - agree, not flipped.
    rows = [_call("AB", 0, pred_label=0), _call("BA", 0, pred_label=1)]
    assert flipped_autoj(rows) is False


def test_flipped_autoj_none_when_an_order_is_missing():
    assert flipped_autoj([_call("AB", 0, pred_label=0)]) is None


# --- _self_consistency_proportion_a / _p_model_a_wins_autoj ---------------


def test_self_consistency_proportion_a_excludes_tie_and_failure():
    # 5 AB draws: Response1, Response1, Response2, Response1, Tie.
    # Canonical (AB): A, A, B, A, None(Tie, dropped) -> 3 A's / 4 counted = 0.75.
    rows = [
        _call("AB", 0, pred_label=0), _call("AB", 1, pred_label=0), _call("AB", 2, pred_label=1),
        _call("AB", 3, pred_label=0), _call("AB", 4, pred_label=2),
    ]
    assert _self_consistency_proportion_a(rows, "AB", k_sc=4) == pytest.approx(0.75)


def test_self_consistency_proportion_a_degenerates_to_single_call_when_no_sampling():
    # BA never gets sampled draws (D6/D19's own schedule asymmetry) - only
    # sample_idx=0 exists, so the "proportion" is just that one call's own
    # translated verdict.
    rows = [_call("BA", 0, pred_label=1)]  # Response 2 under BA -> model_a -> "A"
    assert _self_consistency_proportion_a(rows, "BA", k_sc=4) == pytest.approx(1.0)


def test_p_model_a_wins_autoj_pure_position_bias_nets_to_indifference():
    # The double-translation regression case:
    # "whichever response is displayed first wins", no real model
    # preference - AB's first-displayed is model_a (wins -> "A", p_ab=1.0);
    # BA's first-displayed is model_b (wins, so model_a loses -> canonical
    # "B", p_ba=0.0). A correctly order-corrected average of pure position
    # bias must be 0.5 (indifference), not 1.0.
    rows = [_call("AB", 0, pred_label=0), _call("BA", 0, pred_label=0)]
    assert _p_model_a_wins_autoj(rows, k_sc=4) == pytest.approx(0.5)


def test_p_model_a_wins_autoj_combines_a_multi_draw_ab_with_a_degenerate_ba():
    # AB: 0.75 favor A (from the 5-draw case above). BA: single call,
    # Response 2 -> model_a under BA -> favors A -> p_ba=1.0.
    # Plain average (already-translated inputs): (0.75 + 1.0) / 2 = 0.875.
    rows = [
        _call("AB", 0, pred_label=0), _call("AB", 1, pred_label=0), _call("AB", 2, pred_label=1),
        _call("AB", 3, pred_label=0), _call("AB", 4, pred_label=2),
        _call("BA", 0, pred_label=1),
    ]
    assert _p_model_a_wins_autoj(rows, k_sc=4) == pytest.approx(0.875)


def test_p_model_a_wins_autoj_none_when_an_order_is_missing():
    assert _p_model_a_wins_autoj([_call("AB", 0, pred_label=0)], k_sc=4) is None


def test_verdict_bidir_autoj_uses_order_corrected_mean():
    rows = [
        _call("AB", 0, pred_label=0), _call("AB", 1, pred_label=0), _call("AB", 2, pred_label=1),
        _call("AB", 3, pred_label=0), _call("AB", 4, pred_label=2),
        _call("BA", 0, pred_label=1),
    ]
    assert verdict_bidir_autoj(rows, k_sc=4) == "A"  # 0.875 >= 0.5


# --- conf_sc_autoj (clean only) -------------------------------------------


def test_conf_sc_autoj_fraction_matching_canonical_greedy():
    # Canonical greedy (sample_idx=0): pred_label=0. Sampled draws (1-4):
    # [0, 1, 0, 2] -> matches canonical (0): [True, False, True, False] -> 2/4=0.5.
    rows = [
        _call("AB", 0, pred_label=0), _call("AB", 1, pred_label=0), _call("AB", 2, pred_label=1),
        _call("AB", 3, pred_label=0), _call("AB", 4, pred_label=2),
    ]
    assert conf_sc_autoj(rows, k_sc=4) == pytest.approx(0.5)


def test_conf_sc_autoj_none_on_verbose_no_sampled_draws():
    # verbose never generates sample_idx 1..k_sc at all (D19's schedule,
    # reused unchanged) - only the greedy call exists.
    rows = [_call("AB", 0, pred_label=0)]
    assert conf_sc_autoj(rows, k_sc=4) is None


# --- conf_sc_bpe_autoj (defined on both conditions) ------------------------


def test_conf_sc_bpe_autoj_hand_computed():
    rows = [
        _call("AB", 0, pred_label=0), _call("AB", 1, pred_label=0), _call("AB", 2, pred_label=1),
        _call("AB", 3, pred_label=0), _call("AB", 4, pred_label=2),
        _call("BA", 0, pred_label=1),
    ]
    p = 0.875
    expected_entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    assert conf_sc_bpe_autoj(rows, k_sc=4) == pytest.approx(1 - expected_entropy)


def test_conf_sc_bpe_autoj_still_defined_on_a_verbose_style_degenerate_pair():
    # Both orders reduced to a single greedy call each (verbose's real
    # shape) - must still be defined, or the verbosity test has no signal.
    rows = [_call("AB", 0, pred_label=0), _call("BA", 0, pred_label=1)]
    result = conf_sc_bpe_autoj(rows, k_sc=4)
    assert result is not None
    # Both orders favor model_a (p_ab=1.0, p_ba=1.0) -> p=1.0 -> zero entropy -> conf=1.0.
    assert result == pytest.approx(1.0)


def test_conf_sc_bpe_autoj_greedy_ignores_sampled_draws():
    # Same rows as the hand-computed case above: the pooled signal sees
    # p=0.875, but the greedy-only one sees just AB=model_a, BA=model_a -> p=1.0.
    rows = [
        _call("AB", 0, pred_label=0), _call("AB", 1, pred_label=0), _call("AB", 2, pred_label=1),
        _call("AB", 3, pred_label=0), _call("AB", 4, pred_label=2),
        _call("BA", 0, pred_label=1),
    ]
    assert conf_sc_bpe_autoj_greedy(rows) == pytest.approx(1.0)
    assert conf_sc_bpe_autoj_greedy(rows) != pytest.approx(conf_sc_bpe_autoj(rows, k_sc=4))


# --- compute_item_signals_autoj / build_items_autoj_dataframe -------------


def _calls_df():
    return pd.DataFrame.from_records(
        [
            {
                "item_id": "item1", "question_id": 1, "category": "writing", "turn": 1,
                "condition": "clean", "order": "AB", "sample_idx": 0, "skipped": False,
                "pred_label": 0, "n_prompt_tokens": 500,
            },
            {
                "item_id": "item1", "question_id": 1, "category": "writing", "turn": 1,
                "condition": "clean", "order": "BA", "sample_idx": 0, "skipped": False,
                "pred_label": 1, "n_prompt_tokens": 500,
            },
            {
                "item_id": "item2", "question_id": 2, "category": "coding", "turn": 2,
                "condition": "verbose", "order": "AB", "sample_idx": 0, "skipped": True,
                "pred_label": None, "n_prompt_tokens": None,
            },
            {
                "item_id": "item2", "question_id": 2, "category": "coding", "turn": 2,
                "condition": "verbose", "order": "BA", "sample_idx": 0, "skipped": True,
                "pred_label": None, "n_prompt_tokens": None,
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


def test_build_items_autoj_dataframe_grain_is_item_and_condition():
    items = build_items_autoj_dataframe(_calls_df(), _items_labels_df(), k_sc=4)
    assert len(items) == 2
    assert set(items["item_id"]) == {"item1", "item2"}


def test_build_items_autoj_dataframe_correct_matches_human_label():
    items = build_items_autoj_dataframe(_calls_df(), _items_labels_df(), k_sc=4).set_index("item_id")
    # item1: AB pred_label=0 -> judge_verdict "A", human_label "A" -> correct.
    assert items.loc["item1", "judge_verdict"] == "A"
    assert items.loc["item1", "correct"] is True


def test_build_items_autoj_dataframe_fully_skipped_item_has_null_signals():
    items = build_items_autoj_dataframe(_calls_df(), _items_labels_df(), k_sc=4).set_index("item_id")
    assert pd.isna(items.loc["item2", "judge_verdict"])
    assert pd.isna(items.loc["item2", "correct"])
    assert bool(items.loc["item2", "any_skipped"]) is True
