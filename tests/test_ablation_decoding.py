"""Tests for src/ablation_decoding.py's locally-testable logic.
Deliberately does not import vllm - same D17 pattern as test_judge.py.
"""

import json

from src.ablation_decoding import load_ablation_calls, load_completed_ablation_keys


def test_load_ablation_calls_parses_each_row_and_drops_raw_output(tmp_path):
    rows = [
        {"item_id": "abc", "decoding_mode": "constrained",
         "raw_output": json.dumps({"reasoning": "B is better.", "verdict": "B", "confidence": 0.9})},
        {"item_id": "abc", "decoding_mode": "free_form", "raw_output": "I prefer B."},
    ]
    path = tmp_path / "ablation_decoding.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    calls = load_ablation_calls(path)

    assert calls["decoding_mode"].tolist() == ["constrained", "free_form"]
    assert calls["parse_ok"].tolist() == [True, False]
    assert calls.loc[0, "verdict"] == "B"
    assert "raw_output" not in calls.columns  # invariant 7


def test_load_completed_ablation_keys_empty_when_file_missing(tmp_path):
    assert load_completed_ablation_keys(tmp_path / "does_not_exist.jsonl") == set()


def test_load_completed_ablation_keys_round_trip(tmp_path):
    checkpoint_path = tmp_path / "ablation_decoding.jsonl"
    rows = [
        {"item_id": "abc", "decoding_mode": "constrained"},
        {"item_id": "abc", "decoding_mode": "free_form"},
        {"item_id": "xyz", "decoding_mode": "constrained"},
    ]
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    completed = load_completed_ablation_keys(checkpoint_path)

    assert completed == {"abc|constrained", "abc|free_form", "xyz|constrained"}
