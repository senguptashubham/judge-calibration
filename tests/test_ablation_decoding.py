"""Tests for src/ablation_decoding.py's locally-testable logic (task 4.5).
Deliberately does not import vllm - same D17 pattern as test_judge.py.
"""

import json

from src.ablation_decoding import load_completed_ablation_keys


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
