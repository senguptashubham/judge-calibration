"""Tests for the vacuum test's parsing step (src/vacuum_test.py) and its
confidence-gap CI (analysis/vacuum.py). The GPU runner itself is not tested."""

import json

import pandas as pd
import pytest

from analysis.vacuum import confidence_gap_vs_real
from src.vacuum_test import load_vacuum_calls


def test_load_vacuum_calls_parses_each_row_and_keeps_the_cluster_key(tmp_path):
    rows = [
        {"vacuum_type": "identical", "question_id": 81,
         "raw_output": json.dumps({"reasoning": "Same.", "verdict": "A", "confidence": 0.95})},
        {"vacuum_type": "empty", "question_id": 82, "raw_output": "not json"},
    ]
    path = tmp_path / "vacuum.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    calls = load_vacuum_calls(path)

    assert calls["question_id"].tolist() == [81, 82]
    assert calls["parse_ok"].tolist() == [True, False]
    assert calls.loc[0, "verdict"] == "A"
    assert calls.loc[0, "verbalized_conf"] == 0.95


def test_confidence_gap_is_vacuum_mean_minus_real_mean_with_ci_around_it():
    vacuum = pd.DataFrame({"question_id": [1, 2, 3, 4], "conf": [0.95, 1.0, 0.95, 1.0]})
    real = pd.DataFrame({"question_id": [1, 1, 2, 2, 3, 3, 4, 4], "conf": [0.9, 0.8] * 4})

    point, low, high = confidence_gap_vs_real(vacuum, real, n=200, seed=0)

    assert point == pytest.approx(0.125)
    assert low <= point <= high
