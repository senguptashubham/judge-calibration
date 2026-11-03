"""Tests for analysis/rq7.py's survivorship fill: the padded items auto-j
gave no usable verdict on are put back in with a filled signal."""

import numpy as np
import pandas as pd
import pytest

from analysis.rq7 import fill_missing_verdicts

SIGNAL, PROB = "conf_sc_bpe_autoj_greedy", "conf_sc_bpe_autoj_greedy_prob"
FLOOR = 1 - np.log(2)


def _items() -> pd.DataFrame:
    # Rows: answered and right; no verdict at all; verdict present but the
    # other order failed, so only the signal is missing.
    return pd.DataFrame(
        {
            "correct": [True, None, True],
            SIGNAL: [0.9, None, None],
            PROB: [0.95, None, None],
        }
    )


def test_escalated_counts_no_verdict_as_wrong_at_lowest_confidence():
    filled = fill_missing_verdicts(_items(), SIGNAL, PROB, "escalated")

    assert filled["correct"].tolist() == [True, False, True]
    assert filled[SIGNAL].tolist() == pytest.approx([0.9, FLOOR, FLOOR])
    assert filled[PROB].tolist() == pytest.approx([0.95, 0.5, 0.5])


def test_best_and_worst_case_put_filled_rows_at_opposite_extremes():
    best = fill_missing_verdicts(_items(), SIGNAL, PROB, "best_case")
    worst = fill_missing_verdicts(_items(), SIGNAL, PROB, "worst_case")

    # The wrong row is least confident in the best case and most in the worst;
    # the right row the other way round. The answered row never moves.
    assert best[SIGNAL].tolist() == pytest.approx([0.9, FLOOR, 1.0])
    assert worst[SIGNAL].tolist() == pytest.approx([0.9, 1.0, FLOOR])
