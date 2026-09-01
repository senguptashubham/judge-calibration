"""Tests for src/data.py: build_items()'s tie-policy logic (D1) and
human_human_kappa()'s pairing logic. See TASKS.md tasks 0.8, 0.9.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import build_items, human_human_kappa, summarize_items


def _votes(winners, judges=None, question_id=1, model_a="m1", model_b="m2", turn=1):
    n = len(winners)
    if judges is None:
        judges = [f"judge_{i}" for i in range(n)]
    return pd.DataFrame(
        {
            "question_id": [question_id] * n,
            "model_a": [model_a] * n,
            "model_b": [model_b] * n,
            "turn": [turn] * n,
            "winner": winners,
            "judge": judges,
        }
    )


def _one_item(winners, judges=None):
    votes = _votes(winners, judges=judges)
    items = build_items(votes, tie_policy="mark_tie_strict")
    assert len(items) == 1
    return items.iloc[0]


def test_tie_dominant_group_is_marked_tied():
    item = _one_item(["tie", "tie", "tie", "model_a", "model_b"])
    assert item["is_tie"]
    assert np.isnan(item["frac_prefer_a"])
    assert item["majority_label"] is None
    assert not item["human_unanimous"]
    assert not item["human_agreed"]
    assert np.isnan(item["d_human"])


def test_clean_majority_non_tie():
    item = _one_item(["model_a", "model_a", "model_a", "model_b"])
    assert not item["is_tie"]
    assert item["frac_prefer_a"] == pytest.approx(0.75)
    assert item["majority_label"] == "A"
    assert not item["human_unanimous"]
    assert item["d_human"] == pytest.approx(0.25)


def test_unanimous_non_tie_group():
    item = _one_item(["model_a", "model_a"])
    assert not item["is_tie"]
    assert item["frac_prefer_a"] == pytest.approx(1.0)
    assert item["majority_label"] == "A"
    assert item["human_unanimous"]
    assert item["human_agreed"]


def test_single_vote_item_is_not_agreed_even_if_trivially_unanimous():
    # D16: a single-vote item is trivially "unanimous" (nothing to disagree
    # with) but must not count as agreed ground truth.
    item = _one_item(["model_a"])
    assert item["human_unanimous"]
    assert item["n_human_votes"] == 1
    assert not item["human_agreed"]


def test_tie_tied_with_leader_is_not_marked_tied_under_strict():
    # The exact boundary that motivated "strict" over "lenient" (D1): tie
    # count equals the leading side's count but does not exceed it, so the
    # item is NOT tied, and the leader (A, 2-to-1 over B among decisive
    # voters) gets its real majority_label instead of being discarded.
    item = _one_item(["tie", "tie", "model_a", "model_a", "model_b"])
    assert not item["is_tie"]
    assert item["frac_prefer_a"] == pytest.approx(2 / 3)
    assert item["majority_label"] == "A"


def test_build_items_rejects_unknown_tie_policy():
    votes = _votes(["model_a", "model_b"])
    with pytest.raises(ValueError):
        build_items(votes, tie_policy="drop_ties")


def test_summarize_items_counts():
    votes = pd.concat(
        [
            _votes(["model_a"], question_id=1),  # 1 vote, trivially unanimous
            _votes(["model_a", "model_b"], question_id=2),  # split, contested
            _votes(["model_a", "model_a", "model_b"], question_id=3),  # majority A, contested
            _votes(["tie", "tie", "model_a"], question_id=4),  # tie-dominant
        ],
        ignore_index=True,
    )
    items = build_items(votes, tie_policy="mark_tie_strict")
    summary = summarize_items(items)
    assert summary == {
        "n_total": 4,
        "n_non_tie": 3,
        "n_ge2_votes": 3,
        "n_ge3_votes": 2,
        "n_unanimous": 1,
        "n_contested": 3,
    }


def test_human_human_kappa_hand_computed():
    votes = pd.concat(
        [
            _votes(["model_a", "model_a"], judges=["j1", "j2"], question_id=1),
            _votes(["model_a", "model_b"], judges=["j1", "j2"], question_id=2),
            _votes(["tie", "model_a"], judges=["j1", "j2"], question_id=3),  # 1 non-tie vote -> excluded
            _votes(["model_a"], judges=["j1"], question_id=4),  # 1 vote -> excluded
        ],
        ignore_index=True,
    )
    # rater1=[A, A], rater2=[A, B] -> p_o=0.5, p_e=1.0*0.5 + 0*0.5=0.5, kappa=0.
    kappa, n = human_human_kappa(votes)
    assert n == 2
    assert kappa == pytest.approx(0.0, abs=1e-9)


def test_human_human_kappa_ordering_is_deterministic_regardless_of_row_order():
    # 3 votes on one item: judge A -> model_a, judge B -> model_b,
    # judge C -> model_b. Sorting by judge (A, B, C) always picks A and C
    # as the endpoints: (model_a, model_b). Without sorting by judge, the
    # source data's own row order could instead expose B and C as the
    # apparent "first"/"last" - a different pair, (model_b, model_b) - which
    # must not happen: the result must not depend on incoming row order.
    votes_order1 = _votes(["model_a", "model_b", "model_b"], judges=["A", "B", "C"], question_id=1)
    votes_order2 = _votes(["model_b", "model_a", "model_b"], judges=["B", "A", "C"], question_id=1)

    kappa1, n1 = human_human_kappa(votes_order1)
    kappa2, n2 = human_human_kappa(votes_order2)

    assert n1 == n2 == 1
    assert kappa1 == pytest.approx(kappa2)
