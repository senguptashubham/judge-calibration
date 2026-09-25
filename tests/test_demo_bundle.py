"""Tests for analysis/demo_bundle.py's pure helpers: the position ->
model translation and the showcase picker. The bundle as a whole is checked
against the real data when it is built."""

import pandas as pd

from analysis.demo_bundle import canonical_winner, kev_call_view, pick_showcase_items


def test_canonical_winner_translates_ba_positions():
    assert canonical_winner("first", "AB") == "A"
    assert canonical_winner("second", "AB") == "B"
    # Under BA the answer shown first is model_b's.
    assert canonical_winner("first", "BA") == "B"
    assert canonical_winner("second", "BA") == "A"
    assert canonical_winner(None, "AB") is None


def test_kev_call_view_reads_the_probability_of_its_own_choice():
    row = pd.Series({"skipped": False, "ok": True, "choice": "B", "prob_a": 0.3, "prob_b": 0.7})
    view = kev_call_view(row, "BA")
    assert view["pick"] == "second"
    assert view["winner"] == "A"
    assert view["conf"] == 0.7


def _view(winner, conf):
    return {"status": "ok", "pick": None, "winner": winner, "conf": conf, "text": None}


def _item(item_id, length, ab, ba, padded_ab, others=("A", "A"), same_answers=False):
    calls = {"clean": {"AB": ab, "BA": ba}, "verbose": {"AB": padded_ab, "BA": _view("A", 0.9)}}
    return {
        "item_id": item_id,
        "conversation_a": [{"role": "user", "content": "x" * length}],
        "conversation_b": [{"role": "user", "content": "x" * length if same_answers else ""}],
        "human": {"agreed": True},
        "judges": {
            "Qwen2.5-7B": calls,
            "kev-8b": {"clean": {"AB": _view(others[0], 0.8)}},
            "auto-j-13b": {"clean": {"AB": _view(others[1], None)}},
        },
    }


def test_pick_showcase_items_finds_confident_flips_shortest_first():
    items = [
        _item("long_flip", 200, _view("A", 0.95), _view("B", 0.9), _view("A", 0.95)),
        _item("short_flip", 50, _view("A", 0.95), _view("B", 0.95), _view("A", 0.95)),
        _item("unsure_flip", 10, _view("A", 0.95), _view("B", 0.6), _view("A", 0.95)),
        _item("pad_flip", 60, _view("A", 0.95), _view("A", 0.95), _view("B", 0.92)),
        _item("judges_split", 70, _view("A", 0.95), _view("A", 0.95), _view("A", 0.95), others=("B", "A")),
        _item("twins", 5, _view("B", 0.95), _view("A", 0.95), _view("B", 0.95), same_answers=True),
    ]
    showcase = pick_showcase_items(items, "Qwen2.5-7B")
    by_kind = {}
    for entry in showcase:
        by_kind.setdefault(entry["kind"], []).append(entry["item_id"])
    assert by_kind["order"] == ["short_flip", "long_flip"]  # the 0.6 flip is not "confident"
    assert by_kind["padding"] == ["pad_flip"]
    assert by_kind["judges"] == ["judges_split"]
    assert by_kind["identical"] == ["twins"]  # and kept out of "order" despite flipping
    assert pick_showcase_items(items, "Qwen2.5-7B") == showcase  # deterministic
