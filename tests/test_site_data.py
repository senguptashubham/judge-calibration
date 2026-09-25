"""Tests for analysis/site_data.py's item pickers: the game pool and the random
"Trick the judge" examples. The exported file as a whole is checked against
the real data when it is built."""

from analysis.site_data import pick_game_items, pick_random_examples


def _call(winner):
    return {"status": "ok", "pick": "first", "winner": winner, "conf": 0.9, "text": "reasoning " * 20}


def _item(item_id, question_id, *, turn=1, agreed=True, answer_len=100, same_answers=False, winner="A"):
    calls = {cond: {order: _call(winner) for order in ("AB", "BA")} for cond in ("clean", "verbose")}
    return {
        "item_id": item_id,
        "question_id": question_id,
        "category": "math",
        "turn": turn,
        "model_a": "m1",
        "model_b": "m2",
        "conversation_a": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a" * answer_len}],
        "conversation_b": [{"role": "user", "content": "q"},
                           {"role": "assistant", "content": "a" * answer_len if same_answers else "b" * answer_len}],
        "human": {"label": "A", "n_votes": 2, "frac_prefer_a": 1.0, "agreed": agreed},
        "judges": {judge: calls for judge in ("Qwen2.5-7B", "kev-8b", "auto-j-13b")},
    }


ITEMS = [
    _item("q1-a", 1), _item("q1-b", 1),                  # two eligible on one question
    _item("q2-a", 2), _item("q2-b", 2, agreed=False),    # a single-vote item is never eligible
    _item("q3-a", 3, turn=2),                             # turn 2: out of the game pool
    _item("q4-a", 4, same_answers=True),                  # identical answers: nothing to pick between
    _item("q5-a", 5, answer_len=5000),                    # too long to read in the game
]


def test_game_pool_takes_one_eligible_item_per_question_without_reasoning_text():
    pool = pick_game_items(ITEMS, seed=0)
    assert sorted(it["question_id"] for it in pool) == [1, 2]
    assert next(it for it in pool if it["question_id"] == 2)["item_id"] == "q2-a"
    call = pool[0]["judges"]["Qwen2.5-7B"]["clean"]["AB"]
    assert "text" not in call and call["winner"] == "A"


def test_game_pool_is_deterministic_and_blind_to_the_judges_verdicts():
    """Flipping every verdict must not change which items are picked: the game
    compares you with the judges, so the pool can't favour either side."""
    flipped = [_item(it["item_id"], it["question_id"], turn=it["turn"], agreed=it["human"]["agreed"],
                     answer_len=len(it["conversation_a"][1]["content"]),
                     same_answers=it["conversation_a"] == it["conversation_b"], winner="B") for it in ITEMS]
    ids = lambda pool: [it["item_id"] for it in pool]
    assert ids(pick_game_items(ITEMS, seed=7)) == ids(pick_game_items(ITEMS, seed=7))
    assert ids(pick_game_items(ITEMS, seed=7)) == ids(pick_game_items(flipped, seed=7))


def test_random_examples_respect_exclusions_size_cap_and_count():
    # q1-b is excluded and q5-a is over the cap, so q1 offers q1-a only and q5 nothing.
    examples = pick_random_examples(ITEMS, exclude={"q1-b"}, seed=0, n=10, max_bytes=8000)
    by_question = {ex["question_id"]: ex["item_id"] for ex in examples}
    assert by_question == {1: "q1-a", 2: "q2-a", 3: "q3-a"}
    assert all(ex["kind"] == "random" and "padded_a" in ex for ex in examples)
    assert len(pick_random_examples(ITEMS, exclude=set(), seed=0, n=2, max_bytes=8000)) == 2
