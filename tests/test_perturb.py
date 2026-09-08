"""Property tests for src/perturb.py: verbose_pad preserves verdict-relevant
content. (attribution() is cut, DECISIONS.md D18 - no attribution property
test.) See TASKS.md task 4.1.

vacuum_identical()/vacuum_empty() tests are task 1.8, not 4.1 - see
TASKS.md task 1.8.
"""

from src.perturb import vacuum_empty, vacuum_identical

_CONV = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And of Germany?"},
    {"role": "assistant", "content": "Berlin."},
]


def test_vacuum_identical_is_a_content_copy():
    copy = vacuum_identical(_CONV)
    assert copy == _CONV


def test_vacuum_identical_is_a_deep_copy_not_the_same_object():
    # Mutating the copy must not affect the original - this will be used
    # as BOTH sides of a pair, so accidental aliasing would be a real bug.
    copy = vacuum_identical(_CONV)
    copy[1]["content"] = "mutated"
    assert _CONV[1]["content"] == "Paris."


def test_vacuum_empty_blanks_only_assistant_turns():
    emptied = vacuum_empty(_CONV)
    assert emptied[0]["content"] == "What is the capital of France?"  # user turn unchanged
    assert emptied[1]["content"] == ""  # assistant turn blanked
    assert emptied[2]["content"] == "And of Germany?"  # user turn unchanged
    assert emptied[3]["content"] == ""  # assistant turn blanked


def test_vacuum_empty_preserves_roles_and_length():
    emptied = vacuum_empty(_CONV)
    assert [m["role"] for m in emptied] == [m["role"] for m in _CONV]
    assert len(emptied) == len(_CONV)


def test_vacuum_empty_does_not_mutate_the_original():
    vacuum_empty(_CONV)
    assert _CONV[1]["content"] == "Paris."
    assert _CONV[3]["content"] == "Berlin."
