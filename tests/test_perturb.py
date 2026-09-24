"""Property tests for src/perturb.py: verbose_pad preserves the original
content, and the vacuum helpers build the degenerate pairs they claim to.
"""

from src.perturb import vacuum_empty, vacuum_identical, verbose_pad

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


# --- verbose_pad -----------------------------------------------------------


def test_verbose_pad_preserves_original_content_as_a_prefix():
    # The verdict-relevant content must still be there verbatim, not
    # rewritten or truncated - the attack works by outnumbering it with
    # redundant restatement, not by removing anything.
    padded = verbose_pad(_CONV)
    assert padded[1]["content"].startswith("Paris.")
    assert padded[3]["content"].startswith("Berlin.")


def test_verbose_pad_only_pads_assistant_turns():
    padded = verbose_pad(_CONV)
    assert padded[0]["content"] == "What is the capital of France?"
    assert padded[2]["content"] == "And of Germany?"


def test_verbose_pad_strictly_lengthens_assistant_turns():
    padded = verbose_pad(_CONV)
    assert len(padded[1]["content"]) > len("Paris.")
    assert len(padded[3]["content"]) > len("Berlin.")


def test_verbose_pad_preserves_roles_and_length():
    padded = verbose_pad(_CONV)
    assert [m["role"] for m in padded] == [m["role"] for m in _CONV]
    assert len(padded) == len(_CONV)


def test_verbose_pad_does_not_mutate_the_original():
    verbose_pad(_CONV)
    assert _CONV[1]["content"] == "Paris."
    assert _CONV[3]["content"] == "Berlin."


def test_verbose_pad_n_repeats_controls_padding_amount():
    # More repeats -> strictly more padding - proves the list is actually
    # being repeated n_repeats times, not just appended once regardless.
    small = verbose_pad(_CONV, n_repeats=1)
    large = verbose_pad(_CONV, n_repeats=5)
    assert len(large[1]["content"]) > len(small[1]["content"])


def test_verbose_pad_leaves_empty_content_unpadded():
    # Nothing to restate - must not fabricate a numbered list out of
    # nothing.
    conv = [{"role": "assistant", "content": ""}]
    padded = verbose_pad(conv)
    assert padded[0]["content"] == ""


def test_verbose_pad_pads_content_without_terminal_punctuation():
    # No '.', '!', or '?' to split on still counts as one sentence - the
    # splitter treats the whole string as a single unit rather than
    # discarding it, so a one-word reply like "Paris" still gets padded.
    conv = [{"role": "assistant", "content": "Paris"}]
    padded = verbose_pad(conv)
    assert padded[0]["content"].startswith("Paris")
    assert len(padded[0]["content"]) > len("Paris")
