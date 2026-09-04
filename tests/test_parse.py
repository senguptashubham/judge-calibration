"""Tests for src/parse.py. `split_cot_and_verdict_tokens()` tests moved here
from test_judge.py on 4 Sep 2026 when the function itself relocated to
parse.py (DECISIONS.md D4's amendment). The rest of task 1.5 - verdict/
verbalized_conf extraction and the parse_ok/parse_failure_type taxonomy,
tested against real malformed judge outputs collected in task 1.6 - is not
built yet. See TASKS.md task 1.5.
"""

from src.parse import split_cot_and_verdict_tokens


def test_split_cot_and_verdict_tokens_basic():
    # A well-formed JSON output, split into arbitrary token chunks that
    # concatenate back to the exact full text.
    full_text = '{"reasoning": "A is better because X.", "verdict": "A", "confidence": 0.9}'
    # Token boundaries deliberately don't align to word boundaries, to
    # exercise the character-offset-to-token mapping honestly.
    token_texts = [
        '{"reason', 'ing": "', 'A is bet', 'ter beca', 'use X.",',
        ' "verdi', 'ct": "', 'A', '", "confidence": 0.9}',
    ]
    assert "".join(token_texts) == full_text

    cot_indices, verdict_idx = split_cot_and_verdict_tokens(token_texts, full_text)

    # The verdict token must be the one whose text is exactly "A" - not a
    # neighboring chunk that merely contains "A" as a substring.
    assert token_texts[verdict_idx] == "A"
    # CoT tokens must all fall strictly before the "verdict" key starts.
    verdict_key_start = full_text.index('"verdict"')
    offsets = [0]
    for t in token_texts:
        offsets.append(offsets[-1] + len(t))
    for i in cot_indices:
        assert offsets[i] < verdict_key_start


def test_split_cot_and_verdict_tokens_falls_back_gracefully_on_malformed_text():
    # No '"verdict": "' substring at all (e.g. truncated mid-reasoning) -
    # must not raise, must fall back to "everything is CoT, no verdict token".
    token_texts = ['{"reasoning": "still writ', "ing when it got cut off"]
    full_text = "".join(token_texts)
    cot_indices, verdict_idx = split_cot_and_verdict_tokens(token_texts, full_text)
    assert cot_indices == [0, 1]
    assert verdict_idx is None
