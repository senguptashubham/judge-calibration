"""Tests for src/parse.py. See TASKS.md task 1.5.

split_cot_and_verdict_tokens() tests moved here from test_judge.py on
4 Sep 2026 when the function relocated to parse.py (DECISIONS.md D4's
amendment). Real-malformed-output fixtures from task 1.6 still to be
added once the pilot run exists - the tests below use hand-constructed
cases covering CLAUDE.md §3's full failure taxonomy.
"""

import numpy as np
import pytest

from src.parse import (
    compute_logprob_signals,
    load_logprobs_record,
    parse_verdict_and_confidence,
    split_cot_and_verdict_tokens,
)


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


# --- parse_verdict_and_confidence ---------------------------------------


def test_parse_verdict_and_confidence_success():
    raw = '{"reasoning": "A is better.", "verdict": "A", "confidence": 0.87}'
    assert parse_verdict_and_confidence(raw) == {
        "parse_ok": True,
        "parse_failure_type": "none",
        "verdict": "A",
        "verbalized_conf": 0.87,
    }


def test_parse_verdict_and_confidence_missing_verdict_key():
    raw = '{"reasoning": "no verdict field here", "confidence": 0.9}'
    result = parse_verdict_and_confidence(raw)
    assert result["parse_ok"] is False
    assert result["parse_failure_type"] == "no_verdict"
    assert result["verdict"] is None


def test_parse_verdict_and_confidence_invalid_verdict_value():
    # Structured decoding should prevent this, but parse.py must not trust
    # that blindly - a value outside {"A","B"} is still no_verdict.
    raw = '{"reasoning": "...", "verdict": "C", "confidence": 0.9}'
    assert parse_verdict_and_confidence(raw)["parse_failure_type"] == "no_verdict"


def test_parse_verdict_and_confidence_missing_confidence_still_keeps_verdict():
    # verdict and verbalized_conf are independently nullable columns
    # (CLAUDE.md schema) - a confidence failure shouldn't discard a
    # perfectly good verdict.
    raw = '{"reasoning": "...", "verdict": "B"}'
    result = parse_verdict_and_confidence(raw)
    assert result["parse_ok"] is False
    assert result["parse_failure_type"] == "no_confidence"
    assert result["verdict"] == "B"
    assert result["verbalized_conf"] is None


def test_parse_verdict_and_confidence_confidence_out_of_range():
    raw = '{"reasoning": "...", "verdict": "A", "confidence": 1.5}'
    assert parse_verdict_and_confidence(raw)["parse_failure_type"] == "no_confidence"


def test_parse_verdict_and_confidence_confidence_non_numeric():
    raw = '{"reasoning": "...", "verdict": "A", "confidence": "high"}'
    assert parse_verdict_and_confidence(raw)["parse_failure_type"] == "no_confidence"


def test_parse_verdict_and_confidence_rejects_boolean_confidence():
    # bool is a subclass of int in Python - True would satisfy
    # 0 <= True <= 1 unless explicitly excluded.
    raw = '{"reasoning": "...", "verdict": "A", "confidence": true}'
    assert parse_verdict_and_confidence(raw)["parse_failure_type"] == "no_confidence"


def test_parse_verdict_and_confidence_truncated():
    raw = '{"reasoning": "still writing when it got cut off'
    result = parse_verdict_and_confidence(raw)
    assert result["parse_ok"] is False
    assert result["parse_failure_type"] == "truncated"
    assert result["verdict"] is None


def test_parse_verdict_and_confidence_malformed_json():
    # Ends with "}" (so it isn't classified truncated) but has a trailing
    # comma - invalid JSON syntax.
    raw = '{"reasoning": "text", "verdict": "A", "confidence": 0.9,}'
    result = parse_verdict_and_confidence(raw)
    assert result["parse_ok"] is False
    assert result["parse_failure_type"] == "malformed_json"


# --- compute_logprob_signals ---------------------------------------------


def test_compute_logprob_signals_end_to_end():
    raw_output = '{"reasoning": "ok", "verdict": "A", "confidence": 0.9}'
    token_texts = list(raw_output)  # one token per character - keeps the
    token_ids = list(range(len(token_texts)))  # offset math trivially exact

    expected_cot_indices, expected_verdict_idx = split_cot_and_verdict_tokens(token_texts, raw_output)
    assert token_texts[expected_verdict_idx] == "A"

    token_logprobs = []
    for i, ch in enumerate(token_texts):
        if i == expected_verdict_idx:
            # Chosen token is "A" (p=0.7); "B" (p=0.3) also visible in the
            # top-K, needed for p_a. Already sums to 1, so p_a == 0.7 exactly.
            token_logprobs.append(
                {
                    str(token_ids[i]): {"logprob": float(np.log(0.7)), "decoded_token": "A"},
                    "9999": {"logprob": float(np.log(0.3)), "decoded_token": "B"},
                }
            )
        else:
            # Every non-verdict position has exactly one candidate - a
            # single-candidate distribution has zero entropy after
            # renormalization, and a constant logprob, so mean/min/std/p10
            # are all trivially exact.
            token_logprobs.append({str(token_ids[i]): {"logprob": -0.01, "decoded_token": ch}})

    result = compute_logprob_signals(raw_output, token_ids, token_texts, token_logprobs)

    assert result["verdict_token_logprob"] == pytest.approx(np.log(0.7))
    assert result["p_a"] == pytest.approx(0.7)
    assert result["n_cot_tokens"] == len(expected_cot_indices)
    assert result["cot_logprob_mean"] == pytest.approx(-0.01)
    assert result["cot_logprob_min"] == pytest.approx(-0.01)
    assert result["cot_logprob_std"] == pytest.approx(0.0)
    assert result["cot_logprob_p10"] == pytest.approx(-0.01)
    assert result["cot_entropy_mean"] == pytest.approx(0.0)


def test_compute_logprob_signals_p_a_none_when_b_not_in_topk():
    # If "B" never appears among the verdict position's saved candidates
    # (e.g. an extremely confident model), p_a must be None, not a
    # fabricated value - verdict_token_logprob is still populated.
    raw_output = '{"reasoning": "ok", "verdict": "A", "confidence": 0.9}'
    token_texts = list(raw_output)
    token_ids = list(range(len(token_texts)))
    _, verdict_idx = split_cot_and_verdict_tokens(token_texts, raw_output)

    token_logprobs = [{str(token_ids[i]): {"logprob": -0.01, "decoded_token": ch}} for i, ch in enumerate(token_texts)]
    token_logprobs[verdict_idx] = {str(token_ids[verdict_idx]): {"logprob": -0.001, "decoded_token": "A"}}

    result = compute_logprob_signals(raw_output, token_ids, token_texts, token_logprobs)
    assert result["verdict_token_logprob"] == pytest.approx(-0.001)
    assert result["p_a"] is None


def test_compute_logprob_signals_handles_no_identifiable_verdict():
    # Malformed/truncated raw_output with no '"verdict": "' substring -
    # verdict_token_logprob/p_a must be None, not raise.
    raw_output = '{"reasoning": "still writing when cut off'
    token_texts = list(raw_output)
    token_ids = list(range(len(token_texts)))
    token_logprobs = [{str(token_ids[i]): {"logprob": -0.01, "decoded_token": ch}} for i, ch in enumerate(token_texts)]

    result = compute_logprob_signals(raw_output, token_ids, token_texts, token_logprobs)
    assert result["verdict_token_logprob"] is None
    assert result["p_a"] is None
    assert result["n_cot_tokens"] == len(token_texts)


# --- load_logprobs_record: integration with judge.py's writer ------------


def test_load_logprobs_record_matches_judge_write_logprobs_contract(tmp_path):
    # Proves the two modules actually agree on the saved shape, not just
    # each tested in isolation against its own assumptions.
    from src.judge import write_logprobs

    class _FakeLogprob:
        def __init__(self, logprob, decoded_token):
            self.logprob = logprob
            self.decoded_token = decoded_token

    path = tmp_path / "logprobs" / "call.jsonl.gz"
    token_ids = [101, 303]
    per_token_logprobs = [
        {101: _FakeLogprob(-0.1, "A"), 202: _FakeLogprob(-2.3, "B")},
        {303: _FakeLogprob(-0.05, "!")},
    ]
    write_logprobs(path, token_ids, per_token_logprobs)

    record = load_logprobs_record(path)
    assert record["token_ids"] == [101, 303]
    assert record["token_texts"] == ["A", "!"]
    assert record["token_logprobs"][0]["101"]["decoded_token"] == "A"
    assert record["token_logprobs"][0]["202"]["decoded_token"] == "B"

    # And compute_logprob_signals can actually consume it directly.
    result = compute_logprob_signals(
        "A!", record["token_ids"], record["token_texts"], record["token_logprobs"]
    )
    assert result["n_cot_tokens"] == 2  # no '"reasoning": "' present -> falls back to "everything is CoT"
