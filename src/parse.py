"""Verdict and confidence extraction, failure taxonomy. See TASKS.md task 1.5.

`split_cot_and_verdict_tokens()` lives here (relocated from `judge.py` on
4 Sep 2026, alongside DECISIONS.md D4's amendment to 100% logprob coverage)
because it is itself parsing logic - locating which generated tokens are
the CoT (`"reasoning"` value) versus the verdict token within `raw_output` -
and CLAUDE.md invariant 7 says logic that touches raw model output belongs
only in this file. It's the one function built so far; the rest of task
1.5 (`verdict`/`verbalized_conf` extraction, the `parse_ok`/
`parse_failure_type` taxonomy, and the CoT-aggregate/`verdict_token_logprob`/
`p_a` computation that reads this function's output alongside the saved
per-token logprobs from `runs/logprobs/`) is still to be written.
"""


def split_cot_and_verdict_tokens(
    token_texts: list[str], full_text: str
) -> tuple[list[int], int | None]:
    """Locates which generated tokens fall inside the JSON `"reasoning"`
    string value (the CoT tokens) and which single token is the `"verdict"`
    value, using plain substring search on the assembled text rather than
    full JSON parsing - a truncated/malformed generation may not be valid
    JSON at all, and this should degrade gracefully (falls back to "every
    token is CoT, no identifiable verdict token") rather than raise.

    token_texts: each token's own decoded text piece, in generation order -
    concatenating them all must reconstruct full_text exactly.
    """
    reasoning_key = '"reasoning": "'
    verdict_key = '"verdict": "'

    reasoning_key_start = full_text.find(reasoning_key)
    if reasoning_key_start == -1:
        return list(range(len(token_texts))), None
    reasoning_value_start = reasoning_key_start + len(reasoning_key)

    verdict_key_start = full_text.find(verdict_key, reasoning_value_start)
    if verdict_key_start == -1:
        return list(range(len(token_texts))), None
    verdict_value_start = verdict_key_start + len(verdict_key)

    offsets = [0]
    for t in token_texts:
        offsets.append(offsets[-1] + len(t))

    def _token_at(char_pos: int) -> int:
        for i in range(len(token_texts)):
            if offsets[i] <= char_pos < offsets[i + 1]:
                return i
        return len(token_texts) - 1

    cot_start = _token_at(reasoning_value_start)
    cot_end = _token_at(verdict_key_start)  # exclusive
    cot_token_indices = list(range(cot_start, max(cot_end, cot_start)))
    verdict_token_index = _token_at(verdict_value_start)

    return cot_token_indices, verdict_token_index
