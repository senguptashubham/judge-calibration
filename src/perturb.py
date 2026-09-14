"""verbose_pad / vacuum perturbations. Order lives in prompts.py, not here
(DECISIONS.md D5). attribution() is cut entirely, not built here -
DECISIONS.md D18, 31 Aug 2026. See TASKS.md task 4.1.

vacuum_identical()/vacuum_empty() are task 1.8's "true vacuum" probe
(Dark Current, LEARNING.md A8): does the judge express a preference when
there is genuinely no content difference to distinguish the two options?
"""

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def vacuum_identical(conversation: list[dict]) -> list[dict]:
    """A deep copy of `conversation`, for use as BOTH sides of a pairwise
    prompt. Since both sides end up byte-identical, order (AB vs BA)
    doesn't matter here - swapping which physical content is displayed as
    "A" changes nothing when both physical contents are the same, so only
    one call per pair is needed, not both orders.
    """
    return [dict(msg) for msg in conversation]


def verbose_pad(conversation: list[dict], n_repeats: int = 3) -> list[dict]:
    """Zheng et al. §3.3's "repetitive list" verbosity attack (LEARNING.md
    A2 - the one that fools Claude-v1/GPT-3.5 91.3% of the time on length
    alone, GPT-4 only 8.7%). Every assistant turn's content is kept
    verbatim, then followed by its own sentences restated as a numbered
    list, repeated `n_repeats` times - padding that adds length without
    adding information, which is the entire point of the attack. User
    turns are untouched. An empty assistant turn has no content to restate
    and is left unpadded rather than guessed at.
    """
    padded = []
    for msg in conversation:
        if msg["role"] != "assistant":
            padded.append(dict(msg))
            continue
        content = msg["content"]
        sentences = [s for s in _SENTENCE_SPLIT.split(content.strip()) if s]
        if not sentences:
            padded.append(dict(msg))
            continue
        restated = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
        filler = "\n\n".join([restated] * n_repeats)
        padded.append({**msg, "content": f"{content}\n\n{filler}"})
    return padded


def vacuum_empty(conversation: list[dict]) -> list[dict]:
    """`conversation` with every assistant turn's content replaced by an
    empty string; user turns unchanged. A more degenerate case than
    vacuum_identical: a real question, but a genuinely blank response on
    both sides (both sides use this same emptied conversation, same
    reasoning as vacuum_identical - order doesn't matter).
    """
    return [
        {**msg, "content": ""} if msg["role"] == "assistant" else dict(msg)
        for msg in conversation
    ]
