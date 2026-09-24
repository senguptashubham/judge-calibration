"""Input perturbations: verbose_pad (RQ3's verbosity attack) and the two
vacuum inputs (task 1.8). Order is not a perturbation - it lives in
prompts.py (D5). attribution() is cut (D18).
"""

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def vacuum_identical(conversation: list[dict]) -> list[dict]:
    """A copy of `conversation`, to be used as BOTH sides of a pair - there
    is no content difference to judge, so order doesn't matter either.
    """
    return [dict(msg) for msg in conversation]


def verbose_pad(conversation: list[dict], n_repeats: int = 3) -> list[dict]:
    """Zheng et al. §3.3's "repetitive list" verbosity attack (it fools
    Claude-v1/GPT-3.5 91.3% of the time, GPT-4 8.7%). Each assistant turn
    is kept verbatim, then followed by its own sentences restated as a
    numbered list, `n_repeats` times - length without information. User
    turns are untouched; an empty assistant turn is left as is.
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
    """`conversation` with every assistant turn emptied, user turns kept -
    a real question with a blank answer, used as both sides of a pair.
    """
    return [
        {**msg, "content": ""} if msg["role"] == "assistant" else dict(msg)
        for msg in conversation
    ]
