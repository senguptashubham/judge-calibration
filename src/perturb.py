"""verbose_pad / vacuum perturbations. Order lives in prompts.py, not here
(DECISIONS.md D5). attribution() is cut entirely, not built here -
DECISIONS.md D18, 31 Aug 2026. See TASKS.md task 4.1.

vacuum_identical()/vacuum_empty() are task 1.8's "true vacuum" probe
(Dark Current, LEARNING.md A8): does the judge express a preference when
there is genuinely no content difference to distinguish the two options?
`verbose_pad()` itself is task 4.1 (W4), not built yet.
"""


def vacuum_identical(conversation: list[dict]) -> list[dict]:
    """A deep copy of `conversation`, for use as BOTH sides of a pairwise
    prompt. Since both sides end up byte-identical, order (AB vs BA)
    doesn't matter here - swapping which physical content is displayed as
    "A" changes nothing when both physical contents are the same, so only
    one call per pair is needed, not both orders.
    """
    return [dict(msg) for msg in conversation]


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
