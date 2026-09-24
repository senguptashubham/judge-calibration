"""Judge prompt templates P1 (general MT-Bench rubric, primary), P2
(correctness-first), P3 (helpfulness-first) - D19. FROZEN (invariant 10):
editing any template text changes its hash and requires a full re-run.

Output contract, shared by all three: one JSON object with exactly
`reasoning`, `verdict` ("A"|"B"), `confidence` (0-1), in that order. JSON
rather than a plain "Verdict: A" marker because of how guided decoding
works: a regex like `.*\\nVerdict: (A|B)` has a `.*` that can swallow the
marker itself, so the grammar never forces the model to reach it. A JSON
string's closing quote is unambiguous, so once `reasoning` closes, the
grammar forces the rest. What remains is the model's own choice of when to
stop reasoning - the `truncated` failure type, kept rare by asking for
2-4 sentences.

Turn handling: the dataset stores each side's conversation so far (2
messages for turn=1, 4 for turn=2), and a follow-up like "rewrite your
previous response" is unjudgeable alone, so turn=2 renders the whole
transcript.
"""

import hashlib

VARIANTS = ("P1", "P2", "P3")

_INTRO = (
    "You are an impartial judge evaluating which of two AI assistants gave "
    "the better response(s) in a conversation with a user. Read the full "
    "conversation for each assistant below, then "
)

# The one sentence that differs across variants (D19), so the ensemble
# isolates the rubric's framing, not incidental wording.
_RUBRIC_SENTENCES = {
    "P1": (
        "decide which assistant performed better overall, considering "
        "helpfulness, relevance, accuracy, depth, and level of detail. "
        "This is the general MT-Bench pairwise rubric (Zheng et al.)."
    ),
    "P2": (
        "decide which assistant's response(s) are more factually accurate "
        "and correctly complete the user's request. Treat helpfulness and "
        "detail only as a tiebreaker if both are equally correct."
    ),
    "P3": (
        "decide which assistant's response(s) would be more useful and "
        "satisfying to the user who asked, given what they actually seem "
        "to want. Treat factual precision only as a tiebreaker if both are "
        "equally helpful."
    ),
}

_PREAMBLES = {
    variant: _INTRO + sentence for variant, sentence in _RUBRIC_SENTENCES.items()
}

_TASK_INSTRUCTIONS = """
[Your task]
First, briefly explain your reasoning by comparing the two assistants' \
responses. Then give your verdict and how confident you are in it, as a \
single JSON object with exactly these three keys, in this order:

{"reasoning": "<your reasoning, 2-4 sentences>", "verdict": "A" or "B", \
"confidence": <a number between 0 and 1>}

Respond with only that JSON object."""


def _render_conversation(label: str, conversation: list[dict], turn: int) -> str:
    """One side's transcript up to and including `turn`: [user, assistant]
    for turn=1, [user, assistant, user, assistant] for turn=2.
    """
    lines = [f"[Conversation with Assistant {label}]"]
    lines.append(f"User: {conversation[0]['content']}")
    lines.append(f"Assistant {label}: {conversation[1]['content']}")
    if turn == 2:
        lines.append(f"User: {conversation[2]['content']}")
        lines.append(f"Assistant {label}: {conversation[3]['content']}")
    return "\n".join(lines)


def apply_order(
    order: str, model_a_conversation: list[dict], model_b_conversation: list[dict]
) -> tuple[list[dict], list[dict]]:
    """(model_a, model_b) -> (shown as Assistant A, shown as Assistant B).
    "AB" is the identity; "BA" swaps them - the position-bias probe (D5).
    """
    if order == "AB":
        return model_a_conversation, model_b_conversation
    if order == "BA":
        return model_b_conversation, model_a_conversation
    raise ValueError(f"order must be 'AB' or 'BA', got {order!r}")


def render_prompt(
    variant: str,
    order: str,
    model_a_conversation: list[dict],
    model_b_conversation: list[dict],
    turn: int,
) -> str:
    """The full judge prompt for one call. The conversations are the
    dataset's conversation_a/conversation_b; `order` decides which is shown
    as Assistant A.
    """
    if variant not in _PREAMBLES:
        raise ValueError(f"unknown prompt variant {variant!r} - must be one of {VARIANTS}")
    if turn not in (1, 2):
        raise ValueError(f"turn must be 1 or 2, got {turn!r}")

    displayed_a, displayed_b = apply_order(order, model_a_conversation, model_b_conversation)
    block_a = _render_conversation("A", displayed_a, turn)
    block_b = _render_conversation("B", displayed_b, turn)
    return f"{_PREAMBLES[variant]}\n\n{block_a}\n\n{block_b}\n{_TASK_INSTRUCTIONS}"


def prompt_hash(variant: str) -> str:
    """Hash of a variant's fixed text (preamble + task instructions),
    independent of item content - it changes only if the template does.
    """
    if variant not in _PREAMBLES:
        raise ValueError(f"unknown prompt variant {variant!r} - must be one of {VARIANTS}")
    fixed_text = _PREAMBLES[variant] + _TASK_INSTRUCTIONS
    return hashlib.sha256(fixed_text.encode("utf-8")).hexdigest()[:16]
