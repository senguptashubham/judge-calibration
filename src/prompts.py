"""Judge prompt templates: P1 (existing, primary), P2 (correctness-first
rubric), P3 (helpfulness-first rubric) - DECISIONS.md D19. Each versioned
and hashed independently. FROZEN after Week 1 (CLAUDE.md invariant 10).
See TASKS.md task 1.3.

Output contract (all three variants share it): a single JSON object with
exactly `reasoning` (free text), `verdict` ("A"|"B"), `confidence` (0-1),
in that key order. JSON was chosen over a plain-text "Verdict: A" marker
because of how guided/structured decoding actually behaves: a bare regex
like `.*\\nVerdict: (A|B)` has a `.*` state with a self-loop that swallows
every character - including the literal marker's own characters - so the
grammar never *forces* the model to reach for the marker, it only
constrains what happens once the model gets there. A JSON string value's
closing `"` is not ambiguous the same way (unescaped `"` has no alternate
reading as "more string content"), so the instant the model closes the
`reasoning` string the grammar hard-forces everything after it - the
`malformed_json` failure mode in CLAUDE.md's parse_failure_type taxonomy
already anticipated this design. Neither format eliminates the model's own
choice of *when* to stop reasoning and close the field - that residual
risk is the `truncated` failure type, mitigated by bounding reasoning
length in the prompt itself (see _TASK_INSTRUCTIONS) and measured directly
by task 1.6's pilot parse-failure-rate gate.

Turn handling: `mt_bench_human_judgments` already stores each side's full
conversation-so-far in `conversation_a`/`conversation_b` (2 messages for
turn=1, 4 for turn=2 - confirmed empirically against the live dataset, not
assumed) - so a turn=2 judgment needs the full transcript rendered, not
just the isolated final exchange, or a follow-up like "rewrite your
previous response" is unjudgeable in isolation. This is handled with a
plain Python `if`, not a template-engine conditional, per CLAUDE.md sec 5's
"boring, explicit code" preference.
"""

import hashlib

VARIANTS = ("P1", "P2", "P3")

_INTRO = (
    "You are an impartial judge evaluating which of two AI assistants gave "
    "the better response(s) in a conversation with a user. Read the full "
    "conversation for each assistant below, then "
)

# The one sentence that differs across variants (DECISIONS.md D19) - every
# other word of the prompt is shared, so the ensemble (RQ5) isolates the
# effect of the rubric framing itself, not incidental wording differences.
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
    """Renders one side's transcript up to and including `turn`.

    `conversation` is `conversation_a`/`conversation_b` straight from
    `load_votes()`'s underlying dataset: a list of {"role", "content"}
    dicts, [user, assistant] for turn=1 and [user, assistant, user,
    assistant] for turn=2. Indexing conversation[2]/[3] is guarded by the
    turn check - a turn=1 conversation only ever has 2 messages.
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
    """Maps (model_a, model_b) onto (displayed as Assistant A, displayed as
    Assistant B) for a given order. order="AB" is the identity mapping;
    "BA" swaps which physical model's conversation is labeled A vs B - this
    is the position-bias probe (DECISIONS.md D5), orthogonal to condition.
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
    """Renders the full judge prompt for one call.

    `model_a_conversation`/`model_b_conversation` are the dataset's own
    conversation_a/conversation_b (i.e. keyed to the item's actual
    model_a/model_b identity) - `order` decides which one is displayed as
    "Assistant A" in the rendered text, not the caller.
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
    """Stable hash of a variant's fixed instructional text (preamble +
    task instructions), independent of any item's content - two calls for
    the same variant always match, and it only changes if the template
    text itself is edited (CLAUDE.md invariant 10: that requires a new
    version string and a full re-run, never a silent edit).
    """
    if variant not in _PREAMBLES:
        raise ValueError(f"unknown prompt variant {variant!r} - must be one of {VARIANTS}")
    fixed_text = _PREAMBLES[variant] + _TASK_INSTRUCTIONS
    return hashlib.sha256(fixed_text.encode("utf-8")).hexdigest()[:16]
