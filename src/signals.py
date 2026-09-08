"""conf_verb / conf_lp / conf_sc / conf_bpe (task 1.7), plus conf_ens and
its judge-level Total/Aleatoric/Epistemic entropy decomposition over the
P1/P2/P3 prompt ensemble - clean condition only (DECISIONS.md D20, task 2.2b).

Operates on already-parsed calls - one dict per call, carrying `order`,
`sample_idx`, `verdict`, `verbalized_conf`, `p_a`, `verdict_token_logprob`
(src/parse.py's output already merged in, matching calls.parquet's real
schema - CLAUDE.md invariant 7: parsing itself lives only in parse.py,
this file never touches `raw_output`). All functions here take `rows`:
every call belonging to one (item_id, condition, prompt_variant) group.
"""

import math


def _find_call(rows: list[dict], order: str, sample_idx: int) -> dict | None:
    """Locates one specific call among an item's rows, by order + sample_idx."""
    for row in rows:
        if row["order"] == order and row["sample_idx"] == sample_idx:
            return row
    return None


def judge_verdict(rows: list[dict]) -> str | None:
    """D7 primary: the canonical greedy verdict, AB order specifically.

    AB is `apply_order()`'s identity mapping (prompts.py) - "Assistant A"
    in that rendering already IS model_a, so this raw verdict needs no
    translation to be read as "which model won." This is exactly why the
    primary reference is AB, not BA: it's the one order where the
    judge's own label and the canonical model_a/model_b identity coincide.
    D7: "what a real deployment running one pass would get."
    """
    call = _find_call(rows, "AB", 0)
    return call["verdict"] if call else None


def _p_model_a_wins(rows: list[dict]) -> float | None:
    """Averages both greedy orders' `p_a` onto a single, consistent
    "P(model_a wins)" scale before averaging - not a raw average of the
    two `p_a` values, which would silently blend two different questions.

    `p_a` is P(displayed-A), and "displayed-A" means a different physical
    model depending on order: under AB, displayed-A = model_a, so
    p_a(AB) IS P(model_a wins) directly. Under BA, apply_order() swaps
    which model is shown as A, so displayed-A = model_b, and therefore
    P(model_a wins | BA) = 1 - p_a(BA). Both terms below are now on the
    same "does the judge favor model_a" scale before being combined.
    """
    call_ab = _find_call(rows, "AB", 0)
    call_ba = _find_call(rows, "BA", 0)
    if call_ab is None or call_ba is None:
        return None
    if call_ab["p_a"] is None or call_ba["p_a"] is None:
        return None
    return (call_ab["p_a"] + (1 - call_ba["p_a"])) / 2


def verdict_bidir(rows: list[dict]) -> str | None:
    """D7 secondary: argmax of the order-corrected mean P(model_a wins)
    (see _p_model_a_wins). "A" here means model_a won, "B" means model_b
    won - already in canonical model identity, comparable to human_label
    the same way judge_verdict is.
    """
    p = _p_model_a_wins(rows)
    if p is None:
        return None
    return "A" if p >= 0.5 else "B"


def conf_verb(rows: list[dict]) -> float | None:
    """Verbalized confidence, from the canonical greedy AB call only (D6)."""
    call = _find_call(rows, "AB", 0)
    return call["verbalized_conf"] if call else None


def conf_lp(rows: list[dict]) -> float | None:
    """exp(verdict_token_logprob): the model's own confidence in whichever
    answer it actually gave - symmetric (always in [0.5, 1] for a binary
    forced choice), unlike p_a which is directional toward "A" specifically.

    From the canonical greedy AB call only (D6) - sampled draws
    (sample_idx > 0) run at temperature_sc, and REPORT.md's temperature-
    probe methods note (8 Sep 2026) confirmed empirically, not just
    theoretically, that temperature measurably shifts the reported
    logprobs even when the verdict itself doesn't change.
    """
    call = _find_call(rows, "AB", 0)
    if call is None or call["verdict_token_logprob"] is None:
        return None
    return math.exp(call["verdict_token_logprob"])


def conf_sc(rows: list[dict], k_sc: int) -> float | None:
    """Fraction of the k_sc sampled verdicts (sample_idx=1..k_sc, AB order
    only) matching the canonical greedy verdict. clean/P1 ONLY - sampling
    doesn't exist for any other (condition, prompt_variant) pair (D19,
    D21), so this returns None everywhere else, not a fabricated 0 or 1.
    """
    canonical = _find_call(rows, "AB", 0)
    if canonical is None:
        return None
    sampled = [_find_call(rows, "AB", i) for i in range(1, k_sc + 1)]
    sampled = [s for s in sampled if s is not None]
    if not sampled:
        return None
    matches = sum(1 for s in sampled if s["verdict"] == canonical["verdict"])
    return matches / len(sampled)


def conf_bpe(rows: list[dict]) -> float | None:
    """1 - H(p), where p is the order-corrected mean P(model_a wins) (see
    _p_model_a_wins) and H is binary entropy in nats (not normalized to
    [0,1] - kept on the same raw-nats scale as conf_ens's entropy
    decomposition, D20, task 2.2b, rather than each signal picking its
    own private normalization).

    This is entropy OF the averaged probability, not the average of each
    order's own entropy - a deliberately different, more informative
    quantity: if both orders agree strongly (both push p toward the same
    extreme), the average stays extreme and H stays low. If the orders
    DISAGREE (one order favors model_a, the other favors model_b after
    translation), the average washes toward 0.5 and H rises toward its
    max - directly capturing cross-order inconsistency, not just
    within-order sharpness. (SCOPE, 2026; LEARNING.md C11.)
    """
    p = _p_model_a_wins(rows)
    if p is None:
        return None
    if p <= 0.0 or p >= 1.0:
        entropy = 0.0
    else:
        entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    return 1 - entropy


def compute_item_signals(rows: list[dict], k_sc: int) -> dict:
    """Combines every signal above into one item-level record. `rows` is
    every call belonging to a single (item_id, condition, prompt_variant)
    group - the caller is responsible for that grouping (task 2.2's
    items.parquet-building CLI groups calls.parquet this way).
    """
    return {
        "judge_verdict": judge_verdict(rows),
        "verdict_bidir": verdict_bidir(rows),
        "conf_verb": conf_verb(rows),
        "conf_lp": conf_lp(rows),
        "conf_sc": conf_sc(rows, k_sc),
        "conf_bpe": conf_bpe(rows),
    }
