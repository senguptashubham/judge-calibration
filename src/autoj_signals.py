"""runs/autoj_13b_gptq_4bits/autoj.jsonl -> calls_autoj_13b_gptq_4bits.parquet
-> items_autoj_13b_gptq_4bits.parquet (TASKS.md L4, DECISIONS.md D28/RQ7).

Mirrors src/kev_signals.py's combined parse+signal role (one file, not a
separate parse.py + signals.py split) at auto-j's own grain: (item_id,
condition) only, no prompt_variant axis, but WITH sample_idx sampling
(unlike kev-8b - auto-j supports real autoregressive self-consistency
sampling, D28). Raw-output parsing happens exactly once, at
build_calls_autoj()'s own boundary (CLAUDE.md invariant 7's spirit - not
literally parse.py, but the same "touch raw_output in one place only"
discipline) - every signal function below reads the pre-computed
`pred_label` column, never `raw_output` itself again.

Reuses src.signals._binary_entropy and _sanitize_records directly, same
reuse kev_signals.py already established.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.judge_autoj import AutojConfig
from src.signals import _binary_entropy, _sanitize_records


def extract_autoj_verdict(raw_output: str) -> int:
    """Ported VERBATIM from auto-j's own codes/usage/example.py::
    extract_pariwise_result() (D28) - not reinvented, same precedent as
    using kev's own real choice_confidence formula rather than guessing
    one. Returns 0 (Response 1 wins), 1 (Response 2 wins), 2 (Tie), or -1
    (no parseable "final decision is ..." line found at all - e.g. a
    truncated completion that never reached one).
    """
    raw_output = raw_output.strip()
    pos = raw_output.rfind("final decision is ")
    pred_label = -1
    if pos != -1:
        pred_rest = raw_output[pos + len("final decision is ") :].strip().lower()
        if pred_rest.startswith("response 1"):
            pred_label = 0
        elif pred_rest.startswith("response 2"):
            pred_label = 1
        elif pred_rest.startswith("tie"):
            pred_label = 2
    return pred_label


def build_calls_autoj(checkpoint_path: str | Path) -> pd.DataFrame:
    """runs/autoj_13b_gptq_4bits/autoj.jsonl -> calls_autoj_13b_gptq_4bits.parquet's
    DataFrame. Computes `pred_label` here, once, from `raw_output` - the
    one and only place this module touches raw model output. Skipped
    (over-length) rows have no `raw_output` at all, so `pred_label` is
    None for them, never a fabricated -1 (that value means "the model DID
    respond, but no decision line was found" - a genuinely different
    situation from "this call was never attempted").
    """
    rows = []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            skipped = row.get("skipped", False)
            raw_output = row.get("raw_output")
            pred_label = extract_autoj_verdict(raw_output) if (not skipped and raw_output is not None) else None
            rows.append(
                {
                    "item_id": row["item_id"],
                    "question_id": row["question_id"],
                    "category": row.get("category"),
                    "model_a": row["model_a"],
                    "model_b": row["model_b"],
                    "turn": row["turn"],
                    "condition": row["condition"],
                    "order": row["order"],
                    "sample_idx": row["sample_idx"],
                    "judge_model": row["judge_model"],
                    "skipped": skipped,
                    "skip_reason": row.get("skip_reason"),
                    "finish_reason": row.get("finish_reason"),
                    "pred_label": pred_label,
                    "n_prompt_tokens": row.get("n_prompt_tokens"),
                    "n_out_tokens": row.get("n_out_tokens"),
                }
            )
    return pd.DataFrame.from_records(rows)


def _find_call(rows: list[dict], order: str, sample_idx: int) -> dict | None:
    """Locates one specific call among an item-condition's rows, by order
    + sample_idx - excludes skipped (over-length) calls, same "skipped
    calls return None" treatment kev_signals.py::_find_kev_call already
    established (auto-j has no separate network-failure case the way
    kev's `ok` flag guarded against - a non-skipped call always produced
    SOME raw_output, so `skipped` alone is the right success gate here).
    """
    for row in rows:
        if row["order"] == order and row["sample_idx"] == sample_idx and not row.get("skipped"):
            return row
    return None


def _canonical_letter(pred_label: int | None, order: str) -> str | None:
    """Maps auto-j's raw 0/1/2/-1 label, plus which order the call was
    rendered in, onto the project's canonical A|B|None convention
    (model_a = "A", model_b = "B"). Under AB order, apply_order() is the
    identity mapping (Response 1 = model_a), so label 0->A, 1->B directly;
    under BA, apply_order() swaps which model is displayed first
    (Response 1 = model_b), so the mapping flips: 0->B, 1->A - the same
    translation _p_model_a_wins()/_p_model_a_wins_kev() apply to p_a,
    applied here to a discrete label instead of a continuous probability.

    Tie (2) and parse failure (-1) both return None - neither is a usable
    binary verdict for comparison against human_label. This is a
    deliberate decision (D28, confirmed 24 Sep 2026), not an oversight:
    Tie is tracked separately via calls_autoj_13b_gptq_4bits.parquet's own
    `pred_label` column (2, distinguishable from -1), so it is never
    silently conflated with a genuine parse failure in the report's
    methods notes, even though both collapse to the same `None` here.
    """
    if pred_label == 0:
        return "A" if order == "AB" else "B"
    if pred_label == 1:
        return "B" if order == "AB" else "A"
    return None


def judge_verdict_autoj(rows: list[dict]) -> str | None:
    """D7-style primary: the canonical greedy AB call's own verdict,
    translated to canonical model identity. Mirrors judge_verdict()/
    judge_verdict_kev() exactly.
    """
    call = _find_call(rows, "AB", 0)
    if call is None:
        return None
    return _canonical_letter(call["pred_label"], "AB")


def _canonical_verdict_ba_autoj(rows: list[dict]) -> str | None:
    """The BA-order call's own verdict, translated into canonical
    model_a/model_b identity. Mirrors _canonical_verdict_ba()/
    _canonical_verdict_ba_kev().
    """
    call = _find_call(rows, "BA", 0)
    if call is None:
        return None
    return _canonical_letter(call["pred_label"], "BA")


def flipped_autoj(rows: list[dict]) -> bool | None:
    """Canonical verdict differs between AB and BA order - the position-
    swap flip-rate test's own signal (RQ7's battery, mirrors RQ3a's
    recipe). Needs only the two greedy calls, so this is computable on
    BOTH conditions (no self-consistency dependency at all, unlike
    _p_model_a_wins_autoj below) - same as flipped()/flipped_kev().
    """
    ab = judge_verdict_autoj(rows)
    ba = _canonical_verdict_ba_autoj(rows)
    if ab is None or ba is None:
        return None
    return ab != ba


def _self_consistency_proportion_a(rows: list[dict], order: str, k_sc: int) -> float | None:
    """Fraction of `order`'s available calls (greedy sample_idx=0, plus any
    sampled draws 1..k_sc that exist) whose canonical-translated verdict
    favors model_a. Tie/parse-failure calls are excluded from both
    numerator and denominator - neither is evidence for or against
    model_a specifically.

    For AB on `clean` (which gets k_sc real sampled draws, D19's own
    schedule reused unchanged), this is a genuine proportion over up to
    k_sc+1 calls. For BA (which never gets sampled draws - only
    sample_idx=0 exists, the same schedule asymmetry D6/D19 already
    established for the primary judge) and for AB on `verbose` (which
    also never gets sampled draws), it degenerates to the single greedy
    call's own 0/1 verdict - not a bug, just what a "self-consistency
    proportion" reduces to when there is only one draw to measure (D28,
    confirmed 24 Sep 2026: this degenerate case is what lets
    conf_sc_bpe_autoj stay defined on `verbose` at all, at coarser
    resolution, rather than being undefined there like conf_sc_autoj is).
    """
    calls = [_find_call(rows, order, i) for i in range(0, k_sc + 1)]
    calls = [c for c in calls if c is not None]
    if not calls:
        return None
    letters = [_canonical_letter(c["pred_label"], order) for c in calls]
    letters = [l for l in letters if l is not None]
    if not letters:
        return None
    return sum(1 for l in letters if l == "A") / len(letters)


def _p_model_a_wins_autoj(rows: list[dict], k_sc: int) -> float | None:
    """Order-corrected mean P(model_a wins), built from self-consistency
    PROPORTIONS rather than a per-call logprob ratio (D28's signal-scope
    decision - auto-j has no schema-constrained decoding to guarantee a
    clean two-candidate token position the way conf_lp/conf_bpe rely on
    for the primary judge).

    Deliberately a PLAIN average here, `(p_ab + p_ba) / 2` - NOT
    `(p_ab + (1 - p_ba)) / 2`, the formula _p_model_a_wins()/
    _p_model_a_wins_kev() use. Those two work on an UNTRANSLATED quantity
    (`p_a`/`prob_a` = P(displayed-A wins), a position-relative statement
    that means a different physical model depending on order), so their
    BA term needs the `1 -` flip to become comparable to AB's. This
    function's own inputs are different in kind: _self_consistency_
    proportion_a() already applies the canonical A/B translation
    internally (via _canonical_letter), so p_ab and p_ba arrive already
    on the SAME "P(model_a wins)" scale - applying `(1 - p_ba)` again
    would double-translate.

    Caught via a hand-computed pure-position-bias case, not assumed:
    if "whichever response is displayed first wins" (no real model
    preference, pure position bias), AB's single greedy call has model_a
    displayed first and wins -> p_ab=1.0; BA's has model_b displayed
    first and wins, so model_a loses -> p_ba=0.0 (already correctly
    reflecting "model_a does NOT win under BA", not needing a flip).
    `(p_ab + p_ba)/2 = 0.5` (indifference, the right answer for pure
    position bias); the borrowed `(p_ab + (1-p_ba))/2` formula would have
    given 1.0 instead - confirming the plain average, not the flipped
    one, is correct for this function's own already-translated inputs.
    """
    p_ab = _self_consistency_proportion_a(rows, "AB", k_sc)
    p_ba = _self_consistency_proportion_a(rows, "BA", k_sc)
    if p_ab is None or p_ba is None:
        return None
    return (p_ab + p_ba) / 2


def verdict_bidir_autoj(rows: list[dict], k_sc: int) -> str | None:
    """D7-style secondary: argmax of the order-corrected mean P(model_a
    wins) (see _p_model_a_wins_autoj). Computable on BOTH conditions (see
    that function's own docstring for why `verbose` degenerates rather
    than being undefined).
    """
    p = _p_model_a_wins_autoj(rows, k_sc)
    if p is None:
        return None
    return "A" if p >= 0.5 else "B"


def conf_sc_autoj(rows: list[dict], k_sc: int) -> float | None:
    """Fraction of the k_sc sampled AB-order verdicts matching the
    canonical greedy AB verdict - direct, unmodified port of D6's formula
    (conf_sc). No order translation needed here (unlike judge_verdict_autoj/
    _p_model_a_wins_autoj): every call being compared is already in the
    same AB reference frame, so raw `pred_label` equality is exactly the
    right comparison - Tie==Tie or failure==failure both count as a
    "consistent" outcome too, matching what self-consistency actually
    means (does repeated sampling reach the same conclusion, whatever
    that conclusion is), the same convention D6's own conf_sc uses for
    the primary judge.

    `clean` ONLY - `verbose` never collects sampled draws (D19's schedule,
    reused unchanged for auto-j), so this returns None there, never a
    fabricated 0 or 1. Not an explicit condition check: `sampled` is
    naturally empty because those rows were never generated at all.
    """
    canonical = _find_call(rows, "AB", 0)
    if canonical is None:
        return None
    sampled = [_find_call(rows, "AB", i) for i in range(1, k_sc + 1)]
    sampled = [s for s in sampled if s is not None]
    if not sampled:
        return None
    matches = sum(1 for s in sampled if s["pred_label"] == canonical["pred_label"])
    return matches / len(sampled)


def conf_sc_bpe_autoj(rows: list[dict], k_sc: int) -> float | None:
    """1 - H(p), p = the order-corrected P(model_a wins) built from self-
    consistency proportions (_p_model_a_wins_autoj) - the order-swap
    bidirectional-entropy analog. Deliberately NOT named conf_bpe_autoj,
    to avoid implying false equivalence to the primary judge's logprob-
    based conf_bpe (D28).

    Computed on BOTH conditions (confirmed 24 Sep 2026, reversing D28's
    original "verbose gets no conf_bpe-analog" language) - nulling this on
    `verbose` would have left conf_sc_autoj (already null there) and this
    signal both undefined, making RQ7's verbosity-attack test impossible
    to run on ANY signal. The fix mirrors D21's own already-established
    precedent for the primary judge exactly: `conf_sc` alone is excluded
    from the verbosity-attack test (no verbose data), while `conf_bpe`
    (built from just the two greedy calls) stays in. Here, `conf_sc_autoj`
    plays `conf_sc`'s role and stays clean-only; `conf_sc_bpe_autoj` plays
    `conf_bpe`'s role and survives on both conditions, at a real,
    reportable cost: `verbose`'s own resolution is coarser (a single-draw
    degenerate proportion on the AB side, instead of clean's real k_sc-draw
    one) - state this in REPORT.md's limitations, don't hide it.
    """
    p = _p_model_a_wins_autoj(rows, k_sc)
    if p is None:
        return None
    return 1 - _binary_entropy(p)


def compute_item_signals_autoj(rows: list[dict], k_sc: int) -> dict:
    """Combines every signal into one item-level record. `rows` is every
    call belonging to a single (item_id, condition) group - no
    prompt_variant axis (D28), the caller's groupby is responsible for
    this grouping. No explicit condition gating needed anywhere here:
    conf_sc_autoj naturally returns None on `verbose` (no sampled draws to
    find); everything else is computable on both conditions by
    construction (see each function's own docstring).
    """
    return {
        "judge_verdict": judge_verdict_autoj(rows),
        "verdict_bidir": verdict_bidir_autoj(rows, k_sc),
        "conf_sc_autoj": conf_sc_autoj(rows, k_sc),
        "conf_sc_bpe_autoj": conf_sc_bpe_autoj(rows, k_sc),
        "flipped": flipped_autoj(rows),
    }


def build_items_autoj_dataframe(calls: pd.DataFrame, items_labels: pd.DataFrame, k_sc: int) -> pd.DataFrame:
    """calls_autoj_13b_gptq_4bits.parquet -> items_autoj_13b_gptq_4bits.parquet.
    One row per (item_id, condition) - no prompt_variant axis (D28).
    Mirrors build_items_kev_dataframe's join/NaN-handling pattern exactly.
    """
    labels_by_item = items_labels.set_index("item_id")

    records = []
    for (item_id, condition), group in calls.groupby(["item_id", "condition"]):
        rows = _sanitize_records(group.to_dict("records"))
        first_row = rows[0]
        signals = compute_item_signals_autoj(rows, k_sc)

        label_row = (
            _sanitize_records([labels_by_item.loc[str(item_id)].to_dict()])[0]
            if item_id in labels_by_item.index
            else {}
        )
        human_label = label_row.get("majority_label")

        ab_call = _find_call(rows, "AB", 0)
        n_prompt_tokens = ab_call["n_prompt_tokens"] if ab_call else None

        records.append(
            {
                "item_id": item_id,
                "question_id": first_row["question_id"],
                "category": first_row["category"],
                "condition": condition,
                "turn": first_row["turn"],
                "human_label": human_label,
                "n_human_votes": label_row.get("n_human_votes"),
                "frac_prefer_a": label_row.get("frac_prefer_a"),
                "human_unanimous": label_row.get("human_unanimous"),
                "human_agreed": label_row.get("human_agreed"),
                "d_human": label_row.get("d_human"),
                **signals,
                "correct": (
                    signals["judge_verdict"] == human_label
                    if signals["judge_verdict"] is not None and human_label is not None
                    else None
                ),
                "correct_bidir": (
                    signals["verdict_bidir"] == human_label
                    if signals["verdict_bidir"] is not None and human_label is not None
                    else None
                ),
                "n_prompt_tokens": n_prompt_tokens,
                "any_skipped": any(r.get("skipped") for r in rows),
            }
        )

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = AutojConfig.from_yaml(args.config)
    checkpoint_path = Path(config.paths.runs_dir) / "autoj.jsonl"

    calls = build_calls_autoj(checkpoint_path)
    Path(config.paths.calls_parquet).parent.mkdir(parents=True, exist_ok=True)
    calls.to_parquet(config.paths.calls_parquet)
    print(f"Wrote {len(calls)} calls to {config.paths.calls_parquet}")

    items_labels = pd.read_parquet(config.paths.items_labels_parquet)
    items = build_items_autoj_dataframe(calls, items_labels, config.k_sc)
    items.to_parquet(config.paths.items_parquet)
    print(f"Wrote {len(items)} items to {config.paths.items_parquet}")
