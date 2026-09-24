"""RQ7: runs/autoj_13b_gptq_4bits/autoj.jsonl -> calls -> items parquet.

The auto-j-13b counterpart of parse.py + signals.py, at the grain
(item_id, condition), with self-consistency sampling (unlike kev-8b).
`raw_output` is read in exactly one place, build_calls_autoj(), which turns
it into a `pred_label` column; every signal below works from that label.

auto-j has no confidence field, so both signals come from agreement between
calls: conf_sc_autoj (self-consistency) and conf_sc_bpe_autoj (an order-swap
entropy built from self-consistency proportions - named apart from conf_bpe
because it is not logprob-based).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.judge_autoj import AutojConfig
from src.signals import _binary_entropy, _sanitize_records, prob_on_verdict


def extract_autoj_verdict(raw_output: str) -> int:
    """Ported verbatim from auto-j's codes/usage/example.py::
    extract_pariwise_result() (D28). Returns 0 (Response 1 wins),
    1 (Response 2 wins), 2 (Tie), or -1 (no "final decision is ..." line,
    e.g. a completion truncated before reaching it).
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
    """autoj.jsonl -> one row per call, with `pred_label` extracted.
    Skipped (over-length) calls have no output, so their `pred_label` is
    None - distinct from -1, which means the model answered but no decision
    line was found.
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
    """One call by order + sample_idx, excluding skipped calls."""
    for row in rows:
        if row["order"] == order and row["sample_idx"] == sample_idx and not row.get("skipped"):
            return row
    return None


def _canonical_letter(pred_label: int | None, order: str) -> str | None:
    """auto-j's 0/1 label -> the project's canonical "A" (model_a) / "B"
    (model_b). Under AB, Response 1 is model_a (0->A, 1->B); under BA,
    Response 1 is model_b (0->B, 1->A).

    Tie (2) and parse failure (-1) both return None - neither is a usable
    binary verdict. They stay distinguishable in calls' `pred_label`.
    """
    if pred_label == 0:
        return "A" if order == "AB" else "B"
    if pred_label == 1:
        return "B" if order == "AB" else "A"
    return None


def judge_verdict_autoj(rows: list[dict]) -> str | None:
    """D7 primary: the greedy AB call's verdict, in canonical identity."""
    call = _find_call(rows, "AB", 0)
    if call is None:
        return None
    return _canonical_letter(call["pred_label"], "AB")


def _canonical_verdict_ba_autoj(rows: list[dict]) -> str | None:
    """The greedy BA call's verdict, in canonical identity."""
    call = _find_call(rows, "BA", 0)
    if call is None:
        return None
    return _canonical_letter(call["pred_label"], "BA")


def flipped_autoj(rows: list[dict]) -> bool | None:
    """True when the greedy AB and BA verdicts disagree (both conditions)."""
    ab = judge_verdict_autoj(rows)
    ba = _canonical_verdict_ba_autoj(rows)
    if ab is None or ba is None:
        return None
    return ab != ba


def _self_consistency_proportion_a(rows: list[dict], order: str, k_sc: int) -> float | None:
    """Fraction of `order`'s calls (greedy sample_idx=0 plus any sampled
    draws 1..k_sc) whose canonical verdict is model_a. Tie and parse-failure
    calls are left out of numerator and denominator.

    Only clean AB has sampled draws; for BA, and for AB on verbose, this is
    the single greedy call's 0/1 verdict.
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
    """Order-corrected P(model_a wins) = (p_ab + p_ba) / 2 - a PLAIN average.

    signals.py's version uses (p_ab + (1 - p_ba)) / 2 because its p_a is
    P(displayed-A wins), which needs flipping under BA. Here both
    proportions are already in canonical model_a terms (_canonical_letter
    does the translation), so flipping again would double-translate. Check:
    pure position bias (the first-displayed response always wins) gives
    p_ab = 1, p_ba = 0, and must net to indifference, 0.5.
    """
    p_ab = _self_consistency_proportion_a(rows, "AB", k_sc)
    p_ba = _self_consistency_proportion_a(rows, "BA", k_sc)
    if p_ab is None or p_ba is None:
        return None
    return (p_ab + p_ba) / 2


def verdict_bidir_autoj(rows: list[dict], k_sc: int) -> str | None:
    """D7 secondary: argmax of the order-corrected P(model_a wins)."""
    p = _p_model_a_wins_autoj(rows, k_sc)
    if p is None:
        return None
    return "A" if p >= 0.5 else "B"


def conf_sc_autoj(rows: list[dict], k_sc: int) -> float | None:
    """Fraction of the k_sc sampled AB calls whose raw `pred_label` equals
    the greedy AB call's - D6's conf_sc. All calls share the AB frame, so no
    translation is needed, and Tie==Tie counts as agreement. None on
    verbose, where no sampled draws exist.
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
    """1 - H(p), p = the order-corrected P(model_a wins) from
    self-consistency proportions. Defined on both conditions, but pools 5 AB
    draws on clean and 1 on verbose - use conf_sc_bpe_autoj_greedy for any
    clean-vs-verbose comparison.
    """
    p = _p_model_a_wins_autoj(rows, k_sc)
    if p is None:
        return None
    return 1 - _binary_entropy(p)


def conf_sc_bpe_autoj_greedy(rows: list[dict]) -> float | None:
    """conf_sc_bpe_autoj built from the two greedy calls only (AB and BA,
    sample_idx=0), ignoring any sampled draws. `verbose` has only these two
    calls, so this is the version the clean-vs-verbose comparison must use:
    the full conf_sc_bpe_autoj pools 5 AB draws on clean but 1 on verbose,
    and a paired delta between them would mix the verbosity effect with that
    change in resolution.
    """
    return conf_sc_bpe_autoj(rows, k_sc=0)


def compute_item_signals_autoj(rows: list[dict], k_sc: int) -> dict:
    verdict = judge_verdict_autoj(rows)
    return {
        "judge_verdict": verdict,
        "verdict_bidir": verdict_bidir_autoj(rows, k_sc),
        "conf_sc_autoj": conf_sc_autoj(rows, k_sc),
        "conf_sc_bpe_autoj": conf_sc_bpe_autoj(rows, k_sc),
        "conf_sc_bpe_autoj_greedy": conf_sc_bpe_autoj_greedy(rows),
        # Calibration forms of the two entropy signals: the same p, as a
        # probability on the AB verdict (signals.py::prob_on_verdict).
        "conf_sc_bpe_autoj_prob": prob_on_verdict(_p_model_a_wins_autoj(rows, k_sc), verdict),
        "conf_sc_bpe_autoj_greedy_prob": prob_on_verdict(_p_model_a_wins_autoj(rows, 0), verdict),
        "flipped": flipped_autoj(rows),
    }


def build_items_autoj_dataframe(calls: pd.DataFrame, items_labels: pd.DataFrame, k_sc: int) -> pd.DataFrame:
    """calls -> items, one row per (item_id, condition), human labels
    joined on item_id.
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
