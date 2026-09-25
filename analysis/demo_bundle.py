"""Build the site's data bundle from the recorded runs - no model is called.

`HF_DATASETS_OFFLINE=1 python -m analysis.demo_bundle --config configs/run.yaml --kev-config configs/run_kev.yaml --autoj-config configs/run_autoj.yaml`

For every human-labelled clean item (RQ1's 1,836): the conversation both
models had, the human vote, and each judge's greedy call under
(clean, padded) x (AB, BA) - which position it picked, which model that
means, its stated confidence where it has one, and its own reasoning text.
Plus the auto-accept series (analysis/auto_accept.py), a showcase list for
the site's opening examples, and the headline comparison numbers.

Padded answers are not stored here: analysis/site_data.py renders them with
src/perturb.py::verbose_pad, the same function the runs used.

Writes results/site_bundle.json.gz, an intermediate (not in git, like the
parquet tables) that analysis/site_data.py turns into the site's data files.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.auto_accept import load_series
from analysis.rq1 import load_rq1_items
from src.config import Config
from src.judge import load_full_items_df
from src.judge_autoj import AutojConfig
from src.judge_kev import KevConfig
from src.metrics import auto_accept_stats
from src.plots import judge_name

BUNDLE_PATH = Path("results/site_bundle.json.gz")
CONDITIONS = ["clean", "verbose"]
ORDERS = ["AB", "BA"]


def canonical_winner(pick: str | None, order: str) -> str | None:
    """Displayed position ("first"/"second") -> canonical model ("A" =
    model_a, "B" = model_b). Under AB the first answer shown is model_a;
    under BA it is model_b."""
    if pick not in ("first", "second"):
        return None
    first_is_a = order == "AB"
    return "A" if (pick == "first") == first_is_a else "B"


def _clean(value):
    """NaN/None -> None, numpy scalars -> Python, for JSON."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def qwen_call_view(row: pd.Series | None, order: str) -> dict:
    """One Qwen call: the displayed letter it chose ("A" = shown first), its
    stated confidence, and its reasoning."""
    if row is None or not row["parse_ok"]:
        return {"status": "no_verdict", "pick": None, "winner": None, "conf": None, "text": None}
    pick = {"A": "first", "B": "second"}.get(row["verdict"])
    return {"status": "ok", "pick": pick, "winner": canonical_winner(pick, order),
            "conf": _clean(row["verbalized_conf"]), "text": _clean(row["reasoning_text"])}


def kev_call_view(row: pd.Series | None, order: str) -> dict:
    """One kev-8b call: its choice and the probability it put on it. kev
    writes no reasoning."""
    if row is None or row["skipped"]:
        return {"status": "skipped", "pick": None, "winner": None, "conf": None, "text": None}
    if not bool(row["ok"]):
        return {"status": "no_verdict", "pick": None, "winner": None, "conf": None, "text": None}
    pick = {"A": "first", "B": "second"}.get(row["choice"])
    conf = row["prob_a"] if row["choice"] == "A" else row["prob_b"]
    return {"status": "ok", "pick": pick, "winner": canonical_winner(pick, order), "conf": _clean(conf), "text": None}


def autoj_call_view(row: pd.Series | None, order: str) -> dict:
    """One auto-j greedy call: its decision and critique. auto-j states no
    confidence; "Response 1" in its critique is the answer shown first."""
    if row is None or row["skipped"]:
        return {"status": "skipped", "pick": None, "winner": None, "conf": None, "text": None}
    label = row["pred_label"]
    pick = {0: "first", 1: "second", 2: "tie"}.get(label)
    status = "ok" if pick in ("first", "second") else ("tie" if pick == "tie" else "no_verdict")
    return {"status": status, "pick": pick if pick != "tie" else None, "winner": canonical_winner(pick, order),
            "conf": None, "text": _clean(row["critique"])}


def _index_calls(calls: pd.DataFrame, keys: list[str]) -> dict:
    return {tuple(k): row for k, row in zip(calls[keys].itertuples(index=False, name=None),
                                             (r for _, r in calls.iterrows()))}


def build_judge_views(config: Config, kev_config: KevConfig, autoj_config: AutojConfig) -> dict:
    """item_id -> judge -> condition -> order -> call view."""
    qwen = pd.read_parquet(config.paths.calls_parquet)
    qwen = qwen[(qwen["prompt_variant"] == "P1") & (qwen["sample_idx"] == 0)]
    kev = pd.read_parquet(kev_config.paths.calls_parquet)
    autoj = pd.read_parquet(autoj_config.paths.calls_parquet)
    autoj = autoj[autoj["sample_idx"] == 0]

    sources = [
        (judge_name(config.model_slug), _index_calls(qwen, ["item_id", "condition", "order"]), qwen_call_view),
        (judge_name(kev_config.model_slug), _index_calls(kev, ["item_id", "condition", "order"]), kev_call_view),
        (judge_name(autoj_config.model_slug), _index_calls(autoj, ["item_id", "condition", "order"]), autoj_call_view),
    ]
    item_ids = set(qwen["item_id"])
    views: dict = {}
    for item_id in item_ids:
        views[item_id] = {
            judge: {cond: {order: view_fn(index.get((item_id, cond, order)), order) for order in ORDERS}
                    for cond in CONDITIONS}
            for judge, index, view_fn in sources
        }
    return views


def _conversation(conv, turn: int) -> list[dict]:
    """The conversation up to and including the judged turn - what the judge
    saw. MT-Bench stores both turns even for turn-1 comparisons, so without
    the cut a turn-1 item would show a second exchange the judge never read."""
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in list(conv)[: 2 * turn]]


def pick_showcase_items(items: list[dict], judge: str, n_per_kind: int = 4, max_chars: int = 2500) -> list[dict]:
    """Deterministic opening examples, shortest first (then by item id):

    - "order": the judge picks a different model when the answer order is
      swapped, stating >= 0.9 confidence both times;
    - "padding": padding flips its clean AB verdict, again at >= 0.9 both times;
    - "judges": the three judges' clean AB verdicts don't all agree;
    - "identical": both models gave word-for-word the same answers, yet the
      judge picks one at >= 0.9 - it invents a difference.

    Identical-answer items are kept out of the first three kinds, whose
    point is a flip between answers that genuinely differ. At most one item
    per MT-Bench question across the whole showcase, so it spans topics, and
    only items where at least two humans voted and all agreed, so the
    "reveal" rests on a real consensus rather than one vote.
    """
    def length(item: dict) -> int:
        return sum(len(m["content"]) for side in ("conversation_a", "conversation_b") for m in item[side])

    def confident(view: dict) -> bool:
        return view["status"] == "ok" and view["conf"] is not None and view["conf"] >= 0.9

    order_flips, padding_flips, disagreements, identical = [], [], [], []
    for item in sorted(items, key=lambda it: (length(it), it["item_id"])):
        if length(item) > max_chars or not item["human"]["agreed"]:
            continue
        calls = item["judges"][judge]
        ab, ba, padded_ab = calls["clean"]["AB"], calls["clean"]["BA"], calls["verbose"]["AB"]
        if item["conversation_a"] == item["conversation_b"]:
            if confident(ab):
                identical.append(item["item_id"])
            continue
        if confident(ab) and confident(ba) and ab["winner"] != ba["winner"]:
            order_flips.append(item["item_id"])
        if confident(ab) and confident(padded_ab) and ab["winner"] != padded_ab["winner"]:
            padding_flips.append(item["item_id"])
        winners = {j: v["clean"]["AB"]["winner"] for j, v in item["judges"].items()}
        if None not in winners.values() and len(set(winners.values())) > 1:
            disagreements.append(item["item_id"])

    question_of = {item["item_id"]: item.get("question_id", item["item_id"]) for item in items}
    used_questions = set()
    showcase = []
    # "identical" is picked first so its question is reserved for it; the
    # list is then sorted for display, with an order flip on top.
    for kind, ids in [("identical", identical), ("order", order_flips), ("padding", padding_flips),
                      ("judges", disagreements)]:
        picked = 0
        for item_id in ids:
            if picked == n_per_kind:
                break
            if question_of[item_id] in used_questions:
                continue
            used_questions.add(question_of[item_id])
            showcase.append({"item_id": item_id, "kind": kind})
            picked += 1
    display_order = ["order", "padding", "identical", "judges"]
    return sorted(showcase, key=lambda entry: display_order.index(entry["kind"]))


def build_bundle(config: Config, kev_config: KevConfig, autoj_config: AutojConfig) -> dict:
    labelled = load_rq1_items(config.paths.items_parquet)
    full = load_full_items_df(config).set_index("item_id")
    views = build_judge_views(config, kev_config, autoj_config)

    items = []
    for _, row in labelled.iterrows():
        source = full.loc[row["item_id"]]
        items.append(
            {
                "item_id": row["item_id"],
                "question_id": int(row["question_id"]),
                "category": row["category"],
                "turn": int(row["turn"]),
                "model_a": source["model_a"],
                "model_b": source["model_b"],
                "conversation_a": _conversation(source["conversation_a"], int(row["turn"])),
                "conversation_b": _conversation(source["conversation_b"], int(row["turn"])),
                "human": {
                    "label": row["human_label"],
                    "n_votes": int(row["n_human_votes"]),
                    "frac_prefer_a": _clean(row["frac_prefer_a"]),
                    "agreed": bool(row["human_agreed"]),
                },
                "judges": views[row["item_id"]],
            }
        )

    series = []
    for df in load_series(config, kev_config, autoj_config):
        series.append(
            {
                "judge": df["judge"].iloc[0],
                "signal": df["signal"].iloc[0],
                "condition": df["condition"].iloc[0],
                "confidence": df["confidence"].round(6).tolist(),
                "correct": df["correct"].astype(bool).tolist(),
                "verdict": df["verdict"].tolist(),
                "human_label": df["human_label"].tolist(),
            }
        )

    results_dir = config.paths.results_dir
    return {
        "items": items,
        "showcase": pick_showcase_items(items, judge_name(config.model_slug)),
        "auto_accept_series": series,
        "judge_comparison": pd.read_csv(f"{results_dir}/judge_comparison.csv").to_dict("records"),
        "auto_accept_headline": pd.read_csv(f"{results_dir}/auto_accept.csv").to_dict("records"),
    }


def main(config_path: str, kev_config_path: str, autoj_config_path: str) -> None:
    config = Config.from_yaml(config_path)
    bundle = build_bundle(config, KevConfig.from_yaml(kev_config_path), AutojConfig.from_yaml(autoj_config_path))
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(BUNDLE_PATH, "wt", encoding="utf-8") as f:
        json.dump(bundle, f, default=_clean)
    kinds = pd.Series([s["kind"] for s in bundle["showcase"]]).value_counts().to_dict()
    print(f"Wrote {len(bundle['items'])} items, {len(bundle['auto_accept_series'])} auto-accept series, "
          f"showcase {kinds} to {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size / 1e6:.1f} MB)")
    # A sanity check that the bundle reproduces analysis/auto_accept.py's headline number.
    first = bundle["auto_accept_series"][0]
    stats = auto_accept_stats(first["confidence"], first["correct"], 0.9)
    print(f"check: {first['judge']} {first['signal']} {first['condition']} slip_through@0.9 = {stats['slip_through']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--kev-config", required=True)
    parser.add_argument("--autoj-config", required=True)
    args = parser.parse_args()
    main(args.config, args.kev_config, args.autoj_config)
