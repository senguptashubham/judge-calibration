"""Export the static site's data file from the demo bundle.

`python -m analysis.site_data --config configs/run.yaml`   (after `python -m analysis.demo_bundle ...`)

The site never computes a statistic: every number it shows is computed
here, with the same functions the analysis uses (src/metrics.py::
auto_accept_stats), and padded answers are rendered with the same
src/perturb.py::verbose_pad the runs used. The page only looks values up.

Writes site/data/site-data.js, a plain `window.SITE_DATA = {...}` script so
the page also works opened straight from disk (file://), with no server.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.compare_judges import MEASURES
from analysis.demo_bundle import BUNDLE_PATH
from src.config import Config
from src.metrics import auto_accept_stats
from src.perturb import verbose_pad
from src.plots import SIGNAL_LABELS

SITE_DATA_PATH = Path("site/data/site-data.js")
THRESHOLDS = [round(t, 2) for t in np.arange(0.50, 0.995, 0.01)]
GAME_MAX_CHARS = 1800   # both answers together, so an item reads in under a minute
# Loaded only when a visitor asks for a random example, never with the page.
EXAMPLES_PATH = Path("site/data/examples.js")
RANDOM_EXAMPLES = 40
RANDOM_EXAMPLE_MAX_BYTES = 20 * 1024


def _r(x, digits: int = 4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), digits)


def auto_accept_series(series: dict) -> dict:
    """One series' readouts at every threshold, plus each item as
    [confidence, correct] for the pipeline animation."""
    stats = [
        auto_accept_stats(series["confidence"], series["correct"], t, series["verdict"], series["human_label"])
        for t in THRESHOLDS
    ]
    return {
        "judge": series["judge"],
        "signal": series["signal"],
        "condition": series["condition"],
        "n": stats[0]["n"],
        "base_error": _r(1 - float(np.mean(series["correct"]))),
        "accepted_share": [_r(s["accepted_share"]) for s in stats],
        "slip_through": [_r(s["slip_through"]) for s in stats],
        "error_among_accepted": [_r(s["error_among_accepted"]) for s in stats],
        "kappa_among_accepted": [_r(s["kappa_among_accepted"], 3) for s in stats],
        "n_escalated": [s["n_escalated"] for s in stats],
        "items": [[round(c, 3), int(ok)] for c, ok in zip(series["confidence"], series["correct"])],
    }


def site_item(item: dict) -> dict:
    """A showcase item with its padded conversations pre-rendered."""
    return {
        **item,
        "padded_a": verbose_pad(item["conversation_a"]),
        "padded_b": verbose_pad(item["conversation_b"]),
    }


def verdicts_only(judges: dict) -> dict:
    """A judge view without the reasoning text: the game and the three-judge
    grid show only which model each judge picked and how sure it was."""
    return {judge: {cond: {order: {k: v for k, v in call.items() if k != "text"} for order, call in orders.items()}
                    for cond, orders in conds.items()}
            for judge, conds in judges.items()}


def pick_game_items(items: list[dict], seed: int, max_chars: int = GAME_MAX_CHARS) -> list[dict]:
    """The "you vs the judge" pool: short turn-1 comparisons where at least two
    humans voted and all agreed, and the two answers differ. One item per
    MT-Bench question, drawn at random with a fixed seed.

    Selection never looks at the judges' verdicts, so the judges' score in
    the game is not tilted either way by which items were picked.
    """
    def length(item: dict) -> int:
        return sum(len(m["content"]) for side in ("conversation_a", "conversation_b") for m in item[side])

    by_question: dict[int, list[dict]] = {}
    for item in items:
        if (item["human"]["agreed"] and item["turn"] == 1 and length(item) <= max_chars
                and item["conversation_a"] != item["conversation_b"]):
            by_question.setdefault(item["question_id"], []).append(item)
    rng = np.random.default_rng(seed)
    pool = []
    for question_id in sorted(by_question):
        candidates = sorted(by_question[question_id], key=lambda it: it["item_id"])
        pool.append(candidates[rng.integers(len(candidates))])
    return [{**{k: v for k, v in item.items() if k != "judges"}, "judges": verdicts_only(item["judges"])}
            for item in pool]


def pick_random_examples(items: list[dict], exclude: set[str], seed: int, n: int = RANDOM_EXAMPLES,
                         max_bytes: int = RANDOM_EXAMPLE_MAX_BYTES) -> list[dict]:
    """Extra "Trick the judge" examples, drawn at random rather than picked for a flip.

    Eligible: at least two humans voted and all agreed, the answers differ, the
    item is not already in the showcase or the game, and its full page payload
    (answers, padded answers, every judge's reasoning) fits in max_bytes. At most
    one item per MT-Bench question; n questions drawn with a fixed seed.

    The size cap keeps the file small but favours shorter answers, so these
    are not a representative sample and the page makes no rate claim from them.
    """
    by_question: dict[int, list[dict]] = {}
    for item in items:
        if (item["human"]["agreed"] and item["item_id"] not in exclude
                and item["conversation_a"] != item["conversation_b"]):
            full = site_item(item)
            if len(json.dumps(full, separators=(",", ":")).encode()) <= max_bytes:
                by_question.setdefault(item["question_id"], []).append(full)
    rng = np.random.default_rng(seed)
    questions = sorted(by_question)
    chosen = rng.choice(questions, size=min(n, len(questions)), replace=False)
    out = []
    for question_id in sorted(int(q) for q in chosen):
        candidates = sorted(by_question[question_id], key=lambda it: it["item_id"])
        out.append({"kind": "random", **candidates[rng.integers(len(candidates))]})
    return out


def method_numbers(items: list[dict]) -> dict:
    """Counts for the method diagram, taken from the same items the page shows."""
    calls = {judge: [call for conds in (it["judges"][judge] for it in items)
                     for orders in conds.values() for call in orders.values()]
             for judge in items[0]["judges"]}
    autoj_padded_t2 = [call for it in items if it["turn"] == 2
                       for call in it["judges"]["auto-j-13b"]["verbose"].values() if call["status"] != "skipped"]
    return {
        "n_questions": len({it["question_id"] for it in items}),
        "n_models": len({it["model_a"] for it in items} | {it["model_b"] for it in items}),
        "n_items": len(items),
        "verdicts": {judge: sum(c["status"] == "ok" for c in cs) for judge, cs in calls.items()},
        "autoj_padded_turn2_no_verdict": _r(np.mean([c["status"] == "no_verdict" for c in autoj_padded_t2]), 3),
    }


def headline(results_dir: str, slug: str, bundle: dict) -> dict:
    rq1 = pd.read_csv(f"{results_dir}/rq1_table_{slug}.csv")
    verb = rq1[(rq1["confidence_column"] == "conf_verb") & (rq1["verdict_definition"] == "judge_verdict")].iloc[0]
    head = {(r["judge"], r["signal"], r["condition"], r["threshold"]): r for r in bundle["auto_accept_headline"]}
    comp = {(r["measure"], r["judge"], r["population"]): r for r in bundle["judge_comparison"]}
    return {
        "n_items": len(bundle["items"]),
        "stated_confidence": _r(verb["accuracy"] + verb["overconfidence_gap"], 3),
        "accuracy": _r(verb["accuracy"], 3),
        "slip_stated_09": _r(head[("Qwen2.5-7B", "conf_verb", "clean", 0.9)]["slip_through"], 3),
        "slip_best_09": _r(head[("Qwen2.5-7B", "p_correct_bayesian", "clean", 0.9)]["slip_through"], 3),
        "flip_rate": _r(comp[("flip_rate", "Qwen2.5-7B", "all")]["value"], 3),
    }


def main(config_path: str) -> None:
    config = Config.from_yaml(config_path)
    with gzip.open(BUNDLE_PATH, "rt", encoding="utf-8") as f:
        bundle = json.load(f)
    items = {it["item_id"]: it for it in bundle["items"]}

    data = {
        "headline": headline(config.paths.results_dir, config.model_slug, bundle),
        "thresholds": THRESHOLDS,
        "showcase": [{"kind": s["kind"], **site_item(items[s["item_id"]])} for s in bundle["showcase"]],
        "auto_accept": [auto_accept_series(s) for s in bundle["auto_accept_series"]],
        "signal_labels": SIGNAL_LABELS,
        "game": pick_game_items(bundle["items"], config.seed),
        "judge_comparison": [{**r, "value": _r(r["value"]), "ci_low": _r(r["ci_low"]), "ci_high": _r(r["ci_high"])}
                             for r in bundle["judge_comparison"]],
        "comparison_measures": [{"measure": m, "title": title, "axis": axis, "reference": ref}
                                for m, title, axis, ref in MEASURES],
        "method": method_numbers(bundle["items"]),
    }
    SITE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SITE_DATA_PATH.write_text("window.SITE_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")
    used = {s["item_id"] for s in data["showcase"]} | {g["item_id"] for g in data["game"]}
    examples = pick_random_examples(bundle["items"], used, config.seed)
    EXAMPLES_PATH.write_text("window.SITE_EXAMPLES = " + json.dumps(examples, separators=(",", ":")) + ";\n",
                             encoding="utf-8")
    print(f"Wrote {EXAMPLES_PATH} ({EXAMPLES_PATH.stat().st_size / 1e6:.2f} MB): {len(examples)} random examples")
    print(f"Wrote {SITE_DATA_PATH} ({SITE_DATA_PATH.stat().st_size / 1e6:.2f} MB): "
          f"{len(data['showcase'])} showcase items, {len(data['game'])} game items, "
          f"{len(data['auto_accept'])} auto-accept series")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
