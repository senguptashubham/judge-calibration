"""RQ6: HTTP client for kev-8b's `/v1/systemone` endpoint (D27).

kev-8b makes one deterministic forward pass per request - no prompt
variants, no sampling - so the schedule is (condition, order) only, and a
separate KevConfig avoids forcing placeholder values through Config's
validation, which is built for the primary judge.

Needs `kev.serve` already running at KevConfig.base_url; this module only
POSTs to it. D27 has the working launch configuration and the 8,160-token
ceiling (above ~8,165 tokens kev-8b hits non-deterministic OOMs on an L4).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd
import requests
import yaml

from src.data import item_id
from src.judge import append_checkpoint, git_sha, load_full_items_df
from src.perturb import verbose_pad
from src.prompts import apply_order


@dataclasses.dataclass(frozen=True)
class KevPaths:
    results_dir: str
    runs_dir: str
    calls_parquet: str
    items_parquet: str
    items_labels_parquet: str


@dataclasses.dataclass(frozen=True)
class KevConfig:
    judge_model: str
    dataset: str
    tie_policy: str
    conditions: list[str]
    base_url: str
    max_state_tokens: int
    seed: int
    n_bins: int
    paths: KevPaths

    @property
    def model_slug(self) -> str:
        # Same derivation as Config.model_slug (D26).
        name = self.judge_model.split("/")[-1]
        return name.lower().replace("-", "_")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "KevConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = dict(yaml.safe_load(f))
        raw["paths"] = KevPaths(**raw["paths"])
        return cls(**raw)


@dataclasses.dataclass(frozen=True)
class KevCallSpec:
    condition: str
    order: str  # "AB" | "BA"


def call_schedule(config: KevConfig) -> list[KevCallSpec]:
    """Every (condition, order) pair: len(conditions) * 2 calls per item."""
    return [KevCallSpec(condition, order) for condition in config.conditions for order in ("AB", "BA")]


def checkpoint_key(item: str, condition: str, order: str) -> str:
    return f"{item}|{condition}|{order}"


def load_completed_keys(checkpoint_path: Path) -> set[str]:
    """Keys already in the checkpoint (invariant 9)."""
    completed: set[str] = set()
    if not checkpoint_path.exists():
        return completed
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed.add(checkpoint_key(row["item_id"], row["condition"], row["order"]))
    return completed


def pending_calls(
    items_df: "pd.DataFrame", specs: list[KevCallSpec], completed: set[str]
) -> list[tuple[str, "pd.Series", KevCallSpec]]:
    """Every (item, spec) pair not already completed - pure, testable without a server."""
    pending: list[tuple[str, "pd.Series", KevCallSpec]] = []
    for _, item_row in items_df.iterrows():
        item = item_id(item_row["question_id"], item_row["model_a"], item_row["model_b"], item_row["turn"])
        for spec in specs:
            key = checkpoint_key(item, spec.condition, spec.order)
            if key not in completed:
                pending.append((item, item_row, spec))
    return pending


def _render_side(label: str, conversation: list[dict], turn: int) -> str:
    """One side's transcript. A local copy of prompts.py's private
    _render_conversation(), so the frozen prompts.py needn't change.
    """
    lines = [f"[Response from Assistant {label}]"]
    lines.append(f"User: {conversation[0]['content']}")
    lines.append(f"Assistant {label}: {conversation[1]['content']}")
    if turn == 2:
        lines.append(f"User: {conversation[2]['content']}")
        lines.append(f"Assistant {label}: {conversation[3]['content']}")
    return "\n".join(lines)


def build_state(
    condition: str,
    order: str,
    model_a_conversation: list[dict],
    model_b_conversation: list[dict],
    turn: int,
) -> str:
    """kev's `state` document: the two responses only, no instructions
    (those go in build_payload()'s question). `order` picks which model is
    labeled A; "verbose" applies the same verbose_pad() the primary judge
    saw.
    """
    displayed_a, displayed_b = apply_order(order, model_a_conversation, model_b_conversation)
    if condition == "verbose":
        displayed_a, displayed_b = verbose_pad(displayed_a), verbose_pad(displayed_b)
    return f"{_render_side('A', displayed_a, turn)}\n\n{_render_side('B', displayed_b, turn)}"


def build_payload(state: str) -> dict:
    return {
        "state": state,
        "model": "kev-latest",
        "questions": {
            "verdict": {
                "type": "choice",
                "instructions": "Which response is a better answer, A or B?",
                "criteria": {"A": None, "B": None},
            }
        },
    }


def call_kev(base_url: str, state: str, timeout: int = 120) -> dict:
    """The only network call. Returns the COMPLETE raw response - the full
    JSON body on success, the raw error text on an HTTP error, the exception
    string on a network failure - never a curated subset. Deriving
    choice/probabilities from it is kev_signals.py's job (invariant 7).
    """
    payload = build_payload(state)
    try:
        resp = requests.post(base_url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return {"ok": False, "http_status": None, "raw_response": None, "error": str(exc)}
    if resp.status_code != 200:
        return {"ok": False, "http_status": resp.status_code, "raw_response": resp.text, "error": None}
    return {"ok": True, "http_status": resp.status_code, "raw_response": resp.json(), "error": None}


def _run_calls(config: KevConfig, checkpoint_path: Path, items_df: "pd.DataFrame") -> None:
    """Runs the pending schedule, checking each state's real token length
    (kev's tokenizer, Qwen3-8B-Base) against max_state_tokens before calling.
    Over-length states are logged with skipped=True and a reason, never
    silently dropped.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Base")
    completed = load_completed_keys(checkpoint_path)
    sha = git_sha()

    specs = call_schedule(config)
    pending = pending_calls(items_df, specs, completed)

    for item, item_row, spec in pending:
        state = build_state(
            spec.condition, spec.order, item_row["conversation_a"], item_row["conversation_b"], item_row["turn"]
        )
        true_tokens = len(tok.encode(state, add_special_tokens=False))

        row = {
            "item_id": item,
            "question_id": int(item_row["question_id"]),
            "category": item_row.get("category"),
            "model_a": item_row["model_a"],
            "model_b": item_row["model_b"],
            "turn": int(item_row["turn"]),
            "condition": spec.condition,
            "order": spec.order,
            "judge_model": config.judge_model,
            "git_sha": sha,
            "state_tokens": true_tokens,
        }

        if true_tokens > config.max_state_tokens:
            row.update({"skipped": True, "skip_reason": "over_max_state_tokens", "ok": False})
            append_checkpoint(checkpoint_path, row)
            continue

        result = call_kev(config.base_url, state)
        row["skipped"] = False
        row.update(result)
        append_checkpoint(checkpoint_path, row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        help="Limit to a seeded random sample of N items, for a scoped smoke test before the full run.",
    )
    args = parser.parse_args()

    cfg = KevConfig.from_yaml(args.config)
    # load_full_items_df() reads only dataset/tie_policy, which KevConfig has.
    kev_items_df = load_full_items_df(cfg)

    if args.n_items is not None:
        kev_items_df = kev_items_df.sample(n=args.n_items, random_state=cfg.seed)

    kev_checkpoint_path = Path(cfg.paths.runs_dir) / "kev.jsonl"
    _run_calls(cfg, kev_checkpoint_path, kev_items_df)
