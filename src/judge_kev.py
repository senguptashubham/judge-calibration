"""HTTP client for kev-8b's `/v1/systemone` contract (DECISIONS.md D27) -
the industry-counterexample comparison arm, not part of the primary judge
harness.

Deliberately separate from judge.py/config.py: kev-8b's real schedule has
no prompt_variant axis and no self-consistency sampling (D27 - kev does one
deterministic forward pass per request, no autoregressive generation, so
resampling would just recover its own reported distribution, not an
independent estimate). A separate, smaller `KevConfig` avoids forcing
meaningless placeholder values through `src/config.py::Config`'s validation
(`temperature_sc > 0`, `"P1" in prompt_variants`), which was built for a
different harness's invariants and doesn't apply here at all.

Requires `kev.serve` already running and reachable at `KevConfig.base_url` -
this module only ever POSTs to that URL, it never loads the model itself.
See DECISIONS.md D27 for the confirmed launch config
(`KEV_DTYPE=bf16 KEV_MERGE=0 KEV_ATTN=sdpa`, via `setsid`) and the confirmed
`max_state_tokens` ceiling (8,160 - beyond ~8,165 kev-8b becomes genuinely
non-deterministic on a 24GB-class GPU, confirmed across two independent
probe sessions, not a documentation guess).
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
        # Same derivation as src/config.py::Config.model_slug (D26) -
        # duplicated rather than imported, since Config's own __post_init__
        # validation doesn't apply to this harness and shouldn't be worked
        # around with meaningless placeholder field values.
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
    """Every (condition, order) combination one item needs - D27's schedule
    is deliberately flat compared to judge.py's D19 schedule: no
    prompt_variant axis, no sampling. `len(conditions) * 2` calls/item.
    """
    return [KevCallSpec(condition, order) for condition in config.conditions for order in ("AB", "BA")]


def checkpoint_key(item: str, condition: str, order: str) -> str:
    return f"{item}|{condition}|{order}"


def load_completed_keys(checkpoint_path: Path) -> set[str]:
    """Same resumability guarantee as judge.py's own version (CLAUDE.md
    invariant 9), against this module's simpler 3-field key.
    """
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
    """Same shape as judge.py::pending_calls() - a pure function, no
    network, so resumability is directly testable without a server running.
    """
    pending: list[tuple[str, "pd.Series", KevCallSpec]] = []
    for _, item_row in items_df.iterrows():
        item = item_id(item_row["question_id"], item_row["model_a"], item_row["model_b"], item_row["turn"])
        for spec in specs:
            key = checkpoint_key(item, spec.condition, spec.order)
            if key not in completed:
                pending.append((item, item_row, spec))
    return pending


def _render_side(label: str, conversation: list[dict], turn: int) -> str:
    """Renders one side's transcript - a small, local copy of
    src/prompts.py's own private _render_conversation(), not an import:
    prompts.py is frozen after Week 1 (CLAUDE.md invariant 10, "ask before
    touching src/prompts.py") and this comparison arm postdates that freeze
    by three weeks. Duplicating ~6 lines here is cheaper than reopening a
    frozen file for a private-function visibility change.
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
    """Builds kev's `state` (the document) - just the question+two-responses
    content, no instruction text (that lives in build_payload()'s
    `questions.verdict` branch instead, since kev's typed-question format
    separates the two). `order` picks which physical model is labeled A/B
    (reuses prompts.py::apply_order() - public, generic, no template text,
    so importing it doesn't touch anything frozen). `condition == "verbose"`
    applies the identical verbose_pad() perturbation task 4.2 used for the
    primary judge - the same attack, not a re-derived one.
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
    """The only function in this module that touches the network. Returns
    a dict with `ok` (bool) plus either the parsed verdict fields or an
    `error` string - callers don't need to distinguish a network error, a
    422 validation rejection ("branch too long"), or a 500 server crash to
    decide what to do: log and skip, the same treatment for all three (D27
    - all three are real, observed failure modes on this harness).
    """
    payload = build_payload(state)
    try:
        resp = requests.post(base_url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}
    if resp.status_code != 200:
        return {"ok": False, "error": f"http_{resp.status_code}: {resp.text[:500]}"}
    body = resp.json()
    verdict = body.get("answers", {}).get("verdict", {})
    return {
        "ok": True,
        "choice": verdict.get("choice"),
        "probabilities": verdict.get("probabilities"),
        "input_tokens": body.get("usage", {}).get("input_tokens"),
    }


def _run_calls(config: KevConfig, checkpoint_path: Path, items_df: "pd.DataFrame") -> None:
    """Iterates the pending schedule, checking each state's real token
    length (kev-8b's own tokenizer, Qwen3-8B-Base) against
    config.max_state_tokens BEFORE calling - items over the cap are logged
    to the checkpoint with skipped=True and a reason, never silently
    dropped (same "never silently impute" standard D21 sets for
    conf_sc/conf_ens on verbose). The only function besides call_kev()
    that touches something external (here: the tokenizer download) -
    everything above stays importable/testable without `transformers`
    installed, the same "only touches vllm inside _run_generation" split
    judge.py already uses for its own Colab-only dependency.
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
    # load_full_items_df() only reads config.dataset/config.tie_policy -
    # KevConfig has both by the same names, so this works via duck typing
    # despite the type hint upstream saying Config, not KevConfig.
    kev_items_df = load_full_items_df(cfg)

    if args.n_items is not None:
        kev_items_df = kev_items_df.sample(n=args.n_items, random_state=cfg.seed)

    kev_checkpoint_path = Path(cfg.paths.runs_dir) / "kev.jsonl"
    _run_calls(cfg, kev_checkpoint_path, kev_items_df)
