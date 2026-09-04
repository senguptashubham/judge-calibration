# PREREGISTRATION.md

Written before looking at any judge output (`CLAUDE.md` §7 commands for `src.judge` have not been run). Frozen at Gate 1, per `TASKS.md`.

**Status of this document:** settled. D1's tie-policy comparison, the primary endpoint, the confirmatory/exploratory split, D2, and D3 were all reviewed and decided on 1 Sep 2026.

---

## 1. The five research questions

| RQ | Question | Primary metric |
|----|----------|----------------|
| RQ1 | Is the judge's stated confidence calibrated? | ECE (equal-mass bins), Brier + decomposition, signed overconfidence gap |
| RQ2 | Is any cheap uncertainty signal *informative* about error? | AUROC(uncertainty → error), risk–coverage curve, AURC |
| RQ3 | Does uncertainty flag bias-induced errors, or is the fooled judge confident? | flip rate, mean confidence on flipped vs unflipped, paired ΔECE/ΔAUROC |
| RQ4 | Can a cheap supervised meta-model beat the best single signal at predicting judge error? | AUROC under repeated `StratifiedGroupKFold`, vs best-single-signal baseline and permutation null |
| RQ5 | Does marginalizing over the judge prompt improve uncertainty quality, and does that survive distillation to single-call cost? | Ensemble vs single-call AUROC/ECE/entropy-quality gap; aleatoric-vs-`d_human` validation |

RQ4 is the owner's own idea and RQ5 was added 31 Aug 2026 from professor feedback (`DECISIONS.md` D18–D24) — both are binding, not optional extensions (`CLAUDE.md` §1).

## 2. Primary endpoint (decided)

**RQ4's AUROC vs. the best-single-signal baseline (from RQ2), validated against a permutation null.** RQ4 is the culmination, not RQ2 — RQ2's signals and its AUROC(uncertainty → error) result are the *baseline RQ4 has to beat*, not the headline in their own right. `CLAUDE.md` flags RQ4 as "the owner's own idea... not optional... protect it" — the single strongest editorial signal in the project about what matters most — and it was designed from the outset so that every possible outcome is reportable (`PLAN.md` §2.2: "A ≈ best single signal," "B ≫ A," "C ≫ B," or "nothing beats the permutation null" are all a slide, none is a failed project). That makes it a *safer* choice for a primary endpoint, not a riskier one, since the evidentiary bar was deliberately built to survive a negative result. The primary endpoint decides slide #1 of the defense and the one-sentence framing of the abstract — it does not change what gets built (everything in RQ1–RQ5 is delivered regardless).

## 3. Confirmatory vs exploratory (decided)

**Confirmatory (planned in advance, reported regardless of outcome):** RQ1, RQ2, RQ4's core (Tiers A/B, permutation null, H4's continuous form).

**Exploratory (hypothesis-generating, reported as such, not treated as pre-registered claims):** RQ3 (the specific magnitude of the verbosity/position effects wasn't predicted numerically in advance), Tier C of RQ4 (contingent on `judge.py` capturing the full per-token logprobs cleanly and `parse.py` deriving the CoT aggregates from them correctly, D4), RQ5 in full (new as of 31 Aug 2026, no prior numeric prediction beyond the qualitative "epistemic should beat total" from D20), and every secondary/tertiary fallback listed in `DECISIONS.md` (D9's bucketed H4 variant, D12's transfer-test `LeaveOneGroupOut`).

This is a purely interpretive distinction for how `REPORT.md` discusses findings — mainly to avoid presenting a hypothesis-generating result as though its exact magnitude had been predicted in advance. It is not a scope decision: every RQ1–RQ5 item, confirmatory or exploratory, is still built and delivered. The split matches `PLAN.md` §4's drop order — confirmatory items sit below the "protect everything below this line" marker; exploratory items sit above it.

## 4. D1 — Tie policy (decided)

**Decision: `mark_tie_strict`.**

Four candidates were implemented in `src/data.py::build_items()` and compared empirically on the real vote distribution before deciding:

| | `drop_ties` | `count_half` | `mark_tie_lenient` | `mark_tie_strict` |
|---|---|---|---|---|
| N total | 2396 | 2396 | 2396 | 2396 |
| N non-tie | 1858 | 1858 | 1762 | **1904** |
| N ≥2 votes | 536 | 761 | 761 | 761 |
| N ≥3 votes | 94 | 162 | 162 | 162 |
| N unanimous | 1827 | 2114 | 2142 | **2273** |
| N contested | **101** | 282 | 254 | **123** |

**Reasoning:**
- A "tie" ballot is different information from a weak preference for either side, not a 0.5-strength vote for both. `drop_ties` and `count_half` both try to fold it into a continuous fraction anyway; `mark_tie_strict` treats it as a categorical property of the item instead (`is_tie`), never smoothed into `frac_prefer_a`.
- `drop_ties` put `N contested` at exactly 101 — one rounding away from losing H4's bucketed secondary test at Gate 0 (`TASKS.md` Gate 0's `<100` cutoff). `mark_tie_strict` sits at 123, clear of that boundary but still realistic (not artificially inflated the way `count_half`/`mark_tie_lenient`'s 254–282 are, which come from counting every tie ballot as evidence of "disagreement").
- **Strict vs. lenient** — the deciding sensitivity check: a group of 2 tie / 2 A / 1 B is marked fully tied under lenient (`n_tie >= n_a`), discarding the fact that A beat B 2-to-1 among decisive voters. Strict (`n_tie > n_a and n_tie > n_b`) requires tie ballots to *clearly* dominate before declaring the item undecided, preserving that real signal.
- This mirrors MT-Bench's own S1 (non-tie) primary-analysis convention (Zheng et al., `LEARNING.md` A2) rather than being an arbitrary choice among four options.

Full implementation and unit tests: `src/data.py::build_items()`, `tests/test_data.py`.

## 5. D2 — Label construction (decided)

**Decision:** `human_label` (used from `items.parquet` onward, task 2.2) = `majority_label` from `build_items()`, restricted to items where `is_tie == False`. Items where `is_tie == True` are excluded from the judge-evaluation population entirely — there is no genuine human ground truth to score a forced-choice judge against when humans themselves couldn't converge on a winner (the judge is constrained to `{A, B}` with no tie option, `CLAUDE.md` §3, so it can never express agreement with a "tied" human verdict in the first place).

**The ≥150-item fallback:** the concern this guards against is too few usable (non-tied) items to run the study meaningfully. Under D1's decision, **N non-tie = 1904** — comfortably clear of 150. The fallback is not triggered; no further action needed on this axis. (Had it been triggered, the intended contingency was reconsidering D3's augmentation option or relaxing the tie policy — noted for completeness, not needed here.)

## 6. D3 — RewardBench 2 augmentation (decided)

**Decision: not needed, for now.** RewardBench 2 augmentation exists as a contingency for exactly the low-N scenario D2's fallback describes. With 1,904 usable non-tie items from MT-Bench alone — an order of magnitude above the 150-item floor — there's no data-scarcity justification for adding a second, differently-constructed dataset, which would also introduce its own methodological questions (different annotator pools, different prompting conventions) that aren't worth taking on without a concrete need. **Revisit only if a later gate reveals a real problem** (e.g. an unexpectedly thin subgroup for a specific analysis) — not a decision to reopen speculatively.

## 7. Gate 0 numbers

Computed by `python -m src.data --config configs/run.yaml` (D1's decided policy):

- N total items: **2396**
- N non-tie (usable under D2): **1904**
- N with ≥2 votes: **761**
- N with ≥3 votes: **162**
- N unanimous: **2273**
- N contested: **123** — **≥100, so H4's bucketed secondary comparison stays viable** (`TASKS.md` Gate 0 would otherwise have dropped it in favor of D9's continuous-only version).
- **Human–human Cohen's κ: 0.683** (N = 536 item-pairs). Method: non-tie votes only, one pair per item with ≥2 non-tie votes (not all pairwise combinations, so every pair is an independent observation), the two votes chosen deterministically by sorting each item's votes by annotator ID and taking the first and last (`src/data.py::human_human_kappa()`). This is the ceiling on everything downstream — no judge, however well-calibrated, should be expected to exceed human-human agreement on the same task.

## 8. ⚑-marked decisions, D4–D24

The full text and reasoning for each lives in `DECISIONS.md`, not duplicated here — copying ~15 decisions verbatim would create two documents that could silently drift out of sync. This is the binding list; each is in force as of Gate 1.

- **D4** — Full per-token logprobs captured at generation time for every call (100% coverage, amended 4 Sep 2026), so `src/parse.py` can derive the Tier C CoT aggregates and verdict-level logprob signals from saved data - never reconstructed from `raw_output` text alone.
- **D5** — `order` is an axis orthogonal to `condition`; `swap` is not a condition.
- **D6** — Two temperature config keys, never one; `conf_lp`/`p_a` from the canonical greedy call only.
- **D7** — `judge_verdict` (primary, canonical AB) vs `verdict_bidir` (secondary, order-averaged) — both reported.
- **D8** — `StratifiedGroupKFold(shuffle=True)`, repeated over 10 seeds, not plain `GroupKFold`; no hyperparameter tuning at 80 groups.
- **D9** — Human consensus (`d_human`) as a continuous covariate, primary; bucketed agreed/contested secondary, gated on ≥100 contested items.
- **D14** — `ece()`'s `"auto"` strategy: exact binning for discrete signals, quantile with duplicates dropped otherwise; always returns `n_effective_bins`.
- **D15** — H4's interaction CI is a cluster bootstrap over out-of-fold predictions, never a default standard error.
- **D16** — `human_agreed` requires `n_human_votes >= 2`; a single-vote item is trivially "unanimous" and must not count.
- **D18** — `attribution` condition dropped entirely (supersedes D13).
- **D19** — P1/P2/P3 prompt-ensemble axis; new 12-calls/item schedule (supersedes D5/D12's numbers).
- **D20** — `conf_ens` + judge-level entropy decomposition; `items.parquet`'s grain becomes `(item_id, condition, prompt_variant)`; filter to `prompt_variant == "P1"` (and `condition == "clean"` for RQ1/RQ2/RQ4's core) everywhere in RQ1–RQ4 (invariant 14).
- **D21** — `conf_sc`/`conf_ens` are clean/P1-only; the clean→verbose transfer test and verbose-shift check both use Tier A minus those signals.
- **D22** — Bayesian hierarchical logistic regression joins RQ4; held-out random intercepts marginalized over the population prior, never fitted; convergence diagnostics mandatory.
- **D23** — RQ5: ensemble distribution as teacher, single-call Bayesian model as student; aleatoric estimate validated against `d_human`/H4.

Not ⚑-marked (operational/workflow, not methodologically binding — still real decisions, just not ones that change what gets measured): D10 (pilot scope), D11 (vLLM pinning), D12 (Colab budget method), D13 (superseded by D18), D17 (local/Colab split), D24 (new dependencies).
