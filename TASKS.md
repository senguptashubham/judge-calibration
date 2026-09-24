# TASKS.md

Atomic tasks with a definition of done. Feed one at a time to Claude Code: *"Do task 1.4 from TASKS.md."*
Read `CLAUDE.md` §2 (invariants) and `DECISIONS.md` (D4–D28) before any task touching statistics or the harness.

Legend: **[C]** code · **[A]** analysis · **[W]** writing · **[L]** learning · **⛔** gate

Closeout notes record the result and where it lives. The reasoning behind each choice is in `DECISIONS.md`; the full write-up is in `REPORT.md`.

---

## Week 0 · Aug 24–30 · 9h — Data reality check

- [x] **0.1 [C]** Init repo: `.gitignore`, `pyproject.toml` (python ≥3.11), directory skeleton from `CLAUDE.md` §4. Base install has no `vllm`; an optional `colab` extra adds it (D17). All deps pinned with `==`. Dedicated conda env `judge-calib`. Push to a GitHub remote — that is how Colab gets the code (D17).
  **DoD:** `pytest` runs and collects 0 tests without error. `git log` has one commit, pushed. Every base-install dep has an `==` pin.

- [x] **0.2 [C]** `configs/run.yaml` + a `Config` dataclass loader in `src/config.py`. Keys include **`temperature_canonical: 0.0`**, **`temperature_sc: 0.7`**, **`k_sc: 4`**, `logprobs: 20`, **`conditions: [clean, verbose]`** (no `swap` — D5; no `attribution` — D18), **`prompt_variants: [P1, P2, P3]`** (D19).
  ⚠️ There must be **no single `temperature` key** — one key would make all k draws identical and `conf_sc` a dead constant (D6).
  **DoD:** `Config.from_yaml()` round-trips. Tests assert `temperature_sc > 0`, `"swap"`/`"attribution"` rejected, `"P1"` required.

- [x] **0.3 [L]** Read Guo et al. §§1–4.2 (`LEARNING.md` A1).
  **DoD:** you can write the ECE formula from memory and say why temperature scaling can't change accuracy.

- [x] **0.4 [C]** `src/metrics.py::ece(confidences, correct, n_bins, strategy)` with `strategy ∈ {"uniform", "quantile", "auto"}`, default `"auto"`: if `n_unique(conf) <= n_bins` bin by unique value, else quantile with duplicate edges dropped. **Returns `(ece, n_effective_bins)`** (D14).
  **DoD:** `test_ece_reference` passes on `CLAUDE.md` §5's hand-computed case (**0.222**, tol 1e-3); `test_ece_discrete` returns `n_effective_bins == 5` on a 5-level signal without raising.

- [x] **0.5 [L]** Read MT-Bench §§3–4 + Tables 2, 4, 5 (`LEARNING.md` A2).
  **DoD:** the six numbers in A2 are in your notes, and you can state the S1/S2 distinction.

- [x] **0.6 [C]** `src/metrics.py::cohens_kappa(a, b)` from first principles (p_o, p_e, κ).
  **DoD:** matches `sklearn.metrics.cohen_kappa_score` on 3 random arrays; `test_kappa_balanced` asserts p_o=0.85, p_e=0.5 → κ=0.70.

- [x] **0.7 [C]** `src/data.py::load_votes()` — `lmsys/mt_bench_human_judgments`, split `human`.
  **DoD:** returns 3,355 rows; fails loudly if the dataset shape changed.

- [x] **0.8 [C]** `src/data.py::build_items()` — aggregate votes to items keyed by `(question_id, model_a, model_b, turn)`: `n_human_votes`, `frac_prefer_a`, `majority_label`, `human_unanimous`, `is_tie`, **`d_human`** (D9), **`human_agreed`** (D16), tie policy from config.
  **DoD:** `results/items_labels.parquet` written; summary prints N total, N non-tie, N ≥2/≥3 votes, N unanimous, N contested.

- [x] **0.9 [A]** Human–human Cohen's κ on items with ≥2 non-tie votes.
  **DoD:** a number, with N, in `PREREGISTRATION.md`.

- [x] **0.10 [W]** `PREREGISTRATION.md`: the RQs, D1 (tie policy), D2 (label construction + ≥150-item fallback), D3 (RewardBench augmentation), the ⚑-marked decisions D4–D25, the primary endpoint, which analyses are exploratory.
  **DoD:** committed, written before looking at any judge output.

- [x] ⛔ **GATE 0** — passed. N total 2,396, non-tie 1,904, ≥2 votes 761, unanimous 2,273, contested 123; human–human κ 0.683 (N=536). D1–D3 decided (`PREREGISTRATION.md`).

---

## Week 1 · Aug 31–Sep 6 · 14h (peak week) — Harness

- [x] **1.1 [L]** Course: Getting Structured LLM Output (`LEARNING.md` B1).
  **DoD:** you can explain how guided decoding masks logits per token, and why that means constraining to exactly `{A, B}` and reading the renormalised `p_a`.

- [x] **1.2 [L/C]** Course: vLLM lessons 3, 6, 7, 8 + vLLM docs. Pin `vllm` in a fresh Colab venv (D11).
  **DoD:** a scratch script exercises guided decoding AND `logprobs=20` together. Result: `vllm==0.28.0`, `StructuredOutputsParams`.

- [x] **1.3 [C]** `src/prompts.py` — three pairwise templates (D19): **P1** (general MT-Bench rubric), **P2** (correctness-first), **P3** (helpfulness-first), each versioned + hashed, one shared JSON output contract.
  **DoD:** `prompt_hash()` stable for all three; each renders a real item. Frozen at Gate 1 (invariant 10).

- [x] **1.3b [L]** Theory I — Bayesian inference + NumPyro/NUTS basics (D22).
  **DoD:** you can explain what NUTS samples, why hierarchical intercepts shrink toward the population mean, and why that's the standard fix for too few groups.

- [x] **1.4 [C]** `src/judge.py` — vLLM wrapper: batched generation, verdict constrained to `{A, B}`, `logprobs=20`, JSONL append-checkpoint keyed by `(item_id, condition, prompt_variant, order, sample_idx)`, skip-completed on restart. Schedule (D19): `(clean, P1)` 6 calls (2 greedy × both orders + 4 sampled AB); `(clean, P2)`, `(clean, P3)`, `(verbose, P1)` 2 greedy calls each. Full per-token logprobs saved for every call (D4). No derived signals computed here.
  **DoD:** running twice does not duplicate rows; a kill loses at most one batch; a logprobs file exists for every call; `sample_idx > 0` only ever appears for `(clean, P1)`.

- [x] **1.5 [C]** `src/parse.py` — verdict, verbalized confidence, verdict-token logprob, renormalised `p_a`, CoT logprob aggregates, failure taxonomy (`CLAUDE.md` §3), all from `raw_output` + the saved logprobs.
  **DoD:** a `sample_idx=0` row has all six CoT aggregate columns populated. *The real-malformed-output fixture requirement was dropped: 220 real generations produced zero malformed outputs (structured decoding makes most failure types structurally impossible), so `test_parse.py` uses hand-built cases.*

- [x] **1.6 [C]** Pilot: 20 items × `(clean, P1)` × 6 calls, plus a P2/P3 smoke test, a 10-item verbose smoke test (prompt-token multiplier), and the T=0 vs T=0.7 `p_a` probe (D6).
  **DoD:** parse failure < 5% across P1–P3 — **actual 0%**. GPU hours extrapolated in `PLAN.md` §2.2b. Temperature-probe result in `REPORT.md`'s methods notes.

- [x] **1.7 [C]** `src/signals.py` — `conf_verb`, `conf_lp` (both `sample_idx=0` only, D6), `conf_sc` (`(clean, P1)` only), `conf_bpe`, plus `judge_verdict`/`verdict_bidir` (D7).
  **DoD:** all four populated for the 20 pilot items, no NaNs; a test asserts `conf_lp` is never read from `sample_idx != 0`.

- [x] **1.8 [A]** Vacuum test: 40 identical-response pairs + 20 empty-response pairs.
  **DoD:** a "dark current" number in `REPORT.md` (the judge picks a winner 100% of the time, with confidence ≥ real items').

- [x] **1.9 [L]** Read Tian and Xiong (`LEARNING.md` A3, A4).
  **DoD:** you can state Xiong's negative AUROC result and why it predicts a flat risk–coverage curve.

- [x] ⛔ **GATE 1** — passed. Pilot parse failure 0% across P1–P3, resume proven on real Colab runs, GPU budget written down, `PREREGISTRATION.md` and `src/prompts.py` frozen.

---

## Week 2 · Sep 7–13 · 13h — Clean run (P1+P2+P3) + RQ1

- [x] **2.1 [C]** Full `clean` run, all items, all three prompt variants.
  **DoD:** `results/calls_qwen2.5_7b_instruct.parquet` complete for `clean` — N × 10 calls, no missing keys, CoT aggregates populated for every `(clean, P1)` row.

- [x] **2.2 [C]** `src/signals.py` CLI: calls → `results/items_{model_slug}.parquet` at the `(item_id, condition, prompt_variant)` grain (D20).
  **DoD:** `correct` populated; a schema test checks `prompt_variant ∈ {P1, P2, P3}` on `clean` rows and `== P1` on every `verbose` row.

- [x] **2.2b [C]** `src/signals.py::conf_ens()` — judge-level entropy decomposition over P1/P2/P3 (D20): Total, Aleatoric, Epistemic = Total − Aleatoric, `conf_ens = 1 − Total`.
  **DoD:** populated on every `clean` P1 row, null on P2/P3 rows and on every `verbose` row (the D21 check).

- [x] **2.3 [L]** Block C4 — the ECE⊥AUROC counterexample figure (`LEARNING.md` C4).
  **DoD:** `results/figures/ece_auroc_orthogonal.png`, narratable in one sentence.

- [x] **2.4 [C]** `src/boot.py` — `cluster_bootstrap` and `paired_cluster_bootstrap`.
  **DoD:** `tests/test_boot.py` shows clustered CIs are strictly wider than naive row CIs on the same data.

- [x] **2.5 [C]** Finish `src/metrics.py`: `mce`, `brier`, `brier_decomposition`, `overconfidence_gap`, `auroc_error`.
  **DoD:** each unit-tested; `brier_decomposition` reconstructs the Brier score to 1e-6 on a discrete signal.

- [x] **2.6 [A]** **RQ1** on `(clean, P1)` (invariant 14): reliability diagram, ECE, MCE, Brier + decomposition, overconfidence gap, accuracy, κ — cluster-bootstrap CIs, for both `judge_verdict` and `verdict_bidir` (D7).
  **DoD:** `reliability_{signal}_{model_slug}.png` ×4 + `results/rq1_table_{model_slug}.csv`.

- [x] **2.6b [L]** Read A8 (Dark Current) and A6 (Reliability without Validity).
  **DoD:** you can state why κ is mandatory and what the "true vacuum" probe tests.

- [x] **2.7 [W]** `REPORT.md` RQ1 section.
  **DoD:** written.

- [x] ⛔ **GATE 2** — passed. N=1,836. `conf_verb` overconfidence gap 0.192 [0.172, 0.213] (`judge_verdict`); order-averaging lifts accuracy 0.757 → 0.792, paired gain +0.035 [0.018, 0.053]. `conf_ens` populated on all 1,904 clean P1 rows, null elsewhere.

---

## Week 3 · Sep 14–20 · 11h — RQ2 + human-disagreement + threshold sweep

- [x] **3.1 [C]** `src/metrics.py::risk_coverage`, `aurc`, and the oracle curve.
  **DoD:** unit-tested; the oracle dominates every real signal by construction.

- [x] **3.1b [C]** `src/metrics.py::threshold_sweep` — sweep raw signal thresholds, reporting coverage, accuracy, κ, and ECE on the retained set at each (D20).
  **DoD:** unit-tested against a synthetic signal with a known answer.

- [x] **3.2 [A]** **RQ2** on `(clean, P1)`: risk–coverage per signal + oracle, AURC, accuracy/κ@{90,75,50}% coverage, AUROC(uncertainty→error), cluster-bootstrap CIs.
  **DoD:** `risk_coverage_{model_slug}.png` (the thesis figure) + `results/rq2_table_{model_slug}.csv`.

- [x] **3.2b [A]** **RQ5 threshold sweep** (D20, D23) on `ens_entropy_{total,aleatoric,epistemic}`; test the preregistered prediction that epistemic beats total.
  **DoD:** `entropy_threshold_sweep_{model_slug}.png` + a one-sentence verdict.

- [x] **3.3 [A]** Human-disagreement decomposition — primary: continuous regression of correctness and of confidence on `d_human` (D9); secondary bucketed version only if ≥100 contested items.
  **DoD:** `human_disagreement_{model_slug}.png` + the finding in one sentence. *Bucketed version out of scope: only 31 contested items survive the ≥2-vote filter.*

- [x] **3.4 [A]** Correlation between each of the four original signals and `d_human`.
  **DoD:** four Spearman correlations with CIs in `REPORT.md`.

- [ ] **3.4b [L]** Read A7 (Trust or Escalate) and A5 (SCOPE).
  **DoD:** you can define BPE in one sentence and say what SCOPE leaves open.

- [ ] **3.4c [L]** Theory K — BALD / mutual information (D20).
  **DoD:** you can derive Epistemic = Total − Aleatoric and explain why a position-biased judge maximizing `conf_bpe`'s entropy is the same phenomenon as an ensemble maximizing epistemic uncertainty.

- [x] **3.5 [W]** `REPORT.md` RQ2 section, including the threshold-sweep result.
  **DoD:** written.

- [x] ⛔ **GATE 3** — passed. Oracle AURC 0.0295 vs 1.3–4.7× higher for every real signal; `conf_bpe` best on AUROC (0.794), `conf_sc` best on AURC (0.038, driven by its discreteness) — both reported. Epistemic-beats-total prediction **rejected**: paired AURC gap +0.175 [0.141, 0.207]. 3.4b/3.4c left to the owner's pace.

---

## Week 4 · Sep 21–27 · 8h — RQ3

- [x] **4.1 [C]** `src/perturb.py::verbose_pad()` (repetitive-list attack, Zheng §3.3). `attribution()` is cut (D18).
  **DoD:** `tests/test_perturb.py` shows `verbose_pad` leaves the original content intact.

- [x] **4.1b [C]** Re-extrapolate the GPU budget with real padding (D10, D12).
  **DoD:** updated estimate in `PLAN.md` §2.2b — ~35 compute units against 72.8 available, N not cut. *Found and fixed along the way: `verbose_pad()` had never been wired into `judge.py`, so the first verbose smoke test was unpadded.*

- [x] **4.2 [C]** `verbose` run: P1 only, both orders, greedy — 2 calls per item.
  **DoD:** calls/items rebuilt; `conf_sc`/`conf_ens` null on every verbose row (D21). Result: 3,808 verbose rows, parse_ok 100%, prompt length ≈3.1× clean's.

- [x] **4.3 [A]** **RQ3a** — position bias within `(clean, P1)`: flip rate between orders; mean confidence on flipped vs unflipped items, cluster-bootstrap CI.
  **DoD:** the RQ3 money sentence in `REPORT.md`. Result: flip rate 27.8% [24.8%, 30.8%]; `conf_verb` gap −0.020 [−0.025, −0.015].

- [x] **4.4 [A]** **RQ3b** — paired ΔECE/Δaccuracy/ΔAUROC clean(P1)→verbose(P1) for `conf_verb`, `conf_lp`, `conf_bpe` (`conf_sc` has no verbose data, D21).
  **DoD:** `results/rq3_table_{model_slug}.csv` + the `conf_sc` exclusion noted in `REPORT.md`. Result: accuracy unchanged (+0.011, ns); AUROC drops significantly for all three signals; only `conf_bpe`'s calibration breaks (ΔECE +0.043 [0.011, 0.052], scored on `conf_bpe_prob`).

- [x] **4.5 [A]** Constrained vs free-form decoding on 100 items: parse rate and verdict agreement (D25).
  **DoD:** a limitations paragraph in `REPORT.md`. Result: 100%/100% parse rate; 96% verdict agreement — the 4 flips are all high-confidence, so constraining genuinely moves some verdicts.

- ⛔ **GATE 4** — passed. RQ3a and RQ3b answered with CIs (numbers above); verbosity doesn't fool the judge's verdict but breaks the uncertainty signal meant to catch its errors.

---

## Week 5 · Sep 28–Oct 4 · 15h (second peak week) — RQ4 + RQ5 (protect this week)

- [ ] **5.1 [L]** Read A9 (Know When You're Wrong) and A10 (Meta-Judges).
  **DoD:** you can say in two sentences why RQ4 is neither of them.

- [ ] **5.1b [L]** Theory J (partial pooling) and Theory L (distributional distillation) (D22/D23).
  **DoD:** you can state why a held-out group's intercept must be marginalized over the population prior, and what "the ensemble's predictive distribution is the teacher" means concretely.

- [x] **5.2 [C]** `src/features.py` — tiers A, B, C as in `PLAN.md` §2.2, on `(clean, P1)` only. Tier C carries both `*_greedy` and `*_sampled_t07` CoT aggregates (D4).
  **DoD:** `tests/test_features.py` asserts column sets, no leakage of `correct`/`correct_bidir`/`human_label`/`frac_prefer_a`, no NaNs. Population: 1,819 — the 17 items with undefined `len_ratio`/`longer_is_chosen` are dropped from all three tiers, so tier comparisons stay same-population.

- [x] **5.3 [C]** `src/predictor.py` — `StratifiedGroupKFold(5, shuffle=True)` on `question_id`, 10 seeds (D8); `LogisticRegression(C=1.0)` on standardized features and `HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)`; no tuning.
  **DoD:** `tests/test_predictor.py` asserts no `question_id` in both train and test of any fold, and different seeds give different fold assignments. Result: Tier A/logreg AUROC 0.796 [0.793, 0.799].

- [x] **5.4 [C]** `src/predictor.py::permutation_null(n=200)` — shuffle `correct` within each question_id, one grouped 5-fold CV per permutation.
  **DoD:** null near chance (Tier A/logreg: 0.507 ± 0.023); observed AUROC (0.796) at the 100th percentile. *Switched from a global shuffle on 24 Sep 2026 - within-question keeps each question's error rate, the stricter null for clustered labels; the global version (0.493 ± 0.022) gave the same verdict (D8 amendment).*

- [x] **5.5 [A]** Tier ablation A → B → C, both frequentist models, on `human_agreed` items (D16); baseline = best single signal, recomputed on this population.
  **DoD:** `results/rq4_ablation_{model_slug}.csv` + bar chart. Result: N=556; baseline `conf_bpe` 0.820; every tier/model sits at the 100th percentile of its within-question permutation null (null means 0.53–0.57, observed 0.80–0.82), but every paired step (baseline→A→B→C) has a CI crossing zero — the tiers are statistically indistinguishable.

- [x] **5.6 [A]** **H4**, continuous form (D9, D15): `correct ~ oof_score * d_human` on every item with ≥2 votes, OOF scores from the 10×5 CV, cluster-bootstrap (B=2000, refit per resample) CI on the interaction.
  **DoD:** the interaction coefficient with its CI and a one-sentence reading. Result: N=595, +2.375 [1.685, 3.074] — supports H4.

- [x] **5.7 [A]** Transfer test 1: train on `clean`, test on `verbose`, with Tier A minus `{conf_sc, conf_ens, ens_entropy_*}` for every model (D21).
  **DoD:** ΔAUROC reported. Result: logreg +0.0001, histgbm −0.025 (CIs overlap) — the combined model transfers with little degradation, refuting the "safety net fails under attack" hypothesis.

- [x] **5.8 [A]** Transfer test 2: `LeaveOneGroupOut` over the 8 MT-Bench categories.
  **DoD:** per-category held-out AUROC. Result: 0.739–0.864 (logreg), every category well above chance; weakest are writing and reasoning, not coding.

- [x] **5.9 [A]** Meta-model calibration (frequentist): reliability diagram + ECE of P(judge wrong); logistic-regression coefficients with cluster-bootstrap CIs (Tier C + logreg).
  **DoD:** `rq4_coefficients_{model_slug}.png` (significant coefficients) + `_full` appendix figure + CSV. Result: OOF AUROC 0.799, ECE 0.036; 8/37 coefficients clear zero — 5 in Tier A, 3 in Tier B, none in Tier C.

- [x] **5.9b [C]** `src/bayesian.py` — hierarchical logistic regression (D22), NUTS under the same 10×5 protocol, held-out intercepts marginalized over the population prior, convergence diagnostics for every fold-fit.
  **DoD:** `tests/test_bayesian.py` asserts held-out predictions don't depend on that fold's fitted `α_q`; diagnostics reported; runtime measured. Result: NUTS rung used (no fallback); non-centered parameterization required to avoid Neal's funnel (D22 amendment); full 50-fit run ≈9 min at 2,000 draws per chain. *24 Sep 2026: features now standardized per training fold, as for LogReg; R-hat read unrounded and draws doubled (D22 amendments) — all Bayesian results re-run.*

- [x] **5.9c [A]** Head-to-head: LogReg vs Bayesian model, Tier A — AUROC, ECE, Brier, NLL, bin-aggregate 90% coverage.
  **DoD:** `results/rq4_bayesian_comparison_{model_slug}.csv` + reliability and convergence figures. Result: the two models match - AUROC 0.795 vs 0.796, ECE 0.034 vs 0.037, Brier 0.143 both; Bayesian NLL 0.445, coverage_90 0.80; 0/50 fold-fits flagged (worst R-hat 1.008).

- [x] **5.9d [A]** **RQ5 distillation** (D23): the 3-prompt ensemble vs the single-call Bayesian model on AUROC, ECE, and entropy quality (AUROC of each one's epistemic signal).
  **DoD:** `results/rq5_distillation_{model_slug}.csv` + figure + headline sentence. Result: the 1-call model keeps 98.7% of the ensemble's AUROC edge (paired gap +0.004 [−0.011, 0.020]); its epistemic signal is a far better error predictor (0.779 vs 0.559, gap −0.220 [−0.267, −0.172]); ECE 0.034 vs the ensemble's 0.071. Gaps in `rq5_distillation_gaps_{model_slug}.csv`.

- [x] **5.9e [A]** **RQ5 human-disagreement validation** (D23): Spearman of `ens_entropy_aleatoric` and `ens_entropy_epistemic` against `d_human` (high `d_human` = strong consensus, so aleatoric should correlate *negatively*).
  **DoD:** both correlations with CIs and a one-sentence reading. Result: aleatoric −0.118 [−0.204, −0.016], epistemic +0.014 [−0.071, 0.099] — the decomposition validates.

- [x] **5.9f [A]** **Verbose-shift validation** (D21, D23): fit the Bayesian model once on `clean`, evaluate its meta-model-level decomposition on `clean` and `verbose` via `predict_in_sample()` (never `predict_held_out()`, D21 amendment). Prediction: epistemic rises, aleatoric flat.
  **DoD:** both components on both conditions, with the verdict. Result: prediction **did not hold** — both components fall slightly under `verbose` - aleatoric −0.020 [−0.030, −0.010], epistemic −0.0004 [−0.0006, −0.0001] (`results/rq5_verbose_shift_{model_slug}.csv`).

- [x] **5.10 [W]** `REPORT.md` RQ4 section, framed as *which feature family carries the signal*.
  **DoD:** written (drafted by Claude at the owner's request, every number checked against the CSVs).

- [x] **5.11 [W]** `REPORT.md` RQ5 section: 5.9d, 5.9e, 5.9f.
  **DoD:** written (as 5.10).

- ⛔ **GATE 5** — RQ4 answered against the permutation null and the best-single-signal baseline, for both frequentist and Bayesian models, with convergence diagnostics; H4 tested; RQ5 answered.
  **Status:** all underlying conditions met; left unchecked pending the owner's review of the drafted report sections.

---

## Addendum · RQ6 — kev-8b industry-counterexample stress-test (D27, owner-initiated)

A full RQ (`CLAUDE.md`, `PLAN.md` §7), numbered K1–K5 outside W0–W7 so it can be isolated or trimmed later without renumbering.

- [x] **K1 [C]** Candidate research: Jev (proprietary) vs three open `/v1/systemone` stand-ins vs the rejected `SemIf` cluster.
  **DoD:** a documented decision (D27). kev-8b selected — the only candidate without a disqualifying token cap for this project's prompt lengths.

- [x] **K2 [C]** Diagnostic probing of kev-8b's real token-length behavior and launch configuration.
  **DoD:** a confirmed serving ceiling with population coverage checked. Result: 8,160 tokens (non-deterministic OOM above ~8,165); launch needs `KEV_MERGE=0` and `setsid` (D27).

- [x] **K3a [C]** `src/judge_kev.py` + `configs/run_kev.yaml`; run clean+verbose, both orders.
  **DoD:** a complete, verified `runs/kev_8b/kev.jsonl`. Result: 7,616 rows, 104 verbose calls (52 items) skipped over the ceiling, 0 duplicates. The run predates `call_kev()`'s full-response capture; the one dropped field (`confidence`) is an exact function of `probabilities` (D27), and `kev_signals.py` reads both checkpoint layouts.

- [x] **K3b [C]** `calls_kev_8b.parquet` / `items_kev_8b.parquet`: `judge_verdict`/`correct`, order-corrected `verdict_bidir`, `conf_kev`, `conf_kev_bpe`, `flipped`.
  **DoD:** items table for every non-skipped row. Result: 3,808 item rows; 1,852 verbose items with a verdict, of which 1,784 are also human-labeled — the paired verbosity population.

- [x] **K4 [A]** Calibration, position-swap, verbosity attack, and Bayesian recalibration, split by coverage regime (`input_tokens` ≤ 1,024).
  **DoD:** every metric with a CI, both signals, both regimes — `results/rq6_*_kev_8b.csv`. Result: `conf_kev_bpe` beats `conf_kev` on calibration in both regimes; `conf_kev`'s position-swap gap (−0.096) is ~5× the primary judge's `conf_verb`; verbosity does not break either signal's calibration (`conf_kev_bpe` ΔECE −0.029 in-coverage, ns out); the Bayesian meta-model does not beat the best single signal (0.772 vs 0.772 in-coverage, 0.775 vs 0.781 out; 7/50 and 1/50 fold-fits flagged). *24 Sep 2026: calibration re-scored on `conf_kev_bpe_prob` (D7 amendment) — the earlier "calibration breaks under verbosity" reading was the entropy scale.*

- [x] **K5 [W]** `REPORT.md` RQ6 section, out-of-domain caveat stated up front.
  **DoD:** written, every claim traceable to a results file.

- ⛔ **GATE K** — kev-8b tested against the same battery; out-of-domain caveat prominent.
  **Status:** all conditions met; left unchecked pending the owner's review.

---

## Addendum · RQ7 — auto-j-13b purpose-built-judge generalization test (D28, owner-initiated)

A full RQ (`CLAUDE.md`, `PLAN.md` §8), numbered L1–L6 for the same reason as the K-block.

- [x] **L1 [C]** Confirm auto-j-13b's serving contract, license, context length, and training-data provenance from primary sources.
  **DoD:** the verbatim prompt template and verdict extractor from the authors' repo, plus a contamination assessment. Result: `[INST]…[/INST]` + `pairwise_tie` template, `rfind("final decision is ")` extraction, 8,192-token context, Llama 2 license; MT-Bench's questions are hand-crafted, so leakage from auto-j's Arena training data is unlikely (D28).

- [x] **L2 [C]** GPU/quantization smoke test on the pinned `vllm==0.28.0`.
  **DoD:** a real load with VRAM/timing numbers. Result: `GAIR/autoj-13b-GPTQ-4bits` loads on an L4 via vLLM's Marlin kernel — 6.8 GiB weights, 10.9 GiB KV cache; fp16 would not fit.

- [x] **L3 [C]** `src/judge_autoj.py` + `configs/run_autoj.yaml`; run clean+verbose, both orders, both turns, with the verbatim auto-j contract.
  **DoD:** a complete, verified raw checkpoint. Result: 15,232/15,232 calls, 0 duplicate keys; 362 calls skipped over the 7,168-token prompt ceiling (almost all verbose/turn=2). Raw output, `finish_reason`, and full logprobs captured for every call. Parse failure (no decision line, always `finish_reason == "length"`): 0.04% clean, 13.6% verbose/turn=1, **39.8% verbose/turn=2**. Turn=2 was briefly dropped after the 100-item smoke test, then restored once most of the both-turns run had completed (D28).

- [x] **L4 [C]** `src/autoj_signals.py`: verdict extraction ported verbatim, `conf_sc_autoj`, the order-swap `conf_sc_bpe_autoj` (self-consistency proportions), its greedy-only variant, `flipped_autoj`, calls/items build with `turn` kept.
  **DoD:** `items_autoj_13b_gptq_4bits.parquet` for every non-skipped row. Tie (2.35% of calls) maps to `judge_verdict=None` but stays distinguishable from parse failure in `pred_label`. A hand-computed pure-position-bias test caught a double-translation bug in `_p_model_a_wins_autoj` before it shipped.

- [x] **L4b [A]** Turn=2 validity check, decided before any calibration number existed.
  **DoD:** a stated verdict. Result: a real problem, confined to verbose/turn=2 (the parse-failure rates in L3); clean/turn=2 is as reliable as turn=1.

- [x] **L5 [A]** Calibration, position-swap, verbosity attack, and Bayesian recalibration, split by turn.
  **DoD:** every metric with a CI — `results/rq7_*_autoj_13b_gptq_4bits.csv`. Result: both signals overconfident (gap 0.09–0.13), AUROC ~0.68; flip rate 12.5%/13.8%, about half the other two judges'. Verbosity (like-for-like `conf_sc_bpe_autoj_greedy`, N=1,137 paired): ΔECE −0.02/−0.03 (ns), **ΔAUROC +0.06/+0.07 (significant)**, turn=2 accuracy −0.042. The Bayesian meta-model does not beat the best single signal (0.680 vs 0.683 turn=1, 0.671 vs 0.681 turn=2; 0/50 and 1/50 flagged). *24 Sep 2026: the first verbosity run compared the pooled signal across conditions and was confounded (D28 amendment); Bayesian re-run with standardized features.*

- [x] **L6 [W]** `REPORT.md` RQ7 section: purpose-built framing up front, the turn=2 failure finding stated prominently, L5's numbers.
  **DoD:** written, every claim traceable to a results file.

- ⛔ **GATE L** — auto-j-13b tested against the same battery; turn=2 finding prominent; three-way comparison stated as the headline.
  **Status:** all conditions met; left unchecked pending the owner's review.

---

## Week 6 · Oct 5–11 · 12h — Demo + writeup

- [ ] **6.1 [L]** Course: Gradio (`LEARNING.md` B3).
- [ ] **6.2 [C]** Demo, offline-first: load `items_{model_slug}.parquet` (P1 rows), browse items, show all five signals (including `conf_ens`) and the verdict. **Build the "trick the judge" toggle first** — flip the response order or apply verbosity padding, watch the verdict change and the confidence not.
  **DoD:** runs on the Nitro 5 with no network. The toggle demonstrates the thesis in under 10 seconds.
- [ ] **6.3 [C]** Optional live call: 4-bit Qwen2.5-1.5B/3B via `bitsandbytes`, one item on demand.
  **DoD:** works, or is cleanly disabled by a config flag.
- [ ] **6.4 [W]** Finish `REPORT.md` from the incremental draft: abstract, method, RQ1–RQ7, limitations, extensions.
  **DoD:** complete, every figure referenced.
- [ ] **6.5 [W]** Slides. Anchor on the RQ2 thesis figure, the RQ4 coefficient plot, the RQ5 distillation result, and the three-judge comparison.
  **DoD:** delivered.
- ⛔ **GATE 6** — demo runs offline; report complete, RQ1–RQ7 all covered.

---

## Week 7 · Oct 12–15 · 4h — Reproducibility + buffer

- [ ] **7.1 [C]** Fresh clone, `pip install -e .`, run every stage from config (`CLAUDE.md` §7), confirm the headline numbers reproduce — NUTS results should reproduce the same conclusions, not necessarily bit-identical samples.
  **DoD:** a `REPRODUCE.md` with the exact commands and the expected numbers.
- [ ] **7.2 [W]** README with the one-paragraph result.
- [ ] **7.3** Leave the rest empty. This is re-run insurance, not spare capacity.
