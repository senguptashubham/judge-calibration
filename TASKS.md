# TASKS.md

Atomic tasks with a definition of done. Feed one at a time to Claude Code: *"Do task 1.4 from TASKS.md."*
Read `CLAUDE.md` §2 (invariants) and `DECISIONS.md` (D4–D17) before any task touching statistics or the harness.

Legend: **[C]** code · **[A]** analysis · **[W]** writing · **[L]** learning · **⛔** gate

---

## Week 0 · Aug 24–30 · 9h — Data reality check

- [x] **0.1 [C]** Init repo: `git init`, `.gitignore` (`runs/`, `results/`, `*.parquet`, `__pycache__`, `.env`, standard Python/editor entries), `pyproject.toml` (python ≥3.11), directory skeleton from `CLAUDE.md` §4.
  **Dependencies split (D17):** base install — `datasets`, `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `matplotlib`, `pyyaml`, `pytest` — has no `vllm`. An optional `colab` extra adds `vllm`, installed only on Colab (task 1.2). **All deps pinned to exact versions** (`==`, not `>=`) once resolved by a first `pip install`; `vllm`'s pin is chosen on Colab in task 1.2 per D11, not here.
  **Local environment (D17):** a dedicated conda env, `judge-calib` (`python=3.11`) — not the owner's general AI/ML conda env, so this project's pins can't be silently violated by an unrelated install.
  **Push to a GitHub remote** once the initial commit exists — this is how Colab gets the code in W1 (D17).
  **DoD:** `pytest` runs and collects 0 tests without error. `git log` has one commit, pushed to a GitHub remote. Every base-install dep in `pyproject.toml` has an `==` pin.

- [x] **0.2 [C]** `configs/run.yaml` + a `Config` dataclass loader in `src/config.py`.
  Keys: `judge_model`, `dataset`, `tie_policy`, **`temperature_canonical: 0.0`**, **`temperature_sc: 0.7`**, **`k_sc: 4`**, `max_tokens`, `logprobs: 20`, **`conditions: [clean, verbose, attribution]`** (no `swap` — D5), `seed`, `n_bins`, `paths`.
  ⚠️ There must be **no single `temperature` key**. One key would make all k draws identical and `conf_sc` a dead constant (D6).
  **DoD:** `Config.from_yaml()` round-trips. A test asserts `temperature_sc > 0` and that `"swap"` is not in `conditions`.

- [x] **0.3 [L]** Read Guo et al. §§1–4.2 (`LEARNING.md` A1).
  **DoD:** you can write the ECE formula from memory and say why temperature scaling can't change accuracy.

- [x] **0.4 [C]** Implement `src/metrics.py::ece(confidences, correct, n_bins, strategy)` supporting `strategy ∈ {"uniform", "quantile", "auto"}`, defaulting to `"auto"`. Docstring states the formula, why quantile beats uniform here, and why unique-value binning is **exact** for a discrete signal rather than a fallback.
  **`"auto"`:** if `n_unique(conf) <= n_bins` → bin by unique value; else quantile with duplicate edges dropped. **Returns `(ece, n_effective_bins)`** (D14).
  **DoD:** `test_ece_reference` passes on the hand-computed case in `CLAUDE.md` §5 (**0.222**, tol 1e-3). Plus `test_ece_discrete`: on a 5-level signal like `conf_sc` with `n_bins=10`, returns `n_effective_bins == 5` and does **not** raise on tied quantile edges.

- [ ] **0.5 [L]** Read MT-Bench §§3–4 + Tables 2, 4, 5 (`LEARNING.md` A2).
  **DoD:** the six numbers in A2 are in your notes, and you can state the S1/S2 distinction.

- [ ] **0.6 [C]** `src/metrics.py::cohens_kappa(a, b)` implemented from first principles (p_o, p_e, κ).
  **DoD:** matches `sklearn.metrics.cohen_kappa_score` on 3 random arrays; `test_metrics.py::test_kappa_balanced` asserts p_o=0.85, p_e=0.5 → κ=0.70.

- [ ] **0.7 [C]** `src/data.py::load_votes()` — pull `lmsys/mt_bench_human_judgments` split `human`, return a tidy DataFrame with the fields in `CLAUDE.md` §3.
  **DoD:** returns 3,355 rows. Asserts the row count and fails loudly if the dataset shape changed.

- [ ] **0.8 [C]** `src/data.py::build_items()` — aggregate votes to items keyed by `(question_id, model_a, model_b, turn)`. Emit `n_human_votes`, `frac_prefer_a`, `majority_label`, `human_unanimous`, `is_tie`. Apply the tie policy from config.
  Also emit **`d_human = |frac_prefer_a − 0.5|`** (continuous consensus strength, D9) and **`human_agreed = human_unanimous AND n_human_votes >= 2`** (D16) — a single-vote item is trivially "unanimous" and must not count as agreed ground truth.
  **DoD:** `results/items_labels.parquet` written. Printed summary: N total, N non-tie, N ≥2 votes, N ≥3 votes, **N unanimous, N contested** — the last two printed separately, because D2's fallback threshold is checked **per subset**, not on the total (D9). H4 is the fragile split, not the overall count.

- [ ] **0.9 [A]** Compute **human–human Cohen's κ** on items with ≥2 non-tie votes (pair up votes, or Krippendorff's α if the vote counts are ragged).
  **DoD:** a number, with N, written into `PREREGISTRATION.md`. This is the ceiling on everything downstream.

- [ ] **0.10 [W]** Write `PREREGISTRATION.md`: the four RQs; decisions **D1** (tie policy), **D2** (label construction + the ≥150-item fallback), **D3** (whether RewardBench 2 augmentation is needed); the primary endpoint; which analyses are exploratory.
  **DoD:** committed. Written *before* looking at any judge output.

- ⛔ **GATE 0** — you know N total, N non-tie, N≥2 votes, **N unanimous, N contested**, and human–human κ. D1–D3 decided and committed. **If N contested < 100, H4's bucketed analysis is dropped now and the continuous version (D9) becomes the sole test — record that in `PREREGISTRATION.md` today, not in W5.**

---

## Week 1 · Aug 31–Sep 6 · 11h — Harness

- [ ] **1.1 [L]** Course: Getting Structured LLM Output, full (`LEARNING.md` B1).
  **DoD:** you can explain how Outlines constrains generation by masking logits per token — and why that means you must constrain to exactly `{A, B}` and read the renormalised `p_a`.

- [ ] **1.2 [L/C]** Course: vLLM lessons 3, 6, 7, 8 + 30 min of vLLM docs on `LLM.generate`, `SamplingParams(logprobs=...)`, structured outputs (`LEARNING.md` B2).
  **Pin `vllm==<exact>` now** and install into a **fresh venv** — Colab's preinstalled `torch` will fight vLLM's pinned `torch`; expect one runtime restart. Budget 30 min (D11).
  **DoD:** a scratch script exercises **guided decoding AND `logprobs=20` together** — that combination is what breaks, not either alone. Note whether your pinned version uses `GuidedDecodingParams` or `StructuredOutputsParams`; the API was renamed.

- [ ] **1.3 [C]** `src/prompts.py` — MT-Bench pairwise template, **explanation before verdict** (CoT), plus a verbalized-confidence line. Version string + `prompt_hash` helper.
  **DoD:** `prompt_hash()` is stable across runs. Template renders correctly for a real item.

- [ ] **1.4 [C]** `src/judge.py` — vLLM wrapper. Batched `LLM.generate`; guided decoding constraining the verdict to `{A, B}`; **`logprobs=20`**; JSONL append-checkpoint keyed by `(item_id, condition, order, sample_idx)`; skip-completed on restart.
  **Per-call schedule (D5 + D6):** for each (item, condition) emit **6 calls** — `sample_idx=0` at `temperature_canonical` in **both** orders (2), plus `sample_idx=1..k_sc` at `temperature_sc` in **AB order only** (4). Repeating the sampled draws in BA buys nothing.
  **Compute the CoT logprob aggregates here, at generation time** — `cot_logprob_{mean,min,std,p10}`, `cot_entropy_mean`, `n_cot_tokens`. They cannot be reconstructed from `calls.parquet` later, and Tier C is unbuildable without them (D4).
  **Write the 10% raw logprob sidecar** to `runs/logprobs_sample/` where `hash(item_id) % 10 == 0` (D4).
  **DoD:** running twice does not duplicate rows. Killing mid-run and restarting loses at most one batch. A row for `sample_idx=0` has all six CoT aggregate columns populated. Sidecar files exist for ~10% of items.

- [ ] **1.5 [C]** `src/parse.py` — extract verdict, verbalized confidence, verdict-token logprob, renormalised `p_a`. Failure taxonomy per `CLAUDE.md` §3.
  **DoD:** `tests/test_parse.py` passes against ≥10 real malformed outputs collected in 1.6, saved as fixtures.

- [ ] **1.6 [C]** Pilot run: **20 items × `clean` only × 6 calls = 120 generations** (D5, D6, D10) — 2 canonical greedy across orders + 4 sampled draws in AB. `verbose` and `attribution` don't exist until W4.
  **Plus a 10-item verbose smoke test** with deliberately throwaway padding, for one purpose: measuring the **prompt-token multiplier**. Mark it throwaway in the code.
  **Plus the temperature probe (D6):** run one item at `T=0` and at `T=0.7` and compare the reported `p_a`. If they differ, temperature is scaling the returned logprobs — which is why `conf_lp` comes from `sample_idx=0` only. Record the result in `REPORT.md` as a methods note.
  Collect malformed outputs into `tests/fixtures/`.
  **DoD:** parse failure < 5%. Per-generation wall-clock recorded. GPU hours extrapolated as **N × 3 conditions × 6 calls × 3 (rerun factor)**, adjusted by the verbose multiplier, and written into `PLAN.md` **marked PROVISIONAL** — re-extrapolate after `perturb.py` lands in W4 (D10). Expect verbose to cost ~+20–40% wall-clock, not 2–3×: padding lengthens the prompt, and prefill is cheap next to decode.

- [ ] **1.7 [C]** `src/signals.py` — `conf_verb` and `conf_lp` (both from `sample_idx=0` only, D6), `conf_sc` (fraction of the k_sc sampled verdicts matching the canonical greedy verdict), `conf_bpe` (entropy of mean `p_a` across the two orders, within condition). Docstrings state formulas. Also emit `judge_verdict` / `verdict_bidir` per D7.
  **DoD:** all four populated for the 20 pilot items, no NaNs. A test asserts `conf_lp` is never read from a row with `sample_idx != 0`.

- [ ] **1.8 [A]** Vacuum test: 40 pairs of identical responses + 20 empty-response pairs.
  **DoD:** a "dark current" number — the rate at which the judge picks a winner between identical responses — in `REPORT.md`.

- [ ] **1.9 [L]** Read Tian and Xiong (`LEARNING.md` A3, A4).
  **DoD:** you can state Xiong's negative AUROC result — prompting fixes ECE, AUROC stays ~0.5–0.6 — and why it means a flat risk–coverage curve in W3 is the expected outcome.

- ⛔ **GATE 1** — pilot complete, parse failures <5%, resume proven, GPU budget known. **`PREREGISTRATION.md` frozen and committed. `src/prompts.py` frozen.**

---

## Week 2 · Sep 7–13 · 9h — Clean run + RQ1

- [ ] **2.1 [C]** Full run: `clean` condition, all items, both orders (D5 — there is no `swap` condition).
  **DoD:** `results/calls.parquet` complete. Row count = **N × 1 condition × 6 calls** (2 greedy across orders + 4 sampled in AB). No missing keys. All CoT aggregate columns populated.

- [ ] **2.2 [C]** `src/signals.py` CLI: `calls.parquet` → `results/items.parquet` per `CLAUDE.md` §3.
  **DoD:** one row per (item_id, condition). `correct` populated. Schema asserted in a test.

- [ ] **2.3 [L]** Block C4 — build the ECE⊥AUROC counterexample figure (`LEARNING.md` C4).
  **DoD:** `results/figures/ece_auroc_orthogonal.png` exists and you can narrate it in one sentence.

- [ ] **2.4 [C]** `src/boot.py` — `cluster_bootstrap(df, stat_fn, group_col, n=2000)` and `paired_cluster_bootstrap(...)`.
  **DoD:** `tests/test_boot.py` shows clustered CIs are strictly wider than naive row CIs on the same data.

- [ ] **2.5 [C]** Finish `src/metrics.py`: `mce`, `brier`, `brier_decomposition`, `overconfidence_gap`, `auroc_error`.
  **DoD:** each has a unit test. `brier_decomposition` reconstructs the Brier score to 1e-6.

- [ ] **2.6 [A]** **RQ1**: for each of the four signals — reliability diagram (quantile bins), ECE, MCE, Brier + decomposition, overconfidence gap, accuracy, κ. All with cluster-bootstrap CIs.
  Report headline numbers for **both** `judge_verdict` (canonical AB, the deployed case) and `verdict_bidir` (order-averaged) per D7. **The gap between them quantifies what debiasing-by-averaging buys — that's a finding, not bookkeeping.**
  **DoD:** `results/figures/reliability_{signal}.png` ×4, plus `results/rq1_table.csv` with both verdict definitions.

- [ ] **2.6b [L]** Read A8 (Dark Current) and A6 (Reliability without Validity).
  **DoD:** you can state why κ is mandatory and what the "true vacuum" probe tests.

- [ ] **2.7 [W]** `REPORT.md` RQ1 section: the figures + one sentence stating judge overconfidence with a number and a CI.
  **DoD:** written. Not deferred.

- ⛔ **GATE 2** — RQ1 answered with CIs.

---

## Week 3 · Sep 14–20 · 9h — RQ2 + the differentiator

- [ ] **3.1 [C]** `src/metrics.py::risk_coverage(conf, correct)` and `aurc()`, plus the **oracle** curve (rank by true correctness).
  **DoD:** unit-tested; oracle dominates every real signal by construction.

- [ ] **3.2 [A]** **RQ2**: risk–coverage per signal + oracle on one axis. AURC, accuracy@{90,75,50}% coverage, **κ@coverage**, AUROC(uncertainty→error). Cluster-bootstrap CIs.
  **DoD:** `results/figures/risk_coverage.png` — **this is the thesis figure.** Plus `results/rq2_table.csv`.

- [ ] **3.3 [A]** Human-disagreement decomposition. **Primary: continuous** — regress judge correctness on `d_human`, and judge confidence on `d_human`, with cluster-bootstrap CIs (D9). **Secondary: bucketed** (unanimous / strong majority / contested) *only if Gate 0 found ≥100 contested items.*
  **DoD:** `results/figures/human_disagreement.png` + the finding as one sentence.

- [ ] **3.4 [A]** Does judge uncertainty track *human* uncertainty at all? Correlation between each confidence signal and `d_human`.
  **DoD:** four correlations with CIs in `REPORT.md`. **This is the aleatoric/epistemic result — if judge confidence is uncorrelated with human consensus, the judge is not modelling task ambiguity at all, only its own.**

- [ ] **3.4b [L]** Read A7 (Trust or Escalate) and A5 (SCOPE).
  **DoD:** you can define BPE in one sentence and say what SCOPE leaves open for you.

- [ ] **3.5 [W]** `REPORT.md` RQ2 section.
  **DoD:** written.

- ⛔ **GATE 3** — RQ2 answered. One figure that is the whole thesis. A flat curve is a finding; the oracle overlay makes it legible.

---

## Week 4 · Sep 21–27 · 9h — RQ3

- [ ] **4.1 [C]** `src/perturb.py` — `verbose_pad()` (repetitive-list attack, Zheng §3.3) and `attribution()` (label one response as written by the judge's own model; control labels a different model). **Never named `self_preference` — see D13.**
  **DoD:** `tests/test_perturb.py` — `verbose_pad` leaves semantic content intact; `attribution` changes exactly one line. (The order round-trip property lives in `test_prompts.py`, D5.)

- [ ] **4.1b [C]** **Re-extrapolate the GPU budget** now that real padding exists, against the provisional Gate 1 number (D10).
  **DoD:** updated estimate in `PLAN.md`. If it exceeds the remaining Colab units, **cut N before cutting conditions** (D12).

- [ ] **4.2 [C]** Runs for `verbose` and `attribution`, both orders, 6 calls per (item, condition).
  **DoD:** `calls.parquet` extended; `items.parquet` rebuilt.

- [ ] **4.3 [A]** **RQ3a** — position bias, computed **within the `clean` condition** (D5; it is not a separate run): flip rate between orders; **mean confidence on flipped vs unflipped items**, paired cluster-bootstrap CI on the difference.
  **DoD:** the RQ3 money sentence, with a CI, in `REPORT.md`.

- [ ] **4.4 [A]** **RQ3b** — verbosity and attribution: paired ΔECE, Δaccuracy, ΔAUROC clean→perturbed.
  **DoD:** `results/rq3_table.csv` with paired CIs. In `REPORT.md`, the attribution finding is worded **"susceptibility to authorship framing"**, never "self-enhancement bias" (D13) — and state the design's advantage: decoupling the label from real authorship isolates the attribution effect from stylistic self-similarity, which is exactly the confound that defeated Zheng et al.'s controlled study.

- [ ] **4.5 [A]** Ablation: constrained vs free-form decoding on 100 items — parse rate and verdict agreement.
  **DoD:** a limitations paragraph in `REPORT.md` saying whether constraining moved the verdicts.

- ⛔ **GATE 4** — RQ3 answered in one sentence with a CI. If behind here, cut per `PLAN.md` §4.

---

## Week 5 · Sep 28–Oct 4 · 11h — RQ4 (protect this week)

- [ ] **5.1 [L]** Read A9 (Know When You're Wrong) and A10 (Meta-Judges).
  **DoD:** you can say in two sentences why RQ4 is neither of them.

- [ ] **5.2 [C]** `src/features.py` — tiers A, B, C exactly as in `PLAN.md` §2.2. One function per tier, each returning a named DataFrame.
  Tier C draws on the CoT aggregate columns captured back in task 1.4 (D4). Include both the greedy and the sampled variant of each, named **`*_greedy` and `*_sampled_t07`** — the temperature goes in the column name so nobody averages them later thinking they're the same unit (D4). Label `cot_entropy_*` as **truncated top-20 entropy**; it is a biased estimator and must never be called "predictive entropy".
  **DoD:** `tests/test_features.py` asserts column sets, **no leakage of `correct`, `correct_bidir`, `human_label`, or `frac_prefer_a` into any tier**, no NaNs.

- [ ] **5.3 [C]** `src/predictor.py` — **`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` on `question_id`, repeated over 10 seeds** (D8). `LogisticRegression(C=1.0)` on standardized features and `HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)`. **No nested CV, no tuning** — hyperparameters are fixed and preregistered. Out-of-fold predictions returned per repeat.
  ⚠️ `GroupKFold` has **no `shuffle`** — using it would make all 10 repeats identical and the spread fake.
  **DoD:** `tests/test_predictor.py` asserts (a) no `question_id` in both train and test of any fold, **(b) two different seeds produce different fold assignments.** Both mandatory. Headline uncertainty is the **across-repeat spread**, not a within-split CI.

- [ ] **5.4 [C]** `src/predictor.py::permutation_null(n=200)` — shuffle labels within folds, return the null AUROC distribution.
  **DoD:** null AUROC centres on ~0.5; your observed value reported as a percentile of the null.

- [ ] **5.5 [A]** Tier ablation A → B → C, both models, restricted to items where **`human_agreed`** is true (D16 — `human_unanimous AND n_human_votes >= 2`; single-vote items are excluded, not counted as agreed). Baseline = best single signal's AUROC from RQ2.
  **DoD:** `results/rq4_ablation.csv` + a bar chart with CIs and the baseline drawn as a line.

- [ ] **5.6 [A]** **H4**, continuous form (D9): test the **interaction** between `d_human` and the predictor's output on judge correctness, using every item with ≥2 votes. Hypothesis: the predictor's edge grows with human consensus — error is learnable where humans agree (epistemic) and not where they don't (aleatoric).
  Bucketed agreed-vs-contested comparison **only if Gate 0 found ≥100 contested items**. If even the continuous version is thin, fall back to comparing mean predicted P(wrong) across consensus strata with a cluster-bootstrap on group means — no CV needed.
  **CI method is mandated (D15):** (1) take **out-of-fold** predictions from the fixed 10×5 repeated CV, averaged across repeats to one OOF score per item — in-sample predictions bias the interaction before the CI method even matters; (2) fit `correct ~ oof_score * d_human`; (3) **cluster-bootstrap over `question_id`, B=2000**, refitting each resample, percentile CI on the interaction coefficient. **Never a `statsmodels`/`sklearn` default standard error** — those assume i.i.d. rows and rows are clustered inside 80 questions (invariant #2).
  **DoD:** the interaction coefficient with a **cluster-bootstrap** CI, and the aleatoric/epistemic reading in one sentence.

- [ ] **5.7 [A]** Transfer test 1: train on `clean`, test on `verbose` and `attribution`.
  **DoD:** ΔAUROC reported. The sentence *"the abstention layer trained offline degrades exactly when the judge is attacked"* is either supported with a number or refuted.

- [ ] **5.8 [A]** Transfer test 2: `LeaveOneGroupOut` over the 8 MT-Bench categories.
  **DoD:** per-category held-out AUROC; states whether the predictor generalises or learns "coding is hard".

- [ ] **5.9 [A]** Meta-model calibration: reliability diagram + ECE of the predictor's own P(judge is wrong). Logistic-regression coefficients with CIs.
  **DoD:** `results/figures/rq4_coefficients.png` — **the coefficients are the result, more than the AUROC is.**

- [ ] **5.10 [W]** `REPORT.md` RQ4 section, framed as *which feature family carries the signal*, not *how good is my model*.
  **DoD:** written.

- ⛔ **GATE 5** — RQ4 answered against both the permutation null and the best-single-signal baseline. H4 tested.

---

## Week 6 · Oct 5–11 · 11h — Demo + writeup

- [ ] **6.1 [L]** Course: Gradio (`LEARNING.md` B3).
- [ ] **6.2 [C]** Demo, offline-first: load `items.parquet`, browse items, show all four signals and the verdict. **Build the "trick the judge" toggle first** — flip the response order or apply verbosity padding, watch the verdict change and the confidence not.
  **DoD:** runs on the Nitro 5 with no network. The toggle demonstrates the thesis in under 10 seconds.
- [ ] **6.3 [C]** Optional live call: 4-bit Qwen2.5-1.5B/3B via `bitsandbytes`, one item on demand.
  **DoD:** works, or is cleanly disabled by a config flag.
- [ ] **6.4 [W]** Finish `REPORT.md` from the incremental draft: abstract, method, RQ1–RQ4, limitations, extensions.
  **DoD:** complete, every figure referenced.
- [ ] **6.5 [W]** Slides. Anchor on the RQ2 thesis figure and the RQ4 coefficient plot.
  **DoD:** delivered.
- ⛔ **GATE 6** — demo runs offline; report complete.

---

## Week 7 · Oct 12–15 · 4h — Reproducibility + buffer

- [ ] **7.1 [C]** Fresh clone, `pip install -e .`, run every stage from config, confirm the headline numbers reproduce.
  **DoD:** a `REPRODUCE.md` with the exact commands and the expected numbers.
- [ ] **7.2 [W]** README with the one-paragraph result.
- [ ] **7.3** Leave the rest empty. This is re-run insurance, not spare capacity.
