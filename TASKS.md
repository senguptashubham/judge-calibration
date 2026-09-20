# TASKS.md

Atomic tasks with a definition of done. Feed one at a time to Claude Code: *"Do task 1.4 from TASKS.md."*
Read `CLAUDE.md` §2 (invariants) and `DECISIONS.md` (D4–D26) before any task touching statistics or the harness.

Legend: **[C]** code · **[A]** analysis · **[W]** writing · **[L]** learning · **⛔** gate

---

## Week 0 · Aug 24–30 · 9h — Data reality check

- [x] **0.1 [C]** Init repo: `git init`, `.gitignore` (`runs/`, `results/`, `*.parquet`, `__pycache__`, `.env`, standard Python/editor entries), `pyproject.toml` (python ≥3.11), directory skeleton from `CLAUDE.md` §4.
  **Dependencies split (D17):** base install — `datasets`, `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `matplotlib`, `pyyaml`, `pytest` — has no `vllm`. An optional `colab` extra adds `vllm`, installed only on Colab (task 1.2). **All deps pinned to exact versions** (`==`, not `>=`) once resolved by a first `pip install`; `vllm`'s pin is chosen on Colab in task 1.2 per D11, not here.
  **Local environment (D17):** a dedicated conda env, `judge-calib` (`python=3.11`) — not the owner's general AI/ML conda env, so this project's pins can't be silently violated by an unrelated install.
  **Push to a GitHub remote** once the initial commit exists — this is how Colab gets the code in W1 (D17).
  **DoD:** `pytest` runs and collects 0 tests without error. `git log` has one commit, pushed to a GitHub remote. Every base-install dep in `pyproject.toml` has an `==` pin.

- [x] **0.2 [C]** `configs/run.yaml` + a `Config` dataclass loader in `src/config.py`.
  Keys: `judge_model`, `dataset`, `tie_policy`, **`temperature_canonical: 0.0`**, **`temperature_sc: 0.7`**, **`k_sc: 4`**, `max_tokens`, `logprobs: 20`, **`conditions: [clean, verbose]`** (no `swap` — D5; no `attribution` — D18), **`prompt_variants: [P1, P2, P3]`** (D19), `seed`, `n_bins`, `paths`.
  ⚠️ There must be **no single `temperature` key**. One key would make all k draws identical and `conf_sc` a dead constant (D6).
  **DoD:** `Config.from_yaml()` round-trips. A test asserts `temperature_sc > 0` and that `"swap"` is not in `conditions`.
  **Amendment applied (31 Aug 2026, D18/D19):** `configs/run.yaml`'s `conditions` is now `[clean, verbose]` and `prompt_variants: [P1, P2, P3]` was added, with `Config` validating both (`attribution` rejected, `P1` required in `prompt_variants`) and `tests/test_config.py` covering the new checks.

- [x] **0.3 [L]** Read Guo et al. §§1–4.2 (`LEARNING.md` A1).
  **DoD:** you can write the ECE formula from memory and say why temperature scaling can't change accuracy.

- [x] **0.4 [C]** Implement `src/metrics.py::ece(confidences, correct, n_bins, strategy)` supporting `strategy ∈ {"uniform", "quantile", "auto"}`, defaulting to `"auto"`. Docstring states the formula, why quantile beats uniform here, and why unique-value binning is **exact** for a discrete signal rather than a fallback.
  **`"auto"`:** if `n_unique(conf) <= n_bins` → bin by unique value; else quantile with duplicate edges dropped. **Returns `(ece, n_effective_bins)`** (D14).
  **DoD:** `test_ece_reference` passes on the hand-computed case in `CLAUDE.md` §5 (**0.222**, tol 1e-3). Plus `test_ece_discrete`: on a 5-level signal like `conf_sc` with `n_bins=10`, returns `n_effective_bins == 5` and does **not** raise on tied quantile edges.

- [x] **0.5 [L]** Read MT-Bench §§3–4 + Tables 2, 4, 5 (`LEARNING.md` A2).
  **DoD:** the six numbers in A2 are in your notes, and you can state the S1/S2 distinction.

- [x] **0.6 [C]** `src/metrics.py::cohens_kappa(a, b)` implemented from first principles (p_o, p_e, κ).
  **DoD:** matches `sklearn.metrics.cohen_kappa_score` on 3 random arrays; `test_metrics.py::test_kappa_balanced` asserts p_o=0.85, p_e=0.5 → κ=0.70.

- [x] **0.7 [C]** `src/data.py::load_votes()` — pull `lmsys/mt_bench_human_judgments` split `human`, return a tidy DataFrame with the fields in `CLAUDE.md` §3.
  **DoD:** returns 3,355 rows. Asserts the row count and fails loudly if the dataset shape changed.

- [x] **0.8 [C]** `src/data.py::build_items()` — aggregate votes to items keyed by `(question_id, model_a, model_b, turn)`. Emit `n_human_votes`, `frac_prefer_a`, `majority_label`, `human_unanimous`, `is_tie`. Apply the tie policy from config.
  Also emit **`d_human = |frac_prefer_a − 0.5|`** (continuous consensus strength, D9) and **`human_agreed = human_unanimous AND n_human_votes >= 2`** (D16) — a single-vote item is trivially "unanimous" and must not count as agreed ground truth.
  **DoD:** `results/items_labels.parquet` written. Printed summary: N total, N non-tie, N ≥2 votes, N ≥3 votes, **N unanimous, N contested** — the last two printed separately, because D2's fallback threshold is checked **per subset**, not on the total (D9). H4 is the fragile split, not the overall count.

- [x] **0.9 [A]** Compute **human–human Cohen's κ** on items with ≥2 non-tie votes (pair up votes, or Krippendorff's α if the vote counts are ragged).
  **DoD:** a number, with N, written into `PREREGISTRATION.md`. This is the ceiling on everything downstream.

- [x] **0.10 [W]** Write `PREREGISTRATION.md`: the five RQs (RQ5 added 31 Aug 2026); decisions **D1** (tie policy), **D2** (label construction + the ≥150-item fallback), **D3** (whether RewardBench 2 augmentation is needed), plus every ⚑-marked decision D4–D25 (as of `PREREGISTRATION.md`'s own settled date, 1 Sep 2026 — D26 postdates it); the primary endpoint; which analyses are exploratory.
  **DoD:** committed. Written *before* looking at any judge output.

- [x] ⛔ **GATE 0** — passed. N total=2396, N non-tie=1904, N≥2 votes=761, **N unanimous=2273, N contested=123**, human–human κ=0.683 (N=536). D1–D3 decided and committed (`PREREGISTRATION.md`). N contested ≥100, so H4's bucketed secondary test stays viable — the `<100` contingency did not trigger.

---

## Week 1 · Aug 31–Sep 6 · 14h (peak week) — Harness

- [x] **1.1 [L]** Course: Getting Structured LLM Output, full (`LEARNING.md` B1).
  **DoD:** you can explain how Outlines constrains generation by masking logits per token — and why that means you must constrain to exactly `{A, B}` and read the renormalised `p_a`.

- [x] **1.2 [L/C]** Course: vLLM lessons 3, 6, 7, 8 + 30 min of vLLM docs on `LLM.generate`, `SamplingParams(logprobs=...)`, structured outputs (`LEARNING.md` B2).
  **Pin `vllm==<exact>` now** and install into a **fresh venv** — Colab's preinstalled `torch` will fight vLLM's pinned `torch`; expect one runtime restart. Budget 30 min (D11).
  **DoD:** a scratch script exercises **guided decoding AND `logprobs=20` together** — that combination is what breaks, not either alone. Note whether your pinned version uses `GuidedDecodingParams` or `StructuredOutputsParams`; the API was renamed.

- [x] **1.3 [C]** `src/prompts.py` — **three** MT-Bench pairwise templates (D19): **P1** (the existing template — explanation before verdict / CoT, plus a verbalized-confidence line), **P2** (correctness-first rubric), **P3** (helpfulness-first rubric). Each independently versioned + hashed.
  **DoD:** `prompt_hash()` is stable across runs, for all three variants. Each template renders correctly for a real item. All three are frozen at Gate 1 and touching any of them after requires a new version string and a full re-run (invariant 10).

- [x] **1.3b [L]** Theory I — Bayesian inference + NumPyro/NUTS basics (`LEARNING.md`, new block, D22). Deliberately scheduled here, not W5, since it's the biggest new-concept lift in the plan and needs runway.
  **DoD:** you can explain what NUTS actually samples (the posterior over model parameters, via Hamiltonian trajectories), why a hierarchical model's group-level intercepts shrink toward the population mean (partial pooling), and why that's the standard fix for "too few groups per unit" — not an exception to invariant 11, but the textbook use case for it.

- [x] **1.4 [C]** `src/judge.py` — vLLM wrapper. Batched `LLM.generate`; guided decoding constraining the verdict to `{A, B}`; **`logprobs=20`**; JSONL append-checkpoint keyed by **`(item_id, condition, prompt_variant, order, sample_idx)`** (D19 adds `prompt_variant` to the key); skip-completed on restart.
  **Per-call schedule (D5, D6, D19):** for **`(clean, P1)`** emit **6 calls** — `sample_idx=0` at `temperature_canonical` in both orders (2), plus `sample_idx=1..k_sc` at `temperature_sc` in AB order only (4). For **`(clean, P2)`** and **`(clean, P3)`** emit **2 calls each** — greedy only, both orders, no sampling. For **`(verbose, P1)`** emit **2 calls** — greedy only, both orders, no sampling. **Self-consistency sampling (`sample_idx > 0`) happens only for `(clean, P1)`** — nowhere else (D19, D21).
  **Write the full per-token logprobs for every call** to `runs/logprobs/` (D4, amended 4 Sep 2026: 100% coverage, not a 10% sample - see DECISIONS.md). `judge.py` itself computes no derived signal - it only writes `raw_output` plus provenance to the checkpoint and the full logprobs file; the CoT aggregates and logprob-based signals are computed later, by `src/parse.py` (task 1.5), from this saved data.
  **DoD:** running twice does not duplicate rows. Killing mid-run and restarting loses at most one batch. A logprobs file exists for every completed call (not ~10% - D4's amendment). A test asserts `sample_idx > 0` never appears for any `(condition, prompt_variant)` other than `(clean, P1)`.

- [x] **1.5 [C]** `src/parse.py` — extract verdict, verbalized confidence, verdict-token logprob, renormalised `p_a`, and the CoT logprob aggregates (`cot_logprob_{mean,min,std,p10}`, `cot_entropy_mean`, `n_cot_tokens`) from `raw_output` plus the full per-token logprobs saved by `judge.py` (D4, amended 4 Sep 2026 - this file, not `judge.py`, now owns every logprob-derived field, since 100% coverage means the raw data judge.py saves never disappears). Failure taxonomy per `CLAUDE.md` §3. `split_cot_and_verdict_tokens()` (locating the CoT vs. verdict token boundary within `raw_output`) is already written here.
  **DoD:** ~~`tests/test_parse.py` passes against ≥10 real malformed outputs collected in 1.6, saved as fixtures.~~ A row for `sample_idx=0` has all six CoT aggregate columns populated.
  **Deliberately skipped, 10 Sep 2026:** the real-malformed-fixture requirement. 220 real generations across the pilot, P2/P3 smoke test, and vacuum test (tasks 1.6/1.8) produced **zero** malformed outputs - not bad luck, but structural: `no_verdict`/`no_confidence` are schema-enforced (`required` + `enum`/numeric-range), and `malformed_json` is close to impossible while structured decoding is active at all, since guaranteeing valid JSON syntax is the entire point of the grammar constraint. Only `truncated` is realistically triggerable under normal conditions, and only via an artificially low `max_tokens` stress test - considered, judged not worth the extra Colab time against other priorities right now. `parse.py`'s failure-taxonomy branches stay validated against hand-constructed cases only (`tests/test_parse.py`), not real malformed text. Revisit if task 4.5's free-form-decoding ablation surfaces real examples, or if this stops feeling acceptable later.

- [x] **1.6 [C]** Pilot run: **20 items × `clean`/P1 only × 6 calls = 120 generations** (D5, D6, D10, D19) — 2 canonical greedy across orders + 4 sampled draws in AB. `verbose` doesn't exist until W4; the full P1+P2+P3 clean run doesn't exist until W2.
  **Plus a light P2/P3 smoke test:** run a handful of items through P2 and P3 (greedy, both orders) to confirm all three templates parse cleanly before Gate 1 — catching a P2/P3-specific prompt or parsing bug now is much cheaper than in W2.
  **Plus a 10-item verbose smoke test** with deliberately throwaway padding, for one purpose: measuring the **prompt-token multiplier**. Mark it throwaway in the code.
  **Plus the temperature probe (D6):** run one item at `T=0` and at `T=0.7` and compare the reported `p_a`. If they differ, temperature is scaling the returned logprobs — which is why `conf_lp` comes from `sample_idx=0` only. Record the result in `REPORT.md` as a methods note.
  ~~Collect malformed outputs into `tests/fixtures/`.~~ **Skipped — see 1.5's note; none occurred naturally.**
  **DoD:** parse failure < 5% across P1, P2, and P3 (**actual: 0%**). Per-generation wall-clock recorded. GPU hours extrapolated as **N × 12 calls × 3 (rerun factor)** (D19), adjusted by the verbose multiplier, and written into `PLAN.md` **marked PROVISIONAL** — re-extrapolate after `perturb.py` lands in W4 (D10). Expect verbose to cost ~+20–40% wall-clock, not 2–3×: padding lengthens the prompt, and prefill is cheap next to decode.

- [x] **1.7 [C]** `src/signals.py` — `conf_verb` and `conf_lp` (both from `sample_idx=0` only, D6), `conf_sc` (fraction of the k_sc sampled verdicts matching the canonical greedy verdict, `(clean, P1)` only), `conf_bpe` (entropy of mean `p_a` across the two orders, within `(condition, prompt_variant)`). Docstrings state formulas. Also emit `judge_verdict` / `verdict_bidir` per D7.
  **DoD:** all four populated for the 20 pilot items, no NaNs. A test asserts `conf_lp` is never read from a row with `sample_idx != 0`. (`conf_ens` and its decomposition are a separate task, 2.2b — they need the P2/P3 data this pilot only smoke-tests, not the full W2 run.)

- [x] **1.8 [A]** Vacuum test: 40 pairs of identical responses + 20 empty-response pairs.
  **DoD:** a "dark current" number — the rate at which the judge picks a winner between identical responses — in `REPORT.md`.

- [x] **1.9 [L]** Read Tian and Xiong (`LEARNING.md` A3, A4).
  **DoD:** you can state Xiong's negative AUROC result — prompting fixes ECE, AUROC stays ~0.5–0.6 — and why it means a flat risk–coverage curve in W3 is the expected outcome.

- [x] ⛔ **GATE 1** — passed. Pilot complete (140 real generations, clean/P1+P2/P3), parse failure **0%** across all three prompt variants (well under the <5% bar), resume proven on real Colab runs (not just unit tests), GPU budget known and provisional (**~23.6h** L4 compute, `PLAN.md`, D19 arithmetic). `PREREGISTRATION.md` frozen and committed, including D18–D25. `src/prompts.py` frozen — all three variants.

---

## Week 2 · Sep 7–13 · 13h — Clean run (P1+P2+P3) + RQ1

- [x] **2.1 [C]** Full run: `clean` condition, all items, **all three prompt variants** (D19 — there is no `swap` or `attribution` condition, D5/D18).
  **DoD:** `results/calls_{model_slug}.parquet` complete for `clean` (`results/calls_qwen2.5_7b_instruct.parquet` for this run - D26). Row count = **N × 10 calls** (P1's 6: 2 greedy across orders + 4 sampled in AB; P2's 2 + P3's 2: greedy, both orders, no sampling). No missing keys. All CoT aggregate columns populated for every `(clean, P1)` row.

- [x] **2.2 [C]** `src/signals.py` CLI: `calls_{model_slug}.parquet` → `results/items_{model_slug}.parquet` per `CLAUDE.md` §3 (D26).
  **⚠️ Grain is now `(item_id, condition, prompt_variant)`, not `(item_id, condition)` (D20).** `correct` populated. Schema asserted in a test, including a check that `prompt_variant ∈ {P1, P2, P3}` for `clean` rows and `prompt_variant == P1` for every `verbose` row (since `verbose` never collects P2/P3).

- [x] **2.2b [C]** `src/signals.py::conf_ens()` — the judge-level entropy decomposition (D20), computed from the three variants' greedy `p_a` on `clean` items only: **Total** = H[mean(p_a across P1,P2,P3)], **Aleatoric** = mean(H[p_a]) across the three, **Epistemic** = Total − Aleatoric. Store all three (`ens_entropy_total`, `ens_entropy_aleatoric`, `ens_entropy_epistemic`) plus `conf_ens = 1 - ens_entropy_total`.
  **DoD:** all four columns populated for every `clean` item (one value per item, not per prompt_variant — store on the `P1` row, null on P2/P3 rows, to avoid triplicating an item-level quantity). Null for every `verbose` row (test this explicitly — it's the D21 sanity check that will otherwise surface as a confusing bug in W5). Docstring states the formula and cites D20 for why `conf_ens` is `1 - Total` (mirrors `conf_bpe`'s convention) and never reported without naming which of the three components you mean.

- [x] **2.3 [L]** Block C4 — build the ECE⊥AUROC counterexample figure (`LEARNING.md` C4).
  **DoD:** `results/figures/ece_auroc_orthogonal.png` exists and you can narrate it in one sentence.

- [x] **2.4 [C]** `src/boot.py` — `cluster_bootstrap(df, stat_fn, group_col, n=2000)` and `paired_cluster_bootstrap(...)`.
  **DoD:** `tests/test_boot.py` shows clustered CIs are strictly wider than naive row CIs on the same data.

- [x] **2.5 [C]** Finish `src/metrics.py`: `mce`, `brier`, `brier_decomposition`, `overconfidence_gap`, `auroc_error`.
  **DoD:** each has a unit test. `brier_decomposition` reconstructs the Brier score to 1e-6.

- [x] **2.6 [A]** **RQ1**: for each of the four original signals — reliability diagram (quantile bins), ECE, MCE, Brier + decomposition, overconfidence gap, accuracy, κ. All with cluster-bootstrap CIs.
  **⚠️ Filter to `condition == "clean" AND prompt_variant == "P1"` before this analysis reads `items.parquet`** (invariant 14) — P2/P3 rows inflate sample size, and once `verbose` runs land in W4, `prompt_variant == "P1"` alone would also mix in `(verbose, P1)` rows.
  Report headline numbers for **both** `judge_verdict` (canonical AB, the deployed case) and `verdict_bidir` (order-averaged) per D7. **The gap between them quantifies what debiasing-by-averaging buys — that's a finding, not bookkeeping.**
  **DoD:** `results/figures/reliability_{signal}_{model_slug}.png` ×4, plus `results/rq1_table_{model_slug}.csv` with both verdict definitions, computed on `(clean, P1)` only (`{model_slug}` = `qwen2.5_7b_instruct` for this run, D26).

- [x] **2.6b [L]** Read A8 (Dark Current) and A6 (Reliability without Validity).
  **DoD:** you can state why κ is mandatory and what the "true vacuum" probe tests.

- [x] **2.7 [W]** `REPORT.md` RQ1 section: the figures + one sentence stating judge overconfidence with a number and a CI.
  **DoD:** written. Not deferred.

- [x] ⛔ **GATE 2** — passed. RQ1 answered with CIs on `(clean, P1)`, N=1836: `conf_verb` overconfidence gap 0.192 [0.172, 0.213] (`judge_verdict`) / 0.157 [0.136, 0.178] (`verdict_bidir`); accuracy 0.757→0.792 debiasing gain confirmed via paired cluster-bootstrap, +0.0354 [0.0177, 0.0531] (CI excludes 0). `conf_ens` populated for every `clean` item (1904/1904 P1 rows) and null for P2/P3, per D20/D21. `verbose` doesn't exist until W4, so the "null for every `verbose` item" half of the D21 check is currently vacuous — re-confirm once that run lands.

---

## Week 3 · Sep 14–20 · 11h — RQ2 + human-disagreement + threshold sweep

- [x] **3.1 [C]** `src/metrics.py::risk_coverage(conf, correct)` and `aurc()`, plus the **oracle** curve (rank by true correctness).
  **DoD:** unit-tested; oracle dominates every real signal by construction.

- [x] **3.1b [C]** Extend the risk-coverage machinery with a **threshold-sweep** mode (D20, professor feedback point 5): given a signal, sweep **raw threshold values** (not just percentile coverage) and report coverage, accuracy, κ, **and ECE on the retained set** at each threshold. This is a genuinely different x-axis parametrization from 3.1's percentile-based curve, not a rebuild of it.
  **DoD:** unit-tested against a synthetic signal where the answer is known by construction. Works on `conf_ens`'s `ens_entropy_total`, `ens_entropy_aleatoric`, `ens_entropy_epistemic` independently.

- [x] **3.2 [A]** **RQ2**: risk–coverage per signal + oracle on one axis, on `(clean, P1)` only (invariant 14 — `prompt_variant == "P1"` alone isn't enough once `verbose` exists). AURC, accuracy@{90,75,50}% coverage, **κ@coverage**, AUROC(uncertainty→error). Cluster-bootstrap CIs.
  **DoD:** `results/figures/risk_coverage_{model_slug}.png` — **this is the thesis figure.** Plus `results/rq2_table_{model_slug}.csv` (`{model_slug}` = `qwen2.5_7b_instruct` for this run, D26).

- [x] **3.2b [A]** **RQ5 threshold sweep** (D20, D23): run 3.1b's threshold sweep on `ens_entropy_total`, `ens_entropy_aleatoric`, `ens_entropy_epistemic` separately, on `clean` items (where `conf_ens` exists). **Test the preregistered prediction:** epistemic thresholding beats total thresholding, since epistemic is the reducible part.
  **DoD:** `results/figures/entropy_threshold_sweep_{model_slug}.png` (all three curves + oracle) and a one-sentence verdict on whether the preregistered prediction held.

- [x] **3.3 [A]** Human-disagreement decomposition. **Primary: continuous** — regress judge correctness on `d_human`, and judge confidence on `d_human`, with cluster-bootstrap CIs (D9). **Secondary: bucketed** (unanimous / strong majority / contested) *only if Gate 0 found ≥100 contested items.*
  **DoD:** `results/figures/human_disagreement_{model_slug}.png` + the finding as one sentence.

- [x] **3.4 [A]** Does judge uncertainty track *human* uncertainty at all? Correlation between each of the four original confidence signals and `d_human`.
  **DoD:** four correlations with CIs in `REPORT.md`. **This is the aleatoric/epistemic result — if judge confidence is uncorrelated with human consensus, the judge is not modelling task ambiguity at all, only its own.** (The equivalent check for `conf_ens`'s aleatoric component specifically is D23's dedicated validation, task 5.9e — kept in W5 alongside the rest of RQ5 rather than duplicated here, even though it reuses this same `d_human` machinery.)

- [ ] **3.4b [L]** Read A7 (Trust or Escalate) and A5 (SCOPE).
  **DoD:** you can define BPE in one sentence and say what SCOPE leaves open for you.

- [ ] **3.4c [L]** Theory K — BALD / mutual information (`LEARNING.md`, new block, D20).
  **DoD:** you can derive Epistemic = Total − Aleatoric from first principles (it's the mutual information between the prediction and the model/ensemble parameter), and explain in one sentence why a position-biased judge maximizing `conf_bpe`'s entropy is conceptually the same phenomenon as an ensemble maximizing epistemic uncertainty.

- [x] **3.5 [W]** `REPORT.md` RQ2 section, including the entropy threshold-sweep result (3.2b).
  **DoD:** written.

- [x] ⛔ **GATE 3** — passed. RQ2 answered on `(clean, P1)`, N=1836: oracle AURC 0.0295 vs. every real signal 1.3–4.7x higher; `conf_bpe` best on AUROC (0.794) and every accuracy@coverage/κ@coverage cut, `conf_sc` best on AURC (0.038, driven by its own 5-value discreteness, not cleaner discrimination) — the two rankings disagree and both are reported, not reconciled into one "winner." Entropy threshold-sweep prediction (D23, epistemic beats total) tested and **rejected**: epistemic AURC 0.273 vs. total 0.098, paired gap +0.175 [0.141, 0.207]; mechanism confirmed (65% of items have ~zero epistemic entropy, but accuracy among the lowest-decile-epistemic items is only 46% vs. 75.7% overall — ensemble agreement isn't ensemble correctness). `results/figures/risk_coverage_qwen2.5_7b_instruct.png` is the thesis figure (filename updated 18 Sep 2026 - D26, model-namespaced paths). 3.4b/3.4c (reading) intentionally left open — not gating, owner's own pace.

---

## Week 4 · Sep 21–27 · 8h — RQ3

- [x] **4.1 [C]** `src/perturb.py` — `verbose_pad()` (repetitive-list attack, Zheng §3.3) only. (`attribution()` is cut, D18 — do not build it.)
  **DoD:** `tests/test_perturb.py` — `verbose_pad` leaves semantic content intact. (The order round-trip property lives in `test_prompts.py`, D5.)
  **Built early, 14 Sep 2026:** unblocked (needs no run data) and no longer W4-only — done alongside the Week 2 independent-task pass.

- [x] **4.1b [C]** **Re-extrapolate the GPU budget** now that real padding exists, against the provisional Gate 1 number (D10, D19's 12,000-generation baseline).
  **DoD:** updated estimate in `PLAN.md`. If it exceeds the remaining Colab units, **cut N before cutting conditions** (D12).
  **Done, 18 Sep 2026:** along the way, found and fixed a real bug - `verbose_pad()` was never actually wired into `judge.py`'s generation path (`src/judge.py` commit `3b46ca5`), so the first smoke test's "verbose" data was silently unpadded. After the fix, real timed smoke test (20 items, 40 generations): verbose measured 0.946 sec/gen (below clean's 1.2), but the conservative choice written into `PLAN.md` uses clean's own 1.2 sec/gen rather than banking on an unconfirmed speedup from one small, noisy sample. Total ≈ 22.85h L4 compute ≈ 35.2 compute units against 72.79 available (1.54 units/hour, confirmed live) - **N does not need to be cut**, ≈37.6 units of headroom.

- [x] **4.2 [C]** Run for `verbose`, P1 only, both orders, greedy — **2 calls per item** (D19; no sampling, no P2/P3 for `verbose`).
  **DoD:** `calls_{model_slug}.parquet` extended; `items_{model_slug}.parquet` rebuilt at its `(item_id, condition, prompt_variant)` grain (D26). `conf_sc` and `conf_ens` (+ components) are null for every `verbose` row — expected, not a bug (D21).
  **Done, 18 Sep 2026:** full population, N=1904 items, 3808 verbose rows (N×2, exact item-set match with `clean/P1`) — `calls_qwen2.5_7b_instruct.parquet` now 22,848 rows total, `items_qwen2.5_7b_instruct.parquet` 7,616 rows at the `(item_id, condition, prompt_variant)` grain. `parse_ok` 100% on verbose; padding confirmed applied (mean `n_prompt_tokens` 3060.6 vs clean's 976.5, ≈3.13x). `conf_sc`/`conf_ens`/its 3 entropy components 100% null on verbose as expected (D21); `conf_verb`/`conf_lp`/`conf_bpe`/`flipped`/`judge_verdict` fully populated. `correct`/`human_label` null on 3.57% of rows (68/1904) — a pre-existing `frac_prefer_a == 0.5` tie-policy edge case present at an identical rate across every condition/prompt_variant, including the already-closed `clean` run, not something this run introduced. `pytest` green (147 passed).

- [ ] **4.3 [A]** **RQ3a** — position bias, computed **within `(clean, P1)`** (D5, D19; it is not a separate run, and stays P1-filtered per invariant 14): flip rate between orders; **mean confidence on flipped vs unflipped items**, paired cluster-bootstrap CI on the difference.
  **DoD:** the RQ3 money sentence, with a CI, in `REPORT.md`.

- [x] **4.4 [A]** **RQ3b** — verbosity only (`attribution` is cut, D18): paired ΔECE, Δaccuracy, ΔAUROC clean(P1)→verbose(P1). **`conf_sc`'s clean→verbose comparison is dropped — no data on the `verbose` side (D21).** `conf_verb`, `conf_lp`, `conf_bpe` are unaffected.
  **DoD:** `results/rq3_table_{model_slug}.csv` (D26) with paired CIs for the three signals that survive; a one-line note in `REPORT.md`'s limitations explaining why `conf_sc` isn't in this table.
  **Done, 18 Sep 2026:** same N=1836 paired population as RQ1/RQ3a. Accuracy barely moves and isn't significant (+0.0109 [−0.0070, 0.0281]) — verbosity doesn't make the judge more wrong. But **AUROC drops significantly for all three surviving signals** (`conf_verb` −0.0276 [−0.0571, −0.0023], `conf_lp` −0.0447 [−0.0740, −0.0170], `conf_bpe` −0.0278 [−0.0494, −0.0064]), and `conf_bpe` — RQ1/RQ2's best clean-data performer — is the only one whose calibration itself significantly worsens (ECE +0.0432 [0.0144, 0.0516]). Headline: verbosity doesn't fool the judge, it quietly breaks the uncertainty signal meant to catch when it's wrong. `analysis/rq3.py` now covers both RQ3a and RQ3b (one file, per its own module docstring's stated plan). Added `results/figures/rq3b_deltas_{model_slug}.png` (two forest panels, Δ ECE and Δ AUROC) via `src/plots.py::plot_rq3b_deltas`, sharing `_draw_forest()` with the existing forest-plot figures. `pytest` green (147 passed).

- [x] **4.5 [A]** Ablation: constrained vs free-form decoding on 100 items — parse rate and verdict agreement.
  **DoD:** a limitations paragraph in `REPORT.md` saying whether constraining moved the verdicts.
  **Done, 18 Sep 2026:** 100 items, clean/P1/AB/greedy, both arms scored with the unmodified production parser. Parse rate 100%/100% (constrained/free-form) — no parse-rate cost either way on this population. **Verdict agreement 96% (4/100 disagree)** — all four flips carry high stated confidence (0.90–0.95) on both sides, not marginal coin-tosses, and since both arms are greedy at the same seed, any disagreement is a genuine causal effect of the schema mask, not sampling noise. Confirms D25's concern: constrained decoding moves the actual verdict in a real (if small, ~4%) fraction of items, not just `conf_lp`'s reported scale. `src/ablation_decoding.py` (Colab generation, reuses `judge.py`'s `CallSpec`/`_build_prompts`/`VERDICT_SCHEMA` via new `load_full_items_df()` refactor) + `analysis/decoding_ablation.py` (local scoring). `pytest` green (149 passed).

- ⛔ **GATE 4** — RQ3 answered in one sentence with a CI, verbosity only. If behind here, cut per `PLAN.md` §4.
  **Passed, 18 Sep 2026.** RQ3a: canonical verdict flips between AB/BA order on 27.8% [24.8%, 30.8%] of items; `conf_verb` tracks it weakly but measurably (gap −0.0199 [−0.0248, −0.0152]). RQ3b: verbosity padding doesn't move accuracy (Δ +0.011, ns), but significantly degrades AUROC for all three surviving signals and specifically breaks `conf_bpe`'s calibration (ECE +0.043 [0.014, 0.052]) despite it being RQ1/RQ2's best clean-data signal. Task 4.5's ablation additionally confirms D25's constrained-decoding concern is real, not just cosmetic (4% verdict flip rate) — noted as a limitation, not reopened as a re-run inside this project's timeline.

---

## Week 5 · Sep 28–Oct 4 · 15h (second peak week) — RQ4 + RQ5 (protect this week)

- [ ] **5.1 [L]** Read A9 (Know When You're Wrong) and A10 (Meta-Judges).
  **DoD:** you can say in two sentences why RQ4 is neither of them.

- [ ] **5.1b [L]** Theory J (hierarchical/partial-pooling models) and Theory L (distributional distillation) (`LEARNING.md`, new blocks, D22/D23).
  **DoD:** you can state why a held-out group's random intercept must be marginalized over the population prior rather than fitted (the D22 leakage trap), and what "the ensemble's predictive distribution is the teacher" means concretely (matching more than the mean — matching the spread too).

- [x] **5.2 [C]** `src/features.py` — tiers A, B, C exactly as in `PLAN.md` §2.2 (Tier A now includes `conf_ens` + its three entropy components, added 31 Aug 2026, D20). One function per tier, each returning a named DataFrame.
  **⚠️ Core tiers are built on `condition == "clean" AND prompt_variant == "P1"` — not just P1 alone.** `conf_ens` (+ components) and `conf_sc` are null on `verbose` (D21); including `verbose` rows here would break the "no NaNs" DoD below. The transfer test (task 5.7) builds its own separate, restricted feature view to include `verbose` — that's a different function, not this one.
  Tier C draws on the CoT aggregate columns captured back in task 1.4 (D4). Include both the greedy and the sampled variant of each, named **`*_greedy` and `*_sampled_t07`** — the temperature goes in the column name so nobody averages them later thinking they're the same unit (D4). Label `cot_entropy_*` as **truncated top-20 entropy**; it is a biased estimator and must never be called "predictive entropy".
  **DoD:** `tests/test_features.py` asserts column sets, **no leakage of `correct`, `correct_bidir`, `human_label`, or `frac_prefer_a` into any tier**, no NaNs, and every row is `condition == "clean" AND prompt_variant == "P1"` (RQ4's core analysis, per D20).
  **Done, 18 Sep 2026:** building Tier B/C first required extending `signals.py` with `len_a/len_b/len_ratio/abs_len_diff/longer_is_chosen/turn/judge_output_len/verdict_margin/cot_aggregates_{greedy,sampled_t07}` (a real prerequisite discovered mid-task, not scope creep — those columns didn't exist in `items.parquet` at all before this). Along the way: `src/parse.py::reasoning_length()` (exact `judge_output_len` source, avoids a ~52-char JSON-boilerplate offset `len(raw_output)` would have added) and `src/data.py`'s `len_a`/`len_b` (in `items_labels.parquet`, model-independent). Found and fixed two real bugs during review of the first draft: `verdict_margin` didn't guard `p_a is None` (documented-possible, would `TypeError`) and `cot_aggregates_sampled` wasn't aggregating at all (checked the wrong call, assigned a list of raw call dicts instead of a mean). **Population decision:** 17/1836 items (0.93%) have an undefined `len_ratio`/`longer_is_chosen` (zero-length response, or an exact length tie) — dropped from **all three tiers uniformly** (final N=1819), not just B/C, so "does tier B beat tier A" stays a same-population comparison rather than a confound. Verified against real data: zero NaNs across all 29 tier columns at N=1819. `pytest` green (166 passed).

- [x] **5.3 [C]** `src/predictor.py` — **`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` on `question_id`, repeated over 10 seeds** (D8). `LogisticRegression(C=1.0)` on standardized features and `HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)`. **No nested CV, no tuning** — hyperparameters are fixed and preregistered. Out-of-fold predictions returned per repeat.
  ⚠️ `GroupKFold` has **no `shuffle`** — using it would make all 10 repeats identical and the spread fake.
  **DoD:** `tests/test_predictor.py` asserts (a) no `question_id` in both train and test of any fold, **(b) two different seeds produce different fold assignments.** Both mandatory. Headline uncertainty is the **across-repeat spread**, not a within-split CI.
  **Done, 18 Sep 2026:** `make_fold_splits()` kept separate from any model-fitting (mirrors `judge.py`'s `pending_calls()`/`_run_generation()` split) specifically so D8's two properties are directly testable without fitting a classifier. `encode_features()` one-hot-encodes `category` and casts `longer_is_chosen`/`flipped` to int — applied once, before CV (target-independent, no leakage risk), unlike `StandardScaler`, which lives inside `make_logreg()`'s `Pipeline` so it's fit per-fold, never on held-out data. `LogisticRegression` gets `max_iter=1000` (default 100) purely for convergence robustness — not a hyperparameter deviation from the preregistered `C=1.0`. `HistGradientBoostingClassifier` gets `random_state=seed` (its own internal tie-breaking, not a search). Smoke-tested against real data, all combinations: Tier A/logreg AUROC 0.7962 [0.7932, 0.7986], Tier B/logreg 0.7997 [0.7888, 0.8055], Tier C/histgbm 0.7921 [0.7884, 0.7976] (10-repeat spreads, N=1819) — no crashes, no leftover NaNs anywhere. `pytest` green (178 passed).

- [x] **5.4 [C]** `src/predictor.py::permutation_null(n=200)` — shuffle labels within folds, return the null AUROC distribution.
  **DoD:** null AUROC centres on ~0.5; your observed value reported as a percentile of the null.
  **Done, 19 Sep 2026:** global shuffle of `y` per permutation (not a per-fold-independent shuffle) — the standard permutation-test construction (same shape as sklearn's own `permutation_test_score`), since shuffling values never touches which `question_id` a row belongs to, so the grouping constraint stays exactly as valid on permuted data as real. One 5-fold CV per permutation (not the full 10-repeat protocol — 200×5=1000 fits already, matching standard practice), reusing `repeated_stratified_group_kfold(..., n_repeats=1)` rather than a parallel CV loop. `percentile_of_null()` reports the empirical rank, not a parametric p-value (AUROC's null isn't guaranteed normal/symmetric). Added an optional `--permutation-null` CLI flag (off by default — ~20x the compute of the real result alone). Verified against real data: Tier A/logreg null mean=0.4928, std=0.0222 (centers on chance as required), real AUROC (0.7962) at the 100th percentile — comfortably beats every one of the 200 shuffled runs, and confirms no pipeline leakage. `pytest` green (182 passed, 4 new).

- [x] **5.5 [A]** Tier ablation A → B → C, both frequentist models, restricted to items where **`human_agreed`** is true (D16 — `human_unanimous AND n_human_votes >= 2`; single-vote items are excluded, not counted as agreed). Baseline = best single signal's AUROC from RQ2.
  **DoD:** `results/rq4_ablation_{model_slug}.csv` (D26) + a bar chart with CIs and the baseline drawn as a line.
  **Done, 19 Sep 2026:** population N=556 (79/80 question groups) — a much bigger, non-population-neutral cut from RQ4's N=1819 base than the DoD's wording suggests (accuracy 75.8%→78.4%), so the baseline was recomputed on this exact population rather than reused from `rq2_table_*.csv` (documented in `analysis/rq4.py`'s module docstring). Raw result: baseline (`conf_bpe`) AUROC 0.8196 [0.7746, 0.8622]; Tier A closest (logreg 0.8239, histgbm 0.8188), B and C trending slightly lower. Went beyond the bare DoD to actually test whether that apparent A>B>C pattern is real: **permutation null** (invariant 12, on this population specifically, not the earlier full-population check) confirms every cell clears chance at the 100th percentile; **paired `cluster_bootstrap`** on each step (baseline→A, A→B, B→C, both models) found **every single CI crosses zero** — the tier progression is statistically indistinguishable from no change. Honest finding: the judge's own uncertainty signals (Tier A) already capture what surface and token-distribution features have to offer; adding them neither helps nor measurably hurts. Three figures: `rq4_ablation_{model_slug}.png` (bar chart, baseline as its own bar), `rq4_progression_{model_slug}.png` (forest plot of the 6 paired deltas), `rq4_permutation_nulls_{model_slug}.png` (6-panel null histograms, tier×model grid). Two real bugs found and fixed during review (baseline dict key mismatch, a wrong CI-based tie-break in the winner-selection logic). `pytest` green (182 passed) throughout.

- [x] **5.6 [A]** **H4**, continuous form (D9): test the **interaction** between `d_human` and the (frequentist) predictor's output on judge correctness, using every item with ≥2 votes. Hypothesis: the predictor's edge grows with human consensus — error is learnable where humans agree (epistemic) and not where they don't (aleatoric).
  Bucketed agreed-vs-contested comparison **only if Gate 0 found ≥100 contested items**. If even the continuous version is thin, fall back to comparing mean predicted P(wrong) across consensus strata with a cluster-bootstrap on group means — no CV needed.
  **CI method is mandated (D15):** (1) take **out-of-fold** predictions from the fixed 10×5 repeated CV, averaged across repeats to one OOF score per item — in-sample predictions bias the interaction before the CI method even matters; (2) fit `correct ~ oof_score * d_human`; (3) **cluster-bootstrap over `question_id`, B=2000**, refitting each resample, percentile CI on the interaction coefficient. **Never a `statsmodels`/`sklearn` default standard error** — those assume i.i.d. rows and rows are clustered inside 80 questions (invariant #2).
  **DoD:** the interaction coefficient with a **cluster-bootstrap** CI, and the aleatoric/epistemic reading in one sentence.
  **Done, 19 Sep 2026:** population N=595 (79/80 groups, D9's own ≥2-vote population — larger than D9's rough ~350-item pre-data estimate), deliberately NOT 5.5's `human_agreed`-restricted population or `load_rq4_population()`'s `len_ratio`-filtered one — H4 needs every contested item, that's the point. "The predictor" = Tier A + logreg specifically (documented choice, not left implicit: 5.5 found no tier/model beats another, and Tier A avoids a second population cut Tier B/C would force; logreg keeps the OOF predictor and the interaction model in one coherent logistic-regression family). **Result: interaction coefficient +2.3752 [1.6849, 3.0739] — CI clears 0 comfortably, supports H4.** The predictor's edge genuinely grows with human consensus: judge error is learnable where humans agree (epistemic, reducible) and much less so where they don't (aleatoric, irreducible) — a clean positive result, in contrast to 5.5's null-ish tier-ablation finding. `analysis/rq4.py` now covers both 5.5 and 5.6 behind a `--task {ablation,h4}` flag (mirrors `rq3.py`'s own two-subtask-one-file pattern). Added `results/figures/h4_interaction_{model_slug}.png` (not required by the DoD, added on request) — two panels, probability and log-odds; the probability panel alone visually undersold the interaction (sigmoid saturation where most real data sits), so a log-odds panel was added where the fitted slope difference is undistorted and clearly visible. `pytest` green (182 passed, untouched).

- [x] **5.7 [A]** Transfer test 1: train on `clean`, test on `verbose` (`attribution` is cut, D18).
  **⚠️ Feature-parity fix (D21):** use **Tier A minus `{conf_sc, conf_ens, ens_entropy_total, ens_entropy_aleatoric, ens_entropy_epistemic}`** — `verbose` has none of these. Apply this to every model compared (LogReg, HistGBM, and the Bayesian model, task 5.9f), not just one.
  **DoD:** ΔAUROC reported. The sentence *"the abstention layer trained offline degrades exactly when the judge is attacked"* is either supported with a number or refuted. `REPORT.md` states the feature-parity exception explicitly — never silently impute or leave NaN.
  **Done, 20 Sep 2026:** population = `analysis/rq3.py`'s own `load_rq3b_items()` (N=1836, reused directly rather than re-derived). Feature set (D21) reduces to just `{conf_verb, conf_lp, conf_bpe}`. Two numbers per model, both on the SAME reduced feature set so the comparison isolates the transfer effect from the feature-drop effect: in-domain baseline (repeated grouped CV, D8, on `clean` alone) vs. transfer (fit once on all of `clean`, frozen, evaluated on `verbose`, cluster-bootstrap CI over `verbose`'s own `question_id`). **Result: logreg ΔAUROC = +0.0001 (0.7703→0.7705, CIs heavily overlapping) — essentially no degradation. histgbm ΔAUROC = −0.0254 (0.7889→0.7635, CIs still overlapping) — a small, not clearly significant drop.** This **refutes**, not supports, the headline hypothesis: unlike RQ3b's finding that each *individual* signal's own AUROC drops significantly under verbose, a model that *combines* `conf_verb`/`conf_lp`/`conf_bpe` transfers with little to no measurable degradation — combining signals appears to buy real robustness that no single signal has on its own, since AUROC only needs the combined rank-ordering to survive, not each signal's own calibration. Added `results/figures/rq4_transfer_{model_slug}.png` (not required by the DoD, added on request) — grouped bar chart, in-domain vs. transfer per model. `pytest` green (182 passed, untouched).

- [x] **5.8 [A]** Transfer test 2: `LeaveOneGroupOut` over the 8 MT-Bench categories.
  **DoD:** per-category held-out AUROC; states whether the predictor generalises or learns "coding is hard".
  **Done, 20 Sep 2026:** owner-written (`compute_category_held_out_auroc`/`main_category` in `analysis/rq4.py`), reviewed — fixed 6 bugs (DataFrame integer-indexing where `.iloc` was needed, missing `[:, 1]` on `predict_proba`, a bare-list `DataFrame` losing its column names, `main_category` reading nonexistent `auroc_mean`/`_low`/`_high` columns that don't apply to this deterministic, non-repeated protocol, `pd.DataFrame.from_records` on a list of DataFrames instead of `pd.concat`, and the missing spread-across-categories summary the DoD's reading depends on). Full Tier A used unmodified (no D21 exclusion needed — this never leaves `clean`). **Result: AUROC ranges 0.739–0.864 (logreg) and 0.753–0.852 (histgbm) across the 8 categories — every category comfortably above chance, no collapse.** Notably, `coding` is NOT the weak point (logreg 0.821, solidly mid-range) — the two weakest categories are `writing` (0.739) and `reasoning` (0.744). Reading: the predictor generalises reasonably across task types rather than learning a "coding is hard" shortcut, though with real (not dramatic) category-to-category variation. `results/rq4_category_transfer_{model_slug}.csv`. `pytest` green (182 passed, untouched).

- [ ] **5.9 [A]** Meta-model calibration (frequentist): reliability diagram + ECE of the predictor's own P(judge is wrong). Logistic-regression coefficients with CIs.
  **DoD:** `results/figures/rq4_coefficients_{model_slug}.png` (D26) — **the coefficients are the result, more than the AUROC is.**

- [ ] **5.9b [C]** `src/bayesian.py` — the hierarchical logistic regression (D22): `correct ~ Bernoulli(σ(α + α_q[question] + Xβ))`, `α_q ~ Normal(0, σ_q)`, `σ_q ~ HalfNormal(1)`, `β ~ Normal(0, 1)`. Fit with NumPyro/NUTS under the **same** `StratifiedGroupKFold(5)`×10-seed protocol as 5.3. **Fallback ladder, preregistered:** NUTS → Laplace approximation (MAP via sklearn + Hessian) → bootstrap ensemble of logistic regressions — state which rung was used.
  **⚠️ Held-out random-intercept marginalization is mandatory:** a held-out question's `α_q` is drawn from the population prior (`α_q_new ~ Normal(0, σ_q)`), never its would-be fitted value — the hierarchical-model analogue of the `GroupKFold` leak D8 prevents for the frequentist model.
  **Convergence diagnostics (R-hat, ESS, divergence count) are computed and stored for every fold-fit** — never optional, same status as 5.4's permutation null.
  **DoD:** `tests/test_bayesian.py` asserts a held-out fold's predictions do not depend on that fold's fitted `α_q` (mirrors `test_predictor.py`'s no-leakage assertion). Convergence diagnostics reported for every fit; any fit with R-hat > 1.01 or divergences is flagged, not silently included. Runtime for the full 50-fit protocol measured and recorded — if too slow, the repeat count for this arm specifically is reduced and the reduction is preregistered in `PREREGISTRATION.md`, not discovered mid-week.

- [ ] **5.9c [A]** Head-to-head table: frequentist `LogisticRegression` vs the Bayesian model (5.9b) — AUROC, ECE, Brier, NLL, 90% credible-interval coverage. NLL/coverage exist only for the Bayesian arm.
  **DoD:** `results/rq4_bayesian_comparison_{model_slug}.csv` (D26) + a reliability diagram for the Bayesian model's own P(judge is wrong), alongside the frequentist one from 5.9.

- [ ] **5.9d [A]** **RQ5 distillation comparison** (D23): compare the 3-prompt ensemble's predictive distribution (mean + spread of `p_a` across P1/P2/P3) against the single-call Bayesian model's (5.9b) own posterior predictive spread, trained on P1-only features. **Headline:** how much of the ensemble's AUROC/ECE/entropy-quality benefit survives at 1 call vs 3.
  **DoD:** a table or figure showing the ensemble-vs-single-call gap, and the headline sentence stated with a number.

- [ ] **5.9e [A]** **RQ5 human-disagreement validation** (D23): is `conf_ens`'s **aleatoric** component (2.2b) high specifically where humans actually disagreed (high `d_human`), and is **epistemic** not? Reuses the `d_human` machinery from 3.3/3.4 — a validation, not new statistical infrastructure. Runs on `clean`/P1 rows only (invariant 14).
  **DoD:** the correlation (aleatoric vs `d_human`, and separately epistemic vs `d_human`) with CIs, and the one-sentence reading: does the estimated aleatoric signal track genuine human disagreement?

- [ ] **5.9f [A]** **Verbose-shift validation** (D21, D23, professor feedback "consequences"): train the Bayesian model (5.9b) on `clean` (Tier A minus `conf_sc`/`conf_ens`, per 5.7's fix), evaluate its **meta-model-level** entropy decomposition on `verbose`. **Preregistered prediction:** epistemic uncertainty rises under this distribution shift while aleatoric stays flat — the canonical check that a Bayesian model's epistemic estimate is doing its job. This is **not** a `conf_ens` check — `conf_ens` is undefined for `verbose` (D21).
  **DoD:** the epistemic and aleatoric values on `clean` vs `verbose`, with the verdict on whether the preregistered prediction held.

- [ ] **5.10 [W]** `REPORT.md` RQ4 section, framed as *which feature family carries the signal*, not *how good is my model* — now including the Bayesian head-to-head table (5.9c).
  **DoD:** written.

- [ ] **5.11 [W]** `REPORT.md` RQ5 section: the distillation gap (5.9d), the human-disagreement validation (5.9e), and the verbose-shift check (5.9f).
  **DoD:** written.

- ⛔ **GATE 5** — RQ4 answered against both the permutation null and the best-single-signal baseline, **for both the frequentist and Bayesian models**, with convergence diagnostics reported. H4 tested. **RQ5 answered: distillation gap stated, human-disagreement validation done, verbose-shift prediction tested.**

---

## Week 6 · Oct 5–11 · 12h — Demo + writeup

- [ ] **6.1 [L]** Course: Gradio (`LEARNING.md` B3).
- [ ] **6.2 [C]** Demo, offline-first: load `items_{model_slug}.parquet` (P1 rows, D26), browse items, show all five signals (including `conf_ens`) and the verdict. **Build the "trick the judge" toggle first** — flip the response order or apply verbosity padding, watch the verdict change and the confidence not.
  **DoD:** runs on the Nitro 5 with no network. The toggle demonstrates the thesis in under 10 seconds.
- [ ] **6.3 [C]** Optional live call: 4-bit Qwen2.5-1.5B/3B via `bitsandbytes`, one item on demand.
  **DoD:** works, or is cleanly disabled by a config flag.
- [ ] **6.4 [W]** Finish `REPORT.md` from the incremental draft: abstract, method, RQ1–RQ5, limitations, extensions.
  **DoD:** complete, every figure referenced.
- [ ] **6.5 [W]** Slides. Anchor on the RQ2 thesis figure, the RQ4 coefficient plot, **and the RQ5 distillation result**.
  **DoD:** delivered.
- ⛔ **GATE 6** — demo runs offline; report complete, RQ1–RQ5 all covered.

---

## Week 7 · Oct 12–15 · 4h — Reproducibility + buffer

- [ ] **7.1 [C]** Fresh clone, `pip install -e .`, run every stage from config, confirm the headline numbers reproduce — including the Bayesian model's head-to-head table (rerunning NUTS should reproduce the same conclusions, not necessarily bit-identical posterior samples).
  **DoD:** a `REPRODUCE.md` with the exact commands and the expected numbers.
- [ ] **7.2 [W]** README with the one-paragraph result.
- [ ] **7.3** Leave the rest empty. This is re-run insurance, not spare capacity.
