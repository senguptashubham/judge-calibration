# PLAN.md — design rationale and schedule (v3)

Supersedes build plan v1. Change from v1: **RQ4 (learned error predictor) is restored as a first-class research question**, and the hour budget is raised to match. v3 change: **RQ5 (prompt-ensemble distillation) is added** from 31 Aug 2026 professor feedback, and RQ4 gains a Bayesian hierarchical model.

**v2.1 incorporated the Week-0 design review.** `DECISIONS.md` D4–D17 documented that pass. The three that changed the harness: Tier C needs CoT logprob aggregates captured at generation time (D4); `swap` is not a condition, `order` is an axis (D5); temperature needs two config keys or `conf_sc` is a dead constant (D6). Two more from a second pass changed statistics rather than the harness: `ece()` must handle discrete-valued signals like `conf_sc` exactly rather than binning them (D14), and H4's interaction CI must be a cluster bootstrap over out-of-fold predictions, never a default standard error (D15). A third pass added two more: "human-agreed," used as the Tier-ablation baseline population, needs the same ≥2-vote floor H4 already uses for `d_human` — a single-vote item is trivially unanimous (D16); and D17 settled the local/Colab split — local dev never installs vLLM, Colab gets its own fresh venv, and a GitHub remote bridges code between the two.

**v3 incorporates the professor's 31 Aug 2026 feedback (`PROFESSORFEEDBACK.md`), via `DECISIONS.md` D18–D24.** `attribution` is dropped as a condition (D18, supersedes D13). A third prompt-ensemble axis (P1/P2/P3) and a new 12-calls/item schedule supersede D5's/D12's numbers (D19). A new signal `conf_ens` and its judge-level entropy decomposition change `items.parquet`'s grain and require a `prompt_variant == "P1"` filter everywhere in RQ1–RQ4 (D20). `conf_sc` and `conf_ens` become clean-only, which breaks the existing transfer test unless its feature set is adjusted (D21). RQ4 gains a Bayesian hierarchical logistic regression alongside the existing two models (D22). A new RQ5 formalises prompt distillation and a human-disagreement validation of the aleatoric estimate (D23). New local dependencies: `numpyro`/`jax`/`arviz` (D24). `DECISIONS.md` D4–D26 override anything here that contradicts them — always resolve conflicts there, not here.

---

## 1. Budget

| | v1 | v2 | v3 |
|---|---|---|---|
| Elapsed | 7.4 weeks (Aug 24 → Oct 15) | same | same — W0's ~9h is already spent, ~6.3 weeks remain |
| Core study (RQ1–RQ3) | 55h | 55h | 55h |
| RQ4 (now incl. the Bayesian arm, D22) | cut | **+11h** | +11h |
| RQ5 (new, D23) | — | — | **new** |
| Courses | 0h | **+5h** | +5h |
| Reading rebalance | — | +5h | +5h |
| **Total** | 55h | **~75h** | **~85h** |
| **Per week** | 7.5h | ~10h | **~13.5h, over the ~6.3 weeks remaining** |

⚠️ **Honesty check on the professor's own numbers:** the feedback doc states both "roughly +34 hours" and "~85h total" — against the v2 baseline of 75h these don't fully reconcile (+34h on top of 75h would be ~109h, not 85h). `~85h` / `~13.5h-per-week` is the operative planning number here, since it's the pair that's internally self-consistent (85h ÷ ~6.3 remaining weeks ≈ 13.5h/week) and it's the number accepted going in — flagging the discrepancy rather than silently picking one without saying so.

Weekly figures below are ±1h. Treat them as load indicators, not contracts.

RQ4 costs **zero extra GPU time** — it runs on data already collected for RQ1–RQ3. This still holds for the Bayesian arm: fitting a hierarchical logistic regression on ~1000 rows/~80 groups is CPU-only analysis work (D24), not inference. RQ5 similarly costs no *additional* GPU time beyond the new run schedule (D19) that was going to happen anyway for the ensemble.

**Weekly load is not flat.** W1 is the peak at **~12h** (two courses plus the entire harness, now including 3 prompt templates instead of 1). If that week overruns, the relief valves, in order: (a) skip the vLLM course lessons entirely and take 30 min of vLLM docs instead, (b) slide Xiong et al. to W2. Do **not** relieve W1 by skipping the checkpoint/resume work — that is what protects every later week from a Colab disconnect. **W5 is now the second peak** (Bayesian model + RQ5 distillation + human-disagreement validation, all landing in the same week) — see the revised week plan in §3.

The report is written **incrementally**. Every week's gate includes appending figures and a paragraph to `REPORT.md`. Do not defer writing to Week 6; that is how fixed-deadline projects die.

---

## 2. RQ4 — the learned error predictor

### 2.1 Positioning (checked against the literature, Aug 2026)

Your instinct that this is "common" is half right. Meta-classifiers for failure prediction are standard in the OOD / misclassification-detection literature. But in the **LLM-judge** setting specifically, the field's reflex is to reach for either another LLM or a raw probability, not a trained model:

- [Leveraging LLMs as Meta-Judges](https://arxiv.org/html/2504.17087v1) (Apr 2025) — meta-judging with a seven-criterion rubric and multi-agent scoring. **Trains nothing.**
- [Know When You're Wrong](https://arxiv.org/html/2603.06604) (Mar 2026) — error detection from normalised output token probabilities, AUROC 0.81–0.88. **No separate classifier**, and it's general QA, not judges. *(Note the title collision with your deck — cite it, don't be surprised by it.)*
- [Trust or Escalate](https://arxiv.org/abs/2407.18370), [SCOPE](https://www.alphaxiv.org/abs/2602.13110) — both use a *single* engineered confidence signal plus a threshold. Neither combines signals with a learned model.

So: **not novel as a technique, genuinely under-occupied in this setting.** That's a legitimate slot, and it's an honest thing to say out loud. What makes it more than a bolt-on is the framing below.

### 2.2 The reframe that makes it worth 11 hours

Don't ask "can I build an error predictor." Ask:

> **Which family of cheap features carries the signal about judge error — the judge's own uncertainty, or surface properties of the inputs that the judge is ignoring?**

That question makes *every* outcome reportable, which is how you de-risk an ML component under a fixed deadline. Three feature tiers, each strictly additive:

| Tier | Features | Tests |
|---|---|---|
| **A** | `conf_verb`, `conf_lp`, `conf_sc`, `conf_bpe`, `conf_ens` + its three entropy components (D20, added 31 Aug 2026 — `conf_ens` is `clean`-only, null on `verbose`, which is exactly why the transfer test in §2.4 subtracts it back out) | Do the five uncertainty signals *combine*, or are they redundant? |
| **B** | A + `len_a`, `len_b`, `len_ratio`, `abs_len_diff`, `longer_is_chosen`, `turn`, `category`, `judge_output_len`, `flipped` | Is judge error predictable from surface features the judge itself does not use? |
| **C** | B + verdict-position top-2 margin (exact — the verdict is constrained to 2 tokens) + `cot_logprob_{mean,min,std,p10}` + `cot_entropy_mean` (**truncated** top-20, a biased estimator — never call it "predictive entropy") | Is the usable signal in the token distribution rather than the verbalization? |

⚠️ **Tier C is unbuildable unless `judge.py` captures the full per-token logprobs at generation time (D4, amended 4 Sep 2026: 100% coverage, not a 10% sample).** `src/parse.py` derives the CoT aggregates from that saved data; neither the aggregates nor the raw logprobs they come from can be reconstructed from `raw_output` text alone, and rediscovering that in W5 means re-running the harness. Each aggregate is stored twice: the `sample_idx=0` value and the across-sample mean.

Read the outcomes:

- **A ≈ best single signal** → the signals are redundant; the judge has one internal notion of confidence. Clean finding, directly sharpens RQ2.
  ⚠️ **Reading caveat (D14):** `conf_sc` has only **5 levels** at `k_sc=4` and costs 4 extra generations per item, while `conf_lp` and `conf_bpe` are continuous and cost nothing extra. If self-consistency underperforms here, part of that is **resolution, not information**. Say so — and note the actionable version: *at equal or lower cost, the continuous signals carry finer-grained information than self-consistency at k=4.*
- **B ≫ A** → *judge error is predictable from surface features the judge ignores.* This is the best available outcome and it is very plausible given the verbosity-bias literature. It's a mildly damning result about judges and it's yours.
- **C ≫ B** → the signal lives in the token distribution, not the verbalization. Directly contradicts the "just ask for calibration" prior for open models — quotable.
- **Nothing beats the permutation null** → judge error is not predictable from cheap features at this scale. A real negative result, and the honest ceiling statement for the whole project.

All four are a slide. None of them is a failed project.

### 2.2b Compute, after D5

Fixing the `swap`/`order` double-naming reduced the run rather than growing it, and D19's prompt-ensemble addition reduced it again despite adding a whole new axis:

| condition | variant | sampling | orders | calls |
|---|---|---|---|---|
| clean | P1 | greedy | AB, BA | 2 |
| clean | P1 | sampled ×4 | AB only | 4 |
| clean | P2, P3 | greedy | AB, BA | 4 |
| verbose | P1 | greedy | AB, BA | 2 |
| **Total** | | | | **12 / item** |

At N=1000 × 12 = **12,000 generations**, against 18,000 under the pre-D19 3-condition/6-call schedule (itself already down from ~30,000 under the original pre-D5 schema). `conf_sc` is defined on the canonical AB presentation at clean/P1, so repeating sampled draws in BA order — or for P2/P3, or for `verbose` — buys nothing (and `verbose`'s and P2/P3's lack of sampling is exactly why `conf_sc` and `conf_ens` are clean/P1-only, D21).

**Budget the units, not the hours** (D12): measure seconds-per-generation at Gate 1, multiply by 12,000, **multiply by 3 for reruns** — reruns after a parser or prompt fix are the base case, not a contingency — then read the live burn rate from Colab's resource panel and compare against remaining monthly units *before* committing to W2. **If it's tight, cut N before cutting conditions.** N=600 with everything beats N=1000 with pieces missing. Prefer L4; reach for A100 only if measured L4 throughput fails the budget.

**Gate 1 measurement, PROVISIONAL — 8 Sep 2026.** Two timed cells from task 1.6's pilot on Colab L4: 120 generations in 180s, 20 generations in 60s (both `clean` condition - no `verbose` timing exists yet, task 4.1/4.2 haven't run). Solving both as `total = startup + n × rate` separates the per-call model-reload overhead from the real per-generation cost, rather than blending them: **startup ≈ 36s, rate ≈ 1.2 sec/generation**. A naive single blend (140 gens / 4 min ≈ 1.71 sec/gen) would overstate the steady-state rate, since a fixed ~36s reload cost dominates a 20-generation sample but is negligible at the real run's scale.

Using the **real Gate 0 N = 1904** (not the N=1000 placeholder line 86's "12,000" used before Gate 0 confirmed the actual count): total generations = 1904 × 12 × 3 (rerun factor) = **68,544**. Splitting by the schedule's own clean:verbose ratio (10:2 per item) and applying the measured 1.19x verbose prompt-token multiplier (`REPORT.md`) as an approximation for verbose's marginal rate (1.2 × 1.19 ≈ 1.43 sec/gen — an approximation, since prefill cost doesn't scale 1:1 with wall-clock the way decode does):

- Clean: 57,120 generations × 1.2 sec ≈ 19.04 hours
- Verbose: 11,424 generations × 1.43 sec ≈ 4.53 hours
- **Total ≈ 23.6 hours of L4 compute**

This exceeds a single Colab session (12h free-tier cap, 24h Pro cap) — the full run will span multiple sessions, which is exactly what the checkpoint/resume design (invariant 9) exists for. **Re-extrapolate once `perturb.py` lands (task 4.1b, W4)** with real verbose-condition timing instead of the prompt-token-multiplier approximation used here.

### 2.3 The label problem — and why it makes RQ4 and the human-disagreement angle reinforce each other

`correct` = judge verdict matches human majority. But on items where humans split ~50/50, "correct" is a coin flip, and no predictor can beat chance on those. Left alone, that noise floor caps your AUROC and you'd report a muddy number.

**Use human consensus as a continuous covariate, not a bucket** (D9). Bucketing starves: ~3,355 votes plausibly yields ~350 items with ≥2 votes, of which contested might be ~70 — too thin to survive GroupKFold.

- **Primary:** `d_human = |frac_prefer_a − 0.5|` as a continuous covariate. Test the **interaction** between `d_human` and the predictor's output. Uses every item with ≥2 votes; never needs a bucket to be large enough.
- **Secondary:** the agreed-vs-contested comparison, only if Gate 0 finds ≥100 contested items. "Agreed" means `human_agreed` (D16): `human_unanimous AND n_human_votes ≥ 2` — a single-vote item is trivially unanimous and must not count.
- **Hypothesis H4:** the predictor's edge grows with human consensus — judge error is learnable where humans agree (epistemic) and not where they don't (aleatoric).
- **The interaction CI is a cluster bootstrap over `question_id` on out-of-fold predictions (D15)** — never a default standard error. H4 is the one RQ4 output phrased outside the study's bootstrap-everything convention, which is exactly why it would otherwise slip through invariant #2.

If that holds, you have decomposed judge error into a learnable and an irreducible component, using a supervised model, with the aleatoric/epistemic vocabulary applied *correctly* rather than decoratively. That is the single most Srijith-shaped result available in this project, and RQ4 is what produces it. This is why RQ4 belongs in the study rather than in the extensions.

### 2.4 The transfer test — highest value per hour in the whole project

Train the predictor on `clean`. Test on `verbose` (`attribution` is cut, D18).

> Does an abstention layer trained on well-behaved data still work when the judge is under attack?

It will almost certainly degrade. That result — *"the safety net you trained offline fails exactly when you need it"* — is the most deployment-relevant sentence you will produce, and it costs one extra train/test split on data you already have.

⚠️ **Feature-parity fix required (D21):** `clean` has `conf_sc` and `conf_ens` (+ its decomposition); `verbose` has neither, since self-consistency and the P2/P3 ensemble are both clean/P1-only under D19's schedule. This transfer test uses **Tier A minus `{conf_sc, conf_ens, conf_ens_total, conf_ens_aleatoric, conf_ens_epistemic}`** for every model compared — not a Bayesian-specific carve-out. State this explicitly in `REPORT.md`'s limitations; never silently impute or leave NaN.

Second transfer test: `LeaveOneGroupOut` over the 8 MT-Bench categories. Does error prediction generalise across task types, or is it learning "coding questions are hard"?

### 2.5 Validation protocol — non-negotiable

- **`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` grouped by `question_id`, repeated over 10 seeds** (D8). Plain k-fold leaks (same question → shared prompt and often shared responses) and inflates AUROC by an amount you cannot estimate after the fact. ⚠️ `GroupKFold` has **no `shuffle`** — repeating it across seeds produces ten identical splits and a fake stability number.
- **There are only 80 questions.** Effective N for fold-to-fold variance is ~80, not ~1000. Report the **across-repeat spread** as the headline uncertainty: "AUROC 0.64, 10×5-fold spread [0.58, 0.69]".
- **No hyperparameter tuning, no nested CV.** `C=1.0` standardized; HistGBM at `max_depth=3, max_iter=200, lr=0.05`. Preregister both. At 80 groups, tuning is noise plus an indefensible researcher degree of freedom.
- **Permutation null:** shuffle labels within folds, 200 repeats, report the null AUROC distribution and where your number sits in it.
- **Baseline to beat:** the best single signal's AUROC from RQ2. Not chance — that's a strawman.
- Standardise features; report logistic-regression coefficients with CIs. **The coefficients are the result**, more than the AUROC is.
- **Calibrate the meta-model too.** It outputs P(judge is wrong), so it has its own reliability diagram. Calibrating a calibrator is a good slide and a genuine check.

### 2.6 The Bayesian hierarchical model (D22, professor feedback point 1)

A third model, alongside `LogisticRegression` (kept, frequentist baseline) and `HistGradientBoostingClassifier` (kept, nonlinearity check):

```
correct ~ Bernoulli(σ(α + α_q[question] + Xβ))
α_q ~ Normal(0, σ_q),  σ_q ~ HalfNormal(1),  β ~ Normal(0, 1)
```

Fit with NumPyro/NUTS, run under the **same** `StratifiedGroupKFold(5)`×10-seed protocol as §2.5 — the hierarchical structure adds partial pooling on top of that CV protocol, it does not replace it.

**Fallback ladder, preregistered:** NUTS → Laplace approximation (MAP via sklearn, Gaussian from the Hessian) → bootstrap ensemble of logistic regressions. State whichever rung was actually used in `REPORT.md`.

**Head-to-head table vs the frequentist model:** AUROC, ECE, Brier, NLL, 90% credible-interval coverage. NLL and coverage exist only for the Bayesian arm — no native posterior on the frequentist side to compute them from.

**Two things that are easy to get wrong here, so they're called out explicitly:**
1. A held-out question's `α_q` must be **marginalized over the population prior** (`α_q_new ~ Normal(0, σ_q)`), never its would-be fitted value — otherwise this leaks exactly the way plain `GroupKFold` leaks, which is the entire reason D8 exists. `test_bayesian.py` must assert this.
2. **Convergence diagnostics (R-hat, ESS, divergence count) are mandatory**, the Bayesian-model equivalent of the permutation null in §2.5 — never optional, always reported alongside the headline numbers.

**Runtime is measured, not assumed** (same principle as D12): 5 folds × 10 seeds = 50 fits. Time the first handful before committing to the full repeat count for this arm; reduce and preregister the reduction if 50 full NUTS runs turn out to be too slow for the week's budget, rather than discovering it mid-W5.

This is also where the **meta-model-level entropy decomposition** lives: from the Bayesian model's posterior predictive draws of P(judge wrong | x), compute the same Total/Aleatoric/Epistemic split as `conf_ens` (D20), but over parameter uncertainty rather than prompt variation. This is what the `verbose`-shift validation in §6 actually tests — not `conf_ens`, which is undefined for `verbose` (D21).

---

## 3. Revised week plan

Gates are pass/fail. A failed gate means cutting from §4, not slipping the week.

### W0 · Aug 24–30 · 9h — Data reality check
Theory A (calibration core, 3h) + Theory D (κ, 1.5h). Repo skeleton (1h). `src/data.py`: item table, tie policy, vote aggregation, **human–human Cohen's κ** (3h). Write `PREREGISTRATION.md` decisions D1–D3 *with fallbacks* before looking at results (0.5h).

**GATE 0** — item table exists. You know: N total, N non-tie, N with ≥2 votes, N unanimous, human–human κ.

### W1 · Aug 31–Sep 6 · 14h (peak week) — Harness
Course: **Getting Structured LLM Output** (1h21m) + **vLLM lessons 3, 6, 7, 8** (~46m). Theory E (LLM uncertainty signals, 2h) + **Theory I (Bayesian inference + NumPyro basics, 2h — start now, not W5)**, since it's the biggest new-concept lift in the whole revised plan and deserves runway. vLLM on Colab, **three** CoT pairwise prompt variants — P1 (existing), P2 (correctness-first rubric), P3 (helpfulness-first rubric) — each frozen + hashed (3h, was 2h for one, D19). All four existing signals end-to-end on 20 items; `parse.py` + fixtures + tests (3h). Checkpoint/resume proven by killing the runtime mid-run (1h). Vacuum test: 40 identical/empty-response pairs → dark-current number (0.5h).

**GATE 1** — pilot complete: **20 items × `clean`/P1 × 6 calls = 120 generations**, plus a light P2/P3 smoke test (all three templates parse cleanly), the 10-item verbose smoke test, and the T=0 vs T=0.7 `p_a` probe. Parse failure <5% across all three prompt variants. Full-run GPU hours extrapolated as **N × 12 calls × 3 (rerun factor)** (D19) and written down **marked PROVISIONAL**. **`PREREGISTRATION.md` frozen and committed** — now including D18–D24 as binding. `src/prompts.py` frozen (all three variants).

### W2 · Sep 7–13 · 13h — Clean run (P1+P2+P3) + RQ1
Theory B (discrimination vs calibration, 1h) + F (bootstrap, 1h). Full `clean` run — P1's 6 calls + P2/P3's 4 calls = **10 calls/item** for `clean` (D19) — both orders where applicable (2.5h attention). `metrics.py` + `test_metrics.py` against the hand-computed ECE (2h). **`signals.py`: `conf_ens` + judge-level entropy decomposition (D20) — new (2h).** `items.parquet` built at its new **`(item_id, condition, prompt_variant)`** grain, with the P1-filter (invariant 14) applied and tested (1h). Reliability diagrams, ECE/Brier/gap/accuracy/κ with cluster-bootstrap CIs, on P1 only (2.5h). Append to `REPORT.md` (0.5h).

**GATE 2** — RQ1 answered with CIs, on P1. `conf_ens` populated for every `clean` item, null for every `verbose` item — the D21 sanity check.

### W3 · Sep 14–20 · 11h — RQ2 + human-disagreement + threshold sweep
Theory C (selective prediction, 1.5h) + G (aleatoric/epistemic, 1h). Risk–coverage + oracle curve, AURC, accuracy@coverage, κ@coverage, AUROC per signal (3h). **Extend `risk_coverage()` with the threshold sweep on total/aleatoric/epistemic entropy + ECE-on-retained (D20) — new (2h).** Human-disagreement decomposition (2.5h). `REPORT.md` (1h).

**GATE 3** — RQ2 answered. **One figure that is the whole thesis.** Expect flatness; Xiong et al. predict it, and the oracle overlay makes it interpretable rather than embarrassing. The epistemic-vs-total threshold-sweep prediction is stated and tested (not yet confirmed) — epistemic should beat total.

### W4 · Sep 21–27 · 8h — RQ3
`perturb.py` for `verbose` only, with property tests (1.5h — lighter than before, `attribution` is cut, D18). Re-extrapolate the GPU budget (0.5h). Runs (1h attention). Paired analysis: flip rate, **mean confidence on flipped vs unflipped**, ΔECE/ΔAUROC with paired cluster-bootstrap, verbosity only (2.5h). Constrained-vs-free-form decoding ablation on 100 items (1.5h). `REPORT.md` (1h).

**GATE 4** — RQ3 answered in one sentence with a CI, verbosity only.

### W5 · Sep 28–Oct 4 · 15h (second peak week) — RQ4 + RQ5
`features.py` tiers A/B/C + tests (3h). `predictor.py`: repeated StratifiedGroupKFold, LogReg + HistGBM, permutation null (3h). **`bayesian.py`: hierarchical model, NUTS fit, held-out marginalization, convergence diagnostics (4h — new, the single biggest addition, D22).** Head-to-head table: AUROC/ECE/Brier/NLL/coverage (1h). Tier ablation (1h). Transfer tests: clean→verbose (Tier A minus `conf_sc`/`conf_ens`, D21), `LeaveOneCategoryOut` (1.5h). Meta-model calibration + coefficient plot (0.5h). **RQ5: distillation comparison (ensemble vs single-call Bayesian model) + human-disagreement validation of the aleatoric estimate (D23) (1.5h).** `REPORT.md`, RQ4 and RQ5 sections (0.5h).

**GATE 5** — RQ4 answered against the permutation null and the best-single-signal baseline, for **both** the frequentist and Bayesian models, with convergence diagnostics reported. H4 tested. **RQ5 answered: the distillation gap is stated, and the aleatoric-vs-`d_human` validation is done.**

### W6 · Oct 5–11 · 12h — Demo + writeup
Course: **Building Generative AI Applications with Gradio** (59m). Demo: cached results + one live 4-bit small-model call + the **"trick the judge"** toggle where the verdict flips and the confidence doesn't (5h). Report finished from the incremental draft, now including RQ5 (3.5h). Slides, anchored on RQ2's thesis figure, RQ4's coefficient plot, **and the RQ5 distillation result** (2.5h).

**GATE 6** — demo runs offline on the Nitro 5. Report complete, RQ1–RQ5 all covered.

### W7 · Oct 12–15 · 4h — Reproducibility + buffer
Fresh clone, run from config, confirm numbers match. Leave the rest empty. Re-runs after discovering a parser bug is where every estimate of this kind has failed.

---

## 4. Drop order (last listed dies first)

1. Second judge model — cut
2. Multilingual condition — cut
3. Constrained-vs-free-form ablation
4. Tier C features (RQ4 runs on A/B only)
5. Cross-category transfer test
6. `verbose` condition
7. RQ5's human-disagreement validation of the aleatoric estimate (D23) — the newest, most exploratory piece; the distillation comparison itself is protected below
8. ← *protect everything below this line*
9. RQ5's distillation comparison (D23) — required by the professor, same protected tier as RQ4
10. The Bayesian model (D22) — required; the NUTS → Laplace → bootstrap fallback ladder (D22) is the actual risk mitigation here, not a cut. Reaching this line means all three rungs failed.
11. RQ4 core (tiers A/B, repeated StratifiedGroupKFold, permutation null, H4)
12. Human-disagreement decomposition
13. RQ1, RQ2, the `clean`/P1 run in both orders

`attribution` no longer appears — it isn't a drop candidate, it's gone (D18).

---

## 5. Theory blocks

~18h (was ~13h — professor feedback added Bayesian inference, BALD, and distillation), in weekly slices. Only Block A precedes code. Details, including "what people usually fake", are in `LEARNING.md`.

| Block | Topic | Hours | Week |
|---|---|---|---|
| A | Calibration core (Guo et al.) | 3 | W0 |
| D | Agreement statistics (Cohen's κ) | 1.5 | W0 |
| E | LLM uncertainty signals (Tian, Xiong) | 2 | W1 |
| **I** | **Bayesian inference + NumPyro/NUTS basics (D22)** | **2** | **W1** |
| B | Discrimination vs calibration | 1 | W2 |
| F | Bootstrap (cluster, paired) | 1 | W2 |
| C | Selective prediction | 1.5 | W3 |
| G | Aleatoric vs epistemic | 1 | W3 |
| **K** | **BALD / mutual information (entropy decomposition, D20)** | **1** | **W3** |
| **J** | **Hierarchical / partial-pooling models (D22)** | **1** | **W5** |
| **L** | **Distributional distillation (D23)** | **1** | **W5** |
| H | Conformal prediction (extensions slide only) | 2 | W6, optional |

New blocks (I, J, K, L) exist because everything learned so far (ECE, κ, bootstrap, GroupKFold) is frequentist — Bayesian hierarchical modeling, MCMC/NUTS, and mutual-information decomposition are a genuine step up, not an incremental add. Block I is deliberately placed in W1, not W5, so there's runway before the model itself has to be built.

---

## 6. RQ5 — prompt-ensemble distillation *(added 31 Aug 2026, professor feedback)*

### 6.1 Positioning

Closest prior work: *Auto-Prompt Ensemble for LLM Judge* (Oct 2025 — same Qwen2.5-7B/MT-Bench setup as this project) and *Calibrating MLLM-as-a-Judge via Multimodal Bayesian Prompt Ensembles* (ICCV 2025). Both use prompt ensembles for judging; **neither decomposes entropy into aleatoric/epistemic, and neither validates against real repeated human votes.** That combination — the decomposition plus the human-vote validation — is what RQ5 actually contributes, not the ensemble-prompting idea itself, which is not novel.

### 6.2 The question

> Does marginalizing over the judge prompt (asking the same clean comparison three different ways — P1, P2, P3) produce a better uncertainty estimate than any single prompt, and how much of that benefit survives distilling it down to single-call cost?

Three frozen, hashed prompt variants (D19): **P1** (existing template, primary — RQ1–RQ4 use this alone), **P2** (correctness-first rubric), **P3** (helpfulness-first rubric). Uniform prior over the three.

### 6.3 The signal and its decomposition (D20)

`conf_ens = 1 − ens_entropy_total`, computed from the three variants' greedy `p_a` values on `clean` items only:

- **Total** = H[mean(p_a across P1, P2, P3)]
- **Aleatoric** = mean(H[p_a]) across the three variants
- **Epistemic** = Total − Aleatoric (mutual information / BALD)

This feeds RQ2's threshold-sweep extension (D20) directly — sweep raw thresholds on each of the three separately, report coverage/accuracy/κ/ECE-on-retained at each, and check the preregistered prediction that epistemic thresholding beats total.

### 6.4 The distillation mechanism (D23)

The ensemble's **predictive distribution** — not just its mean — is the teacher. The Bayesian meta-model (§2.6, D22), trained on **single-call, P1-only** features, is the student: does its own posterior predictive spread resemble the expensive 3-call ensemble's actual spread? No LLM fine-tuning is involved. **Headline metric:** how much of the ensemble's AUROC/ECE/entropy-quality benefit survives at 1 call vs 3 — this is the deployment-relevant number, parallel to RQ4's transfer-test framing ("the safety net trained offline...").

### 6.5 Validating the decomposition against human disagreement

MT-Bench's repeated human votes give a real, model-free measure of how contested each item genuinely is (`d_human`, D9) — most judge-uncertainty papers don't have this, because they don't have repeated votes. **Test:** is the estimated **aleatoric** component high specifically where humans actually disagreed (high `d_human`), and is **epistemic** not? This reuses H4's existing machinery rather than inventing new statistics — it validates the aleatoric/epistemic vocabulary against an independent signal, which is stronger evidence than either alone. Runs on `clean`/P1 rows only, per invariant 14.

### 6.6 The verbose-shift check (also D23, via D22's meta-model decomposition, not `conf_ens`)

Train the Bayesian model on `clean`, evaluate on `verbose`. Preregistered prediction: **epistemic** uncertainty rises under this distribution shift while **aleatoric** stays flat — the canonical sanity check that a Bayesian model's epistemic estimate is doing its job. This is necessarily a **meta-model-level** check (§2.6), not a `conf_ens` one — `conf_ens` is undefined for `verbose`, which never collects P2/P3 (D21). Uses the Tier A-minus-`conf_sc`/`conf_ens` feature set from §2.4.
