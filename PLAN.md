# PLAN.md — design rationale and schedule (v2.1)

Supersedes build plan v1. Change from v1: **RQ4 (learned error predictor) is restored as a first-class research question**, and the hour budget is raised to match.

**v2.1 incorporates the Week-0 design review.** `DECISIONS.md` D4–D17 override anything here that contradicts them. The three that changed the harness: Tier C needs CoT logprob aggregates captured at generation time (D4); `swap` is not a condition, `order` is an axis (D5); temperature needs two config keys or `conf_sc` is a dead constant (D6). Two more from a second pass changed statistics rather than the harness: `ece()` must handle discrete-valued signals like `conf_sc` exactly rather than binning them (D14), and H4's interaction CI must be a cluster bootstrap over out-of-fold predictions, never a default standard error (D15). A third pass added two more: "human-agreed," used as the Tier-ablation baseline population, needs the same ≥2-vote floor H4 already uses for `d_human` — a single-vote item is trivially unanimous (D16); and D17 settles the local/Colab split — local dev never installs vLLM, Colab gets its own fresh venv, and a GitHub remote bridges code between the two.

---

## 1. Budget

| | v1 | v2 |
|---|---|---|
| Elapsed | 7.4 weeks (Aug 24 → Oct 15) | same |
| Core study (RQ1–RQ3) | 55h | 55h |
| RQ4 | cut | **+11h** |
| Courses | 0h | **+5h** |
| Reading rebalance | — | +5h |
| **Total** | 55h | **~75h** |
| **Per week** | 7.5h | **~10h** |

Weekly figures below are ±1h. Treat them as load indicators, not contracts.

RQ4 costs **zero extra GPU time** — it runs on data already collected for RQ1–RQ3. It is paid entirely in analysis hours, which is the resource you just offered. That is the argument for putting it back, and it's a good one.

**Weekly load is not flat.** W1 is the peak at **~12h** (two courses plus the entire harness). If that week overruns, the relief valves, in order: (a) skip the vLLM course lessons entirely and take 30 min of vLLM docs instead, (b) slide Xiong et al. to W2. Do **not** relieve W1 by skipping the checkpoint/resume work — that is what protects every later week from a Colab disconnect.

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
| **A** | `conf_verb`, `conf_lp`, `conf_sc`, `conf_bpe` | Do the four uncertainty signals *combine*, or are they redundant? |
| **B** | A + `len_a`, `len_b`, `len_ratio`, `abs_len_diff`, `longer_is_chosen`, `turn`, `category`, `judge_output_len`, `flipped` | Is judge error predictable from surface features the judge itself does not use? |
| **C** | B + verdict-position top-2 margin (exact — the verdict is constrained to 2 tokens) + `cot_logprob_{mean,min,std,p10}` + `cot_entropy_mean` (**truncated** top-20, a biased estimator — never call it "predictive entropy") | Is the usable signal in the token distribution rather than the verbalization? |

⚠️ **Tier C is unbuildable unless `judge.py` captures those CoT aggregates at generation time (D4).** They cannot be reconstructed from `calls.parquet` afterwards, and rediscovering that in W5 means re-running the harness. Each is stored twice: the `sample_idx=0` value and the across-sample mean.

Read the outcomes:

- **A ≈ best single signal** → the signals are redundant; the judge has one internal notion of confidence. Clean finding, directly sharpens RQ2.
  ⚠️ **Reading caveat (D14):** `conf_sc` has only **5 levels** at `k_sc=4` and costs 4 extra generations per item, while `conf_lp` and `conf_bpe` are continuous and cost nothing extra. If self-consistency underperforms here, part of that is **resolution, not information**. Say so — and note the actionable version: *at equal or lower cost, the continuous signals carry finer-grained information than self-consistency at k=4.*
- **B ≫ A** → *judge error is predictable from surface features the judge ignores.* This is the best available outcome and it is very plausible given the verbosity-bias literature. It's a mildly damning result about judges and it's yours.
- **C ≫ B** → the signal lives in the token distribution, not the verbalization. Directly contradicts the "just ask for calibration" prior for open models — quotable.
- **Nothing beats the permutation null** → judge error is not predictable from cheap features at this scale. A real negative result, and the honest ceiling statement for the whole project.

All four are a slide. None of them is a failed project.

### 2.2b Compute, after D5

Fixing the `swap`/`order` double-naming reduced the run rather than growing it:

| | calls per (item, condition) |
|---|---|
| Canonical greedy (T=0), both orders | 2 |
| Self-consistency draws (T=0.7), AB order only | 4 |
| **Total** | **6** |

At N=1000 × 3 real conditions (`clean`, `verbose`, `attribution`) × 6 = **18,000 generations**, against ~30,000 under the old schema. `conf_sc` is defined on the canonical AB presentation, so repeating sampled draws in BA order buys nothing.

**Budget the units, not the hours** (D12): measure seconds-per-generation at Gate 1, multiply by 18,000, **multiply by 3 for reruns** — reruns after a parser or prompt fix are the base case, not a contingency — then read the live burn rate from Colab's resource panel and compare against remaining monthly units *before* committing to W2. **If it's tight, cut N before cutting conditions.** N=600 with all conditions beats N=1000 with two. Prefer L4; reach for A100 only if measured L4 throughput fails the budget.

### 2.3 The label problem — and why it makes RQ4 and the human-disagreement angle reinforce each other

`correct` = judge verdict matches human majority. But on items where humans split ~50/50, "correct" is a coin flip, and no predictor can beat chance on those. Left alone, that noise floor caps your AUROC and you'd report a muddy number.

**Use human consensus as a continuous covariate, not a bucket** (D9). Bucketing starves: ~3,355 votes plausibly yields ~350 items with ≥2 votes, of which contested might be ~70 — too thin to survive GroupKFold.

- **Primary:** `d_human = |frac_prefer_a − 0.5|` as a continuous covariate. Test the **interaction** between `d_human` and the predictor's output. Uses every item with ≥2 votes; never needs a bucket to be large enough.
- **Secondary:** the agreed-vs-contested comparison, only if Gate 0 finds ≥100 contested items. "Agreed" means `human_agreed` (D16): `human_unanimous AND n_human_votes ≥ 2` — a single-vote item is trivially unanimous and must not count.
- **Hypothesis H4:** the predictor's edge grows with human consensus — judge error is learnable where humans agree (epistemic) and not where they don't (aleatoric).
- **The interaction CI is a cluster bootstrap over `question_id` on out-of-fold predictions (D15)** — never a default standard error. H4 is the one RQ4 output phrased outside the study's bootstrap-everything convention, which is exactly why it would otherwise slip through invariant #2.

If that holds, you have decomposed judge error into a learnable and an irreducible component, using a supervised model, with the aleatoric/epistemic vocabulary applied *correctly* rather than decoratively. That is the single most Srijith-shaped result available in this project, and RQ4 is what produces it. This is why RQ4 belongs in the study rather than in the extensions.

### 2.4 The transfer test — highest value per hour in the whole project

Train the predictor on `clean`. Test on `verbose` and `attribution`.

> Does an abstention layer trained on well-behaved data still work when the judge is under attack?

It will almost certainly degrade. That result — *"the safety net you trained offline fails exactly when you need it"* — is the most deployment-relevant sentence you will produce, and it costs one extra train/test split on data you already have.

Second transfer test: `LeaveOneGroupOut` over the 8 MT-Bench categories. Does error prediction generalise across task types, or is it learning "coding questions are hard"?

### 2.5 Validation protocol — non-negotiable

- **`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` grouped by `question_id`, repeated over 10 seeds** (D8). Plain k-fold leaks (same question → shared prompt and often shared responses) and inflates AUROC by an amount you cannot estimate after the fact. ⚠️ `GroupKFold` has **no `shuffle`** — repeating it across seeds produces ten identical splits and a fake stability number.
- **There are only 80 questions.** Effective N for fold-to-fold variance is ~80, not ~1000. Report the **across-repeat spread** as the headline uncertainty: "AUROC 0.64, 10×5-fold spread [0.58, 0.69]".
- **No hyperparameter tuning, no nested CV.** `C=1.0` standardized; HistGBM at `max_depth=3, max_iter=200, lr=0.05`. Preregister both. At 80 groups, tuning is noise plus an indefensible researcher degree of freedom.
- **Permutation null:** shuffle labels within folds, 200 repeats, report the null AUROC distribution and where your number sits in it.
- **Baseline to beat:** the best single signal's AUROC from RQ2. Not chance — that's a strawman.
- Standardise features; report logistic-regression coefficients with CIs. **The coefficients are the result**, more than the AUROC is.
- **Calibrate the meta-model too.** It outputs P(judge is wrong), so it has its own reliability diagram. Calibrating a calibrator is a good slide and a genuine check.

---

## 3. Revised week plan

Gates are pass/fail. A failed gate means cutting from §4, not slipping the week.

### W0 · Aug 24–30 · 9h — Data reality check
Theory A (calibration core, 3h) + Theory D (κ, 1.5h). Repo skeleton (1h). `src/data.py`: item table, tie policy, vote aggregation, **human–human Cohen's κ** (3h). Write `PREREGISTRATION.md` decisions D1–D3 *with fallbacks* before looking at results (0.5h).

**GATE 0** — item table exists. You know: N total, N non-tie, N with ≥2 votes, N unanimous, human–human κ.

### W1 · Aug 31–Sep 6 · 12h (peak week) — Harness
Course: **Getting Structured LLM Output** (1h21m) + **vLLM lessons 3, 6, 7, 8** (~46m). Theory E (LLM uncertainty signals, 2h). vLLM on Colab, CoT pairwise prompt frozen + hashed (2h). All four signals end-to-end on 20 items; `parse.py` + fixtures + tests (3h). Checkpoint/resume proven by killing the runtime mid-run (1h). Vacuum test: 40 identical/empty-response pairs → dark-current number (0.5h).

**GATE 1** — pilot complete: **20 items × `clean` × 6 calls = 120 generations**, plus the 10-item verbose smoke test and the T=0 vs T=0.7 `p_a` probe. Parse failure <5%. Full-run GPU hours extrapolated as **N × 3 conditions × 6 calls × 3 (rerun factor)** and written down **marked PROVISIONAL**. **`PREREGISTRATION.md` frozen and committed.**

### W2 · Sep 7–13 · 9h — Clean run + RQ1
Theory B (discrimination vs calibration, 1h) + F (bootstrap, 1h). Full `clean` run, both orders (2h attention). `metrics.py` + `test_metrics.py` against the hand-computed ECE (2h). Reliability diagrams, ECE/Brier/gap/accuracy/κ with cluster-bootstrap CIs (2.5h). Append to `REPORT.md` (0.5h).

**GATE 2** — RQ1 answered with CIs. You can state judge overconfidence in one sentence with a number.

### W3 · Sep 14–20 · 9h — RQ2 + human-disagreement
Theory C (selective prediction, 1.5h) + G (aleatoric/epistemic, 1h). Risk–coverage + oracle curve, AURC, accuracy@coverage, κ@coverage, AUROC per signal (3h). Human-disagreement decomposition (2.5h). `REPORT.md` (1h).

**GATE 3** — RQ2 answered. **One figure that is the whole thesis.** Expect flatness; Xiong et al. predict it, and the oracle overlay makes it interpretable rather than embarrassing.

### W4 · Sep 21–27 · 9h — RQ3
`perturb.py` for verbose + attribution, with property tests (2h). Re-extrapolate the GPU budget (0.5h). Runs (1h attention). Paired analysis: flip rate, **mean confidence on flipped vs unflipped**, ΔECE/ΔAUROC with paired cluster-bootstrap (3h). Constrained-vs-free-form decoding ablation on 100 items (2h). `REPORT.md` (1h).

**GATE 4** — RQ3 answered in one sentence with a CI.

### W5 · Sep 28–Oct 4 · 11h — RQ4
`features.py` tiers A/B/C + tests (3h). `predictor.py`: repeated StratifiedGroupKFold, logreg + HistGBM, permutation null (3h). Tier ablation (1.5h). Transfer tests: clean→perturbed, LeaveOneCategoryOut (2h). Meta-model calibration + coefficient plot (1h). `REPORT.md` (0.5h).

**GATE 5** — RQ4 answered with a permutation null and the best-single-signal baseline. H4 (agreed vs contested) tested.

### W6 · Oct 5–11 · 11h — Demo + writeup
Course: **Building Generative AI Applications with Gradio** (59m). Demo: cached results + one live 4-bit small-model call + the **"trick the judge"** toggle where the verdict flips and the confidence doesn't (5h). Report finished from the incremental draft (3h). Slides (2h).

**GATE 6** — demo runs offline on the Nitro 5. Report complete.

### W7 · Oct 12–15 · 4h — Reproducibility + buffer
Fresh clone, run from config, confirm numbers match. Leave the rest empty. Re-runs after discovering a parser bug is where every estimate of this kind has failed.

---

## 4. Drop order (last listed dies first)

1. Second judge model — cut
2. Multilingual condition — cut
3. Constrained-vs-free-form ablation
4. `attribution` condition
5. Tier C features (RQ4 runs on A/B only)
6. Cross-category transfer test
7. `verbose` condition
8. ← *protect everything below this line*
9. RQ4 core (tiers A/B, repeated StratifiedGroupKFold, permutation null, H4)
10. Human-disagreement decomposition
11. RQ1, RQ2, the `clean` run in both orders

---

## 5. Theory blocks

~13h, in weekly 2-hour slices. Only Block A precedes code. Details, including "what people usually fake", are in `LEARNING.md`.

| Block | Topic | Hours | Week |
|---|---|---|---|
| A | Calibration core (Guo et al.) | 3 | W0 |
| D | Agreement statistics (Cohen's κ) | 1.5 | W0 |
| E | LLM uncertainty signals (Tian, Xiong) | 2 | W1 |
| B | Discrimination vs calibration | 1 | W2 |
| F | Bootstrap (cluster, paired) | 1 | W2 |
| C | Selective prediction | 1.5 | W3 |
| G | Aleatoric vs epistemic | 1 | W3 |
| H | Conformal prediction (extensions slide only) | 2 | W6, optional |
