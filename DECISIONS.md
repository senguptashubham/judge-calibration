# Design decisions

This file records the design decisions behind the study, one entry per decision, as
they stand now. The code and `REPORT.md` refer to them by number. Each entry gives:

- what was decided;
- why;
- where it shows up.

Decisions marked ⚑ were frozen in [`PREREGISTRATION.md`](PREREGISTRATION.md) before
the main run, or affect how a preregistered result is read. The full development
history is kept at the git tag `dev-phase-end`: the review findings, the
measurements behind each decision, and every amendment.

Read it with `git show dev-phase-end:DECISIONS.md`.

---

## Labels and data

### D1–D3 ⚑ — Tie policy, label construction, no augmentation
These are recorded in `PREREGISTRATION.md` §4–6.

- **Ties:** an item whose human votes are tied is excluded, so it never gets a label.
- **Label:** each remaining item's label is the majority over its non-tie votes.
- **Size:** this leaves 1,904 usable items, well clear of the 150-item fallback.
- **No augmentation:** RewardBench 2 was not needed.

Where: `src/data.py::build_items()`; `src/judge.py` judges non-tie items only.

### D9 ⚑ — Human disagreement is a continuous covariate
**Decision.** Human disagreement enters as a continuous covariate,
`d_human = |frac_prefer_a − 0.5|`, over items with at least two votes. H4 tests the
interaction between `d_human` and the meta-model's out-of-fold score. An
agreed-vs-contested bucket comparison was secondary, and was allowed only if there
were at least 100 contested items.

**Why.** Bucketing starves the analysis. Only 31 contested items survive the two-vote
filter, too few for grouped cross-validation. The continuous form uses every
multi-vote item.

Where: `src/data.py`, `analysis/human_disagreement.py`, `analysis/rq4.py --task h4`.

### D16 ⚑ — "Human-agreed" needs at least two votes
**Decision.** `human_agreed = human_unanimous AND n_human_votes ≥ 2`. The column is
computed once, in `build_items()`.

**Why.** A single-vote item is trivially unanimous: there is nobody to disagree. The
two-vote floor stops the least-informative items from being treated as the most
certain ground truth, for example in RQ4's tier ablation.

Where: `src/data.py`, `analysis/rq4.py`.

---

## Design of the judge runs

### D4 ⚑ — Save every call's full per-token logprobs
**Decision.** Generation uses `SamplingParams(logprobs=20)`. For every call, the full
per-token top-20 logprobs go to `runs/{model}/logprobs/*.jsonl.gz`. `src/parse.py`
derives every logprob signal from those saved files: `p_a`, the verdict-token
logprob, and the CoT aggregates. The aggregates are stored twice, kept apart by
name:

- `cot_*_greedy`, from the T=0 call;
- `cot_*_sampled_t07`, averaged over the T=0.7 draws.

**Why.**

- Tier C's chain-of-thought statistics cannot be rebuilt from text, so missing them
  at generation time would mean re-running the harness.
- Saving everything keeps parsing in one place (`parse.py`). `judge.py` only runs the
  model.
- Top-20 entropy is a truncated, biased estimator. It is labelled "truncated", never
  "predictive entropy".
- Greedy and sampled statistics are on different temperature scales (see D6), so
  separate names make it impossible to mix them up.

Where: `src/judge.py`, `src/parse.py`, `src/features.py` (Tier C).

### D5 ⚑ — `order` is an axis, not a condition
**Decision.** Conditions are content manipulations: `clean`, `verbose`, and `vacuum`
as a sanity check. `order` ∈ {AB, BA} is a separate axis, collected for every
condition. `flipped` and `conf_bpe` are computed within a condition, across its two
orders. RQ3a's position-bias test is the flip rate within `clean`.

**Why.** In a pairwise prompt, swapping the presentation order and swapping the A/B
labels are the same operation. Treating `swap` as a condition as well would have
duplicated the collection.

Where: `src/prompts.py` (order renderer; round-trip test in `test_prompts.py`),
`src/signals.py`.

### D6 ⚑ — Two temperatures; logprob signals come from the greedy call only
**Decision.**

- `temperature_canonical = 0.0` for `sample_idx = 0`.
- `temperature_sc = 0.7` for the `k_sc = 4` draws.
- The per-call seed is `seed + sample_idx`.
- `conf_sc` is the fraction of the sampled verdicts that match the greedy verdict.
- `p_a` and `conf_lp` come from the greedy call only.

**Why.** One temperature of 0 would make every draw identical, so `conf_sc` would be a
constant. vLLM also returns logprobs from the temperature-scaled distribution, so
`p_a` at T=0.7 is not the same quantity as `p_a` at T=0. This was confirmed
empirically; see REPORT's methods notes.

Where: `configs/run.yaml`, `src/config.py` (validation), `src/parse.py`,
`src/signals.py`.

### D18 — No `attribution` condition
**Decision.** The authorship-label condition was dropped on the professor's advice.
The conditions are `clean` and `verbose`, plus the `vacuum` probe. This also retires
D13, which settled how that condition should be named.

Where: `configs/run.yaml`, `src/config.py`, `src/perturb.py` (builds only
`verbose_pad`).

### D19 ⚑ — Three prompt variants, and the 12-call schedule
**Decision.** There are three frozen, hashed judge templates:

- `P1`: the MT-Bench template, the primary one;
- `P2`: a correctness-first rubric;
- `P3`: a helpfulness-first rubric.

`prompt_variant` is an axis like `order`. Each item gets 12 calls:

| condition | variant | sampling | orders | calls |
|---|---|---|---|---|
| clean | P1 | greedy | AB, BA | 2 |
| clean | P1 | sampled ×4 | AB | 4 |
| clean | P2, P3 | greedy | AB, BA | 4 |
| verbose | P1 | greedy | AB, BA | 2 |

**Why.** The ensemble exists to judge the clean comparison three ways (RQ5), so it
only needs `clean`. Self-consistency is defined on the AB presentation, so sampling
in BA, in P2/P3 or in `verbose` would add nothing.

Where: `src/prompts.py`, `src/judge.py::calls_for_item()`.

### D10–D12 — Pilot, vLLM pin, compute budget
**Pilot (D10).** The pilot was 20 clean items. A throwaway verbose smoke test
measured how much padding lengthens the prompt.

**vLLM pin (D11).** `vllm==0.28.0` is pinned in the `colab` extra and was never moved.
It is installed in a fresh venv on Colab, because Colab's own `torch` conflicts with
it. A smoke test confirmed that structured decoding and logprobs work together on
this version.

**Budget (D12).** The budget was measured, not guessed: seconds per generation × calls
× 3 for reruns. If the budget had been short, the rule was to cut N before cutting
conditions. At N = 1,904 it fitted on an L4 without cuts.

### D17 — Split between local machine and Colab
**Decision.**

- Only GPU inference runs on Colab, in a fresh venv with the `colab` extra, which
  adds vLLM.
- Everything else runs locally in the `judge-calib` conda env (Python 3.11), which
  never installs vLLM. `judge.py` imports vLLM inside a function, so the package
  stays importable without it.
- GitHub carries the code between the two. Generated data never travels through git.
- `runs/` and `*.parquet` are gitignored.
- `results/*.csv` and `results/figures/*.png` are tracked. They are small, they are
  the evidence `REPORT.md` cites, and committing them with the report ties each number
  to its text.

Where: `pyproject.toml`, `.gitignore`.

### D24 — The Bayesian dependencies run locally
**Decision.** `numpyro`, `jax` (CPU) and `arviz` are in the base install, not the
`colab` extra.

**Why.** Fitting the meta-model is analysis on data already collected. It needs no
GPU.

### D25 ⚑ — Constrained decoding inflates `conf_lp`, but not `p_a`
**Decision.** `conf_lp` is reported as an upper bound on the judge's confidence.
`conf_verb` is the primary "is the judge falsely confident" signal. `p_a` and
everything built on it (`verdict_bidir`, `conf_bpe`) are unaffected. vLLM stays at
0.28.0.

**Why.** Structured decoding masks every disallowed token, then renormalizes over
`{A, B}`. That inflates the absolute verdict-token probability. A ratio between the
two surviving tokens survives the renormalization unchanged.

vLLM's unmasked-logprob option (`logprobs_mode`) existed only in an unreleased
version, so pinning to it was rejected. The constrained-vs-free-form ablation
measures how much the choice matters: 4% of verdicts change.

Where: `src/parse.py`, `src/ablation_decoding.py`, `analysis/decoding_ablation.py`.

### D26 — Every model-dependent file carries the model slug
**Decision.** `Config.model_slug` turns the judge's name into a filename tag, for
example `Qwen/Qwen2.5-7B-Instruct` → `qwen2.5_7b_instruct`.

- `runs/` gets one subfolder per model.
- `results/` stays a single folder, with `_{model_slug}` on each filename, so a
  cross-model script can glob it.
- `items_labels.parquet` is never suffixed, because it depends only on human votes.

**Why.** A second or third judge can be added without touching existing outputs.

Where: `src/config.py`, `src/plots.py`, every `analysis/*.py` writer.

---

## Item-level signals

### D7 ⚑ — What "the judge's verdict" is, and which confidence goes with it
**Decision.**

- **Primary verdict:** `judge_verdict`, the greedy verdict in AB order. This is what
  a deployment running one pass gets. `correct` is scored against it.
- **Secondary verdict:** `verdict_bidir`, the argmax of the mean `p_a` across both
  orders. `correct_bidir` is scored against it.
- **A confidence is scored only against the verdict it belongs to.**
  - `conf_verb`, `conf_lp` and `conf_sc` describe the AB verdict.
  - Against `verdict_bidir`, calibration uses `conf_verb_bidir` and `conf_lp_bidir`.
- **Entropy signals rank items, but they are not probabilities.**
  - `conf_bpe`, `conf_ens` and the kev/auto-j analogues have the form 1 − H.
  - AUROC uses them as they are.
  - ECE, Brier and the overconfidence gap use their probability form:
    `signals.py::prob_on_verdict()`, which gives `conf_bpe_prob`, `conf_ens_prob`, and
    so on.

**Why.**

- On the roughly 28% of items whose orders flip, `verdict_bidir` is a different
  answer from the AB verdict.
- A 50/50 split scores 0.307 on 1 − H, not 0.5, so a calibration metric computed on
  the entropy form partly measures that difference in scale.

Where: `src/signals.py`, `analysis/rq1.py`; REPORT's methods note "Calibration
forms…".

### D14 ⚑ — Discrete signals: `ece()` binning and tied ranks
**Decision.** `ece()` chooses its own binning strategy.

- If a signal has no more distinct values than there are bins, `ece()` bins by unique
  value. That is the exact ECE, not an approximation.
- Otherwise it uses equal-mass bins, with duplicate edges dropped.

It always returns `(ece, n_effective_bins)`, and every table prints both.

**Why.** At `k_sc = 4`, `conf_sc` takes only five values. Quantile edges would land
inside the ties.

There is also a known confound: `conf_sc` costs 4 extra calls and has 5 levels,
while `conf_lp` and `conf_bpe` are continuous and free. Some of `conf_sc`'s weakness
is therefore resolution, not information. This is reported as such, not fixed by
raising `k_sc`.

**Ties in the ranking metrics.** Coverage-based metrics take a tied group by its
expected value under a random tie-break, never in row order. `risk_coverage()`
gives one point per k = 1..n, with the boundary group contributing its error rate
pro rata. `coverage_weights()` gives the boundary group equal fractional weight
for accuracy and κ at a coverage level. `aurc()` integrates from coverage 0, as
Geifman and El-Yaniv define it. Row order in the item table is correlated with
correctness, so a stable-sort cut biased `conf_verb` and `conf_sc` upward, and
integrating only from a curve's first point let a 5-valued signal skip most of
its area.

Where: `src/metrics.py::ece`, `risk_coverage`, `aurc`, `coverage_weights`;
`tests/test_metrics.py`.

### D20 ⚑ — `conf_ens` and the judge-level entropy decomposition; the P1 filter
**Decision.** For a `clean` item, these are computed over the greedy `p_a` from
P1–P3:

- **total** = H[mean `p_a`];
- **aleatoric** = mean H[`p_a`];
- **epistemic** = total − aleatoric. This is the BALD / mutual-information term.

`conf_ens = 1 − total`. All three components are stored. "Uncertainty" is never
reported without saying which of the three it is.

Adding P2/P3 changes the grain of `items.parquet` to one row per
`(item_id, condition, prompt_variant)`. **RQ1–RQ4 must therefore filter on
`prompt_variant == "P1"`, and on `condition == "clean"` where they apply.** This is
CONTRIBUTING.md invariant 14.

The risk–coverage machinery also sweeps raw thresholds on each component, with ECE
on the retained set. The preregistered prediction was that the epistemic threshold
beats the total one.

Where: `src/signals.py`, `src/metrics.py`, `analysis/rq5.py --task threshold_sweep`.

### D21 ⚑ — Two signals exist for `clean` only
**Decision.** `verbose` has no sampled draws and no P2/P3 calls, so it has no
`conf_sc` and no `conf_ens` or its components. Three consequences:

- RQ3b's paired comparison skips those two signals.
- Transfer test 1 (clean → verbose) and the verbose-shift check use Tier A *minus*
  those features, for every model. Nothing is imputed.
- The verbose-shift check uses the Bayesian model's **real fitted** question
  intercepts (`predict_in_sample()`), not the marginalized held-out ones.

**Why the fitted intercepts.** `verbose` covers the same 80 questions, so they are not
unseen. Marginalizing would also add spread to any input, which would make "epistemic
uncertainty rises under shift" true by construction.

Where: `src/features.py`, `src/bayesian.py::predict_in_sample`, `analysis/rq4.py`,
`analysis/rq5.py`.

---

## Meta-models (RQ4, RQ5)

### D8 ⚑ — Cross-validation with 80 groups
**Decision.**

- Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)`, grouped by
  `question_id` and repeated over 10 seeds.
- The headline uncertainty is the across-repeat spread.
- There is no hyperparameter tuning:
  - `LogisticRegression(C=1.0)` on standardized features;
  - `HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)`.
- The permutation null shuffles `correct` **within each `question_id`**.

**Why.**

- **Grouping.** Items from one question share a prompt and often the same answers, so
  ungrouped CV leaks.
- **Not `GroupKFold`.** sklearn's `GroupKFold` has no `shuffle`, so repeating it over
  seeds silently gives ten identical splits. A test asserts that two seeds give
  different folds.
- **No tuning.** With only 80 groups, tuning is noise plus a researcher degree of
  freedom.
- **Within-question null.** A global shuffle also destroys the question-level
  difficulty structure. That would understate how high a model with no item-level
  information can score.

Where: `src/predictor.py`, `tests/test_predictor.py`.

### D15 ⚑ — H4's confidence interval comes from a cluster bootstrap
**Decision.**

1. Average the out-of-fold scores over the 10 repeats, giving one score per item.
2. Fit `correct ~ oof_score * d_human`.
3. Cluster-bootstrap over `question_id` (B = 2,000) for the interaction's CI.

No default standard error is reported for it.

**Why.** Default standard errors assume independent rows. In-sample scores would bias
the interaction before the CI is even computed.

Where: `analysis/rq4.py --task h4`.

### D22 ⚑ — The Bayesian hierarchical logistic regression
**Decision.** A third RQ4 model, fitted with NumPyro NUTS:

```
correct ~ Bernoulli(σ(α + α_q[question] + Xβ))
α_q ~ Normal(0, σ_q),  σ_q ~ HalfNormal(1),  β ~ Normal(0, 1)
```

- **Sampling setup:** `α_q` uses the non-centered form, `σ_q · α_q_raw`. There are 2
  chains, with 500 warmup draws and 2,000 samples each.
- **Features:** standardized on each training fold.
- **Cross-validation:** the same protocol as D8.
- **Held-out questions:** their intercept is marginalized over the population prior,
  `α_q_new ~ Normal(0, σ_q)`, never set to a fitted value.
- **Required diagnostics:** R-hat at full precision, ESS and the divergence count.
- **Comparison table with LogReg:** AUROC, ECE and Brier for both models; NLL and 90%
  interval coverage for the Bayesian model only.

**Why.**

- **Partial pooling** is the standard tool for few groups, so the model fits the
  spirit of invariant 11. It is not an exception to it.
- **Non-centered form:** the centered form hit Neal's funnel on this data (R-hat 1.07,
  ESS 21). The non-centered form gave R-hat 1.01 and ESS 791.
- **Standardizing per fold:** `β ~ Normal(0, 1)` is only a comparable prior across
  features on the same scale. Without it, narrow-range signals such as `conf_lp` were
  effectively switched off.
- **Marginalized held-out intercepts:** using a held-out question's fitted value would
  leak, the same way ungrouped CV does.
- **R-hat at full precision:** `az.summary` rounds to two decimals by default, which
  hid values just above 1.01.

Where: `src/bayesian.py`, `tests/test_bayesian.py`.

### D23 ⚑ — RQ5: prompt distillation, and checking the decomposition against humans
**Decision.**

- **Teacher and student.** The teacher is the three-prompt ensemble's predictive
  distribution. The student is the single-call, P1-only Bayesian meta-model. No LLM is
  fine-tuned.
- **Headline.** The headline result is how much of the ensemble's AUROC, ECE and
  entropy quality survives at one call.
- **Check against humans.** Separately, the judge-level aleatoric term should track
  `d_human` (where humans disagree), and the epistemic term should not. This runs on
  clean/P1 only.

**Why.** The closest prior work (Auto-Prompt Ensemble, 2025; Bayesian prompt
ensembles for MLLM judges, ICCV 2025) neither decomposes the entropy nor checks it
against real human votes.

Where: `analysis/rq5.py`.

### RQ4 framing (no separate number)
**The question.** RQ4 asks which cheap features carry the signal about judge error:
the judge's own uncertainty, or surface properties of the inputs that the judge
ignores. The tiers are strictly additive:

- **A:** the confidence signals and the entropy components;
- **B:** A plus lengths, turn, category, output length and `flipped`;
- **C:** B plus the verdict margin and the CoT logprob statistics.

**Every outcome is a reportable result.**

| Result | Reading |
|---|---|
| A ≈ the best single signal | the signals are redundant |
| B ≫ A | error is predictable from features the judge ignores |
| C ≫ B | the signal lives in the token distribution |
| nothing beats the permutation null | error isn't predictable from cheap features |

**The transfer tests:**

- **Train on clean, test on verbose.** This asks whether a safety net trained on
  well-behaved data still works under attack.
- **`LeaveOneGroupOut` over the 8 categories.** This asks whether the model learns
  something general or just which categories are hard.

---

## The other two judges (RQ6, RQ7)

### D27 ⚑ — kev-8b as an open stand-in for a decision-only judge
**Decision.** kev-8b (`jaredpalmer/kev-8b`, Apache-2.0) is RQ6's judge. It is
Qwen3-8B-Base with a LoRA adapter and a pointer head, served through its own
`kev.serve` HTTP API.

**Inputs and exclusions.**

- **Signals:** `probabilities[choice]` and an order-corrected bidirectional entropy.
- **Excluded:** kev's `confidence` field. It is exactly `2·p[choice] − 1` for two
  options (`kev/api.py`), so it carries no extra information.
- **Excluded:** self-consistency. kev makes one deterministic pass per request, so
  there is nothing to sample.

**Regimes and ceiling.**

- **Reporting regimes:** results are split by kev's training coverage: `input_tokens`
  ≤ 1,024, and above that.
- **Serving ceiling:** 8,160 tokens, measured directly. Near 8,192 requests fail with
  out-of-memory errors that do not happen every time. 52 verbose items exceed the
  ceiling, and their clean counterparts are dropped from the paired verbosity test.

**Server launch.** The server is launched with `KEV_DTYPE=bf16 KEV_MERGE=0
KEV_ATTN=sdpa`. `KEV_MERGE=0` matters: merging forces fp32, which runs out of memory
on an L4.

**Why kev-8b.** Two other open candidates were rejected on prompt-length grounds:

- Bespoke-Nimble-9B rejects prompts over 2,048 tokens.
- circuit-8b covers too little of the data within its training range.

`SemIf` has no canonical source. kev was never trained to judge answer pairs, so RQ6
is an out-of-domain test, and every RQ6 result says so.

**Limit on the claims.** What is known about Jev itself comes only from Li et al.
(2026). RQ6 makes no claim about Jev's training or design; see D30.

Where: `src/judge_kev.py`, `src/kev_signals.py`, `analysis/rq6.py`,
`configs/run_kev.yaml`.

### D28 ⚑ — auto-j-13b, a judge trained for pairwise judging
**Decision.** auto-j-13b (`GAIR/autoj-13b-GPTQ-4bits`, Li et al., ICLR 2024) is RQ7's
judge. It runs under vLLM 0.28.0 on an L4, with the Marlin GPTQ kernel. It uses the
authors' pairwise template verbatim, and their verdict extractor ported unchanged
(`extract_pariwise_result`).

**Signals.**

- `conf_sc_autoj`: self-consistency.
- `conf_sc_bpe_autoj`: the order-swap entropy, built from the self-consistency
  proportion in each order.
- `conf_sc_bpe_autoj_greedy`: the same signal built from the greedy calls only. The
  verbosity test uses this one, so that it compares like with like.
- `conf_lp` was not attempted, because free text has no fixed verdict-token position.

**Population.**

- **Both turns, always reported separately.** For turn 2, the shared prompt holds both
  user turns, and each response field holds that side's two answers.
- **Ties:** a "Tie" verdict (2.35% of calls) counts as no verdict. It is tracked apart
  from parse failures in `pred_label`.

**Exclusions.** Both are logged, never dropped silently:

- **Input:** prompts over 7,168 tokens (8,192 context minus the 1,024-token output
  budget) are skipped: 362 calls, almost all verbose turn 2. That is 4.75% of the 7,616
  greedy calls and 2.4% of all 15,232.
- **Output:** truncation at `max_tokens = 1024` leaves no verdict on 39.8% of verbose
  turn-2 calls. Every such failure is a truncated completion. This is reported as a
  finding in its own right. `max_tokens` was deliberately not raised.

**Why.** If a model trained for exactly this task shows the same failures, the
finding holds across training objectives.

Where: `src/judge_autoj.py`, `src/autoj_signals.py`, `analysis/rq7.py`,
`configs/run_autoj.yaml`.

---

## Presentation

### D29 — A static site, a post-hoc auto-accept analysis, and a public repo
**Auto-accept analysis.** `analysis/auto_accept.py` asks what share of wrong verdicts
would get through if every verdict above a confidence threshold were auto-accepted.

- It is descriptive and **post-hoc**, not preregistered, and labelled so.
- It uses calibration forms only (D7).
- The meta-models enter through their out-of-fold P(correct).

**The site.** It is plain HTML, CSS and JavaScript (`site/`), and it **computes no
statistic**. Its data is built by `analysis/demo_bundle.py` → `analysis/site_data.py`.

- Example items are selected blind to the judges' verdicts, with a fixed seed.
- The judge palette was validated for colour-vision separation, and `src/plots.py`
  uses the same palette.

**Licensing and publishing.**

- Code is under MIT; text and figures are under CC BY 4.0.
- The site is published on GitHub Pages and as a Hugging Face Space.
- The judge checkpoints are published as the Hugging Face dataset
  `shubhamsengupta/judge-calibration-runs`.

### D30 — Literature check before the write-up
No result changed. Three things did:

1. **Attribution of `conf_bpe`.** It is SCOPE's Bidirectional Preference Entropy
   (Badshah et al., ICML 2026), and REPORT cites it. One construction detail differs:
   here `p_a` is read after the judge's reasoning, not from a bare answer token. So the
   AUROCs are not interchangeable (SCOPE 0.685, RQ2 0.794).
2. **Jev.** It has been evaluated directly by Li et al. (2026, arXiv:2609.26550), and
   that paper, not RQ6, is the evidence about Jev. kev-8b flips 21–25% of verdicts on
   MT-Bench answer swaps. Li et al. report 3.7–11.1% for Jev, on different benchmarks.
   kev is therefore not a close proxy on position bias.
3. **Positioning of the auto-accept section.** It is placed against SCOPE's conformal
   guarantee and Li et al.'s padding result.
