# DECISIONS.md — resolutions from the Week-0 design review

Twenty-five decisions, D4–D28, resolving the review findings and (D18 onward) integrating the professor's 31 Aug 2026 feedback. D27 (22 Sep 2026) and D28 (23 Sep 2026) are later, owner-initiated additions — the industry-counterexample stress test and the purpose-built-judge generalization test — not part of the professor-feedback integration. Each is binding; copy the ones marked ⚑ into `PREREGISTRATION.md` before Gate 1. **D27 and D28 postdate `PREREGISTRATION.md`'s freeze (1 Sep 2026) and are not in it, same as D26.**

Verdict on the review: **eight of ten findings were correct.** #1, #2 and #3 each would have cost a full re-run. #2 turned out to save GPU time rather than cost it. Two findings needed a stronger fix than proposed (#4, #5) and one revealed a second bug underneath it (#3). Nothing was wrong.

---

## D4 ⚑ — Tier C logprob capture *(review #1)*

**Correct, and the most expensive miss.** `verdict_token_logprob` is a scalar; Tier C's CoT-wide statistics don't exist anywhere in the schema. Discovering this at task 5.2 means re-running the harness.

**Decision — capture all three, at generation time:**

1. **Aggregate columns**, computed in `judge.py` as each completion returns, never reconstructed later:
   `cot_logprob_mean`, `cot_logprob_min`, `cot_logprob_std`, `cot_logprob_p10`, `cot_entropy_mean`, `n_cot_tokens`.
2. **`SamplingParams(logprobs=20)`**, not 5. Entropy over top-5 is severely truncated.
3. **A 10% raw sidecar.** For every call where `hash(item_id) % 10 == 0`, dump the full per-token top-20 logprobs to `runs/logprobs_sample/*.jsonl.gz`. Insurance against wanting a statistic you didn't anticipate; full capture for every call is not worth the storage.
   **Sizing, done properly:** ~350 CoT tokens × 20 pairs × ~20 bytes ≈ **140KB per call**; 10% of 18,000 calls = 1,800 calls ≈ **250MB raw, ~70MB gzipped**. **Write it gzipped** — `jsonl.gz`, not `jsonl`. (An earlier draft of this line said "well under 200MB" without doing the multiplication; raw is 250MB.)
   **Updated 31 Aug 2026 (D19):** the total generation count dropped from 18,000 to 12,000. 10% is now 1,200 calls ≈ **170MB raw, ~48MB gzipped**. The policy (10% sample, gzipped) is unchanged — only the illustrative arithmetic moved with the new schedule.

**Two caveats to preregister, not discover:**

- **Top-k entropy is a biased estimator.** It systematically understates true entropy because the tail is truncated. `cot_entropy_mean` is "mean top-20 truncated entropy" and must be labelled that way in every table. Do not call it "predictive entropy."
- **CoT logprob statistics are conditioned on the sampled chain.** Each of the k samples has a different CoT and therefore different statistics. **Item-level aggregation rule:** take the value from the canonical call (`sample_idx=0`) as the primary feature, and the mean across the k sampled calls as a secondary feature. Store both; let the Tier C ablation tell you which carries signal.
- **⚠️ The two are not in the same unit.** The secondary feature is computed from the T=0.7 sampled calls, so the same temperature-scaling effect D6 found for `p_a` applies to it. Standardization means the model doesn't care, but *interpretation* does — a coefficient plot that treats them as one quantity is wrong.
  **The fix is naming, not a comment:** the columns are `cot_logprob_mean_greedy` and `cot_logprob_mean_sampled_t07` (and likewise for `min`, `std`, `p10`, `entropy`). Carrying the temperature in the name makes it structurally impossible for anyone — including you in October — to average them or read them as the same feature.

**Amended, 4 Sep 2026 (task 1.4 implementation review):** point 3's sidecar is raised from a **10% sample to 100% coverage** — every call's full per-token logprobs are now saved, not just calls where `hash(item_id) % 10 == 0`. Two reasons, found while actually writing `judge.py`:

1. **It removes a split this decision was otherwise forcing on `judge.py`.** With only a 10% sample, the raw per-token logprobs needed for `verdict_token_logprob`/`p_a` would vanish for the other 90% of rows before `src/parse.py` (task 1.5) ever runs — so `judge.py` itself had to compute those two fields at generation time, alongside the CoT aggregates. That's a real violation of invariant 7's "parsing lives only in `parse.py`," made out of necessity, not choice. At 100% coverage, `parse.py` can compute `verdict`, `verbalized_conf`, `verdict_token_logprob`, and `p_a` — every logprob-derived field — from saved data, whenever it runs, restoring the intended split cleanly: `judge.py` runs the model and saves raw materials only; `parse.py` is the one place that turns raw materials into signals.
2. **The "insurance against an unanticipated statistic" this decision already argued for is strictly stronger at 100%.** Any future statistic can be computed on the *full* dataset instead of a 10% subsample carrying materially higher variance for anything sliced by category or question.

**Cost, scaled from this decision's own arithmetic above:** roughly **10x** the raw/gzipped sizes already quoted (170MB raw / 48MB gzipped at 10%, post-D19's 12,000-call schedule) — comfortably under 2GB gzipped even generously scaled. The exact absolute total is provisional pending task 4.1b's GPU-budget re-extrapolation, same as every other total-generation-count figure in this file. Judged acceptable against Colab Pro's storage headroom.

Sidecar directory renamed `runs/logprobs_sample/` → `runs/logprobs/`, since "sample" no longer describes 100% coverage. `CLAUDE.md`'s schema and layout sections updated to match.

---

## D5 ⚑ — `order` is the axis; `swap` is not a condition *(review #2)*

**Correct, and this was my error.** In a pairwise judge prompt, "swap the presentation order" and "swap which response is labelled Assistant A" are the same operation. Having both a `swap` condition and an `order` field was double-naming one manipulation, and it would have produced either duplicated collection or an incoherent RQ3a.

**Decision:**

- **Conditions** are *content* manipulations: `clean`, `verbose`, `attribution`, `vacuum` (sanity only).
- **`order` ∈ {AB, BA}** is orthogonal and is collected for **every** condition.
- **`swap` is deleted from the condition vocabulary.** Remove it from `configs/conditions.yaml` and from every DoD.
- `flipped` and `conf_bpe` are computed *within* a condition, across its two orders.
- **RQ3a (position bias) is the flip rate within the `clean` condition.** It is not a separate run.
- `test_perturb.py`'s `swap∘swap = id` property moves to testing the **order renderer** in `prompts.py`, not `perturb.py`.

**This changes the compute budget, downward.** Combined with D6:

| | per (item, condition) |
|---|---|
| Canonical greedy, both orders | 2 calls |
| Self-consistency draws, AB order only | 4 calls |
| **Total** | **6 calls** |

`conf_sc` is defined on the canonical AB presentation, so repeating the sampled draws in BA order buys nothing. At N=1000 × 3 real conditions × 6 = **18,000 generations**, against the ~30,000 the old schema implied. That is a 40% saving that falls straight out of fixing the naming.

*(Rejected extension: swapping the **labels** while holding content order fixed would separate label bias from position bias. Genuinely interesting, out of scope. Note it on the extensions slide.)*

---

## D6 ⚑ — Temperature, and the bug underneath it *(review #3)*

**Correct.** A single `temperature: 0.0` would make all k draws identical and `conf_sc` a constant 1.0 — dead until the W5 Tier A ablation, which is the worst possible place to find it.

**Decision — two keys, never one:**

```yaml
temperature_canonical: 0.0    # sample_idx = 0, greedy, reproducible
temperature_sc:        0.7    # sample_idx = 1..k_sc
k_sc:                  4
seed:                  1234   # per-call seed = seed + sample_idx
```

`conf_sc` = **fraction of the k_sc sampled verdicts that match the canonical greedy verdict.** This is more directly interpretable than raw pairwise agreement among samples, and it keeps the confidence signal aligned with the verdict the `correct` label is computed from.

### The second bug this exposes

vLLM applies temperature scaling to the logits, and in most versions the returned logprobs reflect the **scaled** distribution. So `p_a` measured at T=0.7 is not `p_a` measured at T=0 — the same signal would mean different things across `sample_idx`.

**Decision:** `conf_lp` and `p_a` are extracted from the **canonical greedy call only** (`sample_idx = 0`). Never from a sampled call. Never averaged across temperatures.

**Verify empirically in task 1.6:** run one item at T=0 and T=0.7 and compare the reported `p_a`. If they differ, you have confirmed the behaviour and the decision above is load-bearing. Record the finding in `REPORT.md` — it is a legitimate methods note that most papers skip.

---

## D7 ⚑ — What "the judge's verdict" means at item level *(not in the review — found while resolving D5)*

Once `order` is collected for every condition, `judge_verdict` at item level is ambiguous: the AB verdict, the BA verdict, or the order-averaged one? Left unspecified, this bites in W2 and silently changes every accuracy number.

**Decision:**

- **Primary: `judge_verdict` = the canonical greedy verdict in AB order.** This is what a real deployment running one pass would get, and RQ1/RQ2 should describe the deployed case, not an idealised one.
- **Secondary: `verdict_bidir` = argmax of the mean `p_a` across both orders.** Report RQ1/RQ2 headline numbers for both. The gap between them is itself a finding — it quantifies what debiasing-by-averaging buys.
- `correct` is defined against the primary. `correct_bidir` against the secondary.

**Amended 24 Sep 2026 — a confidence is scored only against the verdict it belongs to, and entropy signals are scored for calibration through a probability.** Two problems found while reviewing the figures:

- *Scoring against the wrong verdict.* RQ1 scored every signal against both verdicts. But `conf_verb`, `conf_lp` and `conf_sc` are confidences in the AB verdict; on the ~28% of items whose orders flip, `verdict_bidir` is a different answer they never described. Against `verdict_bidir`, calibration now uses confidences in *that* verdict: `conf_verb_bidir` (each order's stated confidence counted toward `verdict_bidir` — c if that order agreed, 1 − c if not — averaged) and `conf_lp_bidir` = max(p, 1 − p), with p the order-averaged P(model_a wins). `conf_sc` has no `verdict_bidir` form: its sampled draws exist in AB order only.
- *Entropy is not a probability.* `conf_bpe`, `conf_ens`, `conf_kev_bpe` and `conf_sc_bpe_autoj` are 1 − H(p). That ranks items correctly, so AUROC/AURC use it unchanged, but a 50/50 split scores 0.307, not 0.5, so ECE and the overconfidence gap on it partly measure that difference in scale. Calibration metrics now use the probability form, `signals.py::prob_on_verdict(p, verdict)` — p on the verdict's side: `conf_bpe_prob`, `conf_ens_prob`, `conf_kev_bpe_prob`, `conf_sc_bpe_autoj_prob`, `conf_sc_bpe_autoj_greedy_prob` (all against `judge_verdict`). Against `verdict_bidir`, `prob_on_verdict` gives max(p, 1 − p), which is why `conf_lp_bidir` is also `conf_bpe`'s form for that verdict.

What changed in the results is listed in `REPORT.md`'s methods note "Calibration forms and convergence counts". The two readings that changed: `conf_bpe` is overconfident (not "indistinguishable from perfect calibration"), and the verbosity attack does not break kev-8b's calibration (RQ6 had reported the same breakage as the primary judge).

---

## D8 ⚑ — CV design at 80 groups *(review #4)*

**Correct, and the proposed fix needs two additions.** MT-Bench has 80 questions; effective N for fold-to-fold variance is ~80, not ~1000.

**Decision:**

1. **`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)`, repeated over 10 seeds.**
   ⚠️ **sklearn's `GroupKFold` has no `shuffle` parameter** — it is deterministic, so "repeat across seeds" silently produces ten identical splits. Use `StratifiedGroupKFold`, which supports `shuffle`. This is a real trap and the test in 5.3 must assert that two different seeds produce different fold assignments.
2. **Report the across-repeat spread as the headline uncertainty**, not the within-split CI. The honest number is "AUROC 0.64, 10×5-fold spread [0.58, 0.69]".
3. **Drop hyperparameter tuning entirely.** Fix `C=1.0` on standardized features and preregister it. At 80 groups the nested inner CV subdivides 64 training groups into folds of 16, which is noise, and tuning adds a researcher degree of freedom you cannot afford to defend. For `HistGradientBoostingClassifier`, fix `max_depth=3, max_iter=200, learning_rate=0.05` and preregister those too. This removes the nested CV from task 5.3.

**Amended 24 Sep 2026 (project audit) — the permutation null shuffles within questions, not globally.** Task 5.4 built invariant 12's null as a global shuffle of `correct`. That breaks the within-question clustering of the labels too — some questions are simply harder — so it understates how high a model with no item-level information can score. `src/predictor.py::permutation_null()` now permutes `correct` within each `question_id`, keeping each question's error rate intact; that is the stricter null, since features that only track question difficulty still score above 0.5 under it. Measured on the real populations before switching, the verdict is identical under both: every observed AUROC is 7–16 null standard deviations above either null, at the 100th percentile. New numbers: task 5.4's full-population null (Tier A/logreg) 0.507 ± 0.023 (was 0.493 ± 0.022); the 5.5 ablation nulls 0.53–0.57 (were 0.50–0.51), observed 0.80–0.82.

---

## D9 ⚑ — Human disagreement as a continuous covariate *(review #5)*

**Correct, and the fix should be stronger than a bucket-count check.** Realistic arithmetic: ~3,355 votes, MT-Bench's own human-agreement analysis used ~700 first-turn votes, so items with ≥2 votes are plausibly ~350, of which contested might be ~70. Seventy items split across GroupKFold will not produce a stable AUROC.

**Decision — make the continuous version primary:**

- **Primary:** treat `d = |frac_prefer_a − 0.5|` as a **continuous** covariate. Regress judge correctness on `d`, and test the **interaction** between `d` and the predictor's output. This tests H4 directly, uses every item with ≥2 votes, and never needs a bucket to be large enough.
- **Secondary:** the agreed/contested bucketed comparison, reported only if `n_contested ≥ 100`.
- **Tertiary fallback** if even the continuous version is thin: compare mean predicted P(wrong) across consensus strata using a cluster-bootstrap on group means — no CV required, far lower N demand.

**Gate 0 must print, separately:** N with ≥2 votes, N with ≥3 votes, N unanimous, N contested. The D2 fallback threshold is checked **per subset**, not on the total. Task 0.8 is amended accordingly.

---

## D10 — Pilot scope and a provisional GPU number *(review #6, #7)*

**Both correct.** Task 1.6 said "4 conditions" when only `clean` exists at pilot time, and a clean-only pilot cannot price the verbose condition.

**Decision:**

- **Pilot = 20 items × `clean` × 6 calls = 120 generations**, using the real `prompts.py`. (6 calls per item-condition per D5/D6: 2 canonical greedy across orders + 4 sampled draws in AB. Not 2×5.)
- **Plus a 10-item verbose smoke test** with deliberately throwaway padding logic, for one purpose only: measuring the **prompt-token multiplier**. Label it as throwaway in the code.
- **The Gate 1 GPU number is provisional.** Re-extrapolate after `perturb.py` lands in W4. Task 1.6's DoD is amended to say so.

**Calibration for expectations:** verbose padding lengthens the *prompt*, not the CoT output. Prefill is cheap relative to decode, so expect roughly **+20–40% wall-clock** on that condition, not 2–3×. Budget accordingly rather than over-reserving.

---

## D11 — Pin vLLM *(review #8)*

**Correct, and confirmed as an active hazard.** vLLM deprecated `GuidedDecodingParams` in favour of `StructuredOutputsParams` ([TRL #4617](https://github.com/huggingface/trl/issues/4617)), and there is a documented history of request-level vs engine-level backend mismatch errors ([vLLM #16738](https://github.com/vllm-project/vllm/issues/16738)) and broken doc examples across 0.8.x ([vLLM #15236](https://github.com/vllm-project/vllm/issues/15236)). Course material will not match whatever pip resolves.

**Decision:**

- **Pin `vllm==<exact>` in `pyproject.toml` before task 1.4.** Pick the version on the day, run the guided-decoding smoke test, and do not move it again for the rest of the project.
- Record the resolved version of `vllm`, `torch`, `transformers` in the **run manifest**, alongside `git_sha`.
- **Colab's preinstalled `torch` will fight vLLM's pinned `torch`.** Install into a fresh venv (or `uv venv`) rather than the system environment, and expect one runtime restart. Budget 30 minutes for this in W1; it is the single most likely way the peak week overruns.
- Task 1.2's DoD is amended: the scratch script must exercise **guided decoding + logprobs together**, not logprobs alone, because that is the combination that breaks.

**Resolved, 3 Sep 2026 (task 1.2's smoke test):** pinned `vllm==0.28.0`, installed via `uv venv` into a fresh venv on Colab (T4). This version uses the renamed `StructuredOutputsParams`, not `GuidedDecodingParams`. All three cases — logprobs alone, structured-decoding alone, and **both together** — ran cleanly with no error; the historical incompatibility this decision was written to guard against does not reproduce on this version. `pyproject.toml`'s `colab` extra now pins `vllm==0.28.0` accordingly; per this decision's own rule, that pin does not move again for the rest of the project.

---

## D12 — Colab budget as a method, not a number *(review #9)*

**Correct that it was missing.** Compute units, not calendar time, is the scarce resource, and L4/A100 availability is not guaranteed.

I am not going to quote unit-per-hour rates — they change, and a fabricated number here is worse than none.

**Decision — compute the budget from measurement, in W1:**

1. At Gate 1 you have measured seconds-per-generation on the actual accelerator. Total generations = **N × 3 conditions × 6 calls**, plus pilot, plus the demo.
2. **Multiply by 3.** Reruns after a parser bug, a prompt fix, or a bad seed are not a contingency, they are the base case.
3. Read the live burn rate from Colab's resource panel during the pilot and multiply out. Compare against your remaining monthly units *before* committing to W2.
4. **If the budget is tight, cut N before cutting conditions.** N=600 with all conditions is a better study than N=1000 with two. Preregister the N you can afford at Gate 1 rather than discovering it in W4.
5. Prefer **L4**. Reach for A100 only if measured L4 throughput fails the budget — the unit cost is several times higher and the quality of the study does not change.

---

## D13 — Naming discipline on the attribution condition *(review #10)*

**Correct.** The condition tests susceptibility to an authorship *label*; the response was never written by Qwen2.5-7B.

**Decision:**

- The condition is named **`attribution`** everywhere. The token `self_preference` / `self_enhancement` appears nowhere in the codebase.
- In `REPORT.md` the finding is **"susceptibility to authorship framing"**. Never "self-enhancement bias".
- **State this as a strength, not only a caveat.** Because the label is decoupled from actual authorship, this design isolates the effect of the attribution itself from stylistic self-similarity — a confound that true self-preference studies cannot separate, and the reason Zheng et al. could not establish self-enhancement bias in a controlled setting. You are measuring a cleaner quantity than the thing you gave up.

---

## D14 ⚑ — `ece()` on discrete confidence, and what `conf_sc` costs *(second review)*

**Correct, and binning is the wrong tool here rather than a fiddly one.** With `k_sc=4`, `conf_sc ∈ {0, 0.25, 0.5, 0.75, 1.0}` — five values, with most mass piled at 1.0. Quantile edges will land inside ties; `pd.qcut` raises on duplicate edges, and `duplicates="drop"` silently returns fewer bins than requested. Invariant #4 mandates quantile bins without saying what happens when the signal is discrete.

**Decision — `ece()` picks its own strategy and always reports what it did:**

```
if n_unique(conf) <= n_bins:   bin by unique value   → EXACT, no approximation error
else:                          quantile bins, duplicates dropped
return (ece, n_effective_bins)
```

Binning is an approximation device for *continuous* confidence. When confidence is already discrete with few levels, grouping by unique value is not a fallback — it is the **exact** ECE, and it is strictly better than any binning. `ece()` returns `n_effective_bins` alongside the scalar; every table and every reliability diagram prints it (invariant #4 already requires this, and this is where it bites).

**The limitation this exposes, which belongs in `PREREGISTRATION.md` rather than in W5's surprise pile:**

`conf_sc` has **5 levels and costs 4 extra generations per item**. `conf_lp` and `conf_bpe` are **continuous and cost nothing extra**. So if self-consistency underperforms in the Tier A ablation, part of that gap is **resolution, not information** — a genuine confound, and one that must be stated rather than read as "self-consistency is a weak signal."

That confound is itself a reportable result: *at equal or lower cost, the continuous signals carry finer-grained information than self-consistency at k=4.* For anyone deploying a cheap abstention layer, that is directly actionable. Do **not** fix it by raising `k_sc` — k_sc=9 would take 6 calls/item to 11 and nearly double the run, which D12's budget cannot absorb. Accept 5 levels, preregister the limitation, report it as a finding.

Note also that AUROC on a 5-level score is coarse. `sklearn.roc_auc_score` handles ties correctly via the trapezoid rule, so it is not wrong — just low-resolution. Same caveat, same paragraph.

---

## D15 ⚑ — H4's interaction CI comes from the cluster bootstrap *(second review)*

**Correct, and this is exactly the failure invariant #2 exists to prevent.** D9 and task 5.6 said "the interaction coefficient with a CI" without naming the method — and every default (`statsmodels` OLS/GLM standard errors, `sklearn` anything) assumes i.i.d. rows. Rows are clustered inside 80 questions. H4 was the one RQ4 output phrased differently from the study's bootstrap-everything convention, and that is precisely why it would have slipped through.

**Decision — the H4 interaction CI is a cluster bootstrap over `question_id`, B=2000.** No default standard error is ever reported for it.

**Order of operations matters** — get this wrong and the CI is optimistic for a second, independent reason:

1. Generate **out-of-fold** predictions using the fixed 10×5 repeated `StratifiedGroupKFold` (D8), then average across the 10 repeats to get **one OOF score per item**. In-sample predictions would make the interaction optimistically biased before the CI method even enters.
2. Fit the interaction model — `correct ~ oof_score * d_human` — on that item-level table.
3. **Cluster-bootstrap:** resample `question_id` with replacement, take all rows for each sampled question, refit, collect the interaction coefficient. Percentile CI over the 2000 refits.

The `correct ~ oof_score * d_human` interaction term is H4's whole test. Its CI must survive the same scrutiny as every other number in the study.

---

## D16 ⚑ — "Human-agreed" needs a vote floor *(third review)*

**Correct, and the floor already existed once.** `human_unanimous` (task 0.8) is `True` for any item with 100% vote agreement — including an item with exactly **1 vote**, which is trivially "unanimous" because there is nothing to disagree with. Task 5.5's tier ablation restricts to "human-agreed items" with no minimum vote count, so the least-informative items in the dataset (single-vote items) would be treated as the most confident ground truth, distorting the Gate 5 headline AUROC in an unknown direction. The original pre-review `PLAN.md` had a floor for exactly this reason ("unanimous, or ≥75% majority with **n≥3**") — it was dropped, not deliberately removed, when D9 rewrote the section around the continuous covariate.

**Decision:** define, for task 5.5 only, **`human_agreed` = `human_unanimous AND n_human_votes ≥ 2`.** This reuses the ≥2-vote floor H4 (D9, task 5.6) already established for `d_human`, rather than inventing a second threshold to defend. `src/data.py::build_items()` (task 0.8) emits `human_agreed` as a column alongside `human_unanimous`, so it is computed once, at the source, not re-derived ad hoc in the Tier ablation code.

This does not touch H4 itself — H4's continuous test already uses `d_human` over every item with ≥2 votes and needs no bucket. It only closes the gap in 5.5's baseline population.

---

## D17 — Local/Colab environment split and the GitHub bridge *(workflow, not a review finding)*

Not a bug found in the design — a workflow question the owner raised before starting: how does work move between a local Windows/conda machine and Colab's GPU, given `judge.py` needs vLLM and nothing else in the harness does?

**Decision:**

- **The local machine never installs vLLM.** vLLM's Windows support is unreliable, and nothing that runs locally — `data.py`, `metrics.py`, `parse.py`, `signals.py`, `boot.py`, `features.py`, `predictor.py`, the whole test suite, the Gradio demo — needs it. `judge.py` keeps `import vllm` inside the function that calls it, not at module top, so the rest of the package stays importable without vLLM installed.
- **`pyproject.toml` splits dependencies.** A base install (`datasets`, `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `matplotlib`, `pyyaml`, `pytest`) with no `vllm`, plus an optional `colab` extra (`pip install -e ".[colab]"`) that adds `vllm`. Local `pip install -e .` never touches vLLM.
- **A dedicated local conda env, `judge-calib` (`python=3.11`)** — separate from the owner's general AI/ML conda env. This project pins exact (`==`) versions for reproducibility (task 0.1); sharing an env with other projects is how those pins get silently violated later by an unrelated `pip install`.
- **Colab still gets its own fresh `venv`, per D11** — unrelated to the local conda env, since Colab's preinstalled `torch` conflicts with vLLM's pinned `torch`. Colab installs `pip install -e ".[colab]"` inside that fresh venv, never the system Python.
- **A GitHub remote carries code between the two.** Commit and push locally; `git clone`/`git pull` in Colab at the start of each session. This is the only way Colab sees the latest `src/` changes.
- **`runs/` and `results/` stay gitignored — git is never the bridge for generated data.** Move `calls.parquet` and checkpoint files back from Colab via a Drive-mounted folder or direct download, not by trying to commit them.

**Amended 25 Sep 2026 — `results/*.csv` and `results/figures/*.png` are tracked.** The rule above was aimed at data too heavy or raw for git: `runs/` (checkpoints and per-token logprobs, ~1.6 GB) and the parquet tables (which carry raw model output). The per-RQ CSVs and figures are neither: they are small (~0.7 MB and ~3 MB), aggregate, and they are the evidence `REPORT.md` cites — untracked, a reader of the repo couldn't open any of it. They are now committed alongside the code and the report that describe them; a rerun that changes a number shows up as a diff in the same commit as the `REPORT.md` edit. `runs/` and every `*.parquet` stay ignored, and nothing moves from Colab through git — Colab only produces `runs/`.

---

## D18 ⚑ — `attribution` condition is dropped entirely *(professor feedback, 31 Aug 2026)*

**Decision:** `attribution` is removed from the condition vocabulary. Conditions become **`clean`, `verbose`** (`vacuum` stays, untouched, as the separate small sanity check). This is not the drop-order contingency it used to be (`PLAN.md` sec 4 used to list it as item 4) — it is now unconditional, not something to cut only if behind schedule.

**Consequences:**
- `configs/run.yaml`'s `conditions` list drops `attribution`.
- `src/perturb.py` (task 4.1) builds only `verbose_pad()`, not `attribution()`.
- RQ3b (task 4.4) becomes verbosity-only.
- Transfer test 1 (task 5.7) becomes clean→verbose only.
- **D13 is superseded by this decision.** It resolved a naming problem for a condition that no longer exists. Left in place above, not deleted, for the audit trail — the same treatment every earlier decision that got overtaken has received.

---

## D19 ⚑ — Prompt ensemble: new axis, new schedule *(professor feedback, point 3)*

**Decision:** three frozen, hashed prompt variants — **`P1`** (the existing MT-Bench template, primary; RQ1–RQ4 use this alone), **`P2`** (correctness-first rubric), **`P3`** (helpfulness-first rubric). `prompt_variant` becomes a new axis in `calls.parquet`, orthogonal to `condition` and `order` — exactly the role `order` already plays after D5.

**New per-item schedule, superseding D5's and D12's numbers:**

| condition | variant | sampling | orders | calls |
|---|---|---|---|---|
| clean | P1 | greedy | AB, BA | 2 |
| clean | P1 | sampled ×4 | AB only | 4 |
| clean | P2, P3 | greedy | AB, BA | 4 |
| verbose | P1 | greedy | AB, BA | 2 |

**12 calls/item**, down from 18 under the old 3-condition × 6-call schedule (D5's arithmetic). At N=1000, that's **12,000 generations**, down from 18,000. `vacuum` (task 1.8) stays outside this table entirely — unchanged.

Note where this actually lands in the week plan: rows 1+2+3 (clean, all variants — 10 calls/item) belong with the **existing Week 2 clean run** (task 2.1), since P2/P3 are a property of judging the *clean* comparison three ways, not a perturbation. Row 4 (verbose/P1 — 2 calls/item) stays in **Week 4** alongside `verbose_pad()`. Collecting P2/P3 in W4 instead of W2 would be a mistake — they have nothing to do with the verbose perturbation.

D5 and D12's own *reasoning* (order is an axis, not a condition; budget the units, not the hours) stays correct — only the concrete numbers they computed are superseded here, because the schedule now has a third axis that didn't exist when D5/D12 were written.

---

## D20 ⚑ — `conf_ens`, judge-level entropy decomposition, and the schema break it causes *(professor feedback, points 3, 4, 5)*

**Decision — `conf_ens` and its decomposition.** Computed across the P1/P2/P3 ensemble's greedy `p_a` values for a `clean` item (uniform prior over the three variants):

- **Total** = H[mean(p_a across P1, P2, P3)]
- **Aleatoric** = mean(H[p_a] for each of P1, P2, P3)
- **Epistemic** = Total − Aleatoric (mutual information / BALD)

`conf_ens = 1 − Total`, matching `conf_bpe`'s existing `1 − entropy` convention (SCOPE, 2026) — but now all three quantities are stored, not just the scalar. **Never report "uncertainty" unqualified once this exists — always name which of the three.**

**⚠️ This changes `items.parquet`'s grain.** It was one row per `(item_id, condition)`; it becomes one row per **`(item_id, condition, prompt_variant)`**, since P2/P3 have their own verdicts and confidences. **Every RQ1–RQ4 analysis script must add an explicit `prompt_variant == "P1"` filter wherever it reads `items.parquet`**, or P2/P3 rows silently inflate every sample size and every RQ1–RQ4 headline number is wrong. This gets its own numbered invariant in `CLAUDE.md` (not just a mention) — it is the single easiest thing to get wrong in this whole update.

`conf_bpe` and `flipped`'s "within this condition" language extends to "within this `(condition, prompt_variant)` pair" — P2/P3 also get both orders at greedy (D19's schedule), so they technically support their own `conf_bpe` too, even though RQ1–RQ4 don't use it.

**`conf_ens` (and its three components) exist only for `clean` items** — `verbose` never collects P2/P3, so there is nothing to decompose there. This matters again in D21.

**Decision — threshold-sweep extension to `risk_coverage()` (point 5).** The existing `risk_coverage()`/`aurc()` machinery (built for RQ2, task 3.1) is extended to sweep **raw threshold values** of total/aleatoric/epistemic entropy separately, not just percentile-based coverage, and to report **ECE on the retained set** at every threshold, not just accuracy/κ — "reliability" is the calibration term in the Brier decomposition, so accuracy alone doesn't answer the question asked. **Preregistered prediction:** epistemic thresholding beats total thresholding, because epistemic is the reducible part; aleatoric reflects genuine task ambiguity that no amount of re-asking resolves.

---

## D21 ⚑ — Two signals are clean-only: the transfer test and the verbose-shift check both need a fix *(found while resolving D19/D20)*

Self-consistency sampling and the P2/P3 ensemble both only happen for `clean` under D19's schedule. Previously `verbose` had its own `conf_sc`; now it has neither `conf_sc` **nor** `conf_ens` (nor its total/aleatoric/epistemic components) — there is no ensemble to decompose with only one prompt variant present.

**Consequences:**
- RQ3b's paired clean→verbose comparison is dropped for `conf_sc` and for `conf_ens` specifically — no data exists for either on `verbose`. `conf_verb`, `conf_lp`, `conf_bpe` are unaffected (each only needs the greedy call, still collected for `verbose`).
- **Transfer test 1 (task 5.7)** trains on `clean` (has both signals) and evaluates on `verbose` (has neither) — a feature-parity break, not just a missing comparison.
- **The new verbose-shift validation** (professor feedback "consequences": train the Bayesian model on `clean`, check that epistemic uncertainty rises on `verbose` while aleatoric stays flat) hits the identical break, for the identical reason.

**Decision:** both the transfer test and the verbose-shift validation use **Tier A minus `{conf_sc, conf_ens, conf_ens_total, conf_ens_aleatoric, conf_ens_epistemic}`** as their feature set — for every model (LogReg, HistGBM, and the Bayesian model alike), not a Bayesian-specific carve-out. Tier B and Tier C are untouched (they come from per-call data `verbose` does have). Document this exception explicitly in tasks 5.7 and the new verbose-shift task, and in `REPORT.md`'s limitations section — it must never be silently imputed or left to produce NaN.

Note what this implies for the verbose-shift check specifically: the "epistemic rises under shift" prediction is tested via the **Bayesian meta-model's own posterior-based decomposition** (D22), not via `conf_ens` (which is undefined for `verbose`). This is exactly the distinction the two entropy-decomposition levels exist to support (D20 = judge-level, D22 = meta-model-level) — this check is inherently a meta-model-level one, since it's asking whether a *predictive model* correctly signals its own reduced confidence outside its training distribution.

**Amended 22 Sep 2026 — the verbose-shift check must NOT use `predict_held_out()`'s marginalized `α_q`, and this is not a minor implementation detail.** `clean` and `verbose` are paired on the exact same 80 questions (confirmed empirically: identical `question_id` sets, identical `item_id` sets, same row order) — `verbose`'s rows are **not** unseen questions the way a held-out CV fold's are. `predict_held_out()` exists specifically for genuinely-unseen questions and deliberately discards any real fitted `α_q` in favor of a fresh marginalized draw (D22's own held-out-marginalization requirement) — reusing it here would (a) throw away real, valid posterior information the model actually has for every one of `verbose`'s questions, and (b) **confound the experiment**: marginalization inflates predictive spread on *any* input, so an apparent "epistemic rises on verbose" result could just be measuring "marginalization adds noise" rather than "the model correctly recognizes distribution shift" — making the preregistered prediction unfalsifiable in the wrong direction.

**Decision:** the verbose-shift check uses the model's **real fitted `α_q`** (looked up via the *training* fold's own `question_id → dense index` mapping, from `build_group_index()`'s third return value) for both the `clean` in-sample evaluation and the `verbose` shifted evaluation — never a fresh marginalized draw. Since the question-index mapping is identical on both sides, this isolates the actual variable of interest: `verbose`'s rows carry *different feature values* (`conf_verb`/`conf_lp`/`conf_bpe` computed on the judge's verbose-condition behavior). The mechanism that should make epistemic rise under `verbose` is **`β`'s own posterior spread interacting with those shifted feature values** — different plausible `β` draws naturally diverge more when extrapolating away from the region the training data actually covered. That is a real, uncontaminated epistemic signal. `src/bayesian.py::predict_held_out()` is reserved for genuine cross-validation (5.9b/5.9c/5.9d's use, where held-out questions truly are unseen); a separate function (`predict_in_sample()`, task 5.9f) handles this same-questions-different-condition case. Both functions' own docstrings cross-reference each other and state when to use which, specifically so this choice is never made by which function happens to already exist.

---

## D22 ⚑ — Bayesian hierarchical logistic regression joins RQ4 *(professor feedback, points 1, 4)*

**Decision:** add a third model to RQ4, alongside the existing `LogisticRegression` (kept as the frequentist baseline) and `HistGradientBoostingClassifier` (kept — it tests nonlinearity, a different axis from what the Bayesian model tests; nothing in the feedback cuts it):

```
correct ~ Bernoulli(σ(α + α_q[question] + Xβ))
α_q ~ Normal(0, σ_q),  σ_q ~ HalfNormal(1),  β ~ Normal(0, 1)
```

Fit with NumPyro/NUTS. **Fallback ladder, preregistered, not improvised mid-week:** NUTS → Laplace approximation (MAP via sklearn, Gaussian from the Hessian at the optimum) → bootstrap ensemble of logistic regressions. All three produce posterior-like samples; nothing downstream changes based on which rung is used, except that the rung actually used must be stated in `REPORT.md`.

**Head-to-head table** (frequentist LogReg vs Bayesian): AUROC, ECE, Brier, NLL, 90% credible-interval coverage — the last two exist only for the Bayesian model, since the frequentist model has no native posterior. Computed under the **same `StratifiedGroupKFold(5)` × 10-seed protocol as the frequentist model (D8)** — the hierarchical structure does not replace that CV protocol, it lives alongside it.

**⚠️ Held-out random-intercept marginalization is mandatory, not automatic.** In hierarchical-model cross-validation, a held-out question's `α_q` must be marginalized over the *population-level* prior (`α_q_new ~ Normal(0, σ_q)`), never the value that question would have fitted to had it been in the training fold. Getting this wrong leaks exactly the way plain `GroupKFold` leaks — precisely the failure D8 already exists to prevent for the frequentist model. **A test must assert this** (`test_bayesian.py`, or an addition to `test_predictor.py`), mirroring D8's "two seeds must produce different folds" test.

**Mandatory convergence diagnostics.** A Bayesian result without R-hat, effective sample size, and a divergence count is unreportable — this is the Bayesian-model equivalent of invariant #12's permutation null: never optional, always printed alongside the headline number. **R-hat requires ≥2 chains to compute at all** — it compares between-chain to within-chain variance, so `MCMC(..., num_chains=1)` silently returns `NaN` for every parameter (confirmed empirically: `arviz.summary()` on a single-chain NumPyro fit returns `NaN` R-hat with no error or warning about *why*). Use `num_chains=2` at minimum for every fold-fit, not just a final "official" run.

**Runtime, measured not guessed (same principle as D12):** the existing protocol is 5 folds × 10 seeds = 50 fits. Measure how long 50 NUTS runs actually take before committing to the full 10-seed repeat count for the Bayesian arm specifically; if it's too slow, reduce that arm's repeat count and preregister the reduction rather than discovering the problem mid-W5.

**Amended 21 Sep 2026 — non-centered parameterization required (found during 5.9b's own implementation, measured not assumed):** `α_q` is sampled **non-centered** — `α_q_raw ~ Normal(0, 1)`, `α_q = σ_q · α_q_raw` as a `numpyro.deterministic` transform — rather than directly as `Normal(0, σ_q)` the way this decision's own formula above reads. This is mathematically the SAME prior (`α_q` is still distributed `Normal(0, σ_q)`), not a change to the model this decision specifies — only how NUTS explores it. The direct/centered form hit **Neal's funnel** on real project data: `α_q` and `σ_q` become highly correlated in the posterior geometry when `σ_q` is small, and NUTS's fixed step size can't navigate the resulting narrow region — a geometry problem that more warmup/samples alone doesn't reliably fix (confirmed empirically: increasing `num_warmup` 500→1000 made R-hat *worse*, 1.07→1.20, before a larger 1500/2000 budget only marginally helped, 1.02). Measured on the real Tier A / fold-0 data, `num_warmup=500, num_samples=1000, num_chains=2`: centered gave `max_rhat=1.07` (flagged), `min_ess=21`; non-centered, same settings, gave `max_rhat=1.01` (not flagged), `min_ess=791` — roughly 38× better effective sample size for ~1.5× the runtime cost (9.2s → 14.5s per fit). This is the standard, well-established fix for hierarchical-model funnels (Betancourt & Girolami, *Hamiltonian Monte Carlo for Hierarchical Models*), not a bespoke workaround.

**Runtime, the REAL full-protocol measurement (21 Sep 2026, `python -m src.bayesian --config configs/run.yaml --tier A`):** N=1819, Tier A, 8 features, D8's full 5-fold × 10-repeat protocol (50 real fold-fits, non-centered, `num_warmup=500, num_samples=1000, num_chains=2`) — **421.0s (7.0 minutes) total**, comfortably affordable at the full `n_repeats=10`, no reduction needed. Convergence held up at scale, not just on the single fold-0 fit the reparameterization was originally diagnosed on: **1/50 fold-fits flagged** (barely — `max_rhat=1.020`, just above the 1.01 threshold), **zero divergences across all 50 fits**. AUROC: mean 0.7898, spread [0.7848, 0.7982] (D8's across-repeat spread, not a within-split CI) — the head-to-head comparison against the frequentist Tier A LogReg result (5.5: 0.8239) is 5.9c's job, not asserted here.

**Invariant 11 is reworded, not overturned.** Its original justification — "~80 groups, indefensible at this N" — is exactly the problem hierarchical partial-pooling models exist to solve. The permitted model set becomes `LogisticRegression`, `HistGradientBoostingClassifier`, and the Bayesian hierarchical logistic regression above. "No neural nets, no hyperparameter tuning" still holds for all three.

**Amended 24 Sep 2026 (project audit) — features are standardized per training fold.** The hierarchical model put `β ~ Normal(0, 1)` on raw feature values, while the frequentist LogReg standardizes inside its Pipeline. On Tier A the raw scales differ hugely (`conf_lp`'s standard deviation is 0.014, `conf_verb`'s 0.034, the entropy columns' ~0.3), so the same prior all but switched off the narrow-range signals — a hidden, uneven regularization that made the Bayesian-vs-LogReg comparison unfair. `repeated_stratified_group_kfold_bayesian()` now fits a StandardScaler on each training fold (the same leakage guard as LogReg), and 5.9f fits one on `clean` and applies it unchanged to `verbose` (re-fitting on `verbose` would erase the shift being tested). A test asserts predictions don't change when a feature is rescaled. Every Bayesian result was re-run. The one reading that changes: in the 5.9c head-to-head the Bayesian model no longer looks "slightly worse on AUROC, notably better calibrated" (0.790 / ECE 0.023) — it matches LogReg on every shared metric (AUROC 0.795 vs 0.796, ECE 0.035 vs 0.037, Brier tied). 5.9d (98.7% of the ensemble's AUROC edge retained; epistemic AUROC 0.779 vs 0.559), 5.9f (prediction still did not hold), RQ6 and RQ7 (meta-model still doesn't beat the best single signal) keep their conclusions. RQ6's convergence moves from 0/50 flagged to 5/50 and 2/50 — its two signals are strongly correlated, which standardization makes a sharper posterior ridge.

**Amended 24 Sep 2026 — R-hat was rounded before the check; draws doubled to 2,000 per chain.** `convergence_diagnostics()` read R-hat from `az.summary()`, which rounds to two decimals by default, so an R-hat of 1.014 was stored as 1.01 and passed the `> 1.01` test; the flag counts above (1/50, 5/50, 2/50) were undercounts. Read at full precision (`round_to="none"`, a test now guards it), 10–15 of each run's 50 fold-fits sat just above 1.01 (worst 1.024), none badly mixed. `num_samples` is doubled from 1,000 to 2,000 per chain (`NUM_SAMPLES` in `src/bayesian.py`; warmup 500 and 2 chains unchanged). At 2,000: primary judge 0/50 flagged (worst R-hat 1.008); kev-8b 7/50 in-coverage (one at R-hat 1.012, six with divergences, 11 in total) and 1/50 out-of-coverage; auto-j 0/50 and 1/50. AUROC, ECE and every conclusion moved only in the third or fourth decimal. The Bayesian result tables now record worst R-hat, lowest ESS and divergence counts (`summarize_diagnostics()`), not only the flag count — this decision always required all three to be reported.

---

## D23 ⚑ — RQ5: prompt distillation and the human-disagreement validation *(professor feedback, points 2, 4; "consequences" section)*

**New research question.** RQ5: does marginalizing over the prompt (the P1/P2/P3 ensemble) produce a better uncertainty estimate than any single prompt, and how much of that benefit survives distillation to single-call cost?

**Decision — distillation framing.** The ensemble's **predictive distribution** (not just its mean) is the teacher; the Bayesian meta-model (D22), trained on **single-call, P1-only** features, is the student. No LLM fine-tuning — this is purely about whether the cheap single-call model's own posterior predictive spread resembles the expensive 3-call ensemble's actual spread. Headline metric: how much of the ensemble's benefit (in AUROC/ECE/entropy quality) survives at 1 call vs 3.

**Decision — validate the decomposition against human disagreement.** MT-Bench's repeated human votes give a real, model-free measure of how contested each item genuinely is (`d_human`, D9). Test whether the **judge-level estimated aleatoric** signal (D20) is high specifically where humans actually disagreed, and epistemic is not. This reuses existing infrastructure (`d_human`/H4) rather than building new statistical machinery — it validates the *existing* aleatoric/epistemic vocabulary against an independent signal, which is stronger than either alone. This analysis runs on `clean`/P1 rows only (where `conf_ens`'s components exist), consistent with D20's P1-filter.

**Positioning:** closest prior work is *Auto-Prompt Ensemble for LLM Judge* (Oct 2025, same Qwen2.5-7B/MT-Bench setup) and *Calibrating MLLM-as-a-Judge via Multimodal Bayesian Prompt Ensembles* (ICCV 2025) — neither decomposes entropy into aleatoric/epistemic, and neither validates against real human votes. That combination is RQ5's actual contribution.

---

## D24 — New dependencies: local, not Colab *(professor feedback, mechanical consequence of D22)*

**Decision:** `numpyro` (+ its `jax` dependency, CPU-only — no GPU needed for a hierarchical logistic regression at this N) and `arviz` (for the convergence diagnostics D22 makes mandatory) go into `pyproject.toml`'s **base install**, not the `colab` extra. Fitting the Bayesian model is analysis work on already-collected data, consistent with the existing framing that RQ4 costs zero extra GPU time (`PLAN.md` sec 1) — it runs on the same local `judge-calib` conda env as everything else in `src/` except `judge.py` (D17).

---

## D25 ⚑ — `conf_lp` is inflated by constrained-decoding renormalization; `vllm` version bump investigated and rejected *(found while designing task 1.8's vacuum test, 8 Sep 2026)*

**The problem.** Structured/guided decoding works by masking disallowed tokens to `-inf` before softmax, then renormalizing over whatever survives. At the verdict position, that shrinks the competition pool from the full vocabulary down to just `{A, B}`. This means `conf_lp = exp(verdict_token_logprob)` — an *absolute* probability — gets systematically inflated: a token that would have had modest absolute probability across the full vocabulary (competing against everything else the model might have "wanted" to say) can end up looking highly confident once renormalized against only one other candidate. This is **not** a vacuum-test-specific issue — it affects every use of `conf_lp` throughout RQ1–RQ4, since the whole harness uses constrained decoding.

**Crucially, this does NOT affect everything logprob-derived.** `p_a = P(A)/(P(A)+P(B))` is a *ratio* between exactly the two surviving candidates, and renormalizing a distribution by a constant factor preserves the ratio between any two specific survivors. So `p_a` — and everything built on it (`verdict_bidir`, `conf_bpe`) — is robust to this distortion. Only the absolute-probability signal (`conf_lp`) is compromised.

**Investigated fix: `SamplingParams(logprobs_mode="raw_logprobs")`.** vLLM's `main` branch (post-`0.28.0`) captures logprobs *before* the structured-output mask is applied — confirmed by reading the actual sampler source (`vllm/v1/sample/sampler.py`), which computes `raw_logprobs` from unmasked logits ahead of `apply_logits_processors()`. This would give the true, unconstrained top-K distribution at the verdict position, for free, from a single call.

**Checked empirically against the pinned version, not assumed:** `SamplingParams(logprobs_mode=...)` raises `TypeError: Unexpected keyword argument 'logprobs_mode'` on `vllm==0.28.0` — the feature isn't in this release.

**Considered re-pinning to a newer vLLM specifically for this.** D11 says "do not move it again," but the reasoning behind that rule (preserving comparability of a mostly-collected dataset) doesn't yet apply — this is pre-2.1, before the real full run, and the only data collected so far (task 1.6's 140+20-row pilot) is explicitly disposable, not the final dataset. So this was a genuine reconsideration, not a rule violation, weighed the same way any other version pin gets decided (D11's own precedent: measure, then pin).

**Rejected — there is nothing to re-pin to.** `uv pip install --upgrade-package vllm` against PyPI still resolves to `0.28.0`: it's the latest published release. `logprobs_mode` exists only in vLLM's unreleased `main` branch. The only way to get it would be installing directly from an unreleased git commit — no stable version pin, real risk of unrelated instability, a materially bigger and riskier ask than picking a newer *released* version. Not worth it for an unquantified benefit.

**Decision, final:**
1. Stay on `vllm==0.28.0`. No pin change.
2. For the vacuum test (task 1.8) and generally: `conf_verb` (the model's own stated confidence, not mechanically derived from token renormalization) is the primary "is the judge falsely confident" signal. `conf_lp`/`p_a`-derived signals stay in use but are understood as reporting an upper bound on true confidence, not an exact measurement, wherever `conf_lp` specifically is involved.
3. **Task 4.5's constrained-vs-free-form ablation is the real resolution path** — it will produce actual evidence on how large this gap is. If it turns out to be large, that's a well-justified, evidence-based reason to revisit a version bump for a deliberate re-run (or a future vLLM release may ship `logprobs_mode` by then). Revisiting speculatively, without that evidence, is not.

---

## D26 ⚑ — Every model-dependent output filename carries `Config.model_slug` *(owner asked whether a second judge model could be added later without touching existing results, 18 Sep 2026)*

**The question.** A second (or third) judge model is cut from this pass (drop-order item 1, `PLAN.md` §4; `configs/models.yaml`'s own note) but not ruled out for later, *if time permits*. The owner wanted the option kept open without a structural refactor deep into the project, and without hand-maintained per-model config duplication ("config stitching").

**Decision:**
1. `Config.model_slug` (`src/config.py`) derives a filesystem-safe tag from `judge_model` — org prefix stripped, lowercased, hyphens to underscores (`"Qwen/Qwen2.5-7B-Instruct"` → `"qwen2.5_7b_instruct"`). One field (`judge_model`) is the single source of truth; nothing else needs to be hand-specified per model.
2. **`runs/` gets a subfolder per model** (`runs/{model_slug}/`), not suffixed filenames — `runs/logprobs/` holds one gzipped file per call (~19k in the first run alone), and suffixing thousands of individual filenames per model would be unworkable. `config.paths.runs_dir` is simply set to `runs/{model_slug}` per model config; `judge.py`/`parse.py` needed no code changes since `runs_dir` was already threaded through as a parameter, never hardcoded.
3. **`results/` stays ONE shared directory**; individual filenames carry the `_{model_slug}` suffix instead (`calls_{model_slug}.parquet`, `rq1_table_{model_slug}.csv`, every figure). Deliberately not subfoldered like `runs/` — `results/` holds a handful of aggregate files, small enough that a future cross-model comparison can `glob("results/rq1_table_*.csv")` and concatenate, rather than needing a hardcoded list of per-model directories.
4. **`items_labels.parquet` is the one deliberate exception — never suffixed.** It's built from human votes alone (`src/data.py`) and doesn't depend on `judge_model` at all; suffixing it would just create a driftable duplicate per model for no reason.
5. Filename length was checked empirically, not assumed safe: worst case today (`results/figures/reliability_conf_verb_qwen2.5_7b_instruct.png`) is ~63 characters, full path ~125 — nowhere near Windows' 260-character `MAX_PATH`, and this holds for any realistically long future model name too.

**What this did NOT require:** no changes to `judge.py`, `signals.py`, `data.py`, or `vacuum_test.py` — all of them already read every path from `config.paths.*`, nothing was hardcoded. The only code touched was `src/plots.py` (figure filenames) and the four `analysis/*.py` scripts' own table-writing calls.

**Applied retroactively to the existing single-model (Qwen) results**, not just banked as a design for later: the real `runs/judge_clean.jsonl`, `runs/vacuum.jsonl`, and all 19,100 `runs/logprobs/*.jsonl.gz` files were moved (not regenerated — no GPU involved) into `runs/qwen2.5_7b_instruct/`, and every derived file (`calls.parquet`, `items.parquet`, every RQ1/RQ2/RQ5/human-disagreement table and figure) was regenerated from there and diffed number-for-number against what was already committed in `REPORT.md` before anything old was deleted. Nothing shifted.

---

## D27 ⚑ — kev-8b as the open stand-in for Jev's "System One" calibration claim *(owner's own research find, 22 Sep 2026 — TypeSafe's Jev blog post)*

**The question.** TypeSafe AI's Jev is the one live, publicly claimed counterexample to this project's central thesis — a purpose-built, non-autoregressive "System One Model" trained via "RLCD" specifically to produce well-calibrated typed decisions, explicitly positioned against exactly the failure mode this project studies. Jev itself is proprietary — no weights, no size, no benchmarks disclosed anywhere. Three open, Apache-2.0 candidates explicitly built to the same `/v1/systemone` contract were evaluated as stand-ins: `jaredpalmer/kev-8b`, `bespokelabs/Bespoke-Nimble-9B`, and `jbarney/circuit-8b`.

**What was checked, and what disqualified two of the three.** All three are LoRA adapters (plus a custom pointer/readout head) on a Qwen3/3.5 base, trained on short-context classification/QA tasks (Banking77, BoolQ, AG News, customer-service tickets, NLI, spam, etc.) — **none were trained on pairwise response judging**, making MT-Bench inherently an out-of-domain generalization test for all three, not an apples-to-apples "best-in-class judge" comparison. This must stay prominent in any writeup, not buried in a limitations footnote.

- **Nimble-9B**: hard-rejects (not truncates) prompts over 2,048 tokens. Checked against this project's own real `n_prompt_tokens` (task 4.2): only 41.0% of `verbose` items would survive — breaks the paired Full-N population this study already commits to elsewhere.
- **circuit-8b**: trained at ≤1,024 tokens; only 63.3% of `clean` items and 12.4% of `verbose` items fall in its documented training range — compromises even the basic calibration check, not just the attack battery.
- **kev-8b**: its GitHub README (not the HF model card — the card discloses nothing about this) states training at ≤1,024 tokens but a server-side accept boundary up to 8,192, with no hard rejection below that. The only candidate compatible with the Full-N paired population.

**Decision: kev-8b is the comparison arm.** `Qwen/Qwen3-8B-Base` + a LoRA adapter (r=16) + a pointer head trained from scratch, served via `kev.serve`'s `/v1/systemone` HTTP contract — not loaded through plain `transformers`/`peft`, since the pointer head isn't wired into a standard generation pipeline.

**Real serving ceiling, measured directly, not read off a card** (`kev_token_probe.py` / `kev_token_probe_boundary.py`, scratchpad diagnostics, not committed to `src/`):
- The disclosed "accepts to 8,192" boundary is not a clean cutoff. Requests in roughly the 8,165–8,192 range are **genuinely non-deterministic** — the identical input (same token count, same `pad_side`) succeeded in one probe pass and failed in a later pass, confirmed via `kev_serve.log` to be a real `torch.OutOfMemoryError` inside the LoRA forward pass (PyTorch's own "reserved but unallocated" fragmentation signature), not a validation rejection. This reproduced across **two independent server sessions** — one with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and GPU state contaminated by a prior zombie process, one restarted clean (confirmed via `nvidia-smi`) without that flag — ruling out both the allocator setting and session contamination as the sole cause. This is a real property of kev-8b on a 24GB-class GPU (L4), not an artifact of this project's setup.
- A separate, clean, *deterministic* `422 "branch too long"` rejection exists right at the edge (observed at 8,176 and at 8,192 across the two probe runs — the exact point moves slightly with construction details). Root cause, confirmed from `kev/model.py`: `max_branch` is a fixed total token budget covering the state *and* the question's own tokens together; at these lengths the state alone consumes the whole budget, leaving no room for even the ~17-token question. A different, better-behaved failure mode than the OOM above — both are real and distinct.
- **Practical ceiling adopted for the real run: 8,160 tokens** — the largest value confirmed successful with zero contradictions across both probe sessions.
- **Population impact**, checked against the real `calls_qwen2.5_7b_instruct.parquet`: 100% of `clean` items, **97.06% of `verbose`** items (112/3,808 call-rows, ≈56 unique items) fit under this ceiling. The excluded items are a small, explicit, documented exclusion — not a large biased subsample the way Nimble's cap would have forced, and not silently imputed.

**Launch configuration (load-bearing, not incidental — needed to actually reproduce a working server):** `KEV_DTYPE=bf16 KEV_MERGE=0 KEV_ATTN=sdpa`, process started via `setsid ... < /dev/null` rather than plain `nohup ... &` (Colab's cell-interrupt signal propagates to the whole kernel process group; `nohup` alone does not protect a backgrounded process from it — confirmed empirically after a server died mid-session from interrupting an unrelated log-tailing cell). `KEV_MERGE=0` is the critical one: `kev/checkpoint.py`'s own load path forces `dtype=torch.float32` whenever `merge` is true (the default), **regardless of what `KEV_DTYPE` resolves to** — this caused the first load-time OOM crash (fp32, ~33GB, on a 22GB card) and would not have been caught by setting `KEV_DTYPE=bf16` alone, which is all the README documents.

**Real, unplanned finding surfaced by the diagnostic probing itself** (not yet a formal, controlled experiment — a probe-script observation, flagged as such): across every tested length from 256 to 4,096 tokens, kev-8b's verdict was **100% determined by which side of the input the real content sat on**, independent of content. Real content placed at the *start* of the sequence (filler padding appended after) → verdict "A" every time; real content at the *end* (filler prepended before) → verdict "B" every time. Confidence on the winning choice generally *rose* as more filler padding was added, rather than falling. This held well inside kev's own disclosed training range (256–1,024 tokens), so it is not a truncation artifact — it is a content-blind positional bias, structurally similar to but more severe than this project's own existing AB/BA order-swap probe on the primary judge. A proper controlled follow-up (real MT-Bench content on both sides, not nonsense filler) is needed before this becomes a reportable claim rather than a diagnostic-script observation — flagged here so it isn't lost.

**Excluded from this comparison arm, and why:**
- `conf_sc`/`conf_ens` analogs — kev-8b makes one deterministic forward pass per request; there is no independent-sample self-consistency signal to construct, and resampling from its own reported distribution would just recover the same number, not an independent estimate.
- Bespoke-Nimble-9B, circuit-8b — ruled out above on token-cap grounds specific to this project's prompt lengths, not on any calibration-quality basis.
- **`SemIf`** (surfaced by external research as a fourth candidate) — investigated and rejected. It is not one model release; it is a cluster of near-identical, independently-published repos (`james-see/SemIf`, `TheoLeeCJ/SemIf`, `someka-vrc/SemIf-server`, `Mapika/decider`, and others, all carrying the same tagline and each explicitly disclaiming Jev/TypeSafe affiliation) with no single canonical or authoritative source, and self-published, unverifiable benchmark claims (an informally-named "JevBench," a reported perfect 1.000 score on one metric). Does not meet this project's evidentiary bar.

**Documentation weight, superseded 23 Sep 2026 — treated as RQ6, not a D27-only addendum.** The original call here was a standalone `REPORT.md` section, not a renumbered RQ. The owner's explicit instruction: treat this as RQ6 in full (CLAUDE.md's RQ table, PLAN.md §7), with the understanding that if the professor's later feedback says to scope it down, it gets trimmed then — not preemptively minimized now. `TASKS.md`'s K-block keeps its own K1–K5/GATE K numbering rather than being folded into the W0–W7 sequence, specifically so it stays easy to isolate or trim later without renumbering anything else.

**Amended 22 Sep 2026 — `call_kev()` violated this project's own raw-materials-first principle and was fixed, not patched around.** `src/judge_kev.py::call_kev()` originally extracted only `choice`/`probabilities`/`input_tokens` from each response and discarded the rest — a curated subset, decided unilaterally, not surfaced before the real run. This directly contradicts CLAUDE.md invariant 7 and `judge.py`'s own established pattern (save `raw_output` in full, parse/derive signals later, in a separate step) — a principle already set elsewhere in this codebase, not a novel judgment call that could be excused as an open question. The concrete cost, believed at the time: kev's `/v1/systemone` response schema includes a `confidence` field distinct from `probabilities` — in the schema's own worked example, `confidence` (0.21) is *not* equal to the chosen option's own probability (0.47) — and this field was silently dropped across the entire first 7,616-call run.

**Fix:** `call_kev()` now returns the complete raw response verbatim — the full parsed JSON body on success, the full raw error text on an HTTP failure, the full exception string on a network failure — nothing curated at collection time, going forward. Deriving `choice`/`probabilities`/anything else from the saved `raw_response` is K4's job, not `judge_kev.py`'s, the same layering `parse.py` already provides for the primary judge.

**Amended again 23 Sep 2026 — the first full run was NOT actually discarded; it was restored and is the basis for K3/K4.** Reasoning at the 22 Sep amendment ("the raw response bodies were never saved, so `confidence` cannot be recovered, discard and redo") turned out to be based on incomplete investigation, not a hard fact — the recommendation to discard was made before checking whether the missing field was recoverable some other way. Two things resolved this:
1. **`confidence` is confirmed to be an exact, deterministic function of `probabilities`, not an independent signal**, verified directly against kev's own source (`kev/api.py`, not a secondhand summary): `choice_confidence(p) = (max(p) - 1/K) / (1 - 1/K)`. For this project's 2-option `verdict` question, that's `confidence = 2·probabilities[choice] − 1` exactly. It is not separately RLCD-trained; it carries zero information beyond `probabilities[choice]`.
2. **The first run's `call_kev()` already saved the full `probabilities` dict** (both `"A"` and `"B"` keys) and `input_tokens`, even in its curated-subset form — everything the settled analysis plan (below) actually needs. `confidence`'s absence turned out not to matter, because the plan drops it as redundant anyway.

**Decision: `confidence` is excluded from all analysis, deliberately, not by omission.** Only `probabilities[choice]` is used as kev's raw class-probability signal. This is a different situation from D20's `conf_ens`/`ens_entropy_total` collinearity, which was kept, not dropped — that pair are two members of a pre-registered three-part decomposition (total/aleatoric/epistemic) with independent explanatory value beyond their algebraic redundancy, evaluated with a frequentist `LogisticRegression` that tolerates near-collinear features numerically. `confidence`/`probabilities[choice]` are two notations for the exact same number, with no larger structure justifying keeping both, evaluated partly via the D22 Bayesian model, whose proper-but-weak `Normal(0,1)` priors are more exposed to a genuine non-identifiability ridge under *exact* collinearity than a standardized frequentist fit is. The reasoning differs from D20's on its merits; it isn't a reversal of D20's own call.

**Two more decisions settled 23 Sep 2026, recorded here so they don't live only in conversation:**
- **Population reporting:** results are reported as two explicit regimes — in-coverage (`input_tokens` ≤ ~1,024, kev's own disclosed training range) and out-of-coverage (1,024–8,160) — using the server-reported `input_tokens` from `raw_response`/the saved checkpoint field, not this harness's own `state_tokens`, since `input_tokens` includes the question branch's tokens too and is the quantity kev's documented thresholds are actually stated in.
- **Battery weighting:** calibration check, position-swap attack, verbosity attack, and the D22 Bayesian recalibration check are weighted equally in the writeup by default; if one result is disproportionately striking, that gets decided once the numbers are in, not preregistered as a headline now.

**One methodological point for K4 to get right, flagged here before it's built:** the verbosity attack is a *paired* clean-vs-verbose comparison (invariant 3). 104 `verbose` call-rows (~52 items) were skipped for exceeding the 8,160-token ceiling; their `clean` counterparts were not. K4 must restrict the paired verbosity-attack test to items where *both* conditions succeeded, not just filter `verbose`'s own missing rows — otherwise the paired bootstrap silently loses its pairing for those ~52 items.

**Process fix, not just a code fix — still holds, independent of the run being restored.** The owner set a standing rule following this incident — ask before any consequential decision (what to capture, what to discard, what scope to run) rather than deciding unilaterally and reporting after. Sharpened further once the run was restored: before recommending that already-collected data be discarded, first check whether what's missing is recoverable some other way (derivable from what's already saved, computable from a known formula) — investigate recoverability before proposing disposal, not after.

**K4 done, 23 Sep 2026 — all four tests, both signals, both regimes.** Full results in `results/rq6_{calibration,position_swap,verbosity,bayesian_recalibration}_kev_8b.csv` and `TASKS.md`'s own K4 closeout. Headline: `conf_kev_bpe` beats `conf_kev` on every calibration metric in both regimes (replicates the primary judge's own `conf_bpe`-wins pattern on a different architecture); position-swap confidence gaps are significant and markedly larger than the primary judge's own `conf_verb` gap; `conf_kev_bpe`'s calibration breaks significantly under the verbosity attack while `conf_kev` doesn't (mirrors the primary judge's own finding that its *best* signal was uniquely vulnerable); the Bayesian meta-model does not beat the best single signal in either regime, with clean convergence (0/50 fold-fits flagged) in both.

**Methodological note, since it's a real question a reviewer would reasonably ask: is the Bayesian meta-model check sound given `conf_kev`/`conf_kev_bpe` are related signals?** Yes, and it's a different situation from `confidence` being dropped, not an inconsistency:
- `conf_kev` is a function of one number (`prob_a` from the AB call alone); `conf_kev_bpe` is a function of two (`prob_a` from AB *and* BA, order-corrected). They share one of two inputs — correlated, not collinear. `confidence`/`probabilities[choice]` was exact algebraic identity (one number, two labels); this is not that.
- D22's proper (not flat) `Normal(0,1)` priors handle correlated-but-non-identical predictors without a non-identifiability pathology — and this was checked empirically, not assumed: 0/50 fold-fits flagged in both regimes is the direct evidence the correlation didn't degrade NUTS mixing, the same diagnostic that already caught a real problem once in this project (Neal's funnel).
- The check never interprets individual coefficients (unlike D20's `conf_ens`/`ens_entropy_total` case, where collinearity genuinely would split one effect into two misleading numbers) — it only compares the meta-model's holistic AUROC against the best single signal's, a comparison correlated features don't invalidate.
- The result itself (meta-model ~tied or slightly behind the best single signal) is the coherent consequence of two partially-overlapping signals having limited independent information to combine — not a sign of a modeling problem.

---

## D28 ⚑ — auto-j-13b: does purpose-built judge training resist the same failure modes? *(owner-initiated, 23 Sep 2026 - RQ7)*

**The question.** RQ1–RQ3 found the primary judge (Qwen2.5-7B-Instruct, general-purpose) measurably overconfident and vulnerable to position bias and a verbosity attack that specifically breaks its own best uncertainty signal. RQ6 then asked whether an industry counterclaim (TypeSafe's proprietary "Jev," stood in for by the open `kev-8b`) resists the same failure modes — it doesn't, but kev-8b was never trained to judge anything (an out-of-domain test, stated as RQ6's headline caveat throughout). RQ7 closes the obvious gap: `auto-j-13b` (`GAIR/autoj-13b`, Li et al., ICLR 2024, arXiv:2310.05470) is a model **purpose-trained specifically for pairwise response judging** (GPT-4-distilled critique+verdict data). If a model built for exactly this task still shows the same pattern, that is a materially stronger claim than RQ1 or RQ6 alone — the failure mode generalizes across training objective, not just across one model family.

**What was verified, and how — primary source, not docs summary, same standard D27 was held to.**

- **Serving.** auto-j-13b is a standard Llama-2-13B causal LM with **native vLLM support** — confirmed both from the model card and empirically: a real Colab smoke test loaded `GAIR/autoj-13b-GPTQ-4bits` on this project's pinned `vllm==0.28.0` and generated successfully, using vLLM's own built-in **Marlin GPTQ kernel** (`Using MarlinLinearKernel for AutoGPTQLinearMethod`) — no separate `auto-gptq`/`optimum` runtime dependency needed, better than the repo's own documented `transformers`+`AutoGPTQ` usage path. This is architecturally much closer to the **primary judge's own vLLM-native harness** than to kev-8b's custom `/v1/systemone` HTTP server.
- **Real smoke-test numbers** (not estimates): model weight load 6.78 GiB; with `gpu_memory_utilization=0.85` (18.73 GiB budget out of ~22 GiB usable on the L4), KV cache got 10.93 GiB → 14,320 tokens of cache, "maximum concurrency 3.5x" at a 4,096-token request length — comfortable headroom, confirms the earlier weight-size-based estimate (fp16 would be ~26 GiB, measured from the real HF file sizes, and would not fit an L4 at all — GPTQ-4bit's 7.92 GiB single safetensors file does, easily). Engine init (weight load + `torch.compile` + warmup) ≈ 182s one-time per process; actual weight loading alone is 2.7s.
- **Context.** `config.json` confirms `max_position_embeddings=8192` directly — no empirical ceiling-probing needed the way kev-8b's undocumented practical ceiling was (a genuine simplification vs. RQ6; auto-j's disclosed context comfortably covers 100% of both `clean` and `verbose`, no coverage-regime split needed this time).
- **License.** Llama 2 Community License — fine for this academic use.
- **The real prompt/output contract**, pulled verbatim via `curl` from `github.com/GAIR-NLP/auto-j/codes/usage/constants_prompt.py` and `example.py` (not paraphrased from docs):
  ```
  PROMPT_INPUT_WO_SYSTEM = "[INST] {input} [/INST]"

  pairwise_tie = """You are assessing two submitted responses on a given user's query and judging
  which response is better or they are tied. ...
  [Query]: {prompt}
  [Response 1]: {response}
  [Response 2]: {response_another}
  ... Begin your final decision statement with "So, the final decision is Response 1 / Response 2 / Tie" ..."""
  ```
  Extraction (`extract_pariwise_result`): `raw_output.rfind('final decision is ')`, then checks whether the remainder starts with `'response 1'` / `'response 2'` / `'tie'` (case-insensitive) → labels `0`/`1`/`2`, else `-1` (parse failure). `SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)` in the authors' own example. **No native confidence/probability field anywhere** — free-text critique + this one extractable line. My own first smoke-test prompt did *not* use this real template (it improvised its own wording), which is the most likely reason that test's generated critique read as self-contradictory — confirms the template must be used verbatim, never improvised, for any real data collection.
- **Contamination risk.** auto-j trains on GPT-4-labeled pairwise judgments over `lmsys/chatbot_arena_conversations` (sampled real user traffic) — a *different* dataset from this project's own eval set, `lmsys/mt_bench_human_judgments`. Confirmed from lmsys's own blog post that the 80 MT-Bench questions were **hand-crafted** by the paper's authors — Arena traffic was used only to identify the 8 categories, never to source question text. Direct item-level leakage assessed as unlikely. An exhaustive string-match check against the raw Arena conversations is blocked by a gated HF dataset request only the owner can approve (the HF token has `canReadGatedRepos` capability but hasn't been granted access to this specific dataset) — not required to proceed, available later if wanted.

**Decisions locked with the owner (AskUserQuestion, 23 Sep 2026) — do not relitigate silently:**

1. **GPU/quantization: GPTQ-4bit on L4**, per the confirmed-working smoke test — cheaper (D12's compute-unit framing), no A100 needed. Accepted real caveat: the model's own authors note quantized behavior "might be different" from the full fp16 model — stays a documented limitation, not silently absorbed.
2. **Scope: `clean` + `verbose`**, the full RQ6-parity four-test battery (calibration, position-swap, verbosity attack, Bayesian recalibration) — a real three-way comparison (Qwen vs. kev-8b vs. auto-j) across all four tests.
3. **Signal scope, this pass: minimal** — `conf_sc` (self-consistency) and an order-swap bidirectional-entropy analog only. **`conf_lp` deferred, not attempted.** auto-j is not schema-constrained the way the primary judge is (no `StructuredOutputsParams` forcing exactly two surviving candidates at a fixed position) — `parse.py::compute_logprob_signals()`'s own `p_a` formula only returns a value when *both* candidate tokens appear in the saved top-20 logprobs at the located position, which structured decoding guarantees and free generation does not. Building a free-text verdict-token-locator and accepting an unknown, likely much-higher-than-Qwen's-near-0% null rate is real, non-trivial work — kept as a clearly-flagged stretch item.
4. **Multi-turn (`turn=2`) handling.** auto-j's template takes one flat `{prompt}` field with no native multi-turn support, but MT-Bench is ~half turn=2 items. A real risk was raised and resolved explicitly: deciding "keep vs. discard turn=2" *after* seeing whether its numbers look good would be outcome-dependent reporting — a materially worse researcher degree of freedom than this project's existing "decide emphasis after the numbers are in" precedent (this D28's own point 4 vs. D27's battery-weighting language are not the same kind of choice). **Locked policy:** turn=1 and turn=2 are **always both reported**, as two explicit populations — mirrors D27's in-coverage/out-of-coverage regime split, the identical structural pattern, already proven to work. Turn=2's validity is judged by a **pre-committed, outcome-independent check** (task L4b) — parse-failure rate (`extract_autoj_verdict() == -1`) compared turn=1 vs. turn=2, plus a manual spot-check of real rendered turn=2 prompts — run and decided **before** any calibration/accuracy number exists for either turn. A real problem there earns turn=2 an explicit stated caveat in `REPORT.md`; otherwise it's reported with equal standing to turn=1. Never decided by how the headline numbers look.

**The turn=2 prompt design itself** (`build_autoj_prompt`, task L3), flagged here as the concrete thing L4b validates, not asserted as obviously correct: `{prompt}` carries only the two shared, model-independent user turns ("Earlier in the conversation, the user asked: '{turn1_user}'. Now the user says: '{turn2_user}'"), while each model's own turn-1 answer is folded into *its own* `{response}`/`{response_another}` field as a two-part trajectory. This keeps the shared prompt field honest — it never contains a model-specific answer that would contaminate the other candidate's context — at the cost of a template usage auto-j's own materials never demonstrate.

**Documentation weight**, same treatment RQ6 got: full RQ (`CLAUDE.md`'s RQ table, `PLAN.md` §8), `TASKS.md`'s own **L1–L6/GATE L** block, kept outside the W0–W7 numbering so it stays isolable/trimmable later without renumbering anything else, same as the K-block.

**Amended 23 Sep 2026 - the "8,192-token context covers 100% of both conditions, no coverage-regime split needed" claim above was wrong, and it was wrong because it was never actually measured.** The first real Colab run (`--n-items 20` smoke test) crashed with `vllm.exceptions.VLLMValidationError`: a rendered prompt exceeded 8,192 tokens on its own, before even reserving room for the 1,024-token output budget. This was found and root-caused **locally, offline, with no GPU** - `transformers.AutoTokenizer.from_pretrained("GAIR/autoj-13b-GPTQ-4bits")` loads fine without PyTorch/a GPU, so `build_autoj_prompt()`'s real token length was measured against the full 1,904-item population directly, the same "measure, don't guess" standard D12/D22/D27 already hold this project to - the original claim skipped that step and generalized from the *primary judge's* own verbose-condition average instead, which was the actual mistake.

**Real measured distribution** (3,808 rows = 1,904 items × 2 conditions, `AB` order, `transformers`' own tokenizer):

| | n | mean tokens | max tokens | >7,168 | >8,192 |
|---|---|---|---|---|---|
| clean / turn=1 | 948 | 740 | 2,053 | 0 | 0 |
| clean / turn=2 | 956 | 1,281 | 3,368 | 0 | 0 |
| verbose / turn=1 | 948 | 2,380 | 8,407 | 17 | 1 |
| verbose / turn=2 | 956 | **4,527** | **13,808** | **164 (17.2%)** | 109 |

**Root cause, not a random tail:** `build_autoj_prompt()`'s own turn=2 design (this same decision, above) folds *both* turns' assistant content into each side's `{response}` field, and `verbose_pad()` pads *both* of those assistant turns independently - a real compounding effect specific to the verbose+turn=2 combination, not present for either perturbation alone. clean and verbose/turn=1 stay comfortably within bounds; verbose/turn=2 is the one population segment where auto-j's context genuinely binds.

**Decision, confirmed with the owner (AskUserQuestion, 23 Sep 2026): skip-and-log at 7,168 tokens** (`max_model_len − max_tokens`, reserving the full 1,024-token generation budget for every accepted call, rather than 8,192 with borderline items risking truncated output) - mirrors `judge_kev.py::_run_calls()`'s own `max_state_tokens` skip-and-log pattern exactly: `src/judge_autoj.py::split_by_prompt_length()` checks every call's real rendered-prompt length **before** any batch is sent to `llm.generate()` (a single over-length prompt crashes the whole batch call, not just that one request - confirmed by the crash itself), writes `skipped=True, skip_reason="over_max_prompt_tokens"` rows for the excluded calls, never silently drops them. **Real population impact: 181/3,808 rows (4.75%) skipped overall**, concentrated almost entirely in verbose/turn=2 (164/956, ~17.2% of that specific subgroup) plus a small verbose/turn=1 tail (17/948, ~1.8%) - materially larger than RQ6's kev-8b exclusion (2.89%), and this **must be stated as a real, honest limitation in `REPORT.md`'s RQ7 section (L6)**, not minimized - a meaningful fraction of the hardest attack combination (verbosity × multi-turn) is structurally unreachable at this model's context length, which is itself a substantive finding about auto-j's practical usability under a combined attack, not just a methodological footnote.

**Supersedes, not deletes:** `PLAN.md` §8.5's original "no coverage-regime split needed" sentence is corrected in place with a pointer back here, per this project's own superseding convention (D5/D9/D14/D18's own precedent - the audit trail stays, not the wrong claim standing uncorrected).

**Amended again 23 Sep 2026 - a second, distinct real finding from a 100-item smoke test (`--n-items 100`, run after the skip-and-log fix above, 800 rows, 784 generated + 16 skipped, matching population).** Beyond the input-length skip rate, there is a real, well-measured **output-length parse-failure rate**, concentrated the same way:

| condition / turn | generated | parse failures | rate |
|---|---|---|---|
| clean / turn=1 | 348 | 0 | 0% |
| clean / turn=2 | 252 | 0 | 0% |
| verbose / turn=1 | 112 | 8 | 7.1% |
| verbose / turn=2 | 72 | 26 | **36.1%** |

**The mechanism is exact, not approximate:** every one of the 748 `finish_reason=="stop"` completions parsed successfully (0% failure - the ported `extract_pariwise_result()` and template are completely reliable whenever the model finishes naturally). Failures are 100% concentrated in `finish_reason=="length"` (truncated at `max_tokens=1024`): 34/36 truncated completions never reach a decision line. Truncation itself is sharply condition-dependent: 0% in `clean` (both turns), 8.0% in verbose/turn=1, **37.5% in verbose/turn=2** - verbose padding (especially compounded with turn=2's folded two-turn response fields) makes the model's own critique run long enough to frequently run out of room before concluding.

**Two of the truncated completions inspected by eye showed genuine model-level degenerate repetition, not just "ran out of room near a natural end":** one (`clean` sample, turn=1) starts with coherent, on-topic critique and spirals into token repetition partway through; two turn=2 verbose examples degenerate almost immediately (e.g. looping on "the Alps and the Alps and the Alps..." dozens of times from the first few tokens). This could be GPTQ-4bit's own documented "behavior might differ from the full model" caveat, the turn=2 flattening design's out-of-training-distribution structure, or `verbose_pad()`'s own repetitive-list structure priming a smaller/quantized model into continuing that same repetition - not yet distinguishable with the data collected so far, and not resolved here.

**Decision, confirmed with the owner (AskUserQuestion, 23 Sep 2026): keep `max_tokens=1024`, do not increase it, proceed to the full run.** The failure rate itself is treated as a real, reportable RQ7 finding, not an artifact to engineer away - the verbosity attack derailing auto-j-13b from producing any verdict at all more than a third of the time under the hardest combined condition (verbose + multi-turn) is arguably a more severe failure mode than anything RQ1-RQ3 or RQ6 found, and belongs in `REPORT.md`'s RQ7 section (L6) stated as prominently as the calibration/position-swap/verbosity-attack numbers themselves, not buried as a footnote. Rejected alternative: raising `max_tokens` to trade a higher input-skip rate for a currently-unmeasured, uncertain reduction in output-truncation failures - not worth the added complexity and GPU cost for an unproven benefit.

**Downstream handling (task L4, `src/autoj_signals.py`):** rows where `extract_autoj_verdict()` returns `-1` (no parseable decision line) get `judge_verdict=None`, `correct=None` - the same "missing, not zero, never imputed" convention `parse.py`'s own `parse_failure_type` taxonomy already established for the primary judge. This will materially shrink the effective N for verbose/turn=2's calibration and attack-battery numbers specifically at full-run scale - expected, and must be stated as its own limitation in `REPORT.md`, not silently absorbed into a smaller-but-unexplained sample size.

**Amended a third time, 23 Sep 2026 — turn=2 is dropped from this arm entirely, superseding point 4's "always both reported" locked policy above.** That policy was written to guard against one specific risk: deciding "keep vs. discard turn=2" *after* looking at whether its *substantive research results* (calibration, AUROC) looked good - outcome-dependent reporting. It explicitly pre-committed a *different*, outcome-independent criterion instead (parse-failure rate + spot-check, task L4b), to be decided *before* any calibration number existed. That check has now run, on real data (the 100-item smoke test above), strictly before any calibration/accuracy number for either turn was computed - so acting on its result is exactly what the policy was designed to do, not a violation of it.

**The honest trade-off, laid out before the owner's final call, not decided unilaterally:**
- `clean`/turn=2 was **perfectly reliable** (0% failure, identical to `clean`/turn=1) — this is a `verbose` × turn=2 *interaction*, not a defect in turn=2 on its own. Two of RQ7's four tests (calibration check, position-swap attack) use `clean` data only, mirroring RQ1/RQ3a's own recipes, and would have gotten turn=2 data just fine regardless of the verbose problem.
- Dropping turn=2 outright therefore also discards: (a) that otherwise-fine `clean`/turn=2 population for those two tests, and (b) the turn=1-vs-turn=2 verbosity-degradation *differential* itself (7.1% vs 36.1%) — arguably one of the more interesting findings already sitting in the smoke-test data, which disappears from `REPORT.md` entirely under this decision.
- A narrower alternative was considered and rejected: keep `clean`/turn=2 (free, reliable) and drop only `verbose`/turn=2 specifically. This preserves more research value but only saves ~12.6% of total calls (the `verbose`/turn=2 slice) versus turn=1-only's ~50%, for real added schedule/harness complexity (a condition-conditional-on-turn schedule, more branches in `analysis/rq7.py`'s population loaders) — not worth it given the remaining project timeline (~3 weeks to the mid-Oct deadline) and the number of debugging rounds this one arm has already taken.

**Decision, confirmed with the owner: turn=2 is dropped entirely, not merely caveated.** `src/judge_autoj.py::load_turn1_items_df()` is now the real population loader (turn=1 only, ~948 items); `build_autoj_prompt()` no longer takes a `turn` argument at all — the turn=2 flattening design (folding each side's own turn-1 answer into its own `{response}` field) is removed from the code, not kept as an unreachable branch, since dead code that still *looks* supported is a bigger footgun than a docstring pointer back to this decision. The finding that motivated this (the table and mechanism above) is preserved here in full, and must be stated in `REPORT.md`'s RQ7 section (L6) as the reason the population is turn=1-only, not silently presented as though turn=1-only were the plan from the start. `CLAUDE.md`'s RQ7 row, `PLAN.md` §8, and `TASKS.md`'s L-block are all updated accordingly — the 100-item smoke test's own turn=2 data stays in `runs/autoj_13b_gptq_4bits/` (harmless, unused, not worth the churn of scrubbing) but the real turn=1-only full run starts from that same checkpoint, which resumability already handles correctly (the ~50 turn=1 items already completed in the smoke test are automatically skipped, not redone).

**Amended a fourth time, same day — the turn=1-only decision immediately above is itself reversed, back to both turns, after an operational mistake changed the real cost-benefit math.** The full-run command given to the owner right after the "drop turn=2" decision was started on Colab *before* the turn=1-only fix had been pulled into that already-running process — `git pull` in a separate cell/terminal does not affect a Python process that already has the old module imported into memory, a basic behavior that should have been flagged explicitly at the time and wasn't. That process kept running on the OLD code (the full 1,904-item, both-turns population) for **7 hours** before the mismatch was caught (via a progress check showing turn=2 rows still growing, from 336 at the smoke-test snapshot to 4,116 and climbing). Confirmed no data-integrity problem alongside this - zero duplicate checkpoint keys - so the 7 hours produced real, valid, non-duplicated data; the actual problem was scope, not correctness.

**Recalculated trade-off, at that point 8,250/15,232 calls already done (~54%):** stopping and restarting turn=1-only needed ~3,450 more turn=1 calls (~2.9h at the observed rate); letting the original both-turns run finish needed ~6,982 more calls (~5.9h). The two paths had converged to within ~2.5-3 hours of each other, a much smaller gap than the ~50% difference that motivated dropping turn=2 in the first place, because most of the "expensive" both-turns generation was already sunk. **Decision, confirmed with the owner: let the original run finish (both turns), reverting the turn=1-only decision.** This is a forward-looking call, not sunk-cost reasoning: the marginal ~2.5-3 extra hours buys back real research value this decision's own trade-off analysis (above) already identified as worth having — `clean`/turn=2's perfectly-reliable data (2 of the 4 tests use `clean` only) and the turn=1-vs-turn=2 verbosity-degradation differential (7.1% vs 36.1%) — for a cost small relative to both the remaining project timeline and the time already spent debugging this one arm.

**Consequence for the code:** `src/judge_autoj.py`, `tests/test_judge_autoj.py`, and `configs/run_autoj.yaml` are reverted to their pre-turn1-only state (the commit immediately before the one that removed turn=2 support) - `build_autoj_prompt()` takes `turn` again, `load_turn1_items_df()` is gone, `__main__` uses `load_full_items_df()` again - specifically so that a `git pull` after a Colab disconnect reconnects to code that matches what the running process is actually doing, rather than silently truncating a resumed run to turn=1 only. This reversal is documented here in full rather than silently squashed into the original decision, per this project's own "supersede, don't delete" convention (D5/D9/D13/D14/D18's own precedent) - the back-and-forth is a real part of this arm's history, not something to hide from a defense.

**The turn=2 quality finding itself (36.1% verbose/turn=2 parse-failure rate, the exact mechanism) is unaffected by this reversal and remains a real result** - it will show up directly in the full run's own numbers once complete, and must still be stated prominently in `REPORT.md`'s RQ7 section (L6), same as if turn=1-only had never been the interim plan.

**Full run completed 24 Sep 2026: 15,232/15,232 calls, 1,904/1,904 unique items, 0 duplicate checkpoint keys.** Final, authoritative parse-failure rates (not the earlier partial-sample estimates): clean/turn=1 0.035%, clean/turn=2 0.035%, verbose/turn=1 13.6%, **verbose/turn=2 39.8%** - same pattern as every earlier sample, confirmed at full scale, coming in a bit higher than the smaller-N point estimates (expected - those carried wider uncertainty).

**Two more decisions confirmed with the owner during task L4's build (`src/autoj_signals.py`), before implementing either, not silently:**

1. **Tie handling.** auto-j's own template offers a third option ("Tie") the project's `verdict: A|B|null` schema (CLAUDE.md §3) was never built for. Checked against real data first (D12's own "measure, don't guess" standard): Tie occurs in 349/14,870 generated rows (2.35%) - small but real. Decision: `judge_verdict`/`verdict_bidir` return `None` for Tie, same as a genuine parse failure (neither is a usable binary verdict against `human_label`) - but Tie is tracked SEPARATELY from parse failure via a raw `pred_label` column (0=Response 1, 1=Response 2, 2=Tie, -1=no decision line found) kept in `calls_autoj_13b_gptq_4bits.parquet`, so the two are never silently conflated when the report needs to distinguish "the model explicitly said it couldn't decide" from "the model failed to answer at all."

2. **`conf_sc_bpe_autoj`'s scope, corrected before it was ever built.** D28's original text (above) said this signal should be null on `verbose`, mirroring `conf_sc`. Caught during design, before implementation: since `conf_sc_autoj` is *already*, unavoidably, null on `verbose` (no sampled draws exist there at all), nulling `conf_sc_bpe_autoj` too would leave **both** of RQ7's signals undefined on `verbose` - making the verbosity-attack test (one of RQ7's four core battery tests) impossible to run on anything. Fix, confirmed with the owner: `conf_sc_bpe_autoj` is computed on **both** conditions - its underlying self-consistency-proportion math is mechanically well-defined on `verbose` too (a coarser, single-greedy-call proportion instead of `clean`'s real k_sc-draw one), and this exactly mirrors D21's own already-established precedent for the primary judge: `conf_sc` alone is excluded from the verbosity-attack test there too, while `conf_bpe` (built from just two greedy calls) stays in. Not a new judgment call - an extension of one this project already made and approved. The coarser `verbose`-side resolution is a real, stated limitation for `REPORT.md`, not hidden.

**A real implementation bug caught via a hand-computed test case, before it shipped (task L4's own testing discipline doing its job):** `_p_model_a_wins_autoj`'s first draft copied `_p_model_a_wins()`/`_p_model_a_wins_kev()`'s `(p_ab + (1 - p_ba)) / 2` formula verbatim. But those two combine an *untranslated* `p_a`/`prob_a` (P(displayed-A wins), needing the `1 -` flip to become comparable across orders); `_self_consistency_proportion_a()` (this arm's own helper) already performs the canonical A/B translation internally, so its `p_ab`/`p_ba` outputs arrive pre-translated - applying the `1 -` flip again double-translates. Caught with a hand-computed pure-position-bias case (whichever response is displayed first always "wins," no real model preference): the correct order-corrected average must net to 0.5 (indifference); the buggy formula gave 1.0. Fixed to a plain average, `(p_ab + p_ba) / 2`, kept as a permanent regression test (`tests/test_autoj_signals.py`).

**L4 validated against the real, full dataset, not just synthetic tests:** `src/autoj_signals.py`'s CLI run against all 15,232 checkpoint rows produced 15,232 calls + 3,808 items (1,904 × 2 conditions). `conf_sc_autoj`: 0% null on `clean`, 100% null on `verbose` (exactly as designed). `verdict_bidir`, `flipped`, and `conf_sc_bpe_autoj` all show the *identical* null rate on `verbose` (35.98%) - a strong internal-consistency confirmation, since all three depend on exactly the same "both AB and BA produce a valid canonical letter" condition; `judge_verdict`'s own null rate (34.24%, AB alone) is correctly a bit lower. `any_skipped` (181 items) matches the 362 skipped calls ÷ ~2 orders/item exactly. `correct`: 77.2% (`clean`) / 71.9% (`verbose`) - plausible, same range as the primary judge's and kev-8b's own numbers. 28 new tests, `pytest` green (272 passed).

**L4 (K4-equivalent) done, 24 Sep 2026 — all four tests, both signals (where applicable), both turns.** `analysis/rq7.py`, mirroring `rq6.py`'s shape (population split by `turn`, not coverage regime; every function filters its own population to that signal's non-null rows immediately before computing, since auto-j's two signals have genuinely different null patterns from each other and from `judge_verdict`/`correct`, unlike kev's arm). A real bug was caught against real data, not synthetic tests: the verbosity-attack task's first run returned `delta_ECE=nan` because `load_rq7_paired_items()` filtered on `conf_sc_bpe_autoj.notna()` alone, and that does NOT imply `correct.notna()` (the signal only needs any one of AB's self-consistency draws to succeed; `correct` needs specifically the greedy call to succeed) - fixed by requiring both, confirmed clean afterward.

**Headline: calibration** (N=1,797 clean): both signals overconfident in both turns (gap 0.06–0.11), AUROC ~0.68 throughout - turn=1/turn=2 close in magnitude despite the sharp parse-failure-rate gap between them (D28's earlier finding), i.e. calibration quality among calls that DO succeed doesn't itself degrade much turn-to-turn. **Position-swap** (N=1,770): flip rate 12.5%/13.8% - notably *lower* than the primary judge's 27.8% and kev-8b's 23.7–24.6%, suggestive that auto-j's purpose-built training genuinely helps here; both signals track their own flips with a large, significant confidence gap (`conf_sc_autoj` −0.23 to −0.26, `conf_sc_bpe_autoj` −0.55 to −0.56). **Verbosity attack** (N=1,137 paired, `conf_sc_bpe_autoj` only per D21's precedent): ΔECE positive both turns (+0.042/+0.046, calibration trending worse under verbose) but neither CI excludes 0 at this N; ΔAUROC also not significant. **Bayesian recalibration**: meta-model AUROC 0.679/0.674 vs. best-single-signal 0.683/0.681 (essentially tied-to-slightly-worse, same qualitative pattern as kev-8b's own arm - two correlated signals, limited independent information to combine); convergence 0/50 flagged (turn=1), 3/50 flagged (turn=2) - a small, real uptick worth stating honestly in `REPORT.md`, not hidden. Full numbers in `results/rq7_{calibration,position_swap,verbosity,bayesian_recalibration}_autoj_13b_gptq_4bits.csv`.

**Amended 24 Sep 2026 (project audit) — the verbosity-attack result above was confounded and is superseded.** The pooled `conf_sc_bpe_autoj` pools 5 AB draws on `clean` but only 1 on `verbose` (no sampled draws exist there), so the paired clean-vs-verbose delta mixed the verbosity effect with a change in signal resolution. Fix: a new column, `conf_sc_bpe_autoj_greedy` (the same signal built from the two greedy calls only, on both conditions), used for the verbosity test alone; the calibration, position-swap, and Bayesian tests keep the pooled signal, since they never compare across conditions. Like-for-like result (N=1,137 paired): ΔECE −0.020 [−0.056, 0.018] (turn=1) / −0.019 [−0.058, 0.020] (turn=2), not significant; **ΔAUROC +0.063 [0.012, 0.115] / +0.070 [0.014, 0.128], both significant**; the turn=2 accuracy drop (−0.042) is unchanged, since it doesn't depend on the signal. The reading reverses: auto-j's order-swap signal is not broken by verbosity the way the other two judges' best signal was — it gets better at flagging errors, because under verbosity auto-j's errors are far more often AB/BA flips (47.6% of wrong verdicts on `verbose` vs. 28.4% on `clean`). `REPORT.md`'s RQ7 section is updated to this version, with the confounded one noted in its limitations.
