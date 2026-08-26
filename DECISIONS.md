# DECISIONS.md — resolutions from the Week-0 design review

Fourteen decisions, D4–D17, resolving the review findings. Each is binding; copy the ones marked ⚑ into `PREREGISTRATION.md` before Gate 1.

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

**Two caveats to preregister, not discover:**

- **Top-k entropy is a biased estimator.** It systematically understates true entropy because the tail is truncated. `cot_entropy_mean` is "mean top-20 truncated entropy" and must be labelled that way in every table. Do not call it "predictive entropy."
- **CoT logprob statistics are conditioned on the sampled chain.** Each of the k samples has a different CoT and therefore different statistics. **Item-level aggregation rule:** take the value from the canonical call (`sample_idx=0`) as the primary feature, and the mean across the k sampled calls as a secondary feature. Store both; let the Tier C ablation tell you which carries signal.
- **⚠️ The two are not in the same unit.** The secondary feature is computed from the T=0.7 sampled calls, so the same temperature-scaling effect D6 found for `p_a` applies to it. Standardization means the model doesn't care, but *interpretation* does — a coefficient plot that treats them as one quantity is wrong.
  **The fix is naming, not a comment:** the columns are `cot_logprob_mean_greedy` and `cot_logprob_mean_sampled_t07` (and likewise for `min`, `std`, `p10`, `entropy`). Carrying the temperature in the name makes it structurally impossible for anyone — including you in October — to average them or read them as the same feature.

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

---

## D8 ⚑ — CV design at 80 groups *(review #4)*

**Correct, and the proposed fix needs two additions.** MT-Bench has 80 questions; effective N for fold-to-fold variance is ~80, not ~1000.

**Decision:**

1. **`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)`, repeated over 10 seeds.**
   ⚠️ **sklearn's `GroupKFold` has no `shuffle` parameter** — it is deterministic, so "repeat across seeds" silently produces ten identical splits. Use `StratifiedGroupKFold`, which supports `shuffle`. This is a real trap and the test in 5.3 must assert that two different seeds produce different fold assignments.
2. **Report the across-repeat spread as the headline uncertainty**, not the within-split CI. The honest number is "AUROC 0.64, 10×5-fold spread [0.58, 0.69]".
3. **Drop hyperparameter tuning entirely.** Fix `C=1.0` on standardized features and preregister it. At 80 groups the nested inner CV subdivides 64 training groups into folds of 16, which is noise, and tuning adds a researcher degree of freedom you cannot afford to defend. For `HistGradientBoostingClassifier`, fix `max_depth=3, max_iter=200, learning_rate=0.05` and preregister those too. This removes the nested CV from task 5.3.

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

