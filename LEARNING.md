# LEARNING.md — read / watch / build tracker

Three lists. Everything has a **week**, a **time cost**, and an **extraction target** — what you must be able to do afterwards. If you can't do the extraction target, the item isn't done.

Total: ~18h theory (was ~13h — 31 Aug 2026 professor feedback added Bayesian/NumPyro, hierarchical models, BALD, and distillation, `PLAN.md` §5) + ~5h courses ≈ **23h across 7 weeks**, interleaved with build work. Do not front-load it.

---

## A. TO READ — papers

### Core four (assigned; these are load-bearing)

**A1 · Guo et al. 2017, *On Calibration of Modern Neural Networks*** — §§1–4.2 only · 1.5h · **W0**
> **Extract:** derive ECE on a napkin as a weighted mean of |acc(bin) − conf(bin)|. State the Brier decomposition (reliability − resolution + uncertainty). Explain why temperature scaling cannot change accuracy (monotone in the logits ⇒ preserves the argmax).
> **Usually faked:** that ECE is a *biased estimator* whose bias grows with bin count; that equal-width bins fail when confidences pile up near 1.0; that in a balanced binary task a constant-0.5 predictor scores ECE = 0.

**A2 · Zheng et al. 2023, *Judging LLM-as-a-Judge (MT-Bench)*** — §§3, 4 + Tables 2, 4, 5 · 1.5h · **W0**
> **Memorise these numbers, you will be asked:** GPT-4 position-swap consistency **65%**; Claude-v1 **23.8%**; "repetitive list" verbosity attack fools Claude-v1/GPT-3.5 **91.3%** of the time, GPT-4 **8.7%**; GPT-4↔human agreement **85%** (non-tie), human↔human **81%**.
> **Extract:** the S1/S2 setup distinction (with vs without ties) and which one your primary analysis matches. Note that they *could not* establish self-enhancement bias in a controlled study — that's why your self-preference probe became an attribution probe, and (31 Aug 2026, D18) why that probe was later cut from scope entirely. The historical reasoning is still worth knowing; the probe itself didn't survive.

**A3 · Tian et al. 2023, *Just Ask for Calibration*** · 1h · **W1**
> **Extract:** why verbalized confidence beats conditional logprobs for RLHF'd models — and that the effect is **much weaker for Llama-2-70B-Chat**, i.e. do not assume verbalized wins on your open-weight judge.
> Footnote 5 is your ECE caveat. Quote it in the limitations slide.

**A4 · Xiong et al. 2023, *Can LLMs Express Their Uncertainty?*** · 1.5h · **W1**
> This is the study you are structurally reproducing: prompting × sampling × aggregation.
> **Extract:** their key negative result — prompting fixes ECE but **AUROC stays ~0.5–0.6**; GPT-4 averaged **62.7%**. This is your RQ2 prior. A flat risk–coverage curve is the *expected* outcome, not a failure.

### Modern context (skim; you need to be able to name and place these)

**A5 · [SCOPE](https://www.alphaxiv.org/abs/2602.13110), Feb 2026** · 45m · **W3** — closest paper to yours. Selective conformal pairwise judging, MT-Bench/RewardBench/Arena, Qwen-7B→Llama-70B.
> **Extract:** the **Bidirectional Preference Entropy** definition — judge both orders, average P(A), take entropy. You are implementing it as `conf_bpe`. Be able to explain in one sentence why a position-biased judge maxes it out.

**A6 · [Reliability without Validity](https://www.alphaxiv.org/abs/2606.19544), Jun 2026** · 30m · **W2** — 21 judges, chance-corrected agreement; exact-match overstates ability by **33–41pp** on MT-Bench.
> **Extract:** the sentence "consistency is not validity." This is why κ is mandatory in your repo.

**A7 · [Trust or Escalate](https://arxiv.org/abs/2407.18370), ICLR 2025** · 30m · **W3** — selective evaluation with human-agreement guarantees; Mistral-7B >80% agreement at ~80% coverage.
> **Extract:** that RQ2's *framing* is established prior work, and that your study measures the precondition it assumes. Say this before someone else does.

**A8 · [Dark Current](https://www.alphaxiv.org/abs/2606.15610), Jun 2026** · 30m · **W2** — psychometric judge datasheet; Llama-3.1-8B shows **96.7% positional false preference**.
> **Extract:** the "true vacuum" probe (judge two *identical* responses — does it still pick one?). You are running this in W1 as a sanity check.

**A9 · [Know When You're Wrong](https://arxiv.org/html/2603.06604), Mar 2026** · 20m · **W5** — error detection from normalised token probabilities, no trained classifier, general QA.
> **Extract:** why your RQ4 is not this paper (they don't train; they aren't judging). Also note the title collision with your deck and cite it.

**A10 · [Meta-Judges](https://arxiv.org/html/2504.17087v1), Apr 2025** · 20m · **W5** — rubric-based LLM meta-judging, trains nothing.
> **Extract:** the gap your RQ4 sits in — the field reaches for another LLM, not a cheap supervised model.

### RQ5 positioning (added 31 Aug 2026, professor feedback — D23)

**A11 · Auto-Prompt Ensemble for LLM Judge, Oct 2025** · 30m · **W1** — same Qwen2.5-7B/MT-Bench setup as this project. Closest prior work to RQ5.
> **Extract:** how they aggregate across prompt variants, and what's different about doing it through a Bayesian meta-model instead of a raw ensemble average — that gap is this project's contribution.

**A12 · Calibrating MLLM-as-a-Judge via Multimodal Bayesian Prompt Ensembles, ICCV 2025** · 30m · **W1**
> **Extract:** neither this paper nor A11 decomposes entropy into aleatoric/epistemic, and neither validates against real repeated human votes — that combination (D23) is RQ5's actual contribution. Be ready to say this precisely, not just cite the papers.

---

## B. TO WATCH — DeepLearning.AI

Audited against the catalog. Ranked by whether the project actually breaks without them. **Take the lesson-level cuts, not the full courses** — you don't have 7 hours to spare.

### Load-bearing — take these

**B1 · Getting Structured LLM Output** (DotTxt, 1h21m) — **take in full** · **W1**
> Lessons: intro to structured generation · how to use · retry-based · **structured generation with Outlines** · beyond JSON (regex → FSM).
> **Why it's load-bearing:** it teaches that constrained decoding works by *modifying logits per token*. That is not a detail — see the trap below. It also covers Pydantic schemas (which is what you hand to vLLM's guided decoding) and the `instructor` retry pattern, which is your fallback when constraints fail.
> **⚠️ THE TRAP THIS COURSE LETS YOU AVOID:** constrained decoding renormalises the token distribution. If you mask tokens and *then* read `verdict_token_logprob`, you are reading a probability under a modified distribution, not the model's belief. **Fix:** constrain the verdict to exactly the two tokens `{A, B}` and record the renormalised `p_a` over that pair. That is a clean, defensible restriction — and it is *exactly* the quantity BPE needs. Write this into your limitations section deliberately, not apologetically.

**B2 · Fast & Efficient LLM Inference with vLLM** (Red Hat, 1h38m) — **lessons 3, 6, 7, 8 only (~46m)** · **W1**
> L3 Inference & Memory Fundamentals · L6/L7 Serving with vLLM · L8 Benchmarking.
> **Why partial:** the course covers PagedAttention, continuous batching, prefix caching, quantization and benchmarking — good mental model for *why* batching turns 10 hours into 1. But it does **not** cover `SamplingParams(logprobs=...)`, offline batch `LLM.generate`, or guided decoding, which are the three APIs you actually need. Get those from the vLLM docs (30 min). Skip L1, L2, L4, L5, L9.

**B3 · Building Generative AI Applications with Gradio** (HF, 59m) — **take in full** · **W6**
> **Recommend switching the demo from Streamlit to Gradio purely on time grounds.** DL.AI's Streamlit course is 9h21m; this is 59m and lands you in the same place for a demo this simple.

### Worth it if the hours exist

**B4 · Evaluating AI Agents** (Arize, 2h36m) — **lessons 2, 11, 13 only (~18m)** · **W6, optional**
> L2 "Evaluation in the time of LLMs" · L11 "Adding structure to your evaluations" · **L13 "Improving your LLM-as-a-judge"**.
> Applied/observability framing, not research. Value is vocabulary for the "why this matters in practice" slide, and for talking to industry people about your project. Skip the agent-building labs entirely.

**B5 · Claude Code: A Highly Agentic Coding Assistant** (Anthropic, 2h) · **W0, optional**
> You're using Claude Code as your guide for this project. 2h now probably pays for itself. Pair with **Spec-Driven Development with Coding Agents** (JetBrains, 1h16m) if you want the workflow theory — feeding `CLAUDE.md` + `TASKS.md` to an agent *is* spec-driven development.

### Skip — and why

| Course | Why skip |
|---|---|
| Quantization Fundamentals with HF (1h14m) | Only needed for the laptop demo, and B2's L4/L5 already cover quantization conceptually. The 4-bit `bitsandbytes` path is ~10 lines; read the docs when you get there. |
| Pydantic for LLM Workflows (1h50m) | Overlaps B1, which covers Pydantic-for-structured-output in the part you need. |
| Fast LLM Inference with Cerebras (1h41m) | Hardware-specific API. Irrelevant to a Colab/vLLM stack. |
| Fast Prototyping of GenAI Apps with Streamlit (9h21m) | 9 hours for a demo you can build in Gradio in one. |
| Automated Testing for LLMOps (CircleCI) | Tempting given your SDET background, but it's CI-for-LLM-apps, not statistics. Your testing edge here is unit-testing metric code, which needs no course. |

---

## C. TO BUILD — theory blocks with a coding deliverable

Each block ends in code, not notes. If there's no artifact, it didn't happen.

**C1 · Calibration core** · 3h · **W0** · *(pairs with A1)*
> **Build:** `ece()` from scratch on toy data, both equal-width and equal-mass binning. Plot a reliability diagram for a synthetic overconfident predictor.
> **Prove you got it:** your `ece()` returns 0.222 on the reference case in `CLAUDE.md` §5.

**C2 · Agreement statistics** · 1.5h · **W0** · *(pairs with A2)*
> **Build:** `cohens_kappa()` by hand — p_o, p_e from independent marginals, κ = (p_o − p_e)/(1 − p_e). Verify against `sklearn.metrics.cohen_kappa_score`.
> **Extract:** work one balanced example (p_o = 0.85, p_e = 0.5 → κ = 0.70) and one skewed example. **Understand why κ deflates** — this is the single most-faked statistic in eval papers.

**C3 · LLM uncertainty signals** · 2h · **W1** · *(pairs with A3, A4)*
> **Build:** `signals.py` skeleton with all four estimators and their docstrings stating the formula.
> **Extract:** self-consistency is a Monte-Carlo estimate of the model's verdict distribution — so it ≈ the verdict logprob *when no reasoning precedes the verdict*. This is why you use a CoT judge prompt. **Expect to be asked this.** Also: temperature is a sampling parameter, not a confidence; raising it doesn't make the model less confident, it makes your estimate noisier.

**C4 · Discrimination vs calibration** · 1h · **W2**
> **Build:** a 6-point synthetic example where ECE ≈ 0.3 and AUROC = 1.0, and a second where ECE = 0 and AUROC = 0.5. Save both as a figure.
> **Extract:** AUROC = P(conf on a correct item > conf on an incorrect item), and it is **invariant to any monotone transform of confidence**. If you can say this in one sentence you can explain why RQ1 and RQ2 are different questions. **This is the most likely single question you will be asked.**

**C5 · Bootstrap** · 1h · **W2**
> **Build:** `cluster_bootstrap()` and `paired_cluster_bootstrap()` in `boot.py`, plus `test_boot.py` showing clustered CIs are wider than naive row CIs on the same data.
> **Usually faked:** resampling rows when rows are clustered inside questions. Your CIs come out too narrow and everything looks significant.

**C6 · Selective prediction** · 1.5h · **W3**
> **Build:** `risk_coverage()` + `aurc()`, and the **oracle curve** (rank by true correctness) as the achievable ceiling.
> **Extract:** why plotting the oracle alongside your signals makes a flat curve *interpretable* instead of embarrassing.

**C7 · Aleatoric vs epistemic** · 1h · **W3**
> **Build:** the human-disagreement split — judge error rate and judge confidence, stratified by human consensus strength.
> **Extract:** human disagreement on an item is aleatoric from the judge's point of view; judge-specific bias is epistemic. Using this vocabulary *correctly* is worth more than an extra experiment. Using it decoratively is worse than not using it.

**C8 · Conformal prediction** · 2h · **W6, optional**
> Split conformal, exchangeability, marginal vs conditional coverage. **Enough to defend the extensions slide and discuss SCOPE intelligently. Do not start implementing it.**

### Added 31 Aug 2026, professor feedback (D20, D22, D23) — PLAN.md §5 calls these Blocks I, J, K, L

**C9 · Bayesian inference + NumPyro/NUTS basics** · 2h · **W1** · *(Block I, pairs with task 1.3b)*
> **Build:** fit a toy hierarchical logistic regression in NumPyro on synthetic data with a known group effect; recover it from the posterior.
> **Extract:** NUTS samples the posterior over parameters via Hamiltonian dynamics, not a point estimate. A hierarchical model's group-level intercepts shrink toward the population mean (partial pooling) — say in one sentence why that's exactly the right tool for ~80 groups, not a workaround forced on you by a small N.

**C10 · Hierarchical / partial-pooling models, held-out marginalization** · 1h · **W5** · *(Block J, pairs with task 5.1b/5.9b)*
> **Build:** extend C9's toy model — fit on groups 1–8, hold out groups 9–10, and show that predicting on the held-out groups using their *own* (never-seen) fitted intercept gives a different, optimistic answer than drawing from the population prior (`α_q_new ~ Normal(0, σ_q)`), which is correct.
> **Extract:** why conditioning on a held-out group's own fitted intercept is a leak — structurally identical to `GroupKFold` leaking when it's not shuffled (D8). This is the single most likely way the Bayesian arm's numbers end up quietly too good.

**C11 · BALD / mutual information (entropy decomposition)** · 1h · **W3** · *(Block K, pairs with task 3.4c/2.2b)*
> **Build:** on a toy 3-member ensemble with known per-member probabilities, compute Total/Aleatoric/Epistemic by hand and verify Epistemic = Total − Aleatoric matches the mutual-information formula.
> **Extract:** epistemic uncertainty is the mutual information between the prediction and *which ensemble member (or parameter draw) produced it* — high when members disagree with each other, zero when they're all confidently identical. A position-biased judge maximizing `conf_bpe`'s entropy is the same phenomenon in miniature (SCOPE, A5).

**C12 · Distributional distillation** · 1h · **W5** · *(Block L, pairs with task 5.9d)*
> **Build:** compare a synthetic teacher ensemble's predictive spread (mean *and* variance) against a cheap single-draw student's own spread — a two-moment comparison, not a formal KD loss fit; this project doesn't need the latter.
> **Extract:** be precise about what "distillation" means in RQ5 specifically — whether the single-call Bayesian model's own posterior spread resembles the expensive ensemble's actual spread, not whether an LLM got fine-tuned (it doesn't). "Distillation" is used loosely across the literature; know exactly what you mean by it here.

---

## D. Progress tracker

| | Item | Week | Time | Done |
|---|---|---|---|---|
| A1 | Guo et al. §§1–4.2 | W0 | 1.5h | ☑ |
| A2 | MT-Bench §§3–4 + tables | W0 | 1.5h | ☑ |
| C1 | `ece()` + reliability diagram | W0 | 3h | ☑ |
| C2 | `cohens_kappa()` by hand | W0 | 1.5h | ☑ |
| B5 | Claude Code course *(optional)* | W0 | 2h | ☐ |
| B1 | Structured LLM Output | W1 | 1h21m | ☐ |
| B2 | vLLM L3/6/7/8 + vLLM docs | W1 | 1h15m | ☐ |
| A3 | Tian et al. | W1 | 1h | ☐ |
| A4 | Xiong et al. | W1 | 1.5h | ☐ |
| C3 | `signals.py` + docstrings | W1 | 2h | ☐ |
| **A11** | **Auto-Prompt Ensemble for LLM Judge** | **W1** | **30m** | ☐ |
| **A12** | **Bayesian Prompt Ensembles (ICCV 2025)** | **W1** | **30m** | ☐ |
| **C9** | **Bayesian inference + NumPyro/NUTS basics** | **W1** | **2h** | ☑ |
| A8 | Dark Current | W2 | 30m | ☑ |
| A6 | Reliability without Validity | W2 | 30m | ☑ |
| C4 | ECE⊥AUROC counterexample | W2 | 1h | ☑ |
| C5 | `boot.py` + `test_boot.py` | W2 | 1h | ☑ |
| A7 | Trust or Escalate | W3 | 30m | ☐ |
| A5 | SCOPE | W3 | 45m | ☐ |
| C6 | `risk_coverage()` + oracle | W3 | 1.5h | ☐ |
| C7 | human-disagreement split | W3 | 1h | ☐ |
| **C11** | **BALD / mutual information** | **W3** | **1h** | ☐ |
| A9 | Know When You're Wrong | W5 | 20m | ☐ |
| A10 | Meta-Judges | W5 | 20m | ☐ |
| **C10** | **Hierarchical models, held-out marginalization** | **W5** | **1h** | ☐ |
| **C12** | **Distributional distillation** | **W5** | **1h** | ☐ |
| B3 | Gradio | W6 | 59m | ☐ |
| B4 | Evaluating AI Agents L2/11/13 *(opt)* | W6 | 18m | ☐ |
| C8 | Conformal prediction *(opt)* | W6 | 2h | ☐ |
