# CLAUDE.md — repo constitution

Read this before doing anything in this repo. It is the contract, not a summary.

**Also read `DECISIONS.md`** — D4–D28 (the Week-0 design review, the 31 Aug 2026 professor-feedback integration, and the owner-initiated RQ6/RQ7 additions) override anything here that contradicts them.

---

## 1. What this project is

An empirical study of **whether small open-weight LLM judges know when they are wrong**, and whether their uncertainty is usable for selective evaluation (abstain-and-escalate).

- **Primary judge model:** `Qwen2.5-7B-Instruct` (config key, never hardcoded). RQ6 and RQ7 add `kev-8b` and `auto-j-13b`.
- **Data:** `lmsys/mt_bench_human_judgments`, `human` split (3,355 rows = one row per *human vote*, not per item)
- **Compute:** Google Colab Pro (L4/A100 preferred), vLLM offline batch inference
- **Deadline:** mid-October 2026. This is a certification capstone, not a paper. Working and honest beats clever.

### The research questions

| RQ | Question | Primary metric |
|----|----------|----------------|
| RQ1 | Is the judge's stated confidence calibrated? | ECE (equal-mass bins), Brier + decomposition, signed overconfidence gap |
| RQ2 | Is any cheap uncertainty signal *informative* about error? | AUROC(uncertainty → error), risk–coverage curve, AURC |
| RQ3 | Does uncertainty flag bias-induced errors, or is the fooled judge confident? | flip rate, mean confidence on flipped vs unflipped, paired ΔECE/ΔAUROC |
| RQ4 | Can a cheap supervised meta-model beat the best single signal at predicting judge error? | AUROC under repeated `StratifiedGroupKFold`, vs best-single-signal baseline and permutation null |
| RQ5 | Does marginalizing over the judge prompt (an ensemble of 3 variants) improve uncertainty quality, and does that improvement survive distillation to single-call cost? | Ensemble vs single-call AUROC/ECE/entropy-quality gap; aleatoric-vs-`d_human` validation |
| RQ6 | Does kev-8b — an open, RLCD-trained stand-in for the industry's "System One" calibration counterclaim (Jev/TypeSafe) — actually resist the same failure modes (position bias, verbosity degradation, exploitable residual structure) this project already found in a general-purpose LLM judge? | ECE/overconfidence gap, flip rate, ΔECE/ΔAUROC under verbosity, meta-model-vs-best-single-signal comparison (D22's Bayesian machinery reused) |
| RQ7 | Does auto-j-13b — a model purpose-trained via GPT-4-distilled critique+verdict data specifically for pairwise response judging — still exhibit the same position-bias and verbosity-degradation failure modes found for a general-purpose judge (RQ1–RQ3) and an out-of-domain industry stand-in (RQ6)? | ECE/overconfidence gap, flip rate, ΔECE/ΔAUROC under verbosity, meta-model-vs-best-single-signal comparison, each split by turn=1/turn=2 |

RQ4 is the owner's own idea and is **not** optional. Protect it (`PLAN.md` §2). RQ5 came from professor feedback on 31 Aug 2026 (`DECISIONS.md` D18–D24, `PLAN.md` §6) and is equally binding.

RQ6 and RQ7 are owner-initiated additions (`DECISIONS.md` D27/D28, `PLAN.md` §7–8), tracked in `TASKS.md`'s K- and L-blocks outside the W0–W7 numbering so they can be trimmed later without renumbering anything. Each evaluates a different judge model on the same MT-Bench items. Neither model has a prompt-ensemble axis, so invariant 14's `prompt_variant` filter does not apply to them.
- **RQ6** splits its population by kev-8b's training coverage (`input_tokens` ≤ 1,024 vs above) and skips the 52 verbose items over kev's 8,160-token serving ceiling.
- **RQ7** splits its population by `turn`. It skips the 4.75% of calls whose prompt exceeds auto-j's context (almost all verbose/turn=2), and auto-j produces no verdict at all on 39.8% of verbose/turn=2 calls (output truncated at `max_tokens`). That failure rate must be reported prominently wherever RQ7 is discussed.

---

## 2. Hard invariants

Violating any of these silently corrupts a result. Never do it, never suggest it, and flag it if you find it in existing code.

**Statistics**

1. **Never use plain `KFold` or `train_test_split`.** Items cluster inside MT-Bench `question_id` (same prompt, often same responses). Always group by `question_id`.
   ⚠️ **Use `StratifiedGroupKFold(shuffle=True, random_state=seed)`, not `GroupKFold`.** sklearn's `GroupKFold` has **no `shuffle` parameter** — it is deterministic, so repeating it across seeds silently produces identical splits and a fake stability result. There are only **80 questions**, so effective N for fold-to-fold variance is ~80, not ~1000: repeat 5-fold across **10 seeds** and report the spread as the headline uncertainty. See D8.
2. **Never bootstrap by resampling rows.** Cluster-bootstrap: resample `question_id` values with replacement, then take all rows belonging to the sampled questions. CIs from row-resampling are too narrow and will make noise look significant.
3. **Condition comparisons are paired.** Same items appear in `clean`, `verbose`. Compare with a paired cluster-bootstrap on the per-item difference, never as two independent samples. Order (AB vs BA) is likewise paired within a condition. A paired comparison must also score the *same signal construction* on both sides — e.g. RQ7's verbosity test uses the greedy-only `conf_sc_bpe_autoj_greedy`, because `verbose` has no sampled draws to pool.
4. **ECE uses equal-mass (quantile) bins, not equal-width** — *unless the signal is discrete.* Verbalized confidence piles up at 0.8/0.9/0.95/1.0; equal-width bins leave most bins empty and the number becomes meaningless.
   **`ece()` selects its own strategy and returns `(ece, n_effective_bins)`:** if `n_unique(conf) <= n_bins`, bin by **unique value** — for a discrete signal that is the *exact* ECE, not a fallback; otherwise quantile bins with duplicate edges dropped. `conf_sc` has only 5 possible values at `k_sc=4`, and naive `qcut` either raises on tied edges or silently returns fewer bins than asked. Always emit the reliability diagram alongside the scalar, and **always print `n_effective_bins`**. See D14.
5. **Never report accuracy without Cohen's κ.** Raw agreement overstates judge ability by 33–41pp on MT-Bench (Reliability without Validity, 2026). This applies on the risk–coverage curve too: plot **κ@coverage** as well as accuracy@coverage, because dropping items shifts the base rate and inflates accuracy for free.
6. **Never report ECE alone.** In a binary forced choice confidence is bounded below at 0.5, so a judge that always says 50% has perfect ECE and zero usefulness. ECE always ships with accuracy, AUROC, and the signed overconfidence gap (mean confidence − accuracy).

**Data flow**

7. **Analysis code never touches raw model output.** `calls.parquet` (one row per model call) → `items.parquet` (one row per item×condition) → analysis. Parsing lives only in `src/parse.py` (for RQ6/RQ7, in `src/kev_signals.py` / `src/autoj_signals.py`, each touching raw output in exactly one function). If an analysis function references `raw_output`, it is in the wrong file.
8. **Every call row carries provenance:** `judge_model`, `prompt_hash`, `git_sha`, `seed`, `condition`, `order`, `sample_idx`. This is what makes "reproducible" a true claim. The one exception is the kev-8b arm: kev has no prompt template and no sampling, so its rows carry no `prompt_hash`/`seed`/`sample_idx`.
9. **All runs are resumable.** Append to JSONL after every batch, keyed by `(item_id, condition, order, sample_idx)`. A re-run skips completed keys. Colab disconnects; design for it from the first line. `git pull` does not change an already-running process — restart it to pick up new code.
10. **Prompt templates are frozen and hashed.** After Week 1, changing a template means a new version string and a full re-run. Ask before touching `src/prompts.py`.

**Modelling**

11. **No neural nets for RQ4, and no hyperparameter tuning.** N ≈ 1000 rows but only ~80 groups. The permitted model set is `LogisticRegression` (frequentist baseline, interpretable), `HistGradientBoostingClassifier` (nonlinearity check), and the Bayesian hierarchical logistic regression with question-level random intercepts (D22 — partial pooling is exactly the standard tool for this few-groups problem, not an exception to the invariant's logic). **Hyperparameters are fixed and preregistered** — `LogisticRegression(C=1.0)`; `HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)`. No nested CV, no tuning, for any of the three. At 80 groups, tuning is noise plus a researcher degree of freedom you cannot defend. **LogReg and the Bayesian model both see features standardized on the training fold only** — `β ~ Normal(0, 1)` is only a comparable prior across features that share a scale. See D8, D22.
12. **Every RQ4 result ships with a permutation null.** Shuffle `correct` **within each `question_id`** (one fresh permutation per replicate), re-run the same grouped CV, report the null AUROC distribution and the observed AUROC's percentile in it. Within-question, not global: a global shuffle also destroys the question-level clustering of `correct` and understates how high a no-information model can score (D8 amendment). Without it, 0.62 is not distinguishable from noise.
13. **Every Bayesian RQ4 result ships with convergence diagnostics** — R-hat, effective sample size, divergence count — with the same never-optional status as the permutation null in #12. **Held-out random intercepts are marginalized over the population-level prior** (`α_q_new ~ Normal(0, σ_q)`), never the value a held-out question would have fitted to — getting this wrong leaks exactly the way plain `GroupKFold` leaks. See D22.

**Schema**

14. **Once `prompt_variant` exists as a column, filter to `prompt_variant == "P1"` before any RQ1–RQ4 analysis reads `items.parquet`.** The table's grain became `(item_id, condition, prompt_variant)` when the P1/P2/P3 ensemble was added (D19, D20); P2/P3 rows will silently inflate every sample size and corrupt every RQ1–RQ4 headline number otherwise. This is the single easiest thing to get wrong in the whole prompt-ensemble addition.
   ⚠️ **`prompt_variant == "P1"` alone is not enough for RQ1, RQ2, or RQ4's core analysis.** `items.parquet` holds `(clean, P1)` *and* `(verbose, P1)` rows — filtering on `prompt_variant` alone silently mixes the two. RQ1, RQ2, and RQ4's core tiers additionally need `condition == "clean"`. RQ3 and the transfer tests (5.7) are the deliberate exceptions that read `verbose` too, and each is scoped explicitly where that happens.

---

## 3. Data schemas

**Paths below are logical names (D26).** Every file that depends on which judge model produced it carries a `_{model_slug}` suffix (`Config.model_slug`, e.g. `calls_qwen2.5_7b_instruct.parquet`), and `runs/` is namespaced by model as a subfolder (`runs/{model_slug}/`). The one exception is `results/items_labels.parquet`, never suffixed — it's built from human votes alone and doesn't depend on the judge model. The RQ6/RQ7 tables (`calls_kev_8b`, `items_autoj_13b_gptq_4bits`, ...) have their own, smaller schemas, documented in `src/kev_signals.py` and `src/autoj_signals.py`.

### `results/calls.parquet` — one row per model call
```
item_id             str    stable hash of (question_id, model_a, model_b, turn)
question_id         int    MT-Bench question id  ← THE GROUPING KEY
category            str    writing|roleplay|reasoning|math|coding|extraction|stem|humanities
model_a, model_b    str
turn                int    1 or 2
condition           str    clean|verbose|vacuum   ← NO 'swap', NO 'attribution'. See D5, D18.
prompt_variant      str    P1|P2|P3   ← orthogonal axis, like order. P1 is primary (RQ1-RQ4
                           use it alone); P2/P3 only exist for condition=clean. See D19.
order               str    AB|BA  ← orthogonal to condition; collected for EVERY condition
sample_idx          int    0 = canonical greedy (T=0); 1..k_sc = sampled (T=0.7), clean/P1 ONLY.
                           See D6, D19, D21.
seed                int    per-call = config.seed + sample_idx
judge_model         str
vllm_version        str    pinned exactly; see D11
prompt_hash         str
git_sha             str
raw_output          str
parse_ok            bool
parse_failure_type  str    none|no_verdict|no_confidence|malformed_json|truncated
verdict             str    A|B|null
verbalized_conf     float  [0,1] or null
reasoning_len       int    character length of the "reasoning" field's value alone (not the
                           JSON wrapper); approximate on the rare malformed rows, null only when
                           no reasoning value can be located at all
verdict_token_logprob float ┐
p_a                 float  │ ⚠ p_a VALID ONLY WHEN sample_idx == 0 — temperature scales
                           │ the reported logprobs (D6). Every column on this brace is
cot_logprob_mean    float  │ computed in src/parse.py from the full per-token logprobs
cot_logprob_min     float  │ saved for every call (D4).
cot_logprob_std     float  │
cot_logprob_p10     float  │
cot_entropy_mean    float  │
n_cot_tokens        int    ┘
n_prompt_tokens     int
n_out_tokens        int
latency_ms          float  batch-averaged, not true per-request latency
```

A raw per-token logprob file goes to `runs/logprobs/*.jsonl.gz` (gzipped) for **every** call (D4). `judge.py` only ever writes `raw_output` plus provenance to its checkpoint — it computes no derived signal itself.

### `results/items.parquet` — one row per (item_id, condition, prompt_variant)
⚠ **Filter to `prompt_variant == "P1"` before any RQ1-RQ4 analysis** — see invariant 14.
```
item_id, question_id, category, condition, prompt_variant, turn
human_label         str    A|B          (majority over non-tie human votes)
n_human_votes       int
frac_prefer_a       float  soft label — the differentiator lives here
human_unanimous     bool
human_agreed        bool   human_unanimous AND n_human_votes >= 2 — a single-vote item
                           is trivially unanimous and must not count (D16)
d_human             float  |frac_prefer_a - 0.5|  ← continuous consensus strength (D9)
judge_verdict       str    A|B   PRIMARY: canonical greedy verdict in AB order (D7)
correct             bool   judge_verdict == human_label
verdict_bidir       str    A|B   SECONDARY: argmax of mean p_a across both orders (D7)
correct_bidir       bool
conf_verb           float  from sample_idx=0
conf_lp             float  from sample_idx=0 ONLY (D6)
conf_sc             float  fraction of the k_sc sampled verdicts matching the canonical
                           greedy verdict. clean/P1 ONLY - null elsewhere (D19, D21)
conf_bpe            float  1 - entropy of mean p_a across both orders, WITHIN this
                           (condition, prompt_variant) pair (SCOPE, 2026)
conf_ens            float  1 - ens_entropy_total. clean ONLY - null for verbose,
                           which never collects P2/P3 (D20, D21)
ens_entropy_total       float  H[mean p_a across P1,P2,P3]. clean ONLY (D20)
ens_entropy_aleatoric   float  mean(H[p_a]) across P1,P2,P3. clean ONLY (D20)
ens_entropy_epistemic   float  total - aleatoric (BALD / mutual information). clean ONLY (D20)
flipped             bool   canonical verdict differs between order AB and BA,
                           within this (condition, prompt_variant) pair
                           ── RQ4 Tier B ──
len_a, len_b        int    character length of each side's judged response
len_ratio           float  len_a / len_b; null when len_b == 0
abs_len_diff        int
longer_is_chosen    bool   judge picked the strictly longer side; null on equal lengths
judge_output_len    int    reasoning_len of the canonical AB greedy call
                           ── RQ4 Tier C ──
verdict_margin      float  |2*p_a - 1| on the canonical AB greedy call
cot_*_greedy        the six CoT aggregate columns of the AB greedy call
cot_*_sampled_t07   the same six, averaged over the k_sc sampled calls; clean/P1 only
```

---

## 4. Layout

```
configs/     run.yaml (primary judge)   run_kev.yaml (RQ6)   run_autoj.yaml (RQ7)
src/
  config.py       Config dataclass, YAML loader
  data.py         MT-Bench votes → item table; tie policy; human-human κ
  prompts.py      P1/P2/P3 judge templates, each versioned + hashed   # FROZEN (invariant 10)
  perturb.py      verbose_pad, vacuum inputs
  judge.py        vLLM wrapper for the primary judge: batched, checkpointed, resumable
  parse.py        verdict/confidence extraction, failure taxonomy, logprob signals → calls.parquet
  signals.py      calls → items: conf_verb/lp/sc/bpe/ens + entropy decomposition, Tier B/C columns
  features.py     RQ4 feature tiers A / B / C
  predictor.py    RQ4: repeated StratifiedGroupKFold, LogReg + HistGBM, permutation null
  bayesian.py     RQ4: hierarchical logistic regression (NumPyro/NUTS), convergence
                  diagnostics, meta-model-level entropy decomposition (D22)
  metrics.py      ece, mce, brier+decomposition, auroc, kappa, risk_coverage, aurc, threshold_sweep
  boot.py         cluster_bootstrap, paired_cluster_bootstrap
  plots.py        one figure per function
  vacuum_test.py        task 1.8's identical/empty-pair probe (Colab)
  ablation_decoding.py  task 4.5's constrained vs free-form decoding ablation (Colab)
  judge_kev.py, kev_signals.py        RQ6: kev-8b HTTP client; checkpoint → calls/items
  judge_autoj.py, autoj_signals.py    RQ7: auto-j-13b vLLM wrapper; checkpoint → calls/items
analysis/    rq1.py … rq7.py (one per RQ), human_disagreement.py (tasks 3.3/3.4), vacuum.py (task 1.8),
             decoding_ablation.py (task 4.5) — each a CLI over items.parquet
tests/       one test file per src/ module. analysis/ scripts are verified against
             real data rather than unit-tested.
learning/    study exercises, not part of the pipeline
runs/        per-model checkpoints and logprobs/ (gitignored)
results/     parquet tables, per-RQ CSVs, figures/ (gitignored)
pyproject.toml   pinned deps; base install excludes vllm (`colab` extra adds it, D17)
                 but includes numpyro/jax/arviz (D24)
PLAN.md      design rationale, RQ definitions, week plan
TASKS.md     atomic tasks with definition-of-done and closeout notes
DECISIONS.md resolutions from the design reviews, D4–D28
PREREGISTRATION.md   frozen before the Week 2 full run
REPORT.md    written incrementally, not at the end
LEARNING.md  reading / courses / skills tracker
```

---

## 5. Conventions

- Python 3.11+. Type hints on every public function. No notebooks in `src/` — notebooks call into `src/`.
- Config via YAML → dataclass. **No hardcoded model names, paths, thresholds, or k values anywhere in `src/`.**
- Plots: matplotlib only, one figure per function, always return the `Figure`, always save to `results/figures/` with a deterministic name.
- Logging via `logging`, not `print`, except in CLI entrypoints.
- Randomness: every stochastic function takes an explicit `seed` or `rng`. No bare `np.random.*`.
- Comments and docstrings say what the code does and why a non-obvious choice was made. Dates, task numbers, and the history of how a decision was reached belong in `DECISIONS.md`/`TASKS.md`, not in code.

### Testing — this is the owner's edge, use it

The owner has six years as an SDET. Test code is expected to be good, and it is a deliverable, not overhead.

- `test_metrics.py` must check `ece()` against a **hand-computed** example. Reference case: two bins, (conf 0.9, acc 0.75, weight 0.4) and (conf 0.6, acc 0.33, weight 0.6) → ECE = 0.4·0.15 + 0.6·0.27 = **0.222**.
- `test_data.py` must check `build_items()`'s tie-policy branching against small hand-constructed vote groups (tie-dominant, clean majority, unanimous, single-vote, and the strict-vs-lenient boundary case that motivated D1), and `human_human_kappa()` against a hand-computed κ **and** a determinism check that row order doesn't change which votes get paired.
- `test_parse.py` uses hand-built malformed outputs — no malformed output occurred naturally in the real runs. Add any real failure mode seen later as a fixture.
- `test_perturb.py` property-tests `verbose_pad()`: it preserves the verdict-relevant content. **The `order` round-trip property (rendering AB then BA returns the original assignment) belongs in `test_prompts.py`, not here — order is a renderer concern, not a perturbation (D5).**
- `test_predictor.py` must assert (a) no `question_id` appears in both train and test of any fold, and (b) two different seeds produce **different** fold assignments — the `GroupKFold`-has-no-shuffle trap in §2.1.
- `test_bayesian.py` must assert a held-out question's random intercept is marginalized over the population prior, never its would-be fitted value (D22), and that predictions don't change when a feature is rescaled.
- `test_boot.py` checks that the cluster bootstrap produces wider intervals than a naive row bootstrap on the same data. If it doesn't, the grouping is broken.

Run `pytest` before any commit that touches `src/metrics.py`, `src/boot.py`, `src/predictor.py`, `src/bayesian.py`, or `src/data.py`.

---

## 6. Working with the owner

- He is a beginner in ML research and an expert in test engineering. Explain statistical choices; do not explain software engineering.
- **He is learning this material as he builds it.** When you implement something from `metrics.py` or `predictor.py`, add a docstring that states the formula and why this variant was chosen over the obvious alternative. Those docstrings are study material.
- Do not silently "improve" a method. If you think an invariant in §2 is wrong for a specific case, say so and wait.
- If a task in `TASKS.md` is ambiguous, ask rather than guessing — a wrong statistical choice here is invisible until the viva.
- Prefer boring, explicit code. This will be read aloud by someone defending it.

---

## 7. Commands

```bash
pytest                                                             # always green before commit

# Primary judge (configs/run.yaml)
python -m src.data      --config configs/run.yaml                  # items_labels.parquet
python -m src.judge     --config configs/run.yaml --condition clean    # Colab; then --condition verbose
python -m src.parse     --config configs/run.yaml                  # checkpoints → calls.parquet
python -m src.signals   --config configs/run.yaml                  # calls → items.parquet
python -m analysis.rq1  --config configs/run.yaml                  # likewise rq2, rq3, human_disagreement, vacuum, decoding_ablation
python -m analysis.rq4  --config configs/run.yaml --task {ablation,h4,transfer,category,calibration,bayesian_comparison,verbose_shift}
python -m analysis.rq5  --config configs/run.yaml --task {threshold_sweep,distillation,human_disagreement}

# RQ6 — kev-8b (needs kev.serve running, D27)
python -m src.judge_kev    --config configs/run_kev.yaml
python -m src.kev_signals  --config configs/run_kev.yaml
python -m analysis.rq6     --config configs/run_kev.yaml --task {calibration,position_swap,verbosity,bayesian_recalibration}

# RQ7 — auto-j-13b (Colab)
python -m src.judge_autoj   --config configs/run_autoj.yaml
python -m src.autoj_signals --config configs/run_autoj.yaml
python -m analysis.rq7      --config configs/run_autoj.yaml --task {calibration,position_swap,verbosity,bayesian_recalibration}
```

---

## 8. Environment & workflow

- **Local (VSCode, dedicated conda env `judge-calib`, python 3.11):** everything except running the judge models — writing and testing all of `src/`, all analysis, the Gradio demo. `pip install -e .` here never installs `vllm`.
- **Colab (fresh `venv`, GPU):** the only place `judge.py`/`judge_autoj.py` actually run. `pip install -e ".[colab]"` inside the fresh venv (D11) — never Colab's system Python. The Bayesian model (`bayesian.py`) is analysis, not inference — it runs locally (D24).
- **A GitHub remote** carries code between the two: commit and push locally, `git clone`/`git pull` in Colab.
- **`runs/` and `results/` are gitignored on purpose** — move generated data (checkpoints, `calls.parquet`) back from Colab via a Drive-mounted folder or direct download, never through git.
- See D17 for the full reasoning.

## 9. Current status

RQ1–RQ7 are complete and written up in `REPORT.md`. GATE 5, GATE K, and GATE L are left unchecked in `TASKS.md` pending the owner's review. Remaining: Week 6 (Gradio demo, finishing `REPORT.md`'s framing sections, slides) and Week 7 (fresh-clone reproducibility check, `REPRODUCE.md`).
