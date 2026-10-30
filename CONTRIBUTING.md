# Contributing

How to work in this repo, whether by hand or with any tool. The rules in §2 are
the contract, and `DECISIONS.md` gives the reasoning behind each one. When those two files disagree,
`DECISIONS.md` wins.

## 1. The project

The question is whether small open-weight LLM judges know when they are wrong, and
whether their uncertainty can drive abstain-and-escalate evaluation.

- **Judges:**
  - Qwen2.5-7B-Instruct, the primary judge (RQ1–RQ5);
  - kev-8b (RQ6);
  - auto-j-13b, GPTQ 4-bit (RQ7).
- **Data:** `lmsys/mt_bench_human_judgments`, the `human` split. There are 80
  questions and 1,904 non-tie items.
- **Status:** every analysis, `REPORT.md`, the site and the reproducibility check are
  done. The development-phase planning documents are kept at the git tag
  `dev-phase-end`.

| RQ | Question | Main metrics |
|----|----------|--------------|
| RQ1 | Is stated confidence calibrated? | ECE (equal-mass), Brier + decomposition, overconfidence gap |
| RQ2 | Does any cheap signal flag errors? | AUROC, risk–coverage, AURC, κ@coverage |
| RQ3 | Does uncertainty flag errors caused by order or padding? | flip rate, confidence on flipped vs not, paired ΔECE/ΔAUROC |
| RQ4 | Does a meta-model beat the best single signal? | AUROC under repeated grouped CV, vs. best signal and a permutation null |
| RQ5 | Does a 3-prompt ensemble help, and does it survive distillation to one call? | ensemble vs. one-call AUROC/ECE; aleatoric vs. `d_human` |
| RQ6 | Does kev-8b show the same failures? Split by `input_tokens` ≤ 1,024 | RQ1/RQ3/RQ4 tests |
| RQ7 | Does auto-j-13b show the same failures? Split by turn | RQ1/RQ3/RQ4 tests |

Report auto-j's verbose/turn-2 failure prominently whenever RQ7 comes up: 39.8% of
calls produce no verdict, and 4.75% of all calls are skipped as over-length.

## 2. Invariants

Breaking any of these corrupts a result without any warning. Never do it, and flag it
if you find it in existing code.

**Statistics**

1. **Group every split by `question_id`.** Never use `KFold` or `train_test_split`.
   Use `StratifiedGroupKFold(shuffle=True, random_state=seed)`: 5 folds, repeated over
   10 seeds. **`GroupKFold` has no shuffle**, so repeating it over seeds gives
   identical splits. The headline uncertainty is the across-repeat spread (D8).
2. **Bootstrap by cluster.** Resample `question_id` values, never rows.
3. **Comparisons between conditions are paired.** Use a paired cluster bootstrap on
   the per-item difference, with the *same signal construction* on both sides.
4. **ECE uses equal-mass bins**, or bins by unique value when the signal is discrete.
   `ece()` chooses between them and returns `(ece, n_effective_bins)`; always print
   both, along with the reliability diagram (D14).
5. **Never report accuracy without Cohen's κ**, and that includes κ@coverage on
   risk–coverage curves.
6. **Never report ECE alone.** It always comes with accuracy, AUROC and the signed
   overconfidence gap. Calibration metrics need a probability in the verdict being
   scored. Entropy signals (1 − H) use their `*_prob` form
   (`signals.py::prob_on_verdict`). A confidence in the AB verdict is never scored
   against `verdict_bidir` (D7).

**Data flow**

7. **Analysis code never reads raw model output.** The path is `calls.parquet` →
   `items.parquet` → analysis. Parsing happens only in `src/parse.py`,
   `src/kev_signals.py` and `src/autoj_signals.py`.
8. **Every call row carries its provenance:** `judge_model`, `prompt_hash`, `git_sha`,
   `seed`, `condition`, `order`, `sample_idx`. kev rows have no prompt, seed or
   sample.
9. **Every run can resume.** Each run appends to JSONL, keyed by
   `(item_id, condition, order, sample_idx)`, and skips keys already done. `git pull`
   does not update a process that is already running.
10. **Prompt templates are frozen and hashed.** Changing one means a new version and a
    full re-run.

**Modelling**

11. **RQ4 allows exactly three models:**
    - `LogisticRegression(C=1.0)`;
    - `HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05)`;
    - the Bayesian hierarchical logistic regression (D22).

    No neural nets, no tuning and no nested CV. Features are standardized on the
    training fold only.
12. **Every RQ4 result ships with a permutation null.** It shuffles `correct` *within*
    each `question_id` and re-runs the same grouped CV.
13. **Every Bayesian result ships with R-hat, ESS and the divergence count.** A
    held-out question's intercept is marginalized over the population prior, never
    taken from a fitted value (D22).

**Schema**

14. **Filter `items.parquet` to `prompt_variant == "P1"` before any RQ1–RQ4
    analysis**, and also to `condition == "clean"` for RQ1, RQ2 and RQ4's core tiers.
    RQ3 and the transfer tests read `verbose` deliberately (D20).

## 3. Data

**Paths.** Every file that depends on the judge model carries `_{model_slug}`, e.g.
`calls_qwen2.5_7b_instruct.parquet`. `runs/{model_slug}/` holds the checkpoints. The
exception is `results/items_labels.parquet`, which is built from human votes only
(D26). The kev and auto-j tables are documented in `src/kev_signals.py` and
`src/autoj_signals.py`.

**`calls_*.parquet`** has one row per model call:

- **Identity:** `item_id`, `question_id` (the grouping key), `category`, `model_a`,
  `model_b`, `turn`.
- **Axes:**
  - `condition`: `clean`, `verbose` or `vacuum`;
  - `prompt_variant`: P1–P3; P2 and P3 exist for `clean` only;
  - `order`: AB or BA;
  - `sample_idx`: 0 is greedy; 1–4 are sampled at T=0.7, for `clean`/P1 only.
- **Provenance:** `seed`, `judge_model`, `vllm_version`, `prompt_hash`, `git_sha`.
- **Output:** `raw_output`, `parse_ok`, `parse_failure_type` (`none`, `no_verdict`,
  `no_confidence`, `malformed_json` or `truncated`), `verdict`, `verbalized_conf`,
  `reasoning_len`.
- **Logprobs:** `verdict_token_logprob`, `p_a`, and the six `cot_*` statistics. All
  of them are valid only when `sample_idx == 0` (D6).
- **Counts and timing:** `n_prompt_tokens`, `n_out_tokens`, `latency_ms`.

**`items_*.parquet`** has one row per `(item_id, condition, prompt_variant)`:

- **Labels:** `human_label`, `n_human_votes`, `frac_prefer_a`, `human_unanimous`,
  `human_agreed` (unanimous with at least 2 votes), `d_human`.
- **Verdicts:**
  - `judge_verdict`: the greedy AB verdict, the primary one;
  - `correct`;
  - `verdict_bidir`: the argmax of the mean `p_a` over both orders;
  - `correct_bidir`.
- **Signals:** `conf_verb`, `conf_lp`, `conf_sc` (clean/P1 only), `conf_bpe`, and
  `conf_ens` with `ens_entropy_{total,aleatoric,epistemic}` (clean only).
- **Calibration forms:** `conf_bpe_prob`, `conf_ens_prob`, `conf_verb_bidir`,
  `conf_lp_bidir` (D7).
- **Order:** `flipped`, computed within each `(condition, prompt_variant)`.
- **Tier B:** `len_a`, `len_b`, `len_ratio`, `abs_len_diff`, `longer_is_chosen`,
  `judge_output_len`.
- **Tier C:** `verdict_margin`, `cot_*_greedy`, and `cot_*_sampled_t07` (clean/P1
  only).

## 4. Layout

```
configs/    run.yaml (Qwen) · run_kev.yaml (RQ6) · run_autoj.yaml (RQ7)
src/        config · data · prompts (frozen) · perturb · judge · parse · signals · features ·
            predictor · bayesian · metrics · boot · plots · vacuum_test · ablation_decoding ·
            judge_kev + kev_signals (RQ6) · judge_autoj + autoj_signals (RQ7)
analysis/   rq1 … rq7, human_disagreement, vacuum, decoding_ablation — CLIs over items.parquet;
            compare_judges (three-judge figure), auto_accept (post-hoc, D29),
            demo_bundle → site_data (the site's data)
site/       static HTML/CSS/JS site; computes no statistic; data/ is built, tracked
tests/      one file per src/ module except plots and vacuum_test; test_demo_bundle.py,
            test_site_data.py, test_site_e2e.py (Playwright)
notebooks/  Colab records of the three GPU runs; install, pull and call src/ only
scripts/    deploy_space.py (Hugging Face Space)
docs/readme/  README images
runs/       checkpoints (gitignored; published as HF dataset shubhamsengupta/judge-calibration-runs)
results/    parquet (gitignored); CSVs and figures/ (tracked: REPORT.md's evidence)
REPORT.md · README.md · REPRODUCE.md · DECISIONS.md · PREREGISTRATION.md (frozen; do not edit)
```

## 5. Conventions

- **Language and config:** Python 3.11+, type hints on every public function. Config
  goes YAML → dataclass, and nothing in `src/` hardcodes a model name, path,
  threshold or k.
- **Plots:** matplotlib only. One figure per function, which returns the `Figure` and
  saves it to `results/figures/` under a deterministic name.
- **Output and randomness:** use `logging`, not `print`, except in CLIs. Every
  stochastic function takes an explicit `seed` or `rng`.
- **Comments and docstrings** say what the code does and why. Dates, task numbers and
  decision history do not go in code. In `metrics.py`, `predictor.py` and
  `bayesian.py`, docstrings give the formula and why this variant was chosen, so
  they can be read as study material.
- **Tests are a deliverable.** The core checks:
  - `ece()` against the hand-computed 0.222 case;
  - Cohen's κ against the hand-computed case (p_o = 0.85, p_e = 0.5 → κ = 0.70);
  - the tie-policy branches in `build_items()`;
  - grouped folds that differ between seeds;
  - a cluster bootstrap that is wider than a row bootstrap;
  - held-out intercepts that are marginalized;
  - Bayesian predictions that don't change when a feature is rescaled.

  Run `pytest` before any commit that touches `metrics`, `boot`, `predictor`,
  `bayesian` or `data`.

## 6. Making a change

- **Don't change a method quietly.** A change to what is captured, what is discarded
  or the scope of a run gets a `DECISIONS.md` entry. If an invariant looks wrong for
  a specific case, raise it rather than working around it.
- **Prompts are frozen.** Changing `src/prompts.py` means a new prompt version and a
  full re-run.
- **Results travel with their text.** A changed number in `results/` is committed
  together with the `REPORT.md` text it supports (see §7).

## 7. Commands

The full command sequence, with runtimes and expected numbers, is in
`REPRODUCE.md`. The essentials:

```bash
pytest
python -m src.data --config configs/run.yaml          # then src.parse, src.signals
python -m analysis.rq1 --config configs/run.yaml      # rq2, rq3, human_disagreement, vacuum, decoding_ablation
python -m analysis.rq4 --config configs/run.yaml --task {ablation,h4,transfer,category,calibration,bayesian_comparison,verbose_shift}
python -m analysis.rq5 --config configs/run.yaml --task {threshold_sweep,distillation,human_disagreement}
python -m analysis.rq6 --config configs/run_kev.yaml   --task {calibration,position_swap,verbosity,bayesian_recalibration}
python -m analysis.rq7 --config configs/run_autoj.yaml --task {calibration,position_swap,verbosity,bayesian_recalibration}
# GPU only (Colab, fresh venv, pip install -e ".[colab]"): src.judge, src.judge_autoj; src.judge_kev needs kev.serve
```

After a rerun:

- commit the changed `results/*.csv` and figures together with the `REPORT.md` text
  they support;
- rebuild `site/data/` if a number on the site changed.

GitHub Pages redeploys on a push that touches `site/`. The Hugging Face Space needs
`scripts/deploy_space.py`.
