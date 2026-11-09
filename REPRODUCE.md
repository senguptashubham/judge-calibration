# Reproducing the results

The project has two halves:

1. **GPU inference** writes the checkpoints in `runs/`. It runs on Colab, using the notebooks in `notebooks/`.
2. **Everything after that** runs on a CPU, from those checkpoints to every table, figure and number in `REPORT.md` and on the site.

This file covers the second half and gives the expected numbers. A full rerun of
the second half was checked from a fresh clone, as described below.

## What you need

- Python 3.11.
- The judge checkpoints in `runs/`, about 1.6 GB. They are gitignored, so a clone does not include them. They are published as the Hugging Face dataset [`shubhamsengupta/judge-calibration-runs`](https://huggingface.co/datasets/shubhamsengupta/judge-calibration-runs) under CC BY 4.0. See [Getting the checkpoints](#getting-the-checkpoints).
- The MT-Bench human judgments (`lmsys/mt_bench_human_judgments`). `src.data` downloads them from Hugging Face on the first run. `analysis.demo_bundle` then reads them offline from the local cache.

## Getting the checkpoints

From the repo root, after `pip install -e .`, which installs `huggingface_hub`:

```bash
hf download shubhamsengupta/judge-calibration-runs --repo-type dataset --local-dir runs
tar -xf runs/qwen2.5_7b_instruct/logprobs.tar -C runs/qwen2.5_7b_instruct
tar -xf runs/autoj_13b_gptq_4bits/logprobs.tar -C runs/autoj_13b_gptq_4bits
```

The two `logprobs/` folders hold one small file per call (22,908 and 14,870 files),
so they are stored as `.tar` archives. Once they are extracted, you can delete the
archives and the downloaded `runs/README.md`, which is the dataset card. Neither is
read by the pipeline.

| Folder | Contents | Produced by |
|---|---|---|
| `runs/qwen2.5_7b_instruct/` | `judge_clean.jsonl`, `judge_verbose.jsonl`, `vacuum.jsonl`, `ablation_decoding.jsonl`, `logprobs/` | `notebooks/01_qwen_inference.ipynb` |
| `runs/kev_8b/` | `kev.jsonl` | `notebooks/02_kev_inference.ipynb` |
| `runs/autoj_13b_gptq_4bits/` | `autoj.jsonl`, `logprobs/` | `notebooks/03_autoj_inference.ipynb` |

You can also regenerate the checkpoints yourself by running the three notebooks on a
Colab GPU. They install the `colab` extra in a fresh venv and call `src/judge.py`,
`src/judge_kev.py` and `src/judge_autoj.py`. vLLM sampling on a GPU is not
bit-exact across hardware, so regenerated checkpoints can differ slightly from the
published ones, and so can every number downstream.

## Commands

From the repo root:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
# then download and extract runs/ as above

Q=configs/run.yaml; K=configs/run_kev.yaml; A=configs/run_autoj.yaml

# Concept figure (ece_auroc_orthogonal.png), light and dark
python -m src.plots

# Qwen2.5-7B-Instruct (RQ1-RQ5)
python -m src.data    --config $Q
python -m src.parse   --config $Q
python -m src.signals --config $Q
for m in rq1 rq2 rq3 human_disagreement vacuum decoding_ablation; do
  python -m analysis.$m --config $Q
done
for t in ablation h4 transfer category calibration bayesian_comparison verbose_shift; do
  python -m analysis.rq4 --config $Q --task $t
done
for t in threshold_sweep distillation human_disagreement; do
  python -m analysis.rq5 --config $Q --task $t
done

# kev-8b (RQ6)
python -m src.kev_signals --config $K
for t in calibration position_swap verbosity bayesian_recalibration; do
  python -m analysis.rq6 --config $K --task $t
done

# auto-j-13b (RQ7)
python -m src.autoj_signals --config $A
for t in calibration position_swap verbosity survivorship bayesian_recalibration; do
  python -m analysis.rq7 --config $A --task $t
done

# Cross-judge comparison, auto-accept analysis, site data
python -m analysis.compare_judges --config $Q --kev-config $K --autoj-config $A
python -m analysis.auto_accept    --config $Q --kev-config $K --autoj-config $A
HF_DATASETS_OFFLINE=1 python -m analysis.demo_bundle --config $Q --kev-config $K --autoj-config $A
python -m analysis.site_data --config $Q

pytest
```

Then run `git status`. If everything reproduced, it is empty: every tracked CSV under `results/`, every figure under `results/figures/` and both files under `site/data/` came out identical to the committed versions.

## Expected runtime

A full run takes about 2.5–3 hours on a laptop CPU. Most of that is six steps:

| Step | Time |
|---|---|
| `analysis.rq7 --task bayesian_recalibration` (with permutation nulls) | ~26 min |
| `analysis.rq6 --task bayesian_recalibration` (with permutation nulls) | ~25 min |
| `analysis.rq5 --task distillation` (two Bayesian models, with permutation nulls) | ~43 min |
| `analysis.rq4 --task bayesian_comparison` (with permutation null) | ~25 min |
| `analysis.rq4 --task ablation` (both populations, with permutation nulls) | ~15 min |
| `analysis.auto_accept` | ~5 min |

The Bayesian steps fit NumPyro NUTS models with fixed seeds. Run them one at a time: each holds about 40,000 posterior draws per item, and two at once can exhaust a 16 GB laptop's memory.

## Expected numbers

The 95% intervals come from a cluster bootstrap over `question_id`.

| Result | Value | Source |
|---|---|---|
| Qwen accuracy / Cohen's κ (clean, AB order) | 0.757 / 0.514 | `results/rq1_table_qwen2.5_7b_instruct.csv` |
| Overconfidence gap, stated confidence (`conf_verb`) | +0.192 [0.172, 0.213] | same |
| AUROC of the order-swap signal (`conf_bpe`) | 0.794 [0.767, 0.819] | `results/rq2_table_qwen2.5_7b_instruct.csv` |
| AUROC of stated confidence (`conf_verb`) | 0.639 [0.608, 0.672] | same |
| Verdicts that flip when the answers swap | 27.8% [24.8%, 30.8%] | `results/rq3a_table_qwen2.5_7b_instruct.csv` |
| ΔECE under padding (`conf_bpe`, verbose − clean) | +0.043 [0.011, 0.052] | `results/rq3_table_qwen2.5_7b_instruct.csv` |
| RQ4 Tier A logreg − best single signal, per fold, agreed items | +0.019 [−0.008, 0.047] | `results/rq4_ablation_progression_qwen2.5_7b_instruct.csv` |
| RQ4 transfer, logreg, padded − clean, per fold | −0.030 [−0.051, −0.010] | `results/rq4_transfer_qwen2.5_7b_instruct.csv` |
| RQ4 frequentist vs Bayesian per-fold AUROC | 0.806 vs 0.800; worst R-hat 1.008, 0 divergences | `results/rq4_bayesian_comparison_qwen2.5_7b_instruct.csv` |
| RQ4 Bayesian permutation null (mean / max) | 0.501 / 0.553 | same |
| RQ5 two-call Bayesian − ensemble, per fold | −0.017 [−0.034, 0.003] | `results/rq5_distillation_gaps_qwen2.5_7b_instruct.csv` |
| Wrong verdicts let through when auto-accepting `conf_verb` ≥ 0.9 | 98.4% | `results/auto_accept.csv` |
| Vacuum confidence minus real items (identical / empty) | +0.021 [0.006, 0.033] / +0.023 [0.004, 0.040] | `results/vacuum_qwen2.5_7b_instruct.csv` |
| Multi-vote items (N = 595): `conf_verb` gap, `conf_bpe` AUROC | +0.180 [0.139, 0.218], 0.816 | `results/multivote_sensitivity_qwen2.5_7b_instruct.csv` |
| kev-8b AUROC / flip rate (in coverage) | 0.772 / 24.6% | `results/judge_comparison.csv` |
| auto-j-13b AUROC / flip rate (turn 1) | 0.683 / 12.5% | same |
| auto-j-13b turn-2 share correct, clean → padded, a missing verdict counted as wrong | 0.781 → 0.431 | `results/rq7_survivorship_autoj_13b_gptq_4bits.csv` |

Because the NUTS seeds are fixed, the Bayesian results came out bit-identical on the verification run. Different hardware or a different JAX build may change the samples slightly. If that happens, the conclusions in `REPORT.md` should still hold.

## Verification run

**Date:** 31 October 2026.

**Setup:**
- a fresh `git clone` of `github.com/senguptashubham/judge-calibration` at `6fb2c8a`. Later commits change documentation, the floating-point last digit of one table, and the outputs covered by the in-place rerun (see below);
- a new venv on Python 3.11.15 and `pip install -e .`;
- `runs/` copied in from the original runs. The same files, byte for byte, are the ones published in the dataset.

**Result:**
- Every command above exited with code 0, in about 58 minutes of compute.
- `git status` showed one file: the κ columns of `results/rq2_table_qwen2.5_7b_instruct.csv` differed in the 16th decimal place. The committed table had been generated before a final change to `cohens_kappa()`'s weighted branch. It was regenerated, and is now byte-identical to the fresh-clone output; every other tracked output matched as committed.
- `pytest`: 301 passed, 1 skipped. The skipped module is `tests/test_site_e2e.py` (16 browser tests). It needs Playwright, which the base install leaves out. To run it, use `pip install -e ".[site]" && playwright install chromium`, which brings the total to 317.

**Reruns after D31–D33 (2–3 November 2026).** The RQ5 distillation and the RQ7 survivorship check (D31), every meta-model AUROC, which is now scored per fold (D32), and the permutation nulls added for the remaining meta-models (D33) changed after the run above. The full command sequence was rerun in place on the original machine: every step exited with code 0, `git status` showed only the outputs those decisions change, `site/data/` was unchanged, and `pytest` passed in full. A fresh-clone run was not repeated.

**Reruns after D34 (November 2026).** `analysis.vacuum`, `analysis.human_disagreement` and `analysis.rq5 --task distillation` were rerun in place, each exiting with code 0.
- **What changed:** the vacuum table gained its confidence-gap columns, with the existing values unchanged. Two outputs are new: `multivote_sensitivity_qwen2.5_7b_instruct.csv` and the two-call convergence figure.
- **What stayed the same:** every other tracked output was byte-identical, including all RQ5 tables, and `site/data/` was unchanged.