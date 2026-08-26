# Judge Calibration

An empirical study of whether a small open-weight LLM judge (Qwen2.5-7B-Instruct)
knows when it is wrong, and whether its uncertainty is usable for selective
evaluation (abstain-and-escalate), tested against human judgments on MT-Bench.

Four research questions — calibration, whether cheap uncertainty signals are
actually informative about error, robustness to adversarial manipulation, and
whether judge errors are predictable in advance from cheap features — are
defined in `PLAN.md`, tracked task-by-task with a definition-of-done in
`TASKS.md`, and governed by the hard invariants in `CLAUDE.md`. Every
non-obvious design decision made while stress-testing this plan before
building it is logged in `DECISIONS.md`.

**Status:** Week 0, scaffolding only. Nothing in `src/` is implemented yet —
see `TASKS.md` task 0.1 for the next concrete step.

## Setup

Local development (writing/testing code, all statistics and analysis, the
demo — no GPU needed; see `DECISIONS.md` D17 for why this split exists):

```
conda create -n judge-calib python=3.11
conda activate judge-calib
pip install -e .
```

The judge itself only ever runs on a GPU, on Colab:

```
pip install -e ".[colab]"
```

## Reproducing results

Not yet available. `TASKS.md` task 7.1 writes `REPRODUCE.md` with the exact
commands and expected numbers once the study is complete.
