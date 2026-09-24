# Judge Calibration

An empirical study of whether small open-weight LLM judges know when they are
wrong, and whether their uncertainty is usable for selective evaluation
(abstain-and-escalate), tested against human judgments on MT-Bench.

The primary judge is Qwen2.5-7B-Instruct. Seven research questions cover its
calibration (RQ1), whether cheap uncertainty signals are informative about
error (RQ2), robustness to position and verbosity attacks (RQ3), whether judge
errors are predictable from cheap features (RQ4), and whether a prompt
ensemble's uncertainty survives distillation to a single call (RQ5). RQ6 and
RQ7 repeat the attack battery on two more judges: kev-8b, an open stand-in for
an industry calibration claim, and auto-j-13b, a model trained specifically to
judge.

- `REPORT.md` — results and their reading, one section per RQ
- `PLAN.md` — design rationale and the week plan
- `DECISIONS.md` — every non-obvious design decision and why it was made
- `TASKS.md` — tasks with definitions of done
- `CLAUDE.md` — the hard statistical invariants, schemas, layout, and commands

**Status:** RQ1–RQ7 complete. Remaining: the Gradio demo, the report's
framing sections, and a fresh-clone reproducibility check.

## Setup

Local development (all code, statistics, and analysis — no GPU needed; see
`DECISIONS.md` D17 for why this split exists):

```
conda create -n judge-calib python=3.11
conda activate judge-calib
pip install -e .
```

The judge models themselves only run on a GPU, on Colab:

```
pip install -e ".[colab]"
```

## Reproducing results

The full command sequence is in `CLAUDE.md` §7. `TASKS.md` task 7.1 adds a
`REPRODUCE.md` with expected numbers after the fresh-clone check.

## License

All rights reserved — see `LICENSE`. This is a private certification capstone,
not an open-source release.
