# REPORT.md

Written incrementally as results land, per `CLAUDE.md` §4 — not assembled at the end.

---

## Methods notes

Small, dated empirical observations that inform a design decision but don't belong to
a specific RQ section yet. Promoted into the relevant RQ section once that section is
written.

### Temperature scaling invalidates `p_a` for `sample_idx > 0` (D6) — confirmed empirically, 8 Sep 2026

**Claim (D6):** `p_a` (renormalised P(A) from top-K logprobs) is only valid for the
canonical greedy call (`sample_idx == 0`, T=0). Sampled draws (`sample_idx > 0`, T=0.7)
report logprobs scaled by temperature, so `p_a` computed from them is not comparable to
the T=0 value and must not be used as a calibration signal.

**Method:** for all 20 items in task 1.6's pilot (`clean`/P1/AB), compared the greedy
call's `p_a` against the first sampled draw's `p_a` for the same item. Split by whether
the two calls agreed on the verdict itself, since a verdict *flip* between temperatures
is a different, expected phenomenon (genuine sampling variability — the basis for
`conf_sc`), not evidence about logprob scaling.

**Result:**
- 2/20 items had the sampled draw pick a different verdict than the greedy call —
  expected, not evidence for or against the D6 claim.
- Of the 18/20 items where **both calls agreed on the verdict**, `p_a` still differed
  between T=0 and T=0.7 in every single case: mean absolute difference ≈ 0.00038,
  maximum ≈ 0.0067 (item `5ea559a223e8e898`: 0.0067 at T=0 vs 2.3e-9 at T=0.7 — several
  orders of magnitude in relative terms, despite a small absolute gap, since both values
  sit near the confident extreme).

**Conclusion:** D6's restriction is empirically justified, not just a theoretical
precaution — temperature measurably shifts the reported logprob distribution even when
it doesn't change the model's actual decision. `conf_lp`/`p_a` correctly come from
`sample_idx == 0` only.

Analysis script: ad hoc, not checked in (see `runs/judge_clean.jsonl` + `runs/logprobs/`
from task 1.6's pilot for the underlying data).

### Verbose prompt-token multiplier, throwaway padding — 8 Sep 2026

**Purpose:** an early, rough read on how much longer `verbose` condition prompts get,
to sanity-check the GPU-budget extrapolation ahead of the real `verbose_pad()` (task
4.1, W4). Deliberately throwaway - the padding used here (a fixed filler sentence
repeated 5x, appended to each assistant turn) is not the real repetitive-list-attack
design from Zheng et al. §3.3 that task 4.1 will implement.

**Method:** 10 items sampled from the dataset (seed 1234), `P1`/`AB` prompts rendered
both clean and with the throwaway padding applied to both sides' conversations, real
token counts via `Qwen/Qwen2.5-7B-Instruct`'s own tokenizer (not a character-count
approximation).

**Result:** multiplier ranged 1.07x–1.44x across the 10 items, mean **1.19x**.

**Conclusion:** consistent with `TASKS.md` task 1.6's own stated expectation
("+20-40% wall-clock, not 2-3x - padding lengthens the prompt, and prefill is cheap
next to decode"), now backed by a real tokenizer measurement rather than an assumption.
The real `verbose_pad()` (task 4.1) may land at a different multiplier - this is a
sanity check on the current GPU-budget estimate's order of magnitude, not a
substitute for re-extrapolating once the real perturbation exists.

### Vacuum test — "dark current" (task 1.8, LEARNING.md A8) — 10 Sep 2026

**Purpose:** does the judge express a spurious preference when there is genuinely no
content difference to base one on? Our JSON schema forces a binary `verdict` with no
tie option (unlike the Dark Current paper's own `DC(J) = count(J(o) != 'tie') / N`
metric), so "does it pick a winner" is trivially 100% by construction - the
informative question is *how* it picks, not whether.

**Method:** 40 pairs of identical responses (`vacuum_identical()` - one real item's
response duplicated to both sides) + 20 pairs of empty responses (`vacuum_empty()` -
real questions, both sides' assistant turns blanked), `clean`-schedule item sample
(seed 1234), single greedy `P1`/AB call per pair (order doesn't matter when both
sides are byte-identical - see `src/perturb.py`). Two measurements: the A/B verdict
split (systematic positional skew, since content provides no real signal either way),
and mean `verbalized_conf` (per D25, the primary false-confidence signal - `conf_lp`
is a known post-mask upper bound, reported but not relied on here).

**Result:**

| | n | verdict split (A / B) | binomial p (two-sided, H0: 50/50) | mean `verbalized_conf` |
|---|---|---|---|---|
| Identical pairs | 40 | 72.5% / 27.5% | **0.0064** | 0.970 |
| Empty pairs | 20 | 10.0% / 90.0% | **0.0004** | 0.973 |
| *(real clean/P1 items, for comparison)* | 20 | — | — | 0.945 |

Parse rate: 60/60 (100%) - structured output held up cleanly even on this degenerate
content.

**Conclusion:** two distinct findings, both real:

1. **Genuine positional bias, and its direction flips with content type.** Both splits
   are far from chance (p < 0.01 for identical, p < 0.001 for empty) - not noise at
   these sample sizes. Identical content favors position A; empty content favors
   position B, just as strongly, in the *opposite* direction. The judge's "dark
   current" isn't a single fixed positional preference - what nothing looks like to it
   depends on what kind of nothing it's shown.
2. **False confidence.** Mean `verbalized_conf` on both vacuum types (0.970, 0.973) is
   not lower than - if anything, slightly higher than - the real clean/P1 items'
   mean (0.945). A judge that recognized "there is no real basis for a decision here"
   should show measurably *lower* confidence on these degenerate pairs. It doesn't.

Both findings matter for RQ1 (calibration) and RQ2 (whether confidence is informative
about error): the judge's stated confidence does not distinguish a genuine, considered
judgment from a coin flip forced by schema constraints and possibly-arbitrary
positional preference. Sample sizes (n=40, n=20) are small by design (task 1.8 is a
W1 sanity check, not a powered study) - treat the specific percentages as descriptive,
not as a precise population estimate of the judge's true positional bias rate.

Analysis script: ad hoc, not checked in (see `runs/vacuum.jsonl` + `runs/logprobs/`
for the underlying data; `src/vacuum_test.py` generated it).

### `brier_decomposition()` reconstructs exactly only for discrete-valued signals (D14) — 17 Sep 2026

**Claim:** the reconstruction identity `brier = reliability - resolution + uncertainty`
(`src/metrics.py::brier_decomposition()`, task 2.5) is exact when every bin's forecasts
share one literal confidence value (the "auto" strategy's discrete branch, D14) - it is
only *approximately* exact when a bin groups together items with genuinely different
confidence values (the quantile branch), because `reliability`/`resolution` are computed
from each bin's *mean* confidence, not each item's own value. The gap between the
grouped reconstruction and the raw `brier()` score is real and has a name - "grouping
loss," the information lost by replacing many distinct confidence values with one bin
average. `test_brier_decomposition_reconstructs_brier_score`'s `1e-6` tolerance (task
2.5) was validated only against a 2-unique-value discrete reference case; it was never a
general claim that reconstruction is exact for every signal, and this doesn't
contradict it.

**Method:** RQ1 (task 2.6, `analysis/rq1.py`) computes this reconstruction for all four
original confidence signals on `(clean, P1)` items (N=1836, `human_label` non-null),
`n_bins=10`, `strategy="auto"`. `ece()`'s `"auto"` strategy bins by exact unique value
whenever `n_unique(signal) <= n_bins`; otherwise it falls back to quantile bins (D14).

**Result:**

| signal | unique values | binning (n_bins=10) | verdict def. | `brier()` | `reliability - resolution + uncertainty` | gap |
|---|---|---|---|---|---|---|
| `conf_verb` | 4 | exact (discrete) | judge_verdict | 0.215120 | 0.215120 | 0.0 |
| `conf_verb` | 4 | exact (discrete) | verdict_bidir | 0.186362 | 0.186362 | 0.0 |
| `conf_sc` | 5 | exact (discrete) | judge_verdict | 0.204759 | 0.204759 | 0.0 |
| `conf_sc` | 5 | exact (discrete) | verdict_bidir | 0.196317 | 0.196317 | 0.0 |
| `conf_lp` | 72 | quantile | judge_verdict | 0.242173 | 0.242203 | 0.00003 |
| `conf_lp` | 72 | quantile | verdict_bidir | 0.207322 | 0.207069 | 0.00025 |
| `conf_bpe` | 1521 | quantile | judge_verdict | 0.163734 | 0.160900 | 0.0028 |
| `conf_bpe` | 1521 | quantile | verdict_bidir | 0.177267 | 0.171053 | 0.0062 |

**Conclusion:** the pattern maps exactly onto binning strategy, not onto anything
signal-specific - the two discrete signals (`conf_verb`, `conf_sc`) reconstruct to
floating-point precision; the two continuous, quantile-binned signals (`conf_lp`,
`conf_bpe`) don't, with the gap growing with how many distinct values get compressed
into 10 bins (72 unique values -> a gap of ~0.0001-0.0003; 1521 unique values -> a gap
of ~0.003-0.006). This is expected behavior of a grouped calibration decomposition, not
a bug in `brier_decomposition()` - confirmed by the fact the gap tracks unique-value
count exactly, not signal identity. Reported here so RQ1's eventual write-up (task 2.7)
doesn't need to re-derive this if the arithmetic gets checked in a viva: `conf_lp`'s and
`conf_bpe'`s `reliability`/`resolution` numbers describe the *binned* forecasts, not a
claim that they reconstruct `brier()` to the letter.

Analysis script: `analysis/rq1.py` (task 2.6).
