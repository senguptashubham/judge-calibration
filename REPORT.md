# REPORT.md

Written incrementally as results land, per `CLAUDE.md` §4 — not assembled at the end.

---

## RQ1 — Is the judge's stated confidence calibrated?

*Written 17 Sep 2026, paths updated 18 Sep 2026 (output filenames now carry
`Config.model_slug`, `src/config.py`, so a second judge model's results never
overwrite this one's). Data: `results/items_qwen2.5_7b_instruct.parquet`, filtered to `condition == "clean" AND prompt_variant == "P1"` (invariant 14) and `human_label` non-null — this drops 68 of the 1904 non-tie items, each a genuine 50/50 non-tie human split with no majority label (`data.py::build_items()`; not a data bug), leaving **N = 1836**. Analysis: `analysis/rq1.py`. Figures: `results/figures/reliability_{conf_verb,conf_lp,conf_sc,conf_bpe}_qwen2.5_7b_instruct.png`. All CIs are cluster bootstraps, B=2000, grouped on `question_id` (invariant 2).*

**Headline: the judge is measurably overconfident.** Its own verbalized confidence
(`conf_verb`) overstates its actual accuracy by **0.192 [0.172, 0.213]** when scored
against a single canonical pass (`judge_verdict`), and by **0.157 [0.136, 0.178]** once both presentation orders are averaged (`verdict_bidir`) — both 95% CIs sit well clear of zero.

**Full results** (`judge_verdict` = single canonical AB pass; `verdict_bidir` =
order-averaged, D7):

| signal | verdict def. | accuracy | κ | ECE | MCE | Brier | overconfidence gap |
|---|---|---|---|---|---|---|---|
| `conf_verb` | judge_verdict | 0.757 [0.735, 0.778] | 0.514 [0.471, 0.556] | 0.192 | 0.383 | 0.215 | 0.192 [0.172, 0.213] |
| `conf_lp` | judge_verdict | 0.757 [0.735, 0.778] | 0.514 [0.471, 0.556] | 0.242 | 0.492 | 0.242 | 0.242 [0.221, 0.264] |
| `conf_sc` | judge_verdict | 0.757 [0.735, 0.778] | 0.514 [0.471, 0.556] | 0.183 | 0.313 | 0.205 | 0.169 [0.150, 0.187] |
| `conf_bpe` | judge_verdict | 0.757 [0.735, 0.778] | 0.514 [0.471, 0.556] | 0.115 | 0.229 | 0.164 | 0.049 [0.030, 0.069] |
| `conf_verb` | verdict_bidir | 0.792 [0.771, 0.813] | 0.585 [0.542, 0.626] | 0.157 | 0.216 | 0.186 | 0.157 [0.136, 0.178] |
| `conf_lp` | verdict_bidir | 0.792 [0.771, 0.813] | 0.585 [0.542, 0.626] | 0.207 | 0.397 | 0.207 | 0.207 [0.186, 0.228] |
| `conf_sc` | verdict_bidir | 0.792 [0.771, 0.813] | 0.585 [0.542, 0.626] | 0.177 | 0.688 | 0.196 | 0.133 [0.116, 0.152] |
| `conf_bpe` | verdict_bidir | 0.792 [0.771, 0.813] | 0.585 [0.542, 0.626] | 0.151 | 0.316 | 0.177 | **0.014 [-0.011, 0.038]** |

**Every original signal is overconfident, but the magnitude varies by roughly 5x
depending which one you trust.** 
`conf_lp` (raw exp(verdict-token logprob)) is the worst — 0.242 overconfident under `judge_verdict`, worse than `conf_verb`. This traces directly to its distribution: 99.3% of `conf_lp` values sit at or above 0.99 (median is exactly 1.0) — the model is essentially always "sure" of the token it generated, regardless of whether the underlying judgment was right, a well-documented property of greedy-decoded token probabilities, not a bug in how it's computed.
`conf_bpe` (the order-consistency-based entropy signal) is the best-calibrated by a wide margin, and under `verdict_bidir` specifically its overconfidence gap's CI **crosses zero** (0.014 [-0.011, 0.038]) — not distinguishable from perfect calibration at this sample size. This is plausibly not a coincidence: `conf_bpe` is itself built from agreement across both presentation orders, so it looks best exactly when scored against the order-aware ground truth it was already implicitly modeling.

**Debiasing by averaging both orders buys a real, statistically confirmed accuracy
gain.** Accuracy moves from 0.757 to 0.792 going from `judge_verdict` to `verdict_bidir` — a difference that could just be noise if judged by eye from two separate CIs, so it was tested properly: a **paired** cluster bootstrap on the per-item accuracy difference (invariant 3 — same 1836 items scored two ways is not two independent samples) gives **+0.0354 [0.0177, 0.0531]**, a CI that excludes zero. Concretely: averaging both orders flips 157 items from wrong to right, and only 92 from right to wrong — a net gain of 65/1836 = 0.0354, matching the bootstrap exactly. The debiasing gain is real, not just directionally plausible.

κ improves in step (0.514 → 0.585), but both remain well below the human-human ceiling of κ=0.683 (task 0.9, N=536). That population is smaller and not perfectly matched to RQ1's 1836 items, so treat the comparison as approximate — but debiasing clearly closes only a small fraction of the gap to human-level agreement.

**Corroborating evidence the false confidence is not just a calibration-curve
artifact.** The vacuum test (task 1.8) found the judge's mean `verbalized_conf` on
pairs with *no real content difference to judge* (0.970 identical, 0.973 empty) was, if anything, slightly *higher* than on real items (0.945) — a judge that recognized "there is no basis for a decision here" should show measurably lower confidence on those degenerate pairs, and doesn't.
RQ1's overconfidence finding and the vacuum test's false-confidence finding are two independent measurements pointing at the same underlying problem: the judge's stated confidence tracks something other than its actual likelihood of being right.

**Caveat for `conf_lp`/`conf_bpe`'s `reliability`/`resolution` numbers specifically.**
`brier_decomposition()`'s reconstruction identity (`brier = reliability - resolution + uncertainty`) is exact only when every bin shares one literal confidence value — true for the two discrete signals here (`conf_verb`: 4 unique values, `conf_sc`: 5), so their reconstruction matches `brier()` to floating-point precision. `conf_lp` (72 unique values) and `conf_bpe` (1521 unique values) fall through `ece()`'s `"auto"` strategy to quantile binning instead, where `reliability`/`resolution` are computed from each bin's *mean* confidence rather than each item's own value — a real, expected "grouping loss" gap from `brier()` (up to ~0.006 for `conf_bpe`), not a bug. The table above reports `ece`/`mce`/`brier` (computed directly, unaffected by this) alongside `reliability`/ `resolution` at face value; treat the latter two as describing the *binned* forecast for these two signals specifically, not a claim they reconstruct `brier()` to the letter.

---

## RQ2 — Is any cheap uncertainty signal informative about error?

*Written 17 Sep 2026, paths updated 18 Sep 2026 (model-namespaced filenames -
see RQ1's metadata line). Data: same population as RQ1 — `results/items_qwen2.5_7b_instruct.parquet` filtered
to `(clean, P1)` and `human_label` non-null, **N = 1836** (invariant 14). Analysis:
`analysis/rq2.py` (per-signal risk-coverage) and `analysis/rq5.py` (entropy threshold
sweep, task 3.2b). Figures: `results/figures/risk_coverage_qwen2.5_7b_instruct.png` (the thesis figure),
`results/figures/entropy_threshold_sweep_qwen2.5_7b_instruct.png`. Tables: `results/rq2_table_qwen2.5_7b_instruct.csv`,
`results/rq5_threshold_sweep_table_qwen2.5_7b_instruct.csv`. All CIs are cluster bootstraps, B=2000,
grouped on `question_id` (invariant 2). `accuracy@c%`/`κ@c%` use a rank-based
"top-c% most confident items" cut (`analysis/rq2.py::_top_k_mask`), a different,
deliberately simpler definition from the tie-safe value-threshold curve the figure
itself is built from — see that function's docstring for why the two don't need to
agree, and RQ1's own accuracy/κ at 100% coverage (0.757 [0.735, 0.778] /
0.514 [0.471, 0.556]) as the no-abstention reference point.*

**Headline: every real signal beats chance, none comes remotely close to the oracle,
and the two rankings you'd use to pick "the best" signal disagree with each other.**
The oracle's AURC is **0.0295** — every real signal's AURC is 1.3–4.7x higher. By
AUROC (ranking ability — can the signal tell an error item from a correct one at all),
`conf_bpe` wins clearly (0.794). By AURC (area under the *realized* risk-coverage
curve — how much risk is actually retained as coverage shrinks), `conf_sc` wins
(0.038), not `conf_bpe` (0.100). These measure genuinely different things — AURC is
also shaped by a signal's own value distribution, not just its ranking quality — and
`conf_sc`'s advantage there traces to it having only 5 possible values at all
(`k_sc=4`), which mechanically front-loads coverage into large steps rather than
reflecting cleaner discrimination. Stated plainly rather than picking one "winner":
neither ranking is wrong, they answer different questions, and reporting only one
would misrepresent the other.

**Full results:**

| signal | AURC | AUROC | accuracy@90% | κ@90% | accuracy@75% | κ@75% | accuracy@50% | κ@50% |
|---|---|---|---|---|---|---|---|---|
| `conf_verb` | 0.139 [0.117, 0.162] | 0.639 [0.608, 0.672] | 0.780 [0.756, 0.802] | 0.557 [0.513, 0.602] | 0.814 [0.784, 0.833] | 0.626 [0.566, 0.663] | 0.844 [0.790, 0.850] | 0.683 [0.578, 0.696] |
| `conf_lp` | 0.078 [0.066, 0.092] | 0.715 [0.682, 0.746] | 0.784 [0.763, 0.807] | 0.566 [0.524, 0.611] | 0.832 [0.804, 0.853] | 0.662 [0.608, 0.705] | 0.889 [0.857, 0.906] | 0.777 [0.713, 0.813] |
| `conf_sc` | 0.038 [0.032, 0.046] | 0.623 [0.601, 0.647] | 0.788 [0.766, 0.811] | 0.573 [0.528, 0.619] | 0.816 [0.789, 0.831] | 0.627 [0.573, 0.657] | **0.838 [0.785, 0.836]** | 0.668 [0.565, 0.667] |
| `conf_bpe` | 0.100 [0.083, 0.120] | **0.794 [0.767, 0.819]** | 0.795 [0.769, 0.817] | 0.589 [0.538, 0.633] | 0.861 [0.830, 0.887] | 0.721 [0.661, 0.774] | 0.916 [0.892, 0.942] | 0.832 [0.784, 0.883] |

**`conf_bpe` is the strongest signal by nearly every coverage-restricted metric**
(highest accuracy@c% and κ@c% at all three thresholds, and the best AUROC) — the one
place it isn't "best" is AURC, for the discreteness reason above. `conf_verb`
(the judge's own stated confidence, RQ1's headline signal) is the weakest across the
board on this task specifically — being badly overconfident (RQ1) doesn't
automatically make a signal uninformative for *ranking* errors, but here it's both.

**Caveat: `conf_sc`'s accuracy@50% point estimate (0.838) sits fractionally outside
its own 95% CI** (`[0.785, 0.836]` — the point is 0.0018 above the upper bound).
A real, minor artifact of the rank-based top-k cut interacting with `conf_sc`'s heavy
ties (only 5 distinct values) right at the 50%-coverage boundary, not a computation
bug — flagged rather than silently rounded through, same standard RQ1's `brier()`
reconstruction caveat was held to.

**Entropy threshold sweep (RQ5, task 3.2b): the preregistered prediction did NOT
hold.** D23 predicted epistemic thresholding would beat total thresholding on AURC,
since epistemic is the reducible part of the uncertainty. The opposite happened, and
by a wide margin: AURC(`ens_entropy_total`) = 0.098, AURC(`ens_entropy_aleatoric`) =
0.099, AURC(`ens_entropy_epistemic`) = 0.273 — a paired cluster-bootstrap on the gap
gives **epistemic − total = +0.175 [0.141, 0.207]**, a CI entirely on the "epistemic is
worse" side of zero. The `total`/`aleatoric` curves are visually indistinguishable in
the figure (distinct linestyles/markers used specifically so they stay legible despite
this) because they're nearly numerically identical: **65% of items have epistemic
entropy at machine-epsilon** (all three prompt variants agree), so `total ≈ aleatoric`
for most of the population by construction (`total - aleatoric` exactly equals
`epistemic`, verified). The mechanism: ensemble agreement (low epistemic) does not
mean the ensemble is *right* — among the lowest-decile-epistemic items, accuracy is
only **46%**, far below the 75.7% overall baseline. The judge's three prompt variants
can be confidently, uniformly wrong together just as easily as confidently right
together; low epistemic uncertainty is actively misleading as a trust signal here, not
merely uninformative.

**Human disagreement (tasks 3.3, 3.4): does judge behavior track genuine human
consensus strength at all?** Population for both: `(clean, P1)` further restricted to
`n_human_votes >= 2` (D9 — `d_human` is undefined-as-a-disagreement-signal below 2
votes), **N = 595**. `d_human = |frac_prefer_a - 0.5|` takes only **3 distinct values**
here and is badly imbalanced — **0.167** (28 items, 4.7%), **0.25** (3 items, 0.5%),
**0.5** (564 items, 94.8%), a direct consequence of MT-Bench's small (2-5) per-item
vote counts. D9's secondary bucketed (unanimous/strong-majority/contested) comparison
is correctly out of scope (`n_contested = 31` in this population, below D9's
`n_contested >= 100` bar — a different, smaller population than Gate 0's own
`n_contested = 123`, which was computed on the full 2396-item corpus).

Both `correct` and `conf_verb` rise with `d_human` (i.e. are lower on more-contested
items), CI-backed under both a linear and a rank-only assumption:

| target | OLS slope | Spearman ρ |
|---|---|---|
| `correct ~ d_human` | 0.803 [0.208, 1.376] | 0.141 [0.036, 0.237] |
| `conf_verb ~ d_human` | 0.059 [0.016, 0.100] | 0.113 [0.031, 0.190] |

Spearman was added specifically because 3 points can't support a linear-shape claim;
both CIs excluding 0 under the weaker, monotonic-only assumption is what makes this
finding hold up, not just the OLS number in isolation. Extended to all four original
signals (task 3.4, Spearman only, same population,
`results/figures/d_human_correlations_qwen2.5_7b_instruct.png`):

| signal | Spearman ρ | 95% CI |
|---|---|---|
| `conf_verb` | 0.113 | [0.031, 0.190] |
| `conf_lp` | 0.091 | [-0.004, 0.181] |
| `conf_sc` | 0.024 | [-0.054, 0.123] |
| `conf_bpe` | 0.101 | [0.002, 0.185] |

All four are weak (ρ ≤ 0.11), and **two of four (`conf_lp`, `conf_sc`) don't clear
zero** at this sample size. Per task 3.4's own framing: if judge confidence tracked
genuine task ambiguity, it should track `d_human` meaningfully; instead none of the
four does more than weakly. Combined with RQ1's overconfidence finding and the vacuum
test's false-confidence result, this is a third, independent line of evidence for the
same conclusion — the judge's confidence signals track something other than the
actual difficulty/ambiguity of the item. (Caveat carried over from task 3.3: with
94.8% of `d_human`'s weight at one value, this reads as a comparison between the
dominant near-unanimous group and a small 5.2% minority, not a fine-grained trend
across many disagreement levels — more human votes would require new annotation,
outside this project's scope. D3, RewardBench 2 augmentation, was declined "for now"
in `PREREGISTRATION.md` §6 with an explicit reopen condition — "an unexpectedly thin
subgroup for a specific analysis" — that this arguably meets; revisiting it is a
deliberate scope decision, not a default.)

A genuine floating-point bug was found and fixed while building this analysis:
`d_human` values from 1/3 vs. 2/3 vote splits (both mathematically 1/6) landed on
adjacent float64 values (~6e-17 apart), which `get_bin_edges`' exact-value branch
treated as 2 bins instead of 1, and independently corrupted the Spearman ranking too
(ranks need exact ties detected as ties) until `d_human` was rounded to 6dp once in
`load_disagreement_items()`, upstream of every consumer. Confirmed this does not
affect any already-reported RQ1 number (`conf_verb`/`conf_sc`/`conf_lp` show no such
mismatch; `conf_bpe` does, 1521 raw vs. 797 real unique values, but is always
quantile-binned regardless since it has far more unique values than `n_bins=10`
either way).

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
