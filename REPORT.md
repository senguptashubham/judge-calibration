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

## RQ3 — Does uncertainty flag bias-induced errors, or is the fooled judge confident?

*Written 18 Sep 2026, RQ3b added same day once the `verbose` run (task 4.2) landed.
RQ3a data: same population as RQ1/RQ2 — `results/items_qwen2.5_7b_instruct.parquet`
filtered to `(clean, P1)` and `human_label` non-null, **N = 1836**. This is a
deliberate scope choice, not a data requirement: flip rate and confidence-on-flipped
don't need `human_label` at all (they're pure judge-behavior signals, nothing to do
with correctness) — the maximal population would be all 1904 clean/P1 items. RQ1's
1836-item population is reused instead so the whole report cites one canonical N
across every RQ1–RQ4 core analysis, rather than a second, 68-item-different
population for no analytical reason. RQ3b data: the same 1836 items, paired —
`(clean, P1)` vs. `(verbose, P1)`, both filtered identically; task 4.2 confirmed the
two conditions share an exactly identical 1904-item population, so filtering each
side the same way keeps them paired 1:1. Analysis: `analysis/rq3.py`. Figures:
`results/figures/rq3a_confidence_gap_qwen2.5_7b_instruct.png` (RQ3a),
`results/figures/rq3b_deltas_qwen2.5_7b_instruct.png` (RQ3b — two forest panels, Δ ECE
and Δ AUROC, sharing one signal ordering). Tables:
`results/rq3a_table_qwen2.5_7b_instruct.csv` (RQ3a), `results/rq3_table_qwen2.5_7b_instruct.csv`
(RQ3b). CI is a cluster bootstrap, B=2000, grouped on `question_id` (invariant 2);
RQ3b's deltas use the *paired* cluster bootstrap (invariant 3) since both sides are
the same items scored twice, not independent samples.*

**RQ3a — position bias.** The judge's canonical verdict flips between presentation
orders (AB vs. BA) on **27.8% [24.8%, 30.8%]** of items — a large, common failure
mode, not a rare edge case. The money question: is the judge's own confidence lower
on exactly those items where it got fooled by order?

**Full results** (mean confidence on flipped vs. unflipped items, and the gap
between them):

| signal | conf. when flipped | conf. when stable | gap (flipped − stable) | 95% CI |
|---|---|---|---|---|
| `conf_verb` | 0.935 | 0.955 | −0.0199 | [−0.0248, −0.0152] |
| `conf_lp` | 0.999 | 1.000 | −0.0009 | [−0.0031, 0.0005] |
| `conf_sc` | 0.844 | 0.957 | −0.1127 | [−0.1337, −0.0924] |
| `conf_bpe` | 0.307 | 0.998 | −0.6911 | [−0.6921, −0.6899] |

**Headline: `conf_verb` does track position bias, but only weakly.** Its gap is
small (−0.02) but the CI sits entirely below zero — the judge's stated confidence
*is* measurably lower on items its own order-flip just revealed to be shaky. Read
alongside RQ1's overconfidence finding: the direction is right, but a ~2-point
confidence drop is a faint signal to hang an abstention policy on for a bias this
common (28% of items).

**`conf_lp`'s gap is not distinguishable from zero** (CI crosses 0) — consistent
with D25's finding that constrained-decoding renormalization pins `conf_lp` near
its ceiling almost everywhere, leaving little room for it to move in either
direction.

**`conf_bpe`'s near-total collapse (0.307 vs. 0.998) is expected by construction,
not an independent discovery.** `conf_bpe = 1 - entropy(mean p_a across both
orders)` — it is *built from* cross-order agreement, so it is close to definitionally
minimized exactly when the two orders disagree (a flip). This is the same caveat
RQ1 already raised for `verdict_bidir`'s own suspiciously strong showing: a signal
scored against (or, here, computed directly from) the same order-machinery it's
being evaluated on will look artificially good on that specific axis. `conf_sc`'s
gap (−0.11) is real and worth noting but sits in between — self-consistency
sampling doesn't use cross-order information directly, so its correlation with
flipping is a genuine empirical finding, not a mechanical one.

**Caveat on the bootstrap method used for the gap.** The flipped/unflipped groups
are two *disjoint* subsets of one item population, not the same items measured
twice — `paired_cluster_bootstrap()` (used elsewhere for e.g. `judge_verdict` vs.
`verdict_bidir`) doesn't apply here, since it requires both sides to share the same
`question_id` universe, which a question with zero flipped items would violate. The
gap and its CI instead come from a single `cluster_bootstrap()` call with a
group-difference `stat_fn` computed inside each resampled replicate — the standard
tool for a two-disjoint-subgroup comparison under clustering.

**RQ3b — verbosity bias.** Same N = 1836 items, each scored under both `clean` and
`verbose` (Zheng et al.'s repetitive-list padding). Unlike RQ3a, this *is* a genuine
paired comparison — same items, two conditions — so the deltas below are
`paired_cluster_bootstrap()` CIs on `stat_fn(verbose) − stat_fn(clean)` (invariant 3).
`conf_sc` is excluded — self-consistency sampling only runs for `clean/P1` (D19/D21),
so `verbose` has no `conf_sc` values to compare at all, not a smaller or noisier
sample of them; `conf_verb`, `conf_lp`, and `conf_bpe` are unaffected, since each only
needs the greedy call at both orders, which `verbose` does collect.

| signal | ECE clean | ECE verbose | Δ ECE (verb−clean) | 95% CI | AUROC clean | AUROC verbose | Δ AUROC | 95% CI |
|---|---|---|---|---|---|---|---|---|
| `conf_verb` | 0.192 | 0.182 | −0.0098 | [−0.0268, 0.0079] | 0.639 | 0.611 | −0.0276 | [−0.0571, −0.0023] |
| `conf_lp` | 0.242 | 0.232 | −0.0106 | [−0.0279, 0.0074] | 0.715 | 0.670 | −0.0447 | [−0.0740, −0.0170] |
| `conf_bpe` | 0.115 | 0.159 | **+0.0432** | **[0.0144, 0.0516]** | 0.794 | 0.766 | −0.0278 | [−0.0494, −0.0064] |

Accuracy itself barely moves and isn't significant for any signal (0.757 → 0.768,
Δ +0.0109 [−0.0070, 0.0281], identical across all three rows since accuracy only
depends on `judge_verdict`, not the signal) — verbosity padding doesn't measurably
change how often the judge is *right*.

**Headline: verbosity doesn't fool the judge into more wrong verdicts, but it
quietly breaks the uncertainty signals meant to flag them.** All three surviving
signals' AUROC(uncertainty → error) drops significantly under `verbose` — their CIs
sit entirely below zero. The signal-specific story is sharper still: `conf_bpe` was
RQ1/RQ2's best performer on clean data (lowest ECE, highest AUROC of the four
original signals), and it is the *only* signal whose calibration itself significantly
degrades under verbose (ECE +0.043, CI excludes 0) — while `conf_verb` and `conf_lp`'s
ECE shifts are numerically negative but not significant. Read together with RQ3a:
position bias barely moves confidence at all (a ~2pp gap); verbosity moves accuracy
even less, but erodes exactly the signal — cross-order agreement — that RQ1/RQ2
found most useful for catching errors on clean data. An abstention policy tuned on
`clean` data and deployed where responses vary in verbosity would silently lose
reliability without any accuracy-side symptom to warn it.

---

## RQ4 — Can a cheap supervised meta-model beat the best single signal at predicting judge error?

*Written 22 Sep 2026. Framed per `PLAN.md` §2.2's reframe — the question this
section actually answers is **which feature family carries the signal**, not
"how good is my model." Three feature tiers (`src/features.py`): **Tier A** (8
cols) — the judge's own uncertainty signals (`conf_verb`, `conf_lp`, `conf_sc`,
`conf_bpe`, `conf_ens` + its total/aleatoric/epistemic entropy decomposition).
**Tier B** (+9 cols) — surface properties the judge itself doesn't use as a
signal (response lengths, turn, category, the judge's own output length,
order-flip). **Tier C** (+13 cols) — token-distribution detail (verdict
margin, CoT logprob/entropy aggregates, greedy and sampled). Every tier is
built on one shared population — `features.py::load_rq4_population()`,
`(clean, P1)`, `human_label` non-null, minus 17 items with an undefined
`len_ratio`/`longer_is_chosen` (3 zero-length responses, 14 exact-length
ties) — **N = 1819** — dropped uniformly across all three tiers so a score
change between tiers is a real signal-content difference, never a population
artifact. Two frequentist models throughout (invariant 11): `LogisticRegression(C=1.0)`
and `HistGradientBoostingClassifier(max_depth=3, max_iter=200,
learning_rate=0.05)` — no hyperparameter search, no kernel methods/RandomForest/XGBoost
(rejected at design time: indefensible tuning burden at ~80 groups). A third
model, the Bayesian hierarchical logistic regression (D22, `src/bayesian.py`),
joins from task 5.9b onward. CV protocol throughout: `StratifiedGroupKFold(5)`,
repeated over 10 seeds (D8) — the across-repeat spread is the headline
uncertainty, never a within-split CI, since only ~80 questions means
fold-to-fold variance is real at that scale. Analysis: `analysis/rq4.py`.
Every point estimate not from D8's own repeated-CV spread is a cluster
bootstrap, B=2000, grouped on `question_id` (invariant 2); condition/tier/model
comparisons on the same item set use the *paired* cluster bootstrap
(invariant 3).*

**Headline: the judge's own uncertainty signals already carry essentially all
the recoverable signal — engineered surface and token-distribution features
add nothing measurable, at either the tier level or the individual-feature
level.** This holds up two independent ways: the tier-level ablation (5.5)
found the apparent A→B→C AUROC decline is statistically indistinguishable
from no change at all (every paired-progression CI crosses zero), and the
finer-grained coefficient analysis (5.9) confirms it isn't just underpowered
noise hiding a real pattern — of 37 features in a model containing all three
tiers at once, only 8 have a coefficient whose CI clears zero, and **not one
of Tier C's 13 CoT/token-distribution features is among them.**

### Tier ablation (5.5) — A → B → C, `human_agreed` items only

*Population: `load_rq4_population()` further restricted to `human_agreed`
(D16: `human_unanimous AND n_human_votes >= 2`) — **N = 556** (79/80 question
groups), a much larger, non-population-neutral cut than it looks (69% of the
base population dropped; accuracy rises 75.8%→78.4% on this easier subset,
confirmed empirically) — so the baseline signal's own AUROC is recomputed on
this exact population rather than reused from RQ2's table, which was scored
on the full 1836-item base. Baseline = the best of the four original signals
(`analysis/rq1.py::SIGNALS`) by point AUROC on this population, selection on
the point estimate alone, never a CI-based tie-break (a signal with a
genuinely higher point estimate but a noisier CI must still win).*

| tier | model | AUROC | 95% spread (D8) |
|---|---|---|---|
| baseline (`conf_bpe`) | — | 0.8196 | [0.7746, 0.8622] |
| A | logreg | 0.8239 | [0.8155, 0.8331] |
| A | histgbm | 0.8188 | [0.8080, 0.8282] |
| B | logreg | 0.8120 | [0.7943, 0.8228] |
| B | histgbm | 0.8084 | [0.7983, 0.8176] |
| C | logreg | 0.8065 | [0.7827, 0.8196] |
| C | histgbm | 0.8048 | [0.7943, 0.8156] |

Every cell clears chance decisively — a **permutation null** (invariant 12,
n=200) centers at 0.496–0.512 across all six tier/model cells, and every
observed AUROC sits at the **100th percentile** of its own null. That rules
out "this is noise" for the whole-model AUROC, but not for the *pattern*
across tiers — Tier A looks marginally best and C marginally worst in the
table above, and that apparent decline needed its own, sharper test: a
**paired cluster bootstrap** (invariant 3, same 556 items, different
feature sets) on each step of the progression.

| model | step | Δ AUROC | 95% CI |
|---|---|---|---|
| logreg | A − baseline | +0.0053 | [−0.0310, 0.0389] |
| logreg | B − A | −0.0068 | [−0.0479, 0.0267] |
| logreg | C − B | −0.0042 | [−0.0187, 0.0106] |
| histgbm | A − baseline | +0.0051 | [−0.0157, 0.0257] |
| histgbm | B − A | −0.0052 | [−0.0270, 0.0137] |
| histgbm | C − B | +0.0008 | [−0.0135, 0.0171] |

**Every single CI crosses zero.** The apparent A>B>C decline in the raw table
is not statistically real at this sample size, for either model, at any
step. Honest reading: Tier A's own uncertainty signals already capture
whatever surface and token-distribution features have to offer — adding
them neither measurably helps nor measurably hurts. Figures:
`rq4_ablation_{model_slug}.png` (bar chart, baseline as its own bar),
`rq4_progression_{model_slug}.png` (forest plot of the 6 paired deltas),
`rq4_permutation_nulls_{model_slug}.png` (6-panel null histograms). Tables:
`rq4_ablation_{model_slug}.csv`, `rq4_ablation_progression_{model_slug}.csv`,
`rq4_ablation_null_{model_slug}.csv`.

### H4 — does the predictor's edge track human consensus? (5.6)

*Population: clean/P1, `human_label` non-null, `n_human_votes >= 2` (D9's own
population, deliberately not 5.5's `human_agreed`-restricted one — H4 needs
every contested item) — **N = 595** (79/80 groups). "The predictor" = Tier A
+ logreg specifically, a documented choice: 5.5 found no tier/model reliably
beats another, so Tier A is representative rather than arbitrary, and it
avoids a second population cut Tier B/C's `len_ratio` requirement would
force. Method (D15, mandatory): out-of-fold `P(correct)` from the fixed
10×5 repeated CV, averaged across repeats to one score per item (never
in-sample, which would bias the interaction before the CI method even
matters); fit `correct ~ oof_score * d_human`; cluster-bootstrap (B=2000,
refitting each resample) on the interaction coefficient.*

**Result: interaction coefficient +2.3752 [1.6849, 3.0739] — the CI clears
zero comfortably, H4 holds.** The predictor's edge genuinely grows with
human consensus: judge error is learnable where humans agree (epistemic,
reducible) and much less so where they don't (aleatoric, irreducible) — a
clean positive result, in real contrast to the tier ablation's null finding
above. No figure was required by the DoD; `h4_interaction_{model_slug}.png`
was added anyway (two panels, probability and log-odds) after the
probability-space panel alone visually undersold the effect — sigmoid
saturation compresses the slope difference exactly where most of the real
data sits (high `oof_score`), while the log-odds panel shows it undistorted.

### Transfer test 1 — train on clean, test on verbose (5.7)

*Population: `analysis/rq3.py::load_rq3b_items()` (**N = 1836**, the same
paired clean/verbose population RQ3b uses). **Feature-parity fix (D21,
mandatory for every model compared, not a Bayesian-specific carve-out):**
`verbose` never collects `conf_sc` or the `conf_ens` family (D19 — self-consistency
sampling and the P2/P3 ensemble are both clean/P1-only), so the feature set
here is Tier A minus those five columns — just `{conf_verb, conf_lp,
conf_bpe}` — for every model this test touches, including the Bayesian one
(5.9f). Two numbers per model on the identical reduced feature set, so the
comparison isolates the transfer effect from the feature-drop effect:
in-domain (repeated CV, D8, on `clean` alone) vs. transfer (fit once on all
of `clean`, frozen, evaluated on `verbose`, cluster-bootstrap CI over
`verbose`'s own `question_id`).*

| model | in-domain AUROC | 95% spread | transfer AUROC | 95% CI | Δ AUROC |
|---|---|---|---|---|---|
| logreg | 0.7703 | [0.7662, 0.7773] | 0.7705 | [0.7440, 0.7967] | +0.0001 |
| histgbm | 0.7889 | [0.7859, 0.7925] | 0.7635 | [0.7378, 0.7887] | −0.0254 |

**This refutes, not supports, the headline hypothesis** *("the abstention
layer trained offline degrades exactly when the judge is attacked")*.
`logreg`'s transfer AUROC is statistically indistinguishable from its own
in-domain number; `histgbm` shows a small drop but the CIs still heavily
overlap. This stands in real contrast to RQ3b's own finding that each
*individual* signal's AUROC drops significantly under `verbose` — a model
that *combines* `conf_verb`/`conf_lp`/`conf_bpe` appears to buy real
robustness no single signal has on its own, since AUROC only needs the
combined rank-ordering to survive, not each signal's own calibration.
Figure (not required by the DoD, added on request):
`rq4_transfer_{model_slug}.png`.

### Transfer test 2 — `LeaveOneGroupOut` over category (5.8)

*Population: the full RQ4 base (N=1819) — never restricted, since this test
never leaves `clean`, so the D21 exclusion doesn't apply. Full Tier A, all 8
columns. For each of the 8 MT-Bench categories in turn, fit on the other 7,
evaluate on the held-out one — fully deterministic (no shuffle/seed), one
fixed split per category, so no repeat-spread the way D8's protocol has one.*

| model | weakest category | strongest category | full range |
|---|---|---|---|
| logreg | `writing` (0.739) | `stem` (0.864) | [0.739, 0.864] |
| histgbm | `extraction` (0.753) | `math` (0.852) | [0.753, 0.852] |

**Every category clears chance comfortably; no collapse anywhere.**
Notably, `coding` is *not* the weak point for either model (logreg 0.821,
histgbm 0.795 — solidly mid-range both times) — `writing` and `reasoning`
are the real weak spots. Reading: the predictor generalizes reasonably
across task types rather than learning a "coding is hard" shortcut.
`PLAN.md` §2.4's own "coding is hard" framing turned out, on inspection, to
be an unexamined illustrative phrase carried into the plan text, not a
reasoned hypothesis about this specific judge/dataset — the figure
(`rq4_category_transfer_{model_slug}.png`, categories sorted
weakest-to-strongest) deliberately does not single `coding` out, for the
same reason. Table: `rq4_category_transfer_{model_slug}.csv`.

### Meta-model calibration and coefficients (5.9)

*Population: the full RQ4 base (N=1819). Model: **Tier C + logreg**
specifically — Tier C because it's the only tier containing all three
feature families at once, which is what a coefficient-level "which family
carries the signal" question needs; `logreg` because raw coefficients are
only directly interpretable for the linear model. Reliability diagram/ECE:
out-of-fold `P(correct)` from the standard 10×5 repeated CV, averaged
across repeats (never in-sample). Coefficients: `LogisticRegression(C=1.0)`
fit once on the full population (a coefficient is a property of one fit,
not a per-repeat quantity), CI via a purpose-built cluster bootstrap that
resamples once per replicate and refits the whole 37-coefficient vector
together (not one call per feature, which would both refit ~37× more than
necessary and wrongly treat correlated features' bootstrap draws as
independent).*

**Calibration: OOF AUROC 0.7986, ECE 0.0357 (10 effective bins)** — reasonably
well-calibrated, with the reliability curve sitting a little below the
diagonal (mild overconfidence about `P(wrong)`) in the 0.4–0.7 range.

**Coefficients: 8 of 37 have a CI excluding zero — 5/8 in Tier A, 3/16 in
Tier B, 0/13 in Tier C.** This is the coefficient-level echo of the tier
ablation's null finding above, but sharper, since it isolates individual
features rather than whole tiers: `conf_verb`, `conf_sc`, `conf_ens`
(and its exact algebraic mirror, `ens_entropy_total` — the pair is
perfectly collinear, `conf_ens = 1 - ens_entropy_total`, so standardization
splits one real effect into two equal-and-opposite coefficients, not two
independent findings), and `ens_entropy_aleatoric` carry real, non-zero
weight from Tier A. From Tier B: `abs_len_diff`, `longer_is_chosen`, and
`category_reasoning` (negative — the `reasoning` category's own dummy). From
Tier C: **nothing** — not one of the 13 CoT/token-distribution features
clears zero, at the coefficient level, exactly matching the tier-level
null. Figures: `rq4_coefficients_{model_slug}.png` (the 8 significant
coefficients only — the full 37-row version was unusably tall for a slide
at the annotated forest-plot spacing, ~35in — kept as a dense, unannotated
appendix, `rq4_coefficients_full_{model_slug}.png`) and
`reliability_rq4_meta_model_{model_slug}.png`. Table:
`rq4_coefficients_{model_slug}.csv` (all 37 rows).

### The Bayesian model joins RQ4 (5.9b, 5.9c)

*`src/bayesian.py` implements D22's hierarchical logistic regression —
`correct ~ Bernoulli(σ(α + α_q[question] + Xβ))`, question-level random
intercepts with partial pooling, `α_q ~ Normal(0, σ_q)`, fit with
NumPyro/NUTS under the identical D8 protocol. One real obstacle surfaced
during fitting, not before: the direct ("centered") form of `α_q` hit
**Neal's funnel** on real data — `max_rhat = 1.07` (flagged) at the starting
settings, and *more* warmup made it measurably worse (1.07→1.20), confirming
a posterior-geometry problem rather than an insufficient-sample-count one.
Fixed with the standard non-centered reparameterization (`α_q_raw ~
Normal(0,1)`, `α_q = σ_q · α_q_raw` as a deterministic transform —
mathematically the identical prior, decoupled geometry): `max_rhat` dropped
to 1.01, effective sample size rose ~38× (21→791), at the same settings, on
the same real fold. NUTS (fallback ladder rung 1) was sufficient throughout
— Laplace/bootstrap-ensemble fallbacks were never needed. Real, measured
runtime for the full 5-fold×10-repeat protocol (not extrapolated):
**7.0 minutes**, N=1819, Tier A — `n_repeats=10` used as originally planned.
Convergence held at that full scale: **1/50 fold-fits flagged** (barely,
`max_rhat=1.020`), **zero divergences across all 50 fits** — visible directly
in `rq4_bayesian_convergence_{model_slug}.png` (R-hat per real fold-fit,
D22's 1.01 threshold line).*

**Head-to-head, Tier A on both arms for a fair comparison:**

| | AUROC | 95% CI/spread | ECE | Brier | NLL | 90% coverage |
|---|---|---|---|---|---|---|
| frequentist `LogisticRegression` | 0.7962 | [0.7932, 0.7986] | 0.0366 | 0.1431 | — | — |
| Bayesian hierarchical | 0.7898 | [0.7848, 0.7982] | **0.0231** | 0.1432 | 0.4465 | 0.80 |

*NLL is the proper posterior-predictive log-likelihood — the mean Bernoulli
likelihood averaged across posterior draws first, then `-log`, never a
point-NLL on the mean probability (which would just be Brier with extra
steps and discard exactly what makes this quantity Bayesian). `coverage_90`
required a methodological decision D22 doesn't specify: a single 0/1 draw
can't meaningfully "fall inside" a probability interval the way a
continuous value can, so it's computed as **bin-aggregate coverage** —
reusing `ece()`'s own quantile binning, does each bin's real, aggregated
wrong-rate fall inside that bin's own pooled 90% credible interval. Both
NLL and coverage use posterior draws pooled across all 10 repeats
(concatenated, not averaged), since each repeat is an independent full
refit on a different fold partition.*

**The Bayesian model discriminates slightly worse (lower AUROC) but is
notably, measurably better calibrated** (ECE 0.023 vs. 0.037) — a genuine,
substantive difference, not a restatement of the AUROC gap. Brier is
essentially tied. The frequentist model has no native posterior, so NLL and
credible-interval coverage simply don't exist for it — this is a structural
capability the Bayesian model provides that a point-estimate model cannot,
independent of whether its AUROC wins. Table:
`rq4_bayesian_comparison_{model_slug}.csv`. Figures:
`reliability_rq4_bayesian_meta_model_{model_slug}.png`,
`rq4_bayesian_convergence_{model_slug}.png`.

### Limitations carried forward from RQ4 into RQ5's Bayesian-model sections

Two choices here were not independently re-verified once made and are worth
naming rather than treating as settled: **`α`'s own prior** (`Normal(0,1)`) —
D22's formula names `α` but never states its prior (only `α_q`/`σ_q`/`β`
are preregistered); `Normal(0,1)` was chosen to match `β`'s own scale, a
reasonable but not uniquely-determined choice. **Tier A's use for every
Bayesian-model task** — inherited from 5.5's finding that tiers don't
reliably differ for the *frequentist* models; never independently checked
whether that null result also holds for the Bayesian model specifically.
Neither is expected to change the qualitative findings above, but neither
has been empirically stress-tested either.

---

## RQ5 — Does marginalizing over the judge prompt improve uncertainty quality, and does that improvement survive distillation to single-call cost?

*Written 22 Sep 2026. RQ5's threshold-sweep task (3.2b — does epistemic
thresholding beat total thresholding on AURC?) is already written up inside
RQ2's own section above, where it sits naturally alongside RQ2's other
risk-coverage work; it is not repeated here. This section covers the three
Week 5 pieces: distillation (5.9d), human-disagreement validation (5.9e),
and verbose-shift validation (5.9f) — D23's "consequences" of the
professor-feedback pivot. Two entropy decompositions recur throughout,
sharing one formula (D20) at two different levels: **judge-level**
(`conf_ens`, `signals.py`, over the P1/P2/P3 prompt ensemble — `clean` only,
D19) and **meta-model-level** (`posterior_predictive_entropy_decomposition()`,
`src/bayesian.py`, over the Bayesian model's own posterior predictive draws
— works on any condition the model can be evaluated on, including
`verbose`). Both decompose one total entropy into Aleatoric (mean entropy
within each individual opinion) + Epistemic (the extra entropy that only
appears once you average across opinions — disagreement, not individual
uncertainty).*

**Headline: the single-call Bayesian model is not a worse version of the
3-call ensemble — on the metrics that matter for deployment it is
arguably the better one, except for the one property (recognizing
distribution shift) it was specifically supposed to have.** It retains
essentially all the ensemble's discrimination benefit at a third of the
inference cost, is far better calibrated than the ensemble's own raw
signal, and its own uncertainty is a dramatically better error-predictor
than the ensemble's judge-level uncertainty. But its uncertainty does not
correctly widen under an adversarial distribution shift — the one behavior
that would make it trustworthy specifically under attack, which is exactly
the condition where it would need to be trusted most.

### Distillation comparison — ensemble (teacher) vs. single-call Bayesian (student) (5.9d)

*Population: `load_rq4_population()` (**N = 1819**) — deliberately RQ4's own,
smaller, common population, not RQ5's own wider N=1836 (confirmed
empirically: RQ4's population is a strict subset of RQ5's, differing by
exactly the 17 known `len_ratio`-undefined items) — using the mismatched
wider population for the ensemble side would conflate "which items" with
"which method." Teacher = the 3-call ensemble's `conf_ens` /
`ens_entropy_epistemic` (D20) — `conf_ens`'s own AUROC/ECE had never been
computed anywhere in this project before this task (RQ1/RQ2's `SIGNALS`
list deliberately excludes it as D20's own separate signal). Student = the
1-call Bayesian model (5.9b, Tier A, P1-only features).*

| | AUROC | 95% CI | ECE | entropy-quality AUROC | 95% CI |
|---|---|---|---|---|---|
| ensemble (3-call) | 0.7931 | [0.7660, 0.8199] | 0.1009 | 0.5585 | [0.5141, 0.6019] |
| Bayesian (1-call) | 0.7898 (point) | — | **0.0231** | **0.7767** | [0.7472, 0.8065] |

*"Entropy-quality AUROC" = AUROC(epistemic → error) for each arm's own
epistemic signal — the ensemble's judge-level one vs. the Bayesian model's
own meta-model-level one. Every gap below is a **paired** cluster bootstrap
(invariant 3, same items, two methods), the same rename-to-a-shared-column
trick used throughout this project for one-item-set, two-score comparisons.*

**Headline: the single-call model retains 98.9% of the ensemble's AUROC edge
over chance (0.5)** — paired gap +0.0033 [−0.0116, 0.0188], CI includes
zero, not statistically distinguishable. **ECE strongly favors the Bayesian
model** — expected, since `conf_ens` is a raw judge signal that has never
itself been calibrated via cross-validation, unlike the trained OOF
meta-model. **The genuinely surprising result is entropy quality, and it
runs in the opposite direction the "how much survives" framing would
predict:** paired gap −0.2182 [−0.2659, −0.1691], entirely below zero — the
single-call model's *own* epistemic signal is a *much better* error
predictor than the expensive ensemble's judge-level epistemic signal, not
merely comparable to it. Reading: these are conceptually different
quantities that happen to share the name "epistemic." The ensemble's
captures *prompt-disagreement* — does the judge's raw verdict wobble
depending on how the question is phrased. The Bayesian model's captures the
*meta-model's own task-targeted parameter uncertainty* — how confident is a
model trained specifically to predict error about its prediction for this
item. The task-targeted signal wins decisively, even though it costs a
third as much to obtain. Figure: `rq5_distillation_{model_slug}.png`
(the Bayesian AUROC bar is deliberately shown without an error whisker,
rather than borrowing 5.9c's differently-typed D8-across-repeat-spread
interval and implying a false equivalence with the ensemble's own
cluster-bootstrap CI). Table: `rq5_distillation_{model_slug}.csv`.

### Human-disagreement validation (5.9e)

*Population: `analysis/human_disagreement.py::load_disagreement_items()`
(D9's own — clean/P1, `n_human_votes >= 2`) — **N = 595**, the same
population tasks 3.3/3.4 already use, reused directly, including its own
`d_human` float-tie rounding fix (RQ2's methods note above). Reuses task
3.4's own Spearman-plus-cluster-bootstrap recipe unmodified, over
`ens_entropy_aleatoric`/`ens_entropy_epistemic` — D23's own framing is
explicit that this is a validation against existing `d_human` machinery,
not new statistical infrastructure.*

**A genuine wording contradiction in the task specification was found and
resolved before running anything, not discovered after a wrong result.**
The task text asks whether aleatoric is high "where humans actually
disagreed (high `d_human`)" — but `d_human = |frac_prefer_a - 0.5|`
(`src/data.py`), so *high* `d_human` means *strong consensus* (votes near
0% or 100% for one side), not disagreement; the disagreement zone is *low*
`d_human`, near the 50/50 split. Confirmed directly against the formula in
source, then cross-checked against three independent descriptions that all
agree with each other and disagree only with the task text's own
parenthetical: CLAUDE.md's schema (`d_human`, "continuous consensus
strength"), D23's own plain-English framing, and H4's closeout ("the
predictor's edge grows with human *consensus*"). Tested as originally
intended: does aleatoric correlate *negatively* with `d_human` (high
aleatoric where consensus is weak)?

| signal | Spearman ρ | 95% CI |
|---|---|---|
| `ens_entropy_aleatoric` | **−0.1181** | **[−0.2041, −0.0156]** |
| `ens_entropy_epistemic` | 0.0141 | [−0.0705, 0.0985] |

**A clean validation.** Aleatoric is negatively correlated with `d_human`
(CI excludes zero, modest but real magnitude — this is a genuine, if not
large, relationship at this sample size) — the estimated aleatoric signal
does track real human disagreement. Epistemic shows no such relationship
(CI includes zero). The aleatoric/epistemic vocabulary this project has
used throughout holds up against this independent, model-free check: real
human votes were never used to build the entropy decomposition, and it
still lines up with them exactly the way the vocabulary claims it should.
Figure: `d_human_correlations_ensemble_entropy_{model_slug}.png` (extends
`plot_d_human_correlations()`, task 3.4's own figure function, via a new
`filename_suffix` parameter added specifically so this call couldn't
silently overwrite 3.4's existing figure under the same base filename).
Table: `rq5_human_disagreement_{model_slug}.csv`.

### Verbose-shift validation (5.9f)

*Population: `analysis/rq3.py::load_rq3b_items()` (**N = 1836**, the same
paired clean/verbose population the transfer test (5.7) uses). Feature set:
`TRANSFER_SAFE_COLUMNS` (D21) — the same reduction 5.7 established, applied
to the Bayesian model here as D21 itself requires. The model is fit
**once** on all of `clean` (no CV split — mirrors 5.7's own "one
offline-trained model" design), then evaluated on both `clean` itself and
`verbose`.*

**A methodological question was worked through explicitly before running
real data, not fixed reflexively.** The obvious approach — reuse
`predict_held_out()`, already built for "evaluate on data the model wasn't
trained on" — would have been a real mistake here. `clean` and `verbose`
are paired on the *exact same 80 questions* (confirmed empirically:
identical `question_id` sets, identical `item_id` sets, same row order) —
`verbose`'s rows are not unseen the way a genuinely held-out CV fold's are.
`predict_held_out()`'s marginalization would have discarded real, valid
fitted information *and* confounded the test itself: marginalization
inflates predictive spread on any input, clean or shifted alike, so an
apparent "epistemic rises on verbose" finding could just measure
marginalization noise rather than the model correctly recognizing
distribution shift. A new function, `predict_in_sample()`, uses the
model's real fitted `α_q` (via the training fold's own
`question_id → index` mapping) for both evaluations instead — isolating
the actual variable of interest, since only the *feature values* differ
between the two conditions, not the questions themselves.

A second, subtler concern was raised and checked empirically rather than
assumed: evaluating `clean` *in-sample* (the same rows the model was
fit on) has a theoretical reason to *understate* its own epistemic
uncertainty — posterior parameter uncertainty is smallest exactly where
the likelihood was fit. This would bias the test *toward* a false
"epistemic rises" confirmation, never away from one. Rather than building
an untested, more complex held-out-baseline mechanism on a theoretical
argument alone, the simple version was run first and the real result used
to decide whether the added complexity was actually necessary.

| | clean | verbose | gap (verbose − clean) | 95% CI |
|---|---|---|---|---|
| aleatoric | 0.4645 | 0.4393 | **−0.0253** | [−0.0349, −0.0159] |
| epistemic | 0.0028 | 0.0026 | **−0.0002** | [−0.0003, −0.0001] |

**The preregistered prediction did not hold, in either direction.**
Aleatoric did not stay flat — it fell, significantly. Epistemic did not
rise — it also fell, significantly (both CIs sit entirely below zero).
**This null/contrary result is strengthened, not weakened, by the
in-sample-baseline concern above**: that bias could only ever inflate an
apparent epistemic *rise*, so finding a significant *fall* despite a bias
stacked in the opposite direction makes the fall more credible, not
less — the more complex held-out-baseline check was never actually needed.
Separately notable: epistemic's absolute scale (~0.003 nats) is roughly
150× smaller than aleatoric's (~0.44–0.46 nats) on both conditions — this
simple 3-feature model's parameter uncertainty is nearly negligible next
to the irreducible per-item noise, a real finding about the model itself,
not a side effect of the shift test. Figure:
`rq5_verbose_shift_{model_slug}.png` (two panels, one per metric, on
independent y-axes — a shared axis rendered epistemic's real, significant
bars as visually indistinguishable from zero next to aleatoric's much
larger scale, caught during review and fixed before finalizing). Table:
`rq5_verbose_shift_{model_slug}.csv`.

**Reading: this Bayesian model's meta-model-level epistemic signal does not
correctly recognize the verbose distribution shift.** If anything, both
uncertainty components read *more* confident under the attack, the opposite
of what a trustworthy triage signal would do under adversarial conditions.
This is a genuine, real limitation to carry forward, not a result to
explain away — the same model that performed well in every other RQ4/RQ5
comparison above (competitive AUROC, superior calibration, a
task-targeted epistemic signal that beat the expensive ensemble's own)
fails specifically at the one property — out-of-distribution awareness —
that its Bayesian construction is supposed to provide close to "for free."

### Limitations specific to this section

- **`epistemic-AUROC` (5.9d) and `epistemic` (5.9f, in nats) are not
  directly comparable to each other** — different feature sets (Tier A's 8
  columns vs. `TRANSFER_SAFE_COLUMNS`' 3), different fitting protocols
  (10-repeat CV vs. a single fit), different units entirely (a ranking
  statistic vs. raw entropy). Both are real, correctly-computed findings on
  their own terms; neither should be read against the other's absolute
  scale.
- **Bin-aggregate coverage (5.9c/5.9d) and the pooled-draws NLL
  construction are this project's own invented methodology** — D22 doesn't
  specify either. Defensible and internally consistent, but not a
  preregistered, externally-validated method; treat `coverage_90`
  specifically as the most method-dependent number in this section.
- **5.9f used a single NUTS fit, one seed** — mirroring 5.7's own "frozen
  model" precedent deliberately, but meaning there is no measure of how
  much the reported gaps would move under a different seed's own posterior
  draw. Consistent with established project precedent; still a real,
  uncharacterized source of variance specific to this one result.

---

## RQ6 — Stress-testing the industry's calibration counterclaim (kev-8b)

*Added 22–23 Sep 2026, owner-initiated (not professor feedback) —
`DECISIONS.md` D27, `PLAN.md` §7, `TASKS.md`'s own K1–K5/GATE K addendum
block, kept outside the W0–W7 numbering specifically so it can be trimmed
without renumbering anything else if later feedback says to scope it
down.*

**Out-of-domain caveat, stated here first because it must never be a
footnote: kev-8b was never trained on pairwise response judging.** TypeSafe
AI's Jev is a proprietary, non-autoregressive "System One Model," RLCD-
trained specifically to produce calibrated typed decisions, and publicly
positioned as having solved the exact failure mode this project studies.
It discloses no weights, size, or benchmarks. `kev-8b` (`jaredpalmer/kev`,
Apache-2.0) is an independently-built open stand-in — explicitly "inspired
by the System One approach of TypeSafe's Jev," not a distillation, and
benchmarked by its own authors against real Jev output (93.21% vs. 90.12%
agreement on 324 held-out examples, vs. 66.36% for the untrained base
model) — the strongest evidence among three candidates considered that it
is a fair proxy. Two other open stand-ins (`Bespoke-Nimble-9B`,
`circuit-8b`) and a fourth candidate (`SemIf`, a cluster of non-canonical
hobbyist repos with unverifiable benchmark claims) were evaluated and
rejected on token-cap and evidentiary grounds — full comparison in
`DECISIONS.md` D27. But `kev-8b`'s own training data is Banking77, BoolQ,
AG News, customer-service tickets, NLI, spam — short-context
classification/QA, never a pairwise judgment task. Every result below is
an out-of-domain generalization test, not an apples-to-apples "best-in-
class judge" comparison.

*Population: 1,904 MT-Bench items, both AB/BA orders, `clean` and
`verbose` conditions — the harness (`src/judge_kev.py`) reuses
`load_full_items_df()`/`verbose_pad()`/`apply_order()` from the primary
judge's own pipeline unchanged, so it is the identical item set. kev-8b's
real, empirically-measured serving ceiling (not the documentation's 8,192)
is 8,160 tokens — see D27 for the full diagnostic trail, including a
genuine non-deterministic instability zone confirmed across two
independent probe sessions. This excludes 52 items (104 `verbose` call-
rows, both orders) from `verbose` only — 0 from `clean`. Two signals:
`conf_kev` (`probabilities[choice]`, symmetric in [0.5, 1], the direct
analog of `conf_lp`) and `conf_kev_bpe` (order-corrected bidirectional
entropy across AB/BA, the direct analog of `conf_bpe` — this project's own
best-performing signal). kev's own `confidence` field is deliberately
excluded throughout: confirmed from kev's actual source
(`kev/api.py::choice_confidence`) to be an exact algebraic rescaling of
`conf_kev` for a 2-option question, not an independently-trained signal —
not a judgment call, a proven redundancy. Results are reported across two
coverage regimes — **in-coverage** (`input_tokens` ≤ 1,024, kev's own
disclosed training extent) and **out-of-coverage** (1,024–8,160) — using
each item's stable, unpadded `clean`-side length, so the same item never
carries a different regime label depending on which condition is in
front of you.*

**Headline: kev-8b's best signal shows the identical vulnerability
signature the primary judge's own best signal showed — strong on
calibration and position-bias tracking, but its calibration specifically
breaks under the verbosity attack — and a Bayesian meta-model over both
signals does not beat the better one alone.** Two findings cut against the
"leaves its training range, degrades" story a naive reading of the
out-of-domain caveat might predict: calibration is not worse
out-of-coverage, and the verbosity-attack effect is directionally larger
out-of-coverage but the confidence intervals are wide enough at N=474–526
that this should be read as suggestive, not confirmed.

### Calibration check

*Population: N=1,836 (`clean`, human-labeled — identical population size
to RQ1's own, same filter). `analysis/rq6.py::main_calibration`, reusing
`src/metrics.py::ece`/`brier`/`overconfidence_gap`/`auroc_error` unchanged
— `conf_kev_bpe`'s raw-nats range ([1−ln 2, 1] ≈ [0.307, 1]) needs no
rescaling before these, confirmed the same way `conf_bpe`'s own range
already was.*

| Regime | Signal | ECE | 95% CI | Overconfidence gap | AUROC | 95% CI |
|---|---|---|---|---|---|---|
| in-coverage (N=1,310) | `conf_kev` | 0.1525 | [0.1247, 0.1814] | +0.1525 | 0.6972 | [0.6596, 0.7335] |
| in-coverage | `conf_kev_bpe` | **0.0980** | [0.0763, 0.1158] | **−0.0647** | **0.7722** | [0.7407, 0.8005] |
| out-of-coverage (N=526) | `conf_kev` | 0.1297 | [0.0942, 0.1766] | +0.1297 | 0.7074 | [0.6472, 0.7652] |
| out-of-coverage | `conf_kev_bpe` | **0.1015** | [0.0747, 0.1451] | **−0.0835** | **0.7806** | [0.7231, 0.8313] |

**`conf_kev_bpe` beats `conf_kev` on every metric, in both regimes.** This
replicates, on a completely different non-autoregressive architecture, the
exact pattern this project already found for the primary Qwen judge — its
own bidirectional-entropy signal, `conf_bpe`, was RQ1/RQ2's best performer
too. `conf_kev` is meaningfully overconfident (gap +0.13 to +0.15);
`conf_kev_bpe` is mildly *underconfident* instead (gap −0.06 to −0.08) —
neither is well calibrated in an absolute sense, but the direction of the
miscalibration flips between the two signals. **Out-of-coverage is not
worse than in-coverage** — AUROC is marginally *higher* out of range for
both signals, though the CIs overlap substantially at N=526 vs. 1,310.
This does not support a simple "leaves its training range, degrades"
story for calibration specifically. Figures:
`reliability_conf_kev{,_bpe}_{in,out_of}_coverage_kev_8b.png`. Table:
`rq6_calibration_kev_8b.csv`.

### Position-swap attack

*Same population and recipe as RQ3a (`analysis/rq3.py::compute_flip_rate`/
`compute_confidence_gap`, reused unchanged — both are already generic over
any DataFrame carrying `flipped`/`question_id`/a named signal column,
which `items_kev-8b.parquet` provides under the identical names).*

| Regime | Flip rate | 95% CI | Signal | Confidence gap (flipped − unflipped) | 95% CI |
|---|---|---|---|---|---|
| in-coverage | 24.6% | [19.5%, 30.0%] | `conf_kev` | **−0.0964** | [−0.1174, −0.0777] |
| in-coverage | " | " | `conf_kev_bpe` | **−0.4901** | [−0.5101, −0.4675] |
| out-of-coverage | 21.1% | [15.5%, 28.2%] | `conf_kev` | **−0.0798** | [−0.1167, −0.0453] |
| out-of-coverage | " | " | `conf_kev_bpe` | **−0.4659** | [−0.5027, −0.4262] |

**Both signals track their own position-bias-induced errors strongly, and
significantly, in both regimes** (every CI excludes zero). This is a
notably *stronger* result than the primary judge's own RQ3a finding: its
`conf_verb` gap was −0.0199 [−0.0248, −0.0152] — `conf_kev`'s gap
(−0.096) is roughly **5× larger** on the identical test. This diagnostic
direction was foreshadowed by an earlier, informal probe during token-cap
diagnostics (D27): with content-free filler padding, kev-8b's verdict was
found to be 100% determined by which side of the input the real content
sat on. That earlier finding was explicitly a probe-script observation,
not a controlled result — this section is the controlled version, with
real MT-Bench content on both sides and a proper cluster-bootstrap CI,
and it confirms substantial, statistically real position-sensitivity,
even if not the literal 100%-deterministic effect the content-free probe
showed. Figures: `rq6_position_swap_gap_kev_8b_{in,out_of}_coverage.png`.
Table: `rq6_position_swap_kev_8b.csv`.

### Verbosity attack

*Population: 1,784 paired items — the intersection of `clean` and
`verbose` (D27's pairing requirement: an item missing on either side is
dropped from **both** sides for this specific test, not just the missing
side, or the paired bootstrap silently loses its pairing). Reuses
`analysis/rq3.py::compute_signal_rq3b_metrics` unchanged.*

| Regime | Signal | ΔECE (verbose − clean) | 95% CI | ΔAUROC | 95% CI |
|---|---|---|---|---|---|
| in-coverage (n=1,310) | `conf_kev` | +0.0092 | [−0.0088, 0.0267] | −0.0058 | [−0.0452, 0.0328] |
| in-coverage | `conf_kev_bpe` | **+0.0330** | **[0.0125, 0.0584]** | +0.0055 | [−0.0116, 0.0259] |
| out-of-coverage (n=474) | `conf_kev` | +0.0213 | [−0.0188, 0.0691] | −0.0241 | [−0.0868, 0.0406] |
| out-of-coverage | `conf_kev_bpe` | **+0.0678** | **[0.0206, 0.1091]** | +0.0050 | [−0.0324, 0.0443] |

**`conf_kev_bpe`'s calibration breaks significantly under the verbosity
attack, in both regimes; `conf_kev` shows no significant change in
either.** This is a striking parallel to the primary judge's own RQ3b
finding: there too, the *best* clean-data signal (`conf_bpe`, ECE +0.0432
[0.0144, 0.0516]) was the one whose calibration broke under the identical
attack, while the weaker raw signals were comparatively unaffected. The
effect is directionally larger out-of-coverage (+0.068 vs. +0.033), though
the CIs are wide enough at this N that this should be read as suggestive
of a coverage-dependent effect, not confirmed as one. AUROC does not move
significantly for either signal in either regime — unlike the primary
judge, where AUROC dropped significantly for all three surviving signals
under the same attack; this attack degrades kev-8b's *calibration*
specifically, not its *discrimination*. Accuracy itself does not move
significantly either (Δ −0.014 to −0.025, both CIs include zero) —
verbosity does not make kev-8b more wrong, the same qualitative pattern
the primary judge showed. Figures:
`rq6_verbosity_deltas_kev_8b_{in,out_of}_coverage.png`. Table:
`rq6_verbosity_kev_8b.csv`.

### Bayesian recalibration check

*Full D8 protocol (`StratifiedGroupKFold(5)` × 10 repeats, NUTS) — the
same rigor as the primary judge's own RQ4 Bayesian arm, not a lighter
version, per explicit confirmation given kev's much smaller 2-feature set.
Reuses `repeated_stratified_group_kfold_bayesian`/`build_xyg`/
`encode_features` unchanged; `encode_features` is a no-op on two pure-
float columns, the same way it already is on the primary study's own
Tier A.*

| Regime | Meta-model AUROC | D8 spread | Best single signal AUROC | Fold-fits flagged |
|---|---|---|---|---|
| in-coverage | 0.7723 | [0.7697, 0.7743] | 0.7722 | **0/50** |
| out-of-coverage | 0.7755 | [0.7663, 0.7857] | 0.7806 | **0/50** |

**The meta-model does not beat the best single signal in either regime**
— essentially tied in-coverage, slightly *worse* out-of-coverage.
Convergence is clean throughout (0/50 fold-fits flagged, both regimes —
the same diagnostic that caught a real problem once already in this
project, Neal's funnel, found nothing here). This null is methodologically
sound, not a red flag to explain away: `conf_kev` is a function of one
number (`prob_a` from the AB call alone), `conf_kev_bpe` a function of two
(`prob_a` from AB *and* BA, order-corrected) — they share one of two
inputs, correlated but not collinear the way `confidence`/
`probabilities[choice]` was (confirmed exact algebraic identity, which is
why that one was excluded entirely rather than just noted). D22's proper
`Normal(0,1)` priors handle correlated, non-identical predictors without a
non-identifiability pathology, and the clean convergence diagnostics are
the empirical confirmation of that, not an assumption. The check also
never interprets individual coefficients (unlike this project's own
`conf_ens`/`ens_entropy_total` collinearity, D20, where that really would
have split one effect into two misleading numbers) — it only compares
holistic AUROC, a comparison correlated features don't invalidate. Two
signals built from largely overlapping information having limited
independent value to combine is the expected, coherent outcome, not a
modeling failure. Figures:
`rq4_bayesian_convergence_kev_8b_{in,out_of}_coverage.png`. Table:
`rq6_bayesian_recalibration_kev_8b.csv`.

### Limitations specific to this section

- **Every result here is an out-of-domain generalization test, restated:**
  kev-8b was never trained on pairwise response judging. A weaker
  or absent effect could mean the counterclaim holds, or could mean the
  task is simply unfamiliar to the model in a way that has nothing to do
  with calibration quality — this project cannot fully separate the two
  explanations, and neither should any reading of the results above.
- **The out-of-coverage regime is the smaller population throughout**
  (N=474–526 vs. 1,310) — every regime-comparison claim above ("worse
  out-of-coverage," "not worse out-of-coverage") is reported honestly with
  its own CI, but should be read as suggestive at this sample size, not as
  confirmed as strongly as the in-coverage numbers.
- **The regime boundary (1,024 tokens) comes from kev's own GitHub
  README, not a peer-reviewed source** — the best available disclosure,
  not an independently-verified ground truth for where kev-8b's
  calibration genuinely starts to degrade.
- **The position-bias diagnostic finding (D27, K2) that motivated the
  position-swap test was a content-free filler-padding probe, not a
  controlled experiment** — this section's own position-swap result (real
  content, proper CIs) is the controlled version and is what should be
  cited as the actual finding; the earlier probe should be read only as
  the reason this test was prioritized, not as evidence in its own right.
- **`kev-8b` is one open stand-in for one proprietary system.** Its own
  authors' 90–93% agreement with real Jev output is the best available
  evidence it is a fair proxy, but it is not a guarantee that every
  finding here would replicate against Jev itself, which remains
  untested and untestable given its proprietary status.

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

### Decoding ablation — constrained vs. free-form (task 4.5, D25) — 18 Sep 2026

**Purpose:** D25 found that JSON-schema-constrained decoding renormalizes the
verdict-position logprobs, inflating `conf_lp`. That's a known, bounded distortion of
one *signal*. The open question D25 left for task 4.5 to resolve: does constraining
also change the judge's actual **verdict**, not just how confident `conf_lp` reports
it was?

**Method:** 100 items (seeded sample, clean/P1/AB/greedy - one call per item per arm),
generated twice each: once with the production JSON schema enforced
(`structured_outputs`, exactly what every other call in this project uses), once
completely free-form (schema removed, same prompt - which already asks for the JSON
object in its instructions - same temperature=0, same seed). Both arms scored with the
unmodified production parser (`parse.py::parse_verdict_and_confidence()`), so "did
free-form generation still produce parseable JSON" is answered by the actual parser
every real result depends on, not a purpose-built lenient one. `src/ablation_decoding.py`
(Colab) + `analysis/decoding_ablation.py` (local).

**Result:**

| | n | parse rate |
|---|---|---|
| constrained | 100 | 100% |
| free-form | 100 | 100% |

**Verdict agreement (among the 100/100 items where both parsed): 96.0% (4 disagreements).**
All four are far from marginal - both arms report **high stated confidence (0.90–0.95)
on both sides of every flip**, not a low-confidence coin toss that happened to land
differently. Since both arms use greedy decoding (temperature=0) with the same seed,
any disagreement is a genuine causal effect of the schema mask on the decoding path at
some branch point, not sampling noise.

**Conclusion:** two findings, in opposite directions:

1. **Parse rate is a non-issue here.** Free-form generation held to the JSON output
   contract just as reliably as schema-enforced generation, on this population (clean,
   single greedy call). This doesn't generalize to every condition without re-checking
   - `verbose`'s longer inputs, or the sampled (`temperature=0.7`) calls, could behave
   differently - but for the primary greedy case, constraining buys no parse-rate
   safety margin that wasn't already there.
2. **Constraining does move the verdict, in a small but real fraction of cases (4%).**
   D25's original concern (does the schema mask distort more than `conf_lp`'s reported
   scale) is confirmed, not dismissed - four items out of 100 get a different, equally
   confidently-held answer depending on whether the output format is enforced. At
   N=100 this is a descriptive rate, not a precisely bounded population estimate (no
   bootstrap CI is reported - see this section's own module docstring for why), but it
   is real evidence, not a null result: **every calibration/AUROC number in this report
   is conditional on the constrained-decoding arm specifically**, and roughly 1 in 25
   items would have gotten a different, comparably confident verdict under free-form
   generation instead. Per D25's own decision, this is evidence-based grounds to revisit
   the `vllm` pin (for `logprobs_mode="raw_logprobs"`) in a future run, not something to
   act on inside this project's remaining timeline.
