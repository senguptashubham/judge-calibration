"""Tests for src/bayesian.py: a held-out question's random intercept must be
marginalized over the population prior, never its would-be fitted value
(DECISIONS.md D22) - the hierarchical-model analogue of test_predictor.py's
no-leakage assertion. Also convergence-diagnostics presence (R-hat, ESS,
divergence count) for every fold-fit. See TASKS.md task 5.9b.
"""
