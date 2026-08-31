"""RQ4: Bayesian hierarchical logistic regression (NumPyro/NUTS), with
question-level random intercepts, fit under the same repeated
StratifiedGroupKFold protocol as src/predictor.py.

Fallback ladder, preregistered: NUTS -> Laplace approximation -> bootstrap
ensemble of logistic regressions (DECISIONS.md D22).

Also home to the meta-model-level Total/Aleatoric/Epistemic entropy
decomposition, computed from posterior predictive draws (DECISIONS.md D20,
D22) - the counterpart to signals.py's judge-level (conf_ens) decomposition.

Runs entirely locally (CPU) - no GPU, no Colab (DECISIONS.md D24).
See TASKS.md tasks 5.9b-5.9f.
"""
