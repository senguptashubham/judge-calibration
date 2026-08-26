"""Tests for src/predictor.py: no question_id in both train and test of any
fold, and two different seeds produce different fold assignments (the
GroupKFold-has-no-shuffle trap, DECISIONS.md D8). See TASKS.md task 5.3.
"""
