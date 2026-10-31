"""Tests for src/plots.py's dark variants. The figures' content is checked by
eye and by the byte-identical reproduction of results/figures/; these cover
the theme switch itself."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from src import plots


@pytest.fixture(autouse=True)
def figures_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    yield tmp_path
    plt.close("all")


def test_dark_saves_a_suffixed_copy_on_the_dark_background(figures_in_tmp):
    curves = {"conf_verb": (np.array([0.5, 1.0]), np.array([0.2, 0.25]))}
    oracle = (np.array([0.5, 1.0]), np.array([0.0, 0.25]))

    light = plots.plot_risk_coverage(curves, oracle, filename="rc.png")
    dark = plots.plot_risk_coverage(curves, oracle, filename="rc.png", dark=True)

    assert (figures_in_tmp / "rc.png").exists()
    assert (figures_in_tmp / "rc_dark.png").exists()
    assert to_hex(light.get_facecolor()) == "#ffffff"
    assert to_hex(dark.get_facecolor()) == plots.DARK_RC["figure.facecolor"]


def test_figures_that_save_without_save_helper_get_the_suffix_too(figures_in_tmp):
    # plot_confidence_gap() calls savefig itself rather than _save().
    panel = {"title": "", "signals": ["conf_verb"], "gap": [-0.02], "ci_low": [-0.03], "ci_high": [-0.01]}

    plots.plot_confidence_gap([panel], title="t", filename="gap.png", dark=True)

    assert (figures_in_tmp / "gap_dark.png").exists()
    assert not (figures_in_tmp / "gap.png").exists()


def test_theme_resets_even_when_a_plot_fails():
    with pytest.raises(Exception):
        plots.plot_risk_coverage({}, (None, None), dark=True)

    assert plots._theme is plots.LIGHT
