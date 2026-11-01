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
    assert to_hex(light.get_facecolor()) == plots.LIGHT.rc["figure.facecolor"]
    assert to_hex(dark.get_facecolor()) == plots.DARK.rc["figure.facecolor"]


def test_a_figure_saved_with_a_footnote_margin_gets_the_suffix_too(figures_in_tmp):
    # plot_confidence_gap() passes _save() a layout rect that leaves room for its footnote.
    panel = {"title": "", "signals": ["conf_verb"], "gap": [-0.02], "ci_low": [-0.03], "ci_high": [-0.01]}

    plots.plot_confidence_gap([panel], title="t", filename="gap.png", dark=True)

    assert (figures_in_tmp / "gap_dark.png").exists()
    assert not (figures_in_tmp / "gap.png").exists()


def test_theme_resets_even_when_a_plot_fails():
    with pytest.raises(Exception):
        plots.plot_risk_coverage({}, (None, None), dark=True)

    assert plots._theme is plots.LIGHT


def test_single_judge_forests_colour_by_direction_and_hollow_the_non_significant():
    # ΔECE: > 0 is worse. Rows: significantly worse, significantly better, CI spans 0.
    colors, hollow = plots._direction_styles([0.01, -0.05, -0.01], [0.03, -0.02, 0.02], worse_if_positive=True)

    assert colors == [plots.LIGHT.accent, plots.LIGHT.works, plots.LIGHT.muted]
    assert hollow == [False, False, True]


def test_a_long_title_over_a_right_shifted_axes_is_kept_inside_the_figure(figures_in_tmp):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_yticks([0, 1], ["a very long tick label", "another long tick label"])
    ax.set_title("A title long enough to run off the right edge of this figure")

    plots._save(fig, "t.png")

    box = ax.title.get_window_extent(fig.canvas.get_renderer())
    assert 0 <= box.x0 and box.x1 <= fig.bbox.width


def test_touching_tick_labels_are_thinned(figures_in_tmp):
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.set_xlim(0, 10)
    ax.set_xticks(range(11), [f"{i}.000" for i in range(11)])

    plots._save(fig, "t.png")

    assert len(ax.get_xticks()) < 11
