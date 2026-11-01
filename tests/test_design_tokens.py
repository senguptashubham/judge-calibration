"""Tests for design/tokens.json, the colour roles shared by the site and the
figures: the site's CSS carries exactly these values in both themes, and every
role keeps its contrast on its own theme's paper."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKENS = json.loads((ROOT / "design" / "tokens.json").read_text(encoding="utf-8"))
CSS = (ROOT / "site" / "style.css").read_text(encoding="utf-8")

# Text roles need 4.5:1 (WCAG AA, small text) on every surface text sits on;
# marks (lines, dots, bars) need 3:1 on the paper; white text on a fill needs 4.5:1.
TEXT = ["ink", "ink-2", "ink-3", "accent-text", "works-text"]
SURFACES = ["paper", "paper-2", "window"]
MARKS = ["accent", "works", "qwen", "signal-2", "signal-3"]
# Colours only the figures use (src/plots.py), so the site's CSS doesn't carry them.
FIGURE_ONLY = {"signal-2", "signal-3"}
FILLS = ["accent", "qwen-fill", "works-fill", "kev-fill", "autoj-fill"]


def _block(selector_pattern: str) -> dict[str, str]:
    match = re.search(selector_pattern + r"\s*\{(.*?)\}", CSS, re.S)
    assert match, selector_pattern
    return dict(re.findall(r"--([\w-]+):\s*([^;]+);", match.group(1)))


def _luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted([_luminance(a), _luminance(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_css_carries_the_token_values_in_both_themes():
    light = _block(r"(?m)^:root")
    media_dark = _block(r':root:not\(\[data-theme="light"\]\)')
    toggle_dark = _block(r':root\[data-theme="dark"\]')

    assert media_dark == toggle_dark
    for role, value in TOKENS["light"].items():
        if role not in FIGURE_ONLY:
            assert light[role] == value, role
    for role, value in TOKENS["dark"].items():
        if role not in FIGURE_ONLY:
            assert {**light, **media_dark}[role] == value, role


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_every_role_keeps_its_contrast_on_its_paper(theme):
    t = TOKENS[theme]
    for role in TEXT:
        for surface in SURFACES:
            assert contrast(t[role], t[surface]) >= 4.5, (role, surface)
    for role in MARKS:
        assert contrast(t[role], t["paper"]) >= 3.0, role
    for role in FILLS:
        assert contrast("#ffffff", t[role]) >= 4.5, role
    # Amber and teal marks sit under 3:1 on the light paper by design (each data
    # line carries a text label); on the dark paper they pass.
    if theme == "dark":
        for role in ["kev", "autoj"]:
            assert contrast(t[role], t["paper"]) >= 3.0, role
