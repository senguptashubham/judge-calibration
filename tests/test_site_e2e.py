"""End-to-end tests for the static site (site/), driven by Playwright.

They open site/index.html straight from disk, as a visitor without a server
would, and walk the user flows. Expected values are read from the page's own
data (window.SITE_DATA), never hard-coded, so a data rebuild doesn't break them.

Skipped when Playwright or a Chromium browser is unavailable, or the site data
has not been built (`python -m analysis.site_data --config configs/run.yaml`).
Every test also fails on any page error or console error.
"""

from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

SITE = Path(__file__).resolve().parents[1] / "site"
URL = (SITE / "index.html").as_uri()


@pytest.fixture(scope="module")
def browser():
    if not (SITE / "data" / "site-data.js").exists():
        pytest.skip("site data not built")
    with sync_api.sync_playwright() as p:
        launched = None
        for options in ({"channel": "chrome"}, {}):   # installed Chrome first, then Playwright's own
            try:
                launched = p.chromium.launch(headless=True, **options)
                break
            except Exception:
                continue
        if launched is None:
            pytest.skip("no Chromium browser available")
        yield launched
        launched.close()


@pytest.fixture
def open_site(browser):
    pages = []

    def _open(width=1440, height=900, **context_options):
        context = browser.new_context(viewport={"width": width, "height": height}, **context_options)
        page = context.new_page()
        page.errors = []
        page.requests = []
        page.on("pageerror", lambda e: page.errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: m.type == "error" and page.errors.append(f"console: {m.text}"))
        page.on("request", lambda r: page.requests.append(r.url.rsplit("/", 1)[-1]))
        page.goto(URL)
        page.wait_for_timeout(300)
        pages.append(page)
        return page

    yield _open
    errors = [e for page in pages for e in page.errors]
    for page in pages:
        page.context.close()
    assert errors == []


def wait_until_scroll_settles(page):
    """Resolve once the page has not scrolled for 6 animation frames - a fixed
    sleep is flaky for smooth scrolling on a busy machine."""
    page.evaluate("""() => new Promise(resolve => {
        let last = scrollY, still = 0;
        const tick = () => {
            if (scrollY === last) { if (++still > 6) return resolve(); } else { still = 0; last = scrollY; }
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    })""")


def scroll_to(page, selector, offset=80):
    page.evaluate(f"window.scrollTo({{top: document.querySelector('{selector}').getBoundingClientRect().top"
                  f" + scrollY - {offset}, behavior: 'instant'}})")
    page.wait_for_timeout(150)


# --- page and data ------------------------------------------------------------------

def test_headline_numbers_and_readouts_come_from_the_site_data(open_site):
    page = open_site()
    bound = page.eval_on_selector_all("[data-bind]", "els => els.map(e => e.textContent)")
    assert bound and all(text and text != "—" for text in bound)
    expected = page.evaluate("""() => {
        const s = SITE_DATA.auto_accept.find(s => s.judge === 'Qwen2.5-7B' && s.signal === 'conf_verb' && s.condition === 'clean');
        return Math.round(s.slip_through[SITE_DATA.thresholds.indexOf(0.9)] * 100) + '%'; }""")
    assert page.inner_text("#ro-slip") == expected


# --- 01 trick the judge -------------------------------------------------------------

def test_swapping_the_order_flips_the_first_example(open_site):
    page = open_site()
    page.click("label:has(#swap)")
    assert "flip" in page.get_attribute("#banner", "class")


def test_reveal_marks_the_humans_pick_and_stays_on_across_examples(open_site):
    page = open_site()
    page.click("#reveal")
    assert page.locator("#trick .answer.human-choice").count() == 1
    page.click("#next")
    assert page.locator("#trick .answer.human-choice").count() == 1
    assert page.get_attribute("#reveal", "aria-pressed") == "true"


def test_random_examples_are_fetched_only_when_asked_for(open_site):
    page = open_site()
    assert "examples.js" not in page.requests
    page.click("#random")
    page.wait_for_function("document.getElementById('ex-count').textContent === 'Random example'")
    page.click("#random")
    page.wait_for_timeout(100)
    assert page.requests.count("examples.js") == 1
    page.click("#next")
    assert page.inner_text("#ex-count").startswith("Example ")


# --- 02 you vs the judge ------------------------------------------------------------

def test_a_game_round_shows_each_result_in_view_and_ends_in_the_scoreboard(open_site):
    page = open_site()
    scroll_to(page, "#game .game-status")
    for _ in range(5):
        page.click("#g-answer-A .pick-btn")
        page.click("#g-chips button >> nth=3")
        page.wait_for_timeout(50)    # let the smooth scroll to the result start
        wait_until_scroll_settles(page)
        top, bottom, bar = page.evaluate("""() => { const r = document.getElementById('g-result').getBoundingClientRect();
            return [r.top, r.bottom, document.querySelector('.topbar').offsetHeight]; }""")
        assert top >= bar - 1 and bottom <= page.viewport_size["height"] + 1
        page.click("#g-result .btn.primary")
    assert page.is_visible("#g-final") and not page.is_visible("#g-play")
    assert page.locator("#g-pips li.right, #g-pips li.wrong").count() == 5


# --- 03 three judges ----------------------------------------------------------------

def test_three_judge_grid_has_a_cell_per_judge_and_presentation(open_site):
    page = open_site()
    assert page.locator("#s-grid .vg-cell").count() == 12
    total = page.evaluate("SITE_DATA.showcase.length")   # the stepper starts on the showcase
    page.click("#s-next")
    assert page.inner_text("#s-count").startswith("2 / ") and int(page.inner_text("#s-count").split("/ ")[1]) > total


# --- 04 ship it ---------------------------------------------------------------------

def test_threshold_keys_move_between_thresholds_where_the_result_changes(open_site):
    page = open_site()
    stops = page.evaluate("""() => {
        const s = SITE_DATA.auto_accept.find(s => s.judge === 'Qwen2.5-7B' && s.signal === 'conf_verb' && s.condition === 'clean');
        const a = s.accepted_share, last = a.length - 1;
        return a.map((_, i) => i).filter(i => i === last || a[i] !== a[i + 1]).map(i => SITE_DATA.thresholds[i].toFixed(2)); }""")
    start = stops.index(page.inner_text("#gate-value"))
    page.focus("#threshold")
    page.keyboard.press("ArrowRight")
    assert page.inner_text("#gate-value") == stops[start + 1]
    page.keyboard.press("ArrowLeft")
    page.keyboard.press("ArrowLeft")
    assert page.inner_text("#gate-value") == stops[start - 1]


def test_chart_lines_draw_in_again_on_a_new_judge(open_site):
    page = open_site()
    scroll_to(page, "#curve", 200)
    page.wait_for_timeout(1500)
    width = lambda: float(page.get_attribute("#reveal-clip rect", "width"))
    full = width()
    page.locator("#cost-judge-tabs button").nth(1).click()
    page.wait_for_timeout(200)
    assert width() < full


# --- 05 method ----------------------------------------------------------------------

@pytest.mark.parametrize("width", [390, 1440])
def test_method_connectors_start_and_end_on_box_edges(open_site, width):
    page = open_site(width=width)
    page.click("#m-kev")
    assert page.get_attribute("#m-kev", "aria-pressed") == "true"
    loose = page.evaluate("""() => {
        const nodes = [...document.querySelectorAll('.m-node')].map(n => {
            const l = n.offsetLeft + n.offsetParent.offsetLeft, t = n.offsetTop + n.offsetParent.offsetTop;
            return { l, r: l + n.offsetWidth, t, b: t + n.offsetHeight }; });
        // On a box: inside its rectangle (2px slack) and within 2px of one of its sides.
        const onEdge = (x, y) => nodes.some(q => {
            const inside = x >= q.l - 2 && x <= q.r + 2 && y >= q.t - 2 && y <= q.b + 2;
            const nearSide = [q.l, q.r].some(e => Math.abs(x - e) < 2) || [q.t, q.b].some(e => Math.abs(y - e) < 2);
            return inside && nearSide; });
        return [...document.querySelectorAll('#m-edges path')].filter(p => {
            const a = p.getPointAtLength(0), z = p.getPointAtLength(p.getTotalLength());
            return !(onEdge(a.x, a.y) && onEdge(z.x, z.y)); }).length; }""")
    assert loose == 0


# --- layout -------------------------------------------------------------------------

@pytest.mark.parametrize("width", [1180, 1440])
def test_control_bars_fit_on_one_row_on_laptop_screens(open_site, width):
    """Controls are sized against the bundled fonts; a wider font once pushed the
    padding switch onto a second row."""
    page = open_site(width=width)
    rows = """sel => new Set([...document.querySelector(sel).children].filter(k => k.offsetParent)
        .map(k => { const r = k.getBoundingClientRect(); return Math.round((r.top + r.height / 2) / 20); })).size"""
    for judge in range(3):   # the signal list, and so the dropdown's content, changes per judge
        page.locator("#cost-judge-tabs button").nth(judge).click()
        assert page.evaluate(rows, "#cost .control-bar") == 1
    assert page.evaluate(rows, "#trick .control-bar") == 1


# --- theme, mobile, motion ----------------------------------------------------------

def test_theme_follows_the_device_until_chosen_and_the_choice_survives_a_reload(open_site):
    page = open_site(color_scheme="dark")
    background = lambda: page.evaluate("getComputedStyle(document.body).backgroundColor")
    dark = background()
    page.click("#theme-btn")
    light = background()
    assert light != dark
    page.reload()
    page.wait_for_timeout(200)
    assert background() == light


@pytest.mark.parametrize("width", [360, 820])
def test_narrow_screens_fit_and_fold_the_menu(open_site, width):
    page = open_site(width=width, height=844, is_mobile=True, has_touch=True)
    assert page.evaluate("document.documentElement.scrollWidth - innerWidth") == 0
    overlaps = page.evaluate("""() => {
        const r = [...document.querySelectorAll('.pipeline .node')].map(e => e.getBoundingClientRect());
        let hits = 0;
        for (let i = 0; i < r.length; i++) for (let j = i + 1; j < r.length; j++)
            if (r[i].left < r[j].right && r[j].left < r[i].right && r[i].top < r[j].bottom && r[j].top < r[i].bottom) hits++;
        return hits; }""")
    assert overlaps == 0
    assert not page.is_visible(".topbar nav a >> nth=0")
    page.click("#menu-btn")
    assert page.is_visible(".topbar nav a >> nth=0")
    page.keyboard.press("Escape")
    assert page.get_attribute("#menu-btn", "aria-expanded") == "false"


def test_reduced_motion_shows_everything_without_animating(open_site):
    page = open_site(reduced_motion="reduce")
    scroll_to(page, "#curve", 200)
    rect_width = float(page.get_attribute("#reveal-clip rect", "width"))
    assert rect_width == pytest.approx(page.evaluate("document.getElementById('curve').clientWidth"), abs=1)
