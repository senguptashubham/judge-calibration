/* The page only looks values up: every number comes from data/site-data.js,
   which analysis/site_data.py computes with the project's own analysis code. */
(() => {
  "use strict";

  const D = window.SITE_DATA;
  if (!D) {
    document.querySelector("main").innerHTML =
      "<p class='lede'>site/data/site-data.js is missing - build it with <code>python -m analysis.site_data</code>.</p>";
    return;
  }

  const JUDGES = ["Qwen2.5-7B", "kev-8b", "auto-j-13b"];
  const JUDGE_VAR = { "Qwen2.5-7B": "--qwen", "kev-8b": "--kev", "auto-j-13b": "--autoj" };
  const KIND_LABELS = {
    order: "Swapping the order flips it",
    padding: "Padding the answers flips it",
    identical: "Identical answers, invented difference",
    judges: "The three judges disagree",
    random: "Random short pick, not chosen for a flip",
  };
  const AUTO_ACCEPT_AT = 0.9;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const categoryLabel = (item) => {
    const name = item.category === "stem" ? "STEM" : item.category[0].toUpperCase() + item.category.slice(1);
    return `Category: ${name}${item.turn === 2 ? " · judged on turn 2" : ""}`;
  };
  /** A two-line sticker: a short title over a detail. */
  const sticker = (kind, title, detail) => h("span", { class: `stamp${kind ? ` ${kind}` : ""}` }, h("b", {}, title), detail);
  const humansSticker = (human) => sticker("human", "humans' pick", `${human.n_votes} / ${human.n_votes} votes`);
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const judgeColor = (judge) => cssVar(JUDGE_VAR[judge]);
  // A judge's colour behind text uses its deeper fill shade (see style.css).
  const judgeFill = (judge) => cssVar(`${JUDGE_VAR[judge]}-fill`);
  const onJudgeColor = (judge) => cssVar(JUDGE_VAR[judge].replace("--", "--on-"));
  const pct = (x) => (x === null || x === undefined || Number.isNaN(x) ? "—" : `${Math.round(x * 100)}%`);
  const $ = (id) => document.getElementById(id);

  /** Build an element; string children become text nodes (never HTML). */
  function h(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (key === "class") node.className = value;
      else if (key === "style") Object.assign(node.style, value);
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value);
    }
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  // --- halftone background: a dot grid in the side margins, with a
  // slow travelling wave that swells and shrinks the dots -------------------------
  (() => {
    const canvas = document.querySelector("canvas.dots");
    const ctx = canvas.getContext("2d");
    const GAP = 13, R_MAX = 2.6;
    let W = 0, H = 0, pts = [], running = false, last = 0;

    function measure() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = window.innerWidth; H = window.innerHeight;
      canvas.width = W * dpr; canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      pts = [];
      const main = document.querySelector("main");
      const columnHalf = main ? main.clientWidth / 2 - parseFloat(getComputedStyle(main).paddingLeft) : W / 2;
      for (let y = GAP / 2; y < H; y += GAP) {
        for (let x = GAP / 2; x < W; x += GAP) {
          // Dots live in the side margins and fade to nothing over the text column,
          // top to bottom, so they never sit behind anything you read.
          const outside = Math.abs(x - W / 2) - columnHalf;
          const w = Math.min(1, Math.max(0, (outside + 10) / 110));
          if (w > 0.02) pts.push(x, y, w);
        }
      }
    }

    let dotColor = cssVar("--dots");
    window.addEventListener("themechange", () => { dotColor = cssVar("--dots"); draw(performance.now() / 1000); });
    function draw(t) {
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = dotColor;
      ctx.beginPath();
      for (let i = 0; i < pts.length; i += 3) {
        const x = pts[i], y = pts[i + 1];
        const wave = 0.6 + 0.28 * Math.sin(x * 0.011 + y * 0.007 - t * 0.7) + 0.12 * Math.sin(x * 0.004 - y * 0.013 + t * 0.45);
        const r = R_MAX * pts[i + 2] * wave;
        if (r < 0.4) continue;
        ctx.moveTo(x + r, y);
        ctx.arc(x, y, r, 0, Math.PI * 2);
      }
      ctx.fill();
    }

    function frame(now) {
      if (!running) return;
      if (now - last > 40) { draw(now / 1000); last = now; }   // ~25 fps is plenty for a slow wave
      requestAnimationFrame(frame);
    }
    function setRunning(on) {
      if (reducedMotion) { draw(0); return; }
      if (on && !running) { running = true; requestAnimationFrame(frame); }
      if (!on) running = false;
    }
    measure();
    draw(0);
    setRunning(true);
    window.addEventListener("resize", () => { measure(); draw(performance.now() / 1000); });
    document.addEventListener("visibilitychange", () => setRunning(!document.hidden));
  })();

  // --- headline numbers --------------------------------------------------------
  const HEADLINE_FORMAT = {
    n_items: (v) => v.toLocaleString("en-US"),
  };
  document.querySelectorAll("[data-bind]").forEach((node) => {
    const key = node.dataset.bind;
    node.textContent = (HEADLINE_FORMAT[key] || pct)(D.headline[key]);
  });

  function judgeTabs(container, current, onPick) {
    container.replaceChildren(
      ...JUDGES.map((judge) =>
        h("button", {
          type: "button", role: "tab", "aria-selected": String(judge === current),
          onclick: () => onPick(judge),
        }, h("span", { class: "dot", style: { background: judgeColor(judge) } }), judge)
      )
    );
  }

  // ============================================================================
  // 01 - Trick the judge
  // ============================================================================
  // extra: a random example being shown instead of showcase item i.
  const T = { i: 0, extra: null, judge: "Qwen2.5-7B", swap: false, pad: false, revealed: false };

  const modelOf = (item, winner) => (winner === "A" ? item.model_a : item.model_b);
  const presentation = () => ({ condition: T.pad ? "verbose" : "clean", order: T.swap ? "BA" : "AB" });

  function presentationName(condition, order) {
    if (condition === "clean" && order === "AB") return "original presentation";
    if (condition === "clean") return "answer order swapped";
    if (order === "AB") return "answers padded";
    return "order swapped and answers padded";
  }

  function replies(item, conversation) {
    const turns = conversation.filter((m) => m.role === "assistant").map((m) => m.content);
    return turns.map((text, idx) => {
      const context = item.turn === 2 && idx === 0;
      const tag = item.turn === 2 ? (context ? "turn 1 reply (context)" : "turn 2 reply (judged)") : "reply";
      return h("div", { class: `reply${context ? " context" : ""}` }, h("span", { class: "reply-tag" }, tag), text);
    });
  }

  function renderAnswer(slot, item, letter, conversation, position, call) {
    const picked = call.status === "ok" && call.pick === position;
    const humans = T.revealed && item.human.label === letter;
    slot.classList.toggle("picked", picked);
    slot.classList.toggle("human-choice", humans);
    slot.style.setProperty("--pick", judgeFill(T.judge));
    slot.style.setProperty("--on-pick", onJudgeColor(T.judge));
    const stickers = picked || humans ? h("span", { class: "stickers" },
      picked ? sticker("", "judge's pick", call.conf === null ? "no confidence" : `${pct(call.conf)} sure`) : null,
      humans ? humansSticker(item.human) : null) : null;
    slot.replaceChildren(...[
      h("div", { class: "titlebar" }, h("span", {}, `Answer shown ${position} · ${modelOf(item, letter)}`)),
      stickers,
      h("div", { class: "win-body" }, replies(item, conversation)),
    ].filter(Boolean));
  }

  function verdictSummary(item, call) {
    if (call.status === "skipped") return h("p", { class: "muted" }, "This judge never saw this pair: the prompt was over its length limit.");
    if (call.status === "tie") return h("p", { class: "muted" }, "It declared a tie.");
    if (call.status !== "ok" || !call.winner) return h("p", { class: "muted" }, "It produced no usable verdict.");
    const where = call.pick === "first" ? "shown first" : "shown second";
    return h("div", {},
      h("div", { class: "verdict-line" }, `Picks ${modelOf(item, call.winner)} (${where})`),
      call.conf === null
        ? h("p", { class: "muted" }, "auto-j states no confidence.")
        : [h("div", { class: "conf-meter", title: `stated confidence ${pct(call.conf)}` },
             h("div", { style: { width: pct(call.conf) } })),
           h("div", { class: "muted" }, `stated confidence ${pct(call.conf)}`)]
    );
  }

  const LABEL_NOTE = {
    "Qwen2.5-7B": "\"Assistant A\" below means the answer shown first.",
    "kev-8b": null,
    "auto-j-13b": "\"Response 1\" below means the answer shown first.",
  };

  function reasoningWindow(slot, presentationLabel, item, call, placeholder) {
    const body = placeholder
      ? [h("p", { class: "muted" }, placeholder)]
      : [
          verdictSummary(item, call),
          call.text
            ? [LABEL_NOTE[T.judge] ? h("p", { class: "muted" }, LABEL_NOTE[T.judge]) : null, h("div", {}, call.text)]
            : h("p", { class: "muted" }, T.judge === "kev-8b"
                ? "kev-8b returns a class probability and writes no explanation."
                : "No explanation was recorded for this call."),
        ];
    slot.style.setProperty("--judge", judgeColor(T.judge));
    slot.replaceChildren(
      h("div", { class: "note-head" }, h("span", { class: "dot", style: { background: judgeColor(T.judge) } }),
        `${T.judge}'s reasoning · ${presentationLabel}`),
      h("div", { class: "win-body" }, body));
  }

  function renderBanner(item, call, base, condition, order) {
    const banner = $("banner");
    banner.className = "banner";
    if (condition === "clean" && order === "AB") {
      banner.textContent = "This is the original presentation. Flip a switch above to try to trick the judge.";
      return;
    }
    const trick = presentationName(condition, order);
    if (!call.winner || !base.winner) {
      banner.textContent = `No comparable verdict for this presentation (${trick}).`;
      return;
    }
    if (call.winner === base.winner) {
      banner.classList.add("same");
      banner.textContent = `${T.judge} gives the same verdict (${trick}).`;
      return;
    }
    banner.classList.add("flip");
    const conf = call.conf !== null && base.conf !== null
      ? ` Stated confidence: ${pct(base.conf)} before, ${pct(call.conf)} now.`
      : " auto-j states no confidence.";
    banner.replaceChildren(h("b", {}, "Same two answers, opposite verdict"), ` (${trick}).${conf}`);
  }

  /** The reveal marks the humans' answer on the answer cards themselves. */
  function renderRevealButton() {
    const button = $("reveal");
    button.setAttribute("aria-pressed", String(T.revealed));
    button.textContent = T.revealed ? "Hide what humans chose" : "Reveal what humans chose";
  }

  function renderTrick() {
    const item = T.extra || D.showcase[T.i];
    const { condition, order } = presentation();
    const calls = item.judges[T.judge];
    const call = calls[condition][order];
    const base = calls.clean.AB;
    const convA = T.pad ? item.padded_a : item.conversation_a;
    const convB = T.pad ? item.padded_b : item.conversation_b;
    const shown = T.swap ? [["B", convB], ["A", convA]] : [["A", convA], ["B", convB]];

    $("ex-count").textContent = T.extra ? "Random example" : `Example ${T.i + 1} / ${D.showcase.length}`;
    $("ex-kind").textContent = KIND_LABELS[item.kind] || "";
    $("q-meta").textContent = categoryLabel(item);
    $("question").replaceChildren(
      ...item.conversation_a.filter((m) => m.role === "user").map((m, idx) =>
        h("div", { class: "q-turn" }, h("span", { class: "who" }, `user · turn ${idx + 1}`), m.content))
    );
    renderAnswer($("answer-first"), item, shown[0][0], shown[0][1], "first", call);
    renderAnswer($("answer-second"), item, shown[1][0], shown[1][1], "second", call);
    renderBanner(item, call, base, condition, order);

    const isOriginal = condition === "clean" && order === "AB";
    reasoningWindow($("reason-original"), "original presentation", item, base, null);
    reasoningWindow($("reason-now"), presentationName(condition, order), item, call,
      isOriginal ? "Flip a switch to see what it says when the answers are swapped or padded." : null);
    renderRevealButton();
    judgeTabs($("judge-tabs"), T.judge, (judge) => { T.judge = judge; renderTrick(); });
  }

  // From a random example, the arrows first return to the example you left.
  $("prev").addEventListener("click", () => {
    if (T.extra) T.extra = null; else T.i = (T.i - 1 + D.showcase.length) % D.showcase.length;
    renderTrick();
  });
  $("next").addEventListener("click", () => {
    if (T.extra) T.extra = null; else T.i = (T.i + 1) % D.showcase.length;
    renderTrick();
  });

  // Random examples live in a separate file that is fetched on the first click
  // only, so they add nothing to the page's own load. A script tag (not fetch)
  // keeps it working when the page is opened straight from disk.
  let examplesLoading = null;
  function loadExamples() {
    if (window.SITE_EXAMPLES) return Promise.resolve(window.SITE_EXAMPLES);
    if (!examplesLoading) {
      examplesLoading = new Promise((resolve, reject) => {
        const tag = document.createElement("script");
        tag.src = "data/examples.js";
        tag.onload = () => resolve(window.SITE_EXAMPLES);
        tag.onerror = () => { examplesLoading = null; tag.remove(); reject(new Error("examples.js failed to load")); };
        document.body.append(tag);
      });
    }
    return examplesLoading;
  }
  let unseenExamples = [];
  $("random").addEventListener("click", async () => {
    const button = $("random");
    button.disabled = true; button.textContent = "Loading…";
    try {
      const examples = await loadExamples();
      if (!unseenExamples.length) unseenExamples = shuffle(examples.map((_, i) => i));
      T.extra = examples[unseenExamples.pop()];
      renderTrick();
    } catch {
      $("ex-kind").textContent = "Couldn't load the random examples - check the connection and try again.";
    } finally {
      button.disabled = false; button.textContent = "Random";
    }
  });
  $("swap").addEventListener("change", (e) => { T.swap = e.target.checked; renderTrick(); });
  $("pad").addEventListener("change", (e) => { T.pad = e.target.checked; renderTrick(); });
  // A toggle, like the switches: once on, it stays on across examples, judges and switches.
  $("reveal").addEventListener("click", () => { T.revealed = !T.revealed; renderTrick(); });
  renderTrick();

  // ============================================================================
  // 02 - Ship it?
  // ============================================================================
  const C = { judge: "Qwen2.5-7B", signal: "conf_verb", tIdx: D.thresholds.indexOf(AUTO_ACCEPT_AT), pad: false };
  const SERIES = new Map(D.auto_accept.map((s) => [`${s.judge}|${s.signal}|${s.condition}`, s]));
  const label = (signal) => D.signal_labels[signal] || signal;
  const current = () => SERIES.get(`${C.judge}|${C.signal}|${C.pad ? "verbose" : "clean"}`);

  function signalsFor(judge) {
    const seen = [];
    for (const s of D.auto_accept) if (s.judge === judge && !seen.includes(s.signal)) seen.push(s.signal);
    return seen;
  }

  function renderSignalSelect() {
    const condition = C.pad ? "verbose" : "clean";
    const options = signalsFor(C.judge).map((signal) => {
      const available = SERIES.has(`${C.judge}|${signal}|${condition}`);
      return h("option", available ? { value: signal } : { value: signal, disabled: "" },
        available ? label(signal) : `${label(signal)} (original answers only)`);
    });
    $("signal").replaceChildren(...options);
    if (!SERIES.has(`${C.judge}|${C.signal}|${condition}`)) {
      C.signal = signalsFor(C.judge).find((s) => SERIES.has(`${C.judge}|${s}|${condition}`));
    }
    $("signal").value = C.signal;
  }

  /** Threshold indexes where this series' result changes. Each is the highest
      threshold of a run that accepts exactly the same verdicts: a signal with few
      distinct values (stated confidence has 4) gives identical results in between,
      so the slider snaps to these and marks them with ticks. */
  function stopsFor(series) {
    const a = series.accepted_share, last = a.length - 1;
    return a.map((_, i) => i).filter((i) => i === last || a[i] !== a[i + 1]);
  }
  const snap = (i) => stopsFor(current()).reduce((best, j) => (Math.abs(j - i) < Math.abs(best - i) ? j : best));

  function renderTicks() {
    const last = D.thresholds.length - 1;
    // Chrome's range thumb is 16px wide, so tick positions are inset by half of it.
    $("ticks").replaceChildren(...stopsFor(current()).map((i) =>
      h("i", { style: { left: `calc(8px + (100% - 16px) * ${i / last})` } })));
  }

  function renderReadouts() {
    const s = current();
    const i = C.tIdx;
    const t = D.thresholds[i];
    const slip = s.slip_through[i];
    $("gate-value").textContent = t.toFixed(2);
    $("threshold").value = String(i);
    $("threshold").setAttribute("aria-valuetext", `confidence at least ${t.toFixed(2)}`);
    $("flow-judge").textContent = `${C.judge} · ${label(C.signal)}`;
    $("ro-slip").textContent = pct(slip);
    $("ro-accepted").textContent = pct(s.accepted_share[i]);
    $("ro-error").textContent = pct(s.error_among_accepted[i]);
    $("ro-base").textContent = `overall error rate ${pct(s.base_error)}`;
    $("ro-kappa").textContent = s.kappa_among_accepted[i] === null ? "—" : s.kappa_among_accepted[i].toFixed(2);
    $("ro-reading").textContent = slip >= 0.9
      ? `At ≥ ${t.toFixed(2)} the gate filters out almost none of the errors.`
      : `At ≥ ${t.toFixed(2)} it catches ${pct(1 - slip)} of the errors by sending ${pct(1 - s.accepted_share[i])} of all verdicts to a human.`;
    $("sink-accept").textContent = `${(s.n - s.n_escalated[i]).toLocaleString("en-US")} of ${s.n.toLocaleString("en-US")} verdicts`;
    $("sink-human").textContent = `${s.n_escalated[i].toLocaleString("en-US")} of ${s.n.toLocaleString("en-US")} verdicts`;
  }

  /** A new series (judge, signal or padding) restarts the flow and draws the chart's
      lines in again; moving the threshold only re-sorts the tallies and moves the marker. */
  function updateCost({ newSeries }) {
    judgeTabs($("cost-judge-tabs"), C.judge, (judge) => {
      C.judge = judge; renderSignalSelect(); updateCost({ newSeries: true });
    });
    if (newSeries) { C.tIdx = snap(C.tIdx); renderTicks(); }
    renderReadouts();
    drawChart(newSeries);
    if (newSeries) flow.reset(current()); else flow.refresh();
  }

  $("signal").addEventListener("change", (e) => { C.signal = e.target.value; updateCost({ newSeries: true }); });
  $("cost-pad").addEventListener("change", (e) => { C.pad = e.target.checked; renderSignalSelect(); updateCost({ newSeries: true }); });
  // Pointer drags snap to the nearest stop.
  $("threshold").addEventListener("input", (e) => { C.tIdx = snap(Number(e.target.value)); updateCost({ newSeries: false }); });
  // Keys move stop to stop. The browser's own one-step move would land nearer the
  // current stop than the next one and snap straight back, so it is replaced here.
  $("threshold").addEventListener("keydown", (e) => {
    const stops = stopsFor(current());
    const moves = {
      ArrowRight: () => stops.find((j) => j > C.tIdx), ArrowUp: () => stops.find((j) => j > C.tIdx),
      PageUp: () => stops.find((j) => j > C.tIdx),
      ArrowLeft: () => stops.filter((j) => j < C.tIdx).pop(), ArrowDown: () => stops.filter((j) => j < C.tIdx).pop(),
      PageDown: () => stops.filter((j) => j < C.tIdx).pop(),
      Home: () => stops[0], End: () => stops[stops.length - 1],
    };
    if (!moves[e.key]) return;
    e.preventDefault();
    C.tIdx = moves[e.key]() ?? C.tIdx;
    updateCost({ newSeries: false });
  });

  // --- chart: frameless SVG on the paper; lines draw in on each new series and each
  // time the chart scrolls into view -------------------------------------------------
  const svgNS = "http://www.w3.org/2000/svg";
  function s(tag, attrs = {}) {
    const node = document.createElementNS(svgNS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    return node;
  }

  function drawChart(animate) {
    const svg = $("curve");
    const series = current();
    const W = svg.clientWidth || 800;
    const H = svg.clientHeight || 340;
    // Wide: labels at the line ends. Narrow: they would squeeze the plot, so a key
    // goes above it instead.
    const narrow = W < 560;
    const m = narrow ? { l: 40, r: 12, t: 50, b: 34 } : { l: 46, r: 170, t: 14, b: 34 };
    const x = (t) => m.l + ((t - 0.5) / 0.5) * (W - m.l - m.r);
    const y = (v) => m.t + (1 - v) * (H - m.t - m.b);
    const color = judgeColor(C.judge);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.replaceChildren();

    const grid = s("g", { class: "grid" });
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      grid.append(s("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
      const t = s("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }); t.textContent = pct(v); grid.append(t);
    }
    for (const v of [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) {
      const t = s("text", { x: x(v), y: H - 12, "text-anchor": "middle" }); t.textContent = v.toFixed(1); grid.append(t);
    }
    const xl = s("text", { x: (m.l + W - m.r) / 2, y: H, "text-anchor": "middle" });
    xl.textContent = "auto-accept when confidence ≥"; grid.append(xl);
    svg.append(grid);

    const pathFor = (values) => values.map((v, i) => `${i ? "L" : "M"}${x(D.thresholds[i]).toFixed(1)},${y(v ?? 0).toFixed(1)}`).join("");
    const clip = s("clipPath", { id: "reveal-clip" });
    const clipRect = s("rect", { x: 0, y: 0, width: animate ? 0 : W, height: H });
    clip.append(clipRect);
    svg.append(clip);
    const lines = s("g", { "clip-path": "url(#reveal-clip)" });
    lines.append(s("path", { class: "line accepted", d: pathFor(series.accepted_share), stroke: color }));
    lines.append(s("path", { class: "line", d: pathFor(series.slip_through), stroke: color }));
    svg.append(lines);

    // Direct labels at the right end - identity is never colour alone.
    const last = D.thresholds.length - 1;
    // Keep the two labels at least 16px apart and above the x-axis labels.
    const ya = series.accepted_share[last] ?? 0, ys = series.slip_through[last] ?? 0;
    let py = { slip: y(ys), acc: y(ya) };
    if (Math.abs(py.slip - py.acc) < 16) {
      const mid = (py.slip + py.acc) / 2, upper = ys >= ya ? "slip" : "acc";
      py = upper === "slip" ? { slip: mid - 8, acc: mid + 8 } : { acc: mid - 8, slip: mid + 8 };
    }
    const floor = H - m.b - 8, shift = Math.max(0, Math.max(py.slip, py.acc) - floor);
    const endLabel = (py_, text) => {
      const t = s("text", { class: "line-label", x: x(D.thresholds[last]) + 10, y: py_ - shift + 4 });
      t.textContent = text; svg.append(t);
    };
    if (narrow) {
      [["wrong verdicts shipped", "line"], ["all verdicts accepted", "line accepted"]].forEach(([text, cls], k) => {
        const ky = 10 + k * 18;
        svg.append(s("line", { class: cls, x1: 0, x2: 24, y1: ky, y2: ky, stroke: color }));
        const t = s("text", { class: "line-label", x: 32, y: ky + 4 }); t.textContent = text; svg.append(t);
      });
    } else {
      endLabel(py.slip, "wrong verdicts shipped");
      endLabel(py.acc, "all verdicts accepted (dashed)");
    }

    // The user's threshold.
    const i = C.tIdx, tx = x(D.thresholds[i]);
    svg.append(s("line", { class: "threshold-rule", x1: tx, x2: tx, y1: m.t, y2: H - m.b }));
    for (const [v, r] of [[series.slip_through[i], 6], [series.accepted_share[i], 5]]) {
      svg.append(s("circle", { class: "marker", cx: tx, cy: y(v ?? 0), r, fill: color }));
    }

    // Hover layer: crosshair + tooltip; click moves the threshold there.
    const cross = s("line", { class: "crosshair", y1: m.t, y2: H - m.b, visibility: "hidden" });
    svg.append(cross);
    const hit = s("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
    const tip = $("tooltip");
    const indexAt = (evt) => {
      const px = (evt.offsetX / svg.clientWidth) * W;
      const t = 0.5 + ((px - m.l) / (W - m.l - m.r)) * 0.5;
      return snap(Math.max(0, Math.min(last, Math.round((t - 0.5) * 100))));
    };
    hit.addEventListener("mousemove", (evt) => {
      const j = indexAt(evt), cx = x(D.thresholds[j]);
      cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
      tip.hidden = false;
      tip.textContent = `≥ ${D.thresholds[j].toFixed(2)} · ${pct(series.slip_through[j])} of wrong shipped · ${pct(series.accepted_share[j])} accepted`;
      tip.style.left = `${Math.max(0, Math.min(evt.offsetX + 14, svg.clientWidth - tip.offsetWidth))}px`;
      tip.style.top = `${evt.offsetY - 34}px`;
    });
    hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); tip.hidden = true; });
    hit.addEventListener("click", (evt) => { C.tIdx = indexAt(evt); updateCost({ newSeries: false }); });
    svg.append(hit);

    if (animate && !reducedMotion) {
      const start = performance.now();
      const step = (now) => {
        // rAF's timestamp can precede `start` by a frame, so clamp at 0 too.
        const k = Math.min(1, Math.max(0, (now - start) / 1200));
        clipRect.setAttribute("width", String(W * (1 - Math.pow(1 - k, 3))));
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    } else {
      clipRect.setAttribute("width", String(W));
    }
  }

  // --- pipeline: dots flow judge -> gate -> accepted / human ---------------------
  const flow = (() => {
    const canvas = $("flow");
    const box = canvas.parentElement;
    const ctx = canvas.getContext("2d");
    const CELL = 9, ROWS = 6, HISTORY = 600;
    // Every verdict that reached a sink, oldest first. The two tallies are the last
    // cols x ROWS of these on each side of the *current* threshold, so moving the gate
    // re-sorts the same real verdicts at once instead of mixing two thresholds.
    let dots = [], queue = [], history = [], series = null;
    let geom = null, lastSpawn = 0, running = false, visible = false;

    // Deterministic shuffle (mulberry32), so the stream is the same on every visit.
    function shuffled(items) {
      let a = 1234567;
      const rand = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
      const out = items.slice();
      for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; }
      return out;
    }

    function measure() {
      const dpr = window.devicePixelRatio || 1;
      const r = box.getBoundingClientRect();
      canvas.width = r.width * dpr; canvas.height = r.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const rel = (sel) => { const q = box.querySelector(sel).getBoundingClientRect(); return { l: q.left - r.left, r: q.right - r.left, t: q.top - r.top, b: q.bottom - r.top }; };
      const src = rel(".node.source"), gate = rel(".node.gate"), acc = rel(".node.accept"), hum = rel(".node.human");
      const cols = Math.floor((acc.r - acc.l + 2) / CELL);   // the grid spans its node's width
      const midX = (q) => (q.l + q.r) / 2, midY = (q) => (q.t + q.b) / 2;
      // On a phone the CSS stacks the nodes: judge on top, gate below, the two sinks
      // side by side at the bottom. The flow then runs downward.
      const vertical = gate.t >= src.b - 1;
      if (vertical) {
        const gateOut = { x: midX(gate), y: gate.b };
        const sink = (q) => ({ x: midX(q), y: q.t, ctrl: { x: midX(q), y: gateOut.y }, tallyX: q.l, tallyY: q.b + 10, grow: 1 });
        geom = {
          w: r.width, h: r.height, cols, vertical,
          start: { x: midX(src), y: src.b }, gateIn: { x: midX(gate), y: gate.t }, gateOut,
          accept: sink(acc), human: sink(hum),
        };
      } else {
        const gateOut = { x: gate.r, y: midY(gate) };
        const sink = (q, tallyY, grow) => ({ x: q.l, y: midY(q), ctrl: { x: (gateOut.x + q.l) / 2, y: gateOut.y }, tallyX: q.l, tallyY, grow });
        geom = {
          w: r.width, h: r.height, cols, vertical,
          start: { x: src.r, y: midY(src) }, gateIn: { x: gate.l, y: midY(gate) }, gateOut,
          // Accepted fills downward from under its node; escalated fills upward from
          // just above its node, so each pile sits against the box it belongs to.
          accept: sink(acc, acc.b + 10, 1), human: sink(hum, hum.t - 10 - 7, -1),
        };
      }
    }

    function reset(newSeries) {
      series = newSeries;
      queue = shuffled(series.items);
      dots = []; history = [];
      if (reducedMotion) {
        // No motion: the first verdicts arrive at once and the tallies stand still.
        for (const [conf, ok] of queue.slice(0, HISTORY)) history.push({ conf, ok });
        draw();
      }
    }

    const route = (dot) => (dot.conf >= D.thresholds[C.tIdx] ? "accept" : "human");

    function arrive(dot) {
      history.push({ conf: dot.conf, ok: dot.ok });
      if (history.length > HISTORY) history.shift();
    }

    function currentTallies() {
      const t = D.thresholds[C.tIdx], out = { accept: [], human: [] }, cap = geom.cols * ROWS;
      for (let i = history.length - 1; i >= 0; i--) {
        const pile = out[history[i].conf >= t ? "accept" : "human"];
        if (pile.length < cap) pile.push(history[i].ok);
      }
      out.accept.reverse(); out.human.reverse();
      return out;
    }

    function spawn(now) {
      if (!queue.length) queue = shuffled(series.items);
      const [conf, ok] = queue.pop();
      dots.push({ conf, ok, born: now, jitter: (Math.random() - 0.5) * 36, dest: null });
    }

    const lerp = (a, b, k) => a + (b - a) * k;
    const SEG = 1500;

    function draw(now = performance.now()) {
      if (!geom) measure();
      const g = geom, ink = cssVar("--dot-ok"), red = cssVar("--accent");
      ctx.clearRect(0, 0, g.w, g.h);

      // Faint rails.
      ctx.strokeStyle = cssVar("--rule"); ctx.lineWidth = 2; ctx.setLineDash([3, 6]);
      ctx.beginPath(); ctx.moveTo(g.start.x, g.start.y); ctx.lineTo(g.gateIn.x, g.gateIn.y); ctx.stroke();
      for (const sink of [g.accept, g.human]) {
        ctx.beginPath(); ctx.moveTo(g.gateOut.x, g.gateOut.y);
        ctx.quadraticCurveTo(sink.ctrl.x, sink.ctrl.y, sink.x, sink.y); ctx.stroke();
      }
      ctx.setLineDash([]);

      // Moving dots.
      dots = dots.filter((d) => {
        const age = now - d.born;
        let px, py;
        if (age < SEG) {
          const k = age / SEG;
          if (g.vertical) { px = lerp(g.start.x + d.jitter, g.gateIn.x, k * k); py = lerp(g.start.y, g.gateIn.y, k); }
          else { px = lerp(g.start.x, g.gateIn.x, k); py = lerp(g.start.y + d.jitter, g.gateIn.y, k * k); }
        } else {
          if (!d.dest) d.dest = route(d);
          const k = Math.min(1, (age - SEG) / SEG), sink = g[d.dest];
          px = (1 - k) * (1 - k) * g.gateOut.x + 2 * (1 - k) * k * sink.ctrl.x + k * k * sink.x;
          py = (1 - k) * (1 - k) * g.gateOut.y + 2 * (1 - k) * k * sink.ctrl.y + k * k * sink.y;
          if (k >= 1) { arrive(d); return false; }
        }
        ctx.fillStyle = d.ok ? ink : red;
        ctx.beginPath(); ctx.arc(px, py, d.ok ? 3.5 : 4.5, 0, Math.PI * 2); ctx.fill();
        return true;
      });

      // Tallies: the latest verdicts on each side of the gate, wrong ones in red.
      const tallies = currentTallies();
      for (const key of ["accept", "human"]) {
        const sink = g[key];
        tallies[key].forEach((ok, idx) => {
          ctx.fillStyle = ok ? ink : red;
          ctx.fillRect(sink.tallyX + (idx % g.cols) * CELL, sink.tallyY + sink.grow * Math.floor(idx / g.cols) * CELL, CELL - 2, CELL - 2);
        });
      }
    }

    function frame(now) {
      if (!running) return;
      if (now - lastSpawn > 90 && dots.length < 60) { spawn(now); lastSpawn = now; }
      draw(now);
      requestAnimationFrame(frame);
    }

    function setRunning(on) {
      if (reducedMotion) { draw(); return; }
      if (on && !running) { running = true; requestAnimationFrame(frame); }
      if (!on) running = false;
    }

    new IntersectionObserver((entries) => {
      visible = entries[0].isIntersecting;
      setRunning(visible && !document.hidden);
    }).observe(box);
    document.addEventListener("visibilitychange", () => setRunning(visible && !document.hidden));
    window.addEventListener("resize", () => { measure(); if (!running) draw(); });

    return { reset, refresh: () => { if (!running) draw(); } };
  })();

  // Draw the lines in every time the chart scrolls into view, from above or below.
  new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting) drawChart(true);
  }, { threshold: 0.4 }).observe($("curve"));
  let resizeTimer;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => drawChart(false), 150); });

  // ============================================================================
  // 02 - You vs the judge
  // ============================================================================
  // The pool (analysis/site_data.py::pick_game_items) is a seeded random draw that
  // never looked at the judges' verdicts. Which five you get is random per visit:
  // that only decides what is shown, it is not a statistic.
  const ROUND = 5;
  const CONF_CHOICES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const G = { deck: [], round: [], step: 0, pick: null, answers: [] };
  const letterName = (letter) => (letter === "A" ? "Answer 1" : "Answer 2");

  function shuffle(values) {
    const out = values.slice();
    for (let i = out.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [out[i], out[j]] = [out[j], out[i]]; }
    return out;
  }

  function newRound() {
    if (G.deck.length < ROUND) G.deck = shuffle(D.game.map((_, i) => i));
    G.round = G.deck.splice(0, ROUND);
    G.step = 0; G.pick = null; G.answers = [];
    renderGame();
  }

  const gameItem = () => D.game[G.round[G.step]];

  /** Bring the game back into view after its content changes size.
      A new result is one thing to look at: it is centred on screen if it fits, and
      left alone if it is already fully visible. A new question starts a read from
      the top, so it is aligned just under the sticky top bar. */
  function scrollToGame(id = null) {
    const target = id ? $(id) : document.querySelector("#game .game-status");
    const bar = document.querySelector(".topbar").offsetHeight;
    const r = target.getBoundingClientRect(), room = window.innerHeight - bar;
    let top;
    if (id && r.height <= room - 32) {
      if (r.top >= bar && r.bottom <= window.innerHeight) return;
      top = r.top + window.scrollY - bar - (room - r.height) / 2;
    } else {
      top = r.top + window.scrollY - bar - 16;
    }
    window.scrollTo({ top, behavior: reducedMotion ? "auto" : "smooth" });
  }
  const locked = () => G.answers.length > G.step;

  function gameAnswer(letter) {
    const item = gameItem(), done = locked();
    const slot = $(`g-answer-${letter}`);
    const chosen = (done ? G.answers[G.step].pick : G.pick) === letter;
    slot.classList.toggle("chosen", chosen);
    slot.classList.toggle("human-choice", done && item.human.label === letter);
    const conversation = letter === "A" ? item.conversation_a : item.conversation_b;
    const model = letter === "A" ? item.model_a : item.model_b;
    const mine = done && G.answers[G.step].pick === letter, humans = done && item.human.label === letter;
    slot.replaceChildren(...[
      h("div", { class: "titlebar" }, h("span", {}, done ? `${letterName(letter)} · ${model}` : letterName(letter))),
      mine || humans ? h("span", { class: "stickers" },
        mine ? sticker("", "your pick", `${pct(G.answers[G.step].conf)} sure`) : null,
        humans ? humansSticker(item.human) : null) : null,
      h("div", { class: "win-body" }, replies(item, conversation)),
      h("button", {
        type: "button", class: "btn pick-btn", "aria-pressed": String(chosen), ...(done ? { disabled: "" } : {}),
        onclick: () => { G.pick = letter; renderGame(); },
      }, chosen ? `✓ ${letterName(letter)} is better` : `${letterName(letter)} is better`)
    ].filter(Boolean));
  }

  /** One row of the per-comparison and final tables. */
  function judgeRowCells(call, human) {
    if (!call.winner) return [call.status === "tie" ? "tie" : "no verdict", "—", ""];
    const right = call.winner === human;
    return [letterName(call.winner), call.conf === null ? "states none" : pct(call.conf),
      h("span", { class: right ? "ok-mark" : "bad-mark" }, right ? "✓ right" : "✗ wrong")];
  }

  function nameCell(label, judge) {
    return h("td", {}, judge ? h("span", { class: "dot", style: { background: judgeColor(judge) } }) : null, label);
  }

  function renderGameResult() {
    const box = $("g-result");
    box.hidden = !locked();
    if (!locked()) { box.replaceChildren(); return; }
    const item = gameItem(), mine = G.answers[G.step];
    const human = item.human.label, youRight = mine.pick === human;
    const last = G.step === ROUND - 1;
    box.classList.toggle("wrong", !youRight);
    box.replaceChildren(
      h("div", { class: "titlebar" }, `comparison_${G.step + 1}.log`),
      h("div", { class: `verdict-stamp${youRight ? "" : " wrong"}`, "aria-hidden": "true" }, youRight ? "✓ RIGHT" : "✗ WRONG"),
      h("div", { class: "win-body" },
        h("p", { class: "result-lead" }, `Humans chose ${letterName(human)}. `,
          h("span", { class: youRight ? "ok-mark" : "bad-mark" }, youRight ? "So did you." : "You didn't.")),
        h("p", {}, `${item.human.n_votes} human votes, all agreeing. The judges saw it as published.`),
        h("table", { class: "score-table" },
          h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", {}, "picked"), h("th", {}, "stated confidence"), h("th", {}, ""))),
          h("tbody", {},
            h("tr", { class: "you" }, nameCell("You"), h("td", {}, letterName(mine.pick)), h("td", {}, pct(mine.conf)),
              h("td", {}, h("span", { class: youRight ? "ok-mark" : "bad-mark" }, youRight ? "✓ right" : "✗ wrong"))),
            JUDGES.map((judge) => h("tr", {}, nameCell(judge, judge),
              judgeRowCells(item.judges[judge].clean.AB, human).map((cell) => h("td", {}, cell)))))),
        h("button", { type: "button", class: "btn primary", onclick: () => { G.step += 1; G.pick = null; renderGame(); scrollToGame(); } },
          last ? "See the scoreboard →" : "Next comparison →"))
    );
  }

  /** Right / answered, mean stated confidence, and confidence minus accuracy. */
  function tally(rows) {
    const answered = rows.filter((r) => r.pick);
    const right = answered.filter((r) => r.pick === r.human).length;
    const confs = answered.map((r) => r.conf).filter((c) => c !== null);
    const meanConf = confs.length === answered.length && confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : null;
    const accuracy = answered.length ? right / answered.length : null;
    return { right, answered: answered.length, meanConf, gap: meanConf === null ? null : meanConf - accuracy };
  }

  const signedPoints = (gap) => (gap === null ? "—" : `${gap >= 0 ? "+" : "−"}${Math.round(Math.abs(gap) * 100)} pts`);

  function renderFinal() {
    const box = $("g-final");
    const items = G.round.map((i) => D.game[i]);
    const you = tally(G.answers.map((a, k) => ({ pick: a.pick, conf: a.conf, human: items[k].human.label })));
    const judges = JUDGES.map((judge) => [judge, tally(items.map((item) => {
      const call = item.judges[judge].clean.AB;
      return { pick: call.winner, conf: call.conf, human: item.human.label };
    }))]);
    const reading = you.gap > 0.1
      ? "You were more sure than right, the same failure this project measures in the judges."
      : you.gap < -0.1 ? "You were less sure than right: underconfident." : "Your confidence roughly matched how often you were right.";
    const row = (label, judge, t, cls) => h("tr", cls ? { class: cls } : {}, nameCell(label, judge),
      h("td", {}, `${t.right} / ${t.answered}${t.answered < ROUND ? ` (${ROUND - t.answered} no verdict)` : ""}`),
      h("td", {}, t.meanConf === null ? (judge === "auto-j-13b" ? "states none" : "—") : pct(t.meanConf)),
      h("td", {}, signedPoints(t.gap)));
    box.replaceChildren(
      h("div", { class: "titlebar" }, "scoreboard.log"),
      h("div", { class: "win-body" },
        h("p", { class: "result-lead" }, `You got ${you.right} of ${ROUND} right at an average ${pct(you.meanConf)} confidence.`),
        h("p", {}, reading),
        h("table", { class: "score-table" },
          h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", {}, "right"), h("th", {}, "average stated confidence"),
            h("th", {}, "confidence minus accuracy"))),
          h("tbody", {}, row("You", null, you, "you"), judges.map(([judge, t]) => row(judge, judge, t)))),
        h("p", { class: "note" }, `Five comparisons is far too few to measure anyone, so read this as a feel for the problem.
          These are also easy ones, where every human agreed: across all ${D.headline.n_items.toLocaleString("en-US")}
          comparisons Qwen2.5-7B matched the humans ${pct(D.headline.accuracy)} of the time while stating
          ${pct(D.headline.stated_confidence)} confidence.`),
        h("button", { type: "button", class: "btn primary", onclick: () => { newRound(); scrollToGame(); } }, "Play five more →"))
    );
  }

  function renderGame() {
    const finished = G.step >= ROUND;
    $("g-play").hidden = finished;
    $("g-final").hidden = !finished;
    $("g-progress").textContent = finished ? "Round complete" : `Comparison ${G.step + 1} of ${ROUND}`;
    $("g-pips").replaceChildren(...G.round.map((idx, k) => {
      const a = G.answers[k];
      if (a) {
        const right = a.pick === D.game[idx].human.label;
        return h("li", { class: right ? "right" : "wrong", "aria-label": right ? "right" : "wrong" }, right ? "✓" : "✗");
      }
      return h("li", k === G.step ? { class: "current", "aria-label": "current" } : { "aria-label": "to come" });
    }));
    if (finished) { renderFinal(); return; }

    const item = gameItem();
    $("g-meta").textContent = `${categoryLabel(item)} · which answer is better?`;
    $("g-question").replaceChildren(...item.conversation_a.filter((m) => m.role === "user").map((m) =>
      h("div", { class: "q-turn" }, h("span", { class: "who" }, "user"), m.content)));
    gameAnswer("A");
    gameAnswer("B");
    const asking = G.pick !== null && !locked();
    $("g-confidence").hidden = !asking;
    $("g-chips").replaceChildren(...CONF_CHOICES.map((c) => h("button", {
      type: "button", onclick: () => { G.answers.push({ pick: G.pick, conf: c }); renderGame(); scrollToGame("g-result"); },
    }, pct(c))));
    renderGameResult();
  }
  newRound();

  // ============================================================================
  // 03 - Three judges
  // ============================================================================
  const PRESENTATIONS = [
    ["clean", "AB", "As published"], ["clean", "BA", "Order swapped"],
    ["verbose", "AB", "Padded"], ["verbose", "BA", "Swapped + padded"],
  ];
  // The showcase first (chosen to show something), then the game's random pool.
  const SHOWDOWN = [...D.showcase];
  for (const item of D.game) if (!SHOWDOWN.some((s) => s.item_id === item.item_id)) SHOWDOWN.push(item);
  const S = { i: 0 };

  function verdictCell(item, call, base, name) {
    const statusText = { skipped: "too long for it", tie: "tie", no_verdict: "no verdict" };
    if (!call.winner) {
      return h("div", { class: "vg-cell empty", role: "cell" }, h("span", { class: "pres" }, name), statusText[call.status] || "no verdict");
    }
    const flipped = base.winner && call.winner !== base.winner;
    const right = call.winner === item.human.label;
    return h("div", { class: `vg-cell${flipped ? " flipped" : ""}`, role: "cell" },
      h("span", { class: "pres" }, name),
      h("span", { class: `vg-mark ${right ? "ok-mark" : "bad-mark"}`, title: right ? "matches the humans" : "differs from the humans" }, right ? "✓" : "✗"),
      h("span", { class: "who" }, modelOf(item, call.winner)),
      h("span", { class: "sure" }, call.conf === null ? "no confidence stated" : `${pct(call.conf)} sure`),
      flipped ? h("span", { class: "bad-mark" }, "flipped") : null);
  }

  function renderShowdown() {
    const item = SHOWDOWN[S.i];
    $("s-count").textContent = `${S.i + 1} / ${SHOWDOWN.length}`;
    const kind = KIND_LABELS[item.kind] || "random, humans agreed";
    $("s-meta").textContent = `${categoryLabel(item)} · ${item.model_a} vs ${item.model_b} · ${kind}`;
    $("s-question").replaceChildren(...item.conversation_a.filter((m) => m.role === "user").map((m, idx) =>
      h("div", { class: "q-turn" }, h("span", { class: "who" }, `user · turn ${idx + 1}`), m.content)));
    $("s-human").replaceChildren(`Humans chose ${modelOf(item, item.human.label)} `,
      h("span", { class: "muted" }, `(${item.human.n_votes} votes, all agreeing)`));
    $("s-answers").replaceChildren(...["A", "B"].map((letter) => h("article", { class: "window answer" },
      h("div", { class: "titlebar" }, letter === "A" ? item.model_a : item.model_b),
      h("div", { class: "win-body" }, replies(item, letter === "A" ? item.conversation_a : item.conversation_b)))));

    const cells = [h("div", { class: "vg-head", role: "columnheader" }, "judge"),
      ...PRESENTATIONS.map(([, , name]) => h("div", { class: "vg-head", role: "columnheader" }, name)),
      h("div", { class: "vg-head", role: "columnheader" }, "")];
    for (const judge of JUDGES) {
      const calls = item.judges[judge], base = calls.clean.AB;
      const winners = PRESENTATIONS.map(([c, o]) => calls[c][o].winner).filter(Boolean);
      const flips = winners.filter((w) => base.winner && w !== base.winner).length;
      const summary = !winners.length ? "no verdicts"
        : flips === 0 ? [h("b", {}, `same pick ${winners.length} / ${winners.length}`)]
        : [h("b", {}, `flipped ${flips} of ${winners.length - 1} times`)];
      cells.push(
        h("div", { class: "vg-judge", role: "rowheader" }, h("span", { class: "dot", style: { background: judgeColor(judge) } }), judge),
        ...PRESENTATIONS.map(([c, o, name]) => verdictCell(item, calls[c][o], base, name)),
        h("div", { class: `vg-summary${flips ? " unstable" : ""}`, role: "cell" }, summary));
    }
    $("s-grid").replaceChildren(...cells);
  }
  $("s-prev").addEventListener("click", () => { S.i = (S.i - 1 + SHOWDOWN.length) % SHOWDOWN.length; renderShowdown(); });
  $("s-next").addEventListener("click", () => { S.i = (S.i + 1) % SHOWDOWN.length; renderShowdown(); });
  renderShowdown();

  const M = D.method;
  const autojNoVerdict = `${(M.autoj_padded_turn2_no_verdict * 100).toFixed(1)}%`;

  // --- the four headline measures, each judge with its 95% interval -------------------
  const MEASURE_TEXT = {
    auroc: ["Does its uncertainty flag its own mistakes?", "AUROC: 0.5 (dashed) is chance, higher is better."],
    overconfidence_gap: ["How much more sure than right?", "Mean confidence minus accuracy: 0 (dashed) is honest, above it overconfident."],
    flip_rate: ["Does it flip when the answers swap places?", "Share of verdicts that change with the order alone."],
    delta_ece: ["Does padding make its confidence less honest?", "Calibration error, padded minus original: above 0 (dashed) is worse."],
  };
  const MEASURE_FORMAT = {
    auroc: (v) => v.toFixed(2),
    overconfidence_gap: (v) => (Math.abs(v) < 1e-9 ? "0" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(0)} pts`),
    flip_rate: (v) => pct(v),
    delta_ece: (v) => (Math.abs(v) < 1e-9 ? "0" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`),
  };

  function drawMeasure(container, measure) {
    const rows = D.judge_comparison.filter((r) => r.measure === measure.measure);
    const fmt = MEASURE_FORMAT[measure.measure];
    const [title, sub] = MEASURE_TEXT[measure.measure];
    const svg = s("svg", { role: "img", "aria-label": title });
    container.replaceChildren(h("h4", {}, title), h("p", {}, sub), svg);

    // Wide: label column on the left. Narrow: each label sits above its row, so the
    // plot keeps the full width and the axis numbers don't run together.
    const W = container.clientWidth || 520, narrow = W < 480;
    const rowH = narrow ? 44 : 30, m = narrow ? { l: 0, r: 64, t: 4, b: 26 } : { l: 190, r: 70, t: 6, b: 26 };
    const H = m.t + rows.length * rowH + m.b;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("height", H);
    const lows = rows.map((r) => r.ci_low), highs = rows.map((r) => r.ci_high);
    let lo = Math.min(...lows), hi = Math.max(...highs);
    if (measure.reference !== null) { lo = Math.min(lo, measure.reference); hi = Math.max(hi, measure.reference); }
    const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    const x = (v) => m.l + ((v - lo) / (hi - lo)) * (W - m.l - m.r);
    const yRow = (i) => m.t + i * rowH + (narrow ? 30 : rowH / 2);

    svg.append(s("line", { class: "axis-line", x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    // At least ~64px between ticks, so the labels never collide.
    const step = niceStep(Math.max((hi - lo) / 4, (hi - lo) * 64 / (W - m.l - m.r)));
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
      const t = s("text", { class: "tick", x: x(v), y: H - 8, "text-anchor": "middle" });
      t.textContent = fmt(v); svg.append(t);
    }
    if (measure.reference !== null) {
      svg.append(s("line", { class: "ref", x1: x(measure.reference), x2: x(measure.reference), y1: m.t, y2: H - m.b }));
    }
    const tip = $("s-tooltip"), section = $("judges");
    rows.forEach((r, i) => {
      const y = yRow(i), color = judgeColor(r.judge);
      const label = s("text", { class: "row-label", x: 0, y: narrow ? y - 14 : y + 4 });
      label.textContent = r.population === "all" ? r.judge : `${r.judge} · ${r.population}`;
      svg.append(label);
      svg.append(s("line", { class: "ci", x1: x(r.ci_low), x2: x(r.ci_high), y1: y, y2: y, stroke: color }));
      svg.append(s("circle", { class: "est", cx: x(r.value), cy: y, r: 6, fill: color }));
      const value = s("text", { class: "value-label", x: W - m.r + 12, y: y + 4 });
      value.textContent = fmt(r.value); svg.append(value);
      const hit = s("rect", { x: 0, y: y - rowH / 2, width: W, height: rowH, fill: "transparent" });
      hit.addEventListener("mousemove", (evt) => {
        const box = section.getBoundingClientRect();
        tip.hidden = false;
        tip.textContent = `${label.textContent}: ${fmt(r.value)} (95% interval ${fmt(r.ci_low)} to ${fmt(r.ci_high)})`;
        tip.style.left = `${Math.max(0, Math.min(evt.clientX - box.left + 14, box.width - tip.offsetWidth))}px`;
        tip.style.top = `${evt.clientY - box.top - 36}px`;
      });
      hit.addEventListener("mouseleave", () => { tip.hidden = true; });
      svg.append(hit);
    });
  }

  function niceStep(raw) {
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    return [1, 2, 2.5, 5, 10].map((k) => k * mag).find((k) => k >= raw);
  }

  function drawMeasures() {
    const host = $("s-measures");
    if (!host.children.length) {
      host.replaceChildren(...D.comparison_measures.map(() => h("div", { class: "measure" })));
    }
    D.comparison_measures.forEach((measure, i) => drawMeasure(host.children[i], measure));
  }
  drawMeasures();
  $("s-autoj-note").textContent =
    `For every auto-j-13b number in this section: it often gives no verdict at all. On padded turn-2 comparisons ` +
    `its critique runs out of room before it decides ${autojNoVerdict} of the time, and its numbers cover only the verdicts it gave.`;
  // Intervals grow in each time the panels scroll into view.
  new IntersectionObserver((entries) => {
    $("s-measures").classList.toggle("shown", reducedMotion || entries[0].isIntersecting);
  }, { threshold: 0.3 }).observe($("s-measures"));

  // ============================================================================
  // 05 - How it was tested: an n8n-style graph of the pipeline
  // ============================================================================
  const num = (v) => v.toLocaleString("en-US");
  const totalCalls = M.n_items * 4;
  const METHOD_NODES = [
    { id: "data", col: 0, bar: "01 · data", title: "MT-Bench", sub: `${M.n_questions} questions · ${M.n_models} chatbots`,
      body: [`${M.n_questions} multi-turn questions in eight categories (writing, roleplay, reasoning, math, coding, extraction, STEM, humanities), each answered by ${M.n_models} chatbots, from GPT-4 to LLaMA-13B. People compared pairs of answers and voted for the better one.`,
        "Source: lmsys/mt_bench_human_judgments, CC BY 4.0."] },
    { id: "labels", col: 1, bar: "02 · labels", title: "Human verdicts", sub: `${num(M.n_items)} labelled comparisons`,
      body: ["Votes are grouped by comparison: the same question, the same two chatbots, the same turn. The label is the majority among voters who picked a side.",
        `Comparisons where most voters called it a tie, or where the votes split evenly, get no label and are left out. That leaves ${num(M.n_items)} comparisons: the yardstick every judge verdict is scored against.`] },
    { id: "present", col: 2, bar: "03 · perturb", title: "Four presentations", sub: "as published · swapped · padded · both",
      body: ["Each comparison is shown to each judge four ways: as published; with the two answers in swapped order; with both answers padded with filler that adds length but no information; and both at once.",
        "The content of the answers never changes, so a judge that is reading them should give the same verdict all four times."] },
    { id: "qwen", col: 3, judge: "Qwen2.5-7B", bar: "04 · judge", title: "Qwen2.5-7B", sub: `${num(M.verdicts["Qwen2.5-7B"])} verdicts`,
      body: ["A general-purpose 7B chat model, prompted to act as a judge. It writes its reasoning, a verdict and a stated confidence.",
        "It was also re-asked four more times at a higher sampling temperature, and with two reworded prompts, to measure how stable its verdicts are.",
        `Recorded: ${num(M.verdicts["Qwen2.5-7B"])} main verdicts across the four presentations.`] },
    { id: "kev", col: 3, judge: "kev-8b", bar: "04 · judge", title: "kev-8b", sub: `${num(M.verdicts["kev-8b"])} verdicts`,
      body: ["An 8B model trained to give calibrated probabilities, used as an open stand-in for commercial judges that claim to know when they are wrong. It returns a probability for each answer and writes no explanation.",
        `Recorded: ${num(M.verdicts["kev-8b"])} verdicts; the other ${num(totalCalls - M.verdicts["kev-8b"])} prompts were over its length limit.`] },
    { id: "autoj", col: 3, judge: "auto-j-13b", bar: "04 · judge", title: "auto-j-13b", sub: `${num(M.verdicts["auto-j-13b"])} verdicts`,
      body: ["A 13B model trained specifically to judge pairs of answers, on critiques distilled from GPT-4. It writes a critique and a decision but no confidence, so its confidence comes from agreement across repeated samples and swapped orders.",
        `Recorded: ${num(M.verdicts["auto-j-13b"])} verdicts of ${num(totalCalls)}. The rest were too long for it, ties, or no verdict at all: on padded turn-2 comparisons its critique runs out of room before it decides ${autojNoVerdict} of the time.`] },
    { id: "signals", col: 4, bar: "05 · signals", title: "Confidence signals", sub: "six ways to ask “how sure?”",
      list: ["Stated confidence: the number the judge writes.",
        "Token probability: how much probability the model put on its verdict.",
        "Order-swap agreement: does it pick the same answer in both orders? The one signal all three judges have.",
        "Self-consistency: how often re-sampled verdicts agree.",
        "Prompt ensemble: agreement across three reworded prompts (Qwen only).",
        "Meta-model: a small Bayesian model that combines cheap signals to predict when the judge is wrong."] },
    { id: "scoring", col: 5, bar: "06 · metrics", title: "Scoring", sub: "calibration · error detection · κ",
      list: ["Calibration: when it says 90%, is it right 90% of the time? Expected calibration error with equal-count bins, and the overconfidence gap.",
        "Error detection: does lower confidence pick out the wrong verdicts? AUROC, where 0.5 is chance.",
        "Agreement beyond chance (Cohen's κ), always next to accuracy, because raw agreement flatters a judge.",
        "Position bias: how often a verdict flips when the answers swap places."] },
    { id: "bars", col: 6, bar: "07 · uncertainty", title: "Honest error bars", sub: "resample whole questions",
      body: [`There are only ${M.n_questions} questions, and comparisons on the same question are not independent. So every interval resamples whole questions (a cluster bootstrap), and presentations are compared on the same items, as pairs.`,
        "The meta-model is cross-validated with every question kept on one side of the split, repeated over ten seeds, and checked against a permutation null: the same model trained on shuffled labels."] },
  ];
  const METHOD_EDGES = [["data", "labels"], ["labels", "present"], ["present", "qwen"], ["present", "kev"], ["present", "autoj"],
    ["qwen", "signals"], ["kev", "signals"], ["autoj", "signals"], ["signals", "scoring"], ["scoring", "bars"]];
  const MG = { selected: "data" };

  function buildMethodGraph() {
    const graph = $("m-graph");
    const cols = [...new Set(METHOD_NODES.map((n) => n.col))].map((col) =>
      h("div", { class: `m-col${METHOD_NODES.filter((n) => n.col === col).length > 1 ? " multi" : ""}` },
        METHOD_NODES.filter((n) => n.col === col).map((node) =>
        h("button", { type: "button", class: "m-node", id: `m-${node.id}`, "aria-pressed": "false",
          onclick: () => { MG.selected = node.id; renderMethod(); } },
          h("span", { class: "m-bar" }, node.judge ? h("span", { class: "dot", style: { background: judgeColor(node.judge) } }) : null, node.bar),
          h("span", { class: "m-title" }, node.title),
          h("span", { class: "m-sub" }, node.sub)))));
    graph.append(...cols);
  }

  /** Connectors between node boxes: left-to-right curves on a wide screen,
      top-to-bottom when the columns stack. */
  /** Elbow connector: straight out of one box, a rounded 90° turn at the branch
      line (x = xm, or y = ym when stacked), straight into the other. Edges that
      fan out or in share the segment up to the branch line, so they visibly split
      from, and merge into, one point. */
  function elbowH(x1, y1, x2, y2, xm) {
    if (Math.abs(y2 - y1) < 0.5) return `M${x1},${y1} H${x2}`;
    const r = Math.min(10, Math.abs(y2 - y1) / 2, Math.abs(xm - x1), Math.abs(x2 - xm));
    const sx = Math.sign(x2 - x1), sy = Math.sign(y2 - y1);
    return `M${x1},${y1} H${xm - sx * r} Q${xm},${y1} ${xm},${y1 + sy * r} V${y2 - sy * r} Q${xm},${y2} ${xm + sx * r},${y2} H${x2}`;
  }
  function elbowV(x1, y1, x2, y2, ym) {
    if (Math.abs(x2 - x1) < 0.5) return `M${x1},${y1} V${y2}`;
    const r = Math.min(10, Math.abs(x2 - x1) / 2, Math.abs(ym - y1), Math.abs(y2 - ym));
    const sx = Math.sign(x2 - x1), sy = Math.sign(y2 - y1);
    return `M${x1},${y1} V${ym - sy * r} Q${x1},${ym} ${x1 + sx * r},${ym} H${x2 - sx * r} Q${x2},${ym} ${x2},${ym + sy * r} V${y2}`;
  }

  function drawEdges() {
    const svg = $("m-edges"), graph = $("m-graph");
    svg.setAttribute("viewBox", `0 0 ${graph.clientWidth} ${graph.clientHeight}`);
    const count = (i) => METHOD_EDGES.reduce((acc, e) => ({ ...acc, [e[i]]: (acc[e[i]] || 0) + 1 }), {});
    const outgoing = count(0), incoming = count(1);
    // Layout position inside the graph (node -> its column -> the graph), which
    // ignores the selected node's 2px lift and so never depends on its transition.
    const rel = (el) => {
      const l = el.offsetLeft + el.offsetParent.offsetLeft, t = el.offsetTop + el.offsetParent.offsetTop;
      return { l, r: l + el.offsetWidth, t, b: t + el.offsetHeight };
    };
    // One decision for the whole graph: stacked when step 02 sits below step 01.
    // Deciding per edge breaks when wide boxes happen to overlap sideways.
    const stacked = rel($("m-labels")).t >= rel($("m-data")).b - 1;
    const paths = METHOD_EDGES.map(([from, to]) => {
      const p = rel($(`m-${from}`)), q = rel($(`m-${to}`));
      // A fan-in edge is drawn from its shared end, so every dash on the shared
      // segment lines up; its animation runs in reverse to keep the flow direction.
      const merging = incoming[to] > 1;
      let d;
      if (!stacked) {
        const y1 = (p.t + p.b) / 2, y2 = (q.t + q.b) / 2, xm = (p.r + q.l) / 2;
        d = merging ? elbowH(q.l, y2, p.r, y1, xm) : elbowH(p.r, y1, q.l, y2, xm);
      } else {
        const x1 = (p.l + p.r) / 2, x2 = (q.l + q.r) / 2;
        const gap = q.t - p.b;
        const ym = outgoing[from] > 1 ? p.b + Math.min(15, gap / 2) : merging ? q.t - Math.min(15, gap / 2) : p.b + gap / 2;
        d = merging ? elbowV(x2, q.t, x1, p.b, ym) : elbowV(x1, p.b, x2, q.t, ym);
      }
      const active = from === MG.selected || to === MG.selected;
      return { active, el: s("path", { class: `edge${active ? " active" : ""}${merging ? " rev" : ""}`, d }) };
    });
    // Active edges last, so their colour sits on top of a shared grey segment.
    svg.replaceChildren(...paths.filter((e) => !e.active).map((e) => e.el), ...paths.filter((e) => e.active).map((e) => e.el));
  }

  function renderMethod() {
    const node = METHOD_NODES.find((n) => n.id === MG.selected);
    // The selected step turns ink, or its judge's colour as in "Trick the judge". Red is
    // kept for data meaning (wrong / flipped), so it never marks a selection here.
    for (const el of [$("m-graph"), $("m-detail")]) {
      el.style.setProperty("--sel", node.judge ? judgeFill(node.judge) : cssVar("--ink"));
      el.style.setProperty("--on-sel", node.judge ? onJudgeColor(node.judge) : cssVar("--paper"));
    }
    METHOD_NODES.forEach((n) => $(`m-${n.id}`).setAttribute("aria-pressed", String(n.id === MG.selected)));
    $("m-detail").replaceChildren(
      h("div", { class: "titlebar" }, `${node.bar} · ${node.title}`),
      h("div", { class: "win-body" },
        (node.body || []).map((text) => h("p", {}, text)),
        node.list ? h("ul", {}, node.list.map((text) => h("li", {}, text))) : null));
    drawEdges();
  }
  buildMethodGraph();
  renderMethod();

  let layoutTimer;
  window.addEventListener("resize", () => {
    clearTimeout(layoutTimer);
    layoutTimer = setTimeout(() => { drawEdges(); drawMeasures(); }, 150);
  });

  // Theme: follows the device setting until the viewer picks one, which is then
  // remembered (the inline script in <head> applies it before the first paint).
  const themeButton = $("theme-btn");
  const systemDark = window.matchMedia("(prefers-color-scheme: dark)");
  const effectiveTheme = () => document.documentElement.dataset.theme || (systemDark.matches ? "dark" : "light");
  function syncThemeButton() {
    const label = effectiveTheme() === "dark" ? "Switch to light mode" : "Switch to dark mode";
    themeButton.setAttribute("aria-label", label);
    themeButton.title = label;
  }
  themeButton.addEventListener("click", () => {
    const next = effectiveTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (e) { /* private mode: the choice lasts this visit */ }
    syncThemeButton();
    window.dispatchEvent(new Event("themechange"));
  });
  systemDark.addEventListener("change", () => {
    if (!document.documentElement.dataset.theme) { syncThemeButton(); window.dispatchEvent(new Event("themechange")); }
  });
  // Colours the script writes inline (judge dots, chart lines, the method graph's
  // selection) are read from CSS when drawn, so redraw them for the new theme.
  window.addEventListener("themechange", () => {
    renderTrick(); renderGame(); renderShowdown(); drawMeasures(); renderMethod();
    updateCost({ newSeries: false });
  });
  syncThemeButton();

  // Top bar: on narrow screens the links fold into a menu button (CSS decides when;
  // the has-menu class means the button works, so without JS the links stay visible).
  const topbar = document.querySelector(".topbar"), menuButton = $("menu-btn");
  topbar.classList.add("has-menu");
  function setMenu(open) {
    topbar.classList.toggle("open", open);
    menuButton.setAttribute("aria-expanded", String(open));
    menuButton.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  }
  menuButton.addEventListener("click", () => setMenu(!topbar.classList.contains("open")));
  document.querySelectorAll(".topbar nav a").forEach((a) => a.addEventListener("click", () => setMenu(false)));
  document.addEventListener("click", (e) => { if (!topbar.contains(e.target)) setMenu(false); });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && topbar.classList.contains("open")) { setMenu(false); menuButton.focus(); }
  });

  // Top bar: mark the section currently in view.
  const navLinks = [...document.querySelectorAll(".topbar nav a")];
  // Active = the last section whose top has passed the middle of the viewport;
  // the footer is short, so it also wins once the page is scrolled to the bottom.
  function markActiveSection() {
    const atBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4;
    let active = null;
    for (const a of navLinks) {
      if ($(a.hash.slice(1)).getBoundingClientRect().top <= window.innerHeight / 2) active = a;
    }
    if (atBottom) active = navLinks[navLinks.length - 1];
    navLinks.forEach((a) => a.setAttribute("aria-current", String(a === active)));
  }
  window.addEventListener("scroll", markActiveSection, { passive: true });
  markActiveSection();

  renderSignalSelect();
  updateCost({ newSeries: true });
})();
