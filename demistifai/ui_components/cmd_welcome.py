"""Animated EU AI Act terminal component (per-char colored typing, skip, reduced-motion, progressive post-install)."""

from __future__ import annotations
from textwrap import dedent
from typing import Iterable, List, Optional
import json
from streamlit.components.v1 import html as components_html

_TERMINAL_SUFFIX = "ai_act_fullterm"

_WELCOME_LINES: List[str] = [
    "> What is an AI system?\n",
    "$ fetch EU_AI_ACT.AI_system_definition\n",
    "“AI system” means a machine-based system designed to operate with varying levels of autonomy and that may exhibit adaptiveness after deployment, and that, for explicit or implicit objectives, infers—\n",
    "[…stream truncated…]\n",
    "ERROR 422: Definition overload — too many concepts at once.\n",
    "HINT: Let’s learn it by doing.\n",
    "$ pip install demAI\n",
    "Resolving dependencies… ✓\n",
    "Setting up interactive labs… ✓\n",
    "Verifying examples… ✓\n",
    "Progress 0%   [████████████████████] 100%\n",
    "\nWelcome to demAI — a hands-on way to see how AI works and how the EU AI Act applies in practice.\n",
    "> demonstrateAI\n",
    "Build and run a tiny AI system — from data to predictions — step by step.\n",
    "> demystifyAI\n",
    "Turn buzzwords into concrete actions you can try and understand.\n",
    "> democratizeAI\n",
    "Give everyone the confidence to use AI responsibly with clarity and trust.\n",
    "$ start demo\n",
]

_TERMINAL_STYLE = dedent(f"""
<style>
  .terminal-{_TERMINAL_SUFFIX} {{
    width: 95%;
    height: 95%;
    background: #0d1117;
    color: #e5e7eb;
    font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    border-radius: 12px;
    padding: 1.5rem 1rem 1.3rem;
    position: relative;
    overflow: hidden;
  }}
  .terminal-{_TERMINAL_SUFFIX}::before {{
    content: '●  ●  ●';
    position: absolute; top: 8px; left: 12px;
    color: #ef4444cc; letter-spacing: 6px; font-size: .9rem;
  }}
  .term-body-{_TERMINAL_SUFFIX} {{
    margin-top: .8rem;
    white-space: pre-wrap;
    word-wrap: break-word;
    line-height: 1.6;
    font-size: .96rem;
  }}
  .caret-{_TERMINAL_SUFFIX} {{
    margin-left: 2px;
    display:inline-block; width:6px; height:1rem;
    background:#22d3ee; vertical-align:-0.18rem;
    animation: blink-{_TERMINAL_SUFFIX} .85s steps(1,end) infinite;
  }}
  @keyframes blink-{_TERMINAL_SUFFIX} {{ 50% {{ opacity: 0; }} }}

  .cmdline-{_TERMINAL_SUFFIX} {{ color: #93c5fd; }}
  .kw-act-{_TERMINAL_SUFFIX} {{ color: #facc15; font-weight: 600; }}
  .kw-hint-{_TERMINAL_SUFFIX} {{ color: #34d399; font-weight: 600; }}
  .kw-error-{_TERMINAL_SUFFIX}{{ color: #fb7185; font-weight: 700; }}
  .kw-progress-{_TERMINAL_SUFFIX} {{ color: #a5f3fc; font-weight: 600; }}
  .hl-{_TERMINAL_SUFFIX}       {{ color: #fbbf24; font-weight: 700; }}

  .terminal-wrap-{_TERMINAL_SUFFIX} {{
    opacity: 0; transform: translateY(6px);
    animation: fadein-{_TERMINAL_SUFFIX} .6s ease forwards;
  }}
  @keyframes fadein-{_TERMINAL_SUFFIX} {{ to {{ opacity: 1; transform: translateY(0) }} }}

  .skip-{_TERMINAL_SUFFIX} {{
    position: absolute; top: 10px; right: 10px;
    background: #111827; color: #e5e7eb; border: 1px solid #374151;
    border-radius: 6px; padding: .25rem .6rem; font-size: .8rem; cursor: pointer;
  }}
  .skip-{_TERMINAL_SUFFIX}:hover {{ background:#1f2937; }}

  @media (prefers-reduced-motion: reduce) {{
    .caret-{_TERMINAL_SUFFIX} {{ animation: none; }}
    .terminal-wrap-{_TERMINAL_SUFFIX} {{ animation: none; opacity:1; transform:none; }}
  }}
</style>
""")

def render_ai_act_terminal(
    demai_lines: Optional[Iterable[str]] = None,
    speed_type_ms: int = 20,
    speed_delete_ms: int = 14,        # API compatibility (unused)
    pause_between_ops_ms: int = 360,
    key: str = "ai_act_terminal",
    show_caret: bool = True,
    show_skip: bool = True,
    max_total_duration_ms: int = 12000,
) -> None:
    lines = list(demai_lines) if demai_lines is not None else _WELCOME_LINES

    payload = {
        "lines": lines,
        "speedType": max(0, int(speed_type_ms)),
        "pauseBetween": max(0, int(pause_between_ops_ms)),
        "speedDelete": max(0, int(speed_delete_ms)),
        "showCaret": bool(show_caret),
        "showSkip": bool(show_skip),
        "maxDuration": max(0, int(max_total_duration_ms)),
        "suffix": _TERMINAL_SUFFIX,
        "domId": f"term-{key}",
    }

    final_text = "".join((l if str(l).endswith("\n") else f"{l}\n") for l in lines)

    components_html(
        f"""
{_TERMINAL_STYLE}
<div class="terminal-wrap-{_TERMINAL_SUFFIX}">
  <div id="{payload['domId']}" class="terminal-{_TERMINAL_SUFFIX}" role="region" aria-label="EU AI Act terminal animation">
    {'<button class="skip-' + _TERMINAL_SUFFIX + '" type="button">Skip</button>' if payload['showSkip'] else ''}
    <pre class="term-body-{_TERMINAL_SUFFIX}"></pre>
    <span class="caret-{_TERMINAL_SUFFIX}" style="display:{'inline-block' if payload['showCaret'] else 'none'}"></span>
  </div>
</div>

<noscript>
  <div class="terminal-{_TERMINAL_SUFFIX}">
    <pre class="term-body-{_TERMINAL_SUFFIX}">{final_text}</pre>
  </div>
</noscript>

<script>
(function() {{
  const cfg = {json.dumps(payload)};
  const root  = document.getElementById(cfg.domId);
  if(!root) return;

  const pre   = root.querySelector(".term-body-" + cfg.suffix);
  const caret = root.querySelector(".caret-" + cfg.suffix);
  const skipBtn = root.querySelector(".skip-" + cfg.suffix);

  const rawLines = (cfg.lines || []).map(l => (l == null ? "" : String(l)));
  const toLinesWithNL = (arr) => arr.map(l => l.endsWith("\\n") ? l : (l + "\\n"));
  const esc = (s) => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

  const RULES = [
    {{ name: "cmd",    test: (line) => line.startsWith("$ "),            cls: "cmdline-" + cfg.suffix, fullLine: true }},
    {{ name: "hint",   test: (line) => /^HINT:/i.test(line.trim()),      cls: "kw-hint-" + cfg.suffix }},
    {{ name: "error",  test: (line) => /^ERROR\\b/i.test(line.trim()),   cls: "kw-error-" + cfg.suffix }},
    {{ name: "demo",   test: (line) => /^>\\s*dem[a-z]*ai$/i.test(line.trim()), cls: "hl-" + cfg.suffix, fullLine: true }},
    {{ name: "progress", test: (line) => /\\[█+\\]/.test(line),           cls: "kw-progress-" + cfg.suffix }},
  ];

  const TOKEN_PATTERNS = [
    {{ regex: /EU AI Act/gi,             cls: "kw-act-" + cfg.suffix }},
    {{ regex: /AI system/gi,             cls: "kw-act-" + cfg.suffix }},
  ];

  /* ---------- NEW: Expand lines for progressive post-install ---------- */

  // Per-line pause overrides (after completing each line)
  const LINE_PAUSE_OVERRIDES = [
    {{ regex: /^Resolving dependencies… ✓\\s*$/ }},
    {{ regex: /^Setting up interactive labs… ✓\\s*$/ }},
    {{ regex: /^Verifying examples… ✓\\s*$/ }},
  ];
  const EXTRA_PAUSE_MS = 420; // extra time after each of the above

  // Progress animation generator: turns a single "Progress 0% [...] 100%" into frames
  function expandProgressFrames(line) {{
    if (!/^Progress\\s+0%/.test(line)) return null;
    const frames = [
      "Progress 0%    [█...................] 0%\\n",
      "Progress 35%   [███████.............] 35%\\n",
      "Progress 70%   [██████████████......] 70%\\n",
      "Progress 100%  [████████████████████] 100%\\n",
    ];
    return frames;
  }}

  // Build expanded lines + pause map
  const expanded = [];
  const pauseAfter = [];  // extra pause (ms) after each expanded line
  for (const line of toLinesWithNL(rawLines)) {{
    const prog = expandProgressFrames(line);
    if (prog) {{
      for (const f of prog) {{ expanded.push(f); pauseAfter.push(240); }} // small pauses between frames
      continue;
    }
    expanded.push(line);
    // add extra pause for the three "✓" lines
    const needsExtra = LINE_PAUSE_OVERRIDES.some(p => p.regex.test(line.trim()));
    pauseAfter.push(needsExtra ? EXTRA_PAUSE_MS : 0);
  }}

  /* ---------- Coloring/tokenization (same as before, but applied to expanded) ---------- */

  function segmentsForLine(line) {{
    const segments = [];
    const fullLineRule = RULES.find(r => r.fullLine && r.test(line));
    if (fullLineRule) {{
      segments.push({{ text: line, cls: fullLineRule.cls }});
      return segments;
    }}
    const marks = new Array(line.length).fill(null);
    TOKEN_PATTERNS.forEach(p => {{
      let m;
      const re = new RegExp(p.regex.source, p.regex.flags);
      while ((m = re.exec(line)) !== null) {{
        for (let i = m.index; i < m.index + m[0].length; i++) {{
          if (marks[i] == null) marks[i] = p.cls;
        }}
      }}
    }});
    const lineWide = RULES.find(r => !r.fullLine && r.test(line));
    const baseCls = lineWide ? lineWide.cls : null;

    let i = 0;
    while (i < line.length) {{
      const cls = marks[i] || baseCls;
      let j = i + 1;
      while (j < line.length && (marks[j] || baseCls) === cls) j++;
      const chunk = line.slice(i, j);
      segments.push({{ text: chunk, cls }});
      i = j;
    }}
    return segments;
  }}

  function renderHTMLFromBuffer(linesSegs) {{
    const htmlLines = linesSegs.map(segs => segs.map(s =>
      s.cls ? `<span class="${{s.cls}}">${{esc(s.text)}}</span>` : esc(s.text)
    ).join("")).join("");
    pre.innerHTML = htmlLines;
    autoResize();
  }}

  const autoResize = () => {{
    const height = root.scrollHeight + 24;
    window.parent.postMessage({{ "type": "streamlit:resize", "height": height }}, "*");
  }};
  const ro = new ResizeObserver(() => autoResize());
  ro.observe(root);

  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (prefersReduced || cfg.speedType === 0) {{
    const segsAll = expanded.map(segmentsForLine);
    renderHTMLFromBuffer(segsAll);
    if (caret) caret.style.display = "none";
    return;
  }}

  const LINES_SEGS = expanded.map(segmentsForLine);

  const TYPE_DELAY = Math.max(0, cfg.speedType);
  const BETWEEN_LINES = Math.max(0, cfg.pauseBetween);
  const END_AT = Date.now() + Math.max(0, cfg.maxDuration || 0);

  let li = 0, si = 0, ci = 0;
  const rendered = [];
  let current = [];

  function flush() {{ renderHTMLFromBuffer(rendered.concat([current])); }}

  function finishAll() {{
    renderHTMLFromBuffer(LINES_SEGS);
    if (caret) caret.style.display = "none";
  }}

  function nextLine() {{
    if (li >= LINES_SEGS.length) {{
      if (caret) caret.style.display = "none";
      return;
    }}
    current = LINES_SEGS[li].map(s => ({{ text: "", cls: s.cls, _full: s.text }}));
    si = 0; ci = 0;
    step();
  }}

  function step() {{
    if (Date.now() > END_AT) return finishAll();
    if (li >= LINES_SEGS.length) {{
      if (caret) caret.style.display = "none";
      return;
    }}

    if (si >= current.length) {{
      // line complete: commit & pause (with override if any)
      rendered.push(current.map(s => ({{ text: s._full, cls: s.cls }})));
      const extra = pauseAfter[li] || 0;
      li += 1; si = 0; ci = 0;
      setTimeout(nextLine, BETWEEN_LINES + extra);
      return;
    }}

    const seg = current[si];
    const full = seg._full;
    const CHUNK = 1; // keep per-char; set to 2–3 if you want a bit faster
    const nextIdx = Math.min(full.length, ci + CHUNK);
    seg.text = full.slice(0, nextIdx);
    ci = nextIdx;

    if (ci >= full.length) {{ si += 1; ci = 0; }}

    flush();
    setTimeout(step, TYPE_DELAY);
  }}

  if (skipBtn) {{
    skipBtn.addEventListener("click", () => finishAll());
  }}

  nextLine();
}})();
</script>
        """,
        height=800,
    )