"""Animated EU AI Act terminal component (non-blocking, auto-resizing, f-string safe)."""

from __future__ import annotations
from textwrap import dedent
from typing import Iterable, List, Optional
import json
from streamlit.components.v1 import html as components_html

_TERMINAL_SUFFIX = "ai_act_fullterm"

_DEFAULT_DEMAI_LINES: List[str] = [
    "┌──────── demAI Terminal ──────────────────────────┐\n",
    "│ Session: EU AI Act — Article 3 (AI system)       │\n",
    "└──────────────────────────────────────────────────┘\n",
    "",
    "$ whoami\n",
    "visitor\n",
    "",
    "$ ai-act fetch --article 3 --term \"AI system\"\n",
    "[12:00:01] Connecting to EU AI Act repository… OK\n",
    "[12:00:01] Retrieving Article 3 definition… OK\n",
    "",
    "AI system means a machine-based system designed to operate with varying levels of autonomy that,",
    "for explicit or implicit objectives, infers, from the input it receives, how to generate outputs such as",
    "predictions, content, recommendations, or decisions that can influence physical or virtual environments.\n",
    "",
    "> What does that actually mean?\n",
    "",
    "$ demAI start --module machine\n",
    "[12:00:03] Initializing demAI.machine…\n",
    "[12:00:03] Loading components: ui ▸ model ▸ data ▸ infra\n",
    "[12:00:04] Progress 0%   [░░░░░░░░░░░░░░░░░░░░]\n",
    "[12:00:05] Progress 45%  [██████████░░░░░░░░░░]\n",
    "[12:00:06] Progress 100% [████████████████████]\n",
    "",
    "You are already inside a machine-based system: a user interface (software) running in the cloud (hardware).",
    "On this page you’ll move through each stage with interactive prompts. Use the control room to jump between",
    "stages and enable Nerd Mode for deeper details.\n",
    "",
    ":help Scroll this page to find out more about the demAI machine.\n",
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
  .cmdline-{_TERMINAL_SUFFIX} {{ color: #93c5fd; }}
  .hl-{_TERMINAL_SUFFIX}     {{ color: #a5f3fc; font-weight: 600; }}
  .kw-act-{_TERMINAL_SUFFIX}         {{ color: #facc15; font-weight: 600; }}
  .kw-compliance-{_TERMINAL_SUFFIX}  {{ color: #34d399; font-weight: 600; }}
  .kw-data-{_TERMINAL_SUFFIX}        {{ color: #60a5fa; font-weight: 600; }}
  .kw-oversight-{_TERMINAL_SUFFIX}   {{ color: #f472b6; font-weight: 600; }}
  .kw-risk-{_TERMINAL_SUFFIX}        {{ color: #fb7185; font-weight: 600; }}
  .kw-transparency-{_TERMINAL_SUFFIX}{{ color: #fbbf24; font-weight: 600; }}
  @keyframes blink-{_TERMINAL_SUFFIX} {{ 50% {{ opacity: 0; }} }}

  .terminal-wrap-{_TERMINAL_SUFFIX} {{
    opacity: 0; transform: translateY(6px);
    animation: fadein-{_TERMINAL_SUFFIX} .6s ease forwards;
  }}
  @keyframes fadein-{_TERMINAL_SUFFIX} {{
    to {{ opacity: 1; transform: translateY(0) }}
  }}

  @media (prefers-reduced-motion: reduce) {{
    .caret-{_TERMINAL_SUFFIX} {{ animation: none; }}
    .terminal-wrap-{_TERMINAL_SUFFIX} {{ animation: none; opacity:1; transform:none; }}
  }}
</style>
""")

def render_ai_act_terminal(
    demai_lines: Optional[Iterable[str]] = None,
    speed_type_ms: int = 20,
    speed_delete_ms: int = 14,        # kept for API compatibility
    pause_between_ops_ms: int = 360,  # pause between lines
    key: str = "ai_act_terminal",
    show_caret: bool = True,
) -> None:
    """
    Render the animated EU AI Act terminal sequence using a client-side typing loop with auto-resize.

    - Non-blocking: the rest of the Streamlit page renders immediately.
    - Auto-resizing: iframe height grows with content while typing.
    - Honors 'prefers-reduced-motion': final state is shown if motion is reduced.
    - f-string safe: avoids backslashes inside f-string expressions.
    """
    lines = list(demai_lines) if demai_lines is not None else _DEFAULT_DEMAI_LINES

    payload = {
        "lines": lines,
        "speedType": max(0, int(speed_type_ms)),
        "pauseBetween": max(0, int(pause_between_ops_ms)),
        "speedDelete": max(0, int(speed_delete_ms)),
        "showCaret": bool(show_caret),
        "suffix": _TERMINAL_SUFFIX,
        "domId": f"term-{key}",
    }

    # ---- IMPORTANT: precompute noscript text to avoid backslashes inside f-string expressions
    final_text = "".join((l if str(l).endswith("\n") else f"{l}\n") for l in lines)

    components_html(
        f"""
{_TERMINAL_STYLE}
<div class="terminal-wrap-{_TERMINAL_SUFFIX}">
  <div id="{payload['domId']}" class="terminal-{_TERMINAL_SUFFIX}" role="region" aria-label="EU AI Act terminal animation">
    <pre class="term-body-{_TERMINAL_SUFFIX}"></pre>
    <span class="caret-{_TERMINAL_SUFFIX}" style="display:{'inline-block' if show_caret else 'none'}"></span>
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

  const rawLines = (cfg.lines || []).map(l => (l == null ? "" : String(l)));
  const toLinesWithNL = (arr) => arr.map(l => l.endsWith("\\n") ? l : (l + "\\n"));
  const esc = (s) => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

  const KEYWORD_PATTERNS = [
    {{ regex: /EU AI Act/gi, cls: "kw-act-" + cfg.suffix }},
    {{ regex: /AI system/gi, cls: "kw-act-" + cfg.suffix }},
    {{ regex: /compliance|compliant/gi, cls: "kw-compliance-" + cfg.suffix }},
    {{ regex: /data governance/gi, cls: "kw-data-" + cfg.suffix }},
    {{ regex: /human oversight/gi, cls: "kw-oversight-" + cfg.suffix }},
    {{ regex: /high-risk|risk register/gi, cls: "kw-risk-" + cfg.suffix }},
    {{ regex: /transparency/gi, cls: "kw-transparency-" + cfg.suffix }},
  ];

  function highlight(line) {{
    const stripped = line.trim();
    if (/^dem[a-z]*ai$/i.test(stripped)) return `<span class="hl-${{cfg.suffix}}">${{esc(line)}}</span>`;
    if (line.startsWith("$ ")) return `<span class="cmdline-${{cfg.suffix}}">${{esc(line)}}</span>`;
    let escaped = esc(line);
    KEYWORD_PATTERNS.forEach(({{ regex, cls }}) => {{
      escaped = escaped.replace(regex, (match) => `<span class="${{cls}}">${{match}}</span>`);
    }});
    return escaped;
  }}
  function renderHighlighted(raw) {{
    pre.innerHTML = raw.split("\\n").map(highlight).join("\\n");
    autoResize();
  }}

  // --- AUTO-RESIZE: notify Streamlit when content height changes ---
  const autoResize = () => {{
    const height = root.scrollHeight + 24; // small padding for shadow
    window.parent.postMessage({{ "type": "streamlit:resize", "height": height }}, "*");
  }};
  const resizeObserver = new ResizeObserver(() => autoResize());
  resizeObserver.observe(root);

  // --- Reduced motion: render final state immediately ---
  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (prefersReduced || cfg.speedType === 0) {{
    const finalRaw = toLinesWithNL(rawLines).join("");
    renderHighlighted(finalRaw);
    if (caret) caret.style.display = "none";
    autoResize();
    return;
  }}

  // --- Typing engine (client-side, non-blocking) ---
  const TYPE_DELAY = Math.max(0, cfg.speedType);
  const BETWEEN_LINES = Math.max(0, cfg.pauseBetween);

  let iLine = 0, iChar = 0, buffer = "";

  function step() {{
    if (iLine >= rawLines.length) {{
      renderHighlighted(buffer);
      if (caret) caret.style.display = "none";
      autoResize();
      return;
    }}

    const target = rawLines[iLine].endsWith("\\n") ? rawLines[iLine] : (rawLines[iLine] + "\\n");

    if (iChar < target.length) {{
      buffer += target[iChar++];
      pre.textContent = buffer;   // fast during typing
      autoResize();
      setTimeout(step, TYPE_DELAY);
      return;
    }}

    // End of line: re-render with highlighting for what we have so far
    renderHighlighted(buffer);
    iLine += 1; iChar = 0;
    setTimeout(step, BETWEEN_LINES);
  }}

  // Start after first paint
  requestAnimationFrame(step);
}})();
</script>
        """,
        height=800,  # minimal initial height; JS will grow it dynamically
    )
