"""Animated EU AI Act terminal component (non-blocking, auto-resizing, f-string safe)."""

from __future__ import annotations
from textwrap import dedent
from typing import Iterable, List, Optional, Sequence
import html
import json
import re
from streamlit.components.v1 import html as components_html

_TERMINAL_SUFFIX = "ai_act_fullterm"

_DEFAULT_DEMAI_LINES: List[str] = [
    "$ ai-act fetch --article 3 --term--AI_system\n",
    "[...] Retrieving definition… OK\n",
    "",
    "'AI system means a machine-based system [...]'\n",
    "",
    "> Are you wondering what that means?\n",
    "",
    "$ demAI start --module machine\n",
    "[...] Initializing demAI.machine…\n",
    "[...] Loading components: ui ▸ model ▸ infrastructure\n",
    "[...] Progress 0%   [░░░░░░░░░░░░░░░░░░░░]\n",
    "[... Progress 45%  [██████████░░░░░░░░░░]\n",
    "[...] Progress 100% [████████████████████]\n",
    "",
    "You are already inside a machine-based system: a user interface (software) running in the cloud (hardware).",
    "In each stage, this window will guide you with prompts and key information. Use the control room to jump between",
    "stages and enable Nerd Mode for deeper details.\n",
    "",
    ":help Scroll this page to find out more about the demAI machine.\n",
]

_TERMINAL_STYLE = dedent(f"""
<style>
  .terminal-{_TERMINAL_SUFFIX} {{
    width: min(100%, 680px);
    height: auto;
    background: #0d1117;
    color: #ffffff;
    font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    border-radius: 12px;
    padding: 1.5rem 1.2rem 1.3rem;
    position: relative;
    overflow: hidden;
    margin: 0 auto;
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
  .cmd-{_TERMINAL_SUFFIX} {{ color: #93c5fd; }}
  .err-{_TERMINAL_SUFFIX} {{ color: #fb7185; font-weight: 700; }}
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

  @media (max-width: 640px) {{
    .terminal-{_TERMINAL_SUFFIX} {{
      border-radius: 10px;
      padding: clamp(1rem, 5vw, 1.35rem);
    }}
    .term-body-{_TERMINAL_SUFFIX} {{
      font-size: .9rem;
    }}
  }}
</style>
""")


def _normalize_lines(lines: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for raw in lines:
        line = str(raw)
        normalized.append(line if line.endswith("\n") else f"{line}\n")
    return normalized


def _escape_text(text: str) -> str:
    return html.escape(text, quote=False)


def _highlight_line(line: str, suffix: str) -> str:
    stripped = line.strip()
    if not line:
        return ""
    if line.startswith("$ "):
        return f'<span class="cmd-{suffix}">{_escape_text(line)}</span>'
    if re.match(r"^ERROR\b", stripped, flags=re.IGNORECASE):
        return f'<span class="err-{suffix}">{_escape_text(line)}</span>'
    return _escape_text(line)


def _compute_full_html(lines: Sequence[str], suffix: str) -> str:
    normalized = _normalize_lines(lines)
    final_text = "".join(normalized)
    highlighted_lines = [_highlight_line(part, suffix) for part in final_text.split("\n")]
    return "\n".join(highlighted_lines)

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
    normalized_lines = _normalize_lines(lines)
    full_html = _compute_full_html(lines, _TERMINAL_SUFFIX)

    payload = {
        "lines": lines,
        "fullHtml": full_html,
        "speedType": max(0, int(speed_type_ms)),
        "pauseBetween": max(0, int(pause_between_ops_ms)),
        "speedDelete": max(0, int(speed_delete_ms)),
        "showCaret": bool(show_caret),
        "suffix": _TERMINAL_SUFFIX,
        "domId": f"term-{key}",
    }

    # ---- IMPORTANT: precompute noscript text to avoid backslashes inside f-string expressions
    final_text = "".join(normalized_lines)

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

  const highlight = (line) => {{
    const stripped = line.trim();
    if (line.startsWith("$ ")) return `<span class="cmd-${{cfg.suffix}}">${{esc(line)}}</span>`;
    if (/^ERROR\b/i.test(stripped)) return `<span class="err-${{cfg.suffix}}">${{esc(line)}}</span>`;
    return esc(line);
  }};
  const renderHighlighted = (raw) => {{
    pre.innerHTML = raw.split("\\n").map(highlight).join("\\n");
  }};

  const finalRaw = toLinesWithNL(rawLines).join("");
  const computedFinalHtml = finalRaw ? finalRaw.split("\\n").map(highlight).join("\\n") : "";
  const finalHtml = typeof cfg.fullHtml === "string" ? cfg.fullHtml : computedFinalHtml;
  cfg.fullHtml = finalHtml;

  const ensureMeasuredHeight = () => {{
    const measurement = root.cloneNode(true);
    measurement.removeAttribute("id");
    measurement.style.position = "absolute";
    measurement.style.visibility = "hidden";
    measurement.style.pointerEvents = "none";
    measurement.style.opacity = "0";
    measurement.style.left = "-9999px";
    measurement.style.top = "0";
    measurement.style.height = "auto";
    measurement.style.minHeight = "auto";
    measurement.style.maxHeight = "none";
    const width = root.getBoundingClientRect().width || root.offsetWidth || root.clientWidth;
    if (width) {{
      measurement.style.width = width + "px";
    }}
    const measurePre = measurement.querySelector(".term-body-" + cfg.suffix);
    if (measurePre) {{
      measurePre.innerHTML = finalHtml;
    }}
    const measureCaret = measurement.querySelector(".caret-" + cfg.suffix);
    if (measureCaret) {{
      measureCaret.style.display = "none";
    }}
    (root.parentElement || document.body).appendChild(measurement);
    const height = measurement.scrollHeight;
    measurement.remove();
    if (height) {{
      root.style.minHeight = `${{height}}px`;
      root.style.height = `${{height}}px`;
    }}
    return height;
  }};

  const measuredHeight = ensureMeasuredHeight();
  if (Number.isFinite(measuredHeight) && measuredHeight > 0) {{
    window.parent.postMessage({{ "type": "streamlit:resize", "height": measuredHeight + 24 }}, "*");
  }}

  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (prefersReduced || cfg.speedType === 0) {{
    pre.innerHTML = finalHtml;
    if (caret) caret.style.display = "none";
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
      return;
    }}

    const target = rawLines[iLine].endsWith("\\n") ? rawLines[iLine] : (rawLines[iLine] + "\\n");

    if (iChar < target.length) {{
      buffer += target[iChar++];
      pre.textContent = buffer;   // fast during typing
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
