"""Minimal terminal animation: per-char typing, colored tokens, progressive post-install."""

from __future__ import annotations
from textwrap import dedent
from typing import Iterable, List, Optional, Sequence, Tuple
import html
import json
import re
from streamlit.components.v1 import html as components_html

_SUFFIX = "ai_use"

# Preserve the historical constant name referenced throughout the module. A
# previous refactor renamed the suffix constant but missed the call sites,
# leaving `_TERMINAL_SUFFIX` undefined at import time. Keep both names in sync
# so downstream imports remain stable.
_TERMINAL_SUFFIX = _SUFFIX

_DEFAULT_DEMAI_LINES: List[str] = [
    "> Use: run the spam detector on incoming emails\n",
    "$ quote EU_AI_ACT.AI_system_definition --fragment 'infers how to generate content, predictions, recommendations or decisions'\n",
    "‘An AI system infers how to generate content, predictions, recommendations or decisions’\n",
    "$ predict start --stream\n",
    "In this step, the system takes each email (title + body) as input and produces an output.\n",
    "> Output format\n",
    "Prediction: Spam | Safe  •  Confidence: 0.00–1.00  •  Recommendation: place in Spam | Inbox\n",
    "$ predict example --show\n",
    "Title: “Limited time offer!!!”\n",
    "Body:  “Click to claim your prize — ends today.”\n",
    "→ Prediction: Spam  •  Confidence: 0.97  •  Recommendation: Spam ✓\n",
    "$ predict example --show\n",
    "Title: “Meeting notes: Q4 planning”\n",
    "Body:  “Attached are action items from today’s meeting.”\n",
    "→ Prediction: Safe  •  Confidence: 0.91  •  Recommendation: Inbox ✓\n",
    "$ predict batch --size 100\n",
    "Progress 0%   [████████████████████] 100%\n",
    "$ threshold get\n",
    "Current decision threshold: 0.50 • Change affects Spam/Inbox recommendations ✓\n",
    "HINT: Toggle Nerd Mode to adjust thresholding, view per-email confidence, and export predictions.\n",
    "$ continue\n",
]

# Optional alias if you keep *_LINES naming parallelism
_USE_LINES = _DEFAULT_DEMAI_LINES

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

  @media (max-width: 600px) {{
    .terminal-wrap-{_TERMINAL_SUFFIX} {{
      margin: 0;
      padding: 0;
    }}
    .terminal-{_TERMINAL_SUFFIX} {{
      width: 100vw;
      max-width: none;
      min-height: 100vh;
      margin-block: 0;
      margin-inline: calc(50% - 50vw);
      border-radius: 0;
      padding: clamp(1.2rem, 7vw, 1.8rem);
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      gap: 1.1rem;
    }}
    .terminal-{_TERMINAL_SUFFIX}::before {{
      top: 14px;
      left: 50%;
      transform: translateX(-50%);
    }}
    .term-body-{_TERMINAL_SUFFIX} {{
      font-size: 1.05rem;
      line-height: 1.72;
      width: 100%;
      overflow-wrap: anywhere;
      word-break: break-word;
      overflow-x: hidden;
    }}
    .caret-{_TERMINAL_SUFFIX} {{
      align-self: flex-start;
      margin-top: -.2rem;
      flex-shrink: 0;
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


def _split_segments(line: str, suffix: str) -> List[Tuple[str, Optional[str]]]:
    stripped = line.strip()
    if not line:
        return [(line, None)]
    if line.startswith("$ "):
        return [(line, f"cmd-{suffix}")]
    if re.match(r"^ERROR\b", stripped, flags=re.IGNORECASE):
        return [(line, f"err-{suffix}")]
    return [(line, None)]


def _segments_to_html(segments: Sequence[Tuple[str, Optional[str]]]) -> str:
    rendered: List[str] = []
    for text, css in segments:
        escaped = _escape_text(text)
        if css:
            rendered.append(f'<span class="{css}">{escaped}</span>')
        else:
            rendered.append(escaped)
    return "".join(rendered)


def _highlight_line(line: str, suffix: str) -> str:
    return _segments_to_html(_split_segments(line, suffix))


def _compute_full_html(lines: Sequence[str], suffix: str) -> str:
    normalized = _normalize_lines(lines)
    segments = [_split_segments(line, suffix) for line in normalized]
    return "".join(_segments_to_html(parts) for parts in segments)


def _compute_segment_payload(lines: Sequence[str], suffix: str) -> List[List[Tuple[str, Optional[str]]]]:
    normalized = _normalize_lines(lines)
    return [_split_segments(line, suffix) for line in normalized]

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
    segments = _compute_segment_payload(lines, _TERMINAL_SUFFIX)
    full_html = "".join(_segments_to_html(parts) for parts in segments)
    serializable_segments = [
        [{"t": text, "c": css} for text, css in parts]
        for parts in segments
    ]

    payload = {
        "lines": lines,
        "fullHtml": full_html,
        "speedType": max(0, int(speed_type_ms)),
        "pauseBetween": max(0, int(pause_between_ops_ms)),
        "speedDelete": max(0, int(speed_delete_ms)),
        "showCaret": bool(show_caret),
        "suffix": _TERMINAL_SUFFIX,
        "domId": f"term-{key}",
        "segments": serializable_segments,
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

  const rawLines = (cfg.lines || []).map((l) => (l == null ? "" : String(l)));
  const toLinesWithNL = (arr) => arr.map((l) => (l.endsWith("\\n") ? l : l + "\\n"));
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const normaliseSegments = (segments) =>
    segments.map((line) =>
      Array.isArray(line)
        ? line.map((seg) => ({{
            t: typeof seg.t === "string" ? seg.t : "",
            c: typeof seg.c === "string" && seg.c ? seg.c : null,
          }}))
        : []
    );

  const splitSegments = (line) => {{
    const trimmed = line.trim();
    if (line.startsWith("$ ")) return [{{ t: line, c: "cmd-" + cfg.suffix }}];
    if (/^ERROR\b/i.test(trimmed)) return [{{ t: line, c: "err-" + cfg.suffix }}];
    return [{{ t: line, c: null }}];
  }};

  const normalisedLines = toLinesWithNL(rawLines);
  const perLineSegs = Array.isArray(cfg.segments)
    ? normaliseSegments(cfg.segments)
    : normalisedLines.map(splitSegments);

  const perLineSegmentHtml = perLineSegs.map((segs) =>
    segs.map((seg) => (seg.c ? `<span class="${{seg.c}}">${{esc(seg.t)}}</span>` : esc(seg.t)))
  );
  const perLineHtml = perLineSegmentHtml.map((parts) => parts.join(""));
  const perLineRaw = perLineSegs.map((segs) => segs.map((seg) => seg.t).join(""));

  const finalRaw = perLineRaw.join("");
  const computedFinalHtml = perLineHtml.join("");
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

  const doneHtmlParts = [];
  let activeNode = null;
  let lineIndex = 0;
  let segmentIndex = 0;
  let charIndex = 0;

  const syncDoneHtml = () => {{
    pre.innerHTML = doneHtmlParts.join("");
  }};

  const ensureActiveNode = () => {{
    if (activeNode) {{
      return activeNode;
    }}
    const segments = perLineSegs[lineIndex] || [];
    const current = segments[segmentIndex];
    if (!current) {{
      return null;
    }}
    if (current.c) {{
      const span = document.createElement("span");
      span.className = current.c;
      span.textContent = "";
      pre.appendChild(span);
      activeNode = span;
    }} else {{
      activeNode = document.createTextNode("");
      pre.appendChild(activeNode);
    }}
    return activeNode;
  }};

  const commitActiveSegment = () => {{
    if (!activeNode) {{
      return;
    }}
    if (activeNode.parentNode === pre) {{
      pre.removeChild(activeNode);
    }}
    const segmentHtml = (perLineSegmentHtml[lineIndex] || [])[segmentIndex] || "";
    doneHtmlParts.push(segmentHtml);
    activeNode = null;
    syncDoneHtml();
  }};

  syncDoneHtml();

  function step() {{
    if (lineIndex >= perLineSegs.length) {{
      syncDoneHtml();
      if (caret) {{
        caret.style.display = "none";
      }}
      return;
    }}

    const segments = perLineSegs[lineIndex] || [];
    const current = segments[segmentIndex];

    if (!current) {{
      segmentIndex = 0;
      lineIndex += 1;
      setTimeout(step, BETWEEN_LINES);
      return;
    }}

    const target = current.t;
    if (charIndex < target.length) {{
      const node = ensureActiveNode();
      if (node) {{
        const char = target.charAt(charIndex);
        node.textContent = (node.textContent || "") + char;
      }}
      charIndex += 1;
      setTimeout(step, TYPE_DELAY);
      return;
    }}

    commitActiveSegment();
    segmentIndex += 1;
    charIndex = 0;

    if (segmentIndex >= segments.length) {{
      segmentIndex = 0;
      lineIndex += 1;
      setTimeout(step, BETWEEN_LINES);
      return;
    }}

    setTimeout(step, TYPE_DELAY);
  }}

  requestAnimationFrame(step);
}})();
</script>
        """,
        height=800,  # minimal initial height; JS will grow it dynamically
    )
