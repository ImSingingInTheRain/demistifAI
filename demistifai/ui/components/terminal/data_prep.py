"""Prepare Data terminal animation: per-char typing, colored tokens, progressive status."""

from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple
import html
import json
import re
from streamlit.components.v1 import html as components_html

_SUFFIX = "ai_prepare"

# --- Content for the PREPARE DATA stage (short + progressive) ---
PREPARE_LINES: List[str] = [
    "> Prepare data: teach the model what spam looks like\n",
    "$ quote EU_AI_ACT.AI_system_definition --fragment 'infers from the input it receives'\n",
    "‘An AI system infers from the input it receives […]’\n",
    "$ goal spam_filter --learn 'spam vs safe emails'\n",
    "Your AI system must learn to distinguish a safe email from spam.\n",
    "> Step 1 — Use the Dataset Builder to generate a synthetic dataset.\n",
    "$ dataset.builder --fields subject, body, sender, links --size 2000 --balanced\n",
    "Generating synthetic emails… ✓\n",
    "Progress 0%   [████████████████████] 100%\n",
    "$ dataset.health\n",
    "Health score: 82/100 • Coverage good • Leakage none • Duplicates low ✓\n",
    "Recommendations: add borderline safe examples; diversify link patterns; rebalance 'promotions' subtype.\n",
    "HINT: Toggle Nerd Mode for advanced configuration and diagnostic controls.\n",
    "$ continue\n",
]

# Backwards-compatibility alias for callers that might expect a *_LINES variable
_PREPARE_LINES = PREPARE_LINES

# --- Minimal style (same system as your welcome template) ---
STYLE = """
<style>
  .term-__SFX__{background:#0d1117;color:#e5e7eb;font-family:'Fira Code',ui-monospace,monospace;border-radius:12px;padding:1rem;position:relative}
  .term-__SFX__::before{content:'●  ●  ●';position:absolute;top:8px;left:12px;color:#ef4444cc;letter-spacing:6px;font-size:.9rem}
  .body-__SFX__{white-space:pre-wrap;line-height:1.6;font-size:.96rem;margin-top:.8rem}
  .caret-__SFX__{margin-left:2px;display:inline-block;width:6px;height:1rem;background:#22d3ee;vertical-align:-.18rem;animation:blink-__SFX__ .85s steps(1,end) infinite}
  @keyframes blink-__SFX__{50%{opacity:0}}
  .cmd-__SFX__{color:#93c5fd}
  .hint-__SFX__{color:#34d399;font-weight:600}
  .err-__SFX__{color:#fb7185;font-weight:700}
  .prog-__SFX__{color:#a5f3fc;font-weight:600}
  .pill-__SFX__{color:#fbbf24;font-weight:700}
  .act-__SFX__{color:#facc15;font-weight:600}
  .wrap-__SFX__{opacity:0;transform:translateY(6px);animation:fadein-__SFX__ .4s ease forwards}
  @keyframes fadein-__SFX__{to{opacity:1;transform:none}}
  .skip-__SFX__{position:absolute;top:8px;right:8px;background:#111827;color:#e5e7eb;border:1px solid #374151;border-radius:6px;padding:.2rem .5rem;font-size:.8rem;cursor:pointer}
  .skip-__SFX__:hover{background:#1f2937}
  @media (prefers-reduced-motion: reduce){.caret-__SFX__{animation:none}.wrap-__SFX__{animation:none;opacity:1;transform:none}}
</style>
"""


_TOKEN_REGEX = re.compile(
    r"(AI system|spam|safe|dataset|Health score|Recommendations|Nerd Mode)",
    re.IGNORECASE,
)


def _expand_lines(lines: Sequence[str]) -> Tuple[List[str], List[int]]:
    expanded: List[str] = []
    pauses: List[int] = []
    for raw in lines:
        line = str(raw)
        if not line.endswith("\n"):
            line = f"{line}\n"

        if re.match(r"^Progress\s+0%", line):
            frames = [
                "Progress 0%    [█...................] 0%\n",
                "Progress 40%   [████████............] 40%\n",
                "Progress 75%   [██████████████......] 75%\n",
                "Progress 100%  [████████████████████] 100%\n",
            ]
            expanded.extend(frames)
            pauses.extend([220] * len(frames))
            continue

        expanded.append(line)
        pauses.append(380 if re.search(r"✓\s*$", line) else 0)

    return expanded, pauses


def _split_segments(line: str, suffix: str) -> List[Tuple[str, Optional[str]]]:
    trimmed = line.strip()
    if line.startswith("$ "):
        return [(line, f"cmd-{suffix}")]
    if re.match(r"^HINT:", trimmed, flags=re.IGNORECASE):
        return [(line, f"hint-{suffix}")]
    if re.match(r"^ERROR\b", trimmed, flags=re.IGNORECASE):
        return [(line, f"err-{suffix}")]
    if re.match(r"^>\s", line):
        return [(line, f"pill-{suffix}")]
    if re.search(r"\[█+\]", line):
        return [(line, f"prog-{suffix}")]

    parts: List[Tuple[str, Optional[str]]] = []
    index = 0
    for match in _TOKEN_REGEX.finditer(line):
        start, end = match.span()
        if start > index:
            parts.append((line[index:start], None))
        parts.append((match.group(0), f"act-{suffix}"))
        index = end

    if index < len(line):
        parts.append((line[index:], None))

    return parts or [(line, None)]


def _escape_segment(text: str) -> str:
    return html.escape(text, quote=False)


def _segments_to_html(segments: List[Tuple[str, Optional[str]]]) -> str:
    rendered: List[str] = []
    for text, cls in segments:
        escaped = _escape_segment(text)
        if cls:
            rendered.append(f'<span class="{cls}">{escaped}</span>')
        else:
            rendered.append(escaped)
    return "".join(rendered)


def _compute_full_html(lines: Sequence[str], suffix: str) -> str:
    return "".join(
        _segments_to_html(_split_segments(line, suffix)) for line in lines
    )

HTML = """
{STYLE}
<div class="wrap-__SFX__">
  <div id="__DOMID__" class="term-__SFX__" role="region" aria-label="Prepare Data terminal animation">
    __SKIP__
    <pre class="body-__SFX__"></pre>
    <span class="caret-__SFX__" style="display:__CARET__"></span>
  </div>
</div>
<script>
(function(){
  const cfg = __PAYLOAD__;
  const root = document.getElementById(cfg.domId); if(!root) return;
  const pre = root.querySelector(".body-" + cfg.sfx);
  const caret = root.querySelector(".caret-" + cfg.sfx);
  const skip = root.querySelector(".skip-" + cfg.sfx);

  // helpers
  const esc = s => s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

  // per-line coloring (typed progressively)
  const splitSegments = (line) => {
    const trimmed = line.trim();
    if (line.startsWith("$ ")) return [{t: line, c: "cmd-" + cfg.sfx}];
    if (/^HINT:/i.test(trimmed)) return [{t: line, c: "hint-" + cfg.sfx}];
    if (/^ERROR\\b/i.test(trimmed)) return [{t: line, c: "err-" + cfg.sfx}];
    if (/^>\\s/.test(line)) return [{t: line, c: "pill-" + cfg.sfx}];
    if (/\\[█+\\]/.test(line)) return [{t: line, c: "prog-" + cfg.sfx}];

    // token emphasize: key concepts in this stage
    const parts = [];
    const re = /(AI system|spam|safe|dataset|Health score|Recommendations|Nerd Mode)/gi;
    let idx = 0, m;
    while ((m = re.exec(line)) !== null) {
      if (m.index > idx) parts.push({t: line.slice(idx, m.index), c: null});
      parts.push({t: m[0], c: "act-" + cfg.sfx});
      idx = m.index + m[0].length;
    }
    if (idx < line.length) parts.push({t: line.slice(idx), c: null});
    return parts.length ? parts : [{t: line, c: null}];
  };

  const rawLines = (cfg.expandedLines || cfg.lines || []).map((item) => {
    const value = item == null ? "" : String(item);
    return value.endsWith("\n") ? value : value + "\n";
  });
  const pauses = Array.isArray(cfg.pauses)
    ? cfg.pauses.map(p => Number(p) || 0)
    : new Array(rawLines.length).fill(0);

  const perLineSegs = rawLines.map(splitSegments);
  const computedHtml = perLineSegs
    .map(segs => segs.map(s => s.c ? `<span class="${s.c}">${esc(s.t)}</span>` : esc(s.t)).join(""))
    .join("");
  const finalHtml = typeof cfg.fullHtml === "string" ? cfg.fullHtml : computedHtml;
  cfg.fullHtml = finalHtml;

  const ensureMeasuredHeight = () => {
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
    if (width) {
      measurement.style.width = width + "px";
    }
    const measurePre = measurement.querySelector(".body-" + cfg.sfx);
    if (measurePre) {
      measurePre.innerHTML = finalHtml;
    }
    const measureCaret = measurement.querySelector(".caret-" + cfg.sfx);
    if (measureCaret) {
      measureCaret.style.display = "none";
    }
    (root.parentElement || document.body).appendChild(measurement);
    const height = measurement.scrollHeight;
    measurement.remove();
    if (height) {
      root.style.minHeight = `${height}px`;
      root.style.height = `${height}px`;
    }
    return height;
  };

  const measuredHeight = ensureMeasuredHeight();
  if (Number.isFinite(measuredHeight) && measuredHeight > 0) {
    window.parent.postMessage({type:"streamlit:resize", height: measuredHeight + 24},"*");
  }

  const renderAll = () => {
    pre.innerHTML = finalHtml;
    if (caret) caret.style.display = "none";
  };

  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (prefersReduced || cfg.speed <= 0) { renderAll(); return; }

  // typing state
  let li = 0, si = 0, ci = 0;
  const done = [];

  const flush = (currentHTML="") => {
    pre.innerHTML = done.join("") + currentHTML;
  };

  const typeNext = () => {
    if (li >= perLineSegs.length) { if (caret) caret.style.display = "none"; return; }
    const segs = perLineSegs[li];

    // build typed HTML up to (si, ci)
    let html = "";
    for (let s=0; s<segs.length; s++) {
      const seg = segs[s];
      const full = seg.t;
      const take = (s < si) ? full : (s === si ? full.slice(0, ci) : "");
      html += seg.c ? `<span class="${seg.c}">${esc(take)}</span>` : esc(take);
    }
    flush(html);

    // advance
    const cur = segs[si];
    ci += 1;
    if (ci > cur.t.length) {
      si += 1; ci = 0;
      if (si >= segs.length) {
        // line finished
        const fullHTML = segs.map(seg => seg.c ? `<span class="${seg.c}">${esc(seg.t)}</span>` : esc(seg.t)).join("");
        done.push(fullHTML);
        li += 1; si = 0; ci = 0;
        setTimeout(typeNext, cfg.pause + (pauses[li-1] || 0));
        return;
      }
    }
    setTimeout(typeNext, cfg.speed);
  };

  if (skip) skip.addEventListener("click", renderAll);
  typeNext();
})();
</script>
"""

def render_prepare_terminal(
    lines: Optional[Iterable[str]] = None,
    speed_type_ms: int = 22,          # ~20–50ms per char
    pause_between_lines_ms: int = 300,
    key: str = "ai_prepare",
    show_caret: bool = True,
    show_skip: bool = True,
) -> None:
    """Render the Prepare Data terminal animation (no legacy-kwargs shim)."""
    data = list(lines) if lines is not None else PREPARE_LINES
    expanded_lines, pauses = _expand_lines(data)
    full_html = _compute_full_html(expanded_lines, _SUFFIX)
    payload = {
        "lines": data,
        "expandedLines": expanded_lines,
        "pauses": pauses,
        "fullHtml": full_html,
        "speed": max(0, int(speed_type_ms)),
        "pause": max(0, int(pause_between_lines_ms)),
        "sfx": _SUFFIX,
        "domId": f"term-{key}",
    }
    html_markup = (
        HTML
        .replace("{STYLE}", STYLE.replace("__SFX__", _SUFFIX))
        .replace("__SFX__", _SUFFIX)
        .replace("__DOMID__", payload["domId"])
        .replace("__CARET__", "inline-block" if show_caret else "none")
        .replace("__SKIP__", f'<button class="skip-{_SUFFIX}" type="button">Skip</button>' if show_skip else "")
        .replace("__PAYLOAD__", json.dumps(payload))
    )
    components_html(html_markup, height=720)
