"""Use terminal animation: per-char typing, colored tokens, progressive status."""

from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple
import html
import json
import re
from streamlit.components.v1 import html as components_html

_SUFFIX = "ai_use"

# --- Content for the USE stage (short + progressive) ---
USE_LINES: List[str] = [
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
_USE_LINES = USE_LINES

# --- Minimal style (same system as your other cmd animations) ---
STYLE = """
<style>
  .term-__SFX__{background:#0d1117;color:#ffffff;font-family:'Fira Code',ui-monospace,monospace;border-radius:12px;padding:1rem;position:relative}
  .term-__SFX__::before{content:'●  ●  ●';position:absolute;top:8px;left:12px;color:#ef4444cc;letter-spacing:6px;font-size:.9rem}
  .body-__SFX__{white-space:pre-wrap;line-height:1.6;font-size:.96rem;margin-top:.8rem}
  .caret-__SFX__{margin-left:2px;display:inline-block;width:6px;height:1rem;background:#22d3ee;vertical-align:-.18rem;animation:blink-__SFX__ .85s steps(1,end) infinite}
  @keyframes blink-__SFX__{50%{opacity:0}}
  .cmd-__SFX__{color:#93c5fd}
  .err-__SFX__{color:#fb7185;font-weight:700}
  .wrap-__SFX__{opacity:0;transform:translateY(6px);animation:fadein-__SFX__ .4s ease forwards}
  @keyframes fadein-__SFX__{to{opacity:1;transform:none}}
  .skip-__SFX__{position:absolute;top:8px;right:8px;background:#111827;color:#e5e7eb;border:1px solid #374151;border-radius:6px;padding:.2rem .5rem;font-size:.8rem;cursor:pointer}
  .skip-__SFX__:hover{background:#1f2937}
  @media (prefers-reduced-motion: reduce){.caret-__SFX__{animation:none}.wrap-__SFX__{animation:none;opacity:1;transform:none}}
  @media (max-width: 600px){
    .wrap-__SFX__{margin:0;padding:0}
    .term-__SFX__{width:100%;max-width:100vw;min-height:100vh;margin:0;border-radius:0;padding:clamp(1.1rem,6vw,1.5rem);box-sizing:border-box;display:flex;flex-direction:column;gap:1rem}
    .term-__SFX__::before{top:12px;left:50%;transform:translateX(-50%)}
    .body-__SFX__{font-size:1.05rem;line-height:1.72;width:100%;overflow-wrap:anywhere;word-break:break-word;overflow-x:hidden}
    .caret-__SFX__{align-self:flex-start;margin-top:-.2rem;flex-shrink:0}
    .skip-__SFX__{position:static;width:100%;margin:.75rem 0 0;align-self:stretch;text-align:center;padding:.6rem 1rem;font-size:.9rem}
  }
</style>
"""


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
    if re.match(r"^ERROR\b", trimmed, flags=re.IGNORECASE):
        return [(line, f"err-{suffix}")]
    return [(line, None)]


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
  <div id="__DOMID__" class="term-__SFX__" role="region" aria-label="Use terminal animation">
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

  const normaliseSegments = (segments) =>
    segments.map((line) =>
      Array.isArray(line)
        ? line.map((seg) => ({
            t: typeof seg.t === "string" ? seg.t : "",
            c: typeof seg.c === "string" && seg.c ? seg.c : null,
          }))
        : []
    );

  const splitSegments = (line) => {
    const trimmed = line.trim();
    if (line.startsWith("$ ")) return [{ t: line, c: "cmd-" + cfg.sfx }];
    if (/^ERROR\\b/i.test(trimmed)) return [{ t: line, c: "err-" + cfg.sfx }];
    return [{ t: line, c: null }];
  };

  const rawLines = (cfg.expandedLines || cfg.lines || []).map((item) => {
    const value = item == null ? "" : String(item);
    return value.endsWith("\n") ? value : value + "\n";
  });
  const pauses = Array.isArray(cfg.pauses)
    ? cfg.pauses.map((p) => Number(p) || 0)
    : new Array(rawLines.length).fill(0);

  const perLineSegs = Array.isArray(cfg.segments)
    ? normaliseSegments(cfg.segments)
    : rawLines.map(splitSegments);
  const perLineSegmentHtml = perLineSegs.map((segs) =>
    segs.map((seg) => (seg.c ? `<span class="${seg.c}">${esc(seg.t)}</span>` : esc(seg.t)))
  );
  const perLineHtml = perLineSegmentHtml.map((parts) => parts.join(""));
  const perLineRaw = perLineSegs.map((segs) => segs.map((s) => s.t).join(""));
  const computedHtml = perLineHtml.join("");
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

  let cancelled = false;

  const doneHtmlParts = [];
  let activeNode = null;
  let lineIndex = 0;
  let segmentIndex = 0;
  let charIndex = 0;

  const syncDoneHtml = () => {
    pre.innerHTML = doneHtmlParts.join("");
  };

  const ensureActiveNode = () => {
    if (activeNode) {
      return activeNode;
    }
    const segments = perLineSegs[lineIndex] || [];
    const current = segments[segmentIndex];
    if (!current) {
      return null;
    }
    if (current.c) {
      const span = document.createElement("span");
      span.className = current.c;
      span.textContent = "";
      pre.appendChild(span);
      activeNode = span;
    } else {
      activeNode = document.createTextNode("");
      pre.appendChild(activeNode);
    }
    return activeNode;
  };

  const commitActiveSegment = () => {
    if (!activeNode) {
      return;
    }
    if (activeNode.parentNode === pre) {
      pre.removeChild(activeNode);
    }
    const segmentHtml = (perLineSegmentHtml[lineIndex] || [])[segmentIndex] || "";
    doneHtmlParts.push(segmentHtml);
    activeNode = null;
    syncDoneHtml();
  };

  const showFinal = () => {
    cancelled = true;
    doneHtmlParts.length = 0;
    activeNode = null;
    pre.innerHTML = finalHtml;
    if (caret) caret.style.display = "none";
  };

  if (skip) skip.addEventListener("click", showFinal);

  const baseSpeed = Math.max(0, Number(cfg.speed) || 0);
  const basePause = Math.max(0, Number(cfg.pause) || 0);
  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (prefersReduced || baseSpeed <= 0) { showFinal(); return; }

  syncDoneHtml();

  const typeNext = () => {
    if (cancelled) { return; }
    if (lineIndex >= perLineSegs.length) {
      syncDoneHtml();
      if (caret) caret.style.display = "none";
      return;
    }

    const segments = perLineSegs[lineIndex] || [];
    const current = segments[segmentIndex];

    if (!current) {
      segmentIndex = 0;
      lineIndex += 1;
      const delay = basePause + (pauses[lineIndex - 1] || 0);
      setTimeout(typeNext, delay);
      return;
    }

    const target = current.t;
    if (charIndex < target.length) {
      const node = ensureActiveNode();
      if (node) {
        const char = target.charAt(charIndex);
        node.textContent = (node.textContent || "") + char;
      }
      charIndex += 1;
      setTimeout(typeNext, baseSpeed);
      return;
    }

    commitActiveSegment();
    segmentIndex += 1;
    charIndex = 0;

    if (segmentIndex >= segments.length) {
      segmentIndex = 0;
      lineIndex += 1;
      const delay = basePause + (pauses[lineIndex - 1] || 0);
      setTimeout(typeNext, delay);
      return;
    }

    setTimeout(typeNext, baseSpeed);
  };

  requestAnimationFrame(typeNext);
})();
</script>
"""

def render_use_terminal(
    lines: Optional[Iterable[str]] = None,
    speed_type_ms: int = 22,          # ~20–50ms per char
    pause_between_lines_ms: int = 300,
    key: str = "ai_use",
    show_caret: bool = True,
    show_skip: bool = True,
) -> None:
    """Render the Use stage terminal animation (no legacy-kwargs shim)."""
    data = list(lines) if lines is not None else USE_LINES
    expanded_lines, pauses = _expand_lines(data)
    segments = [_split_segments(line, _SUFFIX) for line in expanded_lines]
    full_html = "".join(_segments_to_html(parts) for parts in segments)
    serializable_segments = [
        [{"t": text, "c": css} for text, css in parts]
        for parts in segments
    ]
    payload = {
        "lines": data,
        "expandedLines": expanded_lines,
        "pauses": pauses,
        "fullHtml": full_html,
        "speed": max(0, int(speed_type_ms)),
        "pause": max(0, int(pause_between_lines_ms)),
        "sfx": _SUFFIX,
        "domId": f"term-{key}",
        "segments": serializable_segments,
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
