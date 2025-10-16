"""Overview terminal animation: per-char typing, colored tokens, progressive status."""

from __future__ import annotations
from typing import Iterable, List, Optional
import json
from streamlit.components.v1 import html as components_html

_OV_SUFFIX = "ai_overview"

# --- Content for the OVERVIEW page (short + progressive) ---
OVERVIEW_LINES: List[str] = [
    "> Overview: inside the demAI machine\n",
    "$ status demAI.machine\n",
    "demAI.machine  ▸ idle  •  0 pending jobs  •  ready\n",
    "$ map stages\n",
    "- [ ] Data intake\n",
    "- [ ] Label & split\n",
    "- [ ] Train & validate\n",
    "- [ ] Evaluate\n",
    "- [ ] Inference\n",
    "- [ ] Oversight & logs\n",
    "> Tip: You can jump to any stage from the Control Room.\n",
    "$ scan components --quick\n",
    "UI (client)……………… OK ✓\n",
    "Bridge (API)…………… OK ✓\n",
    "Model runner…………… OK ✓\n",
    "Storage (datasets)…… OK ✓\n",
    "Progress 0%   [████████████████████] 100%\n",
    "\nHINT: Toggle Nerd Mode anytime for deeper details, metrics, and artifacts.\n",
    "$ diagram pipeline\n",
    "[user] → [UI] → [Bridge] → [Model] → [Outputs]\n",
    "                 ↑              ↓\n",
    "            [Data/Labels]   [Logs/Oversight]\n",
    "$ open overview\n",
]

# Backwards-compatibility alias for callers that might expect this name pattern
_OVERVIEW_LINES = OVERVIEW_LINES

# --- Minimal style (same system as the welcome template) ---
OV_STYLE = """
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

OV_HTML = """
{STYLE}
<div class="wrap-__SFX__">
  <div id="__DOMID__" class="term-__SFX__" role="region" aria-label="demAI overview terminal animation">
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
  const postResize = () => window.parent.postMessage({type:"streamlit:resize", height: root.scrollHeight + 24},"*");

  // per-line coloring (typed progressively)
  const splitSegments = (line) => {
    const trimmed = line.trim();
    if (line.startsWith("$ ")) return [{t: line, c: "cmd-" + cfg.sfx}];
    if (/^HINT:/i.test(trimmed)) return [{t: line, c: "hint-" + cfg.sfx}];
    if (/^ERROR\\b/i.test(trimmed)) return [{t: line, c: "err-" + cfg.sfx}];
    if (/^>\\s/.test(line)) return [{t: line, c: "pill-" + cfg.sfx}];
    if (/\\[█+\\]/.test(line)) return [{t: line, c: "prog-" + cfg.sfx}];

    // token emphasize: stages keywords (simple)
    const parts = [];
    const re = /(Data intake|Label & split|Train & validate|Evaluate|Inference|Oversight & logs)/gi;
    let idx = 0, m;
    while ((m = re.exec(line)) !== null) {
      if (m.index > idx) parts.push({t: line.slice(idx, m.index), c: null});
      parts.push({t: m[0], c: "act-" + cfg.sfx});
      idx = m.index + m[0].length;
    }
    if (idx < line.length) parts.push({t: line.slice(idx), c: null});
    return parts.length ? parts : [{t: line, c: null}];
  };

  // expand progress into frames + extra pause after ✓ lines
  const expand = (arr) => {
    const out = [], pauses = [];
    for (const raw of arr) {
      const line = raw.endsWith("\\n") ? raw : raw + "\\n";
      if (/^Progress\\s+0%/.test(line)) {
        ["Progress 0%    [█...................] 0%\\n",
         "Progress 40%   [████████............] 40%\\n",
         "Progress 75%   [██████████████......] 75%\\n",
         "Progress 100%  [████████████████████] 100%\\n"].forEach(f => { out.push(f); pauses.push(220); });
        continue;
      }
      out.push(line);
      pauses.push(/✓\\s*$/.test(line) ? 380 : 0);
    }
    return {lines: out, pauses};
  };

  const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const {lines, pauses} = expand(cfg.lines.map(x => String(x)));
  const perLineSegs = lines.map(splitSegments);

  const renderAll = () => {
    const html = perLineSegs.map(segs => segs.map(s => s.c ? `<span class="${s.c}">${esc(s.t)}</span>` : esc(s.t)).join("")).join("");
    pre.innerHTML = html;
    if (caret) caret.style.display = "none";
    postResize();
  };

  if (prefersReduced || cfg.speed <= 0) { renderAll(); return; }

  // typing state
  let li = 0, si = 0, ci = 0;
  const done = [];

  const flush = (currentHTML="") => {
    pre.innerHTML = done.join("") + currentHTML;
    postResize();
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

def render_overview_terminal(
    lines: Optional[Iterable[str]] = None,
    speed_type_ms: int = 22,
    pause_between_lines_ms: int = 300,
    key: str = "ai_overview",
    show_caret: bool = True,
    show_skip: bool = True,
) -> None:
    """Render the demAI Overview terminal animation (no legacy-kwargs shim)."""
    data = list(lines) if lines is not None else OVERVIEW_LINES
    payload = {
        "lines": data,
        "speed": max(0, int(speed_type_ms)),
        "pause": max(0, int(pause_between_lines_ms)),
        "sfx": _OV_SUFFIX,
        "domId": f"term-{key}",
    }
    html_markup = (
        OV_HTML
        .replace("{STYLE}", OV_STYLE.replace("__SFX__", _OV_SUFFIX))
        .replace("__SFX__", _OV_SUFFIX)
        .replace("__DOMID__", payload["domId"])
        .replace("__CARET__", "inline-block" if show_caret else "none")
        .replace("__SKIP__", f'<button class="skip-{_OV_SUFFIX}" type="button">Skip</button>' if show_skip else "")
        .replace("__PAYLOAD__", json.dumps(payload))
    )
    components_html(html_markup, height=720)