"""demAI fixed header (HTML) + reliable Streamlit-bridge buttons.

- Full-width, fixed header bar (side-to-side), mobile safe-area aware
- In-bar animated logo + perfectly centered stage title
- Visible header buttons (HTML) trigger the stage grid navigation buttons via JS
- Works on desktop and mobile without losing session state
"""

from __future__ import annotations

import html
from base64 import b64encode
from textwrap import dedent
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from demistifai.constants import STAGES, STAGE_INDEX, StageMeta
from demistifai.core.utils import streamlit_rerun
from .animated_logo import demai_logo_html


# ---------------- Query param compatibility (new/old Streamlit) ----------------
def _qp_get_all(key: str) -> list[str]:
    qp = getattr(st, "query_params", None)
    if qp is not None:
        get_all = getattr(qp, "get_all", None)
        if callable(get_all):
            return get_all(key)
        val = qp.get(key)
        return val if isinstance(val, list) else ([val] if val is not None else [])
    params = st.experimental_get_query_params()  # type: ignore[attr-defined]
    val = params.get(key)
    return val if isinstance(val, list) else ([] if val is None else [val])


# ---------------- Stage helpers ----------------
def _bootstrap_stage_from_query() -> None:
    """If URL has ?stage=… and it's different from session, adopt it."""
    vals = _qp_get_all("stage")
    if not vals:
        return
    key = vals[0]
    if key in STAGE_INDEX and st.session_state.get("active_stage") != key:
        st.session_state["active_stage"] = key
        st.session_state["stage_scroll_to_top"] = True
        streamlit_rerun()


def _resolve_stage_context() -> dict:
    if not STAGES:
        return {"active_key": None, "index": 0, "total": 0, "stage": None, "prev_stage": None, "next_stage": None}

    ss = st.session_state
    default_key = STAGES[0].key
    active_key = ss.get("active_stage", default_key)
    if active_key not in STAGE_INDEX:
        active_key = default_key
        ss["active_stage"] = active_key

    index = STAGE_INDEX[active_key]
    total = len(STAGES)
    stage = STAGES[index]
    prev_stage: Optional[StageMeta] = STAGES[index - 1] if index > 0 else None
    next_stage: Optional[StageMeta] = STAGES[index + 1] if index < total - 1 else None
    return {
        "active_key": active_key,
        "index": index,
        "total": total,
        "stage": stage,
        "prev_stage": prev_stage,
        "next_stage": next_stage,
    }


# ---------------- Fixed header with HTML buttons + stage navigation bridge ----------------
def mount_demai_header(logo_height: int = 56, max_inner_width: int = 1200) -> None:
    """
    Render a full-width fixed header (pure HTML) with visible buttons.
    Those buttons programmatically click the stage grid navigation buttons so navigation
    stays in the same session (no reloads, no new tab).
    """

    _bootstrap_stage_from_query()

    ctx = _resolve_stage_context()
    stage: Optional[StageMeta] = ctx["stage"]
    prev_stage: Optional[StageMeta] = ctx["prev_stage"]
    next_stage: Optional[StageMeta] = ctx["next_stage"]
    index = int(ctx["index"])
    total = int(ctx["total"])

    # Desktop base sizes
    base_logo_h = int(logo_height)
    base_vpad = 10
    base_header_h = base_logo_h + (base_vpad * 2)

    # Mobile sizes (perfect alignment)
    sm_logo_h = max(40, base_logo_h - 12)
    sm_vpad = 8
    sm_header_h = sm_logo_h + (sm_vpad * 2)

    # Logo iframe content
    raw_logo_html = demai_logo_html(frame_marker="demai-header")
    logo_data_url = f"data:text/html;base64,{b64encode(raw_logo_html.encode('utf-8')).decode('ascii')}"

    # Stage text
    if isinstance(stage, StageMeta):
        icon = html.escape(stage.icon)
        title = html.escape(stage.title)
        progress = f"Stage {index + 1} of {total}" if total else f"Stage {index + 1}"
        stage_html = (
            "<div class=\"demai-stage\">"
            f'<span class="progress">{progress}</span>'
            f'<span class="title"><span class="icon" aria-hidden="true">{icon}</span>'
            f'<span class="name">{title}</span></span>'
            "</div>"
        )
    else:
        stage_html = (
            "<div class=\"demai-stage\">"
            '<span class="progress">Stage</span>'
            '<span class="title"><span class="icon" aria-hidden="true">…</span>'
            '<span class="name">Loading…</span></span>'
            "</div>"
        )

    # ---------- CSS ----------
    st.markdown(
        dedent(
            f"""
            <style>
              :root {{
                --demai-logo-h: {base_logo_h}px;
                --demai-header-vpad: {base_vpad}px;
                --demai-header-h: {base_header_h}px;   /* spacer height */
                --demai-btn-min-h: 36px;
                --demai-gap: 12px;
              }}

              header[data-testid="stHeader"] {{ display: none !important; }}

              /* Fixed, full-width header */
              .demai-header-fixed {{
                position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
                background: linear-gradient(90deg, rgba(30,41,59,0.92), rgba(15,23,42,0.96) 45%, rgba(8,11,22,0.98));
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border-bottom: 1px solid rgba(94,234,212,0.28);
                box-shadow: 0 10px 26px rgba(8,15,33,0.28);
              }}

              .demai-header-inner {{
                max-width: {max_inner_width}px; margin: 0 auto;
                padding: var(--demai-header-vpad) 16px;
                display: grid;
                grid-template-columns: auto minmax(0,1fr) auto;
                grid-template-areas: "logo stage actions";
                align-items: center; gap: var(--demai-gap);
                min-height: var(--demai-header-h);
              }}

              .demai-logo {{ grid-area: logo; display: flex; align-items: center; }}
              .demai-stage-wrap {{ grid-area: stage; min-width: 0; }}
              .demai-actions {{ grid-area: actions; display: inline-flex; gap: 10px; align-items: center; }}

              @supports (padding: max(0px)) {{
                .demai-header-inner {{
                  padding-left: max(16px, env(safe-area-inset-left));
                  padding-right: max(16px, env(safe-area-inset-right));
                }}
              }}

              /* Spacer pushes app below fixed bar */
              .demai-header-spacer {{ height: var(--demai-header-h); }}

              .demai-logo-frame {{
                border: 0; background: transparent; height: var(--demai-logo-h); width: auto;
                display: block; pointer-events: none;
              }}

              .demai-stage {{
                display: inline-flex; align-items: center; justify-content: center; gap: 10px;
                flex-wrap: wrap; color: rgba(226,232,240,0.92);
                min-height: max(var(--demai-logo-h), var(--demai-btn-min-h));
                line-height: 1.2; text-align: center;
              }}
              .demai-stage .progress {{
                font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
                font-size: .74rem; color: rgba(226,232,240,.72);
              }}
              .demai-stage .title {{
                display: inline-flex; align-items: center; gap: 8px;
                font-weight: 700; font-size: 1rem;
              }}
              .demai-stage .title .icon {{
                display: inline-flex; align-items: center; justify-content: center;
                font-size: 1.05rem;
              }}
              .demai-stage .title .name {{ display: inline-block; max-width: 20ch; }}

              /* Visible header buttons (HTML) */
              .demai-btn {{
                position: relative; display: inline-flex; align-items: center; justify-content: center; gap: .55rem;
                min-height: var(--demai-btn-min-h); padding: 0 1.1rem; border-radius: 18px;
                font-weight: 700; font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
                letter-spacing: .05em; text-transform: uppercase;
                text-decoration: none; user-select: none; cursor: pointer; overflow: hidden;
                color: rgba(226,232,240,.95);
                background: linear-gradient(140deg, rgba(10,18,35,.94), rgba(15,23,42,.88));
                border: 1px solid rgba(94,234,212,.45);
                text-shadow: 0 0 12px rgba(94,234,212,.24);
                box-shadow: inset 0 0 0 1px rgba(15,23,42,.76), 0 18px 42px rgba(8,47,73,.5);
                transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, filter .18s ease, color .18s ease;
              }}
              .demai-btn::after {{
                content: '';
                position: absolute; inset: 1px; border-radius: 12px;
                background: linear-gradient(140deg, rgba(94,234,212,.18), transparent 60%);
                pointer-events: none;
              }}
              .demai-btn .label {{ font-size: .72rem; letter-spacing: .16em; }}
              .demai-btn .arrow-icon {{ font-size: 1rem; line-height: 1; filter: drop-shadow(0 0 6px rgba(14,165,233,.5)); }}
              .demai-btn.primary {{
                background: linear-gradient(125deg, rgba(45,212,191,.96), rgba(56,189,248,.92));
                color: rgba(12,25,46,.94);
                text-shadow: 0 0 12px rgba(226,232,240,.35);
                border-color: rgba(148,240,223,.8);
                box-shadow: inset 0 0 0 1px rgba(226,232,240,.32), 0 26px 52px rgba(14,165,233,.45);
              }}
              .demai-btn.secondary {{
                background: linear-gradient(135deg, rgba(12,20,38,.96), rgba(17,24,39,.9));
              }}
              .demai-btn:hover {{
                transform: translateY(-1px);
                border-color: rgba(94,234,212,.65);
                box-shadow: inset 0 0 0 1px rgba(15,23,42,.7), 0 28px 58px rgba(8,47,73,.62);
                filter: brightness(1.05) saturate(1.05);
              }}
              .demai-btn.primary:hover {{
                box-shadow: inset 0 0 0 1px rgba(226,232,240,.38), 0 32px 60px rgba(14,165,233,.55);
              }}
              .demai-btn:focus-visible {{
                outline: 2px solid rgba(56,189,248,.7);
                outline-offset: 3px;
              }}
              .demai-btn[aria-disabled="true"] {{
                opacity: .55; pointer-events: none;
                border-color: rgba(148,163,184,.35);
                box-shadow: inset 0 0 0 1px rgba(30,41,59,.7), 0 12px 28px rgba(8,47,73,.35);
                filter: grayscale(12%);
              }}

              /* Small phones: reduce heights paddings to keep perfect centering */
              @media (max-width: 860px) {{
                .demai-header-inner {{
                  grid-template-columns: minmax(0,1fr) auto;
                  grid-template-areas: "stage actions" "logo actions";
                  align-items: center;
                }}
                .demai-logo {{ justify-self: start; }}
                .demai-actions {{ gap: 8px; }}
                .demai-stage {{ justify-content: flex-start; text-align: left; }}
              }}

              @media (max-width: 640px) {{
                .demai-header-inner {{
                  grid-template-columns: auto minmax(0,1fr);
                  grid-template-areas:
                    "logo stage"
                    "actions actions";
                  column-gap: 12px;
                  row-gap: 8px;
                  align-items: center;
                }}
                .demai-logo {{ justify-content: flex-start; }}
                .demai-stage-wrap {{
                  text-align: left;
                  width: 100%;
                }}
                .demai-stage {{
                  justify-content: flex-start;
                  text-align: left;
                  flex-direction: column;
                  align-items: flex-start;
                  gap: 4px;
                  padding: 6px 0;
                }}
                .demai-stage .progress {{ font-size: .68rem; letter-spacing: .12em; opacity: .85; }}
                .demai-stage .title {{ font-size: 1.05rem; gap: 6px; }}
                .demai-stage .title .icon {{ font-size: 1.1rem; }}
                .demai-stage .title .name {{ max-width: none; }}
                .demai-actions {{
                  width: 100%;
                  justify-content: center;
                }}
                .demai-actions .demai-btn {{
                  flex: 1 1 0;
                  max-width: 180px;
                  justify-content: center;
                }}
                .demai-btn .label {{ font-size: .7rem; }}
              }}

              @media (max-width: 420px) {{
                :root {{
                  --demai-logo-h: {sm_logo_h}px;
                  --demai-header-vpad: {sm_vpad}px;
                  --demai-header-h: {sm_header_h}px;
                  --demai-gap: 8px;
                }}
                .demai-stage {{ gap: 3px; }}
                .demai-stage .title {{ font-size: .98rem; gap: 5px; }}
                .demai-stage .title .icon {{ font-size: 1.05rem; }}
                .demai-actions {{ gap: 6px; }}
                .demai-actions .demai-btn {{
                  border-radius: 16px;
                  padding: 0 .9rem;
                }}
                .demai-btn .arrow-icon {{ font-size: .92rem; }}
              }}
            </style>
            """
        ),
        unsafe_allow_html=True,
    )

    # IDs used by visible buttons and the stage-grid nav targets
    prev_visible_id = "demai-btn-prev"
    next_visible_id = "demai-btn-next"
    hidden_prev_id = "demai-stage-nav-prev-btn"
    hidden_next_id = "demai-stage-nav-next-btn"

    # ---------- Fixed header HTML (logo + stage + visible buttons) ----------
    left_disabled = not isinstance(prev_stage, StageMeta)
    right_disabled = not isinstance(next_stage, StageMeta)

    st.markdown(
        dedent(
            f"""
            <div class="demai-header-fixed" role="banner" aria-label="demAI top bar">
              <div class="demai-header-inner">
                <div class="demai-logo">
                  <iframe class="demai-logo-frame" title="demAI animated logo" src="{logo_data_url}" scrolling="no"></iframe>
                </div>
                <div class="demai-stage-wrap">{stage_html}</div>
                <div class="demai-actions" role="group" aria-label="Stage navigation">
                  <button id="{prev_visible_id}" class="demai-btn secondary" {'aria-disabled="true"' if left_disabled else ''}>
                    <span class="arrow-icon" aria-hidden="true">⬅</span>
                    <span class="label">Prev</span>
                  </button>
                  <button id="{next_visible_id}" class="demai-btn primary" {'aria-disabled="true"' if right_disabled else ''}>
                    <span class="label">Next</span>
                    <span class="arrow-icon" aria-hidden="true">➡</span>
                  </button>
                </div>
              </div>
            </div>
            <div class="demai-header-spacer"></div>
            """
        ),
        unsafe_allow_html=True,
    )

    # ---------- JS bridge: visible header buttons -> stage grid buttons ----------
    components.html(
        f"""
        <script>
          (function wireHeaderNav() {{
            const doc = window.parent && window.parent.document ? window.parent.document : document;
            function byId(id) {{ return doc.getElementById(id); }}
            function resolveNavTarget(targetId) {{
              const direct = byId(targetId);
              if (direct) return direct;
              const sentinel = doc.querySelector(`[data-demai-target="${'{'}targetId{'}'}"]`);
              if (!sentinel) return null;
              const container = sentinel.closest('[data-testid="stVerticalBlock"]') || sentinel.parentElement;
              if (!container) return null;
              const button = container.querySelector('button');
              return button || null;
            }}
            function bind(vId, hId) {{
              function attempt() {{
                const v = byId(vId);
                if (!v) return false;
                if (v.dataset.boundTarget === hId) return true;
                const target = resolveNavTarget(hId);
                if (!target) return false;
                v.addEventListener('click', function(e) {{
                  if (v.getAttribute('aria-disabled') === 'true') return;
                  const btn = resolveNavTarget(hId);
                  if (!btn) return;
                  e.preventDefault(); e.stopPropagation();
                  btn.click();
                }});
                v.dataset.boundTarget = hId;
                return true;
              }}
              attempt();
              setInterval(attempt, 400);
            }}
            bind('{prev_visible_id}', '{hidden_prev_id}');
            bind('{next_visible_id}', '{hidden_next_id}');
          }})();
        </script>
        """,
        height=0,
    )
