"""demAI fixed header (HTML) + reliable Streamlit-bridge buttons.

- Full-width, fixed header bar (side-to-side), mobile safe-area aware
- Desktop/tablet: centered stage title
- Phones portrait: buttons inline at the right of the logo; stage hidden
- Phones landscape: ultra-compact single row; progress hidden, title truncated
- Visible header buttons (HTML) trigger stage grid navigation via JS
- Spacer height synced to real header height without feedback loops
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
        return {
            "active_key": None, "index": 0, "total": 0, "stage": None,
            "prev_stage": None, "next_stage": None,
        }

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

    # Base sizes
    base_logo_h = int(logo_height)
    base_vpad = 10

    # Compact scales used on smaller devices
    compact_logo_h = max(36, base_logo_h - 16)
    compact_vpad = 6

    xs_logo_h = max(32, compact_logo_h - 4)
    xs_vpad = max(4, compact_vpad - 2)

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
    # Important: no header height CSS var tied to JS; spacer height is set inline by JS only.
    st.markdown(
        dedent(
            f"""
            <style>
              :root {{
                --demai-logo-h: {base_logo_h}px;
                --demai-header-vpad: {base_vpad}px;
                --demai-btn-min-h: 36px;
                --demai-gap: 12px;
              }}

              header[data-testid="stHeader"] {{ display: none !important; }}

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
                /* Fixed min-height derived from logo + padding only (prevents growth loop) */
                min-height: calc(var(--demai-logo-h) + var(--demai-header-vpad) * 2);
              }}

              @supports (padding: max(0px)) {{
                .demai-header-inner {{
                  padding-left: max(16px, env(safe-area-inset-left));
                  padding-right: max(16px, env(safe-area-inset-right));
                  padding-top: calc(var(--demai-header-vpad) + env(safe-area-inset-top));
                }}
              }}

              .demai-header-spacer {{
                /* JS sets inline height to the measured header height; default fallback below */
                height: calc(var(--demai-logo-h) + var(--demai-header-vpad) * 1);
              }}

              .demai-logo {{ grid-area: logo; display: flex; align-items: center; min-width: 0; }}
              .demai-logo-frame {{
                border: 0; background: transparent; height: var(--demai-logo-h); width: auto;
                display: block; pointer-events: none;
              }}

              .demai-stage-wrap {{ grid-area: stage; min-width: 0; }}
              .demai-stage {{
                display: inline-flex; align-items: center; justify-content:center; gap: 10px;
                flex-wrap: nowrap; color: rgba(226,232,240,0.92);
                min-height: max(var(--demai-logo-h), var(--demai-btn-min-h));
                line-height: 1.2; text-align: center;
              }}
              .demai-stage .progress {{
                font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
                font-size: .72rem; color: rgba(226,232,240,.66);
              }}
              .demai-stage .title {{
                display: inline-flex; align-items: center; gap: 8px;
                font-weight: 700; font-size: .98rem; white-space: nowrap;
                max-width: 32ch; overflow: hidden; text-overflow: ellipsis;
              }}
              .demai-stage .title .icon {{ font-size: 1.05rem; }}

              .demai-actions {{ grid-area: actions; display: inline-flex; gap: 10px; align-items: center; }}
              .demai-btn {{
                position: relative; display: inline-flex; align-items: center; justify-content: center; gap: .55rem;
                min-height: var(--demai-btn-min-h); padding: 0 1.0rem; border-radius: 18px;
                font-weight: 700; font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
                letter-spacing: .05em; text-transform: uppercase;
                text-decoration: none; user-select: none; cursor: pointer; overflow: hidden;
                color: rgba(226,232,240,.95);
                background: linear-gradient(140deg, rgba(10,18,35,.94), rgba(15,23,42,.88));
                border: 1px solid rgba(94,234,212,.45);
                text-shadow: 0 0 12px rgba(94,234,212,.24);
                box-shadow: inset 0 0 0 1px rgba(15,23,42,.76), 0 18px 42px rgba(8,47,73,.5);
                transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease, filter .18s ease, color .18s ease;
                line-height: 1;
              }}
              .demai-btn .label {{ font-size: .7rem; letter-spacing: .14em; }}
              .demai-btn .arrow-icon {{ font-size: .96rem; line-height: 1; }}
              .demai-btn.primary {{
                background: linear-gradient(125deg, rgba(45,212,191,.96), rgba(56,189,248,.92));
                color: rgba(12,25,46,.94);
                border-color: rgba(148,240,223,.8);
                box-shadow: inset 0 0 0 1px rgba(226,232,240,.32), 0 26px 52px rgba(14,165,233,.45);
              }}
              .demai-btn.secondary {{ background: linear-gradient(135deg, rgba(12,20,38,.96), rgba(17,24,39,.9)); }}
              .demai-btn[aria-disabled="true"] {{ opacity: .55; pointer-events: none; }}

              /* ---------- Phones: PORTRAIT ---------- */
              @media (orientation: portrait) and (max-width: 860px) {{
                :root {{
                  --demai-logo-h: {compact_logo_h}px;
                  --demai-header-vpad: {compact_vpad}px;
                }}
                .demai-header-inner {{
                  display: flex; align-items: center; gap: 10px;
                }}
                .demai-stage-wrap {{ display: none; }}
                .demai-actions {{
                  margin-left: auto; display: inline-flex; gap: 8px; align-items: center;
                }}
                .demai-actions .demai-btn {{ min-height: 32px; padding: 0 .72rem; border-radius: 14px; }}
                .demai-btn .label {{ font-size: .64rem; letter-spacing: .12em; }}
                .demai-btn .arrow-icon {{ font-size: .9rem; }}
              }}

              /* ---------- Phones: LANDSCAPE ---------- */
              @media (orientation: landscape) and (max-height: 480px) {{
                :root {{
                  --demai-logo-h: {xs_logo_h}px;
                  --demai-header-vpad: {xs_vpad}px;
                  --demai-gap: 8px;
                }}
                .demai-header-inner {{
                  grid-template-columns: auto minmax(0,1fr) auto;
                  align-items: center; gap: var(--demai-gap);
                }}
                .demai-stage {{ justify-content: center; }}
                .demai-stage .progress {{ display: none; }}
                .demai-stage .title {{ font-size: .9rem; max-width: 24ch; }}
                .demai-actions .demai-btn {{ min-height: 30px; padding: 0 .6rem; border-radius: 12px; }}
                .demai-btn .label {{ font-size: .6rem; letter-spacing: .12em; }}
                .demai-btn .arrow-icon {{ font-size: .86rem; }}
              }}

              /* Very narrow */
              @media (max-width: 380px) {{
                :root {{
                  --demai-logo-h: {xs_logo_h}px;
                  --demai-header-vpad: {xs_vpad}px;
                }}
                .demai-actions {{ gap: 6px; }}
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

    left_disabled = not isinstance(prev_stage, StageMeta)
    right_disabled = not isinstance(next_stage, StageMeta)

    # ---------- Fixed header HTML ----------
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
            <div class="demai-header-spacer" id="demai-header-spacer"></div>
            """
        ),
        unsafe_allow_html=True,
    )

    # ---------- JS bridge + dynamic spacer (one-way; no CSS var writes) ----------
    components.html(
        """
        <script>
          (function initHeader() {
            const doc = (window.parent && window.parent.document) ? window.parent.document : document;
            function byId(id) { return doc.getElementById(id); }

            // Wire visible header buttons to hidden stage-grid nav
            function resolveNavTarget(targetId) {
              const direct = byId(targetId);
              if (direct) return direct;
              const sentinel = doc.querySelector('[data-demai-target="' + targetId + '"]');
              if (!sentinel) return null;
              const container = sentinel.closest('[data-testid="stVerticalBlock"]') || sentinel.parentElement;
              if (!container) return null;
              return container.querySelector('button');
            }
            function bind(vId, hId) {
              function attempt() {
                const v = byId(vId);
                if (!v) return false;
                if (v.dataset.boundTarget === hId) return true;
                const t = resolveNavTarget(hId);
                if (!t) return false;
                v.addEventListener('click', function(e) {
                  if (v.getAttribute('aria-disabled') === 'true') return;
                  e.preventDefault(); e.stopPropagation();
                  const btn = resolveNavTarget(hId);
                  if (btn) btn.click();
                });
                v.dataset.boundTarget = hId;
                return true;
              }
              attempt(); setInterval(attempt, 400);
            }
            bind('demai-btn-prev', 'demai-stage-nav-prev-btn');
            bind('demai-btn-next', 'demai-stage-nav-next-btn');

            // Keep spacer height equal to the actual header height (no feedback to header)
            const header = doc.querySelector('.demai-header-fixed');
            const spacer = byId('demai-header-spacer');
            function syncSpacer() {
              if (!header || !spacer) return;
              const h = Math.ceil(header.getBoundingClientRect().height);
              spacer.style.height = h + 'px';
            }
            if (header && spacer) {
              const ro = new ResizeObserver(syncSpacer);
              ro.observe(header);
              window.addEventListener('orientationchange', () => setTimeout(syncSpacer, 200));
              window.addEventListener('resize', () => setTimeout(syncSpacer, 50));
              syncSpacer();
            }
          })();
        </script>
        """,
        height=0,
    )
