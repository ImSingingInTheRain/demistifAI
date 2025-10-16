"""demAI fixed header (HTML) + Streamlit-bridge buttons.
- Full-width fixed bar with logo + perfectly centered stage title
- Visible header buttons are HTML; they trigger hidden Streamlit buttons via JS
- Clicks stay in-session (session_state updated, no reload/new tab)
- Works on desktop and mobile (safe-area aware)
"""

from __future__ import annotations

import html
from base64 import b64encode
from textwrap import dedent
from typing import Optional

import streamlit as st

from demistifai.constants import STAGES, STAGE_INDEX, StageMeta
from demistifai.core.utils import streamlit_rerun
from .animated_logo import demai_logo_html


# ---------- Query param compatibility (new/old Streamlit) ----------
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


# ---------- Stage helpers ----------
def _bootstrap_stage_from_query() -> None:
    """Adopt ?stage=… from URL once (deep links, reloads)."""
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
            "active_key": None,
            "index": 0,
            "total": 0,
            "stage": None,
            "prev_stage": None,
            "next_stage": None,
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


# ---------- Fixed header with JS-bridged buttons ----------
def mount_demai_header(logo_height: int = 56, max_inner_width: int = 1200) -> None:
    """
    Render a full-width, fixed HTML header. Visible header buttons (HTML) trigger
    hidden Streamlit buttons via JS so navigation stays within the same session.
    The stage title is vertically centered with logo & buttons across screen sizes.
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

    # Mobile sizes (keep vertical centering perfect)
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
            f'<div class="demai-stage"><span class="progress">{progress}</span>'
            f'<span class="title">{icon} {title}</span></div>'
        )
    else:
        stage_html = '<div class="demai-stage"><span class="progress">Stage</span><span class="title">Loading…</span></div>'

    # ---------- CSS ----------
    st.markdown(
        dedent(
            f"""
            <style>
              :root {{
                --demai-logo-h: {base_logo_h}px;
                --demai-header-vpad: {base_vpad}px;
                --demai-header-h: {base_header_h}px;
                --demai-btn-min-h: 36px;
                --demai-gap: 12px;
              }}

              header[data-testid="stHeader"] {{ display: none !important; }}

              .demai-header-fixed {{
                position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
                background: rgba(15,23,42,0.95);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(94,234,212,0.24);
                box-shadow: 0 8px 18px rgba(8,15,33,0.22);
              }}
              .demai-header-inner {{
                max-width: {max_inner_width}px; margin: 0 auto;
                padding: var(--demai-header-vpad) 16px;
                display: grid;
                grid-template-columns: auto minmax(0,1fr) auto;
                align-items: center;
                gap: var(--demai-gap);
                min-height: var(--demai-header-h);
              }}
              @supports (padding: max(0px)) {{
                .demai-header-inner {{
                  padding-left: max(16px, env(safe-area-inset-left));
                  padding-right: max(16px, env(safe-area-inset-right));
                }}
              }}
              .demai-header-spacer {{ height: var(--demai-header-h); }}

              .demai-logo-frame {{
                border: 0; background: transparent; height: var(--demai-logo-h); width: auto;
                display: block; pointer-events: none;
              }}

              .demai-stage {{
                display: inline-flex; align-items: center; gap: 8px; flex-wrap: wrap;
                color: rgba(226,232,240,0.92);
                min-height: max(var(--demai-logo-h), var(--demai-btn-min-h));
                line-height: 1.2;
              }}
              .demai-stage .progress {{
                font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
                font-size: .74rem; color: rgba(226,232,240,.72);
              }}
              .demai-stage .title {{ font-weight: 700; font-size: 1rem; }}

              /* Header action buttons (visible HTML) */
              .demai-actions {{ display: inline-flex; gap: 8px; }}
              .demai-btn {{
                display: inline-flex; align-items: center; justify-content: center;
                min-height: var(--demai-btn-min-h); padding: 0 12px; border-radius: 12px;
                font-weight: 700; text-decoration: none; user-select: none; cursor: pointer;
                color: white;
              }}
              .demai-btn.primary {{ background: #ef4444; }}
              .demai-btn.secondary {{ background: rgba(148,163,184,.18); }}
              .demai-btn[aria-disabled="true"] {{ opacity: .35; pointer-events: none; }}

              /* Hide the hidden Streamlit controls area */
              #demai-hidden-controls {{ height: 0; overflow: hidden; }}

              /* Small phones */
              @media (max-width: 420px) {{
                :root {{
                  --demai-logo-h: {sm_logo_h}px;
                  --demai-header-vpad: {sm_vpad}px;
                  --demai-header-h: {sm_header_h}px;
                  --demai-gap: 8px;
                }}
                .demai-stage .title {{ font-size: .95rem; }}
              }}
            </style>
            """
        ),
        unsafe_allow_html=True,
    )

    # ---------- Fixed header HTML ----------
    # Visible header buttons have stable IDs; JS will click the hidden Streamlit buttons.
    left_btn_id = "demai-btn-prev"
    right_btn_id = "demai-btn-next"
    left_disabled = not isinstance(prev_stage, StageMeta)
    right_disabled = not isinstance(next_stage, StageMeta)

    st.markdown(
        dedent(
            f"""
            <div class="demai-header-fixed" role="banner" aria-label="demAI top bar">
              <div class="demai-header-inner">
                <div><iframe class="demai-logo-frame" title="demAI animated logo" src="{logo_data_url}" scrolling="no"></iframe></div>
                <div>{stage_html}</div>
                <div class="demai-actions">
                  <button id="{left_btn_id}" class="demai-btn secondary" {'aria-disabled="true"' if left_disabled else ''}>⬅️</button>
                  <button id="{right_btn_id}" class="demai-btn primary" {'aria-disabled="true"' if right_disabled else ''}>➡️</button>
                </div>
              </div>
            </div>
            <div class="demai-header-spacer"></div>
            """
        ),
        unsafe_allow_html=True,
    )

    # ---------- Hidden Streamlit controls ----------
    # We render real Streamlit buttons off-screen; JS will click them.
    st.markdown('<div id="demai-hidden-controls"></div>', unsafe_allow_html=True)
    prev_clicked = False
    next_clicked = False

    # Place anchor elements so JS can reliably find the right <button> nodes.
    prev_anchor = st.empty()
    with prev_anchor:
        st.markdown('<div id="demai-prev-hook"></div>', unsafe_allow_html=True)

    if isinstance(prev_stage, StageMeta):
        if st.button("internal_prev", key="demai_header_prev_internal"):
            st.session_state["active_stage"] = prev_stage.key
            st.session_state["stage_scroll_to_top"] = True
            try:
                st.query_params["stage"] = prev_stage.key
            except Exception:
                try:
                    st.experimental_set_query_params(stage=prev_stage.key)
                except Exception:
                    pass
            streamlit_rerun()

    next_anchor = st.empty()
    with next_anchor:
        st.markdown('<div id="demai-next-hook"></div>', unsafe_allow_html=True)

    if isinstance(next_stage, StageMeta):
        if st.button("internal_next", key="demai_header_next_internal"):
            st.session_state["active_stage"] = next_stage.key
            st.session_state["stage_scroll_to_top"] = True
            try:
                st.query_params["stage"] = next_stage.key
            except Exception:
                try:
                    st.experimental_set_query_params(stage=next_stage.key)
                except Exception:
                    pass
            streamlit_rerun()

    # ---------- JS bridge: header HTML buttons -> hidden Streamlit buttons ----------
    # Find the first Streamlit <button> after each hook, then call .click() on it.
    st.markdown(
        dedent(
            f"""
            <script>
              (function () {{
                function findSiblingButton(hookId) {{
                  const hook = document.getElementById(hookId);
                  if (!hook) return null;
                  // Walk forward until we see a real <button>
                  let el = hook.parentElement;
                  for (let i = 0; i < 10 && el; i++) {{
                    // search within this subtree
                    const btn = el.querySelector('button');
                    if (btn) return btn;
                    el = el.nextElementSibling;
                  }}
                  return null;
                }}

                function wireClicks() {{
                  const prevBtn = document.getElementById('{left_btn_id}');
                  const nextBtn = document.getElementById('{right_btn_id}');
                  if (!prevBtn || !nextBtn) return false;

                  const hiddenPrev = findSiblingButton('demai-prev-hook');
                  const hiddenNext = findSiblingButton('demai-next-hook');

                  // Attach only if both the visible and hidden buttons exist
                  if (hiddenPrev) {{
                    prevBtn.addEventListener('click', function (e) {{
                      if (prevBtn.getAttribute('aria-disabled') === 'true') return;
                      e.preventDefault(); e.stopPropagation();
                      hiddenPrev.click();
                    }});
                  }}
                  if (hiddenNext) {{
                    nextBtn.addEventListener('click', function (e) {{
                      if (nextBtn.getAttribute('aria-disabled') === 'true') return;
                      e.preventDefault(); e.stopPropagation();
                      hiddenNext.click();
                    }});
                  }}
                  return true;
                }}

                // Retry wiring a few times in case Streamlit rerendered widgets after the header
                let tries = 0;
                const timer = setInterval(function () {{
                  tries += 1;
                  if (wireClicks() || tries > 15) clearInterval(timer);
                }}, 120);
              }})();
            </script>
            """
        ),
        unsafe_allow_html=True,
    )