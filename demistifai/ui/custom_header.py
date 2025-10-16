"""demAI fixed header (HTML + real buttons):
- Full-width fixed bar that stays at the top
- In-bar animated logo + vertically centered stage title
- Real Streamlit prev/next buttons overlaid (no reloads, session safe)
- Mobile-safe with perfect vertical alignment
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


# ---------------- Fixed header with perfectly centered title ----------------
def mount_demai_header(logo_height: int = 56, max_inner_width: int = 1200) -> None:
    """
    Render a full-width fixed header (HTML) and overlay real Streamlit buttons (fixed)
    so navigation stays in-session. The stage title stays vertically centered with
    logo and buttons on small screens.
    """

    # If navigation happens via URL (e.g., deep link), sync it once.
    _bootstrap_stage_from_query()

    ctx = _resolve_stage_context()
    stage: Optional[StageMeta] = ctx["stage"]
    prev_stage: Optional[StageMeta] = ctx["prev_stage"]
    next_stage: Optional[StageMeta] = ctx["next_stage"]
    index = int(ctx["index"])
    total = int(ctx["total"])

    # Desktop base sizes
    base_logo_h = int(logo_height)
    base_vpad = 10  # header inner vertical padding
    base_header_h = base_logo_h + (base_vpad * 2)

    # Mobile sizes (ensures perfect alignment on small screens)
    sm_logo_h = max(40, base_logo_h - 12)
    sm_vpad = 8
    sm_header_h = sm_logo_h + (sm_vpad * 2)

    # Build the logo as a self-contained iframe via data URL
    raw_logo_html = demai_logo_html(frame_marker="demai-header")
    logo_data_url = f"data:text/html;base64,{b64encode(raw_logo_html.encode('utf-8')).decode('ascii')}"

    # Stage text HTML
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

    # CSS (header, spacer, and perfectly centered alignment)
    st.markdown(
        dedent(
            f"""
            <style>
              :root {{
                --demai-logo-h: {base_logo_h}px;
                --demai-header-vpad: {base_vpad}px;
                --demai-header-h: {base_header_h}px;   /* spacer height */
                --demai-btn-min-h: 36px;               /* Streamlit button min height */
                --demai-gap: 12px;
              }}

              /* Hide native Streamlit header */
              header[data-testid="stHeader"] {{ display: none !important; }}

              /* Fixed, full-width bar */
              .demai-header-fixed {{
                position: fixed; top: 0; left: 0; right: 0; z-index: 1000;
                background: rgba(15,23,42,0.95);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(94,234,212,0.24);
                box-shadow: 0 8px 18px rgba(8,15,33,0.22);
              }}

              /* Centered inner grid, keeps three zones aligned */
              .demai-header-inner {{
                max-width: {max_inner_width}px; margin: 0 auto;
                padding: var(--demai-header-vpad) 16px;
                display: grid;
                grid-template-columns: auto minmax(0,1fr) auto;
                align-items: center;        /* vertical center of all cells */
                gap: var(--demai-gap);
                min-height: var(--demai-header-h);
              }}

              @supports (padding: max(0px)) {{
                .demai-header-inner {{
                  padding-left: max(16px, env(safe-area-inset-left));
                  padding-right: max(16px, env(safe-area-inset-right));
                }}
              }}

              /* Spacer keeps content below the fixed bar */
              .demai-header-spacer {{ height: var(--demai-header-h); }}

              .demai-logo-frame {{
                border: 0; background: transparent; height: var(--demai-logo-h); width: auto;
                display: block; pointer-events: none;
              }}

              /* Middle stage area: flex centered, min-height matches the tallest control */
              .demai-stage {{
                display: inline-flex;
                align-items: center;             /* vertical center */
                gap: 8px; flex-wrap: wrap;
                color: rgba(226,232,240,0.92);
                min-height: max(var(--demai-logo-h), var(--demai-btn-min-h));
                line-height: 1.2;
              }}
              .demai-stage .progress {{
                font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
                font-size: .74rem; color: rgba(226,232,240,.72);
              }}
              .demai-stage .title {{ font-weight: 700; font-size: 1rem; }}

              /* === Fixed overlay for the REAL Streamlit buttons === */
              .demai-controls-fixed {{
                position: fixed; z-index: 1001;
                top: calc(var(--demai-header-vpad));  /* align with inner top padding */
                right: 16px;
                display: grid; grid-auto-flow: column; gap: 8px;
              }}
              @supports (padding: max(0px)) {{
                .demai-controls-fixed {{ right: max(16px, env(safe-area-inset-right)); }}
              }}
              .demai-controls-fixed [data-testid="stButton"] > button {{
                border-radius: 12px; font-weight: 600; min-height: var(--demai-btn-min-h); padding-inline: 12px;
              }}

              /* ---- Small phones: reduce logo/header height & nudge controls to keep perfect centering ---- */
              @media (max-width: 420px) {{
                :root {{
                  --demai-logo-h: {sm_logo_h}px;
                  --demai-header-vpad: {sm_vpad}px;
                  --demai-header-h: {sm_header_h}px;
                  --demai-gap: 8px;
                }}
                .demai-controls-fixed {{
                  top: {sm_vpad}px;     /* match inner padding */
                  gap: 6px;
                }}
                .demai-stage .title {{ font-size: .95rem; }}
              }}
            </style>
            """
        ),
        unsafe_allow_html=True,
    )

    # Entire fixed bar as HTML (logo + centered stage text)
    st.markdown(
        dedent(
            f"""
            <div class="demai-header-fixed" role="banner" aria-label="demAI top bar">
              <div class="demai-header-inner">
                <div><iframe class="demai-logo-frame" title="demAI animated logo" src="{logo_data_url}" scrolling="no"></iframe></div>
                <div>{stage_html}</div>
                <div><!-- buttons are overlaid via .demai-controls-fixed --></div>
              </div>
            </div>
            <div class="demai-header-spacer"></div>
            """
        ),
        unsafe_allow_html=True,
    )

    # Real Streamlit buttons, fixed-positioned into the bar
    with st.container():
        st.markdown('<div class="demai-controls-fixed">', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="small")

        # Back
        with c1:
            if isinstance(prev_stage, StageMeta):
                if st.button("⬅️", key="demai_header_back", help=f"Back to {prev_stage.title}"):
                    st.session_state["active_stage"] = prev_stage.key
                    st.session_state["stage_scroll_to_top"] = True
                    try:
                        st.query_params["stage"] = prev_stage.key  # new API
                    except Exception:
                        try:
                            st.experimental_set_query_params(stage=prev_stage.key)  # old API
                        except Exception:
                            pass
                    streamlit_rerun()
            else:
                st.write("")  # preserve height

        # Next
        with c2:
            if isinstance(next_stage, StageMeta):
                if st.button("➡️", key="demai_header_next", type="primary", help=f"Forward to {next_stage.title}"):
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
            else:
                st.write("")

        st.markdown("</div>", unsafe_allow_html=True)