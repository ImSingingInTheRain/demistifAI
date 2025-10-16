"""demAI fixed header: full-width top bar with logo, stage label and nav.
   - Fixed to viewport (side-to-side)
   - Spacer keeps page content below the bar
   - Works on desktop and mobile (iOS safe-area)
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


# --- Query param compatibility (supports old/new Streamlit) -------------------
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


def _qp_set(**kwargs) -> None:
    qp = getattr(st, "query_params", None)
    if qp is not None:
        for k, v in kwargs.items():
            qp[k] = v
        return
    st.experimental_set_query_params(**kwargs)  # type: ignore[attr-defined]


# --- Stage helpers ------------------------------------------------------------
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


def _set_active_stage(stage_key: str | None) -> None:
    if not stage_key or stage_key not in STAGE_INDEX:
        return
    ss = st.session_state
    changed = ss.get("active_stage") != stage_key
    if changed:
        ss["active_stage"] = stage_key
        ss["stage_scroll_to_top"] = True

    if _qp_get_all("stage") != [stage_key]:
        _qp_set(stage=stage_key)

    if changed:
        streamlit_rerun()


# --- Fixed header -------------------------------------------------------------
def mount_demai_header(logo_height: int = 56, max_inner_width: int = 1200) -> None:
    """
    Render a full-width, fixed header. A spacer div directly after the header
    preserves layout so content never sits under the bar.
    """

    ctx = _resolve_stage_context()
    stage: Optional[StageMeta] = ctx["stage"]
    prev_stage: Optional[StageMeta] = ctx["prev_stage"]
    next_stage: Optional[StageMeta] = ctx["next_stage"]
    index = int(ctx["index"])
    total = int(ctx["total"])

    # Compute a consistent header height (logo + vertical paddings)
    header_h = logo_height + 20  # tweak if you change CSS paddings

    # Global CSS
    st.markdown(
        dedent(
            f"""
            <style>
              :root {{
                --demai-header-h: {header_h}px;
              }}

              /* Hide Streamlit native header */
              header[data-testid="stHeader"] {{ display: none !important; }}

              /* Fixed, full-bleed top bar */
              .demai-header-fixed {{
                position: fixed;
                top: 0; left: 0; right: 0;
                z-index: 1000;
                background: rgba(15,23,42,0.95);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(94,234,212,0.24);
                box-shadow: 0 8px 18px rgba(8,15,33,0.22);
              }}

              /* Center inner content and keep it tight on large screens */
              .demai-header-inner {{
                max-width: {max_inner_width}px;
                margin: 0 auto;
                padding: 10px 16px;
                display: grid;
                grid-template-columns: auto minmax(0,1fr) auto;
                align-items: center;
                gap: 12px;
              }}

              /* Respect notches / safe areas */
              @supports (padding: max(0px)) {{
                .demai-header-inner {{
                  padding-left: max(16px, env(safe-area-inset-left));
                  padding-right: max(16px, env(safe-area-inset-right));
                }}
              }}

              /* Spacer keeps the rest of the app below the fixed bar */
              .demai-header-spacer {{ height: var(--demai-header-h); }}

              .demai-logo-frame {{
                border: none; background: transparent; height: {logo_height}px; width: auto;
                pointer-events: none;
                display: block;
              }}

              .demai-stage {{
                display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
                color: rgba(226,232,240,0.92);
                min-height: {logo_height - 6}px;
              }}
              .demai-stage .progress {{
                font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
                font-size: 0.74rem; color: rgba(226,232,240,0.72);
              }}
              .demai-stage .title {{
                font-weight: 700; font-size: 1rem; line-height: 1.25; white-space: normal;
              }}

              .demai-header-inner [data-testid="stButton"] > button {{
                border-radius: 12px; font-weight: 600; min-height: 36px; padding-inline: 10px;
              }}

              /* Collapse gently on small phones */
              @media (max-width: 420px) {{
                .demai-header-inner {{ gap: 8px; padding-top: 8px; padding-bottom: 8px; }}
                .demai-logo-frame {{ height: {max(40, logo_height - 12)}px; }}
                .demai-stage .title {{ font-size: 0.95rem; }}
              }}
            </style>
            """
        ),
        unsafe_allow_html=True,
    )

    # Render the fixed bar (outside normal flow) + spacer (in flow).
    # Place these as the very first blocks in the app.
    raw_logo_html = demai_logo_html(frame_marker="demai-header")
    data_url = f"data:text/html;base64,{b64encode(raw_logo_html.encode('utf-8')).decode('ascii')}"

    # Fixed bar
    st.markdown('<div class="demai-header-fixed"><div class="demai-header-inner">', unsafe_allow_html=True)

    # Left: logo
    left, middle, right = st.columns([1.1, 2.8, 1.2], gap="small")
    with left:
        st.markdown(
            f'<iframe class="demai-logo-frame" title="demAI animated logo" src="{data_url}" scrolling="no"></iframe>',
            unsafe_allow_html=True,
        )

    # Middle: stage label
    with middle:
        if isinstance(stage, StageMeta):
            icon = html.escape(stage.icon)
            title = html.escape(stage.title)
            progress = f"Stage {index + 1} of {total}" if total else f"Stage {index + 1}"
            st.markdown(
                f'<div class="demai-stage"><span class="progress">{progress}</span>'
                f'<span class="title">{icon} {title}</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="demai-stage"><span class="progress">Stage</span>'
                '<span class="title">Loading…</span></div>',
                unsafe_allow_html=True,
            )

    # Right: nav buttons
    with right:
        c1, c2 = st.columns(2, gap="small")
        with c1:
            if isinstance(prev_stage, StageMeta):
                if st.button("⬅️", key="demai_header_back", use_container_width=True, help=f"Back to {prev_stage.title}"):
                    _set_active_stage(prev_stage.key)
            else:
                st.write("")  # reserve height
        with c2:
            if isinstance(next_stage, StageMeta):
                if st.button(
                    "➡️",
                    key="demai_header_next",
                    use_container_width=True,
                    type="primary",
                    help=f"Forward to {next_stage.title}",
                ):
                    _set_active_stage(next_stage.key)
            else:
                st.write("")

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Spacer directly after the fixed bar so the app content starts below it
    st.markdown('<div class="demai-header-spacer"></div>', unsafe_allow_html=True)