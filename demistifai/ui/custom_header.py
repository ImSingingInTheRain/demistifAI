"""Minimal custom header: hides Streamlit's default header and shows the animated demAI logo with stage nav.
   Sticky-in-flow version with Streamlit query-param compatibility (no :has(), no fixed positioning)."""

from __future__ import annotations

import html
from base64 import b64encode
from textwrap import dedent
from typing import Optional

import streamlit as st

from demistifai.constants import STAGES, STAGE_INDEX, StageMeta
from demistifai.core.utils import streamlit_rerun
from .animated_logo import demai_logo_html


# --- Query param compatibility layer -----------------------------------------
def _qp_get_all(key: str) -> list[str]:
    """Return list of values for a query param across Streamlit versions."""
    qp = getattr(st, "query_params", None)
    if qp is not None:
        # New API (Streamlit ≥ 1.33)
        get_all = getattr(qp, "get_all", None)
        if callable(get_all):
            return get_all(key)
        # Fallback: mapping semantics
        val = qp.get(key)
        return val if isinstance(val, list) else ([val] if val is not None else [])
    # Old API
    params = st.experimental_get_query_params()  # type: ignore[attr-defined]
    val = params.get(key)
    return val if isinstance(val, list) else ([] if val is None else [val])


def _qp_set(**kwargs) -> None:
    """Set query params across Streamlit versions."""
    qp = getattr(st, "query_params", None)
    if qp is not None:
        # New API supports item assignment
        for k, v in kwargs.items():
            qp[k] = v
        return
    # Old API
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
    current = ss.get("active_stage")
    changed = current != stage_key
    if changed:
        ss["active_stage"] = stage_key
        ss["stage_scroll_to_top"] = True

    if _qp_get_all("stage") != [stage_key]:
        _qp_set(stage=stage_key)

    if changed:
        streamlit_rerun()


# --- Header -------------------------------------------------------------------
def mount_demai_header(logo_height: int = 56) -> None:
    """Sticky header that remains within the page flow for robust desktop/mobile rendering."""

    ctx = _resolve_stage_context()
    stage: Optional[StageMeta] = ctx["stage"]
    prev_stage: Optional[StageMeta] = ctx["prev_stage"]
    next_stage: Optional[StageMeta] = ctx["next_stage"]
    index = int(ctx["index"])
    total = int(ctx["total"])

    # Global CSS: hide Streamlit native header; style sticky header block.
    st.markdown(
        dedent(
            f"""
            <style>
              /* Hide Streamlit default header */
              header[data-testid="stHeader"] {{ display: none !important; }}

              /* Sticky header block – in flow (no position:fixed) */
              .demai-header {{
                position: sticky;
                top: 0;
                z-index: 1000;
                background: rgba(15,23,42,0.92);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(94,234,212,0.24);
                box-shadow: 0 8px 18px rgba(8,15,33,0.22);
                padding: 10px 16px;
              }}

              .demai-header .demai-row {{
                display: grid;
                grid-template-columns: auto minmax(0,1fr) auto;
                align-items: center;
                gap: 12px;
              }}

              .demai-logo-frame {{
                border: none; background: transparent; height: {logo_height}px; width: auto;
                pointer-events: none;
              }}

              .demai-stage {{
                display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
                color: rgba(226,232,240,0.92);
              }}
              .demai-stage .progress {{
                font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
                font-size: 0.74rem; color: rgba(226,232,240,0.72);
              }}
              .demai-stage .title {{
                font-weight: 700; font-size: 1rem; white-space: normal; line-height: 1.25;
              }}

              .demai-header [data-testid="stButton"] > button {{
                border-radius: 12px; font-weight: 600; min-height: 36px; padding-inline: 10px;
              }}

              @media (max-width: 420px) {{
                .demai-header {{ padding: 8px 12px; }}
                .demai-row {{ gap: 8px; }}
                .demai-logo-frame {{ height: {max(40, logo_height - 12)}px; }}
                .demai-stage .title {{ font-size: 0.95rem; }}
              }}

              @supports (padding: max(0px)) {{
                .demai-header {{
                  padding-left: max(16px, env(safe-area-inset-left));
                  padding-right: max(16px, env(safe-area-inset-right));
                }}
              }}
            </style>
            """
        ),
        unsafe_allow_html=True,
    )

    # Build sticky header as first container (in normal flow).
    with st.container():
        st.markdown('<div class="demai-header"><div class="demai-row">', unsafe_allow_html=True)

        # Left: animated logo
        raw_logo_html = demai_logo_html(frame_marker="demai-header")
        data_url = f"data:text/html;base64,{b64encode(raw_logo_html.encode('utf-8')).decode('ascii')}"
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

        # Right: navigation buttons
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