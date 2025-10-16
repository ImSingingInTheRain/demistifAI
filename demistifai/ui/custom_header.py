"""demAI fixed header (HTML-based): full-width, side-to-side, with in-bar logo, stage and nav.
   Uses links for nav and syncs session_state from query params.
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


# ---------------- Fixed header (pure HTML inside bar) ----------------
def mount_demai_header(logo_height: int = 56, max_inner_width: int = 1200) -> None:
    """Render a full-width fixed header with an in-flow spacer below it."""

    # adopt URL stage if present (needed because we use links for nav)
    _bootstrap_stage_from_query()

    ctx = _resolve_stage_context()
    stage: Optional[StageMeta] = ctx["stage"]
    prev_stage: Optional[StageMeta] = ctx["prev_stage"]
    next_stage: Optional[StageMeta] = ctx["next_stage"]
    index = int(ctx["index"])
    total = int(ctx["total"])

    header_h = logo_height + 20  # logo + vertical paddings

    # Build logo as data URL so the iframe is self-contained
    raw_logo_html = demai_logo_html(frame_marker="demai-header")
    logo_data_url = f"data:text/html;base64,{b64encode(raw_logo_html.encode('utf-8')).decode('ascii')}"

    # Compose HTML for buttons (as links that reload with ?stage=…)
    def _btn(label: str, key: Optional[str], kind: str) -> str:
        if key:
            return f'<a class="demai-btn {kind}" href="?stage={html.escape(key)}" role="button" aria-label="{label}">{label}</a>'
        # disabled placeholder keeps layout
        return f'<span class="demai-btn {kind} disabled" aria-disabled="true">{label}</span>'

    left_btn = _btn("⬅️", prev_stage.key if isinstance(prev_stage, StageMeta) else None, "secondary")
    right_btn = _btn("➡️", next_stage.key if isinstance(next_stage, StageMeta) else None, "primary")

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

    # CSS (no Streamlit widgets inside the fixed bar)
    st.markdown(
        dedent(
            f"""
            <style>
              :root {{ --demai-header-h: {header_h}px; }}

              /* Hide native Streamlit header */
              header[data-testid="stHeader"] {{ display: none !important; }}

              /* Fixed, full-bleed bar */
              .demai-header-fixed {{
                position: fixed; top: 0; left: 0; right: 0;
                z-index: 1000;
                background: rgba(15,23,42,0.95);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(94,234,212,0.24);
                box-shadow: 0 8px 18px rgba(8,15,33,0.22);
              }}

              .demai-header-inner {{
                max-width: {max_inner_width}px;
                margin: 0 auto;
                padding: 10px 16px;
                display: grid;
                grid-template-columns: auto minmax(0,1fr) auto;
                align-items: center;
                gap: 12px;
              }}
              @supports (padding: max(0px)) {{
                .demai-header-inner {{
                  padding-left: max(16px, env(safe-area-inset-left));
                  padding-right: max(16px, env(safe-area-inset-right));
                }}
              }}

              /* Spacer keeps content below the bar */
              .demai-header-spacer {{ height: var(--demai-header-h); }}

              .demai-logo-frame {{
                border: 0; background: transparent; height: {logo_height}px; width: auto; display: block; pointer-events: none;
              }}

              .demai-stage {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; color: rgba(226,232,240,0.92); }}
              .demai-stage .progress {{ font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
                                        font-size: .74rem; color: rgba(226,232,240,.72); }}
              .demai-stage .title {{ font-weight: 700; font-size: 1rem; line-height: 1.25; }}

              .demai-actions {{ display: inline-flex; gap: 8px; }}

              .demai-btn {{
                display: inline-flex; align-items: center; justify-content: center;
                min-height: 36px; padding: 0 12px; border-radius: 12px; font-weight: 700;
                text-decoration: none; user-select: none;
                transition: transform .04s ease;
              }}
              .demai-btn.primary {{ background: #ef4444; color: white; }}
              .demai-btn.secondary {{ background: rgba(148,163,184,.18); color: white; }}
              .demai-btn.disabled {{ opacity: .35; pointer-events: none; }}
              .demai-btn:active {{ transform: translateY(1px); }}

              @media (max-width: 420px) {{
                .demai-header-inner {{ gap: 8px; padding-top: 8px; padding-bottom: 8px; }}
                .demai-logo-frame {{ height: {max(40, logo_height - 12)}px; }}
                .demai-stage .title {{ font-size: .95rem; }}
              }}
            </style>
            """
        ),
        unsafe_allow_html=True,
    )

    # Entire fixed bar as HTML (everything stays inside it)
    st.markdown(
        dedent(
            f"""
            <div class="demai-header-fixed" role="banner" aria-label="demAI top bar">
              <div class="demai-header-inner">
                <div><iframe class="demai-logo-frame" title="demAI animated logo" src="{logo_data_url}" scrolling="no"></iframe></div>
                <div>{stage_html}</div>
                <div class="demai-actions">{left_btn}{right_btn}</div>
              </div>
            </div>
            <div class="demai-header-spacer"></div>
            """
        ),
        unsafe_allow_html=True,
    )