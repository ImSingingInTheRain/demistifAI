"""Minimal custom header: hides Streamlit header and shows the animated demAI logo at top-left."""

from __future__ import annotations

import html
from base64 import b64encode
from textwrap import dedent
from typing import Optional

import streamlit as st

from demistifai.constants import STAGES, STAGE_INDEX, StageMeta
from demistifai.core.utils import streamlit_rerun

from .animated_logo import demai_logo_html


def _resolve_stage_context() -> dict[str, Optional[StageMeta] | int | str | None]:
    """Return navigation context for the current stage."""

    if not STAGES:
        return {
            "active_key": None,
            "index": 0,
            "total": 0,
            "stage": None,
            "prev_stage": None,
            "next_stage": None,
        }

    session_state = st.session_state
    default_key = STAGES[0].key
    active_key = session_state.get("active_stage", default_key)
    if active_key not in STAGE_INDEX:
        active_key = default_key
        session_state["active_stage"] = active_key

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
    """Synchronise stage navigation state and trigger a rerun when needed."""

    if not stage_key or stage_key not in STAGE_INDEX:
        return

    session_state = st.session_state
    current = session_state.get("active_stage")
    stage_changed = current != stage_key

    if stage_changed:
        session_state["active_stage"] = stage_key
        session_state["stage_scroll_to_top"] = True

    if st.query_params.get_all("stage") != [stage_key]:
        st.query_params["stage"] = stage_key

    if stage_changed:
        streamlit_rerun()


def mount_demai_header(logo_height: int = 56) -> None:
    """
    Hide Streamlit's default header and mount a fixed top bar with navigation.

    Args:
        logo_height: Pixel height of the logo area inside the header.
    """

    stage_context = _resolve_stage_context()
    stage = stage_context["stage"]
    prev_stage = stage_context["prev_stage"]
    next_stage = stage_context["next_stage"]
    index = int(stage_context["index"])
    total = int(stage_context["total"])

    # CSS: hide Streamlit header + add top padding so content doesn't sit under our bar
    st.markdown(
        dedent(
            f"""
            <style>
              /* Hide default Streamlit header */
              header, [data-testid="stHeader"] {{
                display: none !important;
                visibility: hidden !important;
              }}

              /* Ensure main content has room under our fixed header */
              [data-testid="stAppViewContainer"] .main .block-container {{
                padding-top: {logo_height + 28}px !important; /* header height + gap */
              }}

              /* Fixed header bar */
              .demai-header {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1000;

                display: flex;
                align-items: center;
                gap: 18px;

                min-height: {logo_height + 10}px;
                padding: 6px 18px;
                box-sizing: border-box;

                background: rgba(15, 23, 42, 0.92);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(94, 234, 212, 0.24);
                box-shadow: 0 12px 24px rgba(8, 15, 33, 0.28);
              }}

              .demai-header__logo-frame {{
                border: none;
                background: transparent;
                width: auto;
                height: {logo_height}px;
                pointer-events: none; /* prevent focusing the iframe */
              }}

              .demai-header__nav {{
                flex: 1 1 auto;
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 24px;
                min-width: 0;
              }}

              .demai-header__nav [data-testid="stHorizontalBlock"] {{
                width: 100%;
              }}

              .demai-header__nav [data-testid="column"] {{
                padding: 0 !important;
              }}

              .demai-header__stage {{
                display: flex;
                flex-direction: column;
                justify-content: center;
                color: rgba(226, 232, 240, 0.9);
                font-size: 0.86rem;
                min-width: 0;
              }}

              .demai-header__stage-progress {{
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-size: 0.68rem;
                color: rgba(148, 163, 184, 0.92);
                margin-bottom: 2px;
              }}

              .demai-header__stage-name {{
                font-weight: 600;
                font-size: 0.98rem;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
              }}

              .demai-header__controls {{
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 10px;
                width: 100%;
              }}

              .demai-header__controls [data-testid="stButton"] {{
                width: auto;
              }}

              .demai-header__controls [data-testid="stButton"] > button {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                height: 32px;
                background: linear-gradient(135deg, rgba(45, 212, 191, 0.18), rgba(59, 130, 246, 0.12));
                border: 1px solid rgba(94, 234, 212, 0.45);
                color: rgba(226, 232, 240, 0.96);
                border-radius: 999px;
                font-weight: 600;
                font-size: 0.78rem;
                padding: 0 18px;
                line-height: 1;
                box-shadow: none;
                transition: background 120ms ease, border-color 120ms ease, transform 120ms ease;
              }}

              .demai-header__controls [data-testid="stButton"] > button:hover {{
                background: linear-gradient(135deg, rgba(45, 212, 191, 0.3), rgba(59, 130, 246, 0.2));
                border-color: rgba(94, 234, 212, 0.7);
                transform: translateY(-1px);
              }}

              .demai-header__controls [data-testid="stButton"] > button:active {{
                transform: translateY(0);
              }}

              .demai-header__control-placeholder {{
                display: inline-flex;
                min-width: 96px;
                height: 32px;
              }}

              @media (max-width: 960px) {{
                .demai-header {{
                  gap: 12px;
                  padding: 6px 14px;
                }}

                .demai-header__controls [data-testid="stButton"] > button {{
                  padding: 0 14px;
                }}
              }}
            </style>
            """
        ),
        unsafe_allow_html=True,
    )

    # Fixed header with animated logo embedded via iframe ``src``.
    # ``srcdoc`` proved brittle with complex markup because Streamlit re-renders
    # the Markdown block frequently, which occasionally surfaced the raw text
    # instead of the rendered iframe. Using a ``data:`` URL keeps the markup
    # encapsulated without relying on inline escaping.
    raw_logo_html = demai_logo_html(frame_marker="demai-header")
    encoded_logo = b64encode(raw_logo_html.encode("utf-8")).decode("ascii")
    data_url = f"data:text/html;base64,{encoded_logo}"

    with st.container():
        st.markdown('<div class="demai-header" data-testid="demai-header">', unsafe_allow_html=True)

        logo_col, nav_col = st.columns([1, 3.2], gap="large")
        with logo_col:
            st.markdown(
                dedent(
                    f"""
                    <iframe
                      class="demai-header__logo-frame"
                      title="demAI animated logo"
                      src="{data_url}"
                      scrolling="no"
                      frameborder="0"
                    ></iframe>
                    """
                ),
                unsafe_allow_html=True,
            )

        with nav_col:
            st.markdown('<div class="demai-header__nav">', unsafe_allow_html=True)

            nav_stage_col, nav_controls_col = st.columns([3.2, 1.2], gap="small")

            with nav_stage_col:
                if isinstance(stage, StageMeta):
                    icon = html.escape(stage.icon)
                    title = html.escape(stage.title)
                    st.markdown(
                        dedent(
                            f"""
                            <div class="demai-header__stage">
                              <span class="demai-header__stage-progress">Stage {index + 1} of {total}</span>
                              <span class="demai-header__stage-name">{icon} {title}</span>
                            </div>
                            """
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class=\"demai-header__stage\">"
                        "<span class=\"demai-header__stage-progress\">Stage</span>"
                        "<span class=\"demai-header__stage-name\">Loading…</span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

            with nav_controls_col:
                st.markdown('<div class="demai-header__controls">', unsafe_allow_html=True)

                show_back = isinstance(prev_stage, StageMeta)
                show_next = isinstance(next_stage, StageMeta)

                if show_back and prev_stage is not None:
                    if st.button("← Back", key="demai_header_back"):
                        _set_active_stage(prev_stage.key)
                else:
                    st.markdown('<span class="demai-header__control-placeholder"></span>', unsafe_allow_html=True)

                if show_next and next_stage is not None:
                    if st.button("Next →", key="demai_header_next"):
                        _set_active_stage(next_stage.key)
                else:
                    st.markdown('<span class="demai-header__control-placeholder"></span>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
