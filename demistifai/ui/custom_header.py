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
              header[data-testid="stHeader"], [data-testid="stHeader"] {{
                display: none !important;
                visibility: hidden !important;
              }}

              /* Ensure main content has room under our fixed header */
              [data-testid="stAppViewContainer"] .main .block-container {{
                padding-top: {logo_height + 24}px !important; /* header height + gap */
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

                min-height: {logo_height + 12}px;
                padding: 8px 16px;
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
                margin-left: auto;
                display: flex;
                flex-direction: column;
                gap: 6px;
                width: 100%;
                max-width: 480px;
              }}

              .demai-header__stage {{
                display: flex;
                justify-content: space-between;
                align-items: baseline;
                gap: 12px;
                color: rgba(226, 232, 240, 0.92);
                font-size: 0.9rem;
              }}

              .demai-header__stage-progress {{
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.72rem;
                color: rgba(226, 232, 240, 0.72);
              }}

              .demai-header__stage-name {{
                font-weight: 700;
                font-size: 1rem;
                white-space: nowrap;
              }}

              .demai-header__controls {{
                display: flex;
                gap: 8px;
              }}

              .demai-header__controls [data-testid="stVerticalBlock"] {{
                width: 100%;
              }}

              .demai-header__controls [data-testid="stButton"] {{
                width: 100%;
              }}

              .demai-header__controls [data-testid="stButton"] > button {{
                border-radius: 999px;
                font-weight: 700;
              }}

              .demai-header__button-placeholder {{
                height: 38px;
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

        logo_col, nav_col = st.columns([1, 2.2], gap="large")
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

            controls = st.columns(2, gap="small")

            show_back = isinstance(prev_stage, StageMeta)
            show_next = isinstance(next_stage, StageMeta)

            with controls[0]:
                if show_back and prev_stage is not None:
                    if st.button("⬅️ Back", key="demai_header_back", use_container_width=True):
                        _set_active_stage(prev_stage.key)
                else:
                    st.markdown('<div class="demai-header__button-placeholder"></div>', unsafe_allow_html=True)

            with controls[1]:
                if show_next and next_stage is not None:
                    if st.button(
                        "Next ➡️",
                        key="demai_header_next",
                        use_container_width=True,
                        type="primary",
                    ):
                        _set_active_stage(next_stage.key)
                else:
                    st.markdown('<div class="demai-header__button-placeholder"></div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
