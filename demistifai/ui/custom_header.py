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
                padding-top: {logo_height + 28}px !important; /* header height + gap */
              }}

              /* Anchor used to target the fixed header container */
              .demai-header__anchor {{
                display: none;
              }}

              /* Fixed header container */
              [data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] > .demai-header__anchor) {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1000;
              }}

              [data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] > .demai-header__anchor) > [data-testid="element-container"] {{
                display: flex;
                align-items: stretch;
                justify-content: center;
                min-height: {logo_height + 12}px;
                padding: 10px 18px;
                box-sizing: border-box;
                background: rgba(15, 23, 42, 0.92);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(94, 234, 212, 0.24);
                box-shadow: 0 12px 24px rgba(8, 15, 33, 0.28);
              }}

              [data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] > .demai-header__anchor) > [data-testid="element-container"] > [data-testid="stHorizontalBlock"] {{
                margin: 0;
                width: 100%;
                display: grid;
                grid-template-columns: auto minmax(0, 1fr) auto;
                align-items: center;
                column-gap: 18px;
              }}

              [data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] > .demai-header__anchor) [data-testid="column"] {{
                padding: 0 !important;
                display: flex;
                flex-direction: column;
                justify-content: center;
                gap: 6px;
              }}

              [data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] > .demai-header__anchor) [data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"] {{
                margin: 0;
                display: flex;
                gap: 8px;
              }}

              [data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] > .demai-header__anchor) [data-testid="column"] > div:first-child {{
                width: 100%;
              }}

              .demai-header__logo-frame {{
                border: none;
                background: transparent;
                width: auto;
                height: {logo_height}px;
                pointer-events: none; /* prevent focusing the iframe */
              }}

              .demai-header__stage {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                align-items: center;
                gap: 8px;
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
                white-space: normal;
                flex: 1 1 auto;
              }}

              .demai-header__button-placeholder {{
                height: 34px;
                border-radius: 12px;
                background: rgba(148, 163, 184, 0.16);
              }}

              [data-testid="stVerticalBlock"]:has(> [data-testid="element-container"] > .demai-header__anchor) [data-testid="stButton"] > button {{
                border-radius: 12px;
                font-weight: 600;
                min-height: 34px;
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

    header_container = st.container()

    with header_container:
        st.markdown('<div class="demai-header__anchor"></div>', unsafe_allow_html=True)

        logo_col, stage_col, controls_col = st.columns([1.1, 2.8, 1.4], gap="small")

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

        with stage_col:
            if isinstance(stage, StageMeta):
                icon = html.escape(stage.icon)
                title = html.escape(stage.title)
                progress_label = "Stage"
                if total > 0:
                    progress_label = f"Stage {index + 1} of {total}"
                elif index >= 0:
                    progress_label = f"Stage {index + 1}"
                st.markdown(
                    dedent(
                        f"""
                        <div class="demai-header__stage">
                          <span class="demai-header__stage-progress">{progress_label}</span>
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

        with controls_col:
            controls = st.columns(2, gap="small")

            show_back = isinstance(prev_stage, StageMeta)
            show_next = isinstance(next_stage, StageMeta)

            with controls[0]:
                if show_back and prev_stage is not None:
                    if st.button(
                        "⬅️",
                        key="demai_header_back",
                        use_container_width=True,
                        help=f"Back to {prev_stage.title}",
                    ):
                        _set_active_stage(prev_stage.key)
                else:
                    st.markdown('<div class="demai-header__button-placeholder"></div>', unsafe_allow_html=True)

            with controls[1]:
                if show_next and next_stage is not None:
                    if st.button(
                        "➡️",
                        key="demai_header_next",
                        use_container_width=True,
                        type="primary",
                        help=f"Forward to {next_stage.title}",
                    ):
                        _set_active_stage(next_stage.key)
                else:
                    st.markdown('<div class="demai-header__button-placeholder"></div>', unsafe_allow_html=True)
