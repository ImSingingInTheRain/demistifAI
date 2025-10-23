"""Welcome stage rendering utilities for the demistifAI app."""

from __future__ import annotations

from typing import Callable, ContextManager, Optional

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from demistifai.core.navigation import activate_stage
from demistifai.core.utils import streamlit_rerun
from demistifai.constants import STAGE_INDEX, STAGES, StageMeta
from demistifai.ui.components import render_stage_top_grid
from demistifai.ui.components.overview import (
    mailbox_preview_markup,
    mailbox_preview_styles,
    mission_brief_styles,
    mission_overview_column_markup,
)
from demistifai.ui.components.shared.macos_iframe_window import (
    MacWindowConfig,
    MacWindowPane,
    render_macos_iframe_window,
)
from demistifai.ui.components.terminal.intro_terminal import (
    IntroTerminalCommand,
    render_interactive_intro_terminal,
)
from demistifai.ui.primitives import render_eu_ai_quote

EU_AI_ACT_DEF = (
    "An AI system infers how to generate outputs that can influence physical or virtual environments."
)

SectionSurface = Callable[[Optional[str]], ContextManager[None]]


def render_intro_stage(*, section_surface: SectionSurface) -> None:
    """Render the welcome/intro stage surface."""

    next_stage_key: Optional[str] = None
    next_stage_meta: Optional[StageMeta] = None
    intro_index = STAGE_INDEX.get("intro")
    if intro_index is not None and intro_index < len(STAGES) - 1:
        next_stage_meta = STAGES[intro_index + 1]
        next_stage_key = next_stage_meta.key

    show_mission = bool(st.session_state.get("intro_show_mission", False))

    def _render_intro_terminal(slot: DeltaGenerator) -> None:
        with slot:
            command, _ready = render_interactive_intro_terminal(
                speed_type_ms=20,
                pause_between_ops_ms=360,
            )
            if command == IntroTerminalCommand.SHOW_MISSION:
                st.session_state["intro_show_mission"] = True
                streamlit_rerun()
            elif command == IntroTerminalCommand.START:
                st.session_state["intro_show_mission"] = True
                if next_stage_key and activate_stage(next_stage_key):
                    streamlit_rerun()

    def _render_show_mission_cta(slot: DeltaGenerator) -> None:
        if show_mission:
            return

        with slot:
            cta_pressed = st.button(
                "Show Mission",
                key="intro_stage_show_mission_cta",
                type="primary",
                use_container_width=True,
            )
            if cta_pressed:
                st.session_state["intro_show_mission"] = True
                streamlit_rerun()

    render_stage_top_grid(
        "intro",
        left_renderer=_render_intro_terminal,
        right_first_renderer=_render_show_mission_cta,
    )

    with section_surface("section-surface--hero"):
        incoming_records = st.session_state.get("incoming") or []
        preview_records = []
        for record in list(incoming_records)[:5]:
            if hasattr(record, "items"):
                preview_records.append(dict(record))
            else:
                preview_records.append(record)

        if show_mission:
            mission_panes = (
                MacWindowPane(
                    html=mission_overview_column_markup(),
                    css=mission_brief_styles(),
                    min_height=420,
                    pane_id="overview-mission-brief",
                ),
                MacWindowPane(
                    html=mailbox_preview_markup(preview_records),
                    css=mailbox_preview_styles(),
                    min_height=420,
                    pane_id="overview-mission-mailbox",
                ),
            )
            render_macos_iframe_window(
                st,
                MacWindowConfig(
                    panes=mission_panes,
                    rows=1,
                    columns=2,
                    column_ratios=(1.1, 0.9),
                ),
            )

        if next_stage_meta is not None and next_stage_key is not None:
            button_key = f"intro_stage_start_{next_stage_key}"
            with st.container():
                cta_clicked = st.button(
                    f"{next_stage_meta.icon} {next_stage_meta.title} ➡️",
                    key=button_key,
                    type="primary",
                    use_container_width=True,
                    help="Jump to the next stage",
                )
                if cta_clicked and activate_stage(next_stage_key):
                    streamlit_rerun()

