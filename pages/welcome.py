"""Welcome stage rendering utilities for the demistifAI app."""

from __future__ import annotations

from typing import Callable, ContextManager, Optional

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from demistifai.constants import STAGE_INDEX, STAGES
from demistifai.ui.components import render_stage_top_grid
from demistifai.ui.components.intro import render_intro_hero
from demistifai.ui.components.shared import render_mac_window
from demistifai.ui.components.terminal.article3 import (
    _WELCOME_LINES,
    render_ai_act_terminal as render_welcome_ai_act_terminal,
)
from demistifai.ui.primitives import render_eu_ai_quote

EU_AI_ACT_DEF = (
    "An AI system infers how to generate outputs that can influence physical or virtual environments."
)

SectionSurface = Callable[[Optional[str]], ContextManager[None]]


def render_intro_stage(*, section_surface: SectionSurface) -> None:
    """Render the welcome/intro stage surface."""

    next_stage_key: Optional[str] = None
    intro_index = STAGE_INDEX.get("intro")
    if intro_index is not None and intro_index < len(STAGES) - 1:
        next_stage_key = STAGES[intro_index + 1].key

    def _render_intro_terminal(slot: DeltaGenerator) -> None:
        with slot:
            render_welcome_ai_act_terminal(
                demai_lines=_WELCOME_LINES,
                speed_type_ms=20,
                pause_between_ops_ms=360,
            )

    render_stage_top_grid("intro", left_renderer=_render_intro_terminal)

    with section_surface("section-surface--hero"):
        hero_css, left_col_html, right_col_html = render_intro_hero()

        render_mac_window(
            st,
            title="Start your demAI journey",
            ratios=(0.33, 0.67),
            col_html=[left_col_html, right_col_html],
            id_suffix="intro-lifecycle",
            scoped_css=hero_css,
            max_width=1200,
        )

