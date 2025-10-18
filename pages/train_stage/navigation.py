"""UI controls and navigation helpers for the Train stage."""

from __future__ import annotations

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from demistifai.core.utils import streamlit_rerun
from demistifai.ui.components.terminal.train import render_train_terminal


def render_train_terminal_slot(slot: DeltaGenerator) -> None:
    """Render the animated terminal in the stage header grid."""

    with slot:
        render_train_terminal(
            speed_type_ms=22,
            pause_between_lines_ms=320,
        )


def render_prepare_dataset_prompt(stage, section_surface, set_active_stage) -> None:
    """Show the call-to-action that nudges users toward the Data stage."""

    with section_surface():
        st.subheader(
            f"{stage.icon} {stage.title} — How does the spam detector learn from examples?"
        )
        st.info("First prepare and validate your dataset in **📊 Data**.")
        if st.button("Go to Data stage", type="primary"):
            set_active_stage("data")
            streamlit_rerun()


__all__ = ["render_prepare_dataset_prompt", "render_train_terminal_slot"]
