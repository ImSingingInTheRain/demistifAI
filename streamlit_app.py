"""Streamlit entry point for the demistifAI multi-page experience."""

from __future__ import annotations

import streamlit as st
from streamlit_navigation_bar import st_navbar

from demistifai import app_core
import pages as pg

st.set_page_config(
    page_title="demistifAI",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

state, session_state = app_core.initialize_app()

nav_labels = app_core.get_nav_labels()
label_to_stage = {label: app_core.get_stage_key_for_label(label) for label in nav_labels}
current_label = session_state.get("demai_top_nav", nav_labels[0])

selected_label = st_navbar(nav_labels, selected=current_label, key="demai_top_nav")
if not selected_label:
    selected_label = current_label

stage_key = label_to_stage.get(selected_label, app_core.get_active_stage_key())
run_state = state.setdefault("run", {})

if run_state.get("busy") and stage_key != run_state.get("active_stage"):
    stage_key = run_state.get("active_stage", stage_key)
    fallback_label = app_core.get_nav_label_for_stage(stage_key)
    st.session_state["demai_top_nav"] = fallback_label
    selected_label = fallback_label

page_handlers = {
    "intro": pg.home,
    "overview": pg.start_your_machine,
    "data": pg.prepare_data,
    "train": pg.train,
    "evaluate": pg.evaluate,
    "classify": pg.use,
    "model_card": pg.model_card,
}

handler = page_handlers.get(stage_key, pg.home)
handler()
