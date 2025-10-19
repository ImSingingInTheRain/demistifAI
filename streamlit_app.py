from __future__ import annotations

import logging
import streamlit as st
import streamlit.components.v1 as components
from functools import partial

from demistifai.styles import APP_THEME_CSS
from demistifai.constants import (
    STAGE_TEMPLATE_CSS,
    STAGES,
)
from demistifai.core.language import summarize_language_mix
from demistifai.ui.components import render_language_mix_chip_rows
from demistifai.core.state import _apply_pending_advanced_knob_state, ensure_state, validate_invariants
from demistifai.core.navigation import activate_stage, synchronize_stage_selection
from demistifai.core.session_defaults import initialize_session_defaults
from pages.data import render_data_stage as render_data_stage_content
from pages.use import render_classify_stage as render_classify_stage_content
from pages.evaluate import render_evaluate_stage_page
from pages.overview import render_overview_stage as render_overview_stage_content
from pages.train_stage import render_train_stage_page
from demistifai.ui.custom_header import mount_demai_header
from demistifai.ui.primitives import (
    guidance_popover,
    render_email_inbox_table,
    render_eu_ai_quote,
    render_mailbox_panel,
    render_nerd_mode_toggle,
    section_surface,
    shorten_text,
)
from pages.welcome import render_intro_stage as render_intro_stage_content
from pages.model_card import render_model_card_stage as render_model_card_stage_content
logger = logging.getLogger(__name__)

st.set_page_config(page_title="demistifAI", page_icon="📧", layout="wide")

state = ensure_state()
s = state
ss = st.session_state

st.markdown(APP_THEME_CSS, unsafe_allow_html=True)
st.markdown(STAGE_TEMPLATE_CSS, unsafe_allow_html=True)
mount_demai_header()

_apply_pending_advanced_knob_state()
initialized_keys = initialize_session_defaults(ss)
if initialized_keys:
    logger.debug("Initialized session keys: %s", ", ".join(initialized_keys))

stage_sync = synchronize_stage_selection(state=s, session_state=ss)
if stage_sync.toast_message:
    st.toast(stage_sync.toast_message, icon="⏳")


def set_active_stage(stage_key: str) -> None:
    """Update the active stage and synchronize related navigation state."""

    activate_stage(stage_key)


def _set_adaptive_state(new_value: bool, *, source: str) -> None:
    """Synchronize adaptiveness settings across UI controls."""

    current_value = bool(ss.get("adaptive", False))
    desired_value = bool(new_value)
    if desired_value == current_value:
        return

    ss["adaptive"] = desired_value
    ss["use_adaptiveness"] = desired_value

    if source != "stage":
        ss.pop("adaptive_stage", None)


if ss.get("use_adaptiveness") != bool(ss.get("adaptive", False)):
    ss["use_adaptiveness"] = bool(ss.get("adaptive", False))


STAGE_RENDERERS = {
    "intro": partial(render_intro_stage_content, section_surface=section_surface),
    "overview": partial(
        render_overview_stage_content,
        ss,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        section_surface=section_surface,
    ),
    "data": partial(
        render_data_stage_content,
        section_surface=section_surface,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
    ),
    "train": partial(
        render_train_stage_page,
        set_active_stage=set_active_stage,
        render_eu_ai_quote=render_eu_ai_quote,
        render_language_mix_chip_rows=render_language_mix_chip_rows,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        section_surface=section_surface,
        summarize_language_mix=summarize_language_mix,
    ),
    "evaluate": partial(
        render_evaluate_stage_page,
        section_surface=section_surface,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        shorten_text=shorten_text,
    ),
    "classify": partial(
        render_classify_stage_content,
        ss=ss,
        section_surface=section_surface,
        render_eu_ai_quote=render_eu_ai_quote,
        render_email_inbox_table=render_email_inbox_table,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        render_mailbox_panel=render_mailbox_panel,
        set_adaptive_state=_set_adaptive_state,
    ),
    "model_card": partial(
        render_model_card_stage_content,
        section_surface=section_surface,
        guidance_popover=guidance_popover,
    ),
}


validate_invariants(s)
active_stage = stage_sync.active_stage or (STAGES[0].key if STAGES else None)
renderer = STAGE_RENDERERS.get(active_stage)
if renderer is None and STAGES:
    fallback_key = STAGES[0].key
    renderer = STAGE_RENDERERS.get(fallback_key)
if renderer is None:
    logger.warning("No renderer registered for stage %s", active_stage)
    renderer = lambda: None

if ss.pop("stage_scroll_to_top", False):
    components.html(
        """
        <script>
        (function() {
            const main = window.parent.document.querySelector('section.main');
            if (main && typeof main.scrollTo === 'function') {
                main.scrollTo({ top: 0, behavior: 'smooth' });
            }
            if (window.parent && typeof window.parent.scrollTo === 'function') {
                window.parent.scrollTo({ top: 0, behavior: 'smooth' });
            }
        })();
        </script>
        """,
        height=0,
    )

renderer()

st.markdown("---")
st.caption("© demistifAI — Built for interactive learning and governance discussions.")
