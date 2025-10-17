from __future__ import annotations

import logging
import random
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

from demistifai.constants import STAGES
from demistifai.core.state import ensure_state, validate_invariants
from demistifai.core.utils import streamlit_rerun

from demistifai.constants import (
    APP_THEME_CSS,
    AUTONOMY_LEVELS,
    EMAIL_INBOX_TABLE_CSS,
    STAGE_BY_KEY,
    STAGE_TEMPLATE_CSS,
)
from demistifai.core.language import (
    summarize_language_mix,
    render_language_mix_chip_rows,
)
from demistifai.core.state import (
    _set_advanced_knob_state,
    _apply_pending_advanced_knob_state,
)
from demistifai.dataset import (
    DEFAULT_DATASET_CONFIG,
    compute_dataset_summary,
    generate_incoming_batch,
    starter_dataset_copy,
)
from demistifai.modeling import FEATURE_ORDER

from pages.data import render_data_stage as render_data_stage_content
from pages.use import render_classify_stage as render_classify_stage_content
from pages.evaluate import render_evaluate_stage_page
from pages.overview import render_overview_stage as render_overview_stage_content
from pages.train_stage import render_train_stage_page
from demistifai.ui.custom_header import mount_demai_header
from demistifai.ui.primitives import (
    render_email_inbox_table,
    render_eu_ai_quote,
    render_mailbox_panel,
    render_nerd_mode_toggle,
    section_surface,
    shorten_text,
)
from pages.welcome import render_intro_stage as render_intro_stage_content
from stages.model_card import (
    render_model_card_stage as render_model_card_stage_content,
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="demistifAI", page_icon="📧", layout="wide")

state = ensure_state()
s = state
ss = st.session_state

ss.setdefault("viewport_is_mobile", False)

st.markdown(APP_THEME_CSS, unsafe_allow_html=True)
st.markdown(STAGE_TEMPLATE_CSS, unsafe_allow_html=True)
st.markdown(EMAIL_INBOX_TABLE_CSS, unsafe_allow_html=True)
mount_demai_header()

def guidance_popover(title: str, text: str):
    with st.popover(f"❓ {title}"):
        st.write(text)

_apply_pending_advanced_knob_state()
requested_stage_values = st.query_params.get_all("stage")
requested_stage = requested_stage_values[0] if requested_stage_values else None
run_state = s.setdefault("run", {})
if requested_stage in STAGE_BY_KEY and requested_stage != run_state.get("active_stage"):
    run_state["active_stage"] = requested_stage

if not run_state.get("active_stage") and STAGES:
    run_state["active_stage"] = STAGES[0].key

default_stage_key = run_state.get("active_stage")
selected_stage_key = ss.get("active_stage", default_stage_key)

if selected_stage_key not in STAGE_BY_KEY:
    if STAGES:
        selected_stage_key = STAGES[0].key
    else:
        selected_stage_key = None

if default_stage_key not in STAGE_BY_KEY:
    default_stage_key = selected_stage_key

if (
    run_state.get("busy")
    and default_stage_key in STAGE_BY_KEY
    and selected_stage_key not in (None, default_stage_key)
):
    selected_stage_key = default_stage_key
    ss["active_stage"] = default_stage_key
    toast_message = "Training in progress — navigation disabled."
    ui_state = s.setdefault("ui", {})
    toasts = ui_state.setdefault("toasts", [])
    if toast_message not in toasts:
        toasts.append(toast_message)
        st.toast(toast_message, icon="⏳")

if selected_stage_key is not None:
    if selected_stage_key != run_state.get("active_stage"):
        ss["stage_scroll_to_top"] = True
    run_state["active_stage"] = selected_stage_key
    ss["active_stage"] = selected_stage_key
    if st.query_params.get_all("stage") != [selected_stage_key]:
        st.query_params["stage"] = selected_stage_key

ss.setdefault("nerd_mode", False)
ss.setdefault("autonomy", AUTONOMY_LEVELS[0])
ss.setdefault("threshold", 0.6)
ss.setdefault("nerd_mode_eval", False)
ss.setdefault("eval_timestamp", None)
ss.setdefault("eval_temp_threshold", float(ss["threshold"]))
ss.setdefault("adaptive", True)
ss.setdefault("labeled", starter_dataset_copy())      # list of dicts: title, body, label
if "incoming_seed" not in ss:
    ss["incoming_seed"] = None
if not ss.get("incoming"):
    seed = random.randint(1, 1_000_000)
    ss["incoming_seed"] = seed
    ss["incoming"] = generate_incoming_batch(n=30, seed=seed, spam_ratio=0.32)
ss.setdefault("model", None)
ss.setdefault("split_cache", None)
ss.setdefault("mail_inbox", [])  # list of dicts: title, body, pred, p_spam
ss.setdefault("mail_spam", [])
ss.setdefault("metrics", {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
ss.setdefault("last_classification", None)
ss.setdefault("numeric_adjustments", {feat: 0.0 for feat in FEATURE_ORDER})
ss.setdefault("nerd_mode_data", False)
ss.setdefault("nerd_mode_train", False)
ss.setdefault("calibrate_probabilities", False)
ss.setdefault("calibration_result", None)
ss.setdefault(
    "train_params",
    {"test_size": 0.30, "random_state": 42, "max_iter": 1000, "C": 1.0},
)
ss.setdefault(
    "guard_params",
    {
        "assist_center": float(ss.get("threshold", 0.6)),
        "uncertainty_band": 0.08,
        "numeric_scale": 0.5,
        "numeric_logit_cap": 1.0,
        "combine_strategy": "blend",
        "shift_suspicious_tld": -0.04,
        "shift_many_links": -0.03,
        "shift_calm_text": 0.02,
    },
)
ss.setdefault("use_high_autonomy", ss.get("autonomy", AUTONOMY_LEVELS[0]).startswith("High"))
ss.setdefault("train_story_run_id", None)
ss.setdefault("use_batch_results", [])
ss.setdefault("use_adaptiveness", bool(ss.get("adaptive", True)))
ss.setdefault("use_audit_log", [])
ss.setdefault("nerd_mode_use", False)
ss.setdefault("dataset_config", DEFAULT_DATASET_CONFIG.copy())
_set_advanced_knob_state(ss["dataset_config"], force=False)
if "dataset_summary" not in ss:
    ss["dataset_summary"] = compute_dataset_summary(ss["labeled"])
ss.setdefault("dataset_last_built_at", datetime.now().isoformat(timespec="seconds"))
ss.setdefault("previous_dataset_summary", None)
ss.setdefault("dataset_preview", None)
ss.setdefault("dataset_preview_config", None)
ss.setdefault("dataset_preview_summary", None)
ss.setdefault("dataset_manual_queue", None)
ss.setdefault("dataset_controls_open", False)
ss.setdefault("dataset_has_generated_once", False)
ss.setdefault("datasets", [])
ss.setdefault("active_dataset_snapshot", None)
ss.setdefault("dataset_snapshot_name", "")
ss.setdefault("last_dataset_delta_story", None)
ss.setdefault("dataset_compare_delta", None)
ss.setdefault("dataset_preview_lint", None)
ss.setdefault("last_eval_results", None)


def set_active_stage(stage_key: str) -> None:
    """Update the active stage and synchronize related navigation state."""

    if stage_key not in STAGE_BY_KEY:
        return

    run_state = s.setdefault("run", {})
    current_stage = run_state.get("active_stage")
    if current_stage == stage_key:
        return

    run_state["active_stage"] = stage_key
    ss["active_stage"] = stage_key
    ss["stage_scroll_to_top"] = True

    # Mirror the active stage in the URL query parameter for deep-linking and
    # to support refresh persistence.
    if st.query_params.get_all("stage") != [stage_key]:
        st.query_params["stage"] = stage_key


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


ss["use_adaptiveness"] = bool(ss.get("adaptive", False))


def render_intro_stage():

    render_intro_stage_content(section_surface=section_surface)


def render_overview_stage():
    render_overview_stage_content(
        ss,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        section_surface=section_surface,
    )



def render_data_stage():
    render_data_stage_content(
        section_surface=section_surface,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
    )


def render_evaluate_stage():
    render_evaluate_stage_page(
        section_surface=section_surface,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        shorten_text=shorten_text,
    )

def render_classify_stage():
    render_classify_stage_content(
        ss=ss,
        section_surface=section_surface,
        render_eu_ai_quote=render_eu_ai_quote,
        render_email_inbox_table=render_email_inbox_table,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        render_mailbox_panel=render_mailbox_panel,
        set_adaptive_state=_set_adaptive_state,
    )


def render_model_card_stage():
    render_model_card_stage_content(
        section_surface=section_surface,
        guidance_popover=guidance_popover,
    )


def render_train_stage() -> None:
    render_train_stage_page(
        set_active_stage=set_active_stage,
        render_eu_ai_quote=render_eu_ai_quote,
        render_language_mix_chip_rows=render_language_mix_chip_rows,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        section_surface=section_surface,
        summarize_language_mix=summarize_language_mix,
    )


STAGE_RENDERERS = {
    'intro': render_intro_stage,
    'overview': render_overview_stage,
    'data': render_data_stage,
    'train': render_train_stage,
    'evaluate': render_evaluate_stage,
    'classify': render_classify_stage,
    'model_card': render_model_card_stage,
}


validate_invariants(s)
active_stage = s.get("run", {}).get("active_stage") or STAGES[0].key
ss["active_stage"] = active_stage
renderer = STAGE_RENDERERS.get(active_stage, render_intro_stage)

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
