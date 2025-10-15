from __future__ import annotations

from typing import Any, Dict

import streamlit as st
from streamlit.errors import StreamlitAPIException

from demistifai.dataset import (
    ATTACHMENT_MIX_PRESETS,
    DEFAULT_ATTACHMENT_MIX,
    DEFAULT_DATASET_CONFIG,
    DatasetConfig,
)


def _set_advanced_knob_state(
    config: Dict[str, Any] | DatasetConfig,
    *,
    force: bool = False,
) -> None:
    """Ensure Nerd Mode advanced knob widgets reflect the active configuration."""

    if not isinstance(config, dict):
        config = DEFAULT_DATASET_CONFIG

    try:
        links_level = int(str(config.get("susp_link_level", "1")))
    except (TypeError, ValueError):
        links_level = 1
    tld_level = str(config.get("susp_tld_level", "med"))
    caps_level = str(config.get("caps_intensity", "med"))
    money_level = str(config.get("money_urgency", "low"))
    current_mix = config.get("attachments_mix", DEFAULT_ATTACHMENT_MIX)
    attachment_choice = next(
        (name for name, mix in ATTACHMENT_MIX_PRESETS.items() if mix == current_mix),
        "Balanced",
    )
    try:
        noise_pct = float(config.get("label_noise_pct", 0.0))
    except (TypeError, ValueError):
        noise_pct = 0.0
    try:
        seed_value = int(config.get("seed", 42))
    except (TypeError, ValueError):
        seed_value = 42
    poison_demo = bool(config.get("poison_demo", False))

    adv_state = {
        "adv_links_level": links_level,
        "adv_tld_level": tld_level,
        "adv_caps_level": caps_level,
        "adv_money_level": money_level,
        "adv_attachment_choice": attachment_choice,
        "adv_label_noise_pct": noise_pct,
        "adv_seed": seed_value,
        "adv_poison_demo": poison_demo,
    }

    for key, value in adv_state.items():
        if force or key not in st.session_state:
            try:
                st.session_state[key] = value
            except StreamlitAPIException:
                pending = st.session_state.setdefault("_pending_advanced_knob_state", {})
                pending[key] = value
                st.session_state["_needs_advanced_knob_rerun"] = True


def _apply_pending_advanced_knob_state() -> None:
    """Apply any queued advanced knob updates before widgets instantiate."""

    pending = st.session_state.pop("_pending_advanced_knob_state", None)
    if not pending:
        return

    for key, value in pending.items():
        st.session_state[key] = value

    st.session_state.pop("_needs_advanced_knob_rerun", None)


def _push_data_stage_flash(level: str, message: str) -> None:
    """Queue a flash message to render at the top of the data stage."""

    if not message:
        return

    queue = st.session_state.setdefault("data_stage_flash_queue", [])
    queue.append({"level": level, "message": message})
