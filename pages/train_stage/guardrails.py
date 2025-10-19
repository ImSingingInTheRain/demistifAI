from __future__ import annotations

from typing import Any, Dict, MutableMapping

import streamlit as st


def render_guardrail_controls(
    ss: MutableMapping[str, Any],
    *,
    guard_params: Dict[str, Any],
    nerd_mode_train_active: bool,
) -> Dict[str, Any]:
    """Render the launchpad guardrail controls and return updated parameters."""

    auto_mode = st.toggle(
        "Implicit strategy mode (Auto)",
        value=(guard_params.get("assist_center_mode") == "auto"),
        key="launchpad_auto_mode",
    )
    guard_params["assist_center_mode"] = "auto" if auto_mode else "manual"
    if guard_params.get("assist_center_mode") == "manual" and not nerd_mode_train_active:
        guard_params["assist_center"] = st.slider(
            "Where ‘borderline’ lives (0–1)",
            0.30,
            0.90,
            float(guard_params.get("assist_center", ss.get("threshold", 0.6))),
            0.01,
        )
        guard_params["uncertainty_band"] = st.slider(
            "Uncertainty band (±)",
            0.0,
            0.20,
            float(guard_params.get("uncertainty_band", 0.08)),
            0.01,
        )
    elif guard_params.get("assist_center_mode") == "auto":
        st.caption(
            "We’ll pick the center from the hold-out set after training so numeric clues trigger where they help most."
        )
    ss["guard_params"] = guard_params
    return guard_params
