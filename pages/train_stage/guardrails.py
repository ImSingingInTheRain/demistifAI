from __future__ import annotations

from typing import Any, Dict, MutableMapping

import streamlit as st

from pages.train_stage.helpers.guardrails import _guardrail_window_values


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


def render_guardrail_window_preview(ss: MutableMapping[str, Any]) -> None:
    """Display the numeric guardrail window summary in the training context panel."""

    center, _, low, high = _guardrail_window_values(ss)
    low_pct = low * 100.0
    high_pct = high * 100.0
    threshold_pct = max(0.0, min(100.0, center * 100.0))

    band_left = max(0.0, min(100.0, low_pct))
    band_right = max(0.0, min(100.0, high_pct))
    band_width = max(0.0, band_right - band_left)

    gauge_html = f"""
    <div class="train-guardrail-meter">
        <div class="train-guardrail-meter__track">
            <div class="train-guardrail-meter__band" style="left:{band_left:.1f}%;width:{band_width:.1f}%;"></div>
            <div class="train-guardrail-meter__threshold" style="left:{threshold_pct:.1f}%;"></div>
        </div>
        <div class="train-guardrail-meter__labels">
            <span>0</span>
            <span>1</span>
        </div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)
    st.caption(
        "Numeric guardrails wake up when the text score is near τ≈{center:.2f} "
        "(window {low:.2f}–{high:.2f}).".format(center=center, low=low, high=high)
    )
