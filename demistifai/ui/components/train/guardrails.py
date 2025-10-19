from __future__ import annotations

from typing import Any, MutableMapping

import streamlit as st

from pages.train_stage.helpers.guardrails import _guardrail_window_values

GUARDRAIL_PREVIEW_STYLES = """
<style>
.train-guardrail-meter {
    margin-top: 0.25rem;
    margin-bottom: 0.75rem;
}

.train-guardrail-meter__track {
    position: relative;
    height: 10px;
    border-radius: 999px;
    background: rgba(49, 51, 63, 0.12);
    overflow: hidden;
}

.train-guardrail-meter__band {
    position: absolute;
    top: 0;
    bottom: 0;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.55), rgba(37, 99, 235, 0.75));
}

.train-guardrail-meter__threshold {
    position: absolute;
    top: -4px;
    width: 2px;
    bottom: -4px;
    border-radius: 999px;
    background: rgba(37, 99, 235, 0.85);
}

.train-guardrail-meter__labels {
    display: flex;
    justify-content: space-between;
    margin-top: 0.25rem;
    font-size: 0.7rem;
    color: rgba(49, 51, 63, 0.6);
}
</style>
"""


def build_guardrail_window_preview(
    ss: MutableMapping[str, Any]
) -> tuple[str, str, str]:
    """Return the styles, HTML markup, and caption for the guardrail preview."""

    center, _, low, high = _guardrail_window_values(ss)
    low_pct = low * 100.0
    high_pct = high * 100.0
    threshold_pct = max(0.0, min(100.0, center * 100.0))

    band_left = max(0.0, min(100.0, low_pct))
    band_right = max(0.0, min(100.0, high_pct))
    band_width = max(0.0, band_right - band_left)

    html = f"""
    <div class=\"train-guardrail-meter\">
        <div class=\"train-guardrail-meter__track\">
            <div class=\"train-guardrail-meter__band\" style=\"left:{band_left:.1f}%;width:{band_width:.1f}%;\"></div>
            <div class=\"train-guardrail-meter__threshold\" style=\"left:{threshold_pct:.1f}%;\"></div>
        </div>
        <div class=\"train-guardrail-meter__labels\">
            <span>0</span>
            <span>1</span>
        </div>
    </div>
    """

    caption = (
        "Numeric guardrails wake up when the text score is near τ≈{center:.2f} "
        "(window {low:.2f}–{high:.2f}).".format(center=center, low=low, high=high)
    )

    return GUARDRAIL_PREVIEW_STYLES, html, caption


def render_guardrail_window_preview(ss: MutableMapping[str, Any]) -> None:
    """Display the numeric guardrail window summary in the training context panel."""

    styles, html, caption = build_guardrail_window_preview(ss)
    st.markdown(styles, unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)
    st.caption(caption)
