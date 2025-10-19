"""Guardrail-related UI helpers for the Train stage."""

from __future__ import annotations

from dataclasses import dataclass
import textwrap
from typing import Any, MutableMapping

import streamlit as st

from pages.train_stage.helpers.guardrails import (
    _guardrail_window_values,
    _numeric_guardrails_caption_text,
)


@dataclass(frozen=True)
class GuardrailWindowPreview:
    """HTML/CSS payload that renders the numeric guardrail window preview."""

    html: str
    css: str
    caption: str


_GUARDRAIL_WINDOW_CSS = textwrap.dedent(
    """
    .train-guardrail-meter {
      margin: 0.35rem 0 0.55rem 0;
    }
    .train-guardrail-meter__track {
      position: relative;
      height: 12px;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.12);
      overflow: hidden;
    }
    .train-guardrail-meter__band {
      position: absolute;
      top: 0;
      bottom: 0;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(59, 130, 246, 0.4), rgba(96, 165, 250, 0.45));
    }
    .train-guardrail-meter__threshold {
      position: absolute;
      top: -6px;
      bottom: -6px;
      width: 2px;
      border-radius: 999px;
      background: linear-gradient(180deg, #f87171, #ef4444);
      box-shadow: 0 0 0 1px rgba(248, 113, 113, 0.35);
    }
    .train-guardrail-meter__labels {
      display: flex;
      justify-content: space-between;
      font-size: 0.7rem;
      color: rgba(15, 23, 42, 0.55);
      margin-top: 0.25rem;
    }
    """
).strip()


def build_guardrail_window_preview(ss: MutableMapping[str, Any]) -> GuardrailWindowPreview:
    """Construct the guardrail preview HTML, CSS, and caption from session state."""

    center, _, low, high = _guardrail_window_values(ss)
    low_pct = low * 100.0
    high_pct = high * 100.0
    threshold_pct = max(0.0, min(100.0, center * 100.0))

    band_left = max(0.0, min(100.0, low_pct))
    band_right = max(0.0, min(100.0, high_pct))
    band_width = max(0.0, band_right - band_left)

    gauge_html = (
        "<div class=\"train-guardrail-meter\">"
        "<div class=\"train-guardrail-meter__track\">"
        f"<div class=\"train-guardrail-meter__band\" style=\"left:{band_left:.1f}%;width:{band_width:.1f}%;\"></div>"
        f"<div class=\"train-guardrail-meter__threshold\" style=\"left:{threshold_pct:.1f}%;\"></div>"
        "</div>"
        "<div class=\"train-guardrail-meter__labels\">"
        "<span>0</span><span>1</span>"
        "</div>"
        "</div>"
    )

    caption = _numeric_guardrails_caption_text(ss)
    return GuardrailWindowPreview(html=gauge_html, css=_GUARDRAIL_WINDOW_CSS, caption=caption)


def render_guardrail_window_preview(ss: MutableMapping[str, Any]) -> GuardrailWindowPreview:
    """Render the numeric guardrail window summary in the training context panel."""

    preview = build_guardrail_window_preview(ss)
    if preview.css:
        st.markdown(f"<style>{preview.css}</style>", unsafe_allow_html=True)
    st.markdown(preview.html, unsafe_allow_html=True)
    if preview.caption:
        st.caption(preview.caption)
    return preview
