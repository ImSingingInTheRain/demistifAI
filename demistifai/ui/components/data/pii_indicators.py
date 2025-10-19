from __future__ import annotations
from typing import Dict
import streamlit as st
from demistifai.styles.css_blocks import PII_INDICATOR_STYLE
from demistifai.styles.inject import inject_css_once


def render_pii_indicators(counts: Dict[str, int]) -> None:
    """
    Renders a responsive grid of PII counters.
    `counts` expects keys like 'iban', 'credit_card', 'email', 'phone', 'otp6', 'url'.
    """
    inject_css_once(PII_INDICATOR_STYLE)
    st.markdown('<div class="pii-indicators">', unsafe_allow_html=True)
    for label, value in counts.items():
        st.markdown(
            f"""
            <div class="pii-indicator">
              <div class="pii-indicator__label">{label}</div>
              <div class="pii-indicator__value">{int(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
