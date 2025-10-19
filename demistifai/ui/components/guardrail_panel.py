from __future__ import annotations

from typing import Iterable, Mapping

import streamlit as st

from demistifai.styles.css_blocks import GUARDRAIL_PANEL_STYLE
from demistifai.styles.inject import inject_css_once


def render_guardrail_panel(chart, cards: Iterable[Mapping[str, str]]) -> None:
    """
    Renders a panel with a chart and a scrollable list of guardrail cards.
    `chart` can be a matplotlib/altair plot or any st.* render block already created.
    `cards` is an iterable of dicts with keys: 'subject', 'meta_left', 'meta_right', 'body'.
    """
    inject_css_once(GUARDRAIL_PANEL_STYLE)

    st.markdown('<div class="guardrail-panel">', unsafe_allow_html=True)

    # Chart area
    st.markdown('<div class="guardrail-panel__chart">', unsafe_allow_html=True)
    chart  # chart should have been rendered before passing here (or render inline)
    st.markdown('</div>', unsafe_allow_html=True)

    # Cards list
    st.markdown('<div class="guardrail-card-list">', unsafe_allow_html=True)
    for c in cards:
        subject = c.get("subject", "")
        meta_left = c.get("meta_left", "")
        meta_right = c.get("meta_right", "")
        body = c.get("body", "")
        st.markdown(
            f"""
            <div class="guardrail-card">
              <div class="guardrail-card__subject">{subject}</div>
              <div class="guardrail-card__meta">
                <span>{meta_left}</span>
                <span>{meta_right}</span>
              </div>
              <div class="guardrail-card__body">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
