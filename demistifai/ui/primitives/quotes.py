"""Quote-related UI primitives."""

from __future__ import annotations

import html

import streamlit as st


def eu_ai_quote_box(text: str, label: str = "EU AI Act") -> str:
    """Return HTML markup for a styled EU AI Act quote box."""

    escaped_text = html.escape(text)
    escaped_label = html.escape(label)
    return (
        """
        <div class="ai-quote-box">
            <div class="ai-quote-box__icon">⚖️</div>
            <div class="ai-quote-box__content">
                <span class="ai-quote-box__source">{label}</span>
                <p>{text}</p>
            </div>
        </div>
        """
        .format(label=escaped_label, text=escaped_text)
    )


def render_eu_ai_quote(text: str, label: str = "From the EU AI Act, Article 3") -> None:
    """Render a formatted EU AI Act quote."""

    st.markdown(eu_ai_quote_box(text, label), unsafe_allow_html=True)
