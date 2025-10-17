"""Reusable popover helpers."""

from __future__ import annotations

import streamlit as st


def guidance_popover(title: str, text: str) -> None:
    """Render a guidance popover with consistent styling."""

    with st.popover(f"❓ {title}"):
        st.write(text)
