"""Reusable section scaffolding primitives."""

from __future__ import annotations

import html
from contextlib import contextmanager
from typing import Optional

import streamlit as st
from streamlit.delta_generator import DeltaGenerator


@contextmanager
def section_surface(extra_class: Optional[str] = None):
    """Render a consistently styled section surface container."""

    base_class = "section-surface"
    classes = f"{base_class} {extra_class}" if extra_class else base_class

    st.markdown(f'<div class="{classes}">', unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


def render_nerd_mode_toggle(
    *,
    key: str,
    title: str,
    description: Optional[str] = None,
    icon: Optional[str] = "🧠",
    target: DeltaGenerator | None = None,
) -> bool:
    """Render a consistently styled Nerd Mode toggle block."""

    toggle_label = f"{icon} {title}" if icon else title
    wrapper = target.container() if target is not None else st.container()
    default_state = bool(st.session_state.get(key, False))
    icon_html = f"<span class='nerd-toggle__icon'>{html.escape(icon)}</span>" if icon else ""
    safe_title = html.escape(title)
    safe_description = html.escape(description) if description else ""

    with wrapper:
        content_col, toggle_col = st.columns([1, 0.32], gap="large")
        with content_col:
            st.markdown(
                f"<div class='nerd-toggle__title'>{icon_html}<span class='nerd-toggle__title-text'>{safe_title}</span></div>",
                unsafe_allow_html=True,
            )
            if description:
                st.markdown(
                    f"<div class='nerd-toggle__description'>{safe_description}</div>",
                    unsafe_allow_html=True,
                )
        with toggle_col:
            value = st.toggle(
                toggle_label,
                key=key,
                value=default_state,
                label_visibility="collapsed",
            )

    return value
