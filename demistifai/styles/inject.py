from __future__ import annotations

import hashlib
from typing import Iterable

import streamlit as st


def _normalized_blocks(css_blocks: Iterable[str]) -> list[str]:
    """Return a list of non-empty CSS blocks with whitespace trimmed."""

    return [block.strip() for block in css_blocks if block and block.strip()]


def _auto_key(blocks: list[str]) -> str:
    """Generate a stable key for the provided CSS blocks."""

    digest = hashlib.sha1("\n".join(blocks).encode("utf-8")).hexdigest()
    return f"css_injected::{digest}"


def inject_css_once(*css_blocks: str, key: str | None = None) -> None:
    """Inject raw CSS <style> blocks into the page exactly once per session."""

    blocks = _normalized_blocks(css_blocks)
    if not blocks:
        return

    state_key = key or _auto_key(blocks)
    if st.session_state.get(state_key):
        return

    for block in blocks:
        st.markdown(block, unsafe_allow_html=True)

    st.session_state[state_key] = True
