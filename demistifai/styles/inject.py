from __future__ import annotations

import streamlit as st


def inject_css_once(*css_blocks: str, key: str = "css_injected") -> None:
    """
    Inject raw CSS <style> blocks into the page exactly once per session.
    """
    flag = st.session_state.get(key, False)
    if not flag:
        for block in css_blocks:
            if block and block.strip():
                st.markdown(block, unsafe_allow_html=True)
        st.session_state[key] = True
