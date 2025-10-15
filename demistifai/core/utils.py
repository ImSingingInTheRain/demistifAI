from __future__ import annotations
import streamlit as st


def streamlit_rerun() -> None:
    """Compatibility wrapper for st.rerun / st.experimental_rerun."""
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn is None:
        raise AttributeError("Streamlit rerun function unavailable")
    rerun_fn()
