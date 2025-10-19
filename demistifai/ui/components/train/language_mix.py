"""Language mix rendering helpers for Streamlit surfaces."""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

__all__ = ["render_language_mix_chip_rows"]


def render_language_mix_chip_rows(
    train_mix: Optional[Dict[str, Any]],
    test_mix: Optional[Dict[str, Any]],
) -> None:
    """Render paired language mix chip rows for train/test splits."""

    if not train_mix and not test_mix:
        st.caption("Lang mix: unknown")
        return

    train_available = bool(train_mix and train_mix.get("available"))
    test_available = bool(test_mix and test_mix.get("available"))

    if not train_available and not test_available:
        st.caption("Lang mix: unknown")
        return

    col_train, col_test = st.columns(2)
    _render_language_mix_column(col_train, "Train", train_mix)
    _render_language_mix_column(col_test, "Test", test_mix)


def _render_language_mix_column(
    container,
    title: str,
    mix: Optional[Dict[str, Any]],
) -> None:
    container.markdown(f"**{title} language mix**")
    if not mix or not mix.get("available"):
        container.caption("Unknown (language detector unavailable).")
        return

    total = int(mix.get("total", 0))
    top = list(mix.get("top", []))
    if total <= 0 or not top:
        container.caption("No detected language (blank emails).")
        return

    chip_parts = [
        "<div style='display:flex;flex-wrap:wrap;gap:0.35rem;margin-top:0.35rem;'>"
    ]
    for lang, share in top:
        chip_parts.append(
            "<span style='background:rgba(49,51,63,0.08);color:rgba(15,23,42,0.85);"
            "border-radius:999px;padding:0.2rem 0.6rem;font-size:0.75rem;font-weight:600;'>"
            f"{lang.upper()} {share * 100:.0f}%</span>"
        )

    other_share = float(mix.get("other", 0.0))
    if other_share > 0.005:
        chip_parts.append(
            "<span style='background:rgba(49,51,63,0.08);color:rgba(15,23,42,0.65);"
            "border-radius:999px;padding:0.2rem 0.6rem;font-size:0.75rem;font-weight:600;'>"
            f"OTHER {other_share * 100:.0f}%</span>"
        )

    chip_parts.append("</div>")
    container.markdown("".join(chip_parts), unsafe_allow_html=True)
    container.caption(f"n={total}")
