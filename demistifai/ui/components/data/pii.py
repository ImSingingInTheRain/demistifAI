"""Streamlit rendering helpers for the PII cleanup workflow."""

from __future__ import annotations

import html
from typing import Any, Dict, List

import streamlit as st

from demistifai.config.tokens import PII_CHIP_CONFIG

__all__ = [
    "_ensure_pii_state",
    "_highlight_spans_html",
    "pii_chip_row_html",
    "render_pii_cleanup_banner",
]


def pii_chip_row_html(counts: Dict[str, int], extra_class: str = "") -> str:
    classes = ["indicator-chip-row", "pii-chip-row"]
    if extra_class:
        classes.append(extra_class)

    chips: List[str] = []
    for key, icon, label in PII_CHIP_CONFIG:
        count = counts.get(key, 0)
        if isinstance(count, (int, float)):
            value = float(count)
            if abs(value - round(value)) < 1e-6:
                count_display = str(int(round(value)))
            else:
                count_display = f"{value:.2f}".rstrip("0").rstrip(".")
        else:
            count_display = html.escape(str(count))
        chips.append(
            "<span class='lint-chip'><span class='lint-chip__icon'>{icon}</span>"
            "<span class='lint-chip__text'>{label}: {count}</span></span>".format(
                icon=icon,
                label=html.escape(label),
                count=count_display,
            )
        )

    if not chips:
        return ""
    return "<div class='{cls}'>{chips}</div>".format(
        cls=" ".join(classes),
        chips="".join(chips),
    )


def _apply_markup(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def _highlight_spans_html(text: str, spans: List[Dict[str, Any]]) -> str:
    colors = {
        "email": "#ffef9f",
        "phone": "#ffd6a5",
        "iban": "#bde0fe",
        "card16": "#ffc9de",
        "otp6": "#caffbf",
        "url": "#d0bdf4",
    }

    pieces: List[str] = []
    last = 0
    for span in sorted(spans, key=lambda item: item["start"]):
        start, end = span["start"], span["end"]
        color = colors.get(span.get("type", "pii"), "#e0e0e0")
        pieces.append(_apply_markup(text[last:start]))
        fragment = _apply_markup(text[start:end])
        pieces.append(
            "<mark style=\"background:{color}; padding:0 3px; border-radius:3px;\" title=\"{label}\">{frag}</mark>".format(
                color=color,
                label=html.escape(span.get("type", "pii")),
                frag=fragment,
            )
        )
        last = end
    pieces.append(_apply_markup(text[last:]))
    return "".join(pieces)


def render_pii_cleanup_banner(lint_counts: Dict[str, int]) -> bool:
    total_hits = sum(int(value or 0) for value in lint_counts.values())
    if total_hits <= 0:
        return False

    st.markdown("<div class='pii-alert-card'>", unsafe_allow_html=True)
    left, right = st.columns([3.5, 1.25], gap="large")
    with left:
        summary_html = pii_chip_row_html(lint_counts, extra_class="pii-alert-card__chips")
        st.markdown(
            """
            <div class="pii-alert-card__body">
              <div class="pii-alert-card__title">⚠️ Personal data alert</div>
              <p>The data used to build an AI system should not include personal data unless it is really necessary. Click on <strong>Start cleanup</strong> to simulate a data minimization process and replace personal data in your dataset with anonymized tags.</p>
              {summary}
            </div>
            """.format(summary=summary_html or ""),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("<div class='pii-alert-card__action'>", unsafe_allow_html=True)
        start = st.button(
            "🧹 Start cleanup",
            key="pii_btn_start",
            type="primary",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    return start


def _ensure_pii_state() -> None:
    session = st.session_state
    session.setdefault("pii_queue_idx", 0)
    session.setdefault("pii_score", 0)
    session.setdefault("pii_total_flagged", 0)
    session.setdefault("pii_cleaned_count", 0)
    session.setdefault("pii_queue", [])
    session.setdefault("pii_hits_map", {})
    session.setdefault("pii_edits", {})
    session.setdefault("pii_open", False)
