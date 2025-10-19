from __future__ import annotations

import html
from collections import Counter
from typing import Callable, Iterable, Mapping, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from demistifai.core.guardrails import (
    GUARDRAIL_LABEL_ICONS,
    _guardrail_badges_html,
    _guardrail_signals,
)

from .guardrail_panel import render_guardrail_panel


def _as_list(values: Iterable | None) -> list:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    return list(values)


def render_guardrail_audit(
    *,
    p_spam: Sequence[float] | Iterable[float],
    y_true: Sequence | Iterable | None,
    current_threshold: float,
    shorten_text: Callable[[str, int], str],
    subjects: Sequence[str] | Iterable[str] | None = None,
    bodies: Sequence[str] | Iterable[str] | None = None,
    combined_texts: Sequence[str] | Iterable[str] | None = None,
    margin_window: float = 0.15,
    max_cards: int = 8,
) -> None:
    """Render the guardrail audit chart and cards."""

    probs = [float(val) for val in _as_list(p_spam)]
    if len(probs) == 0:
        st.caption("Evaluation set empty — guardrail audit unavailable.")
        return

    y_true_values = _as_list(y_true)
    subjects_list = _as_list(subjects) if subjects is not None else []
    bodies_list = _as_list(bodies) if bodies is not None else []
    combined_text_list = _as_list(combined_texts) if combined_texts is not None else []

    guardrail_cards: list[Mapping[str, str]] = []
    guardrail_counts: Counter[str] = Counter()

    sorted_indices = sorted(
        range(len(probs)),
        key=lambda i: abs(probs[i] - current_threshold),
    )

    for idx in sorted_indices:
        if len(guardrail_cards) >= max_cards:
            break

        prob_val = probs[idx]
        margin = abs(prob_val - current_threshold)
        if margin > margin_window:
            continue

        subject_raw = ""
        body_raw = ""
        if subjects is not None and bodies is not None:
            if idx < len(subjects_list):
                subject_raw = subjects_list[idx]
            if idx < len(bodies_list):
                body_raw = bodies_list[idx]
        else:
            text_val = combined_text_list[idx] if idx < len(combined_text_list) else ""
            text_str = text_val if isinstance(text_val, str) else str(text_val or "")
            parts = text_str.split("\n", 1)
            subject_raw = parts[0] if parts else ""
            body_raw = parts[1] if len(parts) > 1 else ""

        subject_raw = str(subject_raw or "").strip()
        body_raw = str(body_raw or "").strip()

        signals = _guardrail_signals(subject_raw, body_raw)
        active_keys = [key for key, flag in signals.items() if flag]
        if not active_keys:
            continue

        guardrail_counts.update(active_keys)

        subject_display = html.escape(
            shorten_text(subject_raw or "(no subject)", limit=100)
        )
        pred_label = "spam" if prob_val >= current_threshold else "safe"
        icon = GUARDRAIL_LABEL_ICONS.get(pred_label, "✉️")

        true_label_raw = y_true_values[idx] if idx < len(y_true_values) else ""
        true_label_clean = str(true_label_raw or "").strip().title()

        meta_left = html.escape(
            f"P(spam) {prob_val:.2f} • Δ {prob_val - current_threshold:+.2f}"
        )
        meta_right_parts = [f"{icon} {pred_label.title()}"]
        if true_label_clean:
            meta_right_parts.append(f"True: {true_label_clean}")
        meta_right = html.escape(" • ".join(meta_right_parts))

        excerpt = (
            shorten_text(body_raw or "", limit=220)
            .replace("\n", " ")
            .strip()
        )
        excerpt_html = html.escape(excerpt) if excerpt else "No body text."
        badges_html = _guardrail_badges_html(signals)
        body_html = "".join(
            [
                badges_html,
                (
                    "<div style=\"margin-top:0.4rem; color: rgba(55, 65, 81, 0.85);\">"
                    f"{excerpt_html}</div>"
                ),
            ]
        )

        guardrail_cards.append(
            {
                "subject": subject_display,
                "meta_left": meta_left,
                "meta_right": meta_right,
                "body": body_html,
            }
        )

    if not guardrail_cards:
        st.caption("No borderline emails triggered guardrail signals in the test set.")
        return

    chart_obj = None
    counts_sorted = guardrail_counts.most_common()
    if counts_sorted:
        guardrail_df = pd.DataFrame(counts_sorted, columns=["guardrail", "count"])
        guardrail_chart = (
            alt.Chart(guardrail_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("count:Q", title="Emails flagged"),
                y=alt.Y("guardrail:N", sort="-x", title="Signal"),
                color=alt.Color("guardrail:N", legend=None),
                tooltip=[
                    alt.Tooltip("guardrail:N", title="Signal"),
                    alt.Tooltip("count:Q", title="Emails"),
                ],
            )
            .properties(height=220)
        )
        chart_obj = st.altair_chart(guardrail_chart, use_container_width=True)

    render_guardrail_panel(chart=chart_obj, cards=guardrail_cards)
