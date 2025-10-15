from __future__ import annotations
import html
from collections import Counter
from typing import Any, Dict, List
import streamlit as st
from demistifai.core.constants import TOKEN_POLICY, PII_DISPLAY_LABELS, PII_CHIP_CONFIG

def summarize_pii_counts(detailed_hits: Dict[int, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for columns in detailed_hits.values():
        for spans in columns.values():
            for span in spans:
                t = span.get("type")
                if t:
                    counts[t] += 1
    return {
        "iban": counts.get("iban", 0),
        "credit_card": counts.get("card16", 0),
        "email": counts.get("email", 0),
        "phone": counts.get("phone", 0),
        "otp6": counts.get("otp6", 0),
        "url": counts.get("url", 0),
    }

def format_pii_summary(counts: Dict[str, int]) -> str:
    return " • ".join(f"{label}: {int(counts.get(key, 0) or 0)}" for key, label in PII_DISPLAY_LABELS)

def pii_chip_row_html(counts: Dict[str, int], extra_class: str = "") -> str:
    classes = ["indicator-chip-row", "pii-chip-row"]
    if extra_class:
        classes.append(extra_class)
    chips: List[str] = []
    for key, icon, label in PII_CHIP_CONFIG:
        count = counts.get(key, 0)
        if isinstance(count, (int, float)):
            v = float(count)
            count_display = str(int(round(v))) if abs(v - round(v)) < 1e-6 else f"{v:.2f}".rstrip("0").rstrip(".")
        else:
            count_display = html.escape(str(count))
        chips.append(
            "<span class='lint-chip'><span class='lint-chip__icon'>{icon}</span>"
            "<span class='lint-chip__text'>{label}: {count}</span></span>".format(
                icon=icon, label=html.escape(label), count=count_display
            )
        )
    return "" if not chips else "<div class='{cls}'>{chips}</div>".format(cls=" ".join(classes), chips="".join(chips))

def apply_pii_replacements(text: str, spans: List[Dict[str, Any]]) -> str:
    if not spans:
        return text
    ordered = sorted(spans, key=lambda s: s["start"])
    parts: List[str] = []
    last = 0
    for span in ordered:
        start, end = span["start"], span["end"]
        token = TOKEN_POLICY.get(span.get("type", "pii"), "{{PII}}")
        parts.append(text[last:start])
        parts.append(token)
        last = end
    parts.append(text[last:])
    return "".join(parts)

def _highlight_spans_html(text: str, spans: List[Dict[str, Any]]) -> str:
    colors = {"email": "#ffef9f", "phone": "#ffd6a5", "iban": "#bde0fe", "card16": "#ffc9de", "otp6": "#caffbf", "url": "#d0bdf4"}
    fmt = lambda seg: html.escape(seg).replace("\n", "<br>")
    pieces: List[str] = []
    last = 0
    for span in sorted(spans, key=lambda i: i["start"]):
        start, end = span["start"], span["end"]
        color = colors.get(span.get("type", "pii"), "#e0e0e0")
        pieces.append(fmt(text[last:start]))
        frag = fmt(text[start:end])
        pieces.append(
            f'<mark style="background:{color}; padding:0 3px; border-radius:3px;" title="{html.escape(span.get("type","pii"))}">{frag}</mark>'
        )
        last = end
    pieces.append(fmt(text[last:]))
    return "".join(pieces)

def render_pii_cleanup_banner(lint_counts: Dict[str, int]) -> bool:
    total_hits = sum(int(v or 0) for v in lint_counts.values())
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
        start = st.button("🧹 Start cleanup", key="pii_btn_start", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return start

def _ensure_pii_state() -> None:
    s = st.session_state
    s.setdefault("pii_queue_idx", 0)
    s.setdefault("pii_score", 0)
    s.setdefault("pii_total_flagged", 0)
    s.setdefault("pii_cleaned_count", 0)
    s.setdefault("pii_queue", [])
    s.setdefault("pii_hits_map", {})
    s.setdefault("pii_edits", {})
    s.setdefault("pii_open", False)
