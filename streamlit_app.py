from __future__ import annotations

import base64
import html
import math
import json
import logging
import random
import string
import time
import textwrap
import hashlib
import re
from collections import Counter
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode
from uuid import uuid4

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitAPIException

from demistifai.constants import (
    APP_THEME_CSS,
    AUTONOMY_LEVELS,
    BRANDS,
    CLASSES,
    COURIERS,
    DATASET_LEGIT_DOMAINS,
    DATASET_SUSPICIOUS_TLDS,
    EMAIL_INBOX_TABLE_CSS,
    LIFECYCLE_RING_HTML,
    STAGE_BY_KEY,
    STAGE_INDEX,
    STAGE_TEMPLATE_CSS,
    STAGES,
    URGENCY,
    StageMeta,
)
from demistifai.dataset import (
    ATTACHMENT_MIX_PRESETS,
    ATTACHMENT_TYPES,
    DEFAULT_ATTACHMENT_MIX,
    DEFAULT_DATASET_CONFIG,
    EDGE_CASE_TEMPLATES,
    DatasetConfig,
    STARTER_LABELED,
    generate_incoming_batch,
    SUSPICIOUS_TLD_SUFFIXES,
    starter_dataset_copy,
    build_dataset_from_config,
    compute_dataset_hash,
    compute_dataset_summary,
    _estimate_token_stats,
    dataset_delta_story,
    dataset_summary_delta,
    explain_config_change,
    generate_labeled_dataset,
    lint_dataset_detailed,
    lint_dataset,
    lint_text_spans,
    _caps_ratio,
    _count_money_mentions,
    _count_suspicious_links,
    _has_suspicious_tld,
)
from demistifai.modeling import (
    FEATURE_DISPLAY_NAMES,
    FEATURE_ORDER,
    FEATURE_PLAIN_LANGUAGE,
    HybridEmbedFeatsLogReg,
    PlattProbabilityCalibrator,
    URGENCY_TERMS,
    _combine_text,
    _fmt_delta,
    _fmt_pct,
    _pr_acc_cm,
    _predict_proba_batch,
    _y01,
    assess_performance,
    cache_train_embeddings,
    combine_text,
    compute_confusion,
    compute_numeric_features,
    df_confusion,
    encode_texts,
    embedding_backend_info,
    extract_urls,
    _counts,
    features_matrix,
    get_domain_tld,
    get_encoder,
    get_nearest_training_examples,
    make_after_eval_story,
    make_after_training_story,
    model_kind_string,
    numeric_feature_contributions,
    plot_threshold_curves,
    predict_spam_probability,
    threshold_presets,
    top_token_importances,
    verdict_label,
)

from stages.train_stage import render_train_stage
from ui.animated_logo import render_demai_logo
from components.arch_demai import render_demai_architecture
from components.components_cmd import render_ai_act_terminal
from components.components_mac import render_mac_window
logger = logging.getLogger(__name__)


def callable_or_attr(target: Any, attr: str | None = None) -> bool:
    """Return True if ``target`` (or one of its attributes) is callable."""

    try:
        value = getattr(target, attr) if attr else target
    except Exception:
        return False
    return callable(value)


from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


TOKEN_POLICY = {
    "email": "{{EMAIL}}",
    "phone": "{{PHONE}}",
    "iban": "{{IBAN}}",
    "card16": "{{CARD_16}}",
    "otp6": "{{OTP_6}}",
    "url": "{{URL_SUSPICIOUS}}",
}


def _streamlit_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn is None:
        raise AttributeError("Streamlit rerun function unavailable")
    rerun_fn()


PII_DISPLAY_LABELS = [
    ("iban", "IBAN"),
    ("credit_card", "Card"),
    ("email", "Emails"),
    ("phone", "Phones"),
    ("otp6", "OTPs"),
    ("url", "Suspicious URLs"),
]


PII_CHIP_CONFIG = [
    ("credit_card", "💳", "Credit card"),
    ("iban", "🏦", "IBAN"),
    ("email", "📧", "Emails"),
    ("phone", "☎️", "Phones"),
    ("otp6", "🔐", "OTPs"),
    ("url", "🌐", "Suspicious URLs"),
]


PII_INDICATOR_STYLE = """
<style>
.pii-indicators {
    display: grid;
    gap: 0.75rem;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    margin-bottom: 0.5rem;
}
.pii-indicator {
    background: var(--secondary-background-color, rgba(250, 250, 250, 0.85));
    border-radius: 0.75rem;
    border: 1px solid rgba(49, 51, 63, 0.15);
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
    padding: 0.75rem 1rem;
    text-align: center;
}
.pii-indicator__label {
    color: rgba(49, 51, 63, 0.65);
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.pii-indicator__value {
    color: var(--text-color, #0d0d0d);
    font-size: 1.75rem;
    font-weight: 600;
    margin-top: 0.35rem;
}
</style>
"""


GUARDRAIL_PANEL_STYLE = """
<style>
.guardrail-panel {
    display: grid;
    gap: 1.5rem;
}
.guardrail-panel__chart {
    background: var(--secondary-background-color, rgba(248, 250, 252, 0.85));
    border-radius: 1rem;
    border: 1px solid rgba(148, 163, 184, 0.35);
    padding: 1rem;
}
.guardrail-card-list {
    max-height: 320px;
    overflow-y: auto;
    padding-right: 0.5rem;
    display: grid;
    gap: 0.75rem;
}
.guardrail-card {
    background: rgba(255, 255, 255, 0.82);
    border-radius: 0.85rem;
    border: 1px solid rgba(148, 163, 184, 0.4);
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
    padding: 0.85rem 1rem;
}
.guardrail-card__subject {
    font-weight: 600;
    color: var(--text-color, #111827);
    margin-bottom: 0.35rem;
    line-height: 1.3;
}
.guardrail-card__meta {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.8rem;
    color: rgba(55, 65, 81, 0.85);
    margin-bottom: 0.5rem;
    gap: 0.75rem;
}
.guardrail-card__label {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-weight: 500;
}
.guardrail-card__label--spam {
    color: #b91c1c;
}
.guardrail-card__label--safe {
    color: #1d4ed8;
}
.guardrail-card__badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.guardrail-badge {
    border-radius: 999px;
    padding: 0.25rem 0.65rem;
    font-size: 0.8rem;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border: 1px solid rgba(148, 163, 184, 0.55);
}
.guardrail-badge--on {
    background: rgba(250, 204, 21, 0.25);
    border-color: rgba(234, 179, 8, 0.8);
    color: #854d0e;
}
.guardrail-badge--off {
    background: rgba(226, 232, 240, 0.35);
    color: rgba(71, 85, 105, 0.75);
}
.guardrail-panel__example {
    font-size: 0.85rem;
    color: rgba(55, 65, 81, 0.9);
    background: rgba(226, 232, 240, 0.45);
    border-radius: 0.75rem;
    padding: 0.65rem 0.85rem;
}
</style>
"""


GUARDRAIL_BADGE_DEFS = [
    ("link", "🔗", "Suspicious link"),
    ("caps", "🔊", "ALL CAPS"),
    ("money", "💰", "Money"),
    ("urgency", "⚡", "Urgency"),
]


GUARDRAIL_CAPS_THRESHOLD = 0.3


GUARDRAIL_URGENCY_TERMS = {term.lower() for term in URGENCY}
GUARDRAIL_URGENCY_TERMS.update({term.lower() for term in URGENCY_TERMS})


GUARDRAIL_LABEL_ICONS = {"spam": "🚩", "safe": "📥"}


def _guardrail_signals(subject: str, body: str) -> Dict[str, bool]:
    text = f"{subject or ''}\n{body or ''}".strip()
    if not text:
        text = subject or body or ""

    text_lower = text.lower()

    suspicious_links = False
    if text:
        try:
            suspicious_links = (
                _count_suspicious_links(text) > 0
                or _has_suspicious_tld(text)
            )
        except Exception:
            suspicious_links = False

    try:
        caps_ratio = _caps_ratio(text)
    except Exception:
        caps_ratio = 0.0

    try:
        money_hits = _count_money_mentions(text) > 0
    except Exception:
        money_hits = False

    urgency_hits = any(term in text_lower for term in GUARDRAIL_URGENCY_TERMS)

    return {
        "link": bool(suspicious_links),
        "caps": bool(caps_ratio >= GUARDRAIL_CAPS_THRESHOLD),
        "money": bool(money_hits),
        "urgency": bool(urgency_hits),
    }


def _guardrail_badges_html(flags: Dict[str, bool]) -> str:
    badges: List[str] = []
    for key, icon, label in GUARDRAIL_BADGE_DEFS:
        active = bool(flags.get(key))
        classes = "guardrail-badge guardrail-badge--on" if active else "guardrail-badge guardrail-badge--off"
        badges.append(
            f"<span class='{classes}' title='{html.escape(label)}' aria-label='{html.escape(label)}'>{icon}<span style='font-size:0.75rem;'>{html.escape(label)}</span></span>"
        )
    return "".join(badges)


URL_CANDIDATE_RE = re.compile(r"(?i)\b(?:https?://|www\.)[\w\-._~:/?#\[\]@!$&'()*+,;=%]+")
MONEY_CANDIDATE_RE = re.compile(
    r"(?i)(?:\$\s?\d[\d,]*(?:\.\d{2})?|\b(?:usd|eur|gbp|aud|cad|sgd)\s?\$?\d[\d,]*(?:\.\d{2})?|\b\d[\d,]*(?:\.\d{2})?\s?(?:usd|eur|gbp|aud|cad|dollars)\b)"
)
ALERT_PHRASE_RE = re.compile(
    r"(?i)\b(?:verify|reset|account|urgent)(?:\s+(?:verify|reset|account|urgent|your|the|this|that|my|our)){1,3}\b"
)


def extract_candidate_spans(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    """Return candidate spans (URL/money/alert phrases) with start/end indices."""

    if not text:
        return []

    matches: list[Tuple[int, int, str]] = []
    for pattern in (URL_CANDIDATE_RE, MONEY_CANDIDATE_RE, ALERT_PHRASE_RE):
        for match in pattern.finditer(text):
            start, end = match.span()
            snippet = text[start:end]
            if not snippet.strip():
                continue
            matches.append((start, end, snippet))

    if not matches:
        return []

    matches.sort(key=lambda item: (item[0], item[1]))
    deduped: list[Tuple[str, Tuple[int, int]]] = []
    seen: set[Tuple[int, int]] = set()
    for start, end, snippet in matches:
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((snippet, (start, end)))
    return deduped


def summarize_pii_counts(
    detailed_hits: Dict[int, Dict[str, List[Dict[str, Any]]]]
) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for columns in detailed_hits.values():
        for spans in columns.values():
            for span in spans:
                span_type = span.get("type")
                if span_type:
                    counts[span_type] += 1
    return {
        "iban": counts.get("iban", 0),
        "credit_card": counts.get("card16", 0),
        "email": counts.get("email", 0),
        "phone": counts.get("phone", 0),
        "otp6": counts.get("otp6", 0),
        "url": counts.get("url", 0),
    }


def format_pii_summary(counts: Dict[str, int]) -> str:
    return " • ".join(
        f"{label}: {int(counts.get(key, 0) or 0)}" for key, label in PII_DISPLAY_LABELS
    )

try:
    from langdetect import DetectorFactory as _LangDetectFactory
    from langdetect import LangDetectException as _LangDetectException
    from langdetect import detect as _langdetect_detect

    _LangDetectFactory.seed = 0
    _LANG_DETECT_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _langdetect_detect = None
    _LangDetectException = Exception  # type: ignore
    _LANG_DETECT_AVAILABLE = False


has_embed = callable_or_attr(locals().get("encode_texts"))
has_calibration = callable_or_attr(locals().get("PlattProbabilityCalibrator"))
has_langdetect = bool(_LANG_DETECT_AVAILABLE and callable_or_attr(_langdetect_detect))


def _detect_language_code(text: str) -> str:
    """Return a two-letter language code, or blank if detection fails."""

    text = (text or "").strip()
    if not text:
        return ""
    if _LANG_DETECT_AVAILABLE and _langdetect_detect is not None:
        try:
            return str(_langdetect_detect(text)).lower()
        except _LangDetectException:
            return ""
        except Exception:
            return ""
    return "en"


def summarize_language_mix(texts: Iterable[str], top_k: int = 3) -> Dict[str, Any]:
    """Summarize the language distribution for a collection of texts."""

    if not _LANG_DETECT_AVAILABLE:
        return {"available": False, "top": [], "total": 0, "other": 0.0, "counts": Counter()}

    counts: Counter[str] = Counter()
    detected_total = 0
    for raw_text in texts:
        code = _detect_language_code(raw_text)
        if not code:
            continue
        counts[code] += 1
        detected_total += 1

    if detected_total == 0:
        return {
            "available": True,
            "top": [],
            "total": 0,
            "other": 0.0,
            "counts": counts,
        }

    top_items = counts.most_common(max(1, int(top_k)))
    top = [(lang, count / detected_total) for lang, count in top_items]
    covered = sum(share for _, share in top)
    other_share = max(0.0, 1.0 - covered)

    return {
        "available": True,
        "top": top,
        "total": detected_total,
        "other": other_share,
        "counts": counts,
    }


def format_language_mix_summary(label: str, mix: Optional[Dict[str, Any]]) -> str:
    """Format a single-line summary for the language mix."""

    if not mix or not mix.get("available"):
        return "Lang mix: unknown"

    total = int(mix.get("total", 0))
    top = list(mix.get("top", []))
    if total <= 0 or not top:
        return f"Lang mix ({label}): —."

    parts = [f"{lang} {share * 100:.0f}%" for lang, share in top]
    other_share = float(mix.get("other", 0.0))
    if other_share > 0.005:
        parts.append(f"other {other_share * 100:.0f}%")

    return f"Lang mix ({label}): {', '.join(parts)}."


def render_language_mix_chip_rows(
    train_mix: Optional[Dict[str, Any]], test_mix: Optional[Dict[str, Any]]
) -> None:
    """Render side-by-side chip rows summarizing train/test language mix."""

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


def _render_language_mix_column(container, title: str, mix: Optional[Dict[str, Any]]) -> None:
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
        "<div style=\"display:flex;flex-wrap:wrap;gap:0.35rem;margin-top:0.35rem;\">"
    ]
    for lang, share in top:
        chip_parts.append(
            (
                "<span style=\"background:rgba(49,51,63,0.08);color:rgba(15,23,42,0.85);"
                "border-radius:999px;padding:0.2rem 0.6rem;font-size:0.75rem;font-weight:600;\">"
                f"{lang.upper()} {share * 100:.0f}%"
                "</span>"
            )
        )

    other_share = float(mix.get("other", 0.0))
    if other_share > 0.005:
        chip_parts.append(
            (
                "<span style=\"background:rgba(49,51,63,0.08);color:rgba(15,23,42,0.65);"
                "border-radius:999px;padding:0.2rem 0.6rem;font-size:0.75rem;font-weight:600;\">"
                f"OTHER {other_share * 100:.0f}%"
                "</span>"
            )
        )

    chip_parts.append("</div>")
    container.markdown("".join(chip_parts), unsafe_allow_html=True)
    container.caption(f"n={total}")


def _shorten_text(text: str, limit: int = 120) -> str:
    """Return a shortened version of *text* capped at *limit* characters."""

    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _safe_subject(row: dict) -> str:
    return str(row.get("title", "") or "").strip()


def pii_chip_row_html(counts: Dict[str, int], extra_class: str = "") -> str:
    classes = ["indicator-chip-row", "pii-chip-row"]
    if extra_class:
        classes.append(extra_class)
    chip_parts: List[str] = []
    for key, icon, label in PII_CHIP_CONFIG:
        count = counts.get(key, 0)
        if isinstance(count, (int, float)):
            count_value = float(count)
            if abs(count_value - round(count_value)) < 1e-6:
                count_display = str(int(round(count_value)))
            else:
                count_display = f"{count_value:.2f}".rstrip("0").rstrip(".")
        else:
            count_display = html.escape(str(count))
        chip_parts.append(
            "<span class='lint-chip'><span class='lint-chip__icon'>{icon}</span>"
            "<span class='lint-chip__text'>{label}: {count}</span></span>".format(
                icon=icon,
                label=html.escape(label),
                count=count_display,
            )
        )
    if not chip_parts:
        return ""
    return "<div class='{classes}'>{chips}</div>".format(
        classes=" ".join(classes),
        chips="".join(chip_parts),
    )


def apply_pii_replacements(text: str, spans: List[Dict[str, Any]]) -> str:
    if not spans:
        return text
    ordered = sorted(spans, key=lambda span: span["start"])
    pieces: List[str] = []
    last = 0
    for span in ordered:
        start, end = span["start"], span["end"]
        token = TOKEN_POLICY.get(span.get("type", "pii"), "{{PII}}")
        pieces.append(text[last:start])
        pieces.append(token)
        last = end
    pieces.append(text[last:])
    return "".join(pieces)

st.set_page_config(page_title="demistifAI", page_icon="📧", layout="wide")

st.markdown(APP_THEME_CSS, unsafe_allow_html=True)
st.markdown(STAGE_TEMPLATE_CSS, unsafe_allow_html=True)
st.markdown(EMAIL_INBOX_TABLE_CSS, unsafe_allow_html=True)


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


def guidance_popover(title: str, text: str):
    with st.popover(f"❓ {title}"):
        st.write(text)


def eu_ai_quote_box(text: str, label: str = "EU AI Act") -> str:
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
    st.markdown(eu_ai_quote_box(text, label), unsafe_allow_html=True)


# ==== PII Cleanup UI ===========================================================
def _highlight_spans_html(text: str, spans: List[Dict[str, Any]]) -> str:
    colors = {
        "email": "#ffef9f",
        "phone": "#ffd6a5",
        "iban": "#bde0fe",
        "card16": "#ffc9de",
        "otp6": "#caffbf",
        "url": "#d0bdf4",
    }

    def _fmt(segment: str) -> str:
        return html.escape(segment).replace("\n", "<br>")

    pieces: List[str] = []
    last = 0
    for span in sorted(spans, key=lambda item: item["start"]):
        start, end = span["start"], span["end"]
        span_type = span.get("type", "pii")
        color = colors.get(span_type, "#e0e0e0")
        pieces.append(_fmt(text[last:start]))
        fragment = _fmt(text[start:end])
        pieces.append(
            (
                f'<mark style="background:{color}; padding:0 3px; border-radius:3px;" '
                f'title="{html.escape(span_type)}">{fragment}</mark>'
            )
        )
        last = end
    pieces.append(_fmt(text[last:]))
    return "".join(pieces)


def render_pii_cleanup_banner(lint_counts: Dict[str, int]) -> bool:
    total_hits = sum(int(value or 0) for value in lint_counts.values())
    if total_hits <= 0:
        return False
    st.markdown("<div class='pii-alert-card'>", unsafe_allow_html=True)
    column_left, column_right = st.columns([3.5, 1.25], gap="large")
    with column_left:
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
    with column_right:
        st.markdown("<div class='pii-alert-card__action'>", unsafe_allow_html=True)
        start = st.button("🧹 Start cleanup", key="pii_btn_start", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    return start


def _ensure_pii_state() -> None:
    state = st.session_state
    state.setdefault("pii_queue_idx", 0)
    state.setdefault("pii_score", 0)
    state.setdefault("pii_total_flagged", 0)
    state.setdefault("pii_cleaned_count", 0)
    state.setdefault("pii_queue", [])
    state.setdefault("pii_hits_map", {})
    state.setdefault("pii_edits", {})
    state.setdefault("pii_open", False)


VALID_LABELS = {"spam", "safe"}


def _normalize_label(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.strip().lower()
    if x in {"ham", "legit", "legitimate"}:
        return "safe"
    return x


def _validate_csv_schema(df: pd.DataFrame) -> tuple[bool, str]:
    required = {"title", "body", "label"}
    missing = required - set(map(str.lower, df.columns))
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    return True, ""


def route_decision(autonomy: str, y_hat: str, pspam: Optional[float], threshold: float):
    routed = None
    if pspam is not None:
        to_spam = pspam >= threshold
    else:
        to_spam = y_hat == "spam"

    if autonomy.startswith("High"):
        routed = "Spam" if to_spam else "Inbox"
        action = f"Auto-routed to **{routed}** (threshold={threshold:.2f})"
    else:
        action = f"Recommend: {'Spam' if to_spam else 'Inbox'} (threshold={threshold:.2f})"
    return action, routed

def download_text(text: str, filename: str, label: str = "Download"):
    b64 = base64.b64encode(text.encode("utf-8")).decode()
    st.markdown(f'<a href="data:text/plain;base64,{b64}" download="{filename}">{label}</a>', unsafe_allow_html=True)


def _evaluate_dataset_health(
    summary: Dict[str, Any] | None,
    lint_counts: Optional[Dict[str, int]],
) -> Dict[str, Any]:
    summary = summary or {}
    if lint_counts is None:
        lint_counts = {}
        lint_label = "Unknown"
    else:
        lint_flags = sum(int(v or 0) for v in lint_counts.values())
        lint_label = "Clean" if lint_flags == 0 else f"{lint_flags} flag{'s' if lint_flags != 1 else ''}"
    lint_flags = sum(int(v or 0) for v in lint_counts.values())
    spam_ratio = summary.get("spam_ratio")
    total_rows = summary.get("total")

    badge_text = None
    health_emoji = None
    spam_pct = None
    failures = 0
    missing_required = spam_ratio is None or total_rows is None

    if not missing_required:
        try:
            spam_pct = max(0.0, min(100.0, float(spam_ratio) * 100.0))
        except (TypeError, ValueError):
            missing_required = True
        if not isinstance(total_rows, (int, float)):
            missing_required = True
        else:
            if total_rows < 300:
                failures += 1
        if spam_pct is not None and not (40.0 <= spam_pct <= 60.0):
            failures += 1
        if lint_counts and lint_flags > 0:
            failures += 1

    if not missing_required:
        if failures == 0:
            badge_text = "🟢 Good"
            health_emoji = "🟢"
        elif failures <= 2:
            badge_text = "🟡 Needs work"
            health_emoji = "🟡"
        else:
            badge_text = "🔴 Risky"
            health_emoji = "🔴"

    return {
        "badge_text": badge_text,
        "health_emoji": health_emoji,
        "spam_pct": spam_pct,
        "total_rows": total_rows,
        "lint_label": lint_label,
        "lint_flags": lint_flags,
    }


def render_nerd_mode_toggle(
    *,
    key: str,
    title: str,
    description: str,
    icon: Optional[str] = "🧠",
) -> bool:
    """Render a consistently styled Nerd Mode toggle block."""

    toggle_label = f"{icon} {title}" if icon else title
    value = st.toggle(toggle_label, key=key, value=bool(ss.get(key, False)))
    if description:
        st.caption(description)
    return value


def render_email_inbox_table(
    df: pd.DataFrame,
    *,
    title: str,
    subtitle: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> None:
    """Display a small email-centric table with shared styling."""

    with st.container(border=True):
        st.markdown(f"**{title}**")
        if subtitle:
            st.caption(subtitle)

        if df is None or df.empty:
            st.caption("No emails to display.")
            return

        display_df = df.copy()
        if columns:
            existing = [col for col in columns if col in display_df.columns]
            if existing:
                display_df = display_df[existing]

        st.dataframe(display_df, hide_index=True, width="stretch")


def render_mailbox_panel(
    messages: Optional[List[Dict[str, Any]]],
    *,
    mailbox_title: str,
    filled_subtitle: str,
    empty_subtitle: str,
) -> None:
    """Render a mailbox tab with consistent styling and fallbacks."""

    with st.container(border=True):
        st.markdown(f"**{mailbox_title}**")
        records = messages or []
        if not records:
            st.caption(empty_subtitle)
            return

        st.caption(filled_subtitle)
        df_box = pd.DataFrame(records)
        column_order = ["title", "pred", "p_spam", "body"]
        rename_map = {
            "title": "Title",
            "pred": "Predicted",
            "p_spam": "P(spam)",
            "body": "Body",
        }
        existing = [col for col in column_order if col in df_box.columns]
        if existing:
            df_display = df_box[existing].rename(columns=rename_map)
        else:
            df_display = df_box
        st.dataframe(df_display, hide_index=True, width="stretch")


def _append_audit(event: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Record an audit log entry for the current session."""

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": event,
    }
    if details:
        entry["details"] = details

    log = ss.setdefault("use_audit_log", [])
    log.append(entry)


def _export_batch_df(rows: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    """Normalize batch result rows into a consistent DataFrame for export."""

    base_cols = ["title", "body", "pred", "p_spam", "p_safe", "action", "routed_to"]
    if not rows:
        return pd.DataFrame(columns=base_cols)

    df_rows = pd.DataFrame(rows)
    for col in base_cols:
        if col not in df_rows.columns:
            df_rows[col] = None
    return df_rows[base_cols]


def _set_advanced_knob_state(config: Dict[str, Any] | DatasetConfig, *, force: bool = False) -> None:
    """Ensure Nerd Mode advanced knob widgets reflect the active configuration."""

    if not isinstance(config, dict):
        config = DEFAULT_DATASET_CONFIG

    try:
        links_level = int(str(config.get("susp_link_level", "1")))
    except (TypeError, ValueError):
        links_level = 1
    tld_level = str(config.get("susp_tld_level", "med"))
    caps_level = str(config.get("caps_intensity", "med"))
    money_level = str(config.get("money_urgency", "low"))
    current_mix = config.get("attachments_mix", DEFAULT_ATTACHMENT_MIX)
    attachment_choice = next(
        (name for name, mix in ATTACHMENT_MIX_PRESETS.items() if mix == current_mix),
        "Balanced",
    )
    try:
        noise_pct = float(config.get("label_noise_pct", 0.0))
    except (TypeError, ValueError):
        noise_pct = 0.0
    try:
        seed_value = int(config.get("seed", 42))
    except (TypeError, ValueError):
        seed_value = 42
    poison_demo = bool(config.get("poison_demo", False))

    adv_state = {
        "adv_links_level": links_level,
        "adv_tld_level": tld_level,
        "adv_caps_level": caps_level,
        "adv_money_level": money_level,
        "adv_attachment_choice": attachment_choice,
        "adv_label_noise_pct": noise_pct,
        "adv_seed": seed_value,
        "adv_poison_demo": poison_demo,
    }

    for key, value in adv_state.items():
        if force or key not in st.session_state:
            try:
                st.session_state[key] = value
            except StreamlitAPIException:
                pending = st.session_state.setdefault("_pending_advanced_knob_state", {})
                pending[key] = value
                st.session_state["_needs_advanced_knob_rerun"] = True


def _apply_pending_advanced_knob_state() -> None:
    """Apply any queued advanced knob updates before widgets instantiate."""

    pending = st.session_state.pop("_pending_advanced_knob_state", None)
    if not pending:
        return

    for key, value in pending.items():
        st.session_state[key] = value

    st.session_state.pop("_needs_advanced_knob_rerun", None)


def _push_data_stage_flash(level: str, message: str) -> None:
    """Queue a flash message to render at the top of the data stage."""

    if not message:
        return

    queue = st.session_state.setdefault("data_stage_flash_queue", [])
    queue.append({"level": level, "message": message})


@st.cache_data(show_spinner=False)
def _compute_cached_embeddings(dataset_hash: str, texts: tuple[str, ...]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    return encode_texts(list(texts))


ss = st.session_state
_apply_pending_advanced_knob_state()
requested_stage_values = st.query_params.get_all("stage")
requested_stage = requested_stage_values[0] if requested_stage_values else None
default_stage = STAGES[0].key
ss.setdefault("active_stage", default_stage)
if requested_stage in STAGE_BY_KEY:
    if requested_stage != ss["active_stage"]:
        ss["active_stage"] = requested_stage
        ss["stage_scroll_to_top"] = True
else:
    if st.query_params.get_all("stage") != [ss["active_stage"]]:
        st.query_params["stage"] = ss["active_stage"]
ss.setdefault("nerd_mode", False)
ss.setdefault("autonomy", AUTONOMY_LEVELS[0])
ss.setdefault("threshold", 0.6)
ss.setdefault("nerd_mode_eval", False)
ss.setdefault("eval_timestamp", None)
ss.setdefault("eval_temp_threshold", float(ss["threshold"]))
ss.setdefault("adaptive", True)
ss.setdefault("labeled", starter_dataset_copy())      # list of dicts: title, body, label
if "incoming_seed" not in ss:
    ss["incoming_seed"] = None
if not ss.get("incoming"):
    seed = random.randint(1, 1_000_000)
    ss["incoming_seed"] = seed
    ss["incoming"] = generate_incoming_batch(n=30, seed=seed, spam_ratio=0.32)
ss.setdefault("model", None)
ss.setdefault("split_cache", None)
ss.setdefault("mail_inbox", [])  # list of dicts: title, body, pred, p_spam
ss.setdefault("mail_spam", [])
ss.setdefault("metrics", {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
ss.setdefault("last_classification", None)
ss.setdefault("numeric_adjustments", {feat: 0.0 for feat in FEATURE_ORDER})
ss.setdefault("nerd_mode_data", False)
ss.setdefault("nerd_mode_train", False)
ss.setdefault("calibrate_probabilities", False)
ss.setdefault("calibration_result", None)
ss.setdefault(
    "train_params",
    {"test_size": 0.30, "random_state": 42, "max_iter": 1000, "C": 1.0},
)
ss.setdefault(
    "guard_params",
    {
        "assist_center": float(ss.get("threshold", 0.6)),
        "uncertainty_band": 0.08,
        "numeric_scale": 0.5,
        "numeric_logit_cap": 1.0,
        "combine_strategy": "blend",
        "shift_suspicious_tld": -0.04,
        "shift_many_links": -0.03,
        "shift_calm_text": 0.02,
    },
)
ss.setdefault("use_high_autonomy", ss.get("autonomy", AUTONOMY_LEVELS[0]).startswith("High"))
ss.setdefault("train_story_run_id", None)
ss.setdefault("use_batch_results", [])
ss.setdefault("use_adaptiveness", bool(ss.get("adaptive", True)))
ss.setdefault("use_audit_log", [])
ss.setdefault("nerd_mode_use", False)
ss.setdefault("dataset_config", DEFAULT_DATASET_CONFIG.copy())
_set_advanced_knob_state(ss["dataset_config"], force=False)
if "dataset_summary" not in ss:
    ss["dataset_summary"] = compute_dataset_summary(ss["labeled"])
ss.setdefault("dataset_last_built_at", datetime.now().isoformat(timespec="seconds"))
ss.setdefault("previous_dataset_summary", None)
ss.setdefault("dataset_preview", None)
ss.setdefault("dataset_preview_config", None)
ss.setdefault("dataset_preview_summary", None)
ss.setdefault("dataset_manual_queue", None)
ss.setdefault("dataset_controls_open", False)
ss.setdefault("dataset_has_generated_once", False)
ss.setdefault("datasets", [])
ss.setdefault("active_dataset_snapshot", None)
ss.setdefault("dataset_snapshot_name", "")
ss.setdefault("last_dataset_delta_story", None)
ss.setdefault("dataset_compare_delta", None)
ss.setdefault("dataset_preview_lint", None)
ss.setdefault("last_eval_results", None)


def set_active_stage(stage_key: str) -> None:
    """Update the active stage and synchronize related navigation state."""

    if stage_key not in STAGE_BY_KEY:
        return

    if ss.get("active_stage") != stage_key:
        ss["active_stage"] = stage_key
        ss["stage_scroll_to_top"] = True

    # Keep the sidebar radio selection aligned with the active stage so the
    # UI immediately reflects navigation triggered by buttons elsewhere.
    if ss.get("sidebar_stage_nav") != stage_key:
        ss["sidebar_stage_nav"] = stage_key

    # Mirror the active stage in the URL query parameter for deep-linking and
    # to support refresh persistence.
    if st.query_params.get_all("stage") != [stage_key]:
        st.query_params["stage"] = stage_key


def _set_adaptive_state(new_value: bool, *, source: str) -> None:
    """Synchronize adaptiveness settings across UI controls."""

    current_value = bool(ss.get("adaptive", False))
    desired_value = bool(new_value)
    if desired_value == current_value:
        return

    ss["adaptive"] = desired_value
    ss["use_adaptiveness"] = desired_value

    if source != "sidebar":
        ss.pop("adaptive_sidebar", None)
    if source != "stage":
        ss.pop("adaptive_stage", None)


def _handle_sidebar_adaptive_change() -> None:
    _set_adaptive_state(ss.get("adaptive_sidebar", ss.get("adaptive", False)), source="sidebar")


ss["use_adaptiveness"] = bool(ss.get("adaptive", False))

with st.sidebar:
    st.markdown("<div class='sidebar-shell'>", unsafe_allow_html=True)
    st.markdown(
        textwrap.dedent(
            """
        <div class="sidebar-brand">
            <p class="sidebar-title">demistifAI control room</p>
            <p class="sidebar-subtitle">Navigate the lifecycle, review guidance, and manage your session without losing progress.</p>
        </div>
        """
        ),
        unsafe_allow_html=True,
    )

    stage_keys = [stage.key for stage in STAGES]
    active_index = STAGE_INDEX.get(ss.get("active_stage", STAGES[0].key), 0)

    st.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
    selected_stage = st.radio(
        "Navigate demistifAI",
        stage_keys,
        index=active_index,
        key="sidebar_stage_nav",
        label_visibility="collapsed",
        format_func=lambda key: f"{STAGE_BY_KEY[key].icon} {STAGE_BY_KEY[key].title}",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if selected_stage != ss.get("active_stage"):
        set_active_stage(selected_stage)

    current_stage = STAGE_BY_KEY.get(ss.get("active_stage", selected_stage))
    if current_stage is not None:
        st.markdown(
            """
            <div class="sidebar-stage-card">
                <div class="sidebar-stage-card__icon">{icon}</div>
                <div class="sidebar-stage-card__meta">
                    <span class="sidebar-stage-card__eyebrow">Current stage</span>
                    <p class="sidebar-stage-card__title">{title}</p>
                    <p class="sidebar-stage-card__description">{description}</p>
                </div>
            </div>
            """.format(
                icon=html.escape(current_stage.icon),
                title=html.escape(current_stage.title),
                description=html.escape(current_stage.description),
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div class='sidebar-section-title'>Session controls</div>", unsafe_allow_html=True)
    st.toggle(
        "Learn from my corrections (adaptiveness)",
        value=ss.get("adaptive", True),
        key="adaptive_sidebar",
        help="When enabled, your corrections in the Use stage will update the model during the session.",
    )
    _handle_sidebar_adaptive_change()

    if st.button("🔄 Reset demo data", use_container_width=True):
        ss["labeled"] = starter_dataset_copy()
        seed = random.randint(1, 1_000_000)
        ss["incoming_seed"] = seed
        ss["incoming"] = generate_incoming_batch(n=30, seed=seed, spam_ratio=0.32)
        ss["model"] = None
        ss["split_cache"] = None
        ss["mail_inbox"].clear(); ss["mail_spam"].clear()
        ss["metrics"] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        ss["last_classification"] = None
        ss["numeric_adjustments"] = {feat: 0.0 for feat in FEATURE_ORDER}
        ss["use_batch_results"] = []
        ss["use_audit_log"] = []
        ss["nerd_mode_use"] = False
        ss["use_high_autonomy"] = ss.get("autonomy", AUTONOMY_LEVELS[0]).startswith("High")
        ss["adaptive"] = True
        ss["use_adaptiveness"] = True
        ss.pop("adaptive_sidebar", None)
        ss.pop("adaptive_stage", None)
        st.success("Reset complete.")

    st.caption(
        "Need a refresher? Use the navigation above to revisit any step without restarting your scenario."
    )
    st.markdown("</div>", unsafe_allow_html=True)

def render_intro_stage():

    next_stage_key: Optional[str] = None
    intro_index = STAGE_INDEX.get("intro")
    if intro_index is not None and intro_index < len(STAGES) - 1:
        next_stage_key = STAGES[intro_index + 1].key

    with section_surface("section-surface--hero"):
        render_demai_logo()
        render_ai_act_terminal()

        window_css = textwrap.dedent(
            """
            <style>
                .mw-intro-lifecycle {
                    position: relative;
                    margin: clamp(1.2rem, 4vw, 2.4rem) auto 2.4rem;
                    max-width: min(1100px, 100%);
                    border-radius: 20px;
                    overflow: hidden;
                    isolation: isolate;
                    border: 1px solid rgba(15, 23, 42, 0.08);
                    box-shadow: 0 28px 64px rgba(15, 23, 42, 0.16);
                }
                .mw-intro-lifecycle::before {
                    content: "";
                    position: absolute;
                    inset: 0;
                    pointer-events: none;
                    background:
                        radial-gradient(circle at top left, rgba(96, 165, 250, 0.22), transparent 58%),
                        radial-gradient(circle at bottom right, rgba(129, 140, 248, 0.18), transparent 60%);
                    opacity: 0.85;
                }
                .mw-intro-lifecycle__body {
                    position: relative;
                    z-index: 1;
                    background: rgba(248, 250, 252, 0.95);
                    backdrop-filter: blur(22px);
                }
                .mw-intro-lifecycle__grid {
                    gap: clamp(1rem, 2.5vw, 1.8rem);
                }
                .mw-intro-lifecycle__col {
                    position: relative;
                    display: flex;
                    border-radius: 16px;
                    padding: clamp(1rem, 2vw, 1.3rem);
                    background: linear-gradient(180deg, rgba(255, 255, 255, 0.88), rgba(226, 232, 240, 0.68));
                    border: 1px solid rgba(148, 163, 184, 0.25);
                    box-shadow: 0 18px 42px rgba(15, 23, 42, 0.12);
                    overflow: hidden;
                }
                .mw-intro-lifecycle__col::before {
                    content: "";
                    position: absolute;
                    inset: 0;
                    border-radius: inherit;
                    background: linear-gradient(135deg, rgba(59, 130, 246, 0.16), transparent 65%);
                    opacity: 0.8;
                    pointer-events: none;
                }
                .mw-intro-lifecycle__col > * {
                    position: relative;
                    z-index: 1;
                    width: 100%;
                }
                .mw-intro-lifecycle__col:has(> .intro-lifecycle-map) {
                    padding: clamp(0.45rem, 1.6vw, 0.85rem);
                    background: linear-gradient(180deg, rgba(37, 99, 235, 0.16), rgba(14, 116, 144, 0.08));
                    border: 1px solid rgba(37, 99, 235, 0.32);
                    box-shadow: 0 24px 52px rgba(37, 99, 235, 0.22);
                }
                .mw-intro-lifecycle__col:has(> .intro-lifecycle-map)::before {
                    background: radial-gradient(circle at center, rgba(96, 165, 250, 0.38), transparent 70%);
                    opacity: 0.65;
                }
                .intro-lifecycle-sidecar {
                    display: flex;
                    flex-direction: column;
                    gap: 0.85rem;
                    height: 100%;
                }
                .intro-lifecycle-sidecar__eyebrow {
                    font-size: 0.72rem;
                    letter-spacing: 0.18em;
                    text-transform: uppercase;
                    font-weight: 700;
                    color: rgba(15, 23, 42, 0.58);
                }
                .intro-lifecycle-sidecar__title {
                    margin: 0;
                    font-size: clamp(1.2rem, 2.4vw, 1.4rem);
                    font-weight: 700;
                    color: #0f172a;
                }
                .intro-lifecycle-sidecar__body {
                    margin: 0;
                    font-size: 0.97rem;
                    line-height: 1.65;
                    color: rgba(15, 23, 42, 0.78);
                }
                .intro-lifecycle-sidecar__list {
                    margin: 0;
                    padding: 0;
                    list-style: none;
                    display: grid;
                    gap: 0.55rem;
                }
                .intro-lifecycle-sidecar__list li {
                    display: grid;
                    gap: 0.2rem;
                }
                .intro-lifecycle-sidecar__list strong {
                    font-weight: 700;
                    color: #1d4ed8;
                }
                .intro-start-button-slot {
                    margin-top: auto;
                    display: flex;
                    flex-direction: column;
                    gap: 0.6rem;
                }
                .intro-start-button-source {
                    display: none;
                }
                .intro-start-button-source.intro-start-button-source--mounted {
                    display: block;
                }
                .intro-start-button-source--mounted div[data-testid="stButton"] {
                    margin: 0;
                    width: 100%;
                }
                .intro-start-button-source--mounted div[data-testid="stButton"] > button {
                    margin-top: 0.35rem;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    gap: 0.4rem;
                    font-weight: 600;
                    font-size: 0.95rem;
                    border-radius: 999px;
                    padding: 0.68rem 1.6rem;
                    background: linear-gradient(135deg, #2563eb, #4338ca);
                    color: #fff;
                    border: none;
                    box-shadow: 0 18px 36px rgba(37, 99, 235, 0.3);
                    width: 100%;
                    text-align: center;
                    transition: transform 0.18s ease, box-shadow 0.18s ease;
                }
                .intro-start-button-source--mounted div[data-testid="stButton"] > button:hover {
                    transform: translateY(-1px);
                    box-shadow: 0 24px 40px rgba(37, 99, 235, 0.34);
                }
                .intro-start-button-source--mounted div[data-testid="stButton"] > button:focus-visible {
                    outline: 3px solid rgba(59, 130, 246, 0.45);
                    outline-offset: 3px;
                }
                .intro-lifecycle-map {
                    display: flex;
                    flex-direction: column;
                    gap: 0.75rem;
                    height: 100%;
                }
                .intro-lifecycle-map #demai-lifecycle.dlc {
                    flex: 1;
                    width: 100%;
                }
                @media (max-width: 1024px) {
                    .mw-intro-lifecycle {
                        margin: clamp(1rem, 4vw, 1.8rem) auto;
                    }
                }
                @media (max-width: 920px) {
                    .intro-lifecycle-map #demai-lifecycle.dlc {
                        --ring-size: 86vw;
                    }
                }
            </style>
            """
        )

        st.markdown(window_css, unsafe_allow_html=True)

        left_col_html = textwrap.dedent(
            """
            <div class="intro-lifecycle-sidecar" role="complementary" aria-label="Lifecycle guidance">
                <div class="intro-lifecycle-sidecar__eyebrow">What you'll do</div>
                <h5 class="intro-lifecycle-sidecar__title">Build and use an AI spam detector</h5>
                <p class="intro-lifecycle-sidecar__body">
                    In this interactive journey, you’ll build and use your own AI system, an email spam detector. You will experience the key steps of a development lifecycle, step by step: no technical skills are needed.
                    Along the way, you’ll uncover how AI systems learn, make predictions, while applying in practice key concepts from the EU AI Act.
                </p>
                <ul class="intro-lifecycle-sidecar__list">
                    <li>
                        <strong>Discover your journey</strong>
                        To your right, you’ll find an interactive map showing the full lifecycle of your AI system— this is your guide through this hands-on exploration of responsible and transparent AI.
                    </li>
                    <li>
                        <strong>Are you ready to make a machine learn?</strong>
                        Click the button below to start your demAI journey!
                    </li>
                </ul>
                <div id="intro-start-button-slot" class="intro-start-button-slot" aria-live="polite"></div>
            </div>
            """
        ).strip()

        right_col_html = textwrap.dedent(
            f"""
            <div class="intro-lifecycle-map" role="presentation">
                {LIFECYCLE_RING_HTML}
            </div>
            """
        ).strip()

        render_mac_window(
            st,
            title="Start your demAI journey",
            ratios=(0.38, 0.62),
            col_html=[left_col_html, right_col_html],
            id_suffix="intro-lifecycle",
        )

        start_clicked = False

        with st.container():
            st.markdown(
                '<div id="intro-start-button-source" class="intro-start-button-source">',
                unsafe_allow_html=True,
            )
            if next_stage_key:
                start_clicked = st.button(
                    "🚀 Start your machine",
                    key="intro_start_machine",
                    use_container_width=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <script>
            (function () {
                const source = document.getElementById("intro-start-button-source");
                const slot = document.getElementById("intro-start-button-slot");
                if (!source || !slot) return;
                if (!slot.contains(source)) {
                    slot.appendChild(source);
                }
                source.classList.add("intro-start-button-source--mounted");
            })();
            </script>
            """,
            unsafe_allow_html=True,
        )

        if start_clicked and next_stage_key:
            set_active_stage(next_stage_key)

    ai_act_quote_wrapper_open = """
        <style>
            .ai-act-quote-block {
                position: relative;
                display: flex;
                justify-content: center;
                padding: clamp(1.3rem, 2.8vw, 2.1rem);
                border-radius: 1.75rem;
                background: linear-gradient(135deg, rgba(59, 130, 246, 0.16), rgba(236, 72, 153, 0.12));
                border: 1px solid rgba(37, 99, 235, 0.2);
                box-shadow: 0 24px 52px rgba(37, 99, 235, 0.2);
                overflow: hidden;
            }

            .ai-act-quote-block::before {
                content: "";
                position: absolute;
                inset: 0;
                pointer-events: none;
                background: radial-gradient(circle at top right, rgba(59, 130, 246, 0.2), transparent 55%),
                    radial-gradient(circle at bottom left, rgba(236, 72, 153, 0.2), transparent 60%);
                opacity: 0.9;
            }

            .ai-act-quote-block > div[data-testid="stComponent"] {
                width: 100%;
                margin: 0;
                position: relative;
                z-index: 1;
            }
        </style>
        <div class="ai-act-quote-block" role="region" aria-label="From the EU AI Act, Article 3">
    """

def render_overview_stage():
    stage = STAGE_BY_KEY["overview"]

    next_stage_key: Optional[str] = None
    overview_index = STAGE_INDEX.get("overview")
    if overview_index is not None and overview_index < len(STAGES) - 1:
        next_stage_key = STAGES[overview_index + 1].key

    labeled_examples = ss.get("labeled") or []
    dataset_rows = len(labeled_examples)
    dataset_last_built = ss.get("dataset_last_built_at")
    if dataset_last_built:
        try:
            built_dt = datetime.fromisoformat(dataset_last_built)
            dataset_timestamp = f"Dataset build: {built_dt.strftime('%b %d, %H:%M')}"
        except ValueError:
            dataset_timestamp = f"Dataset build: {dataset_last_built}"
    else:
        dataset_timestamp = "Dataset build: pending"

    incoming_records = ss.get("incoming") or []
    incoming_count = len(incoming_records)
    incoming_seed = ss.get("incoming_seed")
    autonomy_label = str(ss.get("autonomy", AUTONOMY_LEVELS[0]))
    adaptiveness_enabled = bool(ss.get("adaptive", False))
    nerd_flag = bool(st.session_state.get("nerd_mode_train") or st.session_state.get("nerd_mode"))
    nerd_enabled = nerd_flag

    with section_surface("section-surface--arch"):
        st.markdown("#### The demAI machine — your system at a glance")
        render_demai_architecture(nerd_mode=nerd_flag, active_stage="overview")

    st.markdown(
        """
        <style>
        .overview-intro-card {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.16), rgba(14, 116, 144, 0.16));
            border-radius: 1.25rem;
            padding: 1.6rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.4rem;
        }
        .overview-intro-card__header {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .overview-intro-card__icon {
            font-size: 1.9rem;
            line-height: 1;
            background: rgba(15, 23, 42, 0.08);
            border-radius: 0.9rem;
            padding: 0.55rem 0.95rem;
            box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
        }
        .overview-intro-card__eyebrow {
            font-size: 0.75rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 700;
            color: rgba(15, 23, 42, 0.6);
            display: inline-block;
            margin-bottom: 0.35rem;
        }
        .overview-intro-card__title {
            margin: 0;
            font-size: 1.5rem;
            font-weight: 700;
            color: #0f172a;
        }
        .overview-intro-card__body {
            margin: 0;
            color: rgba(15, 23, 42, 0.82);
            font-size: 0.98rem;
            line-height: 1.65;
        }
        .overview-checklist {
            margin: 1.1rem 0 0;
            padding: 0;
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }
        .overview-checklist li {
            display: flex;
            gap: 0.6rem;
            align-items: flex-start;
            color: rgba(15, 23, 42, 0.78);
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .overview-checklist li::before {
            content: "✔";
            font-weight: 700;
            color: #1d4ed8;
            margin-top: 0.1rem;
        }
        .overview-intro-actions {
            margin-top: 1.35rem;
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }
        .overview-intro-actions__hint {
            margin: 0;
            font-size: 0.85rem;
            color: rgba(15, 23, 42, 0.65);
        }
        .overview-intro-actions [data-testid="stButton"] > button {
            border-radius: 0.8rem;
            font-weight: 600;
            box-shadow: 0 10px 20px rgba(37, 99, 235, 0.18);
        }
        .overview-subheading {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            margin-bottom: 1.1rem;
        }
        .overview-subheading__eyebrow {
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: rgba(15, 23, 42, 0.6);
            font-weight: 700;
        }
        .overview-subheading h3 {
            margin: 0;
            color: #0f172a;
        }
        .overview-machine-panel {
            background: rgba(255, 255, 255, 0.92);
            border-radius: 1.25rem;
            border: 1px solid rgba(37, 99, 235, 0.22);
            box-shadow: 0 20px 44px rgba(37, 99, 235, 0.12);
            padding: 1.5rem;
        }
        .overview-components {
            gap: 1rem;
        }
        .overview-components.overview-components--compact .callout {
            min-height: 100%;
        }
        .overview-callout {
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 16px 32px rgba(15, 23, 42, 0.08);
            border-radius: 1rem;
        }
        .mission-brief {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(59, 130, 246, 0.08));
            border-radius: 1.35rem;
            border: 1px solid rgba(37, 99, 235, 0.22);
            box-shadow: 0 20px 44px rgba(15, 23, 42, 0.12);
            padding: 1.6rem 1.8rem;
        }
        .mission-brief__header {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        .mission-brief__icon {
            font-size: 1.8rem;
            line-height: 1;
            background: rgba(255, 255, 255, 0.7);
            border-radius: 1rem;
            padding: 0.55rem 0.85rem;
            box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.25);
        }
        .mission-brief__eyebrow {
            font-size: 0.75rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 700;
            color: rgba(15, 23, 42, 0.64);
        }
        .mission-brief__title {
            margin: 0;
            font-size: 1.6rem;
            font-weight: 700;
            color: #0f172a;
        }
        .mission-brief__bridge {
            margin-top: 1rem;
            font-size: 0.95rem;
            line-height: 1.6;
            color: rgba(15, 23, 42, 0.75);
        }
        .mission-brief__grid {
            display: grid;
            grid-template-columns: minmax(0, 1.05fr) minmax(0, 0.95fr);
            gap: 1.8rem;
            margin-top: 1.6rem;
            align-items: stretch;
        }
        .mission-brief__objective {
            display: flex;
            flex-direction: column;
            gap: 1.1rem;
        }
        .mission-brief__objective p {
            margin: 0;
            font-size: 1rem;
            line-height: 1.65;
        }
        .mission-brief__list {
            margin: 0;
            padding-left: 1.1rem;
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }
        .mission-brief__list li {
            font-size: 0.94rem;
            line-height: 1.6;
            color: rgba(15, 23, 42, 0.78);
        }
        .mission-brief__preview {
            display: flex;
        }
        .mission-brief__preview-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 1.15rem;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.16);
            border: 1px solid rgba(37, 99, 235, 0.2);
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
            width: 100%;
        }
        .mission-brief__preview-eyebrow {
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: rgba(37, 99, 235, 0.8);
            font-weight: 700;
        }
        .mission-brief__preview-title {
            margin: 0;
            font-size: 1.1rem;
            color: #1d4ed8;
            font-weight: 600;
        }
        .mission-brief__preview-intro {
            margin: 0;
            font-size: 0.92rem;
            color: rgba(15, 23, 42, 0.7);
        }
        .mission-brief__inbox-list {
            list-style: none;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
        }
        .mission-brief__inbox-item {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        .mission-brief__subject {
            font-weight: 600;
            font-size: 0.96rem;
            color: #0f172a;
        }
        .mission-brief__snippet {
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.68);
            line-height: 1.4;
        }
        .mission-brief__empty {
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.65);
        }
        .mission-brief__preview-note {
            margin: 0;
            font-size: 0.85rem;
            color: rgba(15, 23, 42, 0.6);
        }
        .mission-brief__highlights {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
        }
        .mission-highlight {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            background: rgba(255, 255, 255, 0.85);
            border-radius: 999px;
            padding: 0.55rem 0.95rem;
            font-size: 0.9rem;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.12);
            border: 1px solid rgba(37, 99, 235, 0.18);
        }
        .mission-highlight__icon {
            font-size: 1.1rem;
        }
        .mailbox-preview {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 1.35rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 20px 44px rgba(15, 23, 42, 0.1);
            overflow: hidden;
        }
        .mailbox-preview__header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.1rem 1.4rem;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(59, 130, 246, 0.1));
            border-bottom: 1px solid rgba(37, 99, 235, 0.16);
        }
        .mailbox-preview__header h4 {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 600;
            color: #1e3a8a;
        }
        .mailbox-preview__header span {
            font-size: 0.85rem;
            color: rgba(15, 23, 42, 0.65);
        }
        .mail-rows {
            display: flex;
            flex-direction: column;
        }
        .mail-row {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: start;
            gap: 1rem;
            padding: 1rem 1.4rem;
            border-bottom: 1px solid rgba(15, 23, 42, 0.06);
            background: rgba(248, 250, 252, 0.7);
        }
        .mail-row:nth-child(even) {
            background: rgba(255, 255, 255, 0.92);
        }
        .mail-row__status {
            width: 12px;
            height: 12px;
            border-radius: 999px;
            background: linear-gradient(135deg, #1d4ed8, #2563eb);
            margin-top: 0.35rem;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.18);
        }
        .mail-row__details {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }
        .mail-row__subject {
            margin: 0;
            font-size: 0.98rem;
            font-weight: 600;
            color: #0f172a;
        }
        .mail-row__snippet {
            margin: 0;
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.68);
            line-height: 1.45;
        }
        .mail-row__meta {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 0.4rem;
            font-size: 0.8rem;
            color: rgba(15, 23, 42, 0.6);
        }
        .mail-row__tag {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.14);
            color: #1d4ed8;
            font-weight: 600;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .mail-empty {
            padding: 1.35rem 1.4rem;
            font-size: 0.92rem;
            color: rgba(15, 23, 42, 0.65);
        }
        @media (max-width: 960px) {
            .mission-brief__grid {
                grid-template-columns: 1fr;
                gap: 1.4rem;
            }
            .mission-brief {
                padding: 1.35rem 1.4rem;
            }
            .mission-brief__preview {
                margin-top: 0.4rem;
            }
            .mail-row {
                grid-template-columns: auto 1fr;
            }
            .mail-row__meta {
                align-items: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with section_surface():
        top_left, top_right = st.columns([0.48, 0.52], gap="large")
        with top_left:
            stage_card_html = f"""
                <div class="overview-intro-card">
                    <div class="overview-intro-card__header">
                        <span class="overview-intro-card__icon">{html.escape(stage.icon)}</span>
                        <div>
                            <span class="overview-intro-card__eyebrow">Orientation</span>
                            <h4 class="overview-intro-card__title">{html.escape(stage.title)}</h4>
                        </div>
                    </div>
                    <p class="overview-intro-card__body">
                        You are already inside a <strong>machine-based system</strong>: a Streamlit UI (software)
                        running in the cloud (hardware). Use this control room to <strong>build, evaluate, and operate</strong>
                        a focused email spam detector. We'll guide you through each stage and surface tips when you need them.
                    </p>
                    <ul class="overview-checklist">
                        <li>See how the lifecycle flows from data preparation to deployment.</li>
                        <li>Meet the interface, model, and inbox components you’ll orchestrate.</li>
                        <li>Decide when to enable <em>Nerd Mode</em> for deeper diagnostics.</li>
                    </ul>
                </div>
            """
            st.markdown(stage_card_html, unsafe_allow_html=True)

            nerd_enabled = render_nerd_mode_toggle(
                key="nerd_mode",
                title="Nerd Mode",
                icon="🧠",
                description="Toggle to see technical details and extra functionality. You can enable it at any stage to look under the hood.",
            )

            if next_stage_key:
                st.markdown("<div class='overview-intro-actions'>", unsafe_allow_html=True)
                st.button(
                    "📊 Continue to Prepare data",
                    key="flow_start_machine_cta",
                    type="primary",
                    on_click=set_active_stage,
                    args=(next_stage_key,),
                    use_container_width=True,
                )
                st.markdown(
                    "<p class='overview-intro-actions__hint'>Next you’ll curate examples so the model learns the right patterns.</p>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)

        with top_right:
            machine_panel_html = """
                <div class="overview-machine-panel">
                    <div class="overview-subheading">
                        <span class="overview-subheading__eyebrow">Console map</span>
                        <h3>Meet the machine</h3>
                    </div>
                    <div class="callout-grid overview-components overview-components--compact">
                        <div class="callout callout--info overview-callout">
                            <h5>🖥️ User interface</h5>
                            <p>The control panel for your AI system. Step through <strong>Prepare data</strong>, <strong>Train</strong>, <strong>Evaluate</strong>, and <strong>Use</strong>. Tooltips and short explainers guide you; <em>Nerd Mode</em> reveals more.</p>
                        </div>
                        <div class="callout callout--info overview-callout">
                            <h5>🧠 AI model (how it learns &amp; infers)</h5>
                            <p>The model learns from <strong>labeled examples</strong> you provide to tell <strong>Spam</strong> from <strong>Safe</strong>. For each new email it produces a <strong>spam score</strong> (P(spam)); your <strong>threshold</strong> turns that score into a recommendation or decision.</p>
                        </div>
                        <div class="callout callout--info overview-callout">
                            <h5>📥 Inbox interface</h5>
                            <p>A simulated inbox feeds emails into the system. Preview items, process a batch or review one by one, and optionally enable <strong>adaptiveness</strong> so your confirmations/corrections help the model improve.</p>
                        </div>
                    </div>
                </div>
            """

            st.markdown(machine_panel_html, unsafe_allow_html=True)

    dataset_value = f"{dataset_rows:,} labeled emails ready" if dataset_rows else "Starter dataset loading"
    incoming_value = f"{incoming_count:,} emails queued"
    if incoming_seed is not None:
        incoming_value = f"{incoming_value} • seed {incoming_seed}"

    adaptiveness_value = (
        "Adaptiveness on — confirmations feed future training"
        if adaptiveness_enabled
        else "Adaptiveness off — corrections stay manual"
    )

    preview_records: List[Dict[str, Any]] = []
    if incoming_records:
        df_incoming = pd.DataFrame(incoming_records)
        preview_records = df_incoming.head(5).to_dict("records")

    def _format_snippet(text: Optional[str], *, limit: int = 110) -> str:
        snippet = (text or "").strip()
        snippet = re.sub(r"\s+", " ", snippet)
        if len(snippet) > limit:
            snippet = snippet[: limit - 1].rstrip() + "…"
        return snippet

    inbox_rows_html = []
    for record in preview_records:
        subject = html.escape(record.get("title", "Untitled email"))
        snippet = html.escape(_format_snippet(record.get("body")))
        inbox_rows_html.append(
            textwrap.dedent(
                """
                <div class="mail-row">
                    <div class="mail-row__status"></div>
                    <div class="mail-row__details">
                        <p class="mail-row__subject">{subject}</p>
                        <p class="mail-row__snippet">{snippet}</p>
                    </div>
                    <div class="mail-row__meta">
                        <span class="mail-row__tag">Queued</span>
                    </div>
                </div>
                """
            ).format(subject=subject, snippet=snippet).strip()
        )

    if not inbox_rows_html:
        inbox_rows_html.append(
            "<div class='mail-empty'>Inbox stream is empty. Once new emails arrive you’ll see them queue here for review.</div>"
        )

    mailbox_html = textwrap.dedent(
        """
        <div class="mailbox-preview">
            <div class="mailbox-preview__header">
                <h4>Live inbox preview</h4>
                <span>First {count} messages waiting for triage</span>
            </div>
            <div class="mail-rows">{rows}</div>
        </div>
        """
    ).format(count=len(preview_records) or 0, rows="".join(inbox_rows_html)).strip()

    mission_html = textwrap.dedent(
        f"""
        <div class="mission-brief">
            <div class="mission-brief__header">
                <span class="mission-brief__icon">🎯</span>
                <div>
                    <span class="mission-brief__eyebrow">Mission briefing</span>
                    <h3 class="mission-brief__title">Your mission</h3>
                </div>
            </div>
            <p class="mission-brief__bridge">You’re stepping into the control room of an email triage machine. The inbox snapshot on the right matches the live preview you’ll work from in a moment.</p>
            <div class="mission-brief__grid">
                <div class="mission-brief__objective">
                    <p>Keep unwanted email out while letting the important messages through. You’ll steer the controls, set the operating thresholds, and verify the system’s choices.</p>
                    <ul class="mission-brief__list">
                        <li>Scan the inbox feed and spot risky patterns early.</li>
                        <li>Decide how strict the spam filter should be and when autonomy applies.</li>
                        <li>Confirm or correct decisions so the system learns your judgement.</li>
                    </ul>
                </div>
                <div class="mission-brief__preview">
                    <div class="mission-brief__preview-card mission-brief__preview-card--mailbox">
                        <span class="mission-brief__preview-eyebrow">Inbox stream</span>
                        {mailbox_html}
                    </div>
                </div>
            </div>
        </div>
        """
    ).strip()

    with section_surface():
        st.markdown(mission_html, unsafe_allow_html=True)


    if nerd_enabled:
        with section_surface():
            st.markdown(
                """
                <div class="overview-subheading">
                    <span class="overview-subheading__eyebrow">Deep dive</span>
                    <h3>🔬 Nerd Mode — technical details</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            nerd_details_html = """
            <div class="callout-grid">
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">🖥️</div>
                    <div class="callout-body">
                        <h5>User interface (software &amp; runtime)</h5>
                        <ul>
                            <li>You’re using a simple Streamlit (Python) web app running in the cloud.</li>
                            <li>The app remembers your session choices — data, model, threshold, autonomy — so you can move around without losing progress.</li>
                            <li>Short tips and popovers appear where helpful; toggle <em>Nerd Mode</em> any time to dive deeper.</li>
                        </ul>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">🧠</div>
                    <div class="callout-body">
                        <h5>AI model (how it works, without the math)</h5>
                        <ul>
                            <li><strong>What’s inside:</strong>
                                <ul>
                                    <li>A MiniLM sentence-transformer turns each email’s title + body into meaning-rich numbers.</li>
                                    <li>A Logistic Regression layer draws the boundary between Spam and Safe.</li>
                                </ul>
                            </li>
                            <li><strong>How it learns (training):</strong>
                                <ul>
                                    <li>You supply labeled examples (Spam/Safe).</li>
                                    <li>The app trains on most of them and holds out a slice for fair evaluation later.</li>
                                    <li>Training is repeatable via a fixed random seed; class weights rebalance skewed datasets.</li>
                                </ul>
                            </li>
                            <li><strong>How it predicts (inference):</strong>
                                <ul>
                                    <li>For a new email, the model outputs a spam score between 0 and 1.</li>
                                    <li>A threshold converts that score into action: below = Safe, above = Spam.</li>
                                    <li>In <em>Evaluate</em>, tune the threshold with presets such as Balanced, Protect inbox, or Catch spam.</li>
                                </ul>
                            </li>
                            <li><strong>Why it decided that (interpretability):</strong>
                                <ul>
                                    <li>View similar training emails and simple clues (urgent tone, suspicious links, ALL-CAPS bursts).</li>
                                    <li>Enable numeric signals to see which features nudged the call toward Spam or Safe.</li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">📥</div>
                    <div class="callout-body">
                        <h5>Inbox interface (your data in and out)</h5>
                        <ul>
                            <li>The app manages incoming (unlabeled) emails, labeled training emails, and the routed Inbox/Spam buckets.</li>
                            <li>Process emails in small batches (e.g., the first 10) or handle them one by one.</li>
                            <li><strong>Autonomy levels:</strong>
                                <ul>
                                    <li>Moderate (default): the system recommends a route; you decide.</li>
                                    <li>High autonomy: the system routes automatically using your chosen threshold.</li>
                                </ul>
                            </li>
                            <li><strong>Adaptiveness (optional):</strong> confirm or correct outcomes to add feedback, then retrain to personalize the model.</li>
                        </ul>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">🛡️</div>
                    <div class="callout-body">
                        <h5>Governance &amp; transparency</h5>
                        <ul>
                            <li>A model card records purpose, data summary, metrics, chosen threshold, autonomy, adaptiveness, seed, and timestamps.</li>
                            <li>We track risks: false positives (legit to Spam) and false negatives (Spam to Inbox).</li>
                            <li>An optional audit log lists batch actions, corrections, and retraining events for the session.</li>
                        </ul>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">🧩</div>
                    <div class="callout-body">
                        <h5>Packages (what powers this)</h5>
                        <p>streamlit (UI), pandas/numpy (data), scikit-learn (training &amp; evaluation), optional sentence-transformers + torch/transformers (embeddings), matplotlib (plots)</p>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">📏</div>
                    <div class="callout-body">
                        <h5>Limits (demo scope)</h5>
                        <ul>
                            <li>Uses synthetic or curated text — there’s no live mailbox connection.</li>
                            <li>Designed for learning clarity rather than production-grade email security.</li>
                        </ul>
                    </div>
                </div>
            </div>
            """
            st.markdown(nerd_details_html, unsafe_allow_html=True)



def render_data_stage():

    stage = STAGE_BY_KEY["data"]

    current_summary = compute_dataset_summary(ss["labeled"])
    ss["dataset_summary"] = current_summary

    flash_queue = ss.pop("data_stage_flash_queue", [])
    for flash in flash_queue:
        if not isinstance(flash, dict):
            continue
        message = str(flash.get("message", "")).strip()
        if not message:
            continue
        level = flash.get("level", "info")
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        elif level == "error":
            st.error(message)
        else:
            st.info(message)

    with section_surface():
        render_eu_ai_quote("An AI system “infers, from the input it receives…”.")

    st.markdown(
        """
        <style>
        .prepare-intro-card {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.16), rgba(236, 72, 153, 0.12));
            border-radius: 1.25rem;
            padding: 1.5rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            margin-bottom: 50px;
            margin-top: 20px;
        }
        .prepare-intro-card__header {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 0.9rem;
        }
        .prepare-intro-card__icon {
            font-size: 1.75rem;
            line-height: 1;
            background: rgba(15, 23, 42, 0.08);
            border-radius: 0.9rem;
            padding: 0.55rem 0.9375rem;
            box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
        }
        .prepare-intro-card__eyebrow {
            font-size: 0.75rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 700;
            color: rgba(15, 23, 42, 0.55);
            display: inline-block;
        }
        .prepare-intro-card__title {
            margin: 0;
            padding: 0;
            font-size: 1.4rem;
            font-weight: 700;
            color: #0f172a;
        }
        .prepare-intro-card__body {
            margin: 0;
            color: rgba(15, 23, 42, 0.8);
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .dataset-health-card {
            border-radius: 1.1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 1.35rem;
            background: rgba(255, 255, 255, 0.88);
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.1);
            backdrop-filter: blur(6px);
        }
        .dataset-health-panel {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        .dataset-health-panel__status {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
        }
        .dataset-health-panel__status-copy {
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
        }
        .dataset-health-panel__status-copy h5 {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
        }
        .dataset-health-panel__status-copy small {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: rgba(15, 23, 42, 0.55);
            font-weight: 700;
        }
        .dataset-health-status {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            padding: 0.4rem 1rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            line-height: 1;
        }
        .dataset-health-status__dot {
            width: 14px;
            height: 14px;
            border-radius: 50%;
            box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.85);
        }
        .dataset-health-status--good {
            background: rgba(34, 197, 94, 0.18);
            color: #15803d;
        }
        .dataset-health-status--good .dataset-health-status__dot {
            background: #22c55e;
        }
        .dataset-health-status--warn {
            background: rgba(234, 179, 8, 0.2);
            color: #b45309;
        }
        .dataset-health-status--warn .dataset-health-status__dot {
            background: #fbbf24;
        }
        .dataset-health-status--risk {
            background: rgba(248, 113, 113, 0.2);
            color: #b91c1c;
        }
        .dataset-health-status--risk .dataset-health-status__dot {
            background: #f87171;
        }
        .dataset-health-status--neutral {
            background: rgba(148, 163, 184, 0.22);
            color: #1f2937;
        }
        .dataset-health-status--neutral .dataset-health-status__dot {
            background: rgba(148, 163, 184, 0.9);
        }
        .dataset-health-panel__row {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .dataset-health-panel__row--bar {
            margin-top: 0.35rem;
        }
        .dataset-health-panel__row--meta {
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 0.75rem;
        }
        .dataset-health-panel__meta-primary {
            display: inline-flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            font-size: 0.85rem;
            font-weight: 600;
            color: rgba(15, 23, 42, 0.75);
        }
        .dataset-health-panel__lint-placeholder {
            font-size: 0.8rem;
            color: rgba(15, 23, 42, 0.55);
        }
        .dataset-health-panel__bar {
            flex: 1;
            display: flex;
            height: 10px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.08);
        }
        .dataset-health-panel__bar span {
            display: block;
            height: 100%;
        }
        .dataset-health-panel__bar-spam {
            background: linear-gradient(90deg, #fb7185 0%, #f43f5e 100%);
        }
        .dataset-health-panel__bar-safe {
            background: linear-gradient(90deg, #38bdf8 0%, #0ea5e9 100%);
        }
        .dataset-delta-panel {
            position: sticky;
            top: 5.5rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            padding: 1.25rem;
            border-radius: 1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
            backdrop-filter: blur(6px);
            margin-top: 50px;
        }
        .dataset-delta-panel h5 {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 700;
        }
        .dataset-delta-panel__items {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        .dataset-delta-panel__item {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            font-size: 0.95rem;
        }
        .dataset-delta-panel__item span:first-child {
            color: #0f172a;
            font-weight: 600;
        }
        .delta-arrow {
            font-weight: 700;
        }
        .delta-arrow--up {
            color: #16a34a;
        }
        .delta-arrow--down {
            color: #dc2626;
        }
        .dataset-delta-panel__hint {
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.75);
            border-top: 1px solid rgba(15, 23, 42, 0.08);
            padding-top: 0.75rem;
        }
        .dataset-delta-panel__story {
            font-size: 0.85rem;
            color: rgba(15, 23, 42, 0.7);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    delta_text = ""
    if ss.get("dataset_compare_delta"):
        delta_text = dataset_delta_story(ss["dataset_compare_delta"])
    if not delta_text and ss.get("last_dataset_delta_story"):
        delta_text = ss["last_dataset_delta_story"]
    if not delta_text:
        delta_text = explain_config_change(ss.get("dataset_config", DEFAULT_DATASET_CONFIG))

    def _generate_preview_from_config(config: DatasetConfig) -> Dict[str, Any]:
        dataset_rows = build_dataset_from_config(config)
        preview_summary = compute_dataset_summary(dataset_rows)
        lint_counts = lint_dataset(dataset_rows)

        manual_df = pd.DataFrame(dataset_rows[: min(len(dataset_rows), 200)])
        if not manual_df.empty:
            manual_df.insert(0, "include", True)

        ss["dataset_preview"] = dataset_rows
        ss["dataset_preview_config"] = config
        ss["dataset_preview_summary"] = preview_summary
        ss["dataset_preview_lint"] = lint_counts
        ss["dataset_manual_queue"] = manual_df
        ss["dataset_compare_delta"] = dataset_summary_delta(current_summary, preview_summary)
        ss["last_dataset_delta_story"] = dataset_delta_story(ss["dataset_compare_delta"])
        ss["dataset_has_generated_once"] = True

        st.success("Dataset generated — scroll to **Review** and curate data before committing.")
        explanation = explain_config_change(config, ss.get("dataset_config", DEFAULT_DATASET_CONFIG))


        return preview_summary

    def _build_compare_panel_html(
        base_summary: Optional[Dict[str, Any]],
        target_summary: Optional[Dict[str, Any]],
        delta_summary: Optional[Dict[str, Any]],
        delta_story_text: str,
    ) -> str:
        panel_items: list[tuple[str, str, str, str]] = []
        spam_share_delta_pp: Optional[float] = None

        def _add_panel_item(
            label: str,
            value: Any,
            *,
            unit: str = "",
            decimals: Optional[int] = None,
        ) -> None:
            if value is None:
                return
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return
            if abs(numeric_value) < 1e-6:
                return
            arrow = "▲" if numeric_value > 0 else "▼"
            arrow_class = "delta-arrow--up" if numeric_value > 0 else "delta-arrow--down"
            abs_value = abs(numeric_value)
            if decimals is not None:
                value_str = f"{abs_value:.{decimals}f}".rstrip("0").rstrip(".")
            else:
                if abs(abs_value - round(abs_value)) < 1e-6:
                    value_str = f"{int(round(abs_value))}"
                else:
                    value_str = f"{abs_value:.2f}".rstrip("0").rstrip(".")
            if unit:
                value_str = f"{value_str}{unit}"
            panel_items.append((label, arrow, arrow_class, value_str))

        if base_summary and target_summary:
            try:
                base_ratio = float(base_summary.get("spam_ratio") or 0.0)
                target_ratio = float(target_summary.get("spam_ratio") or 0.0)
                spam_share_delta_pp = (target_ratio - base_ratio) * 100.0
            except (TypeError, ValueError):
                spam_share_delta_pp = None
            if spam_share_delta_pp is not None and abs(spam_share_delta_pp) >= 0.1:
                _add_panel_item("Spam share", spam_share_delta_pp, unit="pp", decimals=1)

        if delta_summary:
            _add_panel_item("Examples", delta_summary.get("total"))
            _add_panel_item(
                "Avg suspicious links",
                delta_summary.get("avg_susp_links"),
                decimals=2,
            )
            _add_panel_item("Suspicious TLD hits", delta_summary.get("suspicious_tlds"))
            _add_panel_item("Money cues", delta_summary.get("money_mentions"))
            _add_panel_item("Attachment lures", delta_summary.get("attachment_lures"))

        effect_hint = ""
        if spam_share_delta_pp is not None and abs(spam_share_delta_pp) >= 0.1:
            if spam_share_delta_pp > 0:
                effect_hint = (
                    "Higher spam share → recall ↑, precision may ↓; adjust threshold later in Evaluate."
                )
            else:
                effect_hint = (
                    "Lower spam share → precision ↑, recall may ↓; consider rebalancing spam examples before training."
                )
        elif delta_summary:
            link_delta = float(delta_summary.get("avg_susp_links") or 0.0)
            tld_delta = float(delta_summary.get("suspicious_tlds") or 0.0)
            money_delta = float(delta_summary.get("money_mentions") or 0.0)
            attachment_delta = float(delta_summary.get("attachment_lures") or 0.0)
            if link_delta > 0:
                effect_hint = "More suspicious links → phishing recall should improve via URL cues."
            elif link_delta < 0:
                effect_hint = "Fewer suspicious links → URL-heavy spam might slip by; monitor precision/recall."
            elif tld_delta > 0:
                effect_hint = "Suspicious TLD hits increased — domain heuristics strengthen spam recall."
            elif tld_delta < 0:
                effect_hint = "Suspicious TLD hits dropped — rely more on text patterns and validate in Evaluate."
            elif money_delta > 0:
                effect_hint = "Money cues rose — expect better coverage on payment scams."
            elif money_delta < 0:
                effect_hint = "Money cues fell — finance-themed recall could dip."
            elif attachment_delta > 0:
                effect_hint = "Attachment lures increased — the model leans on risky file signals."
            elif attachment_delta < 0:
                effect_hint = "Attachment lures decreased — detection may hinge on text clues."

        if not effect_hint:
            if panel_items:
                effect_hint = "Changes logged — move to Evaluate to measure the impact."
            else:
                effect_hint = "No changes yet—adjust and preview."

        panel_html = ["<div class='dataset-delta-panel'>", "<h5>Compare datasets</h5>"]
        if panel_items:
            panel_html.append("<div class='dataset-delta-panel__items'>")
            for label, arrow, arrow_class, value_str in panel_items:
                panel_html.append(
                    "<div class='dataset-delta-panel__item'><span>{label}</span>"
                    "<span class='delta-arrow {cls}'>{arrow}{value}</span></div>".format(
                        label=html.escape(label),
                        cls=arrow_class,
                        arrow=html.escape(arrow),
                        value=html.escape(value_str),
                    )
                )
            panel_html.append("</div>")
        else:
            panel_html.append(
                "<p class='dataset-delta-panel__story'>After you generate a dataset, you can tweak the configuration and preview here how these impact your data.</p>"
            )
            effect_hint = ""

        if effect_hint:
            panel_html.append(
                "<div class='dataset-delta-panel__hint'>{}</div>".format(
                    html.escape(effect_hint)
                )
            )
        if delta_story_text:
            panel_html.append(
                "<div class='dataset-delta-panel__story'>{}</div>".format(
                    html.escape(delta_story_text)
                )
            )
        panel_html.append("</div>")
        return "".join(panel_html)

    nerd_mode_data_enabled = bool(ss.get("nerd_mode_data"))
    delta_summary: Optional[Dict[str, Any]] = ss.get("dataset_compare_delta")
    base_summary_for_delta: Optional[Dict[str, Any]] = None
    target_summary_for_delta: Optional[Dict[str, Any]] = None
    compare_panel_html = ""

    with section_surface():
        info_col, builder_col = st.columns([0.45, 0.55], gap="large")
        with info_col:
            st.markdown(
                """
                <div class="prepare-intro-card">
                    <div class="prepare-intro-card__header">
                        <span class="prepare-intro-card__icon" style="
    padding-bottom: 15px;">🧪</span>
                        <div>
                            <span class="prepare-intro-card__eyebrow">Stage 1</span>
                            <h4 class="prepare-intro-card__title" style="padding-top: 0px;">Prepare data</h4>
                        </div>
                    </div>
                    <p class="prepare-intro-card__body">
                        Your AI system must learn how to distinguish a safe email from spam. The first step is to prepare a dataset representing what spam and safe emails look like. Use the dataset builder to generate a synthetic dataset, then review its health score and recommendations. Toggle Nerd Mode for advanced configuration and diagnostic controls when you need them.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            nerd_mode_data_enabled = render_nerd_mode_toggle(
                key="nerd_mode_data",
                title="Nerd Mode — advanced dataset controls",
                description="Expose feature prevalence, randomness, diagnostics, and CSV import when you need them.",
            )

        with builder_col:
            st.markdown("### Dataset builder")

            if ss.get("dataset_preview_summary"):
                base_summary_for_delta = current_summary
                target_summary_for_delta = ss["dataset_preview_summary"]
            elif ss.get("previous_dataset_summary") and delta_summary:
                base_summary_for_delta = ss.get("previous_dataset_summary")
                target_summary_for_delta = ss.get("dataset_summary", current_summary)
            else:
                base_summary_for_delta = compute_dataset_summary(STARTER_LABELED)
                target_summary_for_delta = current_summary
                if delta_summary is None:
                    delta_summary = dataset_summary_delta(
                        base_summary_for_delta, target_summary_for_delta
                    )

            preview_clicked = False
            reset_clicked = False

            cfg = ss.get("dataset_config", DEFAULT_DATASET_CONFIG)
            _set_advanced_knob_state(cfg)

            spam_ratio_default = float(cfg.get("spam_ratio", 0.5))
            spam_share_default = int(round(spam_ratio_default * 100))
            spam_share_default = min(max(spam_share_default, 20), 80)
            if spam_share_default % 5 != 0:
                spam_share_default = int(5 * round(spam_share_default / 5))

            with st.form("dataset_builder_form"):
                dataset_size = st.radio(
                    "Dataset size",
                    options=[100, 300, 500],
                    index=[100, 300, 500].index(int(cfg.get("n_total", 500)))
                    if int(cfg.get("n_total", 500)) in [100, 300, 500]
                    else 2,
                    help="Preset sizes illustrate how data volume influences learning (guarded ≤500).",
                )
                spam_share_pct = st.slider(
                    "Spam share",
                    min_value=20,
                    max_value=80,
                    value=spam_share_default,
                    step=5,
                    help="Adjust prevalence to explore bias/recall trade-offs.",
                )
                edge_cases = st.slider(
                    "Edge cases",
                    min_value=0,
                    max_value=len(EDGE_CASE_TEMPLATES),
                    value=int(cfg.get("edge_cases", 0)),
                    help="Inject similar-looking spam/safe pairs to stress the model.",
                )

                if nerd_mode_data_enabled:
                    with st.expander("Advanced knobs", expanded=True):
                        st.caption(
                            "Fine-tune suspicious links, domains, tone, attachments, randomness, and demos before generating a preview."
                        )
                        attachment_keys = list(ATTACHMENT_MIX_PRESETS.keys())
                        adv_col_a, adv_col_b = st.columns(2, gap="large")
                        with adv_col_a:
                            st.slider(
                                "Suspicious links per spam email",
                                min_value=0,
                                max_value=2,
                                value=int(st.session_state.get("adv_links_level", 1)),
                                help="Controls how many sketchy URLs appear in spam examples (0–2).",
                                key="adv_links_level",
                            )
                            st.select_slider(
                                "Suspicious TLD frequency",
                                options=["low", "med", "high"],
                                value=str(
                                    st.session_state.get(
                                        "adv_tld_level", cfg.get("susp_tld_level", "med")
                                    )
                                ),
                                key="adv_tld_level",
                            )
                            st.select_slider(
                                "ALL-CAPS / urgency intensity",
                                options=["low", "med", "high"],
                                value=str(
                                    st.session_state.get(
                                        "adv_caps_level", cfg.get("caps_intensity", "med")
                                    )
                                ),
                                key="adv_caps_level",
                            )
                        with adv_col_b:
                            st.select_slider(
                                "Money symbols & urgency",
                                options=["off", "low", "high"],
                                value=str(
                                    st.session_state.get(
                                        "adv_money_level", cfg.get("money_urgency", "low")
                                    )
                                ),
                                key="adv_money_level",
                            )
                            st.selectbox(
                                "Attachment lure mix",
                                options=attachment_keys,
                                index=attachment_keys.index(
                                    st.session_state.get("adv_attachment_choice", "Balanced")
                                )
                                if st.session_state.get("adv_attachment_choice", "Balanced") in attachment_keys
                                else 1,
                                help="Choose how often risky attachments (HTML/ZIP/XLSM/EXE) appear vs. safer PDFs.",
                                key="adv_attachment_choice",
                            )
                            st.slider(
                                "Label noise (%)",
                                min_value=0.0,
                                max_value=5.0,
                                step=1.0,
                                value=float(
                                    st.session_state.get(
                                        "adv_label_noise_pct", cfg.get("label_noise_pct", 0.0)
                                    )
                                ),
                                key="adv_label_noise_pct",
                            )
                            st.number_input(
                                "Random seed",
                                min_value=0,
                                value=int(st.session_state.get("adv_seed", cfg.get("seed", 42))),
                                key="adv_seed",
                                help="Keep this fixed for reproducibility.",
                            )
                            st.toggle(
                                "Data poisoning demo (synthetic)",
                                value=bool(
                                    st.session_state.get(
                                        "adv_poison_demo", cfg.get("poison_demo", False)
                                    )
                                ),
                                key="adv_poison_demo",
                                help="Adds a tiny malicious distribution shift labeled as safe to show metric degradation.",
                            )

                st.markdown("<div class='cta-sticky'>", unsafe_allow_html=True)
                btn_primary, btn_secondary = st.columns([1, 1])
                with btn_primary:
                    preview_clicked = st.form_submit_button("Generate preview", type="primary")
                with btn_secondary:
                    reset_clicked = st.form_submit_button("Reset to baseline", type="secondary")
                st.markdown("</div>", unsafe_allow_html=True)

            spam_ratio = float(spam_share_pct) / 100.0

            if reset_clicked:
                ss["labeled"] = starter_dataset_copy()
                ss["dataset_config"] = DEFAULT_DATASET_CONFIG.copy()
                baseline_summary = compute_dataset_summary(ss["labeled"])
                ss["dataset_summary"] = baseline_summary
                ss["previous_dataset_summary"] = None
                ss["dataset_compare_delta"] = None
                ss["last_dataset_delta_story"] = None
                ss["active_dataset_snapshot"] = None
                ss["dataset_snapshot_name"] = ""
                ss["dataset_last_built_at"] = datetime.now().isoformat(timespec="seconds")
                ss["dataset_preview"] = None
                ss["dataset_preview_config"] = None
                ss["dataset_preview_summary"] = None
                ss["dataset_preview_lint"] = None
                ss["dataset_manual_queue"] = None
                ss["dataset_controls_open"] = False
                _push_data_stage_flash(
                    "success", f"Dataset reset to starter baseline ({len(STARTER_LABELED)} rows)."
                )
                _set_advanced_knob_state(ss["dataset_config"], force=True)
                if ss.get("_needs_advanced_knob_rerun"):
                    _streamlit_rerun()
                current_summary = baseline_summary
                delta_summary = ss.get("dataset_compare_delta")
                delta_text = explain_config_change(ss.get("dataset_config", DEFAULT_DATASET_CONFIG))
                base_summary_for_delta = compute_dataset_summary(STARTER_LABELED)
                target_summary_for_delta = current_summary

        preview_summary_local: Optional[Dict[str, Any]] = None
        if preview_clicked:
            attachment_choice = st.session_state.get(
                "adv_attachment_choice",
                next(
                    (
                        name
                        for name, mix in ATTACHMENT_MIX_PRESETS.items()
                        if mix == cfg.get("attachments_mix", DEFAULT_ATTACHMENT_MIX)
                    ),
                    "Balanced",
                ),
            )
            attachment_mix = ATTACHMENT_MIX_PRESETS.get(attachment_choice, DEFAULT_ATTACHMENT_MIX).copy()
            links_level_value = int(
                st.session_state.get(
                    "adv_links_level",
                    int(str(cfg.get("susp_link_level", "1"))),
                )
            )
            tld_level_value = str(
                st.session_state.get("adv_tld_level", cfg.get("susp_tld_level", "med"))
            )
            caps_level_value = str(
                st.session_state.get("adv_caps_level", cfg.get("caps_intensity", "med"))
            )
            money_level_value = str(
                st.session_state.get("adv_money_level", cfg.get("money_urgency", "low"))
            )
            noise_pct_value = float(
                st.session_state.get("adv_label_noise_pct", float(cfg.get("label_noise_pct", 0.0)))
            )
            seed_value = int(st.session_state.get("adv_seed", int(cfg.get("seed", 42))))
            poison_demo_value = bool(
                st.session_state.get("adv_poison_demo", bool(cfg.get("poison_demo", False)))
            )
            config: DatasetConfig = {
                "seed": int(seed_value),
                "n_total": int(dataset_size),
                "spam_ratio": float(spam_ratio),
                "susp_link_level": str(int(links_level_value)),
                "susp_tld_level": tld_level_value,
                "caps_intensity": caps_level_value,
                "money_urgency": money_level_value,
                "attachments_mix": attachment_mix,
                "edge_cases": int(edge_cases),
                "label_noise_pct": float(noise_pct_value),
                "poison_demo": bool(poison_demo_value),
            }
            preview_summary_local = _generate_preview_from_config(config)
            delta_summary = ss.get("dataset_compare_delta")
            if delta_summary:
                delta_text = dataset_delta_story(delta_summary)
            base_summary_for_delta = current_summary
            target_summary_for_delta = preview_summary_local
        compare_panel_html = _build_compare_panel_html(
            base_summary_for_delta,
            target_summary_for_delta,
            delta_summary,
            delta_text,
        )

    preview_summary_for_health = ss.get("dataset_preview_summary")
    lint_counts_preview = (
        ss.get("dataset_preview_lint") if preview_summary_for_health is not None else None
    )
    spam_pct: Optional[float] = None
    total_rows: Optional[float] = None
    lint_label = ""
    badge_text = ""
    lint_chip_html = ""
    dataset_health_available = False
    dataset_generated_once = bool(ss.get("dataset_has_generated_once"))

    if preview_summary_for_health is not None:
        health = _evaluate_dataset_health(preview_summary_for_health, lint_counts_preview)
        spam_pct = health["spam_pct"]
        total_rows = health["total_rows"]
        lint_label = health["lint_label"]
        badge_text = health["badge_text"]
        lint_flags_total = health["lint_flags"]
        if lint_label == "Unknown":
            lint_icon = "ℹ️"
        elif lint_flags_total:
            lint_icon = "⚠️"
        else:
            lint_icon = "🛡️"
        lint_chip_html = (
            "<span class='lint-chip'><span class='lint-chip__icon'>{icon}</span>"
            "<span class='lint-chip__text'>Personal data alert: {label}</span></span>"
        ).format(icon=lint_icon, label=html.escape(lint_label or "Unknown"))
        dataset_health_available = True

    if dataset_generated_once or preview_summary_for_health is not None:
        with section_surface():
            health_col, compare_col = st.columns([1.4, 1], gap="large")
            with health_col:
                st.markdown(
                    "#### Dataset health",
                    help="Quick pulse on balance, volume, and lint signals for the generated preview.",
                )
                if dataset_health_available:
                    if spam_pct is not None and total_rows is not None:
                        safe_pct = max(0.0, min(100.0, 100.0 - spam_pct))
                        total_display = (
                            int(total_rows) if isinstance(total_rows, (int, float)) else total_rows
                        )
                        status_text = badge_text or "Awaiting checks"
                        status_class = "neutral"
                        if badge_text:
                            lowered = badge_text.lower()
                            if "good" in lowered:
                                status_class = "good"
                            elif "needs work" in lowered:
                                status_class = "warn"
                            elif "risky" in lowered:
                                status_class = "risk"
                        lint_section = (
                            f"<div class='indicator-chip-row'>{lint_chip_html}</div>"
                            if lint_chip_html
                            else "<small class='dataset-health-panel__lint-placeholder'>Personal data results pending.</small>"
                        )
                        st.markdown(
                            """
                            <div class="dataset-health-card">
                                <div class="dataset-health-panel">
                                    <div class="dataset-health-panel__status">
                                        <div class="dataset-health-panel__status-copy">
                                            <small>Preview summary</small>
                                            <h5>Balance &amp; lint</h5>
                                        </div>
                                        <div class="dataset-health-status dataset-health-status--{status_class}">
                                            <span class="dataset-health-status__dot"></span>
                                            <span>{status_text}</span>
                                        </div>
                                    </div>
                                    <div class="dataset-health-panel__row dataset-health-panel__row--bar">
                                        <div class="dataset-health-panel__bar">
                                            <span class="dataset-health-panel__bar-spam" style="width: {spam_width}%"></span>
                                            <span class="dataset-health-panel__bar-safe" style="width: {safe_width}%"></span>
                                        </div>
                                    </div>
                                    <div class="dataset-health-panel__row dataset-health-panel__row--meta">
                                        <div class="dataset-health-panel__meta-primary">
                                            <span>Spam {spam_pct:.0f}%</span>
                                            <span>Safe {safe_pct:.0f}%</span>
                                            <span>Rows {total_rows}</span>
                                        </div>
                                        {lint_section}
                                    </div>
                                </div>
                            </div>
                            """.format(
                                status_class=status_class,
                                status_text=html.escape(status_text),
                                spam_width=f"{spam_pct:.1f}",
                                safe_width=f"{safe_pct:.1f}",
                                spam_pct=spam_pct,
                                safe_pct=safe_pct,
                                total_rows=html.escape(str(total_display)),
                                lint_section=lint_section,
                            ),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption("Dataset summary not available.")
                else:
                    st.caption("Generate a preview to evaluate dataset health.")
            with compare_col:
                st.markdown(compare_panel_html, unsafe_allow_html=True)

    if ss.get("dataset_preview"):
        # ===== PII Cleanup (mini-game) ================================================
        _ensure_pii_state()
        preview_rows = ss["dataset_preview"]
        detailed_hits = lint_dataset_detailed(preview_rows)
        counts = summarize_pii_counts(detailed_hits)
        ss["pii_hits_map"] = detailed_hits
        flagged_ids = sorted(detailed_hits.keys())
        ss["pii_queue"] = flagged_ids
        ss["pii_total_flagged"] = len(flagged_ids)

         
        with section_surface():
            banner_clicked = render_pii_cleanup_banner(counts)
            if banner_clicked:
                ss["pii_open"] = True

        if ss.get("pii_open"):
            with section_surface():
                st.markdown("### 🔐 Personal Data Cleanup")
                st.markdown(PII_INDICATOR_STYLE, unsafe_allow_html=True)
                remaining_to_clean = len(flagged_ids)
                indicator_values = [
                    ("Score", int(ss.get("pii_score", 0) or 0)),
                    ("Cleaned", int(ss.get("pii_cleaned_count", 0) or 0)),
                    ("PII to be cleaned", remaining_to_clean),
                ]
                indicators_html = "".join(
                    (
                        "<div class='pii-indicator'>"
                        f"<div class='pii-indicator__label'>{label}</div>"
                        f"<div class='pii-indicator__value'>{value}</div>"
                        "</div>"
                    )
                    for label, value in indicator_values
                )
                st.markdown(
                    f"<div class='pii-indicators'>{indicators_html}</div>",
                    unsafe_allow_html=True,
                )

                if not flagged_ids:
                    st.success("No PII left to clean in the preview. 🎉")
                else:
                    idx = int(ss.get("pii_queue_idx", 0))
                    if idx >= len(flagged_ids):
                        idx = 0
                    ss["pii_queue_idx"] = idx
                    row_id = flagged_ids[idx]
                    row_data = dict(preview_rows[row_id])

                    col_editor, col_tokens = st.columns([2.6, 1.2], gap="large")

                    with col_editor:
                        st.caption("Edit & highlight")
                        st.caption(
                            f"Cleaning email {idx + 1} of {remaining_to_clean} flagged entries."
                        )
                        title_spans = ss["pii_hits_map"].get(row_id, {}).get("title", [])
                        body_spans = ss["pii_hits_map"].get(row_id, {}).get("body", [])
                        edited_values = ss.get("pii_edits", {}).get(row_id, {})
                        title_default = edited_values.get("title", row_data.get("title", ""))
                        body_default = edited_values.get("body", row_data.get("body", ""))

                        title_key = f"pii_title_{row_id}"
                        body_key = f"pii_body_{row_id}"
                        if title_key not in st.session_state:
                            st.session_state[title_key] = title_default
                        if body_key not in st.session_state:
                            st.session_state[body_key] = body_default

                        pending_token = ss.pop("pii_pending_token", None)
                        if pending_token:
                            if pending_token.get("row_id") == row_id:
                                target_field = pending_token.get("field", "body")
                                target_key = title_key if target_field == "title" else body_key
                                current_text = st.session_state.get(
                                    target_key,
                                    row_data.get(target_field, ""),
                                )
                                token_text = pending_token.get("token", "")
                                if token_text:
                                    spacer = " " if current_text and not current_text.endswith(" ") else ""
                                    st.session_state[target_key] = f"{current_text}{spacer}{token_text}"
                            else:
                                ss["pii_pending_token"] = pending_token

                        st.markdown(
                            "**Title (highlighted)**",
                            help="Highlights show detected PII.",
                        )
                        st.markdown(
                            _highlight_spans_html(st.session_state[title_key], title_spans),
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Body (highlighted)**")
                        st.markdown(
                            _highlight_spans_html(st.session_state[body_key], body_spans),
                            unsafe_allow_html=True,
                        )

                        title_value = st.text_input("✏️ Title (editable)", key=title_key)
                        body_value = st.text_area("✏️ Body (editable)", key=body_key, height=180)

                    with col_tokens:
                        st.caption("Replacements")
                        st.write("Click to insert tokens at cursor or paste them:")
                        token_columns = st.columns(2)
                        def _queue_token(token: str, field: str = "body") -> None:
                            st.session_state["pii_pending_token"] = {
                                "row_id": row_id,
                                "field": field,
                                "token": token,
                            }
                            _streamlit_rerun()

                        with token_columns[0]:
                            if st.button("{{EMAIL}}", key=f"pii_token_email_{row_id}"):
                                _queue_token("{{EMAIL}}")
                            if st.button("{{IBAN}}", key=f"pii_token_iban_{row_id}"):
                                _queue_token("{{IBAN}}")
                            if st.button("{{CARD_16}}", key=f"pii_token_card_{row_id}"):
                                _queue_token("{{CARD_16}}")
                        with token_columns[1]:
                            if st.button("{{PHONE}}", key=f"pii_token_phone_{row_id}"):
                                _queue_token("{{PHONE}}")
                            if st.button("{{OTP_6}}", key=f"pii_token_otp_{row_id}"):
                                _queue_token("{{OTP_6}}")
                            if st.button("{{URL_SUSPICIOUS}}", key=f"pii_token_url_{row_id}"):
                                _queue_token("{{URL_SUSPICIOUS}}")

                        st.divider()
                        action_columns = st.columns([1.2, 1, 1.2])
                        apply_and_next = action_columns[0].button("✅ Apply & Next", key="pii_apply_next", type="primary")
                        skip_row = action_columns[1].button("Skip", key="pii_skip")
                        finish_cleanup = action_columns[2].button("Finish cleanup", key="pii_finish")

                        if apply_and_next:
                            ss["pii_edits"].setdefault(row_id, {})
                            ss["pii_edits"][row_id]["title"] = title_value
                            ss["pii_edits"][row_id]["body"] = body_value
                            relinted_title = lint_text_spans(title_value)
                            relinted_body = lint_text_spans(body_value)
                            preview_rows[row_id]["title"] = title_value
                            preview_rows[row_id]["body"] = body_value
                            ss["dataset_preview_lint"] = lint_dataset(preview_rows)
                            if not relinted_title and not relinted_body:
                                ss["pii_cleaned_count"] = ss.get("pii_cleaned_count", 0) + 1
                                points = 10
                                ss["pii_score"] = ss.get("pii_score", 0) + points
                                st.toast(f"Clean! +{points} points", icon="🎯")
                                ss["pii_hits_map"].pop(row_id, None)
                                ss["pii_queue"] = [rid for rid in flagged_ids if rid != row_id]
                                flagged_ids = ss["pii_queue"]
                                ss["pii_queue_idx"] = min(idx, max(0, len(flagged_ids) - 1))
                                ss["pii_total_flagged"] = len(flagged_ids)
                            else:
                                ss["pii_hits_map"][row_id] = {"title": relinted_title, "body": relinted_body}
                                ss["pii_score"] = max(0, ss.get("pii_score", 0) - 2)
                                st.toast("Still detecting PII — try replacing with tokens.", icon="⚠️")
                            _streamlit_rerun()

                        if skip_row:
                            if flagged_ids:
                                ss["pii_queue_idx"] = (idx + 1) % len(flagged_ids)
                            _streamlit_rerun()

                        if finish_cleanup:
                            updated_detailed = lint_dataset_detailed(preview_rows)
                            new_counts = summarize_pii_counts(updated_detailed)
                            ss["pii_hits_map"] = updated_detailed
                            ss["pii_queue"] = sorted(updated_detailed.keys())
                            ss["pii_total_flagged"] = len(ss["pii_queue"])
                            ss["dataset_preview_lint"] = lint_dataset(preview_rows)
                            st.success(
                                "Cleanup finished. Remaining hits — {summary}.".format(
                                    summary=format_pii_summary(new_counts)
                                )
                            )
                            ss["pii_open"] = False

        with section_surface():
            st.markdown("### Review & approve")
            preview_summary = ss.get("dataset_preview_summary") or compute_dataset_summary(ss["dataset_preview"])
            lint_counts = ss.get("dataset_preview_lint") or {
                "credit_card": 0,
                "iban": 0,
                "email": 0,
                "phone": 0,
                "otp6": 0,
                "url": 0,
            }

            kpi_col, sample_col, edge_col = st.columns([1.1, 1.2, 1.1], gap="large")

            with kpi_col:
                st.write("**KPIs**")
                st.metric("Preview rows", preview_summary.get("total", 0))
                spam_ratio = preview_summary.get("spam_ratio", 0.0)
                st.metric("Spam share", f"{spam_ratio * 100:.1f}%")
                st.metric("Avg suspicious links (spam)", f"{preview_summary.get('avg_susp_links', 0.0):.2f}")

                safe_ratio = 1 - spam_ratio
                spam_pct = spam_ratio * 100
                safe_pct = safe_ratio * 100
                bar_html = f"""
                    <div style="border-radius: 6px; overflow: hidden; border: 1px solid #DDD; font-size: 0.75rem; margin: 0.5rem 0 0.75rem 0;">
                        <div style="display: flex; height: 28px;">
                            <div style="background-color: #ff4b4b; color: white; padding: 4px 8px; flex: {spam_ratio:.4f}; display: flex; align-items: center; justify-content: center;">
                                Spam {spam_pct:.0f}%
                            </div>
                            <div style="background-color: #1c83e1; color: white; padding: 4px 8px; flex: {safe_ratio:.4f}; display: flex; align-items: center; justify-content: center;">
                                Safe {safe_pct:.0f}%
                            </div>
                        </div>
                    </div>
                """
                st.markdown(bar_html, unsafe_allow_html=True)

                chip_html = pii_chip_row_html(lint_counts, extra_class="pii-chip-row--compact")
                if chip_html:
                    st.markdown(chip_html, unsafe_allow_html=True)
                st.caption("Guardrail: no live link fetching, HTML escaped, duplicates dropped.")

            with sample_col:
                st.write("**Stratified sample**")
                preview_rows = ss.get("dataset_preview", [])
                spam_examples = [row for row in preview_rows if row.get("label") == "spam"]
                safe_examples = [row for row in preview_rows if row.get("label") == "safe"]
                max_cards = 4
                cards: List[Dict[str, str]] = []
                idx_spam = idx_safe = 0
                for i in range(max_cards):
                    if i % 2 == 0:
                        if idx_spam < len(spam_examples):
                            cards.append(spam_examples[idx_spam])
                            idx_spam += 1
                        elif idx_safe < len(safe_examples):
                            cards.append(safe_examples[idx_safe])
                            idx_safe += 1
                    else:
                        if idx_safe < len(safe_examples):
                            cards.append(safe_examples[idx_safe])
                            idx_safe += 1
                        elif idx_spam < len(spam_examples):
                            cards.append(spam_examples[idx_spam])
                            idx_spam += 1
                    if len(cards) >= max_cards:
                        break
                if not cards:
                    st.info("Preview examples will appear here once generated.")
                else:
                    for card in cards:
                        label_value = (card.get("label", "") or "").strip().lower()
                        label_text = label_value.title() if label_value else "Unlabeled"
                        label_icon = {"spam": "🚩", "safe": "📥"}.get(label_value, "✉️")
                        body = card.get("body", "") or ""
                        excerpt = (body[:160] + ("…" if len(body) > 160 else "")).replace("\n", " ")
                        st.markdown(
                            f"""
                            <div style="border:1px solid #E5E7EB;border-radius:10px;padding:0.75rem;margin-bottom:0.5rem;">
                                <div class="sample-card__label"><span class="sample-card__label-icon">{label_icon}</span><span>{html.escape(label_text)}</span></div>
                                <div style="font-weight:600;margin:0.25rem 0 0.35rem 0;">{html.escape(card.get('title', ''))}</div>
                                <div style="font-size:0.85rem;color:#374151;line-height:1.35;">{html.escape(excerpt)}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with edge_col:
                st.write("**Edge-case pairs**")
                preview_config = ss.get("dataset_preview_config", {})
                if preview_config.get("edge_cases", 0) <= 0:
                    st.caption("Add edge cases in the builder to surface look-alike contrasts here.")
                else:
                    by_title: Dict[str, Dict[str, Dict[str, str]]] = {}
                    for row in ss.get("dataset_preview", []):
                        title = (row.get("title", "") or "").strip()
                        label = row.get("label", "")
                        if not title or label not in VALID_LABELS:
                            continue
                        by_title.setdefault(title, {})[label] = row
                    pairs = [
                        (data.get("spam"), data.get("safe"))
                        for title, data in by_title.items()
                        if data.get("spam") and data.get("safe")
                    ]
                    if not pairs:
                        st.info("No contrasting pairs surfaced yet — regenerate to refresh examples.")
                    else:
                        for spam_row, safe_row in pairs[:3]:
                            spam_excerpt = ((spam_row.get("body", "") or "")[:120] + ("…" if len(spam_row.get("body", "")) > 120 else "")).replace("\n", " ")
                            safe_excerpt = ((safe_row.get("body", "") or "")[:120] + ("…" if len(safe_row.get("body", "")) > 120 else "")).replace("\n", " ")
                            st.markdown(
                                f"""
                                <div style="border:1px solid #E5E7EB;border-radius:10px;padding:0.75rem;margin-bottom:0.5rem;">
                                    <div style="font-weight:600;margin-bottom:0.4rem;">{html.escape(spam_row.get('title', 'Untitled'))}</div>
                                    <div style="display:flex;gap:0.5rem;">
                                        <div style="flex:1;background:#FEE2E2;border-radius:8px;padding:0.5rem;font-size:0.8rem;">
                                            <div class="edge-case-card__label" style="color:#B91C1C;margin-bottom:0.25rem;"><span class="sample-card__label-icon">🚩</span><span>Spam</span></div>
                                            <div style="color:#7F1D1D;line-height:1.35;">{html.escape(spam_excerpt)}</div>
                                        </div>
                                        <div style="flex:1;background:#DBEAFE;border-radius:8px;padding:0.5rem;font-size:0.8rem;">
                                            <div class="edge-case-card__label" style="color:#1D4ED8;margin-bottom:0.25rem;"><span class="sample-card__label-icon">📥</span><span>Safe</span></div>
                                            <div style="color:#1E3A8A;line-height:1.35;">{html.escape(safe_excerpt)}</div>
                                        </div>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            preview_rows_full: List[Dict[str, Any]] = ss.get("dataset_preview") or []
            manual_df = ss.get("dataset_manual_queue")
            if not isinstance(manual_df, pd.DataFrame) or len(manual_df) != len(preview_rows_full):
                manual_df = pd.DataFrame(preview_rows_full)
            else:
                manual_df = manual_df.copy()
            if not manual_df.empty:
                if "include" not in manual_df.columns:
                    manual_df.insert(0, "include", True)
                else:
                    manual_df["include"] = manual_df["include"].fillna(True)
            elif preview_rows_full:
                manual_df = pd.DataFrame(preview_rows_full)
                if not manual_df.empty and "include" not in manual_df.columns:
                    manual_df.insert(0, "include", True)
            with st.expander("Expand this section if you want to manually review and edit individual emails part of the dataset"):
                edited_df = st.data_editor(
                    manual_df,
                    width="stretch",
                    hide_index=True,
                    key="dataset_manual_editor",
                    column_config={
                        "include": st.column_config.CheckboxColumn("Include?", help="Uncheck to drop before committing."),
                        "label": st.column_config.SelectboxColumn("Label", options=sorted(VALID_LABELS)),
                    },
                )
            ss["dataset_manual_queue"] = edited_df
            st.caption("Manual queue covers the entire preview — re-run the builder to generate more variations.")

        edited_df_for_commit = ss.get("dataset_manual_queue")

        with section_surface():
            st.markdown("### Commit dataset")
            st.markdown("<div class='cta-sticky'>", unsafe_allow_html=True)
            commit_col, discard_col, _ = st.columns([1, 1, 2])

            if commit_col.button("Commit dataset", type="primary", use_container_width=True):
                preview_rows_commit = ss.get("dataset_preview")
                config = ss.get("dataset_preview_config", ss.get("dataset_config", DEFAULT_DATASET_CONFIG))
                if not preview_rows_commit:
                    st.error("Generate a preview before committing.")
                else:
                    edited_records = []
                    if isinstance(edited_df_for_commit, pd.DataFrame):
                        edited_records = edited_df_for_commit.to_dict(orient="records")
                    preview_copy = [dict(row) for row in preview_rows_commit]
                    for idx, record in enumerate(edited_records):
                        if idx >= len(preview_copy):
                            break
                        preview_copy[idx]["title"] = str(record.get("title", preview_copy[idx].get("title", "")))
                        preview_copy[idx]["body"] = str(record.get("body", preview_copy[idx].get("body", "")))
                        preview_copy[idx]["label"] = record.get("label", preview_copy[idx].get("label", "spam"))
                        preview_copy[idx]["include"] = bool(record.get("include", True))
                    final_rows: List[Dict[str, str]] = []
                    for idx, row in enumerate(preview_copy):
                        include_flag = row.pop("include", True)
                        if idx < len(edited_records):
                            include_flag = bool(edited_records[idx].get("include", include_flag))
                        if not include_flag:
                            continue
                        final_rows.append(
                            {
                                "title": row.get("title", "").strip(),
                                "body": row.get("body", "").strip(),
                                "label": row.get("label", "spam"),
                            }
                        )
                    if len(final_rows) < 10:
                        st.warning("Need at least 10 rows to maintain a meaningful dataset.")
                    else:
                        previous_summary = ss.get("dataset_summary", {})
                        previous_config = ss.get("dataset_config", DEFAULT_DATASET_CONFIG)
                        lint_counts_commit = lint_dataset(final_rows) or {}
                        new_summary = compute_dataset_summary(final_rows)
                        delta = dataset_summary_delta(previous_summary, new_summary)
                        ss["previous_dataset_summary"] = previous_summary
                        ss["dataset_summary"] = new_summary
                        ss["dataset_config"] = config
                        ss["dataset_compare_delta"] = delta
                        ss["last_dataset_delta_story"] = dataset_delta_story(delta)
                        ss["labeled"] = final_rows
                        ss["active_dataset_snapshot"] = None
                        ss["dataset_snapshot_name"] = ""
                        ss["dataset_last_built_at"] = datetime.now().isoformat(timespec="seconds")
                        _clear_dataset_preview_state()
                        health_evaluation = _evaluate_dataset_health(new_summary, lint_counts_commit)
                        spam_ratio_commit = new_summary.get("spam_ratio") or 0.0
                        spam_share_pct_commit = max(0.0, min(100.0, float(spam_ratio_commit) * 100.0))
                        committed_rows = new_summary.get("total", len(final_rows))
                        try:
                            committed_display = int(committed_rows)
                        except (TypeError, ValueError):
                            committed_display = len(final_rows)
                        health_token = health_evaluation.get("health_emoji") or "—"
                        summary_line = (
                            f"Committed {committed_display} rows • "
                            f"Spam share {spam_share_pct_commit:.1f}% • Health: {health_token}"
                        )

                        prev_spam_ratio = previous_summary.get("spam_ratio") if previous_summary else None
                        spam_delta_pp = 0.0
                        if prev_spam_ratio is not None:
                            try:
                                spam_delta_pp = (float(spam_ratio_commit) - float(prev_spam_ratio)) * 100.0
                            except (TypeError, ValueError):
                                spam_delta_pp = 0.0
                        prev_edge_cases = previous_config.get("edge_cases", 0)
                        new_edge_cases = config.get("edge_cases", prev_edge_cases)
                        if spam_delta_pp >= 10.0:
                            hint_line = "Expect recall ↑, precision may ↓; tune threshold in Evaluate."
                        elif new_edge_cases > prev_edge_cases:
                            hint_line = "Boundary likely sharpened; training may need more iterations."
                        else:
                            hint_line = "Mix steady; train to refresh the model, then tune the threshold in Evaluate."

                        _push_data_stage_flash("success", f"{summary_line}\n{hint_line}")
                        if any(lint_counts_commit.values()):
                            _push_data_stage_flash(
                                "warning",
                                "Lint warnings persist after commit ({}).".format(
                                    format_pii_summary(lint_counts_commit)
                                ),
                            )

                        _set_advanced_knob_state(config, force=True)
                        if ss.get("_needs_advanced_knob_rerun"):
                            _streamlit_rerun()

            if discard_col.button("Discard preview", type="secondary", use_container_width=True):
                _discard_preview()
                st.info("Preview cleared. The active labeled dataset remains unchanged.")
            st.markdown("</div>", unsafe_allow_html=True)

    if dataset_generated_once or preview_summary_for_health is not None:
        with section_surface():
            st.markdown("### Dataset snapshot")
            current_config = ss.get("dataset_config", DEFAULT_DATASET_CONFIG)
            config_json = json.dumps(current_config, indent=2, sort_keys=True)
            current_summary = ss.get("dataset_summary") or compute_dataset_summary(ss.get("labeled", []))
            st.caption("Save immutable snapshots to reference in the model card and audits.")
    
            total_rows = current_summary.get("total") if isinstance(current_summary, dict) else None
            if total_rows is None:
                total_rows = len(ss.get("labeled", []))
            rows_display = total_rows
            try:
                rows_display = int(total_rows)
            except (TypeError, ValueError):
                rows_display = total_rows or "—"
    
            spam_ratio_val = None
            if isinstance(current_summary, dict):
                spam_ratio_val = current_summary.get("spam_ratio")
            if spam_ratio_val is None:
                spam_share_display = "—"
            else:
                try:
                    spam_share_display = f"{float(spam_ratio_val) * 100:.1f}%"
                except (TypeError, ValueError):
                    spam_share_display = "—"
    
            edge_cases = current_config.get("edge_cases") if isinstance(current_config, dict) else None
            edge_display = None
            if edge_cases is not None:
                try:
                    edge_display = int(edge_cases)
                except (TypeError, ValueError):
                    edge_display = edge_cases
    
            labeled_rows = ss.get("labeled", [])
            try:
                snapshot_hash = compute_dataset_hash(labeled_rows)[:10]
            except Exception:
                snapshot_hash = "—"
            timestamp_display = ss.get("dataset_last_built_at") or "—"
            seed_value = current_config.get("seed") if isinstance(current_config, dict) else None
            seed_display = seed_value if seed_value is not None else "—"
    
            summary_rows_html = [
                "<div style='display:flex;justify-content:space-between;'><span>Rows</span><strong>{}</strong></div>".format(
                    html.escape(str(rows_display))
                ),
                "<div style='display:flex;justify-content:space-between;'><span>Spam share</span><strong>{}</strong></div>".format(
                    html.escape(str(spam_share_display))
                ),
            ]
            if edge_display is not None:
                summary_rows_html.append(
                    "<div style='display:flex;justify-content:space-between;'><span>Edge cases</span><strong>{}</strong></div>".format(
                        html.escape(str(edge_display))
                    )
                )
    
            fingerprint_rows_html = [
                "<div style='display:flex;justify-content:space-between;'><span>Short hash</span><strong>{}</strong></div>".format(
                    html.escape(str(snapshot_hash))
                ),
                "<div style='display:flex;justify-content:space-between;'><span>Timestamp</span><strong>{}</strong></div>".format(
                    html.escape(str(timestamp_display))
                ),
                "<div style='display:flex;justify-content:space-between;'><span>Seed</span><strong>{}</strong></div>".format(
                    html.escape(str(seed_display))
                ),
            ]
    
            card_html = """
                <div style="border:1px solid #E5E7EB;border-radius:12px;padding:1rem 1.25rem;margin-bottom:0.75rem;">
                    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;">
                        <div style="flex:1 1 240px;min-width:220px;">
                            <div style="font-weight:600;font-size:0.95rem;margin-bottom:0.6rem;">Summary</div>
                            <div style="display:flex;flex-direction:column;gap:0.4rem;font-size:0.9rem;">
                                {summary_rows}
                            </div>
                        </div>
                        <div style="flex:1 1 240px;min-width:220px;">
                            <div style="font-weight:600;font-size:0.95rem;margin-bottom:0.6rem;">Fingerprint</div>
                            <div style="display:flex;flex-direction:column;gap:0.4rem;font-size:0.9rem;">
                                {fingerprint_rows}
                            </div>
                        </div>
                    </div>
                </div>
            """.format(
                summary_rows="".join(summary_rows_html),
                fingerprint_rows="".join(fingerprint_rows_html),
            )
            st.markdown(card_html, unsafe_allow_html=True)
    
            with st.expander("View JSON", expanded=False):
                st.json(json.loads(config_json))
            ss["dataset_snapshot_name"] = st.text_input(
                "Snapshot name",
                value=ss.get("dataset_snapshot_name", ""),
                help="Describe the scenario (e.g., 'High links, 5% noise').",
            )
            if st.button("Save dataset snapshot", key="save_dataset_snapshot"):
                snapshot_id = compute_dataset_hash(ss["labeled"])
                entry = {
                    "id": snapshot_id,
                    "name": ss.get("dataset_snapshot_name") or f"snapshot-{len(ss['datasets'])+1}",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "config": ss.get("dataset_config", DEFAULT_DATASET_CONFIG),
                    "config_json": config_json,
                    "rows": len(ss["labeled"]),
                }
                existing = next((snap for snap in ss["datasets"] if snap.get("id") == snapshot_id), None)
                if existing:
                    existing.update(entry)
                else:
                    ss["datasets"].append(entry)
                ss["active_dataset_snapshot"] = snapshot_id
                st.success(f"Snapshot saved with id `{snapshot_id[:10]}…`. Use it in the model card.")
    
            if ss.get("datasets"):
                st.markdown("#### Saved snapshots")
                header_cols = st.columns([3, 2.2, 2.2, 1.4, 1.2])
                for col, label in zip(header_cols, ["Name", "Fingerprint", "Saved", "Rows", ""]):
                    col.markdown(f"**{label}**")
    
                datasets_sorted = sorted(
                    ss["datasets"],
                    key=lambda snap: snap.get("timestamp", ""),
                    reverse=True,
                )
                for idx, snap in enumerate(datasets_sorted):
                    name_col, fp_col, ts_col, rows_col, action_col = st.columns([3, 2.2, 2.2, 1.4, 1.2])
                    is_active = snap.get("id") == ss.get("active_dataset_snapshot")
                    name_value = snap.get("name") or "(unnamed snapshot)"
                    badge_html = ""
                    if is_active:
                        badge_html = (
                            "<span style='margin-left:0.35rem;background:#DCFCE7;color:#166534;font-size:0.7rem;font-weight:600;"
                            "padding:2px 8px;border-radius:999px;vertical-align:middle;'>Active</span>"
                        )
                    name_col.markdown(
                        "<div style='display:flex;align-items:center;gap:0.25rem;'><span style='font-weight:600;'>{}</span>{}</div>".format(
                            html.escape(name_value),
                            badge_html,
                        ),
                        unsafe_allow_html=True,
                    )
    
                    snapshot_id = snap.get("id", "")
                    short_id = "—"
                    if snapshot_id:
                        short_id = snapshot_id if len(snapshot_id) <= 10 else f"{snapshot_id[:10]}…"
                    fp_col.markdown(f"`{short_id}`")
                    ts_col.markdown(snap.get("timestamp", "—"))
                    rows_col.markdown(str(snap.get("rows", "—")))
    
                    button_label = "Set active"
                    button_disabled = is_active
                    button_key = f"set_active_snapshot_{idx}_{snapshot_id[:6]}"
                    if action_col.button(button_label, key=button_key, disabled=button_disabled):
                        config = snap.get("config")
                        if isinstance(config, str) and config:
                            try:
                                config = json.loads(config)
                            except json.JSONDecodeError:
                                config = None
                        if not isinstance(config, dict):
                            st.error("Snapshot is missing a valid configuration.")
                        else:
                            dataset_rows = build_dataset_from_config(config)
                            summary = compute_dataset_summary(dataset_rows)
                            lint_counts = lint_dataset(dataset_rows) or {}
                            ss["labeled"] = dataset_rows
                            ss["dataset_config"] = config
                            ss["dataset_summary"] = summary
                            ss["previous_dataset_summary"] = None
                            ss["dataset_compare_delta"] = None
                            ss["last_dataset_delta_story"] = None
                            ss["dataset_snapshot_name"] = snap.get("name", "")
                            ss["active_dataset_snapshot"] = snap.get("id")
                            ss["dataset_last_built_at"] = datetime.now().isoformat(timespec="seconds")
                            _clear_dataset_preview_state()
                            _push_data_stage_flash(
                                "success",
                                f"Snapshot '{snap.get('name', 'snapshot')}' activated. Dataset rebuilt with {len(dataset_rows)} rows.",
                            )
                            if any(lint_counts.values()):
                                _push_data_stage_flash(
                                    "warning",
                                    "Lint warnings present in restored snapshot ({}).".format(
                                        format_pii_summary(lint_counts)
                                    ),
                                )
                            _set_advanced_knob_state(config, force=True)
                            if ss.get("_needs_advanced_knob_rerun"):
                                _streamlit_rerun()
            else:
                st.caption("No snapshots yet. Save one after curating your first dataset.")
    
    
    if nerd_mode_data_enabled:
        with section_surface():
            st.markdown("### Nerd Mode insights")
            df_lab = pd.DataFrame(ss["labeled"])
            if df_lab.empty:
                st.info("Label some emails or import data to unlock diagnostics.")
            else:
                diagnostics_df = df_lab.head(500).copy()
                st.caption(f"Diagnostics sample: {len(diagnostics_df)} emails (cap 500).")

                st.markdown("#### Feature distributions by class")
                feature_records: list[dict[str, Any]] = []
                for _, row in diagnostics_df.iterrows():
                    label = row.get("label")
                    if label not in VALID_LABELS:
                        continue
                    title = row.get("title", "") or ""
                    body = row.get("body", "") or ""
                    feature_records.append(
                        {
                            "label": label,
                            "suspicious_links": float(_count_suspicious_links(body)),
                            "caps_ratio": float(_caps_ratio(f"{title} {body}")),
                            "money_mentions": float(_count_money_mentions(body)),
                        }
                    )
                feature_df = pd.DataFrame(feature_records)
                if feature_df.empty or feature_df["label"].nunique() < 2 or feature_df.groupby("label").size().min() < 3:
                    st.caption("Need at least 3 emails per class to chart numeric feature distributions.")
                else:
                    feature_specs = [
                        ("suspicious_links", "Suspicious links per email"),
                        ("caps_ratio", "ALL-CAPS ratio"),
                        ("money_mentions", "Money & urgency mentions"),
                    ]
                    for feature_key, feature_label in feature_specs:
                        sub_df = feature_df.loc[:, ["label", feature_key]].rename(columns={feature_key: "value"})
                        sub_df["value"] = pd.to_numeric(sub_df["value"], errors="coerce")
                        base_chart = (
                            alt.Chart(sub_df.dropna())
                            .mark_bar(opacity=0.75)
                            .encode(
                                alt.X("value:Q", bin=alt.Bin(maxbins=10), title="Feature value"),
                                alt.Y("count()", title="Count"),
                                alt.Color(
                                    "label:N",
                                    scale=alt.Scale(domain=["spam", "safe"], range=["#ef4444", "#1d4ed8"]),
                                    legend=None,
                                ),
                                tooltip=[
                                    alt.Tooltip("label:N", title="Class"),
                                    alt.Tooltip("value:Q", bin=alt.Bin(maxbins=10), title="Feature value"),
                                    alt.Tooltip("count()", title="Count"),
                                ],
                            )
                            .properties(height=220)
                        )
                        chart = (
                            base_chart
                            .facet(column=alt.Column("label:N", title=None))
                            .resolve_scale(y="independent")
                            .properties(title=feature_label)
                        )
                        st.altair_chart(chart, use_container_width=True)

                st.markdown("#### Near-duplicate check")
                embed_sample = diagnostics_df.head(min(len(diagnostics_df), 200)).copy()
                if embed_sample.empty or embed_sample["label"].nunique() < 2:
                    st.caption("Need both classes present to run the near-duplicate check.")
                    embeddings = np.empty((0, 0), dtype=np.float32)
                    embed_error = None
                else:
                    embed_sample = embed_sample.sample(frac=1.0, random_state=42).reset_index(drop=True)
                    sample_records = embed_sample.to_dict(orient="records")
                    sample_hash = compute_dataset_hash(sample_records)
                    texts = tuple(
                        combine_text(rec.get("title", ""), rec.get("body", ""))
                        for rec in sample_records
                    )
                    embed_error = None
                    try:
                        embeddings = _compute_cached_embeddings(sample_hash, texts)
                    except Exception as exc:  # pragma: no cover - defensive for encoder availability
                        embeddings = np.empty((0, 0), dtype=np.float32)
                        embed_error = str(exc)

                if embed_error:
                    st.warning(f"Embedding diagnostics unavailable: {embed_error}")
                elif embeddings.size < 4:
                    st.caption("Not enough samples for similarity analysis (need at least two per class).")
                else:
                    st.caption(f"Near-duplicate scan on {embeddings.shape[0]} emails (cap 200).")
                    sims = embeddings @ embeddings.T
                    np.fill_diagonal(sims, -1.0)
                    labels = embed_sample["label"].tolist()
                    top_pairs: list[tuple[float, int, int]] = []
                    n_rows = sims.shape[0]
                    for i in range(n_rows):
                        for j in range(i + 1, n_rows):
                            if labels[i] == labels[j]:
                                continue
                            sim_val = float(sims[i, j])
                            if sim_val > 0.9:
                                top_pairs.append((sim_val, i, j))
                    top_pairs.sort(key=lambda tup: tup[0], reverse=True)
                    top_pairs = top_pairs[:5]
                    if not top_pairs:
                        st.caption("No high-similarity cross-label pairs detected in the sampled set.")
                    else:
                        for sim_val, idx_a, idx_b in top_pairs:
                            row_a = embed_sample.iloc[idx_a]
                            row_b = embed_sample.iloc[idx_b]
                            st.markdown(
                                f"**Similarity {sim_val:.2f}** — {row_a.get('label', '').title()} vs {row_b.get('label', '').title()}"
                            )
                            pair_cols = st.columns(2)
                            for col, row in zip(pair_cols, [row_a, row_b]):
                                with col:
                                    st.caption(row.get("label", "").title() or "Unknown")
                                    st.write(f"**{row.get('title', '(untitled)')}**")
                                    body_text = (row.get("body", "") or "").replace("\n", " ")
                                    excerpt = body_text[:160] + ("…" if len(body_text) > 160 else "")
                                    st.write(excerpt)

                st.markdown("#### Class prototypes")
                if embed_error:
                    st.caption("Embeddings unavailable — prototypes skipped.")
                elif embeddings.size == 0:
                    st.caption("Need labeled examples to compute prototypes.")
                else:
                    proto_cols = st.columns(2)
                    for col, label in zip(proto_cols, ["spam", "safe"]):
                        with col:
                            mask = embed_sample["label"] == label
                            idxs = np.where(mask.to_numpy())[0]
                            if idxs.size == 0:
                                st.caption(f"No {label} examples in the sample.")
                                continue
                            centroid = embeddings[idxs].mean(axis=0)
                            norm = float(np.linalg.norm(centroid))
                            if norm == 0.0:
                                st.caption("Centroid not informative for this class.")
                                continue
                            centroid /= norm
                            sims_label = embeddings[idxs] @ centroid
                            top_local = idxs[np.argsort(-sims_label)[:3]]
                            st.markdown(f"**{label.title()} archetype**")
                            for rank, idx_point in enumerate(top_local, start=1):
                                row = embed_sample.iloc[idx_point]
                                body_text = (row.get("body", "") or "").replace("\n", " ")
                                excerpt = body_text[:140] + ("…" if len(body_text) > 140 else "")
                                st.caption(f"{rank}. {row.get('title', '(untitled)')}")
                                st.write(excerpt)

                st.divider()
                st.markdown("#### Quick lexical snapshot")
                tokens_spam = Counter()
                tokens_safe = Counter()
                for _, row in df_lab.iterrows():
                    text = f"{row.get('title', '')} {row.get('body', '')}".lower()
                    tokens = re.findall(r"[a-zA-Z']+", text)
                    if row.get("label") == "spam":
                        tokens_spam.update(tokens)
                    else:
                        tokens_safe.update(tokens)
                top_spam = tokens_spam.most_common(12)
                top_safe = tokens_safe.most_common(12)
                col_tok1, col_tok2 = st.columns(2)
                with col_tok1:
                    st.markdown("**Class token cloud — Spam**")
                    st.write(", ".join(f"{w} ({c})" for w, c in top_spam) or "—")
                with col_tok2:
                    st.markdown("**Class token cloud — Safe**")
                    st.write(", ".join(f"{w} ({c})" for w, c in top_safe) or "—")

                title_groups: Dict[str, set] = {}
                leakage_titles = []
                for _, row in df_lab.iterrows():
                    title = row.get("title", "").strip().lower()
                    label = row.get("label")
                    title_groups.setdefault(title, set()).add(label)
                for title, labels in title_groups.items():
                    if len(labels) > 1 and title:
                        leakage_titles.append(title)

                strat_df = df_lab.groupby("label").size().reset_index(name="count")
                st.dataframe(strat_df, hide_index=True, width="stretch")

        with st.expander("📤 Upload CSV of labeled emails (strict schema)", expanded=False):
            st.caption(
                "Schema: title, body, label (spam|safe). Limits: ≤2,000 rows, title ≤200 chars, body ≤2,000 chars."
            )
            st.caption("Uploaded data stays in this session only. No emails are sent or fetched.")
            up = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader_labeled")
            if up is not None:
                try:
                    df_up = pd.read_csv(up)
                    df_up.columns = [c.strip().lower() for c in df_up.columns]
                    ok, msg = _validate_csv_schema(df_up)
                    if not ok:
                        st.error(msg)
                    else:
                        if len(df_up) > 2000:
                            st.error("Too many rows (max 2,000). Trim the file and retry.")
                        else:
                            initial_rows = len(df_up)
                            df_up["label"] = df_up["label"].apply(_normalize_label)
                            invalid_mask = ~df_up["label"].isin(VALID_LABELS)
                            dropped_invalid = int(invalid_mask.sum())
                            df_up = df_up[~invalid_mask]
                            for col in ["title", "body"]:
                                df_up[col] = df_up[col].fillna("").astype(str).str.strip()
                            length_mask = (df_up["title"].str.len() <= 200) & (df_up["body"].str.len() <= 2000)
                            dropped_length = int(len(df_up) - length_mask.sum())
                            df_up = df_up[length_mask]
                            nonempty_mask = (df_up["title"] != "") | (df_up["body"] != "")
                            dropped_empty = int(len(df_up) - nonempty_mask.sum())
                            df_up = df_up[nonempty_mask]
                            df_existing = pd.DataFrame(ss["labeled"])
                            dropped_dupes = 0
                            if not df_existing.empty:
                                len_before_duplicates = len(df_up)
                                merged = df_up.merge(df_existing, on=["title", "body", "label"], how="left", indicator=True)
                                df_up = merged[merged["_merge"] == "left_only"].loc[:, ["title", "body", "label"]]
                                dropped_dupes = int(max(0, len_before_duplicates - len(df_up)))
                            total_dropped = max(0, initial_rows - len(df_up))
                            drop_reasons: list[str] = []
                            if dropped_invalid:
                                drop_reasons.append(
                                    f"{dropped_invalid} invalid label{'s' if dropped_invalid != 1 else ''}"
                                )
                            if dropped_length:
                                drop_reasons.append(
                                    f"{dropped_length} over length limit"
                                )
                            if dropped_empty:
                                drop_reasons.append(
                                    f"{dropped_empty} blank title/body"
                                )
                            if dropped_dupes:
                                drop_reasons.append(
                                    f"{dropped_dupes} duplicates vs session"
                                )
                            reason_text = "; ".join(drop_reasons) if drop_reasons else "—"
                            lint_counts = lint_dataset(df_up.to_dict(orient="records"))
                            st.caption(f"Rows dropped: {total_dropped} (reason: {reason_text})")
                            st.dataframe(df_up.head(20), hide_index=True, width="stretch")
                            st.caption(
                                "Rows passing validation: {} | Lint -> {}".format(
                                    len(df_up), format_pii_summary(lint_counts)
                                )
                            )
                            if len(df_up) > 0 and st.button(
                                "Import into labeled dataset", key="btn_import_csv"
                            ):
                                ss["labeled"].extend(df_up.to_dict(orient="records"))
                                ss["dataset_summary"] = compute_dataset_summary(ss["labeled"])
                                st.success(
                                    f"Imported {len(df_up)} rows into labeled dataset. Revisit builder to rebalance if needed."
                                )
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")



def _clear_dataset_preview_state() -> None:
    ss["dataset_preview"] = None
    ss["dataset_preview_config"] = None
    ss["dataset_preview_summary"] = None
    ss["dataset_preview_lint"] = None
    ss["dataset_manual_queue"] = None
    ss["dataset_controls_open"] = False


def _discard_preview() -> None:
    _clear_dataset_preview_state()
    ss["dataset_compare_delta"] = None
    ss["last_dataset_delta_story"] = explain_config_change(ss.get("dataset_config", DEFAULT_DATASET_CONFIG))


def _label_balance_status(labeled: list[dict] | None) -> dict:
    """Return counts, total, ratio, and OK flag for balance."""

    labeled = labeled or []
    counts = _counts(
        [
            (r.get("label") or "").strip().lower()
            for r in labeled
            if isinstance(r, dict)
        ]
    )
    total = counts["spam"] + counts["safe"]
    big = max(counts["spam"], counts["safe"])
    small = min(counts["spam"], counts["safe"])
    ratio = (small / big) if big else 0.0
    ok = (total >= 12) and (counts["spam"] >= 6) and (counts["safe"] >= 6) and (ratio >= 0.60)
    return {"counts": counts, "total": total, "ratio": ratio, "ok": ok}


def _pii_status() -> dict:
    """Read PII scan summary saved during Prepare (if any)."""

    # expected shape: {"status": "clean"|"found"|"unknown", "counts": {"emails":..,"phones":..,"names":..}}
    pii = ss.get("pii_scan") or {}
    status = pii.get("status", "unknown")
    counts = pii.get("counts", {})
    return {"status": status, "counts": counts}


def _go_to_prepare():
    """Jump to Prepare stage (match your stage switching mechanism)."""

    ss["stage"] = "prepare"
    st.experimental_rerun()


def render_evaluate_stage():

    stage = STAGE_BY_KEY["evaluate"]

    if not (ss.get("model") and ss.get("split_cache")):
        with section_surface():
            st.subheader(f"{stage.icon} {stage.title} — How well does your spam detector perform?")
            st.info("Train a model first in the **Train** tab.")
        return

    cache = ss["split_cache"]
    if len(cache) == 4:
        X_tr, X_te, y_tr, y_te = cache
        texts_test = X_te
        X_te_t = X_te_b = None
    else:
        X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = cache
        texts_test = [(t or "") + "\n" + (b or "") for t, b in zip(X_te_t, X_te_b)]

    try:
        if len(cache) == 6:
            probs = ss["model"].predict_proba(X_te_t, X_te_b)
        else:
            probs = ss["model"].predict_proba(texts_test)
    except TypeError:
        probs = ss["model"].predict_proba(texts_test)

    classes = list(getattr(ss["model"], "classes_", []))
    if classes and "spam" in classes:
        idx_spam = classes.index("spam")
    else:
        idx_spam = 1 if probs.shape[1] > 1 else 0
    p_spam = probs[:, idx_spam]
    y_true01 = _y01(list(y_te))

    current_thr = float(ss.get("threshold", 0.5))
    cm = compute_confusion(y_true01, p_spam, current_thr)
    acc = (cm["TP"] + cm["TN"]) / max(1, len(y_true01))
    emoji, verdict = verdict_label(acc, len(y_true01))
    prev_eval = ss.get("last_eval_results") or {}
    acc_cur, p_cur, r_cur, f1_cur, cm_cur = _pr_acc_cm(y_true01, p_spam, current_thr)

    with section_surface():
        narrative_col, metrics_col = st.columns([3, 2], gap="large")
        with narrative_col:
            st.subheader(f"{stage.icon} {stage.title} — How well does your spam detector perform?")
            st.write(
                "Now that your model has learned from examples, it’s time to test how well it works. "
                "During training, we kept some emails aside — the **test set**. The model hasn’t seen these before. "
                "By checking its guesses against the true labels, we get a fair measure of performance."
            )
            st.markdown("### What do these results say?")
            st.markdown(make_after_eval_story(len(y_true01), cm))
        with metrics_col:
            st.markdown("### Snapshot")
            st.success(f"**Accuracy:** {acc:.2%}  |  {emoji} {verdict}")
            st.caption(f"Evaluated on {len(y_true01)} unseen emails at threshold {current_thr:.2f}.")
            st.markdown(
                "- ✅ Spam caught: **{tp}**\n"
                "- ❌ Spam missed: **{fn}**\n"
                "- ⚠️ Safe mis-flagged: **{fp}**\n"
                "- ✅ Safe passed: **{tn}**"
            .format(tp=cm["TP"], fn=cm["FN"], fp=cm["FP"], tn=cm["TN"]))
            dataset_story = ss.get("last_dataset_delta_story")
            metric_deltas: list[str] = []
            if prev_eval:
                metric_deltas.append(f"Δaccuracy {acc_cur - prev_eval.get('accuracy', acc_cur):+.2%}")
                metric_deltas.append(f"Δprecision {p_cur - prev_eval.get('precision', p_cur):+.2%}")
                metric_deltas.append(f"Δrecall {r_cur - prev_eval.get('recall', r_cur):+.2%}")
            extra_caption = " | ".join(part for part in [dataset_story, " · ".join(metric_deltas) if metric_deltas else ""] if part)
            if extra_caption:
                st.caption(f"📂 {extra_caption}")

    with section_surface():
        st.markdown("### Spam threshold")
        presets = threshold_presets(y_true01, p_spam)

        if "eval_temp_threshold" not in ss:
            ss["eval_temp_threshold"] = current_thr

        controls_col, slider_col = st.columns([2, 3], gap="large")
        with controls_col:
            if st.button("Balanced (max F1)", use_container_width=True):
                ss["eval_temp_threshold"] = float(presets["balanced_f1"])
                st.toast(f"Suggested threshold (max F1): {ss['eval_temp_threshold']:.2f}", icon="✅")
            if st.button("Protect inbox (≥95% precision)", use_container_width=True):
                ss["eval_temp_threshold"] = float(presets["precision_95"])
                st.toast(
                    f"Suggested threshold (precision≥95%): {ss['eval_temp_threshold']:.2f}",
                    icon="✅",
                )
            if st.button("Catch spam (≥90% recall)", use_container_width=True):
                ss["eval_temp_threshold"] = float(presets["recall_90"])
                st.toast(
                    f"Suggested threshold (recall≥90%): {ss['eval_temp_threshold']:.2f}",
                    icon="✅",
                )
            if st.button("Adopt this threshold", use_container_width=True):
                ss["threshold"] = float(ss.get("eval_temp_threshold", current_thr))
                st.success(
                    f"Adopted new operating threshold: **{ss['threshold']:.2f}**. This will be used in Classify and Full Autonomy."
                )
        with slider_col:
            temp_threshold = float(
                st.slider(
                    "Adjust threshold (temporary)",
                    0.1,
                    0.9,
                    value=float(ss.get("eval_temp_threshold", current_thr)),
                    step=0.01,
                    key="eval_temp_threshold",
                    help="Lower values catch more spam (higher recall) but risk more false alarms. Higher values protect the inbox (higher precision) but may miss some spam.",
                )
            )

            cm_temp = compute_confusion(y_true01, p_spam, temp_threshold)
            acc_temp = (cm_temp["TP"] + cm_temp["TN"]) / max(1, len(y_true01))
            st.caption(
                f"At {temp_threshold:.2f}, accuracy would be **{acc_temp:.2%}** (TP {cm_temp['TP']}, FP {cm_temp['FP']}, TN {cm_temp['TN']}, FN {cm_temp['FN']})."
            )

        acc_new, p_new, r_new, f1_new, cm_new = _pr_acc_cm(y_true01, p_spam, temp_threshold)

        with st.container(border=True):
            st.markdown("#### What changes when I move the threshold?")
            st.caption("Comparing your **adopted** threshold vs. the **temporary** slider value above:")

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Current (adopted)**")
                st.write(f"- Threshold: **{current_thr:.2f}**")
                st.write(f"- Accuracy: {_fmt_pct(acc_cur)}")
                st.write(f"- Precision (spam): {_fmt_pct(p_cur)}")
                st.write(f"- Recall (spam): {_fmt_pct(r_cur)}")
                st.write(f"- False positives (safe→spam): **{cm_cur['FP']}**")
                st.write(f"- False negatives (spam→safe): **{cm_cur['FN']}**")

            with col_right:
                st.markdown("**If you adopt the slider value**")
                st.write(f"- Threshold: **{temp_threshold:.2f}**")
                st.write(f"- Accuracy: {_fmt_pct(acc_new)} ({_fmt_delta(acc_new, acc_cur)})")
                st.write(f"- Precision (spam): {_fmt_pct(p_new)} ({_fmt_delta(p_new, p_cur)})")
                st.write(f"- Recall (spam): {_fmt_pct(r_new)} ({_fmt_delta(r_new, r_cur)})")
                st.write(
                    f"- False positives: **{cm_new['FP']}** ({_fmt_delta(cm_new['FP'], cm_cur['FP'], pct=False)})"
                )
                st.write(
                    f"- False negatives: **{cm_new['FN']}** ({_fmt_delta(cm_new['FN'], cm_cur['FN'], pct=False)})"
                )

            if temp_threshold > current_thr:
                st.info(
                    "Raising the threshold makes the model **more cautious**: usually **fewer false positives** (protects inbox) but **more spam may slip through**."
                )
            elif temp_threshold < current_thr:
                st.info(
                    "Lowering the threshold makes the model **more aggressive**: it **catches more spam** (higher recall) but may **flag more legit emails**."
                )
            else:
                st.info("Same threshold as adopted — metrics unchanged.")

    with section_surface():
        with st.expander("📌 Suggestions to improve your model"):
            st.markdown(
                """
- Add more labeled emails, especially tricky edge cases
- Balance the dataset between spam and safe
- Use diverse wording in your examples
- Tune the spam threshold for your needs
- Review the confusion matrix to spot mistakes
- Ensure emails have enough meaningful content
"""
            )

    ss["last_eval_results"] = {
        "accuracy": acc_cur,
        "precision": p_cur,
        "recall": r_cur,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    nerd_mode_eval_enabled = render_nerd_mode_toggle(
        key="nerd_mode_eval",
        title="Nerd Mode — technical details",
        description="Inspect precision/recall tables, interpretability cues, and governance notes.",
        icon="🔬",
    )

    if nerd_mode_eval_enabled:
        with section_surface():
            temp_threshold = float(ss.get("eval_temp_threshold", current_thr))
            y_hat_temp = (p_spam >= temp_threshold).astype(int)
            prec_spam, rec_spam, f1_spam, sup_spam = precision_recall_fscore_support(
                y_true01, y_hat_temp, average="binary", zero_division=0
            )
            y_true_safe = 1 - y_true01
            y_hat_safe = 1 - y_hat_temp
            prec_safe, rec_safe, f1_safe, sup_safe = precision_recall_fscore_support(
                y_true_safe, y_hat_safe, average="binary", zero_division=0
            )

            st.markdown("### Detailed metrics (at current threshold)")

            def _as_int(value, fallback):
                if value is None:
                    return int(fallback)
                try:
                    return int(value)
                except TypeError:
                    return int(fallback)

            spam_support = _as_int(sup_spam, np.sum(y_true01))
            safe_support = _as_int(sup_safe, np.sum(1 - y_true01))

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "class": "spam",
                            "precision": prec_spam,
                            "recall": rec_spam,
                            "f1": f1_spam,
                            "support": spam_support,
                        },
                        {
                            "class": "safe",
                            "precision": prec_safe,
                            "recall": rec_safe,
                            "f1": f1_safe,
                            "support": safe_support,
                        },
                    ]
                ).round(3),
                width="stretch",
                hide_index=True,
            )

            st.markdown("### Precision & Recall vs Threshold (validation)")
            fig = plot_threshold_curves(y_true01, p_spam)
            st.pyplot(fig)

            st.markdown("### Interpretability")
            try:
                if hasattr(ss["model"], "named_steps"):
                    clf = ss["model"].named_steps.get("clf")
                    vec = ss["model"].named_steps.get("tfidf")
                    if hasattr(clf, "coef_") and vec is not None:
                        vocab = np.array(vec.get_feature_names_out())
                        coefs = clf.coef_[0]
                        top_spam = vocab[np.argsort(coefs)[-10:]][::-1]
                        top_safe = vocab[np.argsort(coefs)[:10]]
                        col_i1, col_i2 = st.columns(2)
                        with col_i1:
                            st.write("Top signals → **Spam**")
                            st.write(", ".join(top_spam))
                        with col_i2:
                            st.write("Top signals → **Safe**")
                            st.write(", ".join(top_safe))
                    else:
                        st.caption("Coefficients unavailable for this classifier.")
                elif hasattr(ss["model"], "numeric_feature_coefs"):
                    coef_map = ss["model"].numeric_feature_coefs()
                    st.caption("Numeric feature weights (positive → Spam, negative → Safe):")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "feature": k,
                                    "weight_toward_spam": v,
                                }
                                for k, v in coef_map.items()
                            ]
                        ).sort_values("weight_toward_spam", ascending=False),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.caption("Interpretability: no compatible inspector for this model.")
            except Exception as e:
                st.caption(f"Interpretability view unavailable: {e}")

            st.markdown("### Governance & reproducibility")
            try:
                if len(cache) == 4:
                    n_tr, n_te = len(y_tr), len(y_te)
                else:
                    n_tr, n_te = len(y_tr), len(y_te)
                split = ss.get("train_params", {}).get("test_size", "—")
                seed = ss.get("train_params", {}).get("random_state", "—")
                ts = ss.get("eval_timestamp", "—")
                st.write(f"- Train set: {n_tr}  |  Test set: {n_te}  |  Hold-out fraction: {split}")
                st.write(f"- Random seed: {seed}")
                st.write(f"- Training time: {ts}")
                st.write(f"- Adopted threshold: {ss.get('threshold', 0.5):.2f}")
            except Exception:
                st.caption("Governance info unavailable.")


def render_classify_stage():

    stage = STAGE_BY_KEY["classify"]

    with section_surface():
        overview_col, guidance_col = st.columns([3, 2], gap="large")
        with overview_col:
            st.subheader(f"{stage.icon} {stage.title} — Run the spam detector")
            render_eu_ai_quote(
                "The EU AI Act says “an AI system infers, from the input it receives, how to generate outputs such as content, predictions, recommendations or decisions.”"
            )
            st.write(
                "In this step, the system takes each email (title + body) as **input** and produces an **output**: "
                "a **prediction** (*Spam* or *Safe*) with a confidence score. By default, it also gives a **recommendation** "
                "about where to place the email (Spam or Inbox)."
            )
        with guidance_col:
            st.markdown("### Operating tips")
            st.markdown(
                "- Monitor predictions before enabling full autonomy.\n"
                "- Keep an eye on confidence values to decide when to intervene."
            )

    with section_surface():
        st.markdown("### Autonomy")
        default_high_autonomy = ss.get("autonomy", AUTONOMY_LEVELS[0]).startswith("High")
        auto_col, explain_col = st.columns([2, 3], gap="large")
        with auto_col:
            use_high_autonomy = st.toggle(
                "High autonomy (auto-move emails)", value=default_high_autonomy, key="use_high_autonomy"
            )
        with explain_col:
            if use_high_autonomy:
                ss["autonomy"] = AUTONOMY_LEVELS[1]
                st.success("High autonomy ON — the system will **move** emails to Spam or Inbox automatically.")
            else:
                ss["autonomy"] = AUTONOMY_LEVELS[0]
                st.warning("High autonomy OFF — review recommendations before moving emails.")
        if not ss.get("model"):
            st.warning("Train a model first in the **Train** tab.")
            st.stop()

    st.markdown("### Incoming preview")
    if not ss.get("incoming"):
        st.caption("No incoming emails. Add or import more in **📊 Prepare Data**, or paste a custom email below.")
        with st.expander("Add a custom email to process"):
            title_val = st.text_input("Title", key="use_custom_title", placeholder="Subject…")
            body_val = st.text_area("Body", key="use_custom_body", height=100, placeholder="Email body…")
            if st.button("Add to incoming", key="btn_add_to_incoming"):
                if title_val.strip() or body_val.strip():
                    ss["incoming"].append({"title": title_val.strip(), "body": body_val.strip()})
                    st.success("Added to incoming.")
                    _append_audit("incoming_added", {"title": title_val[:64]})
                else:
                    st.warning("Please provide at least a title or a body.")
    else:
        preview_n = min(10, len(ss["incoming"]))
        preview_df = pd.DataFrame(ss["incoming"][:preview_n])
        if not preview_df.empty:
            subtitle = f"Showing the first {preview_n} incoming emails (unlabeled)."
            render_email_inbox_table(preview_df, title="Incoming emails", subtitle=subtitle, columns=["title", "body"])
        else:
            render_email_inbox_table(pd.DataFrame(), title="Incoming emails", subtitle="No incoming emails available.")

        if st.button(f"Process {preview_n} email(s)", type="primary", key="btn_process_batch"):
            batch = ss["incoming"][:preview_n]
            y_hat, p_spam, p_safe = _predict_proba_batch(ss["model"], batch)
            thr = float(ss.get("threshold", 0.5))

            batch_rows: list[dict] = []
            moved_spam = moved_inbox = 0
            for idx, item in enumerate(batch):
                pred = y_hat[idx]
                prob_spam = float(p_spam[idx])
                prob_safe = float(p_safe[idx]) if hasattr(p_safe, "__len__") else float(1.0 - prob_spam)
                action = "Recommend: Spam" if prob_spam >= thr else "Recommend: Inbox"
                routed_to = None
                if ss["use_high_autonomy"]:
                    routed_to = "Spam" if prob_spam >= thr else "Inbox"
                    mailbox_record = {
                        "title": item.get("title", ""),
                        "body": item.get("body", ""),
                        "pred": pred,
                        "p_spam": round(prob_spam, 3),
                    }
                    if routed_to == "Spam":
                        ss["mail_spam"].append(mailbox_record)
                        moved_spam += 1
                    else:
                        ss["mail_inbox"].append(mailbox_record)
                        moved_inbox += 1
                    action = f"Moved: {routed_to}"
                row = {
                    "title": item.get("title", ""),
                    "body": item.get("body", ""),
                    "pred": pred,
                    "p_spam": round(prob_spam, 3),
                    "p_safe": round(prob_safe, 3),
                    "action": action,
                    "routed_to": routed_to,
                }
                batch_rows.append(row)

            ss["use_batch_results"] = batch_rows
            ss["incoming"] = ss["incoming"][preview_n:]
            if ss["use_high_autonomy"]:
                st.success(
                    f"Processed {preview_n} emails — decisions applied (Inbox: {moved_inbox}, Spam: {moved_spam})."
                )
                _append_audit(
                    "batch_processed_auto", {"n": preview_n, "inbox": moved_inbox, "spam": moved_spam}
                )
            else:
                st.info(f"Processed {preview_n} emails — recommendations ready.")
                _append_audit("batch_processed_reco", {"n": preview_n})

    if ss.get("use_batch_results"):
        with section_surface():
            st.markdown("### Results")
            df_res = pd.DataFrame(ss["use_batch_results"])
            show_cols = ["title", "pred", "p_spam", "action", "routed_to"]
            existing_cols = [col for col in show_cols if col in df_res.columns]
            display_df = df_res[existing_cols].rename(
                columns={"pred": "Prediction", "p_spam": "P(spam)", "action": "Action", "routed_to": "Routed"}
            )
            render_email_inbox_table(display_df, title="Batch results", subtitle="Predictions and actions just taken.")
            st.caption(
                "Each row shows the predicted label, confidence (P(spam)), and the recommendation or action taken."
            )

        nerd_mode_enabled = render_nerd_mode_toggle(
            key="nerd_mode_use",
            title="Nerd Mode — details for this batch",
            description="Inspect raw probabilities, distributions, and the session audit trail.",
            icon="🔬",
        )
        if nerd_mode_enabled:
            df_res = pd.DataFrame(ss["use_batch_results"])
            with section_surface():
                st.markdown("### Nerd Mode — batch diagnostics")
                col_nm1, col_nm2 = st.columns([2, 1])
                with col_nm1:
                    st.markdown("**Raw probabilities (per email)**")
                    detail_cols = ["title", "p_spam", "p_safe", "pred", "action", "routed_to"]
                    det_existing = [col for col in detail_cols if col in df_res.columns]
                    st.dataframe(df_res[det_existing], width="stretch", hide_index=True)
                with col_nm2:
                    st.markdown("**Batch metrics**")
                    n_items = len(df_res)
                    mean_conf = float(df_res["p_spam"].mean()) if "p_spam" in df_res else 0.0
                    n_spam = int((df_res["pred"] == "spam").sum()) if "pred" in df_res else 0
                    n_safe = n_items - n_spam
                    st.write(f"- Items: {n_items}")
                    st.write(f"- Predicted Spam: {n_spam} | Safe: {n_safe}")
                    st.write(f"- Mean P(spam): {mean_conf:.2f}")

                    if "p_spam" in df_res:
                        fig, ax = plt.subplots()
                        ax.hist(df_res["p_spam"], bins=10)
                        ax.set_xlabel("P(spam)")
                        ax.set_ylabel("Count")
                        ax.set_title("Spam score distribution")
                        st.pyplot(fig)

            with section_surface():
                st.markdown("### Why did it decide this way? (per email)")

                split_cache = ss.get("split_cache")
                train_texts: list[str] = []
                train_labels: list[str] = []
                train_emb: Optional[np.ndarray] = None
                if split_cache:
                    try:
                        if len(split_cache) == 6:
                            X_tr_t, _, X_tr_b, _, y_tr_vals, _ = split_cache
                            train_texts = [combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)]
                            train_labels = list(y_tr_vals)
                        else:
                            X_tr_texts, _, y_tr_vals, _ = split_cache
                            train_texts = list(X_tr_texts)
                            train_labels = list(y_tr_vals)
                        if train_texts:
                            train_emb = cache_train_embeddings(train_texts)
                    except Exception:
                        train_texts = []
                        train_labels = []
                        train_emb = None

                threshold_val = float(ss.get("threshold", 0.5))
                model_obj = ss.get("model")

                for email_idx, row in enumerate(ss["use_batch_results"]):
                    title = row.get("title", "")
                    body = row.get("body", "")
                    pred_label = row.get("pred", "")
                    p_spam_val = row.get("p_spam")
                    try:
                        p_spam_float = float(p_spam_val)
                    except (TypeError, ValueError):
                        p_spam_float = None

                    header = title or "(no subject)"
                    with st.container(border=True):
                        st.markdown(f"#### {header}")
                        st.caption(f"Predicted **{pred_label or '—'}**")

                        if p_spam_float is not None:
                            margin = p_spam_float - threshold_val
                            decision = "Spam" if p_spam_float >= threshold_val else "Safe"
                            st.markdown(
                                f"**Decision summary:** P(spam) = {p_spam_float:.2f} vs threshold {threshold_val:.2f} → **{decision}** "
                                f"(margin {margin:+.2f})"
                            )
                            if hasattr(ss.get("model"), "last_thr_eff"):
                                idxs_thr = getattr(ss.get("model"), "last_thr_eff", None)
                                if idxs_thr:
                                    idxs, thr_eff = idxs_thr
                                    try:
                                        pos = list(idxs).index(email_idx)
                                        st.caption(
                                            f"Effective threshold (with numeric micro-rules): {thr_eff[pos]:.2f}"
                                        )
                                    except Exception:
                                        pass
                        else:
                            st.caption("Probability not available for this email.")

                        if train_texts and train_labels:
                            try:
                                nn_examples = get_nearest_training_examples(
                                    combine_text(title, body), train_texts, train_labels, train_emb, k=3
                                )
                            except Exception:
                                nn_examples = []
                        else:
                            nn_examples = []

                        if nn_examples:
                            st.markdown("**Similar training emails (semantic evidence):**")
                            for example in nn_examples:
                                text_full = example["text"]
                                title_example = text_full.split("\n", 1)[0]
                                st.write(
                                    f"- *{title_example.strip() or '(no subject)'}* — label: **{example['label']}** "
                                    f"(sim {example['similarity']:.2f})"
                                )
                        else:
                            st.caption("No similar training emails available.")

                        if hasattr(model_obj, "scaler") and (
                            hasattr(model_obj, "lr_num") or hasattr(model_obj, "lr")
                        ):
                            contribs = numeric_feature_contributions(model_obj, title, body)
                            if contribs:
                                contribs_sorted = sorted(contribs, key=lambda x: x[2], reverse=True)
                                st.markdown("**Numeric cues (how they nudged the decision):**")
                                st.dataframe(
                                    pd.DataFrame(
                                        [
                                            {
                                                "feature": feat,
                                                "standardized": val,
                                                "toward_spam_logit": contrib,
                                            }
                                            for feat, val, contrib in contribs_sorted
                                        ]
                                    ).round(3),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                st.caption("Positive values push toward **Spam**; negative toward **Safe**.")
                            else:
                                st.caption("Numeric feature contributions unavailable for this email.")
                        else:
                            st.caption("Numeric cue breakdown requires the hybrid model.")

                        if model_obj is not None:
                            with st.expander("🖍️ Highlight influential words (experimental)", expanded=False):
                                st.caption(
                                    "Runs extra passes to see which words reduce P(spam) the most when removed."
                                )
                                if st.checkbox(
                                    "Compute highlights for this email", key=f"hl_{email_idx}", value=False
                                ):
                                    base_prob, rows_imp = top_token_importances(model_obj, title, body)
                                    if base_prob is None:
                                        st.caption("Unable to compute token importances for this model/email.")
                                    else:
                                        st.caption(
                                            f"Base P(spam) = {base_prob:.2f}. Higher importance means removing the word lowers P(spam) more."
                                        )
                                        if rows_imp:
                                            df_imp = pd.DataFrame(rows_imp[:10])
                                            st.dataframe(df_imp, use_container_width=True, hide_index=True)
                                        else:
                                            st.caption("No influential words found among the sampled tokens.")
                        else:
                            st.caption("Word highlights require a trained model.")

            with section_surface():
                st.markdown("### Audit trail (this session)")
                if ss.get("use_audit_log"):
                    st.dataframe(pd.DataFrame(ss["use_audit_log"]), width="stretch", hide_index=True)
                else:
                    st.caption("No events recorded yet.")

            exp_df = _export_batch_df(ss["use_batch_results"])
            csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
            json_bytes = json.dumps(ss["use_batch_results"], ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                "⬇️ Download results (CSV)", data=csv_bytes, file_name="batch_results.csv", mime="text/csv"
            )
            st.download_button(
                "⬇️ Download results (JSON)", data=json_bytes, file_name="batch_results.json", mime="application/json"
            )

    st.markdown("### Adaptiveness — learn from your corrections")
    render_eu_ai_quote(
        "The EU AI Act says “AI systems may exhibit adaptiveness.” Enable adaptiveness to confirm or correct results; the model can retrain on your feedback."
    )
    def _handle_stage_adaptive_change() -> None:
        _set_adaptive_state(ss.get("adaptive_stage", ss.get("adaptive", False)), source="stage")

    st.toggle(
        "Enable adaptiveness (learn from feedback)",
        value=bool(ss.get("adaptive", False)),
        key="adaptive_stage",
        on_change=_handle_stage_adaptive_change,
    )
    use_adaptiveness = bool(ss.get("adaptive", False))

    if use_adaptiveness and ss.get("use_batch_results"):
        st.markdown("#### Review and give feedback")
        for i, row in enumerate(ss["use_batch_results"]):
            with st.container(border=True):
                st.markdown(f"**Title:** {row.get('title', '')}")
                pspam_value = row.get("p_spam")
                if isinstance(pspam_value, (int, float)):
                    pspam_text = f"{pspam_value:.2f}"
                else:
                    pspam_text = pspam_value
                action_display = row.get("action", "")
                pred_display = row.get("pred", "")
                st.markdown(
                    f"**Predicted:** {pred_display}  •  **P(spam):** {pspam_text}  •  **Action:** {action_display}"
                )
                col_a, col_b, col_c = st.columns(3)
                if col_a.button("Confirm", key=f"use_confirm_{i}"):
                    _append_audit("confirm_label", {"i": i, "pred": pred_display})
                    st.toast("Thanks — recorded your confirmation.", icon="✅")
                if col_b.button("Correct → Spam", key=f"use_correct_spam_{i}"):
                    ss["labeled"].append(
                        {"title": row.get("title", ""), "body": row.get("body", ""), "label": "spam"}
                    )
                    _append_audit("correct_label", {"i": i, "new": "spam"})
                    st.toast("Recorded correction → Spam.", icon="✍️")
                if col_c.button("Correct → Safe", key=f"use_correct_safe_{i}"):
                    ss["labeled"].append(
                        {"title": row.get("title", ""), "body": row.get("body", ""), "label": "safe"}
                    )
                    _append_audit("correct_label", {"i": i, "new": "safe"})
                    st.toast("Recorded correction → Safe.", icon="✍️")

        if st.button("🔁 Retrain now with feedback", key="btn_retrain_feedback"):
            df_all = pd.DataFrame(ss["labeled"])
            if not df_all.empty and len(df_all["label"].unique()) >= 2:
                params = ss.get("train_params", {})
                test_size = float(params.get("test_size", 0.30))
                random_state = int(params.get("random_state", 42))
                max_iter = int(params.get("max_iter", 1000))
                C_value = float(params.get("C", 1.0))

                titles = df_all["title"].fillna("").tolist()
                bodies = df_all["body"].fillna("").tolist()
                labels = df_all["label"].tolist()
                X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = train_test_split(
                    titles,
                    bodies,
                    labels,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=labels,
                )

                gp = ss.get("guard_params", {})
                model = HybridEmbedFeatsLogReg(
                    max_iter=max_iter,
                    C=C_value,
                    random_state=random_state,
                    numeric_assist_center=float(
                        gp.get("assist_center", float(ss.get("threshold", 0.6)))
                    ),
                    uncertainty_band=float(gp.get("uncertainty_band", 0.08)),
                    numeric_scale=float(gp.get("numeric_scale", 0.5)),
                    numeric_logit_cap=float(gp.get("numeric_logit_cap", 1.0)),
                    combine_strategy=str(gp.get("combine_strategy", "blend")),
                    shift_suspicious_tld=float(gp.get("shift_suspicious_tld", -0.04)),
                    shift_many_links=float(gp.get("shift_many_links", -0.03)),
                    shift_calm_text=float(gp.get("shift_calm_text", +0.02)),
                )
                try:
                    pass
                except Exception:
                    pass
                model = model.fit(X_tr_t, X_tr_b, y_tr)
                model.apply_numeric_adjustments(ss["numeric_adjustments"])
                ss["model"] = model
                ss["split_cache"] = (X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te)
                ss["eval_timestamp"] = datetime.now().isoformat(timespec="seconds")
                ss["eval_temp_threshold"] = float(ss.get("threshold", 0.6))
                ss["train_story_run_id"] = uuid4().hex
                for key in (
                    "meaning_map_show_examples",
                    "meaning_map_show_centers",
                    "meaning_map_highlight_borderline",
                    "meaning_map_show_pair_trigger",
                ):
                    ss.pop(key, None)
                st.success("Adaptive learning: model retrained with your feedback.")
                _append_audit("retrain_feedback", {"n_labeled": len(df_all)})
            else:
                st.warning("Need both classes (spam & safe) in labeled data to retrain.")

    st.markdown("### 📥 Mailboxes")
    inbox_tab, spam_tab = st.tabs(
        [
            f"Inbox (safe) — {len(ss['mail_inbox'])}",
            f"Spam — {len(ss['mail_spam'])}",
        ]
    )
    with inbox_tab:
        render_mailbox_panel(
            ss.get("mail_inbox"),
            mailbox_title="Inbox (safe)",
            filled_subtitle="Messages the system kept in your inbox.",
            empty_subtitle="Inbox is empty so far.",
        )
    with spam_tab:
        render_mailbox_panel(
            ss.get("mail_spam"),
            mailbox_title="Spam",
            filled_subtitle="What the system routed away from the inbox.",
            empty_subtitle="No emails have been routed to spam yet.",
        )

    st.caption(
        f"Threshold used for routing: **{float(ss.get('threshold', 0.5)):.2f}**. "
        "Adjust it in **🧪 Evaluate** to change how cautious/aggressive the system is."
    )

def render_model_card_stage():


    with section_surface():
        st.subheader("Model Card — transparency")
        guidance_popover("Transparency", """
Model cards summarize intended purpose, data, metrics, autonomy & adaptiveness settings.
They help teams reason about risks and the appropriate oversight controls.
""")
        algo = "Sentence embeddings (MiniLM) + standardized numeric cues + Logistic Regression"
        n_samples = len(ss["labeled"])
        labels_present = sorted({row["label"] for row in ss["labeled"]}) if ss["labeled"] else []
        metrics_text = ""
        holdout_n = 0
        if ss.get("model") and ss.get("split_cache"):
            _, X_te_t, _, X_te_b, _, y_te = ss["split_cache"]
            y_pred = ss["model"].predict(X_te_t, X_te_b)
            holdout_n = len(y_te)
            metrics_text = f"Accuracy on hold‑out: {accuracy_score(y_te, y_pred):.2%} (n={holdout_n})"
        snapshot_id = ss.get("active_dataset_snapshot")
        snapshot_entry = None
        if snapshot_id:
            snapshot_entry = next((snap for snap in ss.get("datasets", []) if snap.get("id") == snapshot_id), None)
        dataset_config_for_card = (snapshot_entry or {}).get("config", ss.get("dataset_config", DEFAULT_DATASET_CONFIG))
        dataset_config_json = json.dumps(dataset_config_for_card, indent=2, sort_keys=True)
        snapshot_label = snapshot_id if snapshot_id else "— (save one in Prepare Data)"

        card_md = f"""
# Model Card — demistifAI (Spam Detector)
**Intended purpose**: Educational demo to illustrate the AI Act definition of an **AI system** via a spam classifier.

**Algorithm**: {algo}
**Features**: Sentence embeddings (MiniLM) concatenated with small, interpretable numeric features:
- num_links_external, has_suspicious_tld, punct_burst_ratio, money_symbol_count, urgency_terms_count.
These are standardized and combined with the embedding before a linear classifier.

**Classes**: spam, safe
**Dataset size**: {n_samples} labeled examples
**Classes present**: {', '.join(labels_present) if labels_present else '[not trained]'}

**Key metrics**: {metrics_text or 'Train a model to populate metrics.'}

**Autonomy**: {ss['autonomy']} (threshold={ss['threshold']:.2f})
**Adaptiveness**: {'Enabled' if ss['adaptive'] else 'Disabled'} (learn from user corrections).

**Data**: user-augmented seed set (title + body); session-only.
**Dataset snapshot ID**: {snapshot_label}
**Dataset config**:
```
{dataset_config_json}
```
**Known limitations**: tiny datasets; vocabulary sensitivity; no MIME/URL/metadata features.

**AI Act mapping**
- **Machine-based system**: Streamlit app (software) running on cloud runtime (hardware).
- **Inference**: model learns patterns from labeled examples.
- **Output generation**: predictions + confidence; used to recommend/route emails.
    - **Varying autonomy**: user selects autonomy level; at high autonomy, the system acts.
- **Adaptiveness**: optional feedback loop that updates the model.
"""
        content_col, highlight_col = st.columns([3, 2], gap="large")
        with content_col:
            st.markdown(card_md)
            download_text(card_md, "model_card.md", "Download model_card.md")
        with highlight_col:
            st.markdown(
                """
                <div class="info-metric-grid">
                    <div class="info-metric-card">
                        <div class="label">Labeled dataset</div>
                        <div class="value">{samples}</div>
                    </div>
                    <div class="info-metric-card">
                        <div class="label">Hold-out size</div>
                        <div class="value">{holdout}</div>
                    </div>
                    <div class="info-metric-card">
                        <div class="label">Autonomy</div>
                        <div class="value">{autonomy}</div>
                    </div>
                    <div class="info-metric-card">
                        <div class="label">Adaptiveness</div>
                        <div class="value">{adaptive}</div>
                    </div>
                </div>
                """.format(
                    samples=n_samples,
                    holdout=holdout_n or "—",
                    autonomy=html.escape(ss.get("autonomy", AUTONOMY_LEVELS[0])),
                    adaptive="On" if ss.get("adaptive") else "Off",
                ),
                unsafe_allow_html=True,
            )

        with highlight_col:
            st.markdown("#### Dataset provenance")
            if snapshot_id:
                st.write(f"Snapshot ID: `{snapshot_id}`")
            else:
                st.write("Snapshot ID: — (save one in Prepare Data → Snapshot & provenance).")
            st.code(dataset_config_json, language="json")


def render_stage_navigation_controls(active_stage_key: str) -> None:
    """Display previous/next controls for the staged experience."""

    stage_keys = [stage.key for stage in STAGES]
    if active_stage_key not in stage_keys:
        return

    stage_index = stage_keys.index(active_stage_key)
    current_stage = STAGE_BY_KEY[active_stage_key]
    previous_stage = STAGE_BY_KEY.get(stage_keys[stage_index - 1]) if stage_index > 0 else None
    next_stage = (
        STAGE_BY_KEY.get(stage_keys[stage_index + 1])
        if stage_index < len(stage_keys) - 1
        else None
    )

    prev_col, info_col, next_col = st.columns([1, 2, 1], gap="large")

    with prev_col:
        if previous_stage is not None:
            st.button(
                f"⬅️ {previous_stage.icon} {previous_stage.title}",
                key=f"stage_nav_prev_{previous_stage.key}",
                on_click=set_active_stage,
                args=(previous_stage.key,),
                use_container_width=True,
            )

    with info_col:
        st.markdown(
            f"""
            <div class="stage-navigation-info">
                <div class="stage-navigation-step">Stage {stage_index + 1} of {len(stage_keys)}</div>
                <div class="stage-navigation-title">{html.escape(current_stage.icon)} {html.escape(current_stage.title)}</div>
                <p class="stage-navigation-description">{html.escape(current_stage.description)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with next_col:
        if next_stage is not None:
            st.button(
                f"{next_stage.icon} {next_stage.title} ➡️",
                key=f"stage_nav_next_{next_stage.key}",
                on_click=set_active_stage,
                args=(next_stage.key,),
                use_container_width=True,
            )


def _render_train_stage_wrapper() -> None:
    render_train_stage(
        ss,
        streamlit_rerun=_streamlit_rerun,
        has_embed=has_embed,
        has_langdetect=has_langdetect,
        render_eu_ai_quote=render_eu_ai_quote,
        render_language_mix_chip_rows=render_language_mix_chip_rows,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        section_surface=section_surface,
        summarize_language_mix=summarize_language_mix,
    )


STAGE_RENDERERS = {
    'intro': render_intro_stage,
    'overview': render_overview_stage,
    'data': render_data_stage,
    'train': _render_train_stage_wrapper,
    'evaluate': render_evaluate_stage,
    'classify': render_classify_stage,
    'model_card': render_model_card_stage,
}


active_stage = ss['active_stage']
renderer = STAGE_RENDERERS.get(active_stage, render_intro_stage)

if ss.pop("stage_scroll_to_top", False):
    components.html(
        """
        <script>
        (function() {
            const main = window.parent.document.querySelector('section.main');
            if (main && typeof main.scrollTo === 'function') {
                main.scrollTo({ top: 0, behavior: 'smooth' });
            }
            if (window.parent && typeof window.parent.scrollTo === 'function') {
                window.parent.scrollTo({ top: 0, behavior: 'smooth' });
            }
        })();
        </script>
        """,
        height=0,
    )

renderer()
render_stage_navigation_controls(active_stage)

st.markdown("---")
st.caption("© demistifAI — Built for interactive learning and governance discussions.")
