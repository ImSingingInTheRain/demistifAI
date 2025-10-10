from __future__ import annotations

import base64
import html
import math
import json
import logging
import random
import string
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

from demistifai.constants import (
    APP_THEME_CSS,
    AUTONOMY_LEVELS,
    BRANDS,
    CLASSES,
    COURIERS,
    DATASET_LEGIT_DOMAINS,
    DATASET_SUSPICIOUS_TLDS,
    EMAIL_INBOX_TABLE_CSS,
    LIFECYCLE_CYCLE_CSS,
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


logger = logging.getLogger(__name__)


def callable_or_attr(target: Any, attr: str | None = None) -> bool:
    """Return True if ``target`` (or one of its attributes) is callable."""

    try:
        value = getattr(target, attr) if attr else target
    except Exception:
        return False
    return callable(value)


from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
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


def _make_selection_point(
    fields: list[str], *, on: str, empty: str
) -> Any | None:
    """Create an Altair selection/parameter that works across Altair versions."""

    try:
        if hasattr(alt, "selection_point"):
            return alt.selection_point(fields=fields, on=on, empty=empty)
    except Exception:
        pass

    # Altair < 5 uses the legacy selection API
    try:
        if hasattr(alt, "selection_single"):
            return alt.selection_single(fields=fields, on=on, empty=empty)
    except Exception:
        pass

    try:
        if hasattr(alt, "selection"):
            return alt.selection(type="single", fields=fields, on=on, empty=empty)
    except Exception:
        pass

    return None


def _chart_add_params(chart: alt.Chart, *params: Any) -> alt.Chart:
    """Attach parameters/selections using whichever API is available."""

    valid = [param for param in params if param is not None]
    if not valid:
        return chart

    if hasattr(chart, "add_params"):
        return chart.add_params(*valid)
    if hasattr(chart, "add_selection"):
        return chart.add_selection(*valid)
    return chart


def _combine_selections(*selections: Any) -> Any | None:
    """Return the logical union of selections/parameters that support it."""

    active = [sel for sel in selections if sel is not None]
    if not active:
        return None

    combined = active[0]
    for sel in active[1:]:
        try:
            combined = combined | sel
        except Exception:
            # If the union operation is unavailable, prefer the last selection
            combined = sel
    return combined


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


def _excerpt(text: str, n: int = 120) -> str:
    text = (text or "").replace("\\n", " ").strip()
    return text[:n] + ("…" if len(text) > n else "")


def _join_phrases(parts: List[str]) -> str:
    cleaned = [p.strip().rstrip(".") for p in parts if p]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return f"{cleaned[0]}."
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}."
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}."


FEATURE_REASON_SPAM = {
    "num_links_external": "Contains multiple external links",
    "has_suspicious_tld": "Links point to risky domains",
    "punct_burst_ratio": "Uses lots of !!! or $$$",
    "money_symbol_count": "Mentions money terms",
    "urgency_terms_count": "Pushes urgent wording",
}

FEATURE_REASON_SAFE = {
    "num_links_external": "Few links to distract",
    "has_suspicious_tld": "No risky domains detected",
    "punct_burst_ratio": "Calm punctuation",
    "money_symbol_count": "No money-talk cues",
    "urgency_terms_count": "Neutral tone without urgency",
}


def _reason_from_contributions(label: str, contributions: List[Tuple[str, float, float]]) -> str:
    if not contributions:
        return "Mostly positioned by wording similarity."

    threshold = 0.08
    phrases: List[str] = []
    if label == "spam":
        for feat, _z, contrib in contributions:
            if contrib > threshold:
                phrases.append(FEATURE_REASON_SPAM.get(feat, FEATURE_DISPLAY_NAMES.get(feat, feat)))
    else:
        for feat, _z, contrib in contributions:
            if contrib < -threshold:
                phrases.append(FEATURE_REASON_SAFE.get(feat, FEATURE_DISPLAY_NAMES.get(feat, feat)))

    phrases = [p for p in phrases if p]
    if not phrases:
        return "Mostly positioned by wording similarity."
    summary = _join_phrases(phrases[:3])
    if not summary:
        return "Mostly positioned by wording similarity."
    return f"Signals: {summary}"


def _sample_indices_by_label(labels: List[str], limit: int) -> List[int]:
    n = len(labels)
    if n <= limit:
        return list(range(n))

    rng = np.random.default_rng(42)
    per_label: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        per_label.setdefault(label, []).append(idx)

    sampled: List[int] = []
    total = float(n)
    for label, idxs in per_label.items():
        if not idxs:
            continue
        target = max(1, round(limit * (len(idxs) / total)))
        target = min(target, len(idxs))
        picked = rng.choice(idxs, size=target, replace=False)
        sampled.extend(int(i) for i in picked)

    if len(sampled) > limit:
        sampled = rng.choice(sampled, size=limit, replace=False).tolist()

    return sorted(sampled)


def _reduce_embeddings_to_2d(embeddings: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    if embeddings.ndim != 2 or embeddings.shape[1] <= 2:
        coords = embeddings[:, :2] if embeddings.ndim == 2 else np.empty((0, 2))
        return coords, {"kind": "raw"}

    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
        coords = reducer.fit_transform(embeddings)
        return coords, {"kind": "umap"}
    except Exception:
        pass

    try:
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
        return coords, {
            "kind": "pca",
            "components": getattr(reducer, "components_", None),
            "mean": getattr(reducer, "mean_", None),
        }
    except Exception:
        return np.empty((0, 2)), {"kind": "failed"}


def _line_box_intersections(
    bounds: Tuple[float, float, float, float],
    weight: np.ndarray,
    intercept: float,
) -> List[Tuple[float, float]]:
    x_min, x_max, y_min, y_max = bounds
    w_x, w_y = weight
    points: List[Tuple[float, float]] = []
    tol = 1e-9

    if abs(w_y) > tol:
        for x in (x_min, x_max):
            y = -(w_x * x + intercept) / w_y
            if y_min - 1e-6 <= y <= y_max + 1e-6:
                points.append((float(x), float(y)))

    if abs(w_x) > tol:
        for y in (y_min, y_max):
            x = -(w_y * y + intercept) / w_x
            if x_min - 1e-6 <= x <= x_max + 1e-6:
                points.append((float(x), float(y)))

    # Deduplicate points (within tolerance)
    unique: List[Tuple[float, float]] = []
    for x, y in points:
        if not any(abs(x - ux) < 1e-6 and abs(y - uy) < 1e-6 for ux, uy in unique):
            unique.append((x, y))

    if len(unique) >= 2:
        # Sort for stable plotting
        unique.sort(key=lambda item: (item[0], item[1]))
        return unique[:2]

    # Fallback: use center with perpendicular direction
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    perp = np.array([-w_y, w_x], dtype=float)
    norm = np.linalg.norm(perp)
    if norm < tol:
        perp = np.array([1.0, 0.0])
        norm = 1.0
    perp /= norm
    span = max(x_max - x_min, y_max - y_min) or 1.0
    half = span / 2.0
    return [
        (float(cx - perp[0] * half), float(cy - perp[1] * half)),
        (float(cx + perp[0] * half), float(cy + perp[1] * half)),
    ]


def _clip_polygon_with_halfplane(
    polygon: List[Tuple[float, float]],
    weight: np.ndarray,
    intercept: float,
    threshold: float,
    keep_leq: bool,
) -> List[Tuple[float, float]]:
    if not polygon:
        return []

    def eval_point(pt: Tuple[float, float]) -> float:
        return float(np.dot(weight, pt) + intercept - threshold)

    output: List[Tuple[float, float]] = []
    prev = polygon[-1]
    prev_val = eval_point(prev)
    prev_inside = prev_val <= 0 if keep_leq else prev_val >= 0

    for curr in polygon:
        curr_val = eval_point(curr)
        curr_inside = curr_val <= 0 if keep_leq else curr_val >= 0

        if curr_inside:
            if not prev_inside:
                denom = curr_val - prev_val
                if abs(denom) > 1e-9:
                    ratio = prev_val / (prev_val - curr_val)
                    ix = prev[0] + (curr[0] - prev[0]) * ratio
                    iy = prev[1] + (curr[1] - prev[1]) * ratio
                    output.append((float(ix), float(iy)))
            output.append((float(curr[0]), float(curr[1])))
        elif prev_inside:
            denom = curr_val - prev_val
            if abs(denom) > 1e-9:
                ratio = prev_val / (prev_val - curr_val)
                ix = prev[0] + (curr[0] - prev[0]) * ratio
                iy = prev[1] + (curr[1] - prev[1]) * ratio
                output.append((float(ix), float(iy)))

        prev = curr
        prev_val = curr_val
        prev_inside = curr_inside

    return output


def _compute_decision_boundary_overlay(
    coords: np.ndarray,
    logits: np.ndarray,
    model: Any,
    projection_meta: Dict[str, Any],
    probability_margin: float = 0.1,
) -> Optional[Dict[str, Any]]:
    if coords.size == 0 or logits.size == 0:
        return None

    try:
        weight_full = None
        intercept_full = None
        lr_text = getattr(model, "lr_text_base", None)
        if lr_text is not None and hasattr(lr_text, "coef_"):
            coef = np.asarray(lr_text.coef_, dtype=float)
            if coef.ndim == 2:
                coef = coef.reshape(-1)
            if coef.size >= coords.shape[1]:
                weight_full = coef
                intercept_arr = np.asarray(getattr(lr_text, "intercept_", [0.0]), dtype=float)
                intercept_full = float(intercept_arr.reshape(-1)[0])
    except Exception:
        weight_full = None
        intercept_full = None

    weight_2d: Optional[np.ndarray] = None
    intercept_2d: Optional[float] = None
    projection_kind = (projection_meta or {}).get("kind") if projection_meta else None

    if weight_full is not None and intercept_full is not None:
        if projection_kind == "pca":
            components = projection_meta.get("components") if projection_meta else None
            mean_vec = projection_meta.get("mean") if projection_meta else None
            if components is not None and mean_vec is not None:
                components_arr = np.asarray(components, dtype=float)
                mean_arr = np.asarray(mean_vec, dtype=float)
                if components_arr.shape[0] >= 2 and components_arr.shape[1] == weight_full.shape[0]:
                    weight_2d = components_arr @ weight_full
                    intercept_2d = float(intercept_full + float(weight_full @ mean_arr))
        elif projection_kind == "raw":
            weight_2d = np.asarray(weight_full[:2], dtype=float)
            intercept_2d = float(intercept_full)

    if weight_2d is None or intercept_2d is None:
        # Approximate using least squares in the 2D space
        A = np.column_stack([coords, np.ones(coords.shape[0])])
        try:
            sol, *_ = np.linalg.lstsq(A, logits, rcond=None)
            weight_2d = np.asarray(sol[:2], dtype=float)
            intercept_2d = float(sol[2])
        except Exception:
            return None

    if weight_2d is None:
        return None

    norm = float(np.linalg.norm(weight_2d))
    if not math.isfinite(norm) or norm < 1e-9:
        return None

    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    span = max(x_max - x_min, y_max - y_min)
    pad = span * 0.08 if span > 0 else 1.0
    bounds = (x_min - pad, x_max + pad, y_min - pad, y_max + pad)

    line_points = _line_box_intersections(bounds, weight_2d, intercept_2d)

    corner_polygon = [
        (bounds[0], bounds[2]),
        (bounds[1], bounds[2]),
        (bounds[1], bounds[3]),
        (bounds[0], bounds[3]),
    ]

    spam_polygon = _clip_polygon_with_halfplane(
        corner_polygon, weight_2d, intercept_2d, 0.0, keep_leq=False
    )
    safe_polygon = _clip_polygon_with_halfplane(
        corner_polygon, weight_2d, intercept_2d, 0.0, keep_leq=True
    )

    prob_margin = max(1e-6, min(0.49, float(probability_margin)))
    logit_margin = math.log((0.5 + prob_margin) / (0.5 - prob_margin))
    margin_distance = logit_margin / norm

    band_polygon = _clip_polygon_with_halfplane(
        corner_polygon, weight_2d, intercept_2d, logit_margin, keep_leq=True
    )
    band_polygon = _clip_polygon_with_halfplane(
        band_polygon, weight_2d, intercept_2d, -logit_margin, keep_leq=False
    )

    line_df = pd.DataFrame(line_points, columns=["x", "y"])

    shading_rows: List[Dict[str, Any]] = []
    for side, polygon in ("spam", spam_polygon), ("safe", safe_polygon):
        for order, (x, y) in enumerate(polygon):
            shading_rows.append({"side": side, "order": order, "x": x, "y": y})
    shading_df = pd.DataFrame(shading_rows)

    band_rows: List[Dict[str, Any]] = []
    for order, (x, y) in enumerate(band_polygon):
        band_rows.append({"order": order, "x": x, "y": y})
    band_df = pd.DataFrame(band_rows)

    return {
        "weight": weight_2d,
        "intercept": intercept_2d,
        "norm": norm,
        "line_df": line_df,
        "shading_df": shading_df,
        "band_df": band_df,
        "margin_probability": prob_margin,
        "margin_logit": logit_margin,
        "margin_distance": margin_distance,
        "bounds": bounds,
    }


def _prepare_meaning_map(
    titles: List[str],
    bodies: List[str],
    labels: List[str],
    model: Any | None,
    *,
    max_points: int = 500,
) -> tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    rows: List[Tuple[int, str, str, str]] = []
    for idx, (title, body, label) in enumerate(zip(titles, bodies, labels)):
        if label not in {"spam", "safe"}:
            continue
        rows.append((idx, title or "", body or "", label))

    if not rows:
        raise RuntimeError("Meaning Map is unavailable—no labeled emails to display.")

    labels_present = {row[3] for row in rows}
    if len(labels_present) < 2:
        raise RuntimeError("Need both Spam and Safe examples to draw the map.")

    base_indices = list(range(len(rows)))
    if len(rows) > max_points:
        sampled_indices = _sample_indices_by_label([row[3] for row in rows], max_points)
    else:
        sampled_indices = base_indices

    sampled_rows = [rows[i] for i in sampled_indices]
    sampled_labels = [row[3] for row in sampled_rows]
    sampled_titles = [row[1] for row in sampled_rows]
    sampled_bodies = [row[2] for row in sampled_rows]
    sampled_texts = [combine_text(t, b) for t, b in zip(sampled_titles, sampled_bodies)]

    try:
        embeddings = cache_train_embeddings(sampled_texts)
    except Exception as exc:
        raise RuntimeError(f"Meaning Map unavailable ({exc}).") from exc

    if getattr(embeddings, "size", 0) == 0:
        raise RuntimeError("Meaning Map unavailable (embeddings missing).")

    coords, projection_meta = _reduce_embeddings_to_2d(np.asarray(embeddings, dtype=np.float32))
    if coords.shape[0] != len(sampled_rows):
        raise RuntimeError("Meaning Map unavailable (projection failed).")

    df = pd.DataFrame(
        {
            "plot_index": range(len(sampled_rows)),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "label": sampled_labels,
            "label_title": [label.title() for label in sampled_labels],
            "subject_full": sampled_titles,
            "subject_tooltip": [
                (title[:80] + "…") if len(title) > 80 else title for title in sampled_titles
            ],
            "body_excerpt": [
                _excerpt(body, n=180) if body else "(no body text)" for body in sampled_bodies
            ],
            "body_full": sampled_bodies,
        }
    )

    reasons: List[str] = []
    for title, body, label in zip(sampled_titles, sampled_bodies, sampled_labels):
        contribs = numeric_feature_contributions(model, title, body) if model else []
        reasons.append(_reason_from_contributions(label, contribs))
    df["reason"] = reasons

    class_counts = df["label"].value_counts().to_dict()

    centroid_df = (
        df.groupby("label", as_index=False)[["x", "y"]].mean().assign(kind="centroid")
    )

    centroid_distance = None
    if len(centroid_df) == 2:
        pts = centroid_df[["x", "y"]].to_numpy()
        centroid_distance = float(np.linalg.norm(pts[0] - pts[1]))

    pair_info: Optional[Dict[str, Any]] = None
    emb_array = np.asarray(embeddings, dtype=np.float32)
    for target_label in ("spam", "safe"):
        idxs = [i for i, lbl in enumerate(sampled_labels) if lbl == target_label]
        if len(idxs) < 2:
            continue
        subset = emb_array[idxs]
        sims = subset @ subset.T
        np.fill_diagonal(sims, -np.inf)
        flat = sims.argmax()
        if not np.isfinite(sims.flat[flat]):
            continue
        a, b = divmod(flat, sims.shape[1])
        idx_a = idxs[a]
        idx_b = idxs[b]
        pair_info = {
            "indices": [idx_a, idx_b],
            "label": target_label,
            "subjects": [sampled_titles[idx_a], sampled_titles[idx_b]],
            "coords": coords[[idx_a, idx_b]].tolist(),
        }
        break

    boundary_info: Dict[str, Any] | None = None
    logits: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None
    if model is not None:
        try:
            logits_raw = model.predict_logit(sampled_texts)
            logits = np.asarray(logits_raw, dtype=float).reshape(-1)
        except Exception:
            logits = None

        if logits is None:
            try:
                probs_raw = model.predict_proba(sampled_titles, sampled_bodies)
                probs_raw = np.asarray(probs_raw, dtype=float)
                if probs_raw.ndim == 2 and probs_raw.shape[1] >= 2:
                    probs = probs_raw[:, getattr(model, "_i_spam", -1)]
                elif probs_raw.ndim == 2 and probs_raw.shape[1] == 1:
                    probs = probs_raw[:, 0]
                else:
                    probs = None
            except Exception:
                probs = None
            if probs is not None:
                probs = np.clip(probs, 1e-6, 1 - 1e-6)
                logits = np.log(probs / (1 - probs))

        if logits is not None and logits.shape[0] == coords.shape[0]:
            probs = 1.0 / (1.0 + np.exp(-logits))
            df["spam_probability"] = probs
            df["logit"] = logits
            try:
                predicted = model.predict(sampled_titles, sampled_bodies)
                if isinstance(predicted, np.ndarray):
                    predicted = predicted.tolist()
            except Exception:
                predicted = None
            if isinstance(predicted, Iterable) and len(predicted) == len(sampled_rows):
                df["predicted_label"] = list(predicted)
            else:
                df["predicted_label"] = ["spam" if val >= 0 else "safe" for val in logits]

            boundary_info = _compute_decision_boundary_overlay(
                coords,
                logits,
                model,
                projection_meta,
            )
            if boundary_info:
                norm_val = float(boundary_info.get("norm", 0.0) or 0.0)
                if norm_val > 0 and np.all(np.isfinite(logits)):
                    distances = logits / norm_val
                    df["distance_to_line"] = distances
                    df["distance_abs"] = np.abs(distances)
                else:
                    df["distance_to_line"] = np.nan
                    df["distance_abs"] = np.nan
                logit_margin = float(boundary_info.get("margin_logit", 0.0) or 0.0)
                if np.all(np.isfinite(logits)):
                    df["borderline"] = np.abs(logits) <= logit_margin
                else:
                    df["borderline"] = False
        if "distance_to_line" not in df:
            df["distance_to_line"] = np.nan
        if "distance_abs" not in df:
            df["distance_abs"] = np.nan
        if "borderline" not in df:
            df["borderline"] = False
    if "predicted_label" not in df:
        df["predicted_label"] = df["label"].tolist()
    df["predicted_label_title"] = [str(lbl).title() for lbl in df["predicted_label"]]

    if "spam_probability" not in df:
        df["spam_probability"] = np.nan
    if "logit" not in df:
        df["logit"] = np.nan
    if "distance_abs" not in df:
        df["distance_abs"] = np.nan
    if "borderline" not in df:
        df["borderline"] = False

    df["distance_display"] = [
        f"{abs(val):.2f}" if isinstance(val, (int, float)) and math.isfinite(val) else "–"
        for val in df["distance_to_line"]
    ]

    meta = {
        "class_counts": class_counts,
        "centroids": centroid_df,
        "centroid_distance": centroid_distance,
        "pair": pair_info,
        "total": len(rows),
        "shown": len(sampled_rows),
        "sampled": len(sampled_rows) < len(rows),
        "projection": projection_meta,
        "boundary": boundary_info,
        "embedding_backend": embedding_backend_info(),
    }

    return df, meta



def _build_meaning_map_chart(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    *,
    show_examples: bool,
    show_class_centers: bool,
    highlight_borderline: bool = False,
) -> Optional[alt.VConcatChart]:
    if df is None or df.empty:
        return None

    required_cols = {"x", "y", "label", "plot_index"}
    if not required_cols.issubset(df.columns):
        return None

    df = df.copy()
    defaults: Dict[str, Any] = {
        "label_title": "",
        "predicted_label_title": "",
        "spam_probability": np.nan,
        "distance_display": "–",
        "subject_tooltip": "",
        "subject_full": "",
        "body_excerpt": "",
        "reason": "",
        "borderline": False,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    color_scale = alt.Scale(domain=["spam", "safe"], range=["#ef4444", "#3b82f6"])
    hover = _make_selection_point(["plot_index"], on="mouseover", empty="none")
    select = _make_selection_point(["plot_index"], on="click", empty="none")
    combined_selection = _combine_selections(select, hover)

    base = alt.Chart(df)
    tooltip_fields: List[Any] = []
    if "subject_tooltip" in df.columns:
        tooltip_fields.append(alt.Tooltip("subject_tooltip:N", title="Subject"))
    if "label_title" in df.columns:
        tooltip_fields.append(alt.Tooltip("label_title:N", title="True label"))
    if "predicted_label_title" in df.columns:
        tooltip_fields.append(alt.Tooltip("predicted_label_title:N", title="Model prediction"))
    if "spam_probability" in df.columns:
        tooltip_fields.append(alt.Tooltip("spam_probability:Q", title="Spam probability", format=".2f"))
    if "distance_display" in df.columns:
        tooltip_fields.append(alt.Tooltip("distance_display:N", title="Distance to line"))

    scatter = base.mark_circle(size=80).encode(
        x=alt.X(
            "x:Q",
            axis=alt.Axis(title="Meaning dimension 1", grid=False, ticks=False, labels=False),
        ),
        y=alt.Y(
            "y:Q",
            axis=alt.Axis(title="Meaning dimension 2", grid=False, ticks=False, labels=False),
        ),
        color=alt.Color("label:N", scale=color_scale, legend=None),
        tooltip=tooltip_fields,
        opacity=
            alt.condition(combined_selection, alt.value(0.95), alt.value(0.45))
            if combined_selection is not None
            else alt.value(0.65),
        stroke=
            alt.condition(
                combined_selection,
                alt.value("#ffffff"),
                alt.value("rgba(0,0,0,0)"),
            )
            if combined_selection is not None
            else alt.value("rgba(0,0,0,0)"),
        strokeWidth=
            alt.condition(combined_selection, alt.value(2), alt.value(0))
            if combined_selection is not None
            else alt.value(0),
    )
    scatter = _chart_add_params(scatter, hover, select)

    layers = [scatter]

    boundary = meta.get("boundary") if isinstance(meta, dict) else None
    if isinstance(boundary, dict):
        shading_df = boundary.get("shading_df")
        if isinstance(shading_df, pd.DataFrame) and not shading_df.empty:
            shading_layer = (
                alt.Chart(shading_df)
                .mark_line(
                    filled=True,
                    opacity=0.18,
                    strokeOpacity=0,
                    fillOpacity=0.18,
                )
                .encode(
                    x="x:Q",
                    y="y:Q",
                    order="order:O",
                    fill=alt.Color(
                        "side:N",
                        scale=alt.Scale(
                            domain=["spam", "safe"],
                            range=["#fee2e2", "#dbeafe"],
                        ),
                        legend=None,
                    ),
                )
            )
            layers.insert(0, shading_layer)

        band_df = boundary.get("band_df")
        if isinstance(band_df, pd.DataFrame) and not band_df.empty:
            band_layer = (
                alt.Chart(band_df)
                .mark_line(
                    filled=True,
                    color="#f97316",
                    opacity=0.18,
                    strokeOpacity=0,
                    fillOpacity=0.18,
                )
                .encode(
                    x="x:Q",
                    y="y:Q",
                    order="order:O",
                    fill=alt.value("#f97316"),
                )
            )
            layers.append(band_layer)

        line_df = boundary.get("line_df")
        if isinstance(line_df, pd.DataFrame) and not line_df.empty:
            line_layer = (
                alt.Chart(line_df)
                .mark_line(color="#1f2937", strokeWidth=2)
                .encode(x="x:Q", y="y:Q")
            )
            layers.append(line_layer)

    if highlight_borderline and "borderline" in df.columns:
        borderline_df = df.loc[df["borderline"] == True]  # noqa: E712
        if not borderline_df.empty:
            border_layer = (
                alt.Chart(borderline_df)
                .mark_circle(
                    size=230,
                    fillOpacity=0.0,
                    stroke="#1f2937",
                    strokeDash=[4, 3],
                    strokeWidth=2,
                )
                .encode(x="x:Q", y="y:Q")
            )
            layers.append(border_layer)

    if show_examples and isinstance(meta.get("pair"), dict) and "plot_index" in df.columns:
        pair = meta.get("pair") or {}
        indices = pair.get("indices") or []
        coords = pair.get("coords") or []
        if indices:
            highlight_df = df.loc[df["plot_index"].isin(indices)]
            if not highlight_df.empty:
                highlight = alt.Chart(highlight_df).mark_circle(
                    size=180,
                    fillOpacity=0.15,
                    stroke="#f97316",
                    strokeWidth=2,
                ).encode(x="x:Q", y="y:Q")
                layers.append(highlight)
        if coords and len(coords) == 2:
            pair_df = pd.DataFrame(coords, columns=["x", "y"])
            link = alt.Chart(pair_df).mark_line(
                color="#f97316",
                strokeDash=[4, 3],
                strokeWidth=2,
                opacity=0.9,
            ).encode(x="x:Q", y="y:Q")
            layers.append(link)

    centroid_df = meta.get("centroids")
    if show_class_centers and isinstance(centroid_df, pd.DataFrame) and not centroid_df.empty:
        centers = alt.Chart(centroid_df).mark_circle(
            size=230,
            opacity=0.95,
            stroke="#ffffff",
            strokeWidth=1.5,
        ).encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("label:N", scale=color_scale, legend=None),
        )
        layers.append(centers)
        if len(centroid_df) == 2:
            line_df = centroid_df.sort_values("label")[["x", "y"]]
            line = alt.Chart(line_df).mark_line(
                color="#6366f1",
                strokeDash=[3, 3],
                opacity=0.7,
            ).encode(
                x="x:Q",
                y="y:Q",
            )
            layers.append(line)

    scatter_layer = alt.layer(*layers).properties(height=320)

    if select is not None:
        detail_chart = alt.Chart(df).transform_filter(select)

        detail_bg = (
            detail_chart
            .mark_rect(
                color="rgba(226,232,240,0.65)",
                stroke="#94a3b8",
                strokeWidth=1,
                cornerRadius=10,
            )
            .encode(
                x=alt.value(0),
                x2=alt.value(400),
                y=alt.value(0),
                y2=alt.value(110),
            )
        )
        detail_subject = (
            detail_chart
            .mark_text(
                align="left",
                baseline="top",
                dx=16,
                dy=16,
                fontSize=14,
                fontWeight="bold",
                color="#1f2937",
            )
            .encode(text="subject_full:N")
        )
        detail_excerpt = (
            detail_chart
            .mark_text(
                align="left",
                baseline="top",
                dx=16,
                dy=46,
                fontSize=12,
                color="#374151",
            )
            .encode(text="body_excerpt:N")
        )
        detail_reason = (
            detail_chart
            .mark_text(
                align="left",
                baseline="top",
                dx=16,
                dy=78,
                fontSize=11,
                color="#1d4ed8",
            )
            .encode(text="reason:N")
        )

        detail_layer = alt.layer(
            detail_bg, detail_subject, detail_excerpt, detail_reason
        ).properties(height=110)
    else:
        detail_layer = (
            alt.Chart(pd.DataFrame({"x": []}))
            .mark_text()
            .encode(text=alt.value(""))
            .properties(height=0)
        )

    return alt.vconcat(scatter_layer, detail_layer, spacing=12).configure_view(strokeWidth=0)


def _build_borderline_guardrail_chart(
    df: pd.DataFrame,
    meta: Dict[str, Any],
) -> Optional[alt.Chart]:
    if df is None or df.empty:
        return None

    required_cols = {"x", "y", "label"}
    if not required_cols.issubset(df.columns):
        return None

    if "borderline" not in df.columns:
        return None

    df = df.copy()
    defaults: Dict[str, Any] = {
        "label_title": "",
        "predicted_label_title": "",
        "spam_probability": np.nan,
        "distance_display": "–",
        "subject_tooltip": "",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    borderline_df = df.loc[df["borderline"] == True]  # noqa: E712
    if borderline_df.empty:
        return None

    color_scale = alt.Scale(domain=["spam", "safe"], range=["#ef4444", "#3b82f6"])
    tooltip_fields: List[Any] = []
    if "subject_tooltip" in df.columns:
        tooltip_fields.append(alt.Tooltip("subject_tooltip:N", title="Subject"))
    if "label_title" in df.columns:
        tooltip_fields.append(alt.Tooltip("label_title:N", title="True label"))
    if "predicted_label_title" in df.columns:
        tooltip_fields.append(alt.Tooltip("predicted_label_title:N", title="Model prediction"))
    if "spam_probability" in df.columns:
        tooltip_fields.append(alt.Tooltip("spam_probability:Q", title="Spam probability", format=".2f"))
    if "distance_display" in df.columns:
        tooltip_fields.append(alt.Tooltip("distance_display:N", title="Distance to line"))

    layers: List[alt.Chart] = []

    non_borderline_df = df.loc[df["borderline"] == False]  # noqa: E712
    if not non_borderline_df.empty:
        background = (
            alt.Chart(non_borderline_df)
            .mark_circle(size=55, opacity=0.12)
            .encode(
                x=alt.X(
                    "x:Q",
                    axis=alt.Axis(title="Meaning dimension 1", grid=False, ticks=False, labels=False),
                ),
                y=alt.Y(
                    "y:Q",
                    axis=alt.Axis(title="Meaning dimension 2", grid=False, ticks=False, labels=False),
                ),
                color=alt.Color("label:N", scale=color_scale, legend=None),
            )
        )
        layers.append(background)

    boundary = meta.get("boundary") if isinstance(meta, dict) else None
    if isinstance(boundary, dict):
        band_df = boundary.get("band_df")
        if isinstance(band_df, pd.DataFrame) and not band_df.empty:
            band_layer = (
                alt.Chart(band_df)
                .mark_line(
                    filled=True,
                    color="#facc15",
                    opacity=0.18,
                    strokeOpacity=0,
                    fillOpacity=0.18,
                )
                .encode(x="x:Q", y="y:Q", order="order:O", fill=alt.value("#facc15"))
            )
            layers.append(band_layer)

        line_df = boundary.get("line_df")
        if isinstance(line_df, pd.DataFrame) and not line_df.empty:
            line_layer = (
                alt.Chart(line_df)
                .mark_line(color="#1f2937", strokeWidth=2)
                .encode(x="x:Q", y="y:Q")
            )
            layers.append(line_layer)

    borderline_points = (
        alt.Chart(borderline_df)
        .mark_circle(size=170, stroke="#f8fafc", strokeWidth=1.6)
        .encode(
            x=alt.X(
                "x:Q",
                axis=alt.Axis(title="Meaning dimension 1", grid=False, ticks=False, labels=False),
            ),
            y=alt.Y(
                "y:Q",
                axis=alt.Axis(title="Meaning dimension 2", grid=False, ticks=False, labels=False),
            ),
            color=alt.Color("label:N", scale=color_scale, legend=None),
            tooltip=tooltip_fields,
        )
    )
    layers.append(borderline_points)

    outline = (
        alt.Chart(borderline_df)
        .mark_circle(
            size=240,
            fillOpacity=0.0,
            stroke="#facc15",
            strokeDash=[6, 4],
            strokeWidth=2.6,
        )
        .encode(
            x="x:Q",
            y="y:Q",
        )
    )
    layers.append(outline)

    return alt.layer(*layers).properties(height=320)


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
st.markdown(LIFECYCLE_CYCLE_CSS, unsafe_allow_html=True)


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
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def _compute_cached_embeddings(dataset_hash: str, texts: tuple[str, ...]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    return encode_texts(list(texts))


ss = st.session_state
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
        """
        <div class="sidebar-brand">
            <p class="sidebar-title">demistifAI control room</p>
            <p class="sidebar-subtitle">Navigate the lifecycle, review guidance, and manage your session without losing progress.</p>
        </div>
        """,
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

st.title("📧 demistifAI")



def render_intro_stage():

    next_stage_key: Optional[str] = None
    intro_index = STAGE_INDEX.get("intro")
    if intro_index is not None and intro_index < len(STAGES) - 1:
        next_stage_key = STAGES[intro_index + 1].key

    hero_info_html = """
        <div class="hero-info-grid">
            <div class="hero-info-card">
                <h3>What you’ll do</h3>
                <p>
                    Build an email spam detector that identifies patterns in messages. You’ll set how strict the filter is
                    (threshold), choose the autonomy level, and optionally enable adaptiveness to learn from your feedback.
                </p>
            </div>
            <div class="hero-info-card">
                <h3>Why demistifAI</h3>
                <p>
                    AI systems are often seen as black boxes, and the EU AI Act can feel too abstract. This experience demystifies
                    both—showing how everyday AI works in practice.
                </p>
            </div>
        </div>
    """

    with section_surface("section-surface--hero"):
        hero_left, hero_right = st.columns([3, 2], gap="large")
        with hero_left:
            st.subheader("Welcome to demistifAI! 🎉")
            st.markdown(
                "demistifAI is an interactive experience where you will build, evaluate, and operate an AI system—"
                "applying key concepts from the EU AI Act."
            )
            st.markdown(
                "Along the way you’ll see:\n"
                "- how an AI system works end-to-end,\n"
                "- how it infers using AI models,\n"
                "- how models learn from data to achieve an explicit objective,\n"
                "- how autonomy levels affect you as a user, and how optional adaptiveness feeds your feedback back into training."
            )
            st.markdown(hero_info_html, unsafe_allow_html=True)

        with hero_right:
            render_eu_ai_quote(
                "“AI system means a machine-based system that is designed to operate with varying levels of autonomy and that may exhibit "
                "adaptiveness after deployment, and that, for explicit or implicit objectives, infers, from the input it "
                "receives, how to generate outputs such as predictions, content, recommendations, or decisions that can "
                "influence physical or virtual environments.”"
            )

            if next_stage_key:
                st.button(
                    "🚀 Start your machine",
                    key="flow_start_machine_hero",
                    type="primary",
                    on_click=set_active_stage,
                    args=(next_stage_key,),
                    use_container_width=True,
                )

               
    with section_surface():
        st.markdown(
            """
            <div>
                <h4>Your AI system lifecycle at a glance</h4>
                <p>These are the core stages you will navigate. They flow into one another — it’s a continuous loop you can revisit.</p>
                <div class="lifecycle-cycle">
                    <div class="cycle-ring">
                        <div class="cycle-node cycle-node--prepare">
                            <span class="cycle-icon">📊</span>
                            <span class="cycle-title">Prepare Data</span>
                        </div>
                        <div class="cycle-arrow cycle-arrow--prepare-train">
                            <span>➝</span>
                        </div>
                        <div class="cycle-node cycle-node--train">
                            <span class="cycle-icon">🧠</span>
                            <span class="cycle-title">Train</span>
                        </div>
                        <div class="cycle-arrow cycle-arrow--train-evaluate">
                            <span>➝</span>
                        </div>
                        <div class="cycle-node cycle-node--evaluate">
                            <span class="cycle-icon">🧪</span>
                            <span class="cycle-title">Evaluate</span>
                        </div>
                        <div class="cycle-arrow cycle-arrow--evaluate-use">
                            <span>➝</span>
                        </div>
                        <div class="cycle-node cycle-node--use">
                            <span class="cycle-icon">📬</span>
                            <span class="cycle-title">Use</span>
                        </div>
                        <div class="cycle-arrow cycle-arrow--use-prepare">
                            <span>➝</span>
                        </div>
                        <div class="cycle-loop">↺</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


    with section_surface():
        ready_left, ready_right = st.columns([3, 2], gap="large")
        with ready_left:
            st.markdown("### Ready to make a machine learn?")
            st.markdown("No worries — you don’t need to be a developer or data scientist to follow along.")
        with ready_right:
            if next_stage_key:
                st.button(
                    "🚀 Start your machine",
                    key="flow_start_machine_ready",
                    type="primary",
                    on_click=set_active_stage,
                    args=(next_stage_key,),
                )



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
    nerd_enabled = bool(ss.get("nerd_mode"))

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
        render_eu_ai_quote(
            "The EU AI Act says that “An AI system is a machine-based system”."
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
                _set_advanced_knob_state(ss["dataset_config"], force=True)
                st.success(f"Dataset reset to starter baseline ({len(STARTER_LABELED)} rows).")
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
                        _set_advanced_knob_state(config, force=True)
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

                        st.success(f"{summary_line}\n{hint_line}")
                        if any(lint_counts_commit.values()):
                            st.warning(
                                "Lint warnings persist after commit ({}).".format(
                                    format_pii_summary(lint_counts_commit)
                                )
                            )

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
                            _set_advanced_knob_state(config, force=True)
                            ss["dataset_summary"] = summary
                            ss["previous_dataset_summary"] = None
                            ss["dataset_compare_delta"] = None
                            ss["last_dataset_delta_story"] = None
                            ss["dataset_snapshot_name"] = snap.get("name", "")
                            ss["active_dataset_snapshot"] = snap.get("id")
                            ss["dataset_last_built_at"] = datetime.now().isoformat(timespec="seconds")
                            _clear_dataset_preview_state()
                            st.success(
                                f"Snapshot '{snap.get('name', 'snapshot')}' activated. Dataset rebuilt with {len(dataset_rows)} rows."
                            )
                            if any(lint_counts.values()):
                                st.warning(
                                    "Lint warnings present in restored snapshot ({}).".format(
                                        format_pii_summary(lint_counts)
                                    )
                                )
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


def render_train_stage():

    stage = STAGE_BY_KEY["train"]
    ss.setdefault("token_budget_cache", {})

    st.markdown(
        """
        <style>
        .train-intro-card {
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.16), rgba(14, 165, 233, 0.16));
            border-radius: 1.25rem;
            padding: 1.28rem 1.6rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.1);
            margin-bottom: 1.8rem;
        }
        .train-intro-card__header {
            display: flex;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 0.85rem;
        }
        .train-intro-card__icon {
            font-size: 1.75rem;
            line-height: 1;
            background: rgba(15, 23, 42, 0.08);
            border-radius: 1rem;
            padding: 0.55rem 0.95rem;
            box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.05);
        }
        .train-intro-card__eyebrow {
            font-size: 0.75rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 700;
            color: rgba(15, 23, 42, 0.55);
            display: inline-block;
        }
        .train-intro-card__title {
            margin: 0;
            font-size: 1.4rem;
            font-weight: 700;
            color: #0f172a;
        }
        .train-intro-card__body {
            margin: 0 0 1.15rem 0;
            color: rgba(15, 23, 42, 0.82);
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .train-intro-card__steps {
            display: grid;
            gap: 0.75rem;
        }
        .train-intro-card__step {
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
        }
        .train-intro-card__step-index {
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: rgba(79, 70, 229, 0.18);
            color: rgba(30, 64, 175, 0.9);
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
        }
        .train-intro-card__step-body {
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.8);
            line-height: 1.55;
        }
        .train-sidebar-card {
            border-radius: 1.1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.12);
            padding: 0.96rem 1.35rem;
            margin-bottom: 1rem;
        }
        .train-sidebar-card__title {
            margin: 0 0 0.65rem 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
        }
        .train-sidebar-card__list {
            list-style: none;
            margin: 0;
            padding: 0;
            display: grid;
            gap: 0.55rem;
        }
        .train-sidebar-card__list li {
            display: grid;
            grid-template-columns: 1.8rem 1fr;
            gap: 0.55rem;
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.78);
            line-height: 1.4;
        }
        .train-sidebar-card__bullet {
            width: 1.8rem;
            height: 1.8rem;
            border-radius: 0.75rem;
            background: rgba(79, 70, 229, 0.12);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
        }
        .train-sidebar-card--secondary {
            background: rgba(30, 64, 175, 0.06);
            border: 1px dashed rgba(30, 64, 175, 0.35);
        }
        .train-how-card {
            border-radius: 1.1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 0.96rem 1.35rem;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.1);
        }
        .train-how-card__header {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            margin-bottom: 0.75rem;
        }
        .train-how-card__icon {
            font-size: 1.35rem;
        }
        .train-how-card__title {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
        }
        .train-how-card__body {
            margin: 0 0 0.85rem 0;
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.78);
            line-height: 1.55;
        }
        .train-how-card__steps {
            margin: 0 0 0.9rem 1.15rem;
            padding: 0;
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.82);
        }
        .train-how-card__grid {
            display: grid;
            gap: 0.9rem;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            margin-bottom: 0.6rem;
        }
        .train-how-card__panel {
            position: relative;
            border-radius: 1rem;
            border: 1px solid rgba(79, 70, 229, 0.16);
            background: rgba(79, 70, 229, 0.06);
            padding: 0.85rem 1rem;
            display: grid;
            gap: 0.35rem;
        }
        .train-how-card__panel-icon {
            font-size: 1.1rem;
        }
        .train-how-card__panel-title {
            margin: 0;
            font-size: 0.9rem;
            font-weight: 700;
            color: rgba(30, 64, 175, 0.95);
        }
        .train-how-card__panel-body {
            margin: 0;
            font-size: 0.85rem;
            color: rgba(15, 23, 42, 0.75);
            line-height: 1.5;
        }
        .train-how-card__divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(15, 23, 42, 0), rgba(15, 23, 42, 0.25), rgba(15, 23, 42, 0));
            margin: 0.75rem 0 0.9rem 0;
        }
        .train-how-card__body--muted {
            color: rgba(15, 23, 42, 0.6);
            font-size: 0.88rem;
            margin-bottom: 0.65rem;
        }
        .train-how-card__step-grid {
            display: grid;
            gap: 0.75rem;
        }
        .train-how-card__step-box {
            border-radius: 1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: linear-gradient(135deg, rgba(14, 116, 144, 0.08), rgba(59, 130, 246, 0.08));
            padding: 0.95rem 1.1rem;
            display: grid;
            gap: 0.5rem;
        }
        .train-how-card__step-label {
            display: flex;
            align-items: center;
            gap: 0.55rem;
        }
        .train-how-card__step-number {
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: rgba(14, 116, 144, 0.18);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.95rem;
            color: rgba(12, 74, 110, 0.95);
        }
        .train-how-card__step-icon {
            font-size: 1.25rem;
        }
        .train-how-card__step-title {
            font-weight: 700;
            font-size: 0.95rem;
            color: rgba(15, 23, 42, 0.9);
        }
        .train-how-card__step-list {
            margin: 0;
            padding-left: 1.2rem;
            font-size: 0.88rem;
            color: rgba(15, 23, 42, 0.78);
            line-height: 1.55;
        }
        .train-how-card__step-sublist {
            margin: 0.45rem 0 0 1.1rem;
            padding-left: 1rem;
            font-size: 0.85rem;
            color: rgba(15, 23, 42, 0.75);
            line-height: 1.5;
        }
        .train-how-card__step-example {
            margin: 0;
            font-size: 0.82rem;
            color: rgba(15, 23, 42, 0.68);
            background: rgba(255, 255, 255, 0.6);
            border-radius: 0.75rem;
            padding: 0.55rem 0.75rem;
            border: 1px dashed rgba(12, 74, 110, 0.25);
        }
        .train-sidebar-card__grid {
            display: grid;
            gap: 0.65rem;
        }
        .train-sidebar-card__item {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 0.6rem;
            padding: 0.6rem 0.75rem;
            border-radius: 0.85rem;
            background: rgba(15, 23, 42, 0.04);
            align-items: flex-start;
        }
        .train-sidebar-card__badge {
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
        .train-sidebar-card__badge-num {
            width: 1.8rem;
            height: 1.8rem;
            border-radius: 999px;
            background: rgba(79, 70, 229, 0.16);
            color: rgba(30, 64, 175, 0.92);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.85rem;
        }
        .train-sidebar-card__badge-icon {
            font-size: 1.05rem;
        }
        .train-sidebar-card__item-title {
            margin: 0;
            font-size: 0.9rem;
            font-weight: 600;
            color: #0f172a;
        }
        .train-sidebar-card__item-body {
            margin: 0.2rem 0 0 0;
            font-size: 0.83rem;
            color: rgba(15, 23, 42, 0.75);
            line-height: 1.45;
        }
        .train-token-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.4rem 0.85rem;
            background: rgba(59, 130, 246, 0.12);
            color: rgba(30, 64, 175, 0.9);
            font-size: 0.8rem;
            font-weight: 600;
        }
        .train-inline-note {
            margin-top: 0.35rem;
            font-size: 0.8rem;
            color: rgba(30, 64, 175, 0.82);
            background: rgba(191, 219, 254, 0.35);
            border-radius: 0.6rem;
            padding: 0.45rem 0.75rem;
            display: inline-block;
        }
        .train-action-card {
            border-radius: 1.1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 1rem 1.35rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(14, 165, 233, 0.12));
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.1);
            margin-bottom: 0.75rem;
        }
        .train-action-card__eyebrow {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-weight: 700;
            color: rgba(30, 64, 175, 0.75);
        }
        .train-action-card__title {
            margin: 0.35rem 0 0.6rem 0;
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a;
        }
        .train-action-card__body {
            margin: 0;
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.78);
            line-height: 1.5;
        }
        .train-context-card {
            border-radius: 1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 0.8rem 1.1rem;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
            margin-bottom: 0.9rem;
        }
        .train-context-card h5 {
            margin: 0 0 0.5rem 0;
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
        }
        .train-context-card ul {
            margin: 0;
            padding-left: 1.1rem;
            font-size: 0.88rem;
            color: rgba(15, 23, 42, 0.78);
            line-height: 1.5;
        }
        .train-context-card--tip {
            background: rgba(236, 233, 254, 0.6);
            border: 1px dashed rgba(79, 70, 229, 0.35);
            color: rgba(55, 48, 163, 0.95);
        }
        .train-band-card {
            border-radius: 1rem;
            border: 1px solid rgba(79, 70, 229, 0.25);
            padding: 0.95rem 1.1rem 1.05rem 1.1rem;
            background: rgba(79, 70, 229, 0.08);
            box-shadow: inset 0 0 0 1px rgba(79, 70, 229, 0.08);
        }
        .train-band-card__title {
            margin: 0 0 0.55rem 0;
            font-size: 0.95rem;
            font-weight: 700;
            color: rgba(49, 46, 129, 0.9);
        }
        .train-band-card__bar {
            position: relative;
            width: 100%;
            height: 16px;
            border-radius: 999px;
            background: rgba(30, 64, 175, 0.12);
        }
        .train-band-card__band {
            position: absolute;
            top: 0;
            bottom: 0;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.35);
        }
        .train-band-card__threshold {
            position: absolute;
            top: -4px;
            bottom: -4px;
            width: 2px;
            border-radius: 2px;
            background: rgba(30, 64, 175, 0.95);
        }
        .train-band-card__scale {
            display: flex;
            justify-content: space-between;
            font-size: 0.68rem;
            color: rgba(30, 64, 175, 0.8);
            margin-top: 0.35rem;
        }
        .train-band-card__caption {
            margin-top: 0.45rem;
            font-size: 0.75rem;
            color: rgba(49, 46, 129, 0.85);
        }
        .train-band-card__hint {
            margin-top: 0.35rem;
            font-size: 0.78rem;
            color: rgba(49, 46, 129, 0.75);
        }
        .train-nerd-intro {
            border-radius: 1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 1rem 1.1rem;
            background: rgba(14, 165, 233, 0.08);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }
        .train-nerd-intro h4 {
            margin: 0 0 0.35rem 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: rgba(7, 89, 133, 0.9);
        }
        .train-nerd-intro p {
            margin: 0;
            font-size: 0.88rem;
            color: rgba(7, 89, 133, 0.78);
            line-height: 1.5;
        }
        .train-nerd-hint {
            margin-top: 0.85rem;
            font-size: 0.8rem;
            color: rgba(15, 23, 42, 0.7);
            background: rgba(191, 219, 254, 0.45);
            border-radius: 0.75rem;
            padding: 0.55rem 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    stage_number = STAGE_INDEX.get(stage.key, 0) - STAGE_INDEX.get("data", 0) + 1
    if stage_number < 1:
        stage_number = STAGE_INDEX.get(stage.key, 0) + 1

    nerd_mode_train_enabled = bool(ss.get("nerd_mode_train"))

    stage_title_html = html.escape(stage.title)
    stage_desc_html = html.escape(stage.description)
    stage_icon_html = html.escape(stage.icon)

    with section_surface():
        intro_col, sidebar_col = st.columns([0.58, 0.42], gap="large")
        with intro_col:
            st.markdown(
                """
                <div class="train-intro-card">
                    <div class="train-intro-card__header">
                        <span class="train-intro-card__icon">{icon}</span>
                        <div>
                            <span class="train-intro-card__eyebrow">Stage {num}</span>
                            <h4 class="train-intro-card__title">{title}</h4>
                        </div>
                    </div>
                    <p class="train-intro-card__body">
                        {desc} It now learns how to separate spam from safe emails using the examples you curated.
                    </p>
                    <div class="train-intro-card__steps">
                        <div class="train-intro-card__step">
                            <span class="train-intro-card__step-index">1</span>
                            <div class="train-intro-card__step-body"><strong>Start from labeled history.</strong> The dataset built in <em>Prepare data</em> anchors what "Spam" and "Safe" mean.</div>
                        </div>
                        <div class="train-intro-card__step">
                            <span class="train-intro-card__step-index">2</span>
                            <div class="train-intro-card__step-body"><strong>Spot the patterns.</strong> MiniLM encodes each email; the classifier learns a direction that pulls spam and safe apart.</div>
                        </div>
                        <div class="train-intro-card__step">
                            <span class="train-intro-card__step-index">3</span>
                            <div class="train-intro-card__step-body"><strong>Generalize forward.</strong> With the patterns in hand, the model can infer outcomes for new emails it has never seen.</div>
                        </div>
                    </div>
                </div>
                """.format(
                    icon=stage_icon_html,
                    num=stage_number,
                    title=stage_title_html,
                    desc=stage_desc_html,
                ),
                unsafe_allow_html=True,
            )
            render_eu_ai_quote("The EU AI Act says: “An AI system infers from the input it receives…”")
            if not ss.get("nerd_mode_train"):
                st.markdown(
                    """
                    <div class="train-how-card">
                        <div class="train-how-card__header">
                            <div class="train-how-card__icon">🧩</div>
                            <div>
                                <h5 class="train-how-card__title">🎯 Now your AI system will learn how to achieve an objective</h5>
                                <p class="train-how-card__body">Think of this as teaching your AI assistant how to tell the difference between spam and safe emails.</p>
                            </div>
                        </div>
                        <div class="train-how-card__grid">
                            <div class="train-how-card__panel">
                                <div class="train-how-card__panel-icon">🫶</div>
                                <h6 class="train-how-card__panel-title">Your part</h6>
                                <p class="train-how-card__panel-body">Provide examples with clear labels (“This one is spam, that one is safe”).</p>
                            </div>
                            <div class="train-how-card__panel">
                                <div class="train-how-card__panel-icon">🤖</div>
                                <h6 class="train-how-card__panel-title">The system’s part</h6>
                                <p class="train-how-card__panel-body">Spot patterns that generalize to emails it hasn’t seen yet.</p>
                            </div>
                        </div>
                        <div class="train-how-card__divider"></div>
                        <p class="train-how-card__body train-how-card__body--muted">We can break it down into three simple steps:</p>
                        <div class="train-how-card__step-grid">
                            <div class="train-how-card__step-box">
                                <div class="train-how-card__step-label">
                                    <span class="train-how-card__step-number">1</span>
                                    <span class="train-how-card__step-icon">🗺️</span>
                                    <span class="train-how-card__step-title">Turning emails into “meaning points”</span>
                                </div>
                                <ul class="train-how-card__step-list">
                                    <li>Imagine every email as a dot on a map.</li>
                                    <li>A small language model called MiniLM reads the words and places each email on this map so that similar messages land near each other.</li>
                                </ul>
                                <div class="train-how-card__step-example"><strong>Example:</strong> “Win a free iPhone!” and “Claim your prize now!” sit close together, while “Project meeting agenda” lands far away.</div>
                            </div>
                            <div class="train-how-card__step-box">
                                <div class="train-how-card__step-label">
                                    <span class="train-how-card__step-number">2</span>
                                    <span class="train-how-card__step-icon">📏</span>
                                    <span class="train-how-card__step-title">Drawing a dividing line</span>
                                </div>
                                <ul class="train-how-card__step-list">
                                    <li>Once emails are mapped, a simple linear classifier draws the straight line that best separates Spam from Safe.</li>
                                    <li>Everything on one side is predicted as spam; everything on the other is tagged safe.</li>
                                </ul>
                                <div class="train-how-card__step-example"><strong>Example:</strong> Marketing scams cluster on the spam side, while project updates stay on the safe side.</div>
                            </div>
                            <div class="train-how-card__step-box">
                                <div class="train-how-card__step-label">
                                    <span class="train-how-card__step-number">3</span>
                                    <span class="train-how-card__step-icon">🛡️</span>
                                    <span class="train-how-card__step-title">Using extra clues when uncertain</span>
                                </div>
                                <ul class="train-how-card__step-list">
                                    <li>Borderline emails land near the dividing line.</li>
                                    <li>Here, the system checks extra guardrails for hints when the text alone is unsure.</li>
                                </ul>
                                <ul class="train-how-card__step-sublist">
                                    <li>Does it hide suspicious links?</li>
                                    <li>Is it shouting in ALL CAPS?</li>
                                    <li>Does it push urgency or “$$$” language?</li>
                                </ul>
                                <div class="train-how-card__step-example"><strong>Example:</strong> “Your invoice is ready” stays safe, while “URGENT! CLICK NOW TO CLAIM $$$” trips multiple guardrails.</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with sidebar_col:
            st.markdown(
                """
                <div class="train-sidebar-card">
                    <div class="train-sidebar-card__title">🚦 Training readiness checklist</div>
                    <div class="train-sidebar-card__grid">
                        <div class="train-sidebar-card__item">
                            <div class="train-sidebar-card__badge">
                                <span class="train-sidebar-card__badge-num">1</span>
                                <span class="train-sidebar-card__badge-icon">📊</span>
                            </div>
                            <div>
                                <p class="train-sidebar-card__item-title">Balanced labels</p>
                                <p class="train-sidebar-card__item-body">Mix of labeled spam and safe emails (aim for balance).</p>
                            </div>
                        </div>
                        <div class="train-sidebar-card__item">
                            <div class="train-sidebar-card__badge">
                                <span class="train-sidebar-card__badge-num">2</span>
                                <span class="train-sidebar-card__badge-icon">🧪</span>
                            </div>
                            <div>
                                <p class="train-sidebar-card__item-title">Honest evaluation slice</p>
                                <p class="train-sidebar-card__item-body">Keep a hold-out split ready so we can grade the model without leakage.</p>
                            </div>
                        </div>
                        <div class="train-sidebar-card__item">
                            <div class="train-sidebar-card__badge">
                                <span class="train-sidebar-card__badge-num">3</span>
                                <span class="train-sidebar-card__badge-icon">🧹</span>
                            </div>
                            <div>
                                <p class="train-sidebar-card__item-title">Data hygiene</p>
                                <p class="train-sidebar-card__item-body">Scrub or minimise personal data in the dataset preview.</p>
                            </div>
                        </div>
                        <div class="train-sidebar-card__item">
                            <div class="train-sidebar-card__badge">
                                <span class="train-sidebar-card__badge-num">4</span>
                                <span class="train-sidebar-card__badge-icon">📝</span>
                            </div>
                            <div>
                                <p class="train-sidebar-card__item-title">Document assumptions</p>
                                <p class="train-sidebar-card__item-body">Write down any assumptions before you ship the model.</p>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="train-sidebar-card train-sidebar-card--secondary">
                    <div class="train-sidebar-card__title">Need precise control?</div>
                    <p style="margin:0; font-size:0.88rem; color:rgba(30,64,175,0.82);">Toggle Nerd Mode to tune the train/test split and numeric guardrails before you launch training.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            nerd_mode_train_enabled = render_nerd_mode_toggle(
                key="nerd_mode_train",
                title="Nerd Mode — advanced controls",
                description="Tweak the train/test split, solver iterations, and regularization strength.",
                icon="🔬",
            )

    def _parse_split_cache(cache):
        if cache is None:
            raise ValueError("Missing split cache.")
        if len(cache) == 4:
            X_tr, X_te, y_tr, y_te = cache
            train_bodies = ["" for _ in range(len(X_tr))]
            test_bodies = ["" for _ in range(len(X_te))]
            return (
                list(X_tr),
                list(X_te),
                train_bodies,
                test_bodies,
                list(y_tr),
                list(y_te),
            )
        if len(cache) == 6:
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = cache
            return (
                list(X_tr_t),
                list(X_te_t),
                list(X_tr_b),
                list(X_te_b),
                list(y_tr),
                list(y_te),
            )
        y_tr = list(cache[-2]) if len(cache) >= 2 else []
        y_te = list(cache[-1]) if len(cache) >= 1 else []
        return [], [], [], [], y_tr, y_te

    if nerd_mode_train_enabled:
        with section_surface():
            st.markdown(
                """
                <div class="train-nerd-intro">
                    <h4>🛡️ Advanced split & guardrail controls</h4>
                    <p>Fine-tune how much data we hold out, solver behaviour, and the numeric assist rules that complement the text model.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            colA, colB = st.columns(2)
            with colA:
                ss["train_params"]["test_size"] = st.slider(
                    "🧪 Hold-out test fraction",
                    min_value=0.10,
                    max_value=0.50,
                    value=float(ss["train_params"]["test_size"]),
                    step=0.05,
                    help="How much labeled data to keep aside as a mini 'exam' (not used for learning).",
                )
                st.caption("🧪 More hold-out = more honest testing but fewer examples for learning.")
                ss["train_params"]["random_state"] = st.number_input(
                    "Random seed",
                    min_value=0,
                    value=int(ss["train_params"]["random_state"]),
                    step=1,
                    help="Fix this to make your train/test split reproducible.",
                )
                st.caption("Keeps your split and results repeatable.")
                ss.setdefault("guard_params", {})
                gp = ss["guard_params"]
                gp["assist_center"] = st.slider(
                    "🛡️ Numeric assist center (text score)",
                    min_value=0.30,
                    max_value=0.90,
                    step=0.01,
                    value=float(gp.get("assist_center", float(ss.get("threshold", 0.6)))),
                    help=(
                        "Center of the borderline region. When the text-only spam probability is near this "
                        "value, numeric guardrails are allowed to lend a hand."
                    ),
                )
                st.caption(
                    "🛡️ Where ‘borderline’ lives on the 0–1 scale; most emails away from here won’t use numeric cues."
                )
                gp["uncertainty_band"] = st.slider(
                    "🛡️ Uncertainty band (± around threshold)",
                    min_value=0.0,
                    max_value=0.20,
                    step=0.01,
                    value=float(gp.get("uncertainty_band", 0.08)),
                    help="Only consult numeric cues when the text score falls inside this band.",
                )
                st.caption("🛡️ Wider band = numeric cues help more often; narrower = trust text more.")
                gp["numeric_scale"] = st.slider(
                    "🛡️ Numeric blend weight (when consulted)",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    value=float(gp.get("numeric_scale", 0.5)),
                    help="How much numeric probability counts in the blend within the band.",
                )
                st.caption("🛡️ Higher = numeric cues have a stronger say when consulted.")
            with colB:
                ss["train_params"]["max_iter"] = st.number_input(
                    "Max iterations (solver)",
                    min_value=200,
                    value=int(ss["train_params"]["max_iter"]),
                    step=100,
                    help="How many optimization steps the classifier can take before stopping.",
                )
                st.caption("Higher values let the solver search longer; use if it says ‘didn’t converge’.")
                ss["train_params"]["C"] = st.number_input(
                    "Regularization strength C (inverse of regularization)",
                    min_value=0.01,
                    value=float(ss["train_params"]["C"]),
                    step=0.25,
                    format="%.2f",
                    help="Higher C fits training data more tightly; lower C adds regularization to reduce overfitting.",
                )
                st.caption("Higher C hugs the training data (risk overfit). Lower C smooths (better generalization).")
                gp["numeric_logit_cap"] = st.slider(
                    "🛡️ Cap numeric logit (absolute)",
                    min_value=0.2,
                    max_value=3.0,
                    step=0.1,
                    value=float(gp.get("numeric_logit_cap", 1.0)),
                    help="Limits how strongly numeric cues can push toward Spam/Safe.",
                )
                st.caption("🛡️ A safety cap so numeric cues can’t overpower the text score.")
                gp["combine_strategy"] = st.radio(
                    "🛡️ Numeric combination strategy",
                    options=["blend", "threshold_shift"],
                    index=0 if gp.get("combine_strategy", "blend") == "blend" else 1,
                    horizontal=True,
                    help="Blend = mix text & numeric probs; Threshold shift = keep text prob, adjust effective threshold slightly.",
                )

            if gp.get("combine_strategy", "blend") == "threshold_shift":
                st.markdown("**🛡️ Threshold-shift micro-rules** (applied only within the uncertainty band)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    gp["shift_suspicious_tld"] = st.number_input(
                        "🛡️ Shift for suspicious TLD",
                        value=float(gp.get("shift_suspicious_tld", -0.04)),
                        step=0.01,
                        format="%.2f",
                        help="Negative shift lowers the threshold (be stricter) when a suspicious domain is present.",
                    )
                    st.caption("🛡️ Tweaks the cut-off in specific situations (e.g., suspicious domains → stricter).")
                with col2:
                    gp["shift_many_links"] = st.number_input(
                        "🛡️ Shift for many external links",
                        value=float(gp.get("shift_many_links", -0.03)),
                        step=0.01,
                        format="%.2f",
                        help="Negative shift lowers the threshold when multiple external links are detected.",
                    )
                    st.caption("🛡️ Tweaks the cut-off in specific situations (e.g., suspicious domains → stricter).")
                with col3:
                    gp["shift_calm_text"] = st.number_input(
                        "🛡️ Shift for calm text",
                        value=float(gp.get("shift_calm_text", +0.02)),
                        step=0.01,
                        format="%.2f",
                        help="Positive shift raises the threshold when text looks calm (very low ALL-CAPS).",
                    )
                    st.caption("🛡️ Tweaks the cut-off in specific situations (e.g., suspicious domains → stricter).")

            st.markdown(
                """
                <div class="train-nerd-hint">
                    <strong>🎯 Guide:</strong> Hold-out keeps an honest exam set, the seed makes runs reproducible, <em>max iter</em> and <em>C</em> steer the solver, and the numeric guardrails define when structured cues can override the text score.
                </div>
                """,
                unsafe_allow_html=True,
            )

    token_budget_text = "Token budget: —"
    avg_tokens_estimate: Optional[float] = None
    show_trunc_tip = False
    try:
        labeled_rows = ss.get("labeled", [])
        if labeled_rows:
            dataset_hash = compute_dataset_hash(labeled_rows)
            cache = ss.get("token_budget_cache", {})
            stats = cache.get(dataset_hash)
            if stats is None:
                titles = [str(row.get("title", "")) for row in labeled_rows]
                bodies = [str(row.get("body", "")) for row in labeled_rows]
                stats = _estimate_token_stats(titles, bodies, max_tokens=384)
                cache[dataset_hash] = stats
                ss["token_budget_cache"] = cache
            if stats and stats.get("n"):
                avg_tokens = float(stats.get("avg_tokens", 0.0))
                if math.isfinite(avg_tokens) and avg_tokens > 0.0:
                    avg_tokens_estimate = avg_tokens
                pct_trunc = float(stats.get("p_truncated", 0.0)) * 100.0
                token_budget_text = f"Token budget: avg ~{avg_tokens:.0f} • truncated: {pct_trunc:.1f}%"
                show_trunc_tip = float(stats.get("p_truncated", 0.0)) > 0.05
        else:
            token_budget_text = "Token budget: —"
    except Exception:
        token_budget_text = "Token budget: —"
        show_trunc_tip = False

    if show_trunc_tip:
        st.markdown(
            f"<div class='train-token-chip'>{html.escape(token_budget_text)}</div>",
            unsafe_allow_html=True,
        )
    if show_trunc_tip:
        st.markdown(
            "<div class='train-inline-note'>Tip: long emails will be clipped; summaries help.</div>",
            unsafe_allow_html=True,
        )

    with section_surface():
        st.markdown(
            """
            <div style="display:flex;flex-direction:column;gap:0.35rem;margin-bottom:0.75rem;">
                <div style="font-size:1.05rem;font-weight:600;color:#111827;">A peek at your training data</div>
                <div style="font-size:0.9rem;color:#4B5563;">Here are a few examples from your labeled set. The model studies these to learn the difference.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        labeled_rows = ss.get("labeled") or []
        spam_examples: List[Dict[str, Any]] = []
        safe_examples: List[Dict[str, Any]] = []
        if isinstance(labeled_rows, list):
            for row in labeled_rows:
                if not isinstance(row, dict):
                    continue
                label_value = (row.get("label", "") or "").strip().lower()
                if label_value == "spam" and len(spam_examples) < 2:
                    spam_examples.append(row)
                elif label_value == "safe" and len(safe_examples) < 2:
                    safe_examples.append(row)
                if len(spam_examples) >= 2 and len(safe_examples) >= 2:
                    break

        cards_html_parts: List[str] = []
        for card in spam_examples + safe_examples:
            label_value = (card.get("label", "") or "").strip().lower()
            icon = "🚩" if label_value == "spam" else "📥"
            subject = html.escape(_safe_subject(card))
            excerpt = html.escape(_excerpt(card.get("body")))
            cards_html_parts.append(
                textwrap.dedent(
                    f"""
                    <div style="border:1px solid #E5E7EB;border-radius:10px;padding:0.65rem;background-color:#FFFFFF;display:flex;flex-direction:column;gap:0.4rem;">
                        <div style=\"font-size:0.85rem;color:#4B5563;display:flex;align-items:center;gap:0.4rem;\">{icon}<span style=\"font-weight:600;color:#111827;\">{html.escape(label_value.title() or 'Unlabeled')}</span></div>
                        <div style="font-weight:600;color:#111827;font-size:0.95rem;line-height:1.3;">{subject}</div>
                        <div style="font-size:0.85rem;color:#374151;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{excerpt}</div>
                    </div>
                    """
                ).strip()
            )

        if cards_html_parts:
            cards_html = "".join(cards_html_parts)
            st.markdown(
                textwrap.dedent(
                    f"""
                    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:0.75rem;">
                        {cards_html}
                    </div>
                    """
                ).strip(),
                unsafe_allow_html=True,
            )
        else:
            st.info("Label a few emails in Prepare data to see examples here.")

        chip_labels = ["Suspicious links", "SHOUTY caps", "Money/urgency terms"]
        chip_html = "".join(
            f"<span style='display:inline-block;border:1px solid #D1D5DB;border-radius:999px;padding:0.25rem 0.6rem;font-size:0.75rem;color:#1F2937;background-color:#F9FAFB;margin-right:0.35rem;margin-bottom:0.35rem;'>{html.escape(label)}</span>"
            for label in chip_labels
        )
        st.markdown(
            f"<div style='margin-top:0.85rem;'>{chip_html}</div>",
            unsafe_allow_html=True,
        )
        st.caption("These are the kinds of cues the model is likely to notice. You tuned their presence in Prepare.")

    with section_surface():
        action_col, context_col = st.columns([0.55, 0.45], gap="large")
        with action_col:
            st.markdown(
                """
                <div class="train-action-card">
                    <div class="train-action-card__eyebrow">Run training</div>
                    <div class="train-action-card__title">Teach the spam detector</div>
                    <p class="train-action-card__body">When you’re ready, launch the training run. We’ll automatically evaluate on the hold-out split.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            trigger_train = st.button("🚀 Train model", type="primary", use_container_width=True)
        with context_col:
            st.markdown(
                """
                <div class="train-context-card">
                    <h5>What happens when you train</h5>
                    <ul>
                        <li>Uses the labeled dataset curated in <em>Prepare data</em>.</li>
                        <li>Applies your split, solver, and numeric guardrail settings.</li>
                        <li>Captures metrics for the Evaluate stage automatically.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            chip_texts: List[str] = []

            params = ss.get("train_params") or {}
            try:
                raw_test_size = params.get("test_size", 0.0)
                test_size = float(raw_test_size)
            except (TypeError, ValueError):
                test_size = None
            if test_size is not None and math.isfinite(test_size):
                holdout_pct = int(max(0.0, min(1.0, test_size)) * 100)
                chip_texts.append(f"Hold-out: {holdout_pct}%")

            language_chip: Optional[str] = None
            if has_langdetect:
                labeled_for_lang = []
                labeled_rows = ss.get("labeled") or []
                if isinstance(labeled_rows, list):
                    for row in labeled_rows:
                        if not isinstance(row, dict):
                            continue
                        title = str(row.get("title", ""))
                        body = str(row.get("body", ""))
                        if title or body:
                            labeled_for_lang.append(combine_text(title, body))
                if labeled_for_lang:
                    try:
                        mix_preview = summarize_language_mix(
                            labeled_for_lang, top_k=2
                        )
                    except Exception:
                        mix_preview = None
                    if (
                        mix_preview
                        and mix_preview.get("available")
                        and int(mix_preview.get("total", 0)) > 0
                    ):
                        top_pairs = list(mix_preview.get("top", []))[:2]
                        if top_pairs:
                            top_codes = [str(lang).upper() for lang, _ in top_pairs]
                            language_chip = f"Languages: {' + '.join(top_codes)}"
            if language_chip:
                chip_texts.append(language_chip)

            if not show_trunc_tip and avg_tokens_estimate and math.isfinite(avg_tokens_estimate):
                chip_texts.append(f"Avg tokens: ~{avg_tokens_estimate:.0f}")

            if chip_texts:
                chips_html = "".join(
                    f"<div class='train-token-chip'>{html.escape(text)}</div>"
                    for text in chip_texts
                )
                st.markdown(
                    f"<div style='display:flex;flex-wrap:wrap;gap:0.4rem;margin-top:0.75rem;'>{chips_html}</div>",
                    unsafe_allow_html=True,
                )

            if not ss.get("nerd_mode_train"):
                guard_params = ss.get("guard_params", {})
                assist_center = float(
                    guard_params.get("assist_center", float(ss.get("threshold", 0.6)))
                )
                band = float(guard_params.get("uncertainty_band", 0.08))

                low = max(0.0, assist_center - band)
                high = min(1.0, assist_center + band)
                low_pct = low * 100.0
                high_pct = high * 100.0
                threshold_pct = max(0.0, min(100.0, assist_center * 100.0))

                band_left = max(0.0, min(100.0, low_pct))
                band_right = max(0.0, min(100.0, high_pct))
                band_width = max(0.0, band_right - band_left)

                band_card_html = f"""
                <div class='train-band-card'>
                    <div class='train-band-card__title'>Numeric assist window</div>
                    <div class='train-band-card__bar'>
                        <div class='train-band-card__band' style='left:{band_left:.2f}%; width:{band_width:.2f}%;'></div>
                        <div class='train-band-card__threshold' style='left:{threshold_pct:.2f}%;'></div>
                    </div>
                    <div class='train-band-card__scale'><span>0</span><span>1</span></div>
                    <div class='train-band-card__caption'>τ = {assist_center:.2f} • band ±{band:.2f}</div>
                    <div class='train-band-card__hint'>“Inside this zone, numeric cues (links, TLDs, caps, money terms) can gently adjust the text score.”</div>
                </div>
                """

                context_col.markdown(band_card_html, unsafe_allow_html=True)

    if trigger_train:
        if len(ss["labeled"]) < 6:
            st.warning("Please label a few more emails first (≥6 examples).")
        else:
            df = pd.DataFrame(ss["labeled"])
            if len(df["label"].unique()) < 2:
                st.warning("You need both classes (spam and safe) present to train.")
            else:
                params = ss.get("train_params", {})
                test_size = float(params.get("test_size", 0.30))
                random_state = int(params.get("random_state", 42))
                max_iter = int(params.get("max_iter", 1000))
                C_value = float(params.get("C", 1.0))

                titles = df["title"].fillna("").tolist()
                bodies = df["body"].fillna("").tolist()
                y = df["label"].tolist()
                X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = train_test_split(
                    titles,
                    bodies,
                    y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y,
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
                if has_embed:
                    try:
                        train_texts_cache = [
                            combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)
                        ]
                        cache_train_embeddings(train_texts_cache)
                    except Exception:
                        pass

    parsed_split = None
    y_tr_labels = None
    y_te_labels = None
    train_texts_combined_cache: list[str] = []
    test_texts_combined_cache: list[str] = []
    lang_mix_train: Optional[Dict[str, Any]] = None
    lang_mix_test: Optional[Dict[str, Any]] = None
    lang_mix_error: Optional[str] = None
    has_model = ss.get("model") is not None
    has_split_cache = ss.get("split_cache") is not None
    if has_model and has_split_cache:
        try:
            parsed_split = _parse_split_cache(ss["split_cache"])
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr_labels, y_te_labels = parsed_split

            if X_tr_t is not None and X_tr_b is not None:
                train_texts_combined_cache = [
                    combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)
                ]
            if X_te_t is not None and X_te_b is not None:
                test_texts_combined_cache = [
                    combine_text(t, b) for t, b in zip(X_te_t, X_te_b)
                ]

            if has_langdetect:
                try:
                    lang_mix_train = summarize_language_mix(train_texts_combined_cache)
                    lang_mix_test = summarize_language_mix(test_texts_combined_cache)
                except Exception as exc:
                    lang_mix_error = str(exc) or exc.__class__.__name__
                    lang_mix_train = None
                    lang_mix_test = None
            else:
                lang_mix_error = "language detector unavailable"

            # Existing success + story (kept)
            story = make_after_training_story(y_tr_labels, y_te_labels)
            st.success("Training finished.")
            st.markdown(story)

            # --- New: Training Storyboard (plain language, for everyone) ---
            with section_surface():
                st.markdown("### Training story — what just happened")

                # 1) What data was used + Meaning Map
                ct = _counts(list(y_tr_labels))
                total_train = len(y_tr_labels)
                train_titles_story = list(X_tr_t) if X_tr_t is not None else []
                train_bodies_story = list(X_tr_b) if X_tr_b is not None else []
                train_labels_story = list(y_tr_labels) if y_tr_labels is not None else []

                st.markdown(
                    f"- **Turning emails into “meaning points”.** MiniLM read **{total_train} training emails** "
                    f"(Spam: {ct['spam']}, Safe: {ct['safe']}) and places each one as a dot so similar wording sits close together."
                )

                meaning_map_df: Optional[pd.DataFrame] = None
                meaning_map_meta: Dict[str, Any] = {}
                meaning_map_error: Optional[str] = None
                model_for_story = ss.get("model")

                try:
                    meaning_map_df, meaning_map_meta = _prepare_meaning_map(
                        list(X_tr_t) if X_tr_t is not None else [],
                        list(X_tr_b) if X_tr_b is not None else [],
                        list(y_tr_labels) if y_tr_labels is not None else [],
                        model_for_story,
                    )
                except Exception as exc:
                    meaning_map_error = f"Meaning Map unavailable ({exc})."
                    logger.exception("Meaning Map failed")

                if meaning_map_error:
                    st.info(meaning_map_error)
                elif meaning_map_df is not None and not meaning_map_df.empty:
                    st.markdown("#### Meaning Map (what the model “sees”)")
                    st.markdown(
                        "Each dot is one training email. Dots that sit close together use similar wording.\n"
                        "- Red = Spam style, Blue = Safe style\n"
                        "- Try hovering some dots: “Win a free iPhone!” and “Claim your prize now!” sit near each other, while “Project meeting agenda” lives far away.\n"
                        "- If dots overlap, the emails are look-alikes—that’s where mistakes are more likely."
                    )

                    backend_info = meaning_map_meta.get("embedding_backend", {})
                    if backend_info.get("kind") == "hashing-vectorizer":
                        error_hint = backend_info.get("error")
                        details = (
                            f" (reason: {error_hint})" if error_hint else ""
                        )
                        st.info(
                            "Using a lightweight hashed embedding fallback because the MiniLM "
                            f"model isn't available{details}. Meaning Map still works, but "
                            "cluster shapes are approximate."
                        )

                    toggles_col1, toggles_col2, toggles_col3 = st.columns(3)
                    show_examples = toggles_col1.checkbox(
                        "Show examples",
                        value=bool(ss.get("meaning_map_show_examples", False)),
                        help="Highlight a look-alike pair the model sees as very similar.",
                        key="meaning_map_show_examples",
                    )
                    show_centers = toggles_col2.checkbox(
                        "Show class centers",
                        value=bool(ss.get("meaning_map_show_centers", False)),
                        help="Plot the average spam vs safe position to show how far apart the styles sit.",
                        key="meaning_map_show_centers",
                    )
                    highlight_borderline = toggles_col3.checkbox(
                        "Highlight borderline emails",
                        value=bool(ss.get("meaning_map_highlight_borderline", False)),
                        help="Circle dots where the spam probability sits within ±0.10 of the dividing line.",
                        key="meaning_map_highlight_borderline",
                    )

                    if st.button(
                        "Show similar pair",
                        help="Highlight the closest spam look-alikes on the map.",
                        key="meaning_map_show_pair_button",
                    ):
                        st.session_state["meaning_map_show_examples"] = True
                        show_examples = True

                    chart = _build_meaning_map_chart(
                        meaning_map_df,
                        meaning_map_meta,
                        show_examples=show_examples,
                        show_class_centers=show_centers,
                        highlight_borderline=highlight_borderline,
                    )
                    if chart is not None:
                        st.altair_chart(chart, use_container_width=True)
                        st.caption(
                            "Line shows where the model splits Spam vs Safe. Emails close to the line are hardest to judge."
                        )
                    else:
                        st.info("Couldn’t render the map (empty or invalid chart data).")

                    class_counts_shown = meaning_map_meta.get("class_counts", {})
                    st.caption(
                        f"Training emails shown: {meaning_map_meta.get('shown', len(meaning_map_df))} "
                        f"(Spam: {class_counts_shown.get('spam', 0)}, Safe: {class_counts_shown.get('safe', 0)})."
                    )

                    if meaning_map_meta.get("sampled"):
                        st.caption("Showing 500 training emails for clarity.")

                    distance = meaning_map_meta.get("centroid_distance")
                    if isinstance(distance, (int, float)) and math.isfinite(distance):
                        st.caption(
                            f"Distance between centers ≈ {float(distance):.2f}. More distance = easier to classify."
                        )

                    if show_examples and isinstance(meaning_map_meta.get("pair"), dict):
                        pair_info = meaning_map_meta["pair"]
                        subjects = pair_info.get("subjects", [])
                        if len(subjects) >= 2:
                            label_pair = str(pair_info.get("label", "spam")).title()
                            st.markdown(
                                f"_{label_pair} look-alikes the model clusters together:_\n"
                                f"• {subjects[0]}\n"
                                f"• {subjects[1]}"
                            )

                    st.markdown("##### The Model’s Dividing Line")
                    st.markdown(
                        "The AI draws a simple straight line to separate Spam from Safe.\n"
                        "- Dots on the red side are predicted as spam.\n"
                        "- Dots on the blue side are predicted as safe.\n"
                        "- Emails close to the line are “borderline cases” — the trickiest to get right."
                    )
                    st.caption(
                        "How to read this: The straight line is the model’s “rule of thumb.” Emails far from the line are classified with confidence; emails near the line are uncertain and need more or clearer training examples."
                    )
                else:
                    st.info("Meaning Map not available for this run.")

                # 2) & 3) remain below
                st.markdown(
                    "- **Drawing a dividing line.** A simple linear classifier draws the straight line that separates spam from safe.\n"
                    "- **Using extra clues when uncertain.** Numeric guardrails step in when the text score is borderline."
                )

                st.markdown(
                    "Your labels define the explicit objective: ‘Spam vs Safe’. "
                    "MiniLM + the classifier discover an implicit strategy: a direction in meaning space that "
                    "separates the two. The numeric cues only assist when the text score is borderline."
                )

                if lang_mix_error:
                    st.caption(f"Language mix unavailable ({lang_mix_error}).")
                else:
                    st.caption(
                        format_language_mix_summary("train", lang_mix_train)
                    )

                guard_caption = None
                audit_info = None
                if hasattr(ss["model"], "audit_numeric_interplay"):
                    try:
                        audit_info = ss["model"].audit_numeric_interplay(X_tr_t, X_tr_b)
                    except Exception:
                        audit_info = None

                if audit_info:
                    pct_consulted = float(audit_info.get("pct_consulted", 0.0))
                    combine_strategy = getattr(ss["model"], "combine_strategy", "blend")
                    if combine_strategy == "threshold_shift":
                        avg_shift = float(audit_info.get("avg_threshold_shift", 0.0))
                        guard_caption = (
                            "Numeric guardrails consulted on "
                            f"{pct_consulted:.1f}% of borderline cases; avg shift Δτ={avg_shift:+.3f}."
                        )
                    else:
                        avg_blend = float(audit_info.get("avg_prob_blend_weight", 0.0))
                        guard_caption = (
                            "Numeric guardrails consulted on "
                            f"{pct_consulted:.1f}% of borderline cases; avg blend Δp={avg_blend:+.3f}."
                        )
                if not guard_caption:
                    guard_caption = (
                        "Numeric guardrails kick in when text scores are borderline,"
                        " lending numeric cues before final decisions."
                    )

                st.caption(guard_caption)
                st.markdown(GUARDRAIL_PANEL_STYLE, unsafe_allow_html=True)
                st.markdown("<div class='guardrail-panel'>", unsafe_allow_html=True)
                st.markdown("#### Guardrails for Borderline Emails")
                st.markdown(
                    "Some emails land right near the dividing line — the model isn’t sure.\n"
                    "In these cases, it looks at a few extra clues:"
                )
                st.markdown(
                    "- 🔗 suspicious links\n"
                    "- 🔊 ALL CAPS\n"
                    "- 💰 money symbols\n"
                    "- ⚡ urgency terms"
                )
                st.markdown(
                    "If several guardrails are triggered, the email tips toward spam."
                )
                st.markdown(
                    "<div class='guardrail-panel__example'><strong>Example shown:</strong> “Your invoice is ready” (0 guardrails) → Safe zone<br>“URGENT! CLICK NOW TO CLAIM $$$” (3 guardrails) → Spam zone</div>",
                    unsafe_allow_html=True,
                )

                try:
                    if meaning_map_df is not None and "borderline" in meaning_map_df.columns:
                        guardrail_chart = _build_borderline_guardrail_chart(
                            meaning_map_df,
                            meaning_map_meta,
                        )
                        left_col, right_col = st.columns([0.58, 0.42], gap="large")
                        with left_col:
                            st.markdown("<div class='guardrail-panel__chart'>", unsafe_allow_html=True)
                            if guardrail_chart is not None:
                                st.altair_chart(guardrail_chart, use_container_width=True)
                            else:
                                st.info(
                                    "Label more emails near the decision boundary to see guardrails in action."
                                )
                            st.markdown("</div>", unsafe_allow_html=True)

                        with right_col:
                            has_distance = "distance_abs" in meaning_map_df.columns
                            borderline_df = (
                                meaning_map_df.loc[meaning_map_df["borderline"] == True]
                                if "borderline" in meaning_map_df.columns
                                else pd.DataFrame()
                            )
                            cards_html_parts: List[str] = []
                            if not borderline_df.empty and has_distance:
                                sorted_borderline = borderline_df.sort_values(
                                    "distance_abs",
                                    ascending=True,
                                    na_position="last",
                                ).head(20)
                                for row in sorted_borderline.itertuples():
                                    subject_value = getattr(row, "subject_full", "")
                                    subject_text = str(subject_value).strip()
                                    if not subject_text:
                                        subject_text = "(no subject)"
                                    body_raw = getattr(row, "body_full", "")
                                    body_text = body_raw if isinstance(body_raw, str) else ""
                                    label_value = str(getattr(row, "label", "") or "")
                                    label_title = label_value.title() if label_value else "Unknown"
                                    label_icon = GUARDRAIL_LABEL_ICONS.get(label_value, "✉️")
                                    spam_prob = getattr(row, "spam_probability", float("nan"))
                                    if isinstance(spam_prob, (int, float)) and math.isfinite(spam_prob):
                                        score_text = f"p(spam) {spam_prob:.2f}"
                                    else:
                                        score_text = "Near the line"
                                    signals = _guardrail_signals(subject_text, body_text)
                                    badges_html = _guardrail_badges_html(signals)
                                    card_html = """
                                    <div class='guardrail-card'>
                                        <div class='guardrail-card__subject'>{subject}</div>
                                        <div class='guardrail-card__meta'>
                                            <span class='guardrail-card__label guardrail-card__label--{label_cls}'>{icon} {label}</span>
                                            <span>{score}</span>
                                        </div>
                                        <div class='guardrail-card__badges'>{badges}</div>
                                    </div>
                                    """.format(
                                        subject=html.escape(subject_text),
                                        label_cls=html.escape(label_value or ""),
                                        icon=label_icon,
                                        label=html.escape(label_title),
                                        score=html.escape(score_text),
                                        badges=badges_html,
                                    )
                                    cards_html_parts.append(card_html)

                            if cards_html_parts:
                                cards_html = "".join(cards_html_parts)
                                st.markdown(
                                    f"<div class='guardrail-card-list'>{cards_html}</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.info("No borderline emails yet.")
                    else:
                        st.info("Guardrails appear when borderline emails are present.")
                except Exception as exc:
                    st.warning(f"Guardrail view failed: {exc}")
                    logger.exception("Guardrail view failed")

                st.caption(
                    "How to read this: Guardrails don’t matter for clear-cut emails. But for borderline cases near the line, they add extra weight. More guardrails = higher chance of spam classification."
                )
                st.markdown("</div>", unsafe_allow_html=True)
                with st.expander("Debug (Meaning Map)"):
                    backend_info = (meaning_map_meta or {}).get("embedding_backend", {})
                    st.write(
                        {
                            "backend": backend_info,
                            "df_shape": None if meaning_map_df is None else meaning_map_df.shape,
                            "df_cols": [] if meaning_map_df is None else list(meaning_map_df.columns),
                        }
                    )
                labeled_rows_story = ss.get("labeled") or []
                if model_for_story and labeled_rows_story:
                    def _select_story_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                        spam_rows = [r for r in rows if (r.get("label") == "spam")]
                        safe_rows = [r for r in rows if (r.get("label") == "safe")]
                        selected: List[Dict[str, Any]] = []
                        for idx in range(2):
                            if idx < len(spam_rows):
                                selected.append({"label": "spam", "row": spam_rows[idx]})
                            if idx < len(safe_rows):
                                selected.append({"label": "safe", "row": safe_rows[idx]})
                        return selected

                    contrast_examples = _select_story_examples(labeled_rows_story)
                    if contrast_examples:
                        titles_story = [str(ex["row"].get("title", "")) for ex in contrast_examples]
                        bodies_story = [str(ex["row"].get("body", "")) for ex in contrast_examples]
                        combined_story = [
                            combine_text(t, b) for t, b in zip(titles_story, bodies_story)
                        ]

                        logits_story: Optional[np.ndarray] = None
                        try:
                            if hasattr(model_for_story, "predict_logit"):
                                raw_logits = model_for_story.predict_logit(combined_story)
                                logits_story = np.asarray(raw_logits, dtype=float).reshape(-1)
                        except Exception:
                            logits_story = None

                        if (
                            logits_story is None
                            or logits_story.size != len(contrast_examples)
                            or not np.all(np.isfinite(logits_story))
                        ):
                            try:
                                probs_story = model_for_story.predict_proba(
                                    titles_story, bodies_story
                                )
                                probs_story = np.asarray(probs_story, dtype=float)
                                if probs_story.ndim == 1:
                                    probs_story = probs_story.reshape(-1, 1)
                                if probs_story.shape[0] == len(contrast_examples):
                                    if probs_story.shape[1] == 1:
                                        p_spam_story = np.clip(
                                            probs_story[:, 0], 1e-6, 1 - 1e-6
                                        )
                                    else:
                                        spam_idx = getattr(
                                            model_for_story, "_i_spam", None
                                        )
                                        if spam_idx is None:
                                            classes_story = list(
                                                getattr(
                                                    model_for_story, "classes_", []
                                                )
                                            )
                                            if classes_story and "spam" in classes_story:
                                                spam_idx = classes_story.index("spam")
                                            else:
                                                spam_idx = (
                                                    1
                                                    if probs_story.shape[1] > 1
                                                    else 0
                                                )
                                        p_spam_story = np.clip(
                                            probs_story[:, int(spam_idx)], 1e-6, 1 - 1e-6
                                        )
                                    logits_story = np.log(
                                        p_spam_story / (1.0 - p_spam_story)
                                    )
                            except Exception:
                                logits_story = None

                        logits_story = (
                            logits_story
                            if logits_story is not None
                            and logits_story.size == len(contrast_examples)
                            else None
                        )

                        feature_weights: Dict[str, float] = {}
                        feature_names: Dict[str, str] = {}
                        if callable_or_attr(model_for_story, "numeric_feature_details"):
                            try:
                                nfd_story = model_for_story.numeric_feature_details().copy()
                                feature_weights = {
                                    str(row["feature"]): float(row["weight_per_std"])
                                    for _, row in nfd_story.iterrows()
                                }
                                feature_names = {
                                    key: FEATURE_DISPLAY_NAMES.get(key, key)
                                    for key in feature_weights
                                }
                            except Exception:
                                feature_weights = {}
                                feature_names = {}

                        reasons: List[str] = []
                        for ex_idx, ex in enumerate(contrast_examples):
                            label = ex["label"]
                            reason_text = (
                                "Typical spam phrasing"
                                if label == "spam"
                                else "Routine business phrasing"
                            )
                            if feature_weights:
                                try:
                                    feats = compute_numeric_features(
                                        titles_story[ex_idx], bodies_story[ex_idx]
                                    )
                                    if label == "spam":
                                        candidates = [
                                            feat
                                            for feat, value in feats.items()
                                            if value > 0 and feature_weights.get(feat, 0.0) > 0
                                        ]
                                        candidates.sort(
                                            key=lambda f: feature_weights.get(f, 0.0),
                                            reverse=True,
                                        )
                                    else:
                                        candidates = [
                                            feat
                                            for feat, value in feats.items()
                                            if value > 0 and feature_weights.get(feat, 0.0) < 0
                                        ]
                                        candidates.sort(
                                            key=lambda f: feature_weights.get(f, 0.0)
                                        )
                                    if candidates:
                                        top_feat = candidates[0]
                                        reason_text = feature_names.get(
                                            top_feat, top_feat
                                        )
                                except Exception:
                                    pass
                            reasons.append(reason_text)

                        clarity_badges: List[str] = []
                        for idx in range(len(contrast_examples)):
                            logit_value = (
                                float(logits_story[idx])
                                if logits_story is not None
                                and idx < logits_story.size
                                and np.isfinite(logits_story[idx])
                                else None
                            )
                            borderline = (
                                abs(logit_value) < 0.4 if logit_value is not None else False
                            )
                            clarity_badges.append("Borderline" if borderline else "Clear")

                        before_col, after_col = st.columns(2, gap="large")
                        with before_col:
                            st.markdown("**Before training**")
                            for ex in contrast_examples:
                                row = ex["row"]
                                label_value = ex["label"]
                                label_icon = {"spam": "🚩", "safe": "📥"}.get(
                                    label_value, "✉️"
                                )
                                title_text = _safe_subject(row)
                                excerpt = _excerpt(row.get("body"), 110)
                                st.markdown(
                                    f"""
                                    <div style="border:1px solid #E5E7EB;border-radius:10px;padding:0.75rem;margin-bottom:0.6rem;">
                                        <div style="font-size:0.8rem;color:#6B7280;margin-bottom:0.25rem;">{label_icon} {html.escape(label_value.title())}</div>
                                        <div style="font-weight:600;margin-bottom:0.35rem;">{html.escape(title_text)}</div>
                                        <div style="font-size:0.85rem;color:#374151;line-height:1.35;">{html.escape(excerpt)}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        with after_col:
                            st.markdown("**What the model sees now**")
                            for idx, ex in enumerate(contrast_examples):
                                row = ex["row"]
                                title_text = _safe_subject(row)
                                badge_label = clarity_badges[idx]
                                is_borderline = badge_label == "Borderline"
                                badge_color = "#f97316" if is_borderline else "#10b981"
                                reason_text = html.escape(reasons[idx])
                                st.markdown(
                                    f"""
                                    <div style="border:1px solid #CBD5F5;border-radius:10px;padding:0.75rem;margin-bottom:0.6rem;background:rgba(226,232,240,0.35);">
                                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem;">
                                            <div style="font-weight:600;color:#1E3A8A;">{html.escape(title_text)}</div>
                                            <span style="background:{badge_color};color:white;font-size:0.7rem;font-weight:600;padding:0.2rem 0.55rem;border-radius:999px;">{badge_label}</span>
                                        </div>
                                        <div style="font-size:0.85rem;color:#1F2937;line-height:1.35;">{reason_text}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.caption("Unavailable")
                elif model_for_story:
                    st.caption("Unavailable")

                # 2) Top signals the model noticed (plain list)
                shown_any_signals = False
                try:
                    # Prefer numeric-feature view if available (Hybrid model)
                    if hasattr(ss["model"], "numeric_feature_details"):
                        nfd = ss["model"].numeric_feature_details().copy()
                        nfd["friendly_name"] = nfd["feature"].map(FEATURE_DISPLAY_NAMES)
                        # Positive weights → Spam, Negative → Safe
                        top_spam = (
                            nfd.sort_values("weight_per_std", ascending=False)
                            .head(3)["friendly_name"].tolist()
                        )
                        top_safe = (
                            nfd.sort_values("weight_per_std", ascending=True)
                            .head(3)["friendly_name"].tolist()
                        )
                        st.markdown("**Top signals the model picked up**")
                        st.write(f"• Toward **Spam**: {', '.join(top_spam) if top_spam else '—'}")
                        st.write(f"• Toward **Safe**: {', '.join(top_safe) if top_safe else '—'}")
                        st.caption(
                            "These are simple cues (e.g., links, ALL-CAPS bursts, money/urgency hints) that nudged decisions."
                        )
                        shown_any_signals = True
                except Exception:
                    pass

                if not shown_any_signals:
                    # Fallback wording if coefficients aren’t available
                    st.markdown("**What it learned**")
                    st.write(
                        "The model pays more attention to words and cues that frequently appear in spam (e.g., urgent offers, suspicious links) "
                        "and learns to ignore everyday business phrases that tend to be safe."
                    )

                # 3) A couple of concrete examples the model saw (subjects only)
                paraphrase_demo: Optional[Tuple[str, str, float]] = None
                paraphrase_ready = bool(X_tr_t and X_tr_b and y_tr_labels)
                paraphrase_error: Optional[str] = None
                try:
                    if paraphrase_ready:
                        train_subjects = list(X_tr_t)
                        y_arr = list(y_tr_labels)
                        # pick first spam + first safe subject line available
                        spam_subj = next((s for s, y in zip(train_subjects, y_arr) if y == "spam"), None)
                        safe_subj = next((s for s, y in zip(train_subjects, y_arr) if y == "safe"), None)
                        if spam_subj or safe_subj:
                            st.markdown("**Examples it learned from**")
                            if spam_subj:
                                st.write(
                                    f"• Spam example: *{spam_subj[:100]}{'…' if len(spam_subj)>100 else ''}*"
                                )
                            if safe_subj:
                                st.write(
                                    f"• Safe example: *{safe_subj[:100]}{'…' if len(safe_subj)>100 else ''}*"
                                )

                        spam_subjects = [
                            s
                            for s, label in zip(train_subjects, y_arr)
                            if label == "spam" and isinstance(s, str) and s.strip()
                        ]
                        if len(spam_subjects) >= 2:
                            limited_subjects = spam_subjects[:100]
                            try:
                                subject_embeddings = encode_texts(limited_subjects)
                            except Exception as exc:
                                subject_embeddings = None
                                paraphrase_ready = False
                                paraphrase_error = (
                                    str(exc) or exc.__class__.__name__
                                )
                                logger.exception("Paraphrase embedding encoding failed")
                            if subject_embeddings is not None:
                                subject_embeddings = np.asarray(subject_embeddings)
                                if subject_embeddings.ndim == 2 and subject_embeddings.shape[0] >= 2:
                                    norms = np.linalg.norm(subject_embeddings, axis=1)
                                    normalized_subjects = [
                                        s.strip().lower() for s in limited_subjects
                                    ]
                                    best_score = -1.0
                                    best_pair: Optional[Tuple[int, int]] = None
                                    for i in range(len(limited_subjects)):
                                        if norms[i] == 0.0:
                                            continue
                                        for j in range(i + 1, len(limited_subjects)):
                                            if norms[j] == 0.0:
                                                continue
                                            if normalized_subjects[i] == normalized_subjects[j]:
                                                continue
                                            score = float(
                                                np.clip(
                                                    np.dot(subject_embeddings[i], subject_embeddings[j])
                                                    / (norms[i] * norms[j]),
                                                    -1.0,
                                                    1.0,
                                                )
                                            )
                                            if score > best_score:
                                                best_score = score
                                                best_pair = (i, j)
                                    if best_pair and best_score >= 0.7:
                                        paraphrase_demo = (
                                            limited_subjects[best_pair[0]],
                                            limited_subjects[best_pair[1]],
                                            best_score,
                                        )
                except Exception as exc:
                    paraphrase_demo = None
                    paraphrase_ready = False
                    paraphrase_error = str(exc) or exc.__class__.__name__
                    logger.exception("Paraphrase demo failed")

                # 4) What this means / next step
                st.markdown(
                    "Your model now has a simple **mental map** of what Spam vs. Safe looks like. "
                    "Next, we’ll check how well this map works on emails it hasn’t seen before."
                )
                if paraphrase_demo:
                    subj_a, subj_b, cos_sim = paraphrase_demo
                    subj_a_fmt = html.escape(_shorten_text(subj_a.strip()))
                    subj_b_fmt = html.escape(_shorten_text(subj_b.strip()))
                    st.markdown(
                        f"""
                        <div style="background: rgba(49, 51, 63, 0.05); border: 1px solid rgba(49, 51, 63, 0.08); border-radius: 0.6rem; padding: 0.75rem 1rem; margin-top: 0.75rem;">
                            <p style="font-size: 0.85rem; font-weight: 600; margin: 0 0 0.35rem 0;">Paraphrase demo:</p>
                            <ul style="margin: 0 0 0.5rem 1.1rem; padding: 0;">
                                <li style="font-size: 0.85rem; margin-bottom: 0.25rem;"><em>{subj_a_fmt}</em></li>
                                <li style="font-size: 0.85rem; margin-bottom: 0.25rem;"><em>{subj_b_fmt}</em></li>
                            </ul>
                            <p style="font-size: 0.8rem; color: rgba(49, 51, 63, 0.8); margin: 0;">Cosine similarity: {cos_sim:.2f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    if paraphrase_error:
                        st.caption(f"Paraphrase demo unavailable ({paraphrase_error}).")
                    else:
                        st.caption("Paraphrase demo unavailable.")
                if not ss.get("nerd_mode_train"):
                    st.markdown(
                        """
                        <div style="margin-top: 0.85rem; background: rgba(49, 51, 63, 0.05); border: 1px solid rgba(49, 51, 63, 0.08); border-radius: 0.6rem; padding: 0.85rem 1rem;">
                            <p style="font-size: 0.85rem; font-weight: 600; margin: 0 0 0.5rem 0;">What to expect from MiniLM</p>
                            <div style="display: flex; flex-wrap: wrap; gap: 1.25rem;">
                                <div style="flex: 1 1 180px;">
                                    <p style="font-size: 0.8rem; font-weight: 600; margin: 0 0 0.25rem 0;">Good at</p>
                                    <ul style="margin: 0; padding-left: 1.1rem; font-size: 0.8rem;">
                                        <li>Semantic paraphrases</li>
                                        <li>Phishing-y phrasing</li>
                                        <li>Short/mid emails</li>
                                    </ul>
                                </div>
                                <div style="flex: 1 1 180px;">
                                    <p style="font-size: 0.8rem; font-weight: 600; margin: 0 0 0.25rem 0;">Watch-outs</p>
                                    <ul style="margin: 0; padding-left: 1.1rem; font-size: 0.8rem;">
                                        <li>Truncation</li>
                                        <li>Non-English</li>
                                        <li>Ultra-short emails — rely more on numeric cues in these cases.</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.info("Go to **3) Evaluate** to test performance and choose a spam threshold.")

        except Exception as exc:
            st.caption(f"Training storyboard unavailable ({exc}).")
            logger.exception("Training storyboard failed")
    elif has_model or has_split_cache:
        st.caption("Unavailable")

    if ss.get("nerd_mode_train") and ss.get("model") is not None and parsed_split:
        with st.expander("Nerd Mode — what just happened (technical)", expanded=True):
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr_labels, y_te_labels = parsed_split
            train_texts_combined: list[str] = list(train_texts_combined_cache)
            train_embeddings: Optional[np.ndarray] = None
            train_embeddings_error: Optional[str] = None
            if not train_texts_combined and X_tr_t and X_tr_b:
                train_texts_combined = [combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)]
            if train_texts_combined:
                if has_embed:
                    try:
                        train_embeddings = cache_train_embeddings(train_texts_combined)
                        if getattr(train_embeddings, "size", 0) == 0:
                            train_embeddings = None
                    except Exception as exc:
                        train_embeddings = None
                        train_embeddings_error = str(exc) or exc.__class__.__name__
                    if train_embeddings is None:
                        try:
                            train_embeddings = encode_texts(train_texts_combined)
                            if getattr(train_embeddings, "size", 0) == 0:
                                train_embeddings = None
                        except Exception as exc:
                            train_embeddings = None
                            train_embeddings_error = str(exc) or exc.__class__.__name__
                        else:
                            train_embeddings_error = None
                else:
                    train_embeddings_error = "text encoder unavailable"
            try:
                st.markdown("**Data split**")
                st.markdown(
                    f"- Train set size: {len(y_tr_labels)}  \n"
                    f"- Test set size: {len(y_te_labels)}  \n"
                    f"- Class balance (train): {_counts(list(y_tr_labels))}  \n"
                    f"- Class balance (test): {_counts(list(y_te_labels))}"
                )
            except Exception:
                st.caption("Split details unavailable.")

            if lang_mix_error:
                st.caption(f"Language mix unavailable ({lang_mix_error}).")
            elif has_langdetect:
                try:
                    render_language_mix_chip_rows(lang_mix_train, lang_mix_test)
                except Exception as exc:
                    msg = str(exc) or exc.__class__.__name__
                    st.caption(f"Language mix unavailable ({msg}).")

            centroid_distance: Optional[float] = None
            centroid_message: Optional[str] = None
            try:
                if not train_texts_combined:
                    centroid_message = "Centroid distance unavailable (no training texts)."
                elif not has_embed:
                    centroid_message = "Centroid distance unavailable (text encoder unavailable)."
                elif train_embeddings is None or getattr(train_embeddings, "size", 0) == 0:
                    detail = train_embeddings_error or "embeddings missing"
                    centroid_message = f"Centroid distance unavailable ({detail})."
                elif not y_tr_labels:
                    centroid_message = "Centroid distance unavailable (labels missing)."
                else:
                    y_train_arr = np.asarray(y_tr_labels)
                    if train_embeddings.shape[0] != y_train_arr.shape[0]:
                        centroid_message = "Centroid distance unavailable (embedding count mismatch)."
                    else:
                        spam_mask = y_train_arr == "spam"
                        safe_mask = y_train_arr == "safe"
                        if not np.any(spam_mask) or not np.any(safe_mask):
                            centroid_message = "Centroid distance requires at least one spam and one safe email."
                        else:
                            spam_centroid = train_embeddings[spam_mask].mean(axis=0)
                            safe_centroid = train_embeddings[safe_mask].mean(axis=0)
                            spam_norm = float(np.linalg.norm(spam_centroid))
                            safe_norm = float(np.linalg.norm(safe_centroid))
                            if spam_norm == 0.0 or safe_norm == 0.0:
                                centroid_message = "Centroid distance unavailable (zero-length centroid)."
                            else:
                                cosine_similarity = float(
                                    np.clip(
                                        np.dot(spam_centroid, safe_centroid)
                                        / (spam_norm * safe_norm),
                                        -1.0,
                                        1.0,
                                    )
                                )
                                centroid_distance = 1.0 - cosine_similarity
            except Exception:
                centroid_message = "Centroid distance unavailable."

            if centroid_distance is not None:
                st.metric("Centroid cosine distance", f"{centroid_distance:.2f}")
                meter_width = float(np.clip(centroid_distance, 0.0, 1.0)) * 100.0
                meter_html = f"""
                <div style="margin-top:-0.5rem; margin-bottom:0.75rem;">
                    <div style="background:rgba(49, 51, 63, 0.1); border-radius:999px; height:10px; width:100%;">
                        <div style="background:linear-gradient(90deg, #4ade80, #22c55e); border-radius:999px; height:100%; width:{meter_width:.0f}%;"></div>
                    </div>
                    <div style="font-size:0.7rem; color:rgba(49,51,63,0.6); margin-top:0.25rem;">0 = identical • 1 = orthogonal</div>
                </div>
                """
                st.markdown(meter_html, unsafe_allow_html=True)
                st.caption(
                    "Bigger distance means spam and safe live farther apart in meaning space—good separation."
                )
            elif centroid_message:
                st.caption(centroid_message)

            st.markdown("#### Decision margin spread (text head)")
            margins: Optional[np.ndarray] = None
            margin_error = False
            model_obj = ss.get("model")
            if not train_texts_combined:
                st.caption("Margin distribution unavailable (no training texts).")
            else:
                logits: Optional[np.ndarray] = None
                try:
                    if hasattr(model_obj, "predict_logit"):
                        logits = np.asarray(model_obj.predict_logit(train_texts_combined), dtype=float)
                    if logits is None or logits.size == 0:
                        probs = model_obj.predict_proba(X_tr_t, X_tr_b)[:, getattr(model_obj, "_i_spam", 1)]
                        probs = np.clip(probs, 1e-6, 1 - 1e-6)
                        logits = np.log(probs / (1.0 - probs))
                    logits = np.asarray(logits, dtype=float).reshape(-1)
                    logits = logits[np.isfinite(logits)]
                    if logits.size > 0:
                        margins = np.abs(logits)
                except Exception as exc:
                    st.caption(f"Margin distribution unavailable: {exc}")
                    margins = None
                    margin_error = True

            if margins is not None and margins.size > 0:
                try:
                    bins = min(12, max(5, int(np.ceil(np.log2(margins.size + 1)))))
                    counts, edges = np.histogram(margins, bins=bins)
                    labels = [
                        f"{edges[i]:.2f}–{edges[i + 1]:.2f}" for i in range(len(edges) - 1)
                    ]
                    hist_df = pd.DataFrame({"margin": labels, "count": counts})
                    st.bar_chart(hist_df.set_index("margin"), width="stretch")
                    st.caption(
                        "Higher margins = clearer decisions; lots of small margins means many borderline emails."
                    )
                except Exception as exc:
                    st.caption(f"Could not render margin histogram: {exc}")
            elif train_texts_combined and not margin_error:
                st.caption("Margin distribution unavailable (no valid logit values).")

            params = ss.get("train_params", {})
            st.markdown("**Parameters used**")
            st.markdown(
                f"- Hold-out fraction: {params.get('test_size', '—')}  \n"
                f"- Random seed: {params.get('random_state', '—')}  \n"
                f"- Max iterations: {params.get('max_iter', '—')}  \n"
                f"- C (inverse regularization): {params.get('C', '—')}"
            )

            calibrate_default = bool(ss.get("calibrate_probabilities", False))
            calib_toggle = st.toggle(
                "Calibrate probabilities (test set)",
                value=calibrate_default,
                key="train_calibrate_toggle",
                help="Platt scaling if test size ≥ 30, else isotonic disabled.",
                disabled=not has_calibration,
            )
            calib_active = bool(calib_toggle and has_calibration)
            ss["calibrate_probabilities"] = bool(calib_active)

            calibration_details = None
            if not has_calibration:
                st.caption("Unavailable")
                if hasattr(model_obj, "set_calibration"):
                    try:
                        model_obj.set_calibration(None)
                    except Exception:
                        pass
                ss["calibration_result"] = None
            elif calib_active:
                test_size = len(y_te_labels) if y_te_labels is not None else 0
                if test_size < 30:
                    st.caption("Unavailable")
                    if hasattr(model_obj, "set_calibration"):
                        try:
                            model_obj.set_calibration(None)
                        except Exception:
                            pass
                    calibration_details = {"status": "too_small", "test_size": test_size}
                    ss["calibration_result"] = calibration_details
                elif model_obj is None:
                    st.caption("Unavailable")
                else:
                    try:
                        spam_index = getattr(model_obj, "_i_spam", 1)
                        if hasattr(model_obj, "predict_proba_base"):
                            base_matrix = model_obj.predict_proba_base(X_te_t, X_te_b)
                        else:
                            base_matrix = model_obj.predict_proba(X_te_t, X_te_b)
                        base_matrix = np.asarray(base_matrix, dtype=float)
                        if base_matrix.ndim != 2 or base_matrix.shape[0] == 0:
                            raise ValueError("Empty probability matrix from model.")
                        base_probs = base_matrix[:, spam_index]
                        base_probs = np.clip(base_probs, 1e-6, 1 - 1e-6)
                        y_true01 = np.asarray(_y01(list(y_te_labels)), dtype=float)
                        base_logits = np.log(base_probs / (1.0 - base_probs))
                        calibrator = PlattProbabilityCalibrator(
                            random_state=int(params.get("random_state", 42))
                        )
                        calibrator.fit(base_logits, list(y_te_labels))
                        if hasattr(model_obj, "set_calibration"):
                            model_obj.set_calibration(calibrator)
                        calibrated_matrix = model_obj.predict_proba(X_te_t, X_te_b)
                        calibrated_matrix = np.asarray(calibrated_matrix, dtype=float)
                        calibrated_probs = calibrated_matrix[:, spam_index]
                        calibrated_probs = np.clip(calibrated_probs, 1e-6, 1 - 1e-6)
                        brier_before = float(np.mean((base_probs - y_true01) ** 2))
                        brier_after = float(np.mean((calibrated_probs - y_true01) ** 2))
                        bins = np.linspace(0.0, 1.0, 11)
                        reliability_rows: List[Dict[str, object]] = []
                        stages = [
                            ("Before calibration", base_probs),
                            ("After calibration", calibrated_probs),
                        ]
                        for stage_label, probs in stages:
                            bin_ids = np.digitize(probs, bins, right=True) - 1
                            bin_ids = np.clip(bin_ids, 0, len(bins) - 2)
                            for b in range(len(bins) - 1):
                                mask = bin_ids == b
                                if not np.any(mask):
                                    continue
                                reliability_rows.append(
                                    {
                                        "stage": stage_label,
                                        "bin": b,
                                        "expected": float(np.mean(probs[mask])),
                                        "observed": float(np.mean(y_true01[mask])),
                                        "count": int(mask.sum()),
                                    }
                                )
                        reliability_df = pd.DataFrame(reliability_rows)
                        reliability_df["bin_label"] = reliability_df["bin"].map(
                            lambda b: f"{bins[b]:.1f}–{bins[b + 1]:.1f}"
                        )
                        calibration_details = {
                            "status": "ok",
                            "brier_before": brier_before,
                            "brier_after": brier_after,
                            "test_size": test_size,
                            "reliability": reliability_df,
                        }
                        ss["calibration_result"] = calibration_details
                    except Exception:
                        st.caption("Unavailable")
                        if hasattr(model_obj, "set_calibration"):
                            try:
                                model_obj.set_calibration(None)
                            except Exception:
                                pass
                        calibration_details = {"status": "error"}
                        ss["calibration_result"] = calibration_details
            else:
                if hasattr(model_obj, "set_calibration"):
                    try:
                        model_obj.set_calibration(None)
                    except Exception:
                        pass
                ss["calibration_result"] = None

            if calibration_details and calibration_details.get("status") == "ok":
                brier_before = calibration_details["brier_before"]
                brier_after = calibration_details["brier_after"]
                delta = brier_after - brier_before
                col_b1, col_b2 = st.columns(2)
                col_b1.metric("Brier score (uncalibrated)", f"{brier_before:.4f}")
                col_b2.metric(
                    "Brier score (calibrated)",
                    f"{brier_after:.4f}",
                    delta=f"{delta:+.4f}",
                    delta_color="inverse",
                )
                st.caption(
                    f"Calibrated on {calibration_details['test_size']} hold-out examples using Platt scaling."
                )
                reliability_df = calibration_details.get("reliability")
                if reliability_df is not None and not reliability_df.empty:
                    chart = (
                        alt.Chart(reliability_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("expected:Q", title="Mean predicted probability"),
                            y=alt.Y("observed:Q", title="Observed spam rate"),
                            color=alt.Color("stage:N", title=""),
                            tooltip=[
                                alt.Tooltip("stage:N", title="Series"),
                                alt.Tooltip("bin_label:N", title="Bin"),
                                alt.Tooltip("expected:Q", title="Predicted", format=".2f"),
                                alt.Tooltip("observed:Q", title="Observed", format=".2f"),
                                alt.Tooltip("count:Q", title="Count"),
                            ],
                        )
                        .properties(height=260)
                    )
                    diagonal = alt.Chart(
                        pd.DataFrame({"expected": [0.0, 1.0], "observed": [0.0, 1.0]})
                    ).mark_line(strokeDash=[4, 4], color="gray")
                    st.altair_chart(chart + diagonal, use_container_width=True)
                    st.caption(
                        "We align predicted probabilities to reality. If the curve hugs the diagonal, scores are trustworthy."
                    )

            st.markdown(f"**Model object**: `{model_kind_string(ss['model'])}`")

            st.markdown("### Interpretability & tuning")
            try:
                coef_details = ss["model"].numeric_feature_details().copy()
                coef_details["friendly_name"] = coef_details["feature"].map(
                    FEATURE_DISPLAY_NAMES
                )
                st.caption(
                    "Positive weights push toward the **spam** class; negative toward **safe**. "
                    "Values are log-odds after standardization."
                )

                chart_data = (
                    coef_details.sort_values("weight_per_std", ascending=True)
                    .set_index("friendly_name")["weight_per_std"]
                )
                st.bar_chart(chart_data, width="stretch")
                st.caption(
                    "Bars to the right push toward 'Spam'; left bars push toward 'Safe'. Longer bar = stronger nudge."
                )

                display_df = coef_details.assign(
                    odds_multiplier_plus_1sigma=coef_details["odds_multiplier_per_std"],
                    approx_pct_change_odds=(coef_details["odds_multiplier_per_std"] - 1.0) * 100.0,
                )[
                    [
                        "friendly_name",
                        "base_weight_per_std",
                        "user_adjustment",
                        "weight_per_std",
                        "odds_multiplier_plus_1sigma",
                        "approx_pct_change_odds",
                        "train_mean",
                        "train_std",
                    ]
                ]

                st.dataframe(
                    display_df.rename(
                        columns={
                            "friendly_name": "Feature",
                            "base_weight_per_std": "Learned log-odds (+1σ)",
                            "user_adjustment": "Your adjustment (+1σ)",
                            "weight_per_std": "Adjusted log-odds (+1σ)",
                            "odds_multiplier_plus_1sigma": "Adjusted odds multiplier (+1σ)",
                            "approx_pct_change_odds": "%Δ odds from adjustment (+1σ)",
                            "train_mean": "Train mean",
                            "train_std": "Train std",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "Base weights come from training. Use the sliders below to nudge each cue if your domain knowledge "
                    "suggests it should count more or less. Adjustments apply per standard deviation of the raw feature."
                )

                st.markdown("#### What influenced the score (span knockout demo)")
                if not train_texts_combined:
                    st.caption("Span influence demo unavailable (no training emails).")
                elif model_obj is None or not hasattr(model_obj, "predict_logit"):
                    st.caption("Span influence demo unavailable (text-head logits missing).")
                else:
                    options = list(range(len(train_texts_combined)))

                    def _format_train_option(i: int) -> str:
                        label = (
                            y_tr_labels[i]
                            if y_tr_labels and 0 <= i < len(y_tr_labels)
                            else "?"
                        )
                        subject = X_tr_t[i] if X_tr_t and 0 <= i < len(X_tr_t) else ""
                        if not isinstance(subject, str) or not subject.strip():
                            subject = train_texts_combined[i][:80]
                        subject_short = _shorten_text(str(subject).strip(), limit=80)
                        return f"{i + 1}. [{label.upper()}] {subject_short}" if label else subject_short

                    selected_idx = st.selectbox(
                        "Pick a training email",
                        options,
                        format_func=_format_train_option,
                        key="nerd_span_email_index",
                    )

                    selected_text = train_texts_combined[selected_idx]
                    candidate_spans = extract_candidate_spans(selected_text)

                    if not candidate_spans:
                        st.info("No influential spans detected for this email.")
                    else:
                        cache_bucket = ss.setdefault("nerd_span_cache", {})
                        text_hash = hashlib.sha1(selected_text.encode("utf-8")).hexdigest()
                        cache_key = f"{id(model_obj)}:{text_hash}"
                        cached = cache_bucket.get(cache_key)

                        if not cached:
                            try:
                                base_logit = float(model_obj.predict_logit([selected_text])[0])
                            except Exception as exc:
                                base_logit = None
                                cached = {"error": str(exc)}
                            else:
                                influence_rows: list[dict[str, float | str]] = []
                                for span_text, (start, end) in candidate_spans:
                                    modified = selected_text[:start] + selected_text[end:]
                                    try:
                                        new_logit = float(
                                            model_obj.predict_logit([modified])[0]
                                        )
                                    except Exception:
                                        continue
                                    delta = base_logit - new_logit
                                    influence_rows.append(
                                        {
                                            "span": span_text,
                                            "delta": delta,
                                        }
                                    )

                                influence_rows.sort(
                                    key=lambda row: row.get("delta", 0.0), reverse=True
                                )
                                cached = {
                                    "base_logit": base_logit,
                                    "rows": influence_rows,
                                }
                            cache_bucket[cache_key] = cached

                        if cached.get("error"):
                            st.caption(
                                "Could not compute span influence: "
                                f"{cached['error']}"
                            )
                        else:
                            base_logit = cached.get("base_logit")
                            rows = cached.get("rows", [])
                            positive_rows = [
                                row
                                for row in rows
                                if isinstance(row.get("delta"), (int, float))
                                and float(row["delta"]) > 0.0
                            ]

                            if base_logit is not None:
                                st.caption(
                                    f"Base text-head logit: {float(base_logit):+.3f}"
                                )

                            if not positive_rows:
                                st.info(
                                    "Removing detected spans did not lower the score."
                                )
                            else:
                                top_rows = positive_rows[:8]
                                display_rows = []
                                for row in top_rows:
                                    span_text = str(row.get("span", "")).replace("\n", " ")
                                    span_text = " ".join(span_text.split())
                                    if len(span_text) > 120:
                                        span_text = span_text[:117] + "…"
                                    display_rows.append(
                                        {
                                            "Span": span_text,
                                            "Δ logit (drop)": round(float(row["delta"]), 4),
                                        }
                                    )

                                df_spans = pd.DataFrame(display_rows)
                                st.dataframe(df_spans, hide_index=True, width="stretch")
                                st.caption(
                                    "We remove phrases and see how the score drops; bigger drops = more influence."
                                )

                st.markdown("#### Plain-language explanations & manual tweaks")
                for row in coef_details.itertuples():
                    feat = row.feature
                    friendly = FEATURE_DISPLAY_NAMES.get(feat, feat)
                    explanation = FEATURE_PLAIN_LANGUAGE.get(feat, "")
                    st.markdown(f"**{friendly}** — {explanation}")
                    slider_key = f"adj_slider_{feat}"
                    current_setting = ss["numeric_adjustments"][feat]
                    if slider_key in st.session_state and st.session_state[slider_key] != current_setting:
                        st.session_state[slider_key] = current_setting
                    new_adj = st.slider(
                        f"Adjustment for {friendly} (log-odds per +1σ)",
                        min_value=-1.5,
                        max_value=1.5,
                        value=float(current_setting),
                        step=0.1,
                        key=slider_key,
                    )
                    if new_adj != ss["numeric_adjustments"][feat]:
                        ss["numeric_adjustments"][feat] = new_adj
                        if ss.get("model"):
                            ss["model"].apply_numeric_adjustments(ss["numeric_adjustments"])
            except Exception as e:
                st.caption(f"Coefficients unavailable: {e}")

            st.markdown("#### Embedding prototypes & nearest neighbors")
            try:
                if X_tr_t and X_tr_b:
                    X_train_texts = train_texts_combined or [
                        combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)
                    ]
                    X_train_emb = train_embeddings
                    if X_train_emb is None or getattr(X_train_emb, "size", 0) == 0:
                        X_train_emb = encode_texts(X_train_texts)
                    y_train_arr = np.array(y_tr_labels)

                    def prototype_for(cls):
                        mask = y_train_arr == cls
                        if not np.any(mask):
                            return None
                        return X_train_emb[mask].mean(axis=0, keepdims=True)

                    def top_nearest(query_vec, k=5):
                        if query_vec is None:
                            return np.array([]), np.array([])
                        sims = (X_train_emb @ query_vec.T).ravel()
                        order = np.argsort(-sims)
                        top_k = order[: min(k, len(order))]
                        return top_k, sims[top_k]

                    rendered_prototypes = False
                    for cls in CLASSES:
                        proto = prototype_for(cls)
                        if proto is None:
                            st.write(f"No training emails for {cls} yet.")
                            continue
                        rendered_prototypes = True
                        idx, sims = top_nearest(proto, k=5)
                        st.markdown(f"**{cls.capitalize()} prototype — most similar training emails**")
                        for i, (ix, sc) in enumerate(zip(idx, sims), 1):
                            text_full = X_train_texts[ix]
                            parts = text_full.split("\n", 1)
                            title_i = parts[0]
                            body_i = parts[1] if len(parts) > 1 else ""
                            st.write(f"{i}. *{title_i}*  — sim={sc:.2f}")
                            preview = body_i[:200]
                            st.caption(preview + ("..." if len(body_i) > 200 else ""))
                    if rendered_prototypes:
                        st.caption(
                            "We average each class’s meaning; these are the emails closest to that average."
                        )
                else:
                    st.caption("Embedding details unavailable (no training texts).")
            except Exception as e:
                st.caption(f"Interpretability view unavailable: {e}")



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


STAGE_RENDERERS = {
    'intro': render_intro_stage,
    'overview': render_overview_stage,
    'data': render_data_stage,
    'train': render_train_stage,
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
