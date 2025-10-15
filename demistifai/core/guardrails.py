from __future__ import annotations
import html, re
from typing import Any, Dict, List, Tuple

# If these exist in your constants module, import them:
# from demistifai.core.constants import URGENCY, URGENCY_TERMS
# Otherwise, keep these two lists here or pass them in via a setup function.
try:
    from demistifai.core.constants import URGENCY, URGENCY_TERMS  # type: ignore
except Exception:
    URGENCY, URGENCY_TERMS = [], []  # fallback to empty

GUARDRAIL_BADGE_DEFS = [
    ("link", "🔗", "Suspicious link"),
    ("caps", "🔊", "ALL CAPS"),
    ("money", "💰", "Money"),
    ("urgency", "⚡", "Urgency"),
]

GUARDRAIL_CAPS_THRESHOLD = 0.3
GUARDRAIL_URGENCY_TERMS = {term.lower() for term in URGENCY} | {term.lower() for term in URGENCY_TERMS}
GUARDRAIL_LABEL_ICONS = {"spam": "🚩", "safe": "📥"}

URL_CANDIDATE_RE = re.compile(r"(?i)\b(?:https?://|www\.)[\w\-._~:/?#\[\]@!$&'()*+,;=%]+")
MONEY_CANDIDATE_RE = re.compile(
    r"(?i)(?:\$\s?\d[\d,]*(?:\.\d{2})?|\b(?:usd|eur|gbp|aud|cad|sgd)\s?\$?\d[\d,]*(?:\.\d{2})?|\b\d[\d,]*(?:\.\d{2})?\s?(?:usd|eur|gbp|aud|cad|dollars)\b)"
)
ALERT_PHRASE_RE = re.compile(
    r"(?i)\b(?:verify|reset|account|urgent)(?:\s+(?:verify|reset|account|urgent|your|the|this|that|my|our)){1,3}\b"
)

# These helpers are expected to exist in your codebase.
# Keep the names to avoid breaking references:
# _count_suspicious_links, _has_suspicious_tld, _caps_ratio, _count_money_mentions
from demistifai.core.utils import (  # make sure these exist; otherwise keep them here
    _count_suspicious_links, _has_suspicious_tld, _caps_ratio, _count_money_mentions,  # type: ignore
)

def _guardrail_signals(subject: str, body: str) -> Dict[str, bool]:
    text = f"{subject or ''}\n{body or ''}".strip() or (subject or body or "")
    text_lower = text.lower()

    suspicious_links = False
    if text:
        try:
            suspicious_links = (_count_suspicious_links(text) > 0) or _has_suspicious_tld(text)
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
            f"<span class='{classes}' title='{html.escape(label)}' aria-label='{html.escape(label)}'>"
            f"{icon}<span style='font-size:0.75rem;'>{html.escape(label)}</span></span>"
        )
    return "".join(badges)

def extract_candidate_spans(text: str) -> List[Tuple[str, Tuple[int, int]]]:
    if not text:
        return []
    matches: list[Tuple[int, int, str]] = []
    for pattern in (URL_CANDIDATE_RE, MONEY_CANDIDATE_RE, ALERT_PHRASE_RE):
        for match in pattern.finditer(text):
            start, end = match.span()
            snippet = text[start:end]
            if snippet.strip():
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
