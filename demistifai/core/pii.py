"""PII data helpers shared across analytics and UI layers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from demistifai.config.tokens import TOKEN_POLICY, PII_DISPLAY_LABELS

__all__ = [
    "summarize_pii_counts",
    "format_pii_summary",
    "apply_pii_replacements",
]


def summarize_pii_counts(
    detailed_hits: Dict[int, Dict[str, List[Dict[str, Any]]]]
) -> Dict[str, int]:
    """Aggregate detected PII spans across dataset rows."""

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
    """Format a summary string describing PII counts by type."""

    return " • ".join(
        f"{label}: {int(counts.get(key, 0) or 0)}" for key, label in PII_DISPLAY_LABELS
    )


def apply_pii_replacements(text: str, spans: List[Dict[str, Any]]) -> str:
    """Replace detected spans within text using configured PII tokens."""

    if not spans:
        return text

    ordered = sorted(spans, key=lambda span: span["start"])
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
