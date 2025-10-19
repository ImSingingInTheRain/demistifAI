from __future__ import annotations

from typing import Any, List, Tuple

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

FEATURE_CLUE_CHIPS = {
    "num_links_external": {
        "spam": "🔗 Many external links",
        "safe": "🔗 Few external links",
    },
    "has_suspicious_tld": {
        "spam": "🌐 Risky domain in links",
        "safe": "🌐 Links look safe",
    },
    "punct_burst_ratio": {
        "spam": "❗ Intense punctuation",
        "safe": "❗ Calm punctuation",
    },
    "money_symbol_count": {
        "spam": "💰 Money cues",
        "safe": "💰 No money cues",
    },
    "urgency_terms_count": {
        "spam": "⏱️ Urgent wording",
        "safe": "⏱️ Neutral urgency",
    },
}


def _join_phrases(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _reason_from_contributions(
    label: str, contributions: List[Tuple[str, float, float]]
) -> str:
    if not contributions:
        return "Mostly positioned by wording similarity."

    threshold = 0.08
    phrases: List[str] = []
    if label == "spam":
        for feat, _z, contrib in contributions:
            if contrib > threshold:
                phrases.append(FEATURE_REASON_SPAM.get(feat, feat))
    else:
        for feat, _z, contrib in contributions:
            if contrib < -threshold:
                phrases.append(FEATURE_REASON_SAFE.get(feat, feat))

    phrases = [p for p in phrases if p]
    if not phrases:
        return "Mostly positioned by wording similarity."
    summary = _join_phrases(phrases[:3])
    if not summary:
        return "Mostly positioned by wording similarity."
    return f"Signals: {summary}"


def _extract_numeric_clues(
    contributions: List[Tuple[str, float, float]],
    *,
    threshold: float = 0.08,
) -> list[dict[str, Any]]:
    """Return structured clue details for sizable numeric contributions."""

    clues: list[dict[str, Any]] = []
    if not contributions:
        return clues

    for feature, _z_score, contrib in contributions:
        direction: str | None = None
        if contrib >= threshold:
            direction = "spam"
        elif contrib <= -threshold:
            direction = "safe"
        if direction is None:
            continue
        mapping = FEATURE_CLUE_CHIPS.get(feature, {}) if isinstance(feature, str) else {}
        label = mapping.get(direction)
        if not label:
            label = str(feature)
        clues.append(
            {
                "feature": feature,
                "direction": direction,
                "label": label,
                "contribution": float(contrib),
            }
        )

    clues.sort(key=lambda item: abs(item.get("contribution", 0.0)), reverse=True)
    return clues
