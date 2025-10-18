from __future__ import annotations

import html
import math
from typing import Any, List, Optional, Tuple

import pandas as pd
import streamlit as st


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


def _render_numeric_clue_preview(assist_center: float, uncertainty_band: float) -> None:
    chip_html_parts = [
        "<span class='numeric-clue-preview__chip'>🔗 Suspicious link</span>",
        "<span class='numeric-clue-preview__chip'>🔊 ALL CAPS</span>",
        "<span class='numeric-clue-preview__chip'>💰 Money cue</span>",
        "<span class='numeric-clue-preview__chip'>⚡ Urgent phrasing</span>",
    ]

    center_text = "τ"
    if isinstance(assist_center, (int, float)) and math.isfinite(assist_center):
        center_text = f"τ ≈ {assist_center:.2f}"

    band_amount: str | None = None
    if isinstance(uncertainty_band, (int, float)) and math.isfinite(uncertainty_band):
        band_amount = f"{uncertainty_band:.2f}"

    band_label = "Assist window"
    low_label = "τ − band"
    high_label = "τ + band"
    if band_amount is not None:
        band_label = f"Assist window ±{band_amount}"
        low_label = f"τ − {band_amount}"
        high_label = f"τ + {band_amount}"

    preview_html = """
<div class='numeric-clue-preview'>
  <div class='numeric-clue-preview__header'>
    <span class='numeric-clue-preview__center'>{center}</span>
    <span class='numeric-clue-preview__band-label'>{band}</span>
  </div>
  <div class='numeric-clue-preview__band'>
    <div class='numeric-clue-preview__ticks'>
      <span>{low}</span>
      <span>Inside band</span>
      <span>{high}</span>
    </div>
    <div class='numeric-clue-preview__chips'>
      {chips}
    </div>
  </div>
  <p class='numeric-clue-preview__note'>Numeric guardrails watch for these structured cues before overriding the text score.</p>
</div>
""".format(
        center=html.escape(center_text),
        band=html.escape(band_label),
        low=html.escape(low_label),
        high=html.escape(high_label),
        chips="".join(chip_html_parts),
    )

    st.markdown(preview_html, unsafe_allow_html=True)


def _render_numeric_clue_cards(df: Optional[pd.DataFrame]) -> None:
    if df is None or df.empty:
        st.info("No emails to review yet — train the model to surface numeric clues.")
        return

    if "numeric_clues" not in df.columns:
        st.info("Numeric clue details were unavailable for this run.")
        return

    working = df.copy()
    try:
        mask = working["numeric_clues"].apply(lambda val: bool(val))
    except Exception:
        mask = pd.Series([False] * len(working), index=working.index)

    subset = working.loc[mask]
    if "borderline" in subset.columns:
        try:
            borderline_mask = subset["borderline"].astype(bool)
        except Exception:
            borderline_mask = pd.Series([False] * len(subset), index=subset.index)
        subset = subset.loc[borderline_mask]
    if subset.empty:
        st.info(
            "No emails inside the assist window needed the extra numeric guardrails — the text score was decisive."
        )
        return

    cards: List[str] = []
    for _, row in subset.iterrows():
        subject = html.escape(row.get("subject_tooltip", row.get("subject_full", "(untitled)")) or "")
        reason = html.escape(row.get("reason", "Mostly positioned by wording similarity."))
        clues = row.get("numeric_clues") or []
        clue_html = "".join(
            f"<span class='numeric-clue-chip numeric-clue-chip--{html.escape(clue.get('direction', 'unknown'))}'><span>{html.escape(str(clue.get('label', '')))}</span></span>"
            for clue in clues
        )
        cards.append(
            """
<div class='numeric-clue-card'>
  <div class='numeric-clue-card__header'>
    <div class='numeric-clue-card__subject'>{subject}</div>
  </div>
  <div class='numeric-clue-card__reason'>{reason}</div>
  <div class='numeric-clue-card__chips'>{chips}</div>
</div>
""".format(subject=subject, reason=reason, chips=clue_html)
        )

    run_marker_raw = subset.get("run_marker") if isinstance(subset, pd.DataFrame) else None
    run_marker = html.escape(str(run_marker_raw or "initial"))
    wrapper = "<div class='numeric-clue-card-grid' data-run='{}'>{}</div>".format(
        run_marker,
        "".join(cards),
    )
    st.markdown(wrapper, unsafe_allow_html=True)
