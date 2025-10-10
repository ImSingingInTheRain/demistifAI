from __future__ import annotations

import html
import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.decomposition import PCA

from demistifai.modeling import (
    cache_train_embeddings,
    combine_text,
    embedding_backend_info,
    numeric_feature_contributions,
)


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


def _make_selection_point(fields: list[str], *, on: str, empty: str) -> Any | None:
    """Create an Altair selection/parameter that works across Altair versions."""

    try:
        if hasattr(alt, "selection_point"):
            return alt.selection_point(fields=fields, on=on, empty=empty)
    except Exception:
        pass

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
            combined = sel
    return combined


def _excerpt(text: str, n: int = 120) -> str:
    return (text or "")[:n] + ("…" if text and len(text) > n else "")


def _join_phrases(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _reason_from_contributions(label: str, contributions: List[Tuple[str, float, float]]) -> str:
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

def _line_box_intersections(
    bounds: Tuple[float, float, float, float],
    weight: np.ndarray,
    intercept: float,
) -> List[Tuple[float, float]]:
    """Return intersection points of a line with the bounding box."""

    x_min, x_max, y_min, y_max = bounds
    w1, w2 = float(weight[0]), float(weight[1])
    points: List[Tuple[float, float]] = []

    if abs(w2) > 1e-9:
        for x in (x_min, x_max):
            y = (-w1 * x - intercept) / w2
            if y_min - 1e-9 <= y <= y_max + 1e-9:
                points.append((x, y))
    if abs(w1) > 1e-9:
        for y in (y_min, y_max):
            x = (-w2 * y - intercept) / w1
            if x_min - 1e-9 <= x <= x_max + 1e-9:
                points.append((x, y))

    if len(points) < 2:
        return [(x_min, y_min), (x_max, y_max)]
    return points[:2]


def _clip_polygon_with_halfplane(
    polygon: List[Tuple[float, float]],
    weight: np.ndarray,
    intercept: float,
    threshold: float,
    *,
    keep_leq: bool,
) -> List[Tuple[float, float]]:
    """Clip polygon with a half-plane defined by weight·x + intercept <= threshold."""

    if not polygon:
        return []

    def inside(point: Tuple[float, float]) -> bool:
        value = weight[0] * point[0] + weight[1] * point[1] + intercept
        return value <= threshold + 1e-9 if keep_leq else value >= threshold - 1e-9

    output: List[Tuple[float, float]] = []
    prev = polygon[-1]
    prev_inside = inside(prev)
    for curr in polygon:
        curr_inside = inside(curr)
        if curr_inside:
            if not prev_inside:
                denom = (weight[0] * (curr[0] - prev[0]) + weight[1] * (curr[1] - prev[1]))
                if abs(denom) < 1e-12:
                    intersection = curr
                else:
                    t = (
                        threshold
                        - (weight[0] * prev[0] + weight[1] * prev[1] + intercept)
                    ) / denom
                    t = max(0.0, min(1.0, t))
                    intersection = (
                        prev[0] + t * (curr[0] - prev[0]),
                        prev[1] + t * (curr[1] - prev[1]),
                    )
                output.append(intersection)
            output.append(curr)
        elif prev_inside:
            denom = (weight[0] * (curr[0] - prev[0]) + weight[1] * (curr[1] - prev[1]))
            if abs(denom) < 1e-12:
                intersection = curr
            else:
                t = (
                    threshold
                    - (weight[0] * prev[0] + weight[1] * prev[1] + intercept)
                ) / denom
                t = max(0.0, min(1.0, t))
                intersection = (
                    prev[0] + t * (curr[0] - prev[0]),
                    prev[1] + t * (curr[1] - prev[1]),
                )
            output.append(intersection)
        prev, prev_inside = curr, curr_inside
    return output


def _reduce_embeddings_to_2d(embeddings: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    if embeddings.ndim != 2 or embeddings.shape[1] <= 2:
        coords = embeddings[:, :2] if embeddings.ndim == 2 else np.empty((0, 2))
        return coords, {"kind": "raw"}

    try:  # pragma: no cover - optional dependency
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=42, metric="cosine")
        coords = reducer.fit_transform(embeddings)
        return coords, {"kind": "umap"}
    except Exception:
        pass

    reducer = PCA(n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)
    return coords, {
        "kind": "pca",
        "components": getattr(reducer, "components_", None),
        "mean": getattr(reducer, "mean_", None),
    }


def _compute_decision_boundary_overlay(
    coords: np.ndarray,
    logits: np.ndarray,
    model: Any,
    projection_meta: Dict[str, Any],
    *,
    probability_center: float = 0.5,
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

    center_prob = float(probability_center)
    center_prob = max(1e-4, min(1.0 - 1e-4, center_prob))

    max_margin = min(center_prob - 1e-4, 1.0 - center_prob - 1e-4)
    if max_margin < 0:
        max_margin = 0.0
    requested_margin = max(0.0, float(probability_margin))
    prob_margin = min(max_margin, requested_margin)

    low_prob = max(1e-4, center_prob - prob_margin)
    high_prob = min(1.0 - 1e-4, center_prob + prob_margin)

    center_logit = math.log(center_prob / (1.0 - center_prob))
    low_logit = math.log(low_prob / (1.0 - low_prob))
    high_logit = math.log(high_prob / (1.0 - high_prob))

    margin_logit_upper = max(0.0, high_logit - center_logit)
    margin_logit_lower = max(0.0, center_logit - low_logit)
    margin_logit = max(margin_logit_upper, margin_logit_lower)
    margin_distance_upper = margin_logit_upper / norm if margin_logit_upper > 0 else 0.0
    margin_distance_lower = margin_logit_lower / norm if margin_logit_lower > 0 else 0.0
    margin_distance = max(margin_distance_upper, margin_distance_lower)

    intercept_centered = float(intercept_2d - center_logit)

    line_points = _line_box_intersections(bounds, weight_2d, intercept_centered)

    corner_polygon = [
        (bounds[0], bounds[2]),
        (bounds[1], bounds[2]),
        (bounds[1], bounds[3]),
        (bounds[0], bounds[3]),
    ]

    spam_polygon = _clip_polygon_with_halfplane(
        corner_polygon, weight_2d, intercept_2d, center_logit, keep_leq=False
    )
    safe_polygon = _clip_polygon_with_halfplane(
        corner_polygon, weight_2d, intercept_2d, center_logit, keep_leq=True
    )

    band_polygon = corner_polygon
    if high_logit > low_logit:
        band_polygon = _clip_polygon_with_halfplane(
            band_polygon, weight_2d, intercept_2d, high_logit, keep_leq=True
        )
        band_polygon = _clip_polygon_with_halfplane(
            band_polygon, weight_2d, intercept_2d, low_logit, keep_leq=False
        )
    else:
        band_polygon = []

    line_df = pd.DataFrame(line_points, columns=["x", "y"])

    shading_rows: List[Dict[str, Any]] = []
    for polygon, label in ((spam_polygon, "spam"), (safe_polygon, "safe")):
        if not polygon:
            continue
        for x, y in polygon:
            shading_rows.append({"x": x, "y": y, "label": label})

    shading_df = pd.DataFrame(shading_rows) if shading_rows else None

    band_df = pd.DataFrame(band_polygon, columns=["x", "y"]) if band_polygon else None

    return {
        "line_df": line_df,
        "band_df": band_df,
        "shading_df": shading_df,
        "norm": norm,
        "center_logit": center_logit,
        "low_logit": low_logit,
        "high_logit": high_logit,
        "center_probability": center_prob,
        "margin_probability": prob_margin,
        "margin_probability_low": low_prob,
        "margin_probability_high": high_prob,
        "margin_distance": margin_distance,
    }

def _conceptual_meaning_sketch():
    domain = (-3, 3)
    cards = pd.DataFrame(
        [
            {
                "label": "Promotions",
                "details": "discounts, offers",
                "tone": "Spam-like",
                "x1": -2.8,
                "x2": -1.4,
                "y1": -1.6,
                "y2": -0.4,
            },
            {
                "label": "Security alerts",
                "details": "password resets",
                "tone": "Spam-like",
                "x1": -1.4,
                "x2": 0.0,
                "y1": -1.2,
                "y2": 0.2,
            },
            {
                "label": "Courier tracking",
                "details": "delivery notices",
                "tone": "Borderline",
                "x1": -0.4,
                "x2": 1.0,
                "y1": -0.2,
                "y2": 1.2,
            },
            {
                "label": "Project updates",
                "details": "status summaries",
                "tone": "Safe-like",
                "x1": 1.2,
                "x2": 2.6,
                "y1": 0.6,
                "y2": 1.8,
            },
            {
                "label": "Meeting emails",
                "details": "agenda, follow-ups",
                "tone": "Safe-like",
                "x1": 1.8,
                "x2": 3.0,
                "y1": 1.6,
                "y2": 2.8,
            },
        ]
    )
    cards["xc"] = (cards["x1"] + cards["x2"]) / 2
    cards["yc"] = (cards["y1"] + cards["y2"]) / 2

    base = (
        alt.Chart(pd.DataFrame({"x": domain, "y": domain}))
        .mark_point(opacity=0)
        .encode(
            x=alt.X(
                "x:Q",
                scale=alt.Scale(domain=domain, nice=False),
                title="Meaning dimension 1",
                axis=alt.Axis(labelFontSize=12, titleFontSize=13, labelColor="#1f2937"),
            ),
            y=alt.Y(
                "y:Q",
                scale=alt.Scale(domain=domain, nice=False),
                title="Meaning dimension 2",
                axis=alt.Axis(labelFontSize=12, titleFontSize=13, labelColor="#1f2937"),
            ),
        )
        .properties(height=360)
    )

    background = alt.Chart(
        pd.DataFrame(
            [
                {"x1": domain[0], "x2": 0, "y1": domain[0], "y2": 0, "shade": "spam"},
                {"x1": 0, "x2": domain[1], "y1": 0, "y2": domain[1], "shade": "safe"},
                {"x1": domain[0], "x2": 0, "y1": 0, "y2": domain[1], "shade": "mixed"},
                {"x1": 0, "x2": domain[1], "y1": domain[0], "y2": 0, "shade": "mixed"},
            ]
        )
    ).mark_rect(opacity=0.22).encode(
        x=alt.X("x1:Q", scale=alt.Scale(domain=domain)),
        x2="x2:Q",
        y=alt.Y("y1:Q", scale=alt.Scale(domain=domain)),
        y2="y2:Q",
        color=alt.Color(
            "shade:N",
            scale=alt.Scale(
                domain=["safe", "mixed", "spam"],
                range=["#e0f2fe", "#fef3c7", "#fee2e2"],
            ),
            legend=None,
        ),
    )

    guard_band = alt.Chart(
        pd.DataFrame(
            [
                {
                    "x1": -0.5,
                    "x2": 0.5,
                    "y1": domain[0],
                    "y2": domain[1],
                }
            ]
        )
    ).mark_rect(
        color="#fde68a",
        opacity=0.18,
    ).encode(
        x=alt.X("x1:Q", scale=alt.Scale(domain=domain)),
        x2="x2:Q",
        y=alt.Y("y1:Q", scale=alt.Scale(domain=domain)),
        y2="y2:Q",
    )

    tone_scale = alt.Scale(
        domain=["Safe-like", "Borderline", "Spam-like"],
        range=["#bfdbfe", "#fde68a", "#fecaca"],
    )

    card_rects = (
        alt.Chart(cards)
        .mark_rect(
            stroke="#1e293b",
            strokeWidth=1.5,
            cornerRadius=26,
            opacity=0.92,
        )
        .encode(
            x=alt.X("x1:Q", scale=alt.Scale(domain=domain)),
            x2="x2:Q",
            y=alt.Y("y1:Q", scale=alt.Scale(domain=domain)),
            y2="y2:Q",
            color=alt.Color("tone:N", title="", scale=tone_scale),
            tooltip=[
                alt.Tooltip("label:N", title="Cluster"),
                alt.Tooltip("details:N", title="Typical content"),
                alt.Tooltip("tone:N", title="Tone"),
            ],
        )
    )

    card_titles = (
        alt.Chart(cards)
        .mark_text(fontSize=12, fontWeight=700, color="#0f172a", dy=-6)
        .encode(x="xc:Q", y="yc:Q", text="label:N")
    )

    card_details = (
        alt.Chart(cards)
        .mark_text(fontSize=11, color="#1f2937", dy=12)
        .encode(x="xc:Q", y="yc:Q", text="details:N")
    )

    crosshair = (
        alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
        .mark_rule(color="#64748b", strokeDash=[6, 4], strokeWidth=1.5)
        .encode(x="x:Q")
        + alt.Chart(pd.DataFrame({"x": [0], "y": [0]}))
        .mark_rule(color="#64748b", strokeDash=[6, 4], strokeWidth=1.5)
        .encode(y="y:Q")
    )

    frame = alt.Chart(
        pd.DataFrame({"x1": [domain[0]], "x2": [domain[1]], "y1": [domain[0]], "y2": [domain[1]]})
    ).mark_rect(fillOpacity=0, stroke="#1d4ed8", strokeWidth=1.5, cornerRadius=22)

    chart = (
        (background + guard_band + base + card_rects + card_titles + card_details + crosshair + frame)
        .properties(title="Conceptual meaning map (before training)")
        .configure_axis(labelColor="#334155", titleColor="#0f172a", gridColor="#e2e8f0", gridDash=[2, 3])
        .configure_legend(labelColor="#334155", title=None)
        .configure_title(color="#0f172a", fontSize=16, anchor="start", subtitleColor="#475569")
        .configure_view(strokeOpacity=0)
    )
    return chart


def _ghost_meaning_map(height: int = 220):
    base = (
        alt.Chart(pd.DataFrame({"x": [-1, 1], "y": [-1, 1]}))
        .mark_point(opacity=0)
        .encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[-1, 1]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[-1, 1]), axis=None),
        )
        .properties(height=height)
    )
    return base


def _guardrail_window_values(ss) -> Tuple[float, float, float, float]:
    guard_params = ss.get("guard_params", {}) or {}
    threshold_default = float(ss.get("threshold", 0.6))
    try:
        center = float(guard_params.get("assist_center", threshold_default))
    except (TypeError, ValueError):
        center = threshold_default
    try:
        band = float(guard_params.get("uncertainty_band", 0.08))
    except (TypeError, ValueError):
        band = 0.08
    center = max(0.0, min(1.0, center))
    band = max(0.0, band)
    low = max(0.0, min(1.0, center - band))
    high = max(0.0, min(1.0, center + band))
    return center, band, low, high


def _ghost_meaning_map_enhanced(
    ss,
    *,
    height: int = 220,
    title: str = "",
    show_divider: bool = True,
    show_band: bool = True,
) -> "alt.Chart":
    base = alt.Chart(pd.DataFrame({"x": [-1, 1], "y": [-1, 1]})).mark_point(opacity=0).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-1, 1]), title="meaning dimension 1"),
        y=alt.Y("y:Q", scale=alt.Scale(domain=[-1, 1]), title="meaning dimension 2"),
    ).properties(height=height, title=title or None)

    c, b, low, high = _guardrail_window_values(ss)
    tau = float(ss.get("threshold", c))
    x_center = 2.0 * (c - 0.5)
    x_tau = 2.0 * (tau - 0.5)
    band_left = 2.0 * (max(0.0, c - b) - 0.5)
    band_right = 2.0 * (min(1.0, c + b) - 0.5)

    layers = [base]

    if show_band and band_right > band_left:
        band_df = pd.DataFrame(
            {
                "x1": [band_left],
                "x2": [band_right],
                "y1": [-1.0],
                "y2": [1.0],
            }
        )
        rect = alt.Chart(band_df).mark_rect(opacity=0.18).encode(
            x=alt.X("x1:Q"),
            x2="x2:Q",
            y=alt.Y("y1:Q"),
            y2="y2:Q",
            tooltip=[alt.Tooltip("x1:Q", title="band left"), alt.Tooltip("x2:Q", title="band right")],
        )
        layers.append(rect)

    if show_divider:
        line_df = pd.DataFrame({"x": [x_tau, x_tau], "y": [-1.0, 1.0]})
        divider = alt.Chart(line_df).mark_rule(strokeDash=[5, 5], strokeOpacity=0.9).encode(
            x="x:Q",
            y="y:Q",
        )
        layers.append(divider)

    center_df = pd.DataFrame({"x": [x_center], "y": [0.0], "τ": [c]})
    dot = alt.Chart(center_df).mark_point(size=60, filled=True, opacity=0.9).encode(
        x="x:Q",
        y="y:Q",
        tooltip=[alt.Tooltip("τ:Q", title="guard center τ", format=".2f")],
    )
    layers.append(dot)

    return alt.layer(*layers)


def _numeric_guardrails_caption_text(ss) -> str:
    center, _, low, high = _guardrail_window_values(ss)
    return (
        f"Numeric guardrails watch emails when the text score is near τ≈{center:.2f} "
        f"(window {low:.2f}–{high:.2f})."
    )


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

    return alt.layer(*layers).properties(height=360)


def _render_training_examples_preview() -> None:
    """Placeholder hook for future training examples preview."""

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
    numeric_contribs: List[List[Tuple[str, float, float]]] = []
    for title, body, label in zip(sampled_titles, sampled_bodies, sampled_labels):
        contribs = numeric_feature_contributions(model, title, body) if model else []
        numeric_contribs.append(contribs or [])
        reasons.append(_reason_from_contributions(label, contribs))
    df["reason"] = reasons
    df["_numeric_contribs"] = numeric_contribs

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
    guard_center = 0.5
    guard_band = 0.1
    if model is not None:
        try:
            guard_center = float(getattr(model, "numeric_assist_center", guard_center))
        except Exception:
            guard_center = 0.5
        try:
            guard_band = float(getattr(model, "uncertainty_band", guard_band))
        except Exception:
            guard_band = 0.1
    guard_center = max(1e-4, min(1.0 - 1e-4, guard_center))
    guard_band = max(0.0, guard_band)
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
                probability_center=guard_center,
                probability_margin=guard_band,
            )
            if boundary_info:
                norm_val = float(boundary_info.get("norm", 0.0) or 0.0)
                center_logit = float(boundary_info.get("center_logit", 0.0) or 0.0)
                low_logit_val = float(boundary_info.get("low_logit", center_logit))
                high_logit_val = float(boundary_info.get("high_logit", center_logit))
                if norm_val > 0 and np.all(np.isfinite(logits)):
                    distances = (logits - center_logit) / norm_val
                    df["distance_to_line"] = distances
                    df["distance_abs"] = np.abs(distances)
                else:
                    df["distance_to_line"] = np.nan
                    df["distance_abs"] = np.nan
                if np.all(np.isfinite(logits)):
                    lo = min(low_logit_val, high_logit_val)
                    hi = max(low_logit_val, high_logit_val)
                    df["borderline"] = (logits >= lo) & (logits <= hi)
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

    guard_center_effective = guard_center
    guard_margin_requested = guard_band
    guard_margin_effective = guard_band
    guard_low = max(0.0, guard_center_effective - guard_margin_effective)
    guard_high = min(1.0, guard_center_effective + guard_margin_effective)
    if isinstance(boundary_info, dict):
        guard_center_effective = float(
            boundary_info.get("center_probability", guard_center_effective)
        )
        guard_margin_effective = float(
            boundary_info.get("margin_probability", guard_margin_effective)
        )
        guard_low = float(boundary_info.get("margin_probability_low", guard_low))
        guard_high = float(boundary_info.get("margin_probability_high", guard_high))
    guard_center_effective = max(0.0, min(1.0, guard_center_effective))
    guard_margin_effective = max(0.0, guard_margin_effective)
    guard_low = max(0.0, min(1.0, guard_low))
    guard_high = max(0.0, min(1.0, guard_high))

    if "numeric_clues" not in df.columns:
        if "_numeric_contribs" in df.columns:
            contrib_lists = df["_numeric_contribs"].tolist()
            df.drop(columns=["_numeric_contribs"], inplace=True)
        else:
            contrib_lists = [[] for _ in range(len(df))]
        df["numeric_clues"] = [
            _extract_numeric_clues(contribs) if contribs else [] for contribs in contrib_lists
        ]
    else:
        if "_numeric_contribs" in df.columns:
            df.drop(columns=["_numeric_contribs"], inplace=True)

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
        "guard_center_probability": guard_center_effective,
        "guard_margin_probability": guard_margin_effective,
        "guard_margin_requested": guard_margin_requested,
        "guard_window_low": guard_low,
        "guard_window_high": guard_high,
    }

    return df, meta


def _meaning_map_zoom_subset(
    df: Optional[pd.DataFrame],
    meta: Optional[Dict[str, Any]],
    *,
    max_line_points: int = 24,
    max_center_points_per_class: int = 6,
) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df

    working = df.copy()
    if "label" not in working.columns:
        return working

    subset_indices: set[int] = set()

    if "distance_abs" in working.columns and working["distance_abs"].notna().any():
        labels = sorted({lbl for lbl in working["label"].dropna().unique()})
        if labels:
            per_label = max(1, max_line_points // max(len(labels), 1))
            for label in labels:
                class_df = working.loc[working["label"] == label]
                if class_df.empty:
                    continue
                top = class_df.nsmallest(per_label, "distance_abs")
                subset_indices.update(top.index.tolist())

    centroids = None
    if isinstance(meta, dict):
        centroids = meta.get("centroids")
    if isinstance(centroids, pd.DataFrame) and not centroids.empty:
        if "x" in working.columns and "y" in working.columns and "label" in centroids.columns:
            for _, row in centroids.iterrows():
                label = row.get("label")
                if label not in working["label"].values:
                    continue
                class_df = working.loc[working["label"] == label]
                if class_df.empty:
                    continue
                cx = float(row.get("x", 0.0) or 0.0)
                cy = float(row.get("y", 0.0) or 0.0)
                distances = pd.Series(
                    np.hypot(class_df["x"] - cx, class_df["y"] - cy), index=class_df.index
                )
                per_label = max(1, max_center_points_per_class)
                indices = distances.nsmallest(per_label).index
                top = class_df.loc[indices]
                subset_indices.update(top.index.tolist())

    if not subset_indices:
        return working

    subset = working.loc[sorted(subset_indices)].copy()
    return subset.reset_index(drop=True)

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

    x_series = df["x"].replace([np.inf, -np.inf], np.nan).dropna()
    y_series = df["y"].replace([np.inf, -np.inf], np.nan).dropna()

    def _dimension_scale(series: pd.Series) -> alt.Scale:
        if series.empty:
            return alt.Scale(zero=False)
        min_val = float(series.min())
        max_val = float(series.max())
        span = max_val - min_val
        padding = max(0.35, span * 0.12) if math.isfinite(span) else 0.35
        domain = [min_val - padding, max_val + padding]
        return alt.Scale(domain=domain, nice=False, zero=False)

    x_scale = _dimension_scale(x_series)
    y_scale = _dimension_scale(y_series)

    axis_kwargs = dict(
        labelColor="#475569",
        labelFontSize=11,
        titleColor="#0f172a",
        titleFontSize=12,
        tickCount=5,
        ticks=False,
        domain=False,
        grid=True,
        gridColor="#e2e8f0",
        gridDash=[3, 5],
        gridOpacity=0.55,
    )

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
            axis=alt.Axis(title="Meaning dimension 1", **axis_kwargs),
            scale=x_scale,
        ),
        y=alt.Y(
            "y:Q",
            axis=alt.Axis(title="Meaning dimension 2", **axis_kwargs),
            scale=y_scale,
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

    halo = base.mark_circle(size=420, opacity=0.12).encode(
        x="x:Q",
        y="y:Q",
        color=alt.Color(
            "label:N",
            scale=alt.Scale(domain=["spam", "safe"], range=["#fecaca", "#bfdbfe"]),
            legend=None,
        ),
    )
    layers = [halo, scatter]

    boundary = meta.get("boundary") if isinstance(meta, dict) else None
    if isinstance(boundary, dict):
        shading_df = boundary.get("shading_df")
        if isinstance(shading_df, pd.DataFrame) and not shading_df.empty:
            shading_layer = (
                alt.Chart(shading_df)
                .mark_line(
                    filled=True,
                    opacity=0.22,
                    strokeOpacity=0,
                    fillOpacity=0.22,
                )
                .encode(
                    x="x:Q",
                    y="y:Q",
                    order="order:O",
                    color=alt.Color("label:N", scale=color_scale, legend=None),
                )
            )
            layers.append(shading_layer)

            try:
                label_positions = []
                for label, group in shading_df.groupby("label"):
                    if group.empty:
                        continue
                    label_positions.append(
                        {
                            "label": label,
                            "label_display": "Spam-leaning zone"
                            if str(label).lower() == "spam"
                            else ("Safe-leaning zone" if str(label).lower() == "safe" else str(label)),
                            "x": float(np.nanmean(group["x"])),
                            "y": float(np.nanmean(group["y"])),
                        }
                    )
            except Exception:
                label_positions = []

            if label_positions:
                label_layer = (
                    alt.Chart(pd.DataFrame(label_positions))
                    .mark_text(fontSize=12, fontWeight=600, opacity=0.82, color="#0f172a", dy=-6)
                    .encode(
                        x="x:Q",
                        y="y:Q",
                        text="label_display:N",
                    )
                )
                layers.append(label_layer)

        band_df = boundary.get("band_df")
        if isinstance(band_df, pd.DataFrame) and not band_df.empty:
            band_layer = (
                alt.Chart(band_df)
                .mark_line(
                    filled=True,
                    color="#facc15",
                    opacity=0.24,
                    strokeOpacity=0,
                    fillOpacity=0.24,
                )
                .encode(x="x:Q", y="y:Q", order="order:O", fill=alt.value("#facc15"))
            )
            layers.append(band_layer)

        line_df = boundary.get("line_df")
        if isinstance(line_df, pd.DataFrame) and not line_df.empty:
            line_layer = (
                alt.Chart(line_df)
                .mark_line(color="#1f2937", strokeWidth=2.6, strokeDash=[8, 6])
                .encode(x="x:Q", y="y:Q")
            )
            layers.append(line_layer)

    if show_examples and "body_excerpt" in df.columns:
        text_layer = base.mark_text(
            align="center",
            baseline="middle",
            color="#0f172a",
            fontSize=10,
            fontWeight=600,
        ).encode(
            x="x:Q",
            y="y:Q",
            text="body_excerpt:N",
        )
        layers.append(text_layer)

    if show_class_centers and isinstance(meta.get("centroids"), pd.DataFrame):
        centroids_df = meta["centroids"]
        center_layer = (
            alt.Chart(centroids_df)
            .mark_point(size=260, shape="diamond", filled=True, opacity=0.9)
            .encode(
                x="x:Q",
                y="y:Q",
                color=alt.Color("label:N", scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("label:N", title="Class"),
                    alt.Tooltip("x:Q", title="Meaning 1", format=".2f"),
                    alt.Tooltip("y:Q", title="Meaning 2", format=".2f"),
                ],
            )
        )
        layers.append(center_layer)

    if highlight_borderline and "borderline" in df.columns:
        borderline_df = df.loc[df["borderline"] == True]  # noqa: E712
        if not borderline_df.empty:
            borderline_layer = (
                alt.Chart(borderline_df)
                .mark_point(size=260, stroke="#f8fafc", strokeWidth=1.4)
                .encode(
                    x="x:Q",
                    y="y:Q",
                    color=alt.Color("label:N", scale=color_scale, legend=None),
                )
            )
            layers.append(borderline_layer)

    chart_main = alt.layer(*layers).properties(height=360)

    detail_columns = [
        "subject_tooltip",
        "label_title",
        "predicted_label_title",
        "spam_probability",
        "distance_display",
        "reason",
    ]
    available_detail_cols = [col for col in detail_columns if col in df.columns]
    if available_detail_cols:
        detail_df = df[["plot_index", *available_detail_cols]].copy()
        detail_df.rename(
            columns={
                "subject_tooltip": "Subject",
                "label_title": "True label",
                "predicted_label_title": "Model prediction",
                "spam_probability": "Spam probability",
                "distance_display": "Distance to line",
                "reason": "Signals",
            },
            inplace=True,
        )

        detail_bg = (
            alt.Chart(detail_df)
            .mark_rect(opacity=0.0)
            .encode(
                y=alt.Y("plot_index:O", title=""),
                color=alt.Color("plot_index:N", legend=None),
                opacity=alt.condition(combined_selection, alt.value(0.18), alt.value(0.02)),
            )
        )
        detail_subject = (
            alt.Chart(detail_df)
            .mark_text(align="left", dx=4, fontWeight=600, fontSize=12)
            .encode(y="plot_index:O", text="Subject:N")
        )
        detail_excerpt = (
            alt.Chart(detail_df)
            .mark_text(align="left", dx=4, dy=16, fontSize=11, color="#475569")
            .encode(text="Signals:N")
        )
        detail_reason = (
            alt.Chart(detail_df)
            .mark_text(align="left", dx=4, dy=32, fontSize=11, color="#1f2937")
            .encode(text="Distance to line:N")
        )
        detail_chart = alt.layer(detail_bg, detail_subject, detail_excerpt, detail_reason).properties(
            height=180
        )
    else:
        detail_chart = None

    if detail_chart is None:
        return chart_main

    return alt.vconcat(chart_main, detail_chart).resolve_scale(color="independent")

def _render_unified_training_storyboard(
    ss,
    *,
    has_model: bool,
    has_split: bool,
    has_embed: bool,
    section_surface: Callable[[], Any],
    request_meaning_map_refresh: Callable[[Optional[str]], None],
    parse_split_cache: Callable[[Any], tuple],
    rerun: Callable[[], None],
    logger,
) -> None:
    meaning_map_df, meaning_map_meta, chart_ready, meaning_map_error = None, {}, False, None
    if has_model and has_split and has_embed:
        try:
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = parse_split_cache(ss["split_cache"])
            meaning_map_df, meaning_map_meta = _prepare_meaning_map(
                list(X_tr_t) if X_tr_t else [],
                list(X_tr_b) if X_tr_b else [],
                list(y_tr) if y_tr else [],
                ss.get("model"),
            )
            gp = ss.get("guard_params", {}) or {}
            center_default = float(gp.get("assist_center", float(ss.get("threshold", 0.6))))
            band_default = float(gp.get("uncertainty_band", 0.08))
            meaning_map_meta.setdefault("guard_center_probability", center_default)
            meaning_map_meta.setdefault("guard_margin_probability", band_default)
            c = float(meaning_map_meta["guard_center_probability"])
            b = float(meaning_map_meta["guard_margin_probability"])
            meaning_map_meta.setdefault("guard_window_low", max(0.0, c - b))
            meaning_map_meta.setdefault("guard_window_high", min(1.0, c + b))
            chart_ready = meaning_map_df is not None and not meaning_map_df.empty
        except Exception as exc:
            meaning_map_error = f"Meaning Map unavailable ({exc})."
            meaning_map_df, meaning_map_meta = None, {}
            chart_ready = False

    ss["train_storyboard_payload"] = {
        "meaning_map_df": meaning_map_df if chart_ready else None,
        "meaning_map_meta": meaning_map_meta if meaning_map_meta else {},
        "meaning_map_error": meaning_map_error,
        "chart_ready": chart_ready,
    }

    refresh_expected = bool(ss.get("train_refresh_expected"))
    refresh_attempts = int(ss.get("train_refresh_attempts", 0) or 0)
    if refresh_expected:
        if chart_ready:
            ss["train_refresh_expected"] = False
            ss["train_refresh_attempts"] = 0
        elif refresh_attempts > 0:
            ss["train_refresh_attempts"] = refresh_attempts - 1
            logger.debug(
                "Meaning map still warming after training; scheduling auto-rerun (%s attempts left)",
                refresh_attempts - 1,
            )
            st.caption("Refreshing training meaning maps…")
            time.sleep(0.35)
            rerun()
            return
        else:
            ss["train_refresh_expected"] = False
            ss["train_refresh_attempts"] = 0

    with section_surface():
        st.markdown("### How training works — live storyboard")

        left1, right1 = st.columns([0.48, 0.52], gap="large")
        with left1:
            pre_training = not (has_model and has_split)
            show_examples = False
            if pre_training:
                st.markdown(
                    "**1) Meaning points**  \n"
                    "Think of each email as a dot placed by MiniLM in a 2-D meaning space:\n\n"
                    "- **Meaning dimension 1**: a blend of traits such as *formality ↔ salesy tone*, *generic phrasing ↔ brand-specific details*, or *routine biz-speak ↔ promotional language*.\n"
                    "- **Meaning dimension 2**: a complementary blend such as *calm, factual ↔ urgent, persuasive*, *low CTA ↔ strong CTA*, or *neutral ↔ attention-grabbing*.\n\n"
                    "Below is a **conceptual meaning map** (an illustrative example shown *before* training). "
                    "Instead of dots, it uses chip-style cards to preview likely clusters from this dataset — "
                    "meeting notes, courier tracking, promotional blasts, security alerts, and other themes — each placed where that tone usually lands. "
                    "This helps you imagine the semantic neighborhoods MiniLM will uncover and why borderline content often sits near the middle. "
                    "After you train on your labeled emails, this sketch will be replaced by the **real** map built from your data."
                )
            else:
                st.markdown("**1) Meaning points**")
                st.markdown(
                    "MiniLM places each email so **similar wording lands close**. "
                    "These two axes are the first two directions of that meaning space — "
                    "_meaning dimension 1_ and _meaning dimension 2_."
                )
                if has_model and has_split:
                    if st.button(
                        "Refresh chart",
                        key=f"refresh_meaning_map_section1_{ss.get('train_story_run_id') or 'initial'}",
                    ):
                        request_meaning_map_refresh("section1")

        with right1:
            pre_training = not (has_model and has_split)
            story_run_id = ss.get("train_story_run_id") or "initial"
            if pre_training:
                try:
                    chart = _conceptual_meaning_sketch()
                except Exception:
                    st.info("Illustrative map unavailable (Altair missing)")
                else:
                    st.altair_chart(
                        chart,
                        use_container_width=True,
                        key=f"conceptual_meaning_sketch_{story_run_id}",
                    )
                _render_training_examples_preview()
            elif meaning_map_error:
                st.info(meaning_map_error)
                _render_training_examples_preview()
            elif chart_ready:
                chart1 = _build_meaning_map_chart(
                    meaning_map_df,
                    meaning_map_meta,
                    show_examples=show_examples,
                    show_class_centers=False,
                    highlight_borderline=False,
                )
                if chart1 is not None:
                    st.altair_chart(
                        chart1,
                        use_container_width=True,
                        key=f"meaning_map_chart1_live_{story_run_id}",
                    )
                else:
                    st.info("Meaning Map unavailable for this run.")
                _render_training_examples_preview()
            else:
                st.altair_chart(
                    _ghost_meaning_map_enhanced(
                        ss,
                        title="Meaning map (schematic)",
                        show_divider=False,
                        show_band=False,
                    ),
                    use_container_width=True,
                    key=f"ghost_live_map1_{story_run_id}",
                )
                _render_training_examples_preview()

        left2, right2 = st.columns([0.48, 0.52], gap="large")
        with left2:
            st.markdown("**2) A simple dividing line**")
            st.markdown(
                "A straight line will separate spam from safe. Before training, we show where the **current configuration** "
                "would place that dividing line conceptually. After training, you’ll see the actual learned boundary in a "
                "zoomed-in view that spotlights borderline emails and those closest to each class center."
            )
            show_centers = bool(has_model and has_split)
            if has_model and has_split:
                if st.button(
                    "Refresh chart",
                    key=f"refresh_meaning_map_section2_{story_run_id}",
                ):
                    request_meaning_map_refresh("section2")

        with right2:
            if meaning_map_error:
                st.info(meaning_map_error)
            elif chart_ready:
                zoom_df = _meaning_map_zoom_subset(meaning_map_df, meaning_map_meta)
                chart2 = _build_meaning_map_chart(
                    zoom_df if zoom_df is not None else meaning_map_df,
                    meaning_map_meta,
                    show_examples=False,
                    show_class_centers=show_centers,
                    highlight_borderline=False,
                )
                if chart2 is not None:
                    st.altair_chart(
                        chart2,
                        use_container_width=True,
                        key=f"meaning_map_chart2_live_{story_run_id}",
                    )
                else:
                    st.info("Meaning Map unavailable for this run.")
            else:
                st.altair_chart(
                    _ghost_meaning_map_enhanced(
                        ss,
                        title="Dividing line (schematic)",
                        show_divider=True,
                        show_band=False,
                    ),
                    use_container_width=True,
                    key=f"ghost_live_map2_{story_run_id}",
                )

        guard_params = ss.get("guard_params", {}) if hasattr(ss, "get") else {}
        try:
            assist_center = float(guard_params.get("assist_center", ss.get("threshold", 0.6)))
        except Exception:
            assist_center = float(ss.get("threshold", 0.6))
        try:
            uncertainty_band = float(guard_params.get("uncertainty_band", 0.08))
        except Exception:
            uncertainty_band = 0.08

        left3, right3 = st.columns([0.48, 0.52], gap="large")
        with left3:
            st.markdown("**3) Extra clues when unsure**")
            st.markdown(
                f"When the text score is near **τ≈{assist_center:.2f}**, numeric guardrails help out within a small window "
                f"(**±{uncertainty_band:.2f}**) — we look at links, ALL-CAPS, money or urgency hints."
            )
            guard_low = None
            guard_high = None
            if isinstance(meaning_map_meta, dict):
                guard_low = meaning_map_meta.get("guard_window_low")
                guard_high = meaning_map_meta.get("guard_window_high")
            if chart_ready and isinstance(guard_low, (int, float)) and isinstance(guard_high, (int, float)):
                st.caption(
                    f"Showing emails where the spam probability falls between {guard_low:.2f} and {guard_high:.2f}."
                )
            else:
                st.caption("Train the model to see which emails triggered these numeric clues.")
            if has_model and has_split:
                if st.button(
                    "Refresh chart",
                    key=f"refresh_meaning_map_section3_{story_run_id}",
                ):
                    request_meaning_map_refresh("section3")

        with right3:
            if meaning_map_error:
                st.info(meaning_map_error)
            elif chart_ready:
                _render_numeric_clue_cards(meaning_map_df)
            else:
                _render_numeric_clue_preview(assist_center, uncertainty_band)

        st.caption(_numeric_guardrails_caption_text(ss))
