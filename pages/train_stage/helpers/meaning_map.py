from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from demistifai.constants import CLASSES
from demistifai.modeling import (
    cache_train_embeddings,
    combine_text,
    embedding_backend_info,
    numeric_feature_contributions,
)

from .numeric_clues import _extract_numeric_clues, _reason_from_contributions
from .sampling import _sample_indices_by_label


CLASS_DOMAIN = tuple(CLASSES)
CLASS_SET = set(CLASSES)


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
                "y1": 1.2,
                "y2": 2.4,
            },
        ]
    )

    base = alt.Chart(pd.DataFrame({"x": [domain[0], domain[1]], "y": [domain[0], domain[1]]})).mark_point(
        opacity=0
    )

    background = (
        alt.Chart(cards)
        .mark_rect(cornerRadius=18, strokeOpacity=0.0)
        .encode(
            x=alt.X("x1:Q", title="meaning dimension 1", scale=alt.Scale(domain=domain)),
            x2="x2:Q",
            y=alt.Y("y1:Q", title="meaning dimension 2", scale=alt.Scale(domain=domain)),
            y2="y2:Q",
            color=alt.Color("tone:N", scale=alt.Scale(range=["#f97316", "#facc15", "#38bdf8"])),
        )
    )

    card_rects = (
        alt.Chart(cards)
        .mark_rect(cornerRadius=18, fillOpacity=0.08, stroke="#0f172a", strokeOpacity=0.12, strokeWidth=2)
        .encode(x="x1:Q", x2="x2:Q", y="y1:Q", y2="y2:Q")
    )

    card_titles = (
        alt.Chart(cards)
        .mark_text(fontSize=14, fontWeight=700, color="#0f172a", dy=-8)
        .encode(x="x1:Q", y="y2:Q", text="label:N")
    )

    card_details = (
        alt.Chart(cards)
        .mark_text(fontSize=12, color="#1f2937", dy=10)
        .encode(x="x1:Q", y="y2:Q", text="details:N")
    )

    guard_band = (
        alt.Chart(pd.DataFrame({"x": [0.0], "y": [0.0], "width": [1.2], "height": [6.0]}))
        .mark_rect(fill="#facc15", opacity=0.18, strokeOpacity=0.0)
        .encode(
            x=alt.X("x:Q", title="meaning dimension 1"),
            y=alt.Y("y:Q", title="meaning dimension 2"),
            x2=alt.X("width:Q"),
            y2=alt.Y("height:Q"),
        )
    )

    base_crosshair = pd.DataFrame({"x": [0.0, 0.0], "y": [domain[0], domain[1]]})
    crosshair = (
        alt.Chart(base_crosshair)
        .mark_rule(color="#0f172a", strokeDash=[4, 4], strokeWidth=2)
        .encode(x="x:Q", y="y:Q")
    ) + (
        alt.Chart(pd.DataFrame({"x": [domain[0], domain[1]], "y": [0.0, 0.0]}))
        .mark_rule(color="#0f172a", strokeDash=[4, 4], strokeWidth=2)
        .encode(x="x:Q", y="y:Q")
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

    color_scale = alt.Scale(domain=CLASS_DOMAIN, range=["#ef4444", "#3b82f6"])
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

    borderline_points = (
        alt.Chart(borderline_df)
        .mark_circle(size=200, opacity=0.82, stroke="#f8fafc", strokeWidth=1.4)
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
        if label not in CLASS_SET:
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

    color_scale = alt.Scale(domain=CLASS_DOMAIN, range=["#ef4444", "#3b82f6"])
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
    if "reason" in df.columns:
        tooltip_fields.append(alt.Tooltip("reason:N", title="Signals"))

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
            scale=alt.Scale(domain=CLASS_DOMAIN, range=["#fecaca", "#bfdbfe"]),
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
    detail_selection = select if select is not None else combined_selection

    if available_detail_cols and detail_selection is not None:
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
                opacity=alt.condition(detail_selection, alt.value(0.18), alt.value(0.02)),
            )
            .transform_filter(detail_selection)
        )
        detail_subject = (
            alt.Chart(detail_df)
            .mark_text(align="left", dx=4, fontWeight=600, fontSize=12)
            .encode(y="plot_index:O", text="Subject:N")
            .transform_filter(detail_selection)
        )
        detail_excerpt = (
            alt.Chart(detail_df)
            .mark_text(align="left", dx=4, dy=16, fontSize=11, color="#475569")
            .encode(text="Signals:N")
            .transform_filter(detail_selection)
        )
        detail_reason = (
            alt.Chart(detail_df)
            .mark_text(align="left", dx=4, dy=32, fontSize=11, color="#1f2937")
            .encode(text="Distance to line:N")
            .transform_filter(detail_selection)
        )
        detail_chart = alt.layer(
            detail_bg, detail_subject, detail_excerpt, detail_reason
        ).properties(height=180)
        detail_chart = _chart_add_params(detail_chart, detail_selection)
    else:
        detail_chart = None

    if detail_chart is None:
        return chart_main

    return alt.vconcat(chart_main, detail_chart).resolve_scale(color="independent")
