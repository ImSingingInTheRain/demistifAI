from __future__ import annotations

from typing import Tuple

import altair as alt
import pandas as pd


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
            x2=alt.X2("x2"),
            y=alt.Y("y1:Q"),
            y2=alt.Y2("y2"),
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
