"""Chart builders for the Train stage."""
from __future__ import annotations

import altair as alt
import pandas as pd


def build_calibration_chart(reliability_df: pd.DataFrame) -> alt.Chart | None:
    """Return an Altair chart comparing predicted vs. observed spam rates."""

    if reliability_df is None or reliability_df.empty:
        return None

    base_chart = (
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

    return base_chart + diagonal


__all__ = ["build_calibration_chart"]
