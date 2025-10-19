"""Shared UI components used across multiple stages."""

from .mac_window import render_mac_window
from .stage_navigation import (
    StageBlockRenderer,
    StageTopCardContent,
    render_stage_top_grid,
)

__all__ = [
    "render_mac_window",
    "StageBlockRenderer",
    "StageTopCardContent",
    "render_stage_top_grid",
]
