"""Shared UI components used across multiple stages."""

from .macos_iframe_window import (
    MacWindowConfig,
    MacWindowPane,
    build_srcdoc,
    render_macos_iframe_window,
)
from .stage_navigation import (
    StageBlockRenderer,
    StageTopCardContent,
    render_stage_top_grid,
)

__all__ = [
    "MacWindowConfig",
    "MacWindowPane",
    "build_srcdoc",
    "render_macos_iframe_window",
    "StageBlockRenderer",
    "StageTopCardContent",
    "render_stage_top_grid",
]
