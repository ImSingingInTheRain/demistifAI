"""Deprecated stage navigation module maintained for backward compatibility."""

from __future__ import annotations

import warnings

from demistifai.ui.components.shared.stage_navigation import (
    StageBlockRenderer,
    StageTopCardContent,
    _build_default_stage_top_card_content,
    _render_stage_navigation_controls,
    _render_stage_top_card,
    _stage_navigation_context,
    render_stage_top_grid,
)

warnings.warn(
    "`demistifai.core.nav` is deprecated; import from "
    "`demistifai.ui.components.shared.stage_navigation` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "StageBlockRenderer",
    "StageTopCardContent",
    "render_stage_top_grid",
    "_build_default_stage_top_card_content",
    "_render_stage_navigation_controls",
    "_render_stage_top_card",
    "_stage_navigation_context",
]

