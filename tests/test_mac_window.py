"""Tests for the mac window component helpers."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from demistifai.ui.components.shared.mac_window import (  # type: ignore  # noqa: E402
    _format_grid_template,
    _format_max_width,
)


def test_format_max_width_uses_px_for_numbers() -> None:
    """Numeric ``max_width`` values are converted into pixel tokens."""

    assert _format_max_width(1280) == "1280px"


def test_format_max_width_accepts_css_tokens() -> None:
    """String ``max_width`` values are returned verbatim after stripping."""

    assert _format_max_width(" 72rem ") == "72rem"


def test_format_grid_template_balances_ratios() -> None:
    """Ratios are normalised into percentage-based grid tracks."""

    columns = 2
    ratios = (1, 2)
    template = _format_grid_template(columns, ratios)
    assert template.split() == ["33.33333%", "66.66667%"]


def test_format_grid_template_validates_lengths() -> None:
    """Mismatched column counts raise a ``ValueError``."""

    with pytest.raises(ValueError):
        _format_grid_template(3, (1, 1))
