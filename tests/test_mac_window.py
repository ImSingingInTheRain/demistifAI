"""Tests for the mac window component helpers."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from demistifai.ui.components.shared.mac_window import mac_window_html


def test_mac_window_uses_default_max_width() -> None:
    """Omitting ``max_width`` keeps the legacy 1020px constraint."""

    html = mac_window_html()

    assert "width: min(100%, 1020px);" in html


def test_mac_window_accepts_custom_max_width() -> None:
    """A numeric ``max_width`` is converted to a pixel-based constraint."""

    html = mac_window_html(max_width=1280)

    assert "width: min(100%, 1280px);" in html


def test_mac_window_accepts_css_max_width_tokens() -> None:
    """String ``max_width`` values flow straight into the style rule."""

    html = mac_window_html(max_width="72rem")

    assert "width: min(100%, 72rem);" in html
