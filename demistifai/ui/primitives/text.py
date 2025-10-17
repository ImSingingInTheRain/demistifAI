"""Text-related UI helpers."""

from __future__ import annotations


def shorten_text(text: str, limit: int = 120) -> str:
    """Return *text* truncated to *limit* characters, preserving readability."""

    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"
