from __future__ import annotations

import re
from typing import Iterable

import streamlit as st

from demistifai.constants import SUSPICIOUS_TLD_SUFFIXES

__all__ = [
    "streamlit_rerun",
    "_count_suspicious_links",
    "_has_suspicious_tld",
    "_caps_ratio",
    "_count_money_mentions",
]


def streamlit_rerun() -> None:
    """Compatibility wrapper for st.rerun / st.experimental_rerun."""
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn is None:
        raise AttributeError("Streamlit rerun function unavailable")
    rerun_fn()


def _count_suspicious_links(text: str, suspicious_suffixes: Iterable[str] | None = None) -> int:
    """Count links with suspicious TLDs in ``text``."""

    suffixes = tuple(suspicious_suffixes) if suspicious_suffixes is not None else SUSPICIOUS_TLD_SUFFIXES
    urls = re.findall(r"https?://[^\s]+", text, re.IGNORECASE)
    return sum(1 for url in urls if any(tld in url for tld in suffixes))


def _count_money_mentions(text: str) -> int:
    """Count occurrences of money and wire-transfer cues in ``text``."""

    patterns = [r"€\s?\d+", r"\$\s?\d+", r"£\s?\d+", r"wire", r"bank", r"transfer", r"invoice"]
    text_lower = text.lower()
    return sum(len(re.findall(pattern, text_lower)) for pattern in patterns)


def _caps_ratio(text: str) -> float:
    """Return the ratio of ALL CAPS tokens to total tokens in ``text``."""

    words = text.split()
    if not words:
        return 0.0
    total = len(words)
    caps = sum(1 for word in words if len(word) > 2 and word.isupper())
    return caps / total


def _has_suspicious_tld(text: str, suspicious_suffixes: Iterable[str] | None = None) -> bool:
    """Return True if ``text`` contains a suspicious TLD."""

    suffixes = tuple(suspicious_suffixes) if suspicious_suffixes is not None else SUSPICIOUS_TLD_SUFFIXES
    return any(tld in text for tld in suffixes)
