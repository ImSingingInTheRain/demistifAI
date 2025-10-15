"""Custom demAI header helpers."""

from __future__ import annotations

from functools import lru_cache
from textwrap import dedent

import streamlit as st

from .animated_logo import demai_logo_srcdoc

_FRAME_MARKER = "custom-header"

_HEADER_CSS = dedent(
    """
    header, [data-testid="stHeader"] {
        visibility: hidden;
    }

    [data-testid="stHeader"] {
        display: none;
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0 !important;
    }

    [data-testid="stMainBlock"] {
        padding-top: 0 !important;
    }

    [data-testid="stMainBlock"] > .block-container {
        padding-top: 0 !important;
    }

    .demai-custom-header {
        position: fixed;
        top: 0;
        z-index: 1000;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: clamp(0.5rem, 2vw, 1.5rem);
        padding: clamp(0.6rem, 1.8vw, 1rem) clamp(1rem, 3.2vw, 1.75rem);
        margin-bottom: clamp(1rem, 2.8vw, 1.8rem);
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.92), rgba(15, 118, 110, 0.92));
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
    }

    .demai-custom-header__logo-slot {
        display: flex;
        align-items: center;
        min-height: clamp(3.2rem, 5vw, 4.6rem);
    }

    .demai-custom-header__logo-frame {
        width: clamp(9.8rem, 16vw, 14.5rem) !important;
        height: clamp(3.2rem, 5vw, 4.6rem) !important;
        border: none !important;
        background: transparent !important;
        pointer-events: none;
    }

    .demai-custom-header__meta {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        color: rgba(241, 245, 249, 0.92);
        text-align: right;
    }

    .demai-custom-header__meta-title {
        font-size: clamp(0.92rem, 2.2vw, 1.05rem);
        font-weight: 600;
        letter-spacing: 0.4px;
    }

    .demai-custom-header__meta-caption {
        font-size: clamp(0.75rem, 1.6vw, 0.88rem);
        opacity: 0.85;
    }

    @media (max-width: 768px) {
        .demai-custom-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.85rem;
            padding: 0.85rem clamp(1rem, 6vw, 1.4rem);
        }

        .demai-custom-header__logo-slot,
        .demai-custom-header__meta {
            width: 100%;
        }

        .demai-custom-header__logo-slot {
            justify-content: center;
        }

        .demai-custom-header__meta {
            text-align: center;
        }
    }
    """
)

def _custom_header_html(*, logo_height: int) -> str:
    """Return the HTML wrapper that hosts the custom header."""

    logo_srcdoc = demai_logo_srcdoc(frame_marker=_FRAME_MARKER)
    return dedent(
        f"""
        <div class="demai-custom-header" data-testid="demai-custom-header">
          <div class="demai-custom-header__logo-slot">
            <iframe
              class="demai-custom-header__logo-frame"
              title="demAI animated logo"
              srcdoc="{logo_srcdoc}"
              data-demai-logo-marker="{_FRAME_MARKER}"
              scrolling="no"
              frameborder="0"
              height="{logo_height}"
              width="100%"
            ></iframe>
          </div>
          <div class="demai-custom-header__meta">
            <span class="demai-custom-header__meta-title">demistifAI Lab</span>
            <span class="demai-custom-header__meta-caption">Walking the EU AI Act journey together</span>
          </div>
        </div>
        """
    )


@lru_cache(maxsize=1)
def _header_css() -> str:
    """Return the CSS required to style the custom header."""

    return _HEADER_CSS


@lru_cache(maxsize=None)
def _header_html(logo_height: int) -> str:
    """Return the HTML wrapper that hosts the custom header."""

    return _custom_header_html(logo_height=logo_height)


def mount_demai_header_logo(*, logo_height: int = 96) -> None:
    """Render the animated demAI logo inside the sticky header."""

    st.markdown(f"<style>{_header_css()}</style>", unsafe_allow_html=True)
    st.markdown(_header_html(logo_height), unsafe_allow_html=True)
