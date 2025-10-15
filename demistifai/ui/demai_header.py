"""Custom demAI header helpers."""

from __future__ import annotations

from functools import lru_cache
from textwrap import dedent

import streamlit as st
from streamlit.components.v1 import html as components_html

from .animated_logo import render_demai_logo

_FRAME_MARKER = "custom-header"
_STATE_KEY = "__demaiCustomHeaderState"

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

    div[data-demai-logo-wrapper="true"] {
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
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
        padding-right: 100px
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

_CUSTOM_HEADER_HTML = dedent(
    """
    <div class="demai-custom-header" data-testid="demai-custom-header">
      <div class="demai-custom-header__logo-slot" id="demai-custom-header-logo"></div>
      <div class="demai-custom-header__meta">
        <span class="demai-custom-header__meta-title">demistifAI Lab</span>
        <span class="demai-custom-header__meta-caption">Walking the EU AI Act journey together</span>
      </div>
    </div>
    """
)

_RELOCATION_SCRIPT = dedent(
    f"""
    <script>
    (function () {{
      const FRAME_SELECTOR = 'iframe[data-demai-logo-marker="{_FRAME_MARKER}"]';
      const SLOT_ID = 'demai-custom-header-logo';
      const WRAPPER_ATTR = 'data-demai-logo-wrapper';
      const rootWindow = window.parent || window;
      const rootDocument = rootWindow.document;
      const raf = rootWindow.requestAnimationFrame.bind(rootWindow);
      const state = rootWindow['{_STATE_KEY}'] || (rootWindow['{_STATE_KEY}'] = {{}});

      function relocate() {{
        const frame = rootDocument.querySelector(FRAME_SELECTOR);
        const slot = rootDocument.getElementById(SLOT_ID);

        if (!frame || !slot) {{
          raf(relocate);
          return;
        }}

        frame.classList.add('demai-custom-header__logo-frame');

        const wrapper = frame.closest('div[data-testid="stVerticalBlock"]');
        if (wrapper && wrapper.getAttribute(WRAPPER_ATTR) !== 'true') {{
          wrapper.setAttribute(WRAPPER_ATTR, 'true');
        }}

        if (frame.parentElement !== slot) {{
          slot.replaceChildren(frame);
        }}
      }}

      if (state.relocatorObserver) {{
        state.relocatorObserver.disconnect();
      }}

      function ensureBody() {{
        if (!rootDocument.body) {{
          raf(ensureBody);
          return;
        }}

        relocate();

        const observer = new MutationObserver(relocate);
        observer.observe(rootDocument.body, {{ childList: true, subtree: true }});
        state.relocatorObserver = observer;
      }}

      ensureBody();
    }})();
    </script>
    """
)


@lru_cache(maxsize=1)
def _header_css() -> str:
    """Return the CSS required to style the custom header."""

    return _HEADER_CSS


@lru_cache(maxsize=1)
def _header_html() -> str:
    """Return the HTML wrapper that hosts the custom header."""

    return _CUSTOM_HEADER_HTML


@lru_cache(maxsize=1)
def _relocation_script() -> str:
    """Return the JavaScript that relocates the logo iframe into the header."""

    return _RELOCATION_SCRIPT


def mount_demai_header_logo(*, logo_height: int = 96) -> None:
    """Render the animated demAI logo inside the sticky header."""

    render_demai_logo(height=logo_height, frame_marker=_FRAME_MARKER)
    st.markdown(f"<style>{_header_css()}</style>", unsafe_allow_html=True)
    st.markdown(_header_html(), unsafe_allow_html=True)
    components_html(_relocation_script(), height=0, width=0)
