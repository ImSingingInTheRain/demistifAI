"""Minimal custom header: hides Streamlit header and shows the animated demAI logo at top-left."""

from __future__ import annotations

from html import escape
from textwrap import dedent

import streamlit as st

from .animated_logo import demai_logo_html


def mount_demai_header(logo_height: int = 56) -> None:
    """
    Hide Streamlit's default header and mount a fixed, simple top bar
    with the animated demAI logo on the left.

    Args:
        logo_height: Pixel height of the logo area inside the header.
    """

    # 1) CSS: hide Streamlit header + add top padding so content doesn't sit under our bar
    st.markdown(
        dedent(
            f"""
            <style>
              /* Hide default Streamlit header */
              header, [data-testid="stHeader"] {{
                display: none !important;
                visibility: hidden !important;
              }}

              /* Ensure main content has room under our fixed header */
              [data-testid="stAppViewContainer"] .main .block-container {{
                padding-top: {logo_height + 18}px !important; /* header height + small gap */
              }}

              /* Simple fixed header bar */
              .demai-header {{
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1000;

                display: flex;
                align-items: center;
                gap: 12px;

                height: {logo_height + 12}px;
                padding: 6px 14px;
                box-sizing: border-box;

                background: rgba(15, 23, 42, 0.92);
                backdrop-filter: blur(8px);
                border-bottom: 1px solid rgba(148, 163, 184, 0.28);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
              }}

              .demai-header__logo-frame {{
                border: none;
                background: transparent;
                width: auto;
                height: {logo_height}px;
                pointer-events: none; /* prevent focusing the iframe */
              }}
            </style>
            """
        ),
        unsafe_allow_html=True,
    )

    # 2) HTML: fixed header with the animated logo embedded via iframe srcdoc
    # We escape the HTML so it can't "break out" and show raw CSS in your page.
    raw_logo_html = demai_logo_html(frame_marker="demai-header")
    safe_srcdoc = escape(raw_logo_html, quote=True)

    st.markdown(
        f"""
        <div class="demai-header" data-testid="demai-header">
          <iframe
            class="demai-header__logo-frame"
            title="demAI animated logo"
            srcdoc="{safe_srcdoc}"
            scrolling="no"
            frameborder="0"
          ></iframe>
        </div>
        """,
        unsafe_allow_html=True,
    )
