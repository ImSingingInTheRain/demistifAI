from __future__ import annotations

from textwrap import dedent
from typing import Iterable, List, Sequence

import streamlit as st
import streamlit.components.v1 as components

from ..components.terminal.terminal_base import render_ai_act_terminal
from ..primitives.typing_quote import (
    get_eu_ai_act_typing_inline_bootstrap,
    get_eu_ai_act_typing_inline_markup,
)


DEFAULT_TERMINAL_LINES: Sequence[str] = (
    "Welcome to demAI — an interactive experience where you will build and operate an AI system, while discovering and applying key concepts from the EU AI Act.\n",
    "",
    "demonstrateAI",
    "Experience how an AI system actually works, step by step — from data preparation to predictions — through an interactive, hands-on journey.\n",
    "",
    "demistifyAI",
    "Break down complex AI concepts into clear, tangible actions so that anyone can understand what’s behind the model’s decisions.\n",
    "",
    "democratizeAI",
    "Empower everyone to engage responsibly with AI, making transparency and trust accessible to all.",
)


def _prepare_lines(lines: Iterable[str]) -> List[str]:
    return ["" if line is None else str(line) for line in lines]


def render_command_grid(lines=None, title: str = ""):
    """Render the welcome command grid with the EU AI Act terminal intro."""

    prepared_lines = _prepare_lines(lines) if lines is not None else list(DEFAULT_TERMINAL_LINES)

    suffix = "welcome_cmd"
    quote_id_prefix = "eu-typing-welcome"
    quote_markup = get_eu_ai_act_typing_inline_markup(id_prefix=quote_id_prefix)
    quote_bootstrap = get_eu_ai_act_typing_inline_bootstrap(id_prefix=quote_id_prefix)

    st.markdown(
        dedent(
            f"""
            <style>
              .placeholder-{suffix} {{
                background: linear-gradient(155deg, rgba(248,250,252,.95), rgba(226,232,240,.6));
                border-radius: 12px;
                box-shadow: 0 12px 28px rgba(15,23,42,.08), inset 0 0 0 1px rgba(148,163,184,.25);
                padding: 1rem 1.1rem;
                display: flex;
                flex-direction: column;
                align-items: stretch;
                justify-content: center;
                min-height: 260px;
                height: 100%;
                color: #334155;
                text-align: left;
              }}
              .placeholder-{suffix} .eu-typing {{
                width: 100%;
              }}
            </style>
            """
        ),
        unsafe_allow_html=True,
    )

    with st.container():
        cols = st.columns([2, 1], gap="large")
        with cols[0]:
            render_ai_act_terminal(demai_lines=prepared_lines)
        with cols[1]:
            st.markdown(
                f"""
                <div class="placeholder-{suffix}">
                  {quote_markup}
                </div>
                """,
                unsafe_allow_html=True,
            )
            components.html(quote_bootstrap, height=0)

