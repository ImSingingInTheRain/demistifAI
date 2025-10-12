import html
import re
import time
from textwrap import dedent
from typing import Iterable, List, Sequence

import streamlit as st

import streamlit.components.v1 as components

from components.ui_typing_quote import (
    get_eu_ai_act_typing_inline_bootstrap,
    get_eu_ai_act_typing_inline_markup,
)


def _prepare_lines(lines: Iterable[str]) -> List[str]:
    return ["" if line is None else str(line) for line in lines]


def _line_css_class(index: int, text: str, suffix: str) -> str:
    if index == 0:
        return f"cmdline-{suffix}"
    trimmed = text.strip()
    if trimmed and re.match(r"^dem[a-z]*ai$", trimmed, re.IGNORECASE):
        return f"hl-{suffix}"
    return ""


def _build_grid_html(
    *,
    suffix: str,
    typed_lines: Sequence[str],
    css_classes: Sequence[str],
    show_caret: bool,
    quote_markup: str,
) -> str:
    segments = []
    for typed, css_class in zip(typed_lines, css_classes):
        safe_text = html.escape(typed)
        if css_class:
            safe_text = f'<span class="{css_class}">{safe_text}</span>'
        segments.append(safe_text)

    body_html = "".join(segments)
    caret_html = f'<span class="caret-{suffix}"></span>' if show_caret else ""

    return dedent(
        f"""
        <div class="cmdgrid-{suffix}">
          <div class="terminal-{suffix}">
            <pre class="term-body-{suffix}">{body_html}{caret_html}</pre>
          </div>
          <div class="placeholder-{suffix}">
            {quote_markup}
          </div>
        </div>
        """
    ).strip()


def render_command_grid(lines=None, title: str = ""):
    """Render the intro command grid directly in Streamlit without an iframe."""

    if lines is None:
        lines = [
            "$ pip install demAI",
            "Welcome to demAI — an interactive experience where you will build and operate an AI system, while discovering and applying key concepts from the EU AI Act.\n",
            "",
            "",
            "demonstrateAI",
            "Experience how an AI system actually works, step by step — from data preparation to predictions — through an interactive, hands-on journey.\n",
            "",
            "demistifyAI",
            "Break down complex AI concepts into clear, tangible actions so that anyone can understand what’s behind the model’s decisions.\n",
            "",
            "democratizeAI",
            "Empower everyone to engage responsibly with AI, making transparency and trust accessible to all.",
        ]

    suffix = "welcome_cmd"
    quote_id_prefix = "eu-typing-welcome"
    quote_markup = get_eu_ai_act_typing_inline_markup(id_prefix=quote_id_prefix)
    quote_bootstrap = get_eu_ai_act_typing_inline_bootstrap(id_prefix=quote_id_prefix)

    st.markdown(
        dedent(
            f"""
            <style>
              .cmdgrid-{suffix} {{
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 1.1rem;
                align-items: stretch;
              }}
              @media (max-width: 860px) {{
                .cmdgrid-{suffix} {{ grid-template-columns: 1fr; }}
              }}
              .terminal-{suffix} {{
                background: #0d1117;
                color: #e5e7eb;
                font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
                border-radius: 12px;
                padding: 1.1rem 1rem 1.3rem;
                box-shadow: 0 14px 34px rgba(0,0,0,.25);
                position: relative;
                overflow: hidden;
                min-height: 240px;
                display: flex;
                flex-direction: column;
              }}
              .terminal-{suffix}::before {{
                content: '●';
                position: absolute; top: 10px; left: 12px;
                color: #ef4444cc; letter-spacing: 6px; font-size: .9rem;
              }}
              .term-body-{suffix} {{
                margin-top: 1rem;
                white-space: pre-wrap;
                word-wrap: break-word;
                line-height: 1.55;
                font-size: .95rem;
                flex: 1 1 auto;
              }}
              .cmdline-{suffix} {{ color: #93c5fd; }}
              .hl-{suffix} {{ color: #a5f3fc; }}
              .caret-{suffix} {{
                display:inline-block;
                width:6px;
                height:1rem;
                background:#22d3ee;
                vertical-align:-0.18rem;
                animation: blink-{suffix} .85s steps(1,end) infinite;
              }}
              @keyframes blink-{suffix} {{ 50% {{ opacity: 0; }} }}
              .placeholder-{suffix} {{
                background: linear-gradient(155deg, rgba(248,250,252,.95), rgba(226,232,240,.6));
                border-radius: 12px;
                box-shadow: 0 12px 28px rgba(15,23,42,.08), inset 0 0 0 1px rgba(148,163,184,.25);
                padding: 1rem 1.1rem;
                display: flex;
                flex-direction: column;
                align-items: stretch;
                justify-content: center;
                min-height: 240px;
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

    prepared_lines = _prepare_lines(lines)
    css_classes = [_line_css_class(idx, line, suffix) for idx, line in enumerate(prepared_lines)]
    key_prefix = f"cmdgrid_{suffix}"
    lines_key = tuple(prepared_lines)

    placeholder = st.empty()

    cached_lines = st.session_state.get(f"{key_prefix}_lines")
    cached_render = st.session_state.get(f"{key_prefix}_render")

    if cached_lines == lines_key and cached_render is not None:
        placeholder.markdown(
            _build_grid_html(
                suffix=suffix,
                typed_lines=list(cached_render),
                css_classes=css_classes,
                show_caret=False,
                quote_markup=quote_markup,
            ),
            unsafe_allow_html=True,
        )
        components.html(quote_bootstrap, height=0)
        return

    typed_lines: List[str] = [""] * len(prepared_lines)

    # Initial pause to mimic the original animation timing.
    time.sleep(0.32)

    for idx, line in enumerate(prepared_lines):
        for char in line:
            typed_lines[idx] += char
            placeholder.markdown(
                _build_grid_html(
                    suffix=suffix,
                    typed_lines=typed_lines,
                    css_classes=css_classes,
                    show_caret=True,
                    quote_markup=quote_markup,
                ),
                unsafe_allow_html=True,
            )
            time.sleep(0.024)

        if idx < len(prepared_lines) - 1 and not line.endswith("\n"):
            typed_lines[idx] += "\n"
            placeholder.markdown(
                _build_grid_html(
                    suffix=suffix,
                    typed_lines=typed_lines,
                    css_classes=css_classes,
                    show_caret=True,
                    quote_markup=quote_markup,
                ),
                unsafe_allow_html=True,
            )

        time.sleep(0.36)

    placeholder.markdown(
        _build_grid_html(
            suffix=suffix,
            typed_lines=typed_lines,
            css_classes=css_classes,
            show_caret=False,
            quote_markup=quote_markup,
        ),
        unsafe_allow_html=True,
    )

    st.session_state[f"{key_prefix}_lines"] = lines_key
    st.session_state[f"{key_prefix}_render"] = tuple(typed_lines)
    components.html(quote_bootstrap, height=0)
