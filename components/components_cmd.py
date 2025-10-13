from __future__ import annotations

import html
import re
import time
from textwrap import dedent
from typing import Iterable, List

import streamlit as st


_DEFAULT_DEMAI_LINES: List[str] = [
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
]

_DEFAULT_OPS = [
    {"kind": "type", "text": "$ get EU-AI-Act.definition\n\n"},
    {"kind": "type", "text": " ‘AI system’ means a machine-based system that is designed to operate with varying levels of autonomy and that may exhibit adaptiveness after deployment, and that, for explicit or implicit objectives, infers, from the input it receives, how to generate outputs such as predictions, content, recommendations, or decisions that can influence physical or virtual environments; \n\n"},
    {"kind": "type", "text": "confused?\n\n"},
    {"kind": "type", "text": "$ pip install demAI\n\n"},
]

_TERMINAL_SUFFIX = "ai_act_fullterm"
_CSS_KEY = "_ai_act_terminal_css_injected"
_FINAL_STATE_KEY = "_ai_act_terminal_final_raw"


def _ensure_terminal_css() -> None:
    if st.session_state.get(_CSS_KEY):
        return

    css = dedent(
        f"""
        <style>
          .terminal-{_TERMINAL_SUFFIX} {{
            width: 100%;
            background: #0d1117;
            color: #e5e7eb;
            font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
            border-radius: 12px;
            padding: 1.1rem 1rem 1.3rem;
            box-shadow: 0 14px 34px rgba(0,0,0,.25);
            position: relative;
            overflow: hidden;
            min-height: 260px;
          }}
          .terminal-{_TERMINAL_SUFFIX}::before {{
            content: '●  ●  ●';
            position: absolute; top: 8px; left: 12px;
            color: #ef4444cc; letter-spacing: 6px; font-size: .9rem;
          }}
          .term-body-{_TERMINAL_SUFFIX} {{
            margin-top: .8rem;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
            font-size: .96rem;
          }}
          .caret-{_TERMINAL_SUFFIX} {{
            margin-left: 2px;
            display:inline-block; width:6px; height:1rem;
            background:#22d3ee; vertical-align:-0.18rem;
            animation: blink-{_TERMINAL_SUFFIX} .85s steps(1,end) infinite;
          }}
          .cmdline-{_TERMINAL_SUFFIX} {{ color: #93c5fd; }}
          .hl-{_TERMINAL_SUFFIX}     {{ color: #a5f3fc; font-weight: 600; }}
          @keyframes blink-{_TERMINAL_SUFFIX} {{ 50% {{ opacity: 0; }} }}
        </style>
        """
    )

    st.markdown(css, unsafe_allow_html=True)
    st.session_state[_CSS_KEY] = True


def _highlight_line(line: str) -> str:
    stripped = line.strip()
    safe = html.escape(line)
    if re.fullmatch(r"dem[a-z]*ai", stripped, flags=re.IGNORECASE):
        return f'<span class="hl-{_TERMINAL_SUFFIX}">{safe}</span>'
    if line.startswith("$ "):
        return f'<span class="cmdline-{_TERMINAL_SUFFIX}">{safe}</span>'
    return safe


def _render_terminal_html(placeholder, raw: str, show_caret: bool) -> None:
    highlighted_parts = []
    for segment in raw.splitlines(keepends=True):
        if segment.endswith("\n"):
            content = segment[:-1]
            suffix = "\n"
        else:
            content = segment
            suffix = ""
        highlighted_parts.append(_highlight_line(content) + suffix)

    highlighted = "".join(highlighted_parts)
    caret_style = "display:inline-block;" if show_caret else "display:none;"
    html_payload = dedent(
        f"""
        <div class="terminal-{_TERMINAL_SUFFIX}">
          <pre class="term-body-{_TERMINAL_SUFFIX}">{highlighted}</pre>
          <span class="caret-{_TERMINAL_SUFFIX}" style="{caret_style}"></span>
        </div>
        """
    )
    placeholder.markdown(html_payload, unsafe_allow_html=True)


def _run_typewriter_animation(
    placeholder,
    ops: Iterable[dict],
    demai_lines: Iterable[str],
    speed_type_ms: int,
    speed_delete_ms: int,
    pause_between_ops_ms: int,
) -> str:
    raw = ""
    type_delay = max(speed_type_ms, 0) / 1000.0
    delete_delay = max(speed_delete_ms, 0) / 1000.0
    pause_delay = max(pause_between_ops_ms, 0) / 1000.0

    def render(show_caret: bool = True) -> None:
        _render_terminal_html(placeholder, raw, show_caret)

    render(True)

    def type_text(text: str) -> None:
        nonlocal raw
        if not text:
            return
        for ch in text:
            raw += ch
            render(True)
            if type_delay:
                time.sleep(type_delay)

    def delete_text(text: str) -> None:
        nonlocal raw
        if not text or not raw.endswith(text):
            render(True)
            return
        for _ in range(len(text)):
            raw = raw[:-1]
            render(True)
            if delete_delay:
                time.sleep(delete_delay)

    for op in ops:
        kind = op.get("kind")
        text = op.get("text", "")
        if kind == "type":
            type_text(text)
        elif kind == "delete":
            delete_text(text)
        else:
            render(True)
        if pause_delay:
            time.sleep(pause_delay)

    # Add spacing before the deMAI manifesto
    type_text("\n")
    for line in demai_lines:
        type_text(line)
        if not line.endswith("\n"):
            type_text("\n")

    render(False)
    return raw


def render_ai_act_terminal(
    demai_lines=None,
    speed_type_ms: int = 20,
    speed_delete_ms: int = 14,
    pause_between_ops_ms: int = 360,
):
    """Render the animated EU AI Act terminal sequence using native Streamlit primitives."""

    if demai_lines is None:
        demai_lines = _DEFAULT_DEMAI_LINES

    _ensure_terminal_css()
    container = st.container()
    placeholder = container.empty()

    final_state_key = f"{_FINAL_STATE_KEY}:{hash(tuple(demai_lines))}"
    final_state = st.session_state.get(final_state_key)
    if final_state:
        _render_terminal_html(placeholder, final_state, show_caret=False)
        return

    raw = _run_typewriter_animation(
        placeholder,
        _DEFAULT_OPS,
        demai_lines,
        speed_type_ms=speed_type_ms,
        speed_delete_ms=speed_delete_ms,
        pause_between_ops_ms=pause_between_ops_ms,
    )

    st.session_state[final_state_key] = raw
