from __future__ import annotations

import html
import re
from textwrap import dedent
from typing import Iterable, List

import streamlit as st
import time


_DEFAULT_DEMAI_LINES: List[str] = [
    "> What is an AI system?",
    "",
    "LOADING EU AI ACT, Article 3 \n",
    "",
    "0% ■■■■■■■■■■■■■■■■■■■■■■■■ 100% \n",
    "",
    "AI system means a machine-based system...",
    "",
    "> Wait, but what that actually means? \n",
    "",
    "...",
    "",
    "You are already inside a machine-based system: a user interface (software) running in the cloud (hardware). You will be guided you through each stage with interactive prompts like this. Browse this page to discover more information about the demAI machine. Use the control room to advance to different stages and enable a Nerd Mode when you are thirsty for more details.",
    ""
    ""
    "STARTING demAI.machine",
    "",
    "0% ■■■■■■■■■■■■■■■■■■■■■■■■ 100%",
]

_TERMINAL_SUFFIX = "ai_act_fullterm"
_FINAL_STATE_KEY = "_ai_act_terminal_final_raw"

_TERMINAL_STYLE = dedent(
    f"""
    <style>
      .terminal-{_TERMINAL_SUFFIX} {{
        width: 100%;
        background: #0d1117;
        color: #e5e7eb;
        font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        border-radius: 12px;
        padding: 1.5rem 1rem 1.3rem;
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
      @media (prefers-reduced-motion: reduce) {{
        .caret-{_TERMINAL_SUFFIX} {{ animation: none; }}
      }}
    </style>
    """
)


def _highlight_line(line: str) -> str:
    stripped = line.strip()
    safe = html.escape(line)
    if re.fullmatch(r"dem[a-z]*ai", stripped, flags=re.IGNORECASE):
        return f'<span class="hl-{_TERMINAL_SUFFIX}">{safe}</span>'
    if line.startswith("$ "):
        return f'<span class="cmdline-{_TERMINAL_SUFFIX}">{safe}</span>'
    return safe


def _highlight_raw(raw: str) -> str:
    highlighted_parts: List[str] = []
    for segment in raw.splitlines(keepends=True):
        if segment.endswith("\n"):
            content = segment[:-1]
            suffix = "\n"
        else:
            content = segment
            suffix = ""
        highlighted_parts.append(_highlight_line(content) + suffix)
    return "".join(highlighted_parts)


def _build_terminal_shell(pre_inner: str, caret_visible: bool) -> str:
    caret_style = "display:inline-block;" if caret_visible else "display:none;"
    return dedent(
        f"""
        <div class="terminal-{_TERMINAL_SUFFIX}">
          <pre class="term-body-{_TERMINAL_SUFFIX}">{pre_inner}</pre>
          <span class="caret-{_TERMINAL_SUFFIX}" style="{caret_style}"></span>
        </div>
        """
    )


def _prepare_lines(lines: Iterable[str]) -> List[str]:
    return ["" if line is None else str(line) for line in lines]


def _compute_final_state(demai_lines: Iterable[str]) -> str:
    """Final raw content once the animation completes (for caching)."""
    raw = ""
    for line in demai_lines:
        raw += line
        if not line.endswith("\n"):
            raw += "\n"
    return raw


def render_ai_act_terminal(
    demai_lines=None,
    speed_type_ms: int = 20,
    speed_delete_ms: int = 14,         # kept for API compatibility (unused)
    pause_between_ops_ms: int = 360,   # used as per-line pause
):
    """Render the animated EU AI Act terminal sequence using native Streamlit primitives."""

    if demai_lines is None:
        demai_lines = _DEFAULT_DEMAI_LINES

    prepared_lines = _prepare_lines(demai_lines)

    final_state_key = f"{_FINAL_STATE_KEY}:{hash(tuple(prepared_lines))}"
    final_state = st.session_state.get(final_state_key)

    container = st.container()
    container.markdown(_TERMINAL_STYLE, unsafe_allow_html=True)
    shell_slot = container.empty()

    def _render_shell(raw: str, caret_visible: bool) -> None:
        shell_slot.markdown(
            _build_terminal_shell(_highlight_raw(raw), caret_visible=caret_visible),
            unsafe_allow_html=True,
        )

    if final_state:
        _render_shell(final_state, caret_visible=False)
        return

    final_raw = _compute_final_state(prepared_lines)

    type_delay_s = max(speed_type_ms, 0) / 1000
    pause_delay_s = max(pause_between_ops_ms, 0) / 1000

    raw = ""
    _render_shell(raw, caret_visible=True)

    for line in prepared_lines:
        safe_line = line if isinstance(line, str) else str(line)
        for char in safe_line:
            raw += char
            _render_shell(raw, caret_visible=True)
            if type_delay_s:
                time.sleep(type_delay_s)
        if not safe_line.endswith("\n"):
            raw += "\n"
            _render_shell(raw, caret_visible=True)
            if type_delay_s:
                time.sleep(type_delay_s)
        if pause_delay_s:
            time.sleep(pause_delay_s)

    _render_shell(raw, caret_visible=False)

    st.session_state[final_state_key] = final_raw
