"""Minimal terminal animation: per-char typing, colored tokens, progressive post-install."""

from __future__ import annotations

import time
from typing import List, Sequence

import streamlit as st

from .shared_renderer import make_terminal_renderer

_SUFFIX = "ai_term"

# Preserve the historical constant name referenced throughout the module. A
# previous refactor renamed the suffix constant but missed the call sites,
# leaving `_TERMINAL_SUFFIX` undefined at import time. Keep both names in sync
# so downstream imports remain stable.
_TERMINAL_SUFFIX = _SUFFIX

_DEFAULT_DEMAI_LINES: List[str] = [
    "> What is an AI system?\n",
    "$ fetch EU_AI_ACT.AI_system_definition\n",
    "“AI system” means a machine-based system designed to operate with varying levels of autonomy and that may exhibit adaptiveness after deployment, and that, for explicit or implicit objectives, infers—\n",
    "[…stream truncated…]\n",
    "ERROR 422: Definition overload — too many concepts at once.\n",
    "HINT: Let’s learn it by doing.\n",
    "$ pip install demAI\n",
    "Resolving dependencies… ✓\n",
    "Setting up interactive labs… ✓\n",
    "Verifying examples… ✓\n",
    "Progress 0%   [████████████████████] 100%\n",
    "\nWelcome to demAI — a hands-on way to see how AI works and how the EU AI Act applies in practice.\n",
    "> demonstrateAI\n",
    "Build and run a tiny AI system — from data to predictions — step by step.\n",
    "> demystifyAI\n",
    "Turn buzzwords into concrete actions you can try and understand.\n",
    "> democratizeAI\n",
    "Give everyone the confidence to use AI responsibly with clarity and trust.\n",
    "$ start demo\n",
    "> Type Show Mission or click the button to find out what’s your goal\n",
]

# Backwards-compatibility alias for callers importing the old constant name.
_WELCOME_LINES = _DEFAULT_DEMAI_LINES

render_ai_act_terminal = make_terminal_renderer(
    suffix=_TERMINAL_SUFFIX,
    default_lines=_DEFAULT_DEMAI_LINES,
    default_key="ai_act_terminal",
)


def _estimate_terminal_duration(
    lines: Sequence[str], *, speed_type_ms: int, pause_between_ops_ms: int
) -> float:
    """Approximate the time required to finish the intro typing animation."""

    total_chars = sum(len(line) for line in lines)
    typing_ms = max(0, total_chars * max(0, speed_type_ms))
    pauses_ms = max(0, (len(lines) - 1) * max(0, pause_between_ops_ms))
    buffer_ms = 600  # account for layout/iframe setup time on slower clients
    return (typing_ms + pauses_ms + buffer_ms) / 1000.0


def render_intro_terminal_with_prompt(
    *,
    command_key: str = "intro_show_mission_cmd",
    speed_type_ms: int = 20,
    pause_between_ops_ms: int = 360,
) -> bool:
    """Render the intro terminal and capture the "Show Mission" command."""

    clear_flag_key = f"{command_key}_clear_pending"
    ready_at_key = f"{command_key}_ready_at"
    ready_flag_key = f"{command_key}_ready"
    animation_duration = _estimate_terminal_duration(
        _DEFAULT_DEMAI_LINES,
        speed_type_ms=speed_type_ms,
        pause_between_ops_ms=pause_between_ops_ms,
    )

    if st.session_state.get(clear_flag_key):
        st.session_state[command_key] = ""
        st.session_state[clear_flag_key] = False
        st.session_state[ready_flag_key] = False
        st.session_state[ready_at_key] = time.time() + animation_duration

    if ready_at_key not in st.session_state:
        st.session_state[ready_at_key] = time.time() + animation_duration

    render_ai_act_terminal(
        speed_type_ms=speed_type_ms,
        pause_between_ops_ms=pause_between_ops_ms,
    )

    style_injected_flag = f"{command_key}_style_injected"

    if not st.session_state.get(style_injected_flag):
        st.session_state[style_injected_flag] = True
        st.markdown(
            """
            <style>
            div[data-testid="element-container"]:has(> iframe[title="ai_act_terminal"]) {
                padding-bottom: 0 !important;
                margin-bottom: 0 !important;
            }

            div[data-testid="element-container"]:has(> div[data-testid="stTextInput"] input[placeholder="Show Mission"]) {
                padding-top: 0 !important;
                margin-top: 0 !important;
            }

            div[data-testid="stVerticalBlock"]:has(div[data-testid="stTextInput"] input[placeholder="Show Mission"]) {
                row-gap: 0 !important;
                gap: 0 !important;
            }

            div[data-testid="stTextInput"]:has(input[placeholder="Show Mission"]) {
                background: #0d1117;
                font-family: 'Fira Code', monospace;
                width: min(100%, 680px);
                margin: 0 auto;
                padding: 12px 16px;
                display: flex;
                align-items: center;
                gap: 12px;
                border-radius: 0 0 12px 12px;
                border: 1px solid #30363d;
                border-top: 0;
                box-shadow: none;
                position: relative;
                overflow: hidden;
            }

            div[data-testid="stTextInput"]:has(input[placeholder="Show Mission"]) div[data-baseweb="input"],
            div[data-testid="stTextInput"]:has(input[placeholder="Show Mission"]) div[data-baseweb="input"] > div {
                background: transparent;
                border: 0;
                box-shadow: none;
            }

            div[data-testid="stTextInput"]:has(input[placeholder="Show Mission"])::before {
                content: "$";
                color: #58a6ff;
                font-weight: 500;
            }

            div[data-testid="stTextInput"]:has(input[placeholder="Show Mission"]) > label {
                display: none;
            }

            div[data-testid="stTextInput"]:has(input[placeholder="Show Mission"]) > div {
                flex: 1;
            }

            div[data-testid="stTextInput"]:has(input[placeholder="Show Mission"]) input {
                background: #0d1117;
                border: 0;
                box-shadow: none;
                color: #c9d1d9;
                caret-color: #00c2ff;
                font: inherit;
                width: 100%;
                padding: 0;
                outline: none;
            }

            div[data-testid="stTextInput"]:has(input[placeholder="Show Mission"]) input::placeholder {
                color: rgba(88, 166, 255, 0.7);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    ready_at = st.session_state.get(ready_at_key, 0.0)
    ready = st.session_state.get(ready_flag_key, False)
    now = time.time()

    if not ready and now >= ready_at:
        ready = True
        st.session_state[ready_flag_key] = True

    command = st.text_input(
        "Show Mission command",
        key=command_key,
        placeholder="Show Mission",
        label_visibility="collapsed",
    )

    if not ready:
        remaining_ms = max(0, int((st.session_state.get(ready_at_key, 0.0) - now) * 1000))
        st.markdown(
            f"""
            <script>
            (function() {{
              const delay = {remaining_ms};
              const selector = 'div[data-testid="stTextInput"] input[placeholder="Show Mission"]';
              const findRoot = () => {{
                const inputEl = document.querySelector(selector);
                if (!inputEl) {{
                  window.setTimeout(findRoot, 120);
                  return;
                }}
                const root = inputEl.closest('div[data-testid="stTextInput"]');
                if (!root || root.dataset.introReveal === 'pending') {{
                  return;
                }}
                root.dataset.introReveal = 'pending';
                if (!root.dataset.introOriginalDisplay) {{
                  root.dataset.introOriginalDisplay = root.style.display;
                }}
                root.style.display = 'none';
                window.setTimeout(() => {{
                  const original = root.dataset.introOriginalDisplay;
                  root.style.display = original && original !== 'none' ? original : 'flex';
                  root.dataset.introReveal = 'done';
                }}, delay);
              }};
              findRoot();
            }})();
            </script>
            """,
            unsafe_allow_html=True,
        )

    if command.strip().lower() == "show mission":
        st.session_state[clear_flag_key] = True
        return True

    return False
