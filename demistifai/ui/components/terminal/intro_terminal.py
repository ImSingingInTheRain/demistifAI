"""Minimal terminal animation: per-char typing, colored tokens, progressive post-install."""

from __future__ import annotations

from typing import List

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


def render_intro_terminal_with_prompt(
    *,
    command_key: str = "intro_show_mission_cmd",
    speed_type_ms: int = 20,
    pause_between_ops_ms: int = 360,
) -> bool:
    """Render the intro terminal and capture the "Show Mission" command."""

    clear_flag_key = f"{command_key}_clear_pending"

    if st.session_state.get(clear_flag_key):
        st.session_state[command_key] = ""
        st.session_state[clear_flag_key] = False

    render_ai_act_terminal(
        speed_type_ms=speed_type_ms,
        pause_between_ops_ms=pause_between_ops_ms,
    )

    style_injected_flag = f"{command_key}_style_injected"

    if not st.session_state.get(style_injected_flag):
        st.session_state[style_injected_flag] = True
        st.markdown(
            f"""
            <style>
            .intro-terminal-command {{
                background: #0d1117;
                font-family: 'Fira Code', monospace;
                width: min(100%, 680px);
                margin: -8px auto 0;
                padding: 12px 16px;
                display: flex;
                align-items: center;
                gap: 12px;
                border-radius: 0 0 12px 12px;
                border-top: 0;
            }}

            .intro-terminal-command span.intro-terminal-prompt {{
                color: #58a6ff;
                font-weight: 500;
            }}

            .intro-terminal-command [data-testid="stTextInput"] {{
                flex: 1;
                margin: 0;
            }}

            .intro-terminal-command [data-testid="stTextInput"] label {{
                display: none;
            }}

            .intro-terminal-command [data-testid="stTextInput"] > div {{
                margin: 0;
                padding: 0;
                background: transparent;
            }}

            .intro-terminal-command input[id="{command_key}"] {{
                background: transparent;
                border: 0;
                box-shadow: none;
                color: #c9d1d9;
                caret-color: #00c2ff;
                font: inherit;
                width: 100%;
                padding: 0;
                outline: none;
            }}

            .intro-terminal-command input[id="{command_key}"]::placeholder {{
                color: rgba(88, 166, 255, 0.7);
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="intro-terminal-command">
            <span class="intro-terminal-prompt">$</span>
        """,
        unsafe_allow_html=True,
    )

    command = st.text_input(
        "Show Mission command",
        key=command_key,
        placeholder="Show Mission",
        label_visibility="collapsed",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if command.strip().lower() == "show mission":
        st.session_state[clear_flag_key] = True
        return True

    return False
