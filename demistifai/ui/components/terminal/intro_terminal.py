"""Minimal terminal animation: per-char typing, colored tokens, progressive post-install."""

from __future__ import annotations

import time
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import streamlit as st

from .interactive_terminal_component import render_interactive_terminal
from .shared_renderer import make_terminal_renderer

_SUFFIX = "ai_term"

# Preserve the historical constant name referenced throughout the module. A
# previous refactor renamed the suffix constant but missed the call sites,
# leaving `_TERMINAL_SUFFIX` undefined at import time. Keep both names in sync
# so downstream imports remain stable.
_TERMINAL_SUFFIX = _SUFFIX

_SHOW_MISSION_USER_LINE = "> Show Mission\n"
_SHOW_MISSION_RESPONSE_LINE: Tuple[str, str] = (
    ">show mission\n",
    "> I opened the mission overview below, check it out and when you are ready type \"start\" here\n",
)


class IntroTerminalCommand(str, Enum):
    """Command identifiers emitted by the intro terminal component."""

    SHOW_MISSION = "show_mission"
    START = "start"


_DEFAULT_DEMAI_LINES: List[str] = [
    "> What is an AI system?\n",
    "$ fetch EU_AI_ACT\n",
    "“AI system” means a machine-based system designed to operate with varying levels of autonomy and that may exhibit adaptiveness after deployment, and that, for explicit or implicit objectives, infers—\n",
    "[…stream truncated…]\n",
    "ERROR 422: Definition overload — too many concepts at once.\n",
    "",
    "$ pip install demAI\n",
    "Progress 0%   [████████████████████] 100%\n",
    "",
    "\nWelcome to demAI — a hands-on way to see how AI works and how the EU AI Act applies in practice.\n",
    "> demonstrateAI\n",
    "Build and run a tiny AI system — from data to predictions — step by step.\n",
    "> demystifyAI\n",
    "Turn buzzwords into concrete actions you can try and understand.\n",
    "> democratizeAI\n",
    "Give everyone the confidence to use AI responsibly with clarity and trust.\n",
    "",
    "> Ready to start?\n"
    "> Type \"show mission\" here or click the button to find out what’s your goal\n",
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
    """Render the intro terminal and capture the "Show Mission" command.

    The helper exists for callers that still expect the legacy "prompt" API;
    under the hood it proxies to the inline animation renderer so the
    typing sequence and the embedded input remain in sync. It returns ``True``
    when the "show mission" command is submitted, mirroring the legacy
    behaviour expected by older call sites.
    """
    command, _ = render_interactive_intro_terminal(
        command_key=command_key,
        speed_type_ms=speed_type_ms,
        pause_between_ops_ms=pause_between_ops_ms,
        placeholder="Type here",
    )

    return command == IntroTerminalCommand.SHOW_MISSION


def render_interactive_intro_terminal(
    *,
    command_key: str = "intro_show_mission_cmd",
    speed_type_ms: int = 20,
    speed_delete_ms: int = 14,
    pause_between_ops_ms: int = 360,
    placeholder: str = "Type here",
) -> Tuple[Optional[IntroTerminalCommand], bool]:
    """Render the intro terminal animation and capture the command submission.

    Returns a tuple containing the parsed command (if any) and whether the
    terminal is ready for direct user input without replaying the animation.
    """

    clear_flag_key = f"{command_key}_clear_pending"
    ready_at_key = f"{command_key}_ready_at"
    ready_flag_key = f"{command_key}_ready"
    lines_state_key = f"{command_key}_lines"
    lines_signature_key = f"{command_key}_lines_signature"
    append_pending_key = f"{command_key}_append_pending"
    preserve_state_key = f"{command_key}_preserve_state"
    component_key = "intro_inline_terminal"
    now = time.time()

    if st.session_state.get(clear_flag_key):
        st.session_state.pop(command_key, None)
        st.session_state.pop(component_key, None)
        st.session_state.pop(ready_at_key, None)
        append_pending = bool(st.session_state.pop(append_pending_key, False))
        preserve_state = bool(st.session_state.pop(preserve_state_key, False))
        reset_lines_state = not (append_pending or preserve_state)
        if reset_lines_state:
            st.session_state.pop(lines_state_key, None)
            st.session_state.pop(lines_signature_key, None)
        st.session_state[clear_flag_key] = False
        if reset_lines_state:
            st.session_state[ready_flag_key] = False

    lines = st.session_state.get(lines_state_key)
    if not isinstance(lines, list) or not lines:
        lines = list(_DEFAULT_DEMAI_LINES)
        st.session_state[lines_state_key] = lines

    animation_duration = _estimate_terminal_duration(
        lines,
        speed_type_ms=speed_type_ms,
        pause_between_ops_ms=pause_between_ops_ms,
    )

    ready = bool(st.session_state.get(ready_flag_key, False))
    previous_signature: Optional[Tuple[str, ...]] = st.session_state.get(
        lines_signature_key
    )
    current_signature = tuple(lines)
    skip_animation = ready and previous_signature == current_signature

    if not ready and ready_at_key not in st.session_state:
        st.session_state[ready_at_key] = now + animation_duration

    component_payload = render_interactive_terminal(
        suffix=_TERMINAL_SUFFIX,
        lines=lines,
        speed_type_ms=0 if skip_animation else speed_type_ms,
        speed_delete_ms=speed_delete_ms,
        pause_between_ops_ms=0 if skip_animation else pause_between_ops_ms,
        key=component_key,
        placeholder=placeholder,
        accept_keystrokes=True,
        show_caret=True,
    )

    st.session_state[lines_signature_key] = current_signature

    component_text = st.session_state.get(command_key, "")
    component_ready = False
    component_submitted = False

    if component_payload:
        component_text = component_payload.get("text", "")
        component_ready = bool(component_payload.get("ready", False))
        component_submitted = bool(component_payload.get("submitted", False))
        st.session_state[command_key] = component_text

    ready = ready or component_ready
    if not ready:
        ready_at = st.session_state.get(ready_at_key, now)
        if now >= ready_at:
            ready = True
            st.session_state.pop(ready_at_key, None)

    if component_ready:
        ready = True
        st.session_state.pop(ready_at_key, None)

    st.session_state[ready_flag_key] = ready

    command: Optional[IntroTerminalCommand] = None
    text_value = component_text.strip().lower()
    if ready and component_submitted:
        if text_value == "show mission":
            command = IntroTerminalCommand.SHOW_MISSION
            if _SHOW_MISSION_USER_LINE not in lines:
                lines.extend([
                    _SHOW_MISSION_USER_LINE,
                    *_SHOW_MISSION_RESPONSE_LINE,
                ])
            st.session_state[append_pending_key] = True
            st.session_state[clear_flag_key] = True
            st.session_state.pop(component_key, None)
            st.session_state.pop(command_key, None)
        elif text_value == "start":
            command = IntroTerminalCommand.START
            st.session_state[preserve_state_key] = True
            st.session_state[clear_flag_key] = True
            st.session_state.pop(component_key, None)
            st.session_state.pop(command_key, None)

    return command, ready
