"""Minimal terminal animation: per-char typing, colored tokens, progressive post-install."""

from __future__ import annotations

import time
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import streamlit as st

from .interactive_terminal_component import render_interactive_terminal
from .intro_session import IntroTerminalSession
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
    pause_between_ops_ms: int = 360,
    placeholder: str = "Type here",
) -> Tuple[Optional[IntroTerminalCommand], bool]:
    """Render the intro terminal animation and capture the command submission.

    Returns a tuple containing the parsed command (if any) and whether the
    terminal is ready for direct user input without replaying the animation.
    """

    session = IntroTerminalSession(command_key=command_key)
    component_key = session.component_key
    now = time.time()

    lines = session.ensure_lines(_DEFAULT_DEMAI_LINES)

    animation_duration = _estimate_terminal_duration(
        lines,
        speed_type_ms=speed_type_ms,
        pause_between_ops_ms=pause_between_ops_ms,
    )

    ready = session.ready
    previous_signature = session.lines_signature
    current_signature = tuple(lines)
    keep_input_active = session.consume_input_focus_request()

    pending_component_state = st.session_state.get(component_key)
    pending_component_ready = False
    pending_component_payload: Optional[dict] = None
    if isinstance(pending_component_state, dict):
        pending_value = pending_component_state.get("value")
        if isinstance(pending_value, dict):
            pending_component_payload = pending_value
            pending_component_ready = bool(pending_value.get("ready", False))

    if pending_component_ready:
        if not ready:
            ready = True
        session.clear_ready_deadline()

    skip_animation = ready and previous_signature == current_signature

    if not ready and session.ready_deadline is None:
        session.set_ready_deadline(now + animation_duration)

    persisted_text = session.input_text
    if pending_component_payload:
        pending_text = pending_component_payload.get("text")
        if isinstance(pending_text, str):
            persisted_text = pending_text

    prefilled_line_count = session.prefilled_line_count
    if prefilled_line_count <= 0 and _SHOW_MISSION_USER_LINE in lines:
        try:
            inferred_prefill_index = lines.index(_SHOW_MISSION_USER_LINE)
        except ValueError:  # pragma: no cover - defensive guard
            inferred_prefill_index = 0
        if inferred_prefill_index > 0:
            prefilled_line_count = inferred_prefill_index
            session.set_prefilled_line_count(inferred_prefill_index)

    line_delta_hint: Optional[dict[str, object]]
    if previous_signature is None:
        line_delta_hint = {"action": "replace", "lines": list(lines)}
    elif previous_signature == current_signature:
        line_delta_hint = {"action": "none"}
    elif len(previous_signature) <= len(current_signature) and current_signature[: len(previous_signature)] == previous_signature:
        appended = list(lines[len(previous_signature) :])
        line_delta_hint = {"action": "append", "lines": appended} if appended else {"action": "none"}
    else:
        line_delta_hint = {"action": "replace", "lines": list(lines)}

    component_payload = render_interactive_terminal(
        suffix=_TERMINAL_SUFFIX,
        lines=lines,
        speed_type_ms=0 if skip_animation else speed_type_ms,
        pause_between_ops_ms=0 if skip_animation else pause_between_ops_ms,
        key=component_key,
        placeholder=placeholder,
        accept_keystrokes=True,
        show_caret=True,
        value=persisted_text,
        prefilled_line_count=prefilled_line_count,
        line_delta=line_delta_hint,
        keep_input_active=keep_input_active,
    )

    session.set_lines_signature(current_signature)

    component_text = persisted_text
    component_ready = False
    component_submitted = False

    if component_payload:
        component_text = component_payload.get("text", "")
        component_ready = bool(component_payload.get("ready", False))
        component_submitted = bool(component_payload.get("submitted", False))
        session.set_input_text(component_text)

    ready = ready or component_ready
    if not ready:
        ready_at = session.ready_deadline or now
        if now >= ready_at:
            ready = True
            session.clear_ready_deadline()

    if component_ready:
        ready = True
        session.clear_ready_deadline()

    command: Optional[IntroTerminalCommand] = None
    text_value = component_text.strip().lower()
    if ready and component_submitted:
        if text_value == "show mission":
            command = IntroTerminalCommand.SHOW_MISSION
            if _SHOW_MISSION_USER_LINE not in lines:
                rendered_line_count = len(lines)
                session.append_lines(
                    [
                        _SHOW_MISSION_USER_LINE,
                        *_SHOW_MISSION_RESPONSE_LINE,
                    ],
                    prefill_line_count=rendered_line_count,
                    keep_input_active=True,
                )
            session.request_input_focus()
        elif text_value == "start":
            command = IntroTerminalCommand.START
            session.preserve_input(component_text)
            session.request_input_focus()

    session.set_ready(ready)
    return command, ready
