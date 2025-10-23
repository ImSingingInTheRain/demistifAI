"""Minimal terminal animation: per-char typing, colored tokens, progressive post-install."""

from __future__ import annotations

import json
import time
from functools import lru_cache
from importlib import resources
from typing import List, Sequence, Tuple

import streamlit as st
from streamlit.components.v1 import html as components_html

from .shared_renderer import build_terminal_render_bundle, make_terminal_renderer

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


@lru_cache(maxsize=1)
def _load_terminal_script() -> str:
    frontend_root = resources.files(__package__).joinpath("frontend")
    with resources.as_file(frontend_root.joinpath("terminal.js")) as path:
        return path.read_text(encoding="utf-8")


def _render_intro_terminal_surface(
    *,
    speed_type_ms: int,
    speed_delete_ms: int,
    pause_between_ops_ms: int,
) -> None:
    bundle = build_terminal_render_bundle(
        suffix=_TERMINAL_SUFFIX,
        lines=_DEFAULT_DEMAI_LINES,
        speed_type_ms=speed_type_ms,
        speed_delete_ms=speed_delete_ms,
        pause_between_ops_ms=pause_between_ops_ms,
        key="intro_inline_terminal",
        show_caret=True,
        accept_keystrokes=False,
    )
    typing_config = {
        "speedType": bundle.payload["speedType"],
        "speedDelete": bundle.payload["speedDelete"],
        "pauseBetween": bundle.payload["pauseBetween"],
    }
    props = {
        "markup": bundle.markup,
        "payload": bundle.payload,
        "serializedLines": bundle.serializable_segments,
        "typingConfig": typing_config,
        "acceptKeystrokes": False,
    }
    js_props = json.dumps(props, ensure_ascii=False, separators=(",", ":"))
    script = _load_terminal_script()
    html = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\" /></head><body>"
        "<div id=\"root\"></div>"
        f"<script>window.__STREAMLIT_TERMINAL_PROPS__ = {js_props};</script>"
        f"<script>{script}</script>"
        "</body></html>"
    )
    components_html(html, height=420, scrolling=False)


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
    typing sequence and the embedded input remain in sync.
    """
    command_triggered, _ = render_interactive_intro_terminal(
        command_key=command_key,
        speed_type_ms=speed_type_ms,
        pause_between_ops_ms=pause_between_ops_ms,
        placeholder="Show Mission",
    )

    return command_triggered


def render_interactive_intro_terminal(
    *,
    command_key: str = "intro_show_mission_cmd",
    speed_type_ms: int = 20,
    speed_delete_ms: int = 14,
    pause_between_ops_ms: int = 360,
    placeholder: str = "Show Mission",
) -> Tuple[bool, bool]:
    """Render the intro terminal animation and capture the command submission."""

    clear_flag_key = f"{command_key}_clear_pending"
    ready_at_key = f"{command_key}_ready_at"
    ready_flag_key = f"{command_key}_ready"
    submit_flag_key = f"{command_key}_submitted"
    animation_duration = _estimate_terminal_duration(
        _DEFAULT_DEMAI_LINES,
        speed_type_ms=speed_type_ms,
        pause_between_ops_ms=pause_between_ops_ms,
    )
    now = time.time()

    if st.session_state.get(clear_flag_key):
        st.session_state[command_key] = ""
        st.session_state[clear_flag_key] = False
        st.session_state[ready_flag_key] = False
        st.session_state[ready_at_key] = now + animation_duration

    if ready_at_key not in st.session_state:
        st.session_state[ready_at_key] = now + animation_duration

    ready = bool(st.session_state.get(ready_flag_key, False))
    if not ready and now >= st.session_state.get(ready_at_key, now):
        ready = True
    st.session_state[ready_flag_key] = ready

    _render_intro_terminal_surface(
        speed_type_ms=speed_type_ms,
        speed_delete_ms=speed_delete_ms,
        pause_between_ops_ms=pause_between_ops_ms,
    )

    def _on_submit() -> None:
        st.session_state[submit_flag_key] = True

    text_value = st.text_input(
        "",
        key=command_key,
        placeholder=placeholder,
        label_visibility="collapsed",
        disabled=not ready,
        on_change=_on_submit,
    )

    submitted = bool(st.session_state.pop(submit_flag_key, False))

    command_triggered = False
    if ready and submitted and text_value.strip().lower() == "show mission":
        command_triggered = True
        st.session_state[clear_flag_key] = True

    if not ready:
        remaining = st.session_state[ready_at_key] - now
        if remaining > 0:
            time.sleep(min(0.2, remaining))
            st.experimental_rerun()

    return command_triggered, ready
