"""Animated EU AI Act terminal component (non-blocking, auto-resizing, f-string safe)."""

from __future__ import annotations

from typing import List

from .shared_renderer import make_terminal_renderer

_TERMINAL_SUFFIX = "ai_act_fullterm"

_DEFAULT_DEMAI_LINES: List[str] = [
    "$ ai-act fetch --article 3 --term--AI_system\n",
    "[...] Retrieving definition… OK\n",
    "",
    "'AI system means a machine-based system [...]'\n",
    "",
    "> Are you wondering what that means?\n",
    "",
    "$ demAI start --module machine\n",
    "[...] Initializing demAI.machine…\n",
    "[...] Loading components: ui ▸ model ▸ infrastructure\n",
    "[...] Progress 0%   [░░░░░░░░░░░░░░░░░░░░]\n",
    "[... Progress 45%  [██████████░░░░░░░░░░]\n",
    "[...] Progress 100% [████████████████████]\n",
    "",
    "You are already inside a machine-based system: a user interface (software) running in the cloud (hardware).",
    "In each stage, this window will guide you with prompts and key information. Use the control room to jump between",
    "stages and enable Nerd Mode for deeper details.\n",
    "",
    ":help Scroll this page to find out more about the demAI machine.\n",
]

render_ai_act_terminal = make_terminal_renderer(
    suffix=_TERMINAL_SUFFIX,
    default_lines=_DEFAULT_DEMAI_LINES,
    default_key="ai_act_terminal",
)
