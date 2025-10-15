"""High-level UI components for the demistifAI experience."""

from .arch_demai import (
    ArchitectureCard,
    demai_architecture_markup,
    demai_architecture_styles,
    render_demai_architecture,
)
from .components_cmd import render_ai_act_terminal
from .components_mac import render_mac_window
from .stage_control_room import stage_control_room
from .ui_command_grid import render_command_grid
from .ui_typing_quote import (
    get_eu_ai_act_typing_inline_bootstrap,
    get_eu_ai_act_typing_inline_markup,
)

__all__ = [
    "ArchitectureCard",
    "demai_architecture_markup",
    "demai_architecture_styles",
    "render_demai_architecture",
    "render_ai_act_terminal",
    "render_mac_window",
    "stage_control_room",
    "render_command_grid",
    "get_eu_ai_act_typing_inline_bootstrap",
    "get_eu_ai_act_typing_inline_markup",
]
