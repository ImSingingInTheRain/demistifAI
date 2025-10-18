"""Reusable UI components for the demistifAI experience."""

from .arch_demai import (
    ArchitectureCard,
    demai_architecture_markup,
    demai_architecture_styles,
    render_demai_architecture,
)
from .control_room import stage_control_room
from .intro_hero import (
    intro_ai_act_quote_wrapper_close,
    intro_ai_act_quote_wrapper_open,
    intro_hero_scoped_css,
    intro_lifecycle_columns,
    render_intro_hero,
)
from .mac_window import render_mac_window
from .overview_mission import (
    mailbox_preview_markup,
    mission_brief_markup,
    mission_brief_styles,
)
from .train_animation import (
    build_training_animation_column,
    render_training_animation,
)
from . import terminal

__all__ = [
    "ArchitectureCard",
    "demai_architecture_markup",
    "demai_architecture_styles",
    "render_demai_architecture",
    "intro_ai_act_quote_wrapper_close",
    "intro_ai_act_quote_wrapper_open",
    "intro_hero_scoped_css",
    "intro_lifecycle_columns",
    "render_mac_window",
    "mailbox_preview_markup",
    "mission_brief_markup",
    "mission_brief_styles",
    "render_intro_hero",
    "build_training_animation_column",
    "render_training_animation",
    "stage_control_room",
    "terminal",
]
