"""Reusable UI components for the demistifAI experience."""

from .arch_demai import (
    ArchitectureCard,
    demai_architecture_markup,
    demai_architecture_styles,
    render_demai_architecture,
)
from .control_room import stage_control_room
from .mac_window import render_mac_window
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
    "render_mac_window",
    "build_training_animation_column",
    "render_training_animation",
    "stage_control_room",
    "terminal",
]
