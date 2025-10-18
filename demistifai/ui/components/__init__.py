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
from .data_review import (
    data_review_styles,
    dataset_balance_bar_html,
    edge_case_pairs_html,
    stratified_sample_cards_html,
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
from .train_intro import (
    TrainingNotesColumn,
    build_inline_note,
    build_launchpad_card,
    build_launchpad_status_item,
    build_token_chip,
    build_train_intro_card,
    build_training_notes_column,
    build_nerd_intro_card,
    training_stage_stylesheet,
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
    "TrainingNotesColumn",
    "build_inline_note",
    "build_launchpad_card",
    "build_launchpad_status_item",
    "build_token_chip",
    "build_train_intro_card",
    "build_training_notes_column",
    "build_nerd_intro_card",
    "training_stage_stylesheet",
    "stage_control_room",
    "terminal",
    "data_review_styles",
    "dataset_balance_bar_html",
    "edge_case_pairs_html",
    "stratified_sample_cards_html",
]
