"""Reusable UI components for the demistifAI experience."""

from .arch_demai import (
    ArchitectureCard,
    demai_architecture_markup,
    demai_architecture_styles,
    render_demai_architecture,
)
from .intro_hero import (
    intro_hero_scoped_css,
    intro_lifecycle_columns,
    render_intro_hero,
)
from .guardrail_panel import render_guardrail_panel
from .data_review import (
    data_review_styles,
    dataset_balance_bar_html,
    dataset_snapshot_active_badge,
    dataset_snapshot_card_html,
    dataset_snapshot_styles,
    edge_case_pairs_html,
    stratified_sample_cards_html,
)
from .language_mix import render_language_mix_chip_rows
from .pii_indicators import render_pii_indicators
from .mac_window import render_mac_window
from .overview_mission import (
    mailbox_preview_markup,
    mission_brief_markup,
    mission_brief_styles,
)
from .pii import (
    _ensure_pii_state,
    _highlight_spans_html,
    pii_chip_row_html,
    render_pii_cleanup_banner,
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
from .stage_navigation import (
    StageBlockRenderer,
    StageTopCardContent,
    render_stage_top_grid,
)
from . import terminal

__all__ = [
    "ArchitectureCard",
    "demai_architecture_markup",
    "demai_architecture_styles",
    "render_demai_architecture",
    "intro_hero_scoped_css",
    "intro_lifecycle_columns",
    "render_mac_window",
    "mailbox_preview_markup",
    "mission_brief_markup",
    "mission_brief_styles",
    "render_intro_hero",
    "render_guardrail_panel",
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
    "terminal",
    "data_review_styles",
    "dataset_balance_bar_html",
    "dataset_snapshot_active_badge",
    "dataset_snapshot_card_html",
    "dataset_snapshot_styles",
    "edge_case_pairs_html",
    "stratified_sample_cards_html",
    "render_pii_indicators",
    "StageBlockRenderer",
    "StageTopCardContent",
    "render_stage_top_grid",
    "render_language_mix_chip_rows",
    "_ensure_pii_state",
    "_highlight_spans_html",
    "pii_chip_row_html",
    "render_pii_cleanup_banner",
]
