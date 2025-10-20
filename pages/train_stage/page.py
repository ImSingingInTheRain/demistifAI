from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

from demistifai.constants import STAGE_BY_KEY, STAGE_INDEX, StageMeta
from demistifai.core.language import HAS_LANGDETECT
from demistifai.core.state import ensure_state, validate_invariants
from demistifai.core.utils import streamlit_rerun
from demistifai.modeling import encode_texts
from pages.train_stage.helpers.meaning_map import (
    _build_meaning_map_chart,
    _meaning_map_zoom_subset,
    _prepare_meaning_map,
)
from pages.train_stage.helpers.storyboard import (
    _render_training_examples_preview,
    _render_unified_training_storyboard,
)

from demistifai.ui.components import render_stage_top_grid
from demistifai.ui.components.train import (
    build_inline_note,
    build_training_animation_column,
    build_token_chip,
    build_training_notes_column,
    render_guardrail_window_preview,
    render_numeric_clue_cards,
    render_numeric_clue_preview,
    training_stage_stylesheet,
)
from demistifai.ui.theme.macos_window import (
    build_macos_window,
    inject_macos_window_theme,
)
from .callbacks import (
    callable_or_attr,
    parse_split_cache,
    prime_session_state_from_store,
    request_meaning_map_refresh,
    sync_training_artifacts,
)
from .navigation import render_prepare_dataset_prompt, render_train_terminal_slot
from .visualizations import build_calibration_chart
from .panels import render_launchpad_panel, render_nerd_mode_panels
from .results import render_training_results
from .state import (
    compute_token_budget_summary,
    ensure_train_stage_state,
    execute_training_pipeline,
    parse_train_params,
    persist_training_outcome,
    register_training_refresh,
    reset_meaning_map_flags,
)


logger = logging.getLogger(__name__)
def render_train_stage_page(
    *,
    set_active_stage: Callable[[str], None],
    render_eu_ai_quote,
    render_language_mix_chip_rows,
    render_nerd_mode_toggle,
    section_surface,
    summarize_language_mix,
):
    """Entry point for the Train stage when called from ``streamlit_app``."""

    s = ensure_state()
    validate_invariants(s)

    ss = st.session_state

    run_state = s.setdefault("run", {})
    run_state.setdefault("busy", False)
    data_state = s.setdefault("data", {})
    split_state = s.setdefault("split", {})
    model_state = s.setdefault("model", {})
    stage = STAGE_BY_KEY["train"]

    prepared_df = data_state.get("prepared")

    render_stage_top_grid("train", left_renderer=render_train_terminal_slot)

    if prepared_df is None:
        render_prepare_dataset_prompt(stage, section_surface, set_active_stage)
        return

    prime_session_state_from_store(ss, model_state, split_state)

    animation_column = build_training_animation_column()
    notes_column = build_training_notes_column()

    inject_macos_window_theme(st)
    st.markdown(
        build_macos_window(
            title="Training dynamics monitor",
            subtitle="MiniLM organising emails by meaning",
            column_blocks=(notes_column.html, animation_column.html),
            columns=2,
            ratios=(0.3, 0.7),
            id_suffix="train-animation",
            scoped_css=notes_column.css,
        ),
        unsafe_allow_html=True,
    )

    has_embed = callable_or_attr(encode_texts)

    render_train_stage(
        ss,
        streamlit_rerun=streamlit_rerun,
        has_embed=has_embed,
        has_langdetect=HAS_LANGDETECT,
        render_eu_ai_quote=render_eu_ai_quote,
        render_language_mix_chip_rows=render_language_mix_chip_rows,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        section_surface=section_surface,
        summarize_language_mix=summarize_language_mix,
    )

    if isinstance(prepared_df, pd.DataFrame) and not prepared_df.empty:
        try:
            sync_training_artifacts(
                prepared_df,
                ss=ss,
                run_state=run_state,
                data_state=data_state,
                split_state=split_state,
                model_state=model_state,
            )
        except RuntimeError as exc:
            logger.exception("Failed to update cached training artifacts")
            detail = exc.__cause__ or exc
            st.error(f"Failed to update training artifacts: {detail}")


def render_train_stage(
    ss,
    *,
    streamlit_rerun,
    has_embed,
    has_langdetect,
    render_eu_ai_quote,
    render_language_mix_chip_rows,
    render_nerd_mode_toggle,
    section_surface,
    summarize_language_mix,
):


    stage = STAGE_BY_KEY["train"]
    ss.setdefault("token_budget_cache", {})
    ss.setdefault("guard_params", {})
    ss.setdefault("train_in_progress", False)
    ss.setdefault("train_refresh_expected", False)
    ss.setdefault("train_refresh_attempts", 0)
    ss["guard_params"].setdefault("assist_center_mode", "manual")  # "manual" | "auto"
    ss["guard_params"].setdefault("assist_center", float(ss.get("threshold", 0.6)))
    ss["guard_params"].setdefault("uncertainty_band", 0.08)
    ss["guard_params"].setdefault("numeric_scale", 0.5)
    ss["guard_params"].setdefault("numeric_logit_cap", 1.0)
    ss["guard_params"].setdefault("combine_strategy", "blend")

    st.markdown(training_stage_stylesheet(), unsafe_allow_html=True)

    stage_number = STAGE_INDEX.get(stage.key, 0) - STAGE_INDEX.get("data", 0) + 1
    if stage_number < 1:
        stage_number = STAGE_INDEX.get(stage.key, 0) + 1

    guard_params = ensure_train_stage_state(ss, threshold=float(ss.get("threshold", 0.6)))

    nerd_mode_train_enabled = render_launchpad_panel(
        ss,
        stage=stage,
        stage_number=stage_number,
        render_eu_ai_quote=render_eu_ai_quote,
        section_surface=section_surface,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        guard_params=guard_params,
    )

    if nerd_mode_train_enabled:
        render_nerd_mode_panels(ss, section_surface=section_surface)

    token_summary, token_cache = compute_token_budget_summary(
        ss.get("labeled", []), ss.get("token_budget_cache", {})
    )
    ss["token_budget_cache"] = token_cache

    if token_summary.show_truncation_tip and token_summary.text:
        st.markdown(build_token_chip(token_summary.text), unsafe_allow_html=True)
        st.markdown(
            build_inline_note("Tip: long emails will be clipped; summaries help."),
            unsafe_allow_html=True,
        )

    should_rerun_after_training = False
    training_successful = False

    with section_surface():
        action_col, context_col = st.columns([0.55, 0.45], gap="large")
        with action_col:
            st.markdown(
                """
                <div class="train-action-card">
                    <div class="train-action-card__eyebrow">Run training</div>
                    <div class="train-action-card__title">Teach the spam detector</div>
                    <p class="train-action-card__body">When you’re ready, launch the training run. We’ll automatically evaluate on the hold-out split.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            train_disabled = bool(ss.get("train_in_progress"))
            trigger_train = st.button(
                "🚀 Train model",
                type="primary",
                use_container_width=True,
                disabled=train_disabled,
            )
            if train_disabled:
                st.caption("Training in progress… hang tight while we refresh the charts.")
            if ss.get("train_flash_finished"):
                st.success("Training finished.")
                ss["train_flash_finished"] = False
        with context_col:

            if not ss.get("nerd_mode_train"):
                render_guardrail_window_preview(ss)


    has_model = ss.get("model") is not None
    has_split_cache = ss.get("split_cache") is not None

    def _request_refresh(section: str | None) -> None:
        request_meaning_map_refresh(ss, section, streamlit_rerun)

    _render_unified_training_storyboard(
        ss,
        has_model=has_model,
        has_split=has_split_cache,
        has_embed=bool(has_embed),
        section_surface=section_surface,
        request_meaning_map_refresh=_request_refresh,
        parse_split_cache=parse_split_cache,
        rerun=streamlit_rerun,
        logger=logger,
    )

    if trigger_train:
        if len(ss["labeled"]) < 6:
            st.warning("Please label a few more emails first (≥6 examples).")
        else:
            df = pd.DataFrame(ss["labeled"])
            if len(df["label"].unique()) < 2:
                st.warning("You need both classes (spam and safe) present to train.")
            else:
                params = parse_train_params(ss.get("train_params", {}))
                titles = df["title"].fillna("").tolist()
                bodies = df["body"].fillna("").tolist()
                labels = df["label"].tolist()
                ss["train_in_progress"] = True
                try:
                    with st.spinner("Training the model and refreshing charts…"):
                        training_outcome = execute_training_pipeline(
                            titles,
                            bodies,
                            labels,
                            params=params,
                            guard_params=dict(ss.get("guard_params", {})),
                            numeric_adjustments=ss.get("numeric_adjustments"),
                            has_embed=bool(has_embed),
                            fallback_center=float(ss.get("threshold", 0.6)),
                        )
                        if training_outcome.auto_select_error:
                            logger.error(
                                "Failed to auto-select assist center from hold-out set",
                                exc_info=training_outcome.auto_select_error,
                            )
                        persist_training_outcome(
                            ss,
                            model=training_outcome.model,
                            split=training_outcome.split,
                            guard_params=training_outcome.guard_params,
                        )
                        register_training_refresh(
                            ss, threshold=float(ss.get("threshold", 0.6))
                        )
                        reset_meaning_map_flags(ss)
                        should_rerun_after_training = True
                        training_successful = True
                finally:
                    ss["train_in_progress"] = False

    if should_rerun_after_training and training_successful:
        streamlit_rerun()
        return

    render_training_results(
        ss,
        has_embed=bool(has_embed),
        has_langdetect=has_langdetect,
        render_language_mix_chip_rows=render_language_mix_chip_rows,
        summarize_language_mix=summarize_language_mix,
        parse_split_cache=parse_split_cache,
    )
