"""Streamlit page orchestration for the Prepare/Data stage."""

from __future__ import annotations

from typing import Any, Callable, ContextManager, Optional

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from demistifai.core.dataset import compute_dataset_summary
from demistifai.ui.components import render_stage_top_grid
from demistifai.core.state import ensure_state, validate_invariants
from demistifai.ui.components.terminal.data_prep import render_prepare_terminal

from pages.data.builder import DatasetBuilderResult, render_dataset_builder, render_prepare_panel
from pages.data.dataset_io import SessionState, prepare_records
from pages.data.pii import render_pii_cleanup
from pages.data.review import (
    render_csv_upload,
    render_dataset_health_section,
    render_dataset_snapshot_section,
    render_nerd_mode_insights,
    render_preview_and_commit,
)


SectionSurface = Callable[[Optional[str]], ContextManager[DeltaGenerator]]
RenderToggle = Callable[..., bool]


def render_data_stage(
    *,
    section_surface: SectionSurface,
    render_nerd_mode_toggle: RenderToggle,
) -> None:
    """Render the Prepare/Data stage surface."""

    state_root = ensure_state()
    ss: SessionState = st.session_state
    validate_invariants(state_root)

    data_state = state_root.setdefault("data", {})
    split_state = state_root.setdefault("split", {})
    model_state = state_root.setdefault("model", {})
    ui_state = state_root.setdefault("ui", {})
    data_state.setdefault("params", {})

    def _drain_toasts() -> None:
        toast_messages = list(ui_state.get("toasts") or [])
        if not toast_messages:
            return
        for toast in toast_messages:
            if not isinstance(toast, str):
                continue
            lowered = toast.lower()
            if any(keyword in lowered for keyword in ("warn", "caution", "risk", "error")):
                st.warning(toast)
            else:
                st.info(toast)
        ui_state["toasts"] = []

    _drain_toasts()

    def _invalidate_downstream() -> None:
        model_state["status"] = "stale"
        model_state["metrics"] = {}
        split_state.update(
            {
                "X_train": None,
                "X_test": None,
                "y_train": None,
                "y_test": None,
                "hash": "",
            }
        )

    def _prepare_records(records: Any, *, invalidate: bool = True) -> None:
        prepare_records(
            records,
            data_state=data_state,
            state_root=state_root,
            drain_toasts=_drain_toasts,
            invalidate_callback=_invalidate_downstream,
            invalidate=invalidate,
        )

    if data_state.get("prepared") is None:
        _prepare_records(ss.get("labeled"), invalidate=False)

    def _render_prepare_terminal(slot: DeltaGenerator) -> None:
        with slot:
            render_prepare_terminal()

    current_summary = compute_dataset_summary(ss["labeled"])
    ss["dataset_summary"] = current_summary

    builder_result: Optional[DatasetBuilderResult] = None
    nerd_mode_data_enabled = bool(ss.get("nerd_mode_data"))

    def _render_prepare(slot: DeltaGenerator) -> None:
        nonlocal nerd_mode_data_enabled
        nerd_mode_data_enabled = render_prepare_panel(
            slot=slot,
            section_surface=section_surface,
            render_nerd_mode_toggle=render_nerd_mode_toggle,
        )

    def _render_builder(slot: DeltaGenerator) -> None:
        nonlocal builder_result, current_summary
        builder_result = render_dataset_builder(
            slot=slot,
            section_surface=section_surface,
            ss=ss,
            current_summary=current_summary,
            nerd_mode_data_enabled=nerd_mode_data_enabled,
            prepare_records_callback=_prepare_records,
        )
        current_summary = builder_result.current_summary

    render_stage_top_grid(
        "data",
        left_renderer=_render_prepare_terminal,
        right_first_renderer=_render_prepare,
        right_second_renderer=_render_builder,
    )

    if builder_result is None:
        builder_result = DatasetBuilderResult(
            current_summary=current_summary,
            compare_panel_html="",
        )
    else:
        current_summary = builder_result.current_summary
    ss["dataset_summary"] = current_summary
    compare_panel_html = builder_result.compare_panel_html

    flash_queue = ss.pop("data_stage_flash_queue", [])
    for flash in flash_queue:
        if not isinstance(flash, dict):
            continue
        message = str(flash.get("message", "")).strip()
        if not message:
            continue
        level = flash.get("level", "info")
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        elif level == "error":
            st.error(message)
        else:
            st.info(message)

    preview_summary_for_health = ss.get("dataset_preview_summary")
    lint_counts_preview = (
        ss.get("dataset_preview_lint") if preview_summary_for_health is not None else None
    )
    dataset_generated_once = bool(ss.get("dataset_has_generated_once"))

    render_dataset_health_section(
        section_surface=section_surface,
        compare_panel_html=compare_panel_html,
        preview_summary_for_health=preview_summary_for_health,
        lint_counts_preview=lint_counts_preview,
        dataset_generated_once=dataset_generated_once,
    )

    render_pii_cleanup(section_surface=section_surface, ss=ss)

    render_preview_and_commit(
        section_surface=section_surface,
        ss=ss,
        prepare_records_callback=_prepare_records,
    )

    render_dataset_snapshot_section(
        section_surface=section_surface,
        ss=ss,
        dataset_generated_once=dataset_generated_once,
        preview_summary_for_health=preview_summary_for_health,
        prepare_records_callback=_prepare_records,
    )

    render_nerd_mode_insights(
        section_surface=section_surface,
        ss=ss,
        enabled=nerd_mode_data_enabled,
    )

    render_csv_upload(
        ss=ss,
        prepare_records_callback=_prepare_records,
        enabled=nerd_mode_data_enabled,
    )
