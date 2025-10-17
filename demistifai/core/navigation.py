"""Navigation helpers for stage selection and synchronization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any, Optional, cast

import streamlit as st

from demistifai.constants import STAGES, STAGE_BY_KEY


@dataclass(frozen=True)
class StageSelectionResult:
    """Result of synchronizing the active stage across state containers."""

    active_stage: Optional[str]
    changed: bool
    toast_message: Optional[str] = None


def _first_query_param(name: str) -> Optional[str]:
    values = st.query_params.get_all(name)
    return values[0] if values else None


def synchronize_stage_selection(
    *,
    state: MutableMapping[str, object],
    session_state: MutableMapping[str, object],
    stages: Iterable[object] = STAGES,
    stage_by_key: Mapping[str, object] = STAGE_BY_KEY,
) -> StageSelectionResult:
    """Resolve the active stage and propagate it to state containers."""

    run_state = cast(MutableMapping[str, Any], state.setdefault("run", {}))
    ui_state = cast(MutableMapping[str, Any], state.setdefault("ui", {}))
    toasts = cast(list[str], ui_state.setdefault("toasts", []))

    requested_stage = _first_query_param("stage")
    if requested_stage in stage_by_key and requested_stage != run_state.get("active_stage"):
        run_state["active_stage"] = requested_stage

    stage_keys = [stage.key for stage in stages if hasattr(stage, "key")]
    default_stage = run_state.get("active_stage")
    if default_stage not in stage_by_key:
        default_stage = stage_keys[0] if stage_keys else None
        run_state["active_stage"] = default_stage

    previous_session_stage = session_state.get("active_stage")
    selected_stage = previous_session_stage if previous_session_stage in stage_by_key else default_stage

    if selected_stage is None and default_stage is not None:
        selected_stage = default_stage

    busy = bool(run_state.get("busy"))
    toast_message: Optional[str] = None
    if (
        busy
        and default_stage in stage_by_key
        and selected_stage not in (None, default_stage)
    ):
        selected_stage = default_stage
        toast_message = "Training in progress — navigation disabled."
        if toast_message not in toasts:
            toasts.append(toast_message)

    final_stage = selected_stage if selected_stage in stage_by_key else default_stage
    if final_stage is None and stage_keys:
        final_stage = stage_keys[0]

    changed = final_stage != previous_session_stage

    run_state["active_stage"] = final_stage
    session_state["active_stage"] = final_stage
    if changed and final_stage is not None:
        session_state["stage_scroll_to_top"] = True

    if final_stage and st.query_params.get_all("stage") != [final_stage]:
        st.query_params["stage"] = final_stage

    return StageSelectionResult(active_stage=final_stage, changed=bool(changed), toast_message=toast_message)


def activate_stage(stage_key: str) -> bool:
    """Set the requested stage as active and synchronize navigation affordances."""

    if stage_key not in STAGE_BY_KEY:
        return False

    state = st.session_state.get("demai")
    if not isinstance(state, MutableMapping):
        state = {}
        st.session_state["demai"] = state

    run_state = cast(MutableMapping[str, Any], state.setdefault("run", {}))
    current_stage = run_state.get("active_stage")
    if current_stage == stage_key:
        return False

    run_state["active_stage"] = stage_key
    st.session_state["active_stage"] = stage_key
    st.session_state["stage_scroll_to_top"] = True
    if st.query_params.get_all("stage") != [stage_key]:
        st.query_params["stage"] = stage_key

    return True
