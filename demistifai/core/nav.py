from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import html
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

# Expect STAGES, STAGE_BY_KEY in app scope; import if you have a module for them.
try:
    from demistifai.core.constants import STAGES, STAGE_BY_KEY  # type: ignore
except Exception:
    STAGES, STAGE_BY_KEY = [], {}

StageBlockRenderer = Callable[[DeltaGenerator], None]

@dataclass
class StageTopGridSlots:
    left: DeltaGenerator
    right_primary: DeltaGenerator
    right_secondary: DeltaGenerator
    prev_clicked: bool
    next_clicked: bool

def _stage_navigation_context(stage_key: str):
    stage_keys = [stage.key for stage in STAGES]
    total = len(stage_keys)
    if total == 0 or stage_key not in STAGE_BY_KEY:
        raise ValueError("Stage navigation requested for unknown stage.")
    index = stage_keys.index(stage_key)
    stage = STAGE_BY_KEY[stage_key]
    prev_stage = STAGE_BY_KEY.get(stage_keys[index - 1]) if index > 0 else None
    next_stage = STAGE_BY_KEY.get(stage_keys[index + 1]) if index < total - 1 else None
    return index, total, stage, prev_stage, next_stage

def _render_stage_navigation_panel(stage_key: str, card_slot: DeltaGenerator, next_slot: DeltaGenerator, prev_slot: DeltaGenerator):
    try:
        index, total, stage, prev_stage, next_stage = _stage_navigation_context(stage_key)
    except ValueError:
        return False, False

    html_card = f"""
      <div class="stage-top-grid__nav-card">
        <div class="stage-top-grid__nav-card-header">
          <span class="stage-top-grid__nav-prompt">$ stage.status</span>
          <span class="stage-top-grid__nav-stage">[{index+1}/{total}]</span>
        </div>
        <div class="stage-top-grid__nav-title">
          <span class="stage-top-grid__nav-icon">{html.escape(stage.icon)}</span>
          <span>{html.escape(stage.title)}</span>
        </div>
        <p class="stage-top-grid__nav-description">{html.escape(stage.description)}</p>
      </div>
    """
    with card_slot:
        st.markdown(html_card, unsafe_allow_html=True)

    next_label = "Proceed" if next_stage is None else f"{next_stage.icon} {next_stage.title} ➡️"
    prev_label = "Back" if prev_stage is None else f"⬅️ {prev_stage.icon} {prev_stage.title}"

    sentinel_next_id = f"demai-stage-nav-next-sentinel-{stage_key}"
    sentinel_prev_id = f"demai-stage-nav-prev-sentinel-{stage_key}"

    next_clicked = next_slot.button(
        next_label, key=f"stage_grid_next_{stage_key}",
        use_container_width=True, type="primary", disabled=next_stage is None, help="Jump to the next stage"
    )
    if next_clicked and next_stage is not None:
        st.session_state["active_stage"] = next_stage.key
        st.session_state["stage_scroll_to_top"] = True
        st.query_params["stage"] = next_stage.key
        from demistifai.core.utils import streamlit_rerun
        streamlit_rerun()

    next_slot.markdown(
        f"""
        <div id="{sentinel_next_id}" data-stage-key="{html.escape(stage_key)}" data-demai-target="demai-stage-nav-next-btn"></div>
        """,
        unsafe_allow_html=True,
    )

    prev_clicked = prev_slot.button(
        prev_label, key=f"stage_grid_prev_{stage_key}",
        use_container_width=True, disabled=prev_stage is None, help="Return to the previous stage"
    )
    if prev_clicked and prev_stage is not None:
        st.session_state["active_stage"] = prev_stage.key
        st.session_state["stage_scroll_to_top"] = True
        st.query_params["stage"] = prev_stage.key
        from demistifai.core.utils import streamlit_rerun
        streamlit_rerun()

    prev_slot.markdown(
        f"""
        <div id="{sentinel_prev_id}" data-stage-key="{html.escape(stage_key)}" data-demai-target="demai-stage-nav-prev-btn"></div>
        """,
        unsafe_allow_html=True,
    )

    return prev_clicked, next_clicked

def render_stage_top_grid(stage_key: str, *, left_renderer: StageBlockRenderer | None = None,
                          right_first_renderer: StageBlockRenderer | None = None,
                          right_second_renderer: StageBlockRenderer | None = None) -> StageTopGridSlots:
    grid_container = st.container()
    left_col, right_col = grid_container.columns([0.65, 0.35], gap="large")
    left_slot = left_col.container()
    nav_slot = right_col.container()
    right_col.markdown("<div class='stage-top-grid__gap'></div>", unsafe_allow_html=True)
    right_first_slot = right_col.container()
    right_col.markdown("<div class='stage-top-grid__gap'></div>", unsafe_allow_html=True)
    right_second_slot = right_col.container()
    right_col.markdown("<div class='stage-top-grid__gap'></div>", unsafe_allow_html=True)
    next_slot = right_col.container()
    right_col.markdown("<div class='stage-top-grid__gap'></div>", unsafe_allow_html=True)
    prev_slot = right_col.container()

    prev_clicked, next_clicked = _render_stage_navigation_panel(stage_key, nav_slot, next_slot, prev_slot)

    if left_renderer is not None:
        left_renderer(left_slot)
    else:
        left_slot.markdown(
            """
            <div class="stage-top-grid__placeholder">
              <strong>Stage content placeholder</strong>
              Customize this area with rich content unique to the current stage.
            </div>
            """, unsafe_allow_html=True
        )

    if right_first_renderer is not None:
        right_first_renderer(right_first_slot)
    else:
        right_first_slot.markdown(
            """
            <div class="stage-top-grid__placeholder stage-top-grid__placeholder--compact">
              Right column placeholder — add supplemental callouts or metrics here.
            </div>
            """, unsafe_allow_html=True
        )

    if right_second_renderer is not None:
        right_second_renderer(right_second_slot)
    else:
        right_second_slot.markdown(
            """
            <div class="stage-top-grid__placeholder stage-top-grid__placeholder--compact">
              Secondary placeholder ready for per-stage widgets or notes.
            </div>
            """, unsafe_allow_html=True
        )

    return StageTopGridSlots(
        left=left_slot, right_primary=right_first_slot, right_secondary=right_second_slot,
        prev_clicked=prev_clicked, next_clicked=next_clicked
    )
