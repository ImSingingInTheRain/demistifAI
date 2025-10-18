from __future__ import annotations

import time
from typing import Any, Callable, Optional

import streamlit as st

from .guardrails import (
    _ghost_meaning_map_enhanced,
    _numeric_guardrails_caption_text,
)
from .meaning_map import (
    _build_meaning_map_chart,
    _conceptual_meaning_sketch,
    _meaning_map_zoom_subset,
    _prepare_meaning_map,
)
from .numeric_clues import _render_numeric_clue_cards, _render_numeric_clue_preview


def _render_training_examples_preview() -> None:
    """Placeholder hook for future training examples preview."""


def _render_unified_training_storyboard(
    ss,
    *,
    has_model: bool,
    has_split: bool,
    has_embed: bool,
    section_surface: Callable[[], Any],
    request_meaning_map_refresh: Callable[[Optional[str]], None],
    parse_split_cache: Callable[[Any], tuple],
    rerun: Callable[[], None],
    logger,
) -> None:
    meaning_map_df, meaning_map_meta, chart_ready, meaning_map_error = None, {}, False, None
    if has_model and has_split and has_embed:
        try:
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = parse_split_cache(ss["split_cache"])
            meaning_map_df, meaning_map_meta = _prepare_meaning_map(
                list(X_tr_t) if X_tr_t else [],
                list(X_tr_b) if X_tr_b else [],
                list(y_tr) if y_tr else [],
                ss.get("model"),
            )
            gp = ss.get("guard_params", {}) or {}
            center_default = float(gp.get("assist_center", float(ss.get("threshold", 0.6))))
            band_default = float(gp.get("uncertainty_band", 0.08))
            meaning_map_meta.setdefault("guard_center_probability", center_default)
            meaning_map_meta.setdefault("guard_margin_probability", band_default)
            c = float(meaning_map_meta["guard_center_probability"])
            b = float(meaning_map_meta["guard_margin_probability"])
            meaning_map_meta.setdefault("guard_window_low", max(0.0, c - b))
            meaning_map_meta.setdefault("guard_window_high", min(1.0, c + b))
            chart_ready = meaning_map_df is not None and not meaning_map_df.empty
        except Exception as exc:
            meaning_map_error = f"Meaning Map unavailable ({exc})."
            meaning_map_df, meaning_map_meta = None, {}
            chart_ready = False

    ss["train_storyboard_payload"] = {
        "meaning_map_df": meaning_map_df if chart_ready else None,
        "meaning_map_meta": meaning_map_meta if meaning_map_meta else {},
        "meaning_map_error": meaning_map_error,
        "chart_ready": chart_ready,
    }

    refresh_expected = bool(ss.get("train_refresh_expected"))
    refresh_attempts = int(ss.get("train_refresh_attempts", 0) or 0)
    if refresh_expected:
        if chart_ready:
            ss["train_refresh_expected"] = False
            ss["train_refresh_attempts"] = 0
        elif refresh_attempts > 0:
            ss["train_refresh_attempts"] = refresh_attempts - 1
            logger.debug(
                "Meaning map still warming after training; scheduling auto-rerun (%s attempts left)",
                refresh_attempts - 1,
            )
            st.caption("Refreshing training meaning maps…")
            time.sleep(0.35)
            rerun()
            return
        else:
            ss["train_refresh_expected"] = False
            ss["train_refresh_attempts"] = 0

    with section_surface():
        st.markdown("### How training works — live storyboard")

        left1, right1 = st.columns([0.48, 0.52], gap="large")
        with left1:
            pre_training = not (has_model and has_split)
            show_examples = False
            if pre_training:
                st.markdown(
                    "**1) Meaning points**  \n"
                    "Think of each email as a dot placed by MiniLM in a 2-D meaning space:\n\n"
                    "- **Meaning dimension 1**: a blend of traits such as *formality ↔ salesy tone*, *generic phrasing ↔ brand-specific details*, or *routine biz-speak ↔ promotional language*.\n"
                    "- **Meaning dimension 2**: a complementary blend such as *calm, factual ↔ urgent, persuasive*, *low CTA ↔ strong CTA*, or *neutral ↔ attention-grabbing*.\n\n"
                    "Below is a **conceptual meaning map** (an illustrative example shown *before* training). "
                    "Instead of dots, it uses chip-style cards to preview likely clusters from this dataset — "
                    "meeting notes, courier tracking, promotional blasts, security alerts, and other themes — each placed where that tone usually lands. "
                    "This helps you imagine the semantic neighborhoods MiniLM will uncover and why borderline content often sits near the middle. "
                    "After you train on your labeled emails, this sketch will be replaced by the **real** map built from your data."
                )
            else:
                st.markdown("**1) Meaning points**")
                st.markdown(
                    "MiniLM places each email so **similar wording lands close**. "
                    "These two axes are the first two directions of that meaning space — "
                    "_meaning dimension 1_ and _meaning dimension 2_."
                )
                if has_model and has_split:
                    if st.button(
                        "Refresh chart",
                        key=f"refresh_meaning_map_section1_{ss.get('train_story_run_id') or 'initial'}",
                    ):
                        request_meaning_map_refresh("section1")

        with right1:
            pre_training = not (has_model and has_split)
            story_run_id = ss.get("train_story_run_id") or "initial"
            if pre_training:
                try:
                    chart = _conceptual_meaning_sketch()
                except Exception:
                    st.info("Illustrative map unavailable (Altair missing)")
                else:
                    st.altair_chart(
                        chart,
                        use_container_width=True,
                        key=f"conceptual_meaning_sketch_{story_run_id}",
                    )
                _render_training_examples_preview()
            elif meaning_map_error:
                st.info(meaning_map_error)
                _render_training_examples_preview()
            elif chart_ready:
                chart1 = _build_meaning_map_chart(
                    meaning_map_df,
                    meaning_map_meta,
                    show_examples=show_examples,
                    show_class_centers=False,
                    highlight_borderline=False,
                )
                if chart1 is not None:
                    st.altair_chart(
                        chart1,
                        use_container_width=True,
                        key=f"meaning_map_chart1_live_{story_run_id}",
                    )
                else:
                    st.info("Meaning Map unavailable for this run.")
                _render_training_examples_preview()
            else:
                st.altair_chart(
                    _ghost_meaning_map_enhanced(
                        ss,
                        title="Meaning map (schematic)",
                        show_divider=False,
                        show_band=False,
                    ),
                    use_container_width=True,
                    key=f"ghost_live_map1_{story_run_id}",
                )
                _render_training_examples_preview()

        left2, right2 = st.columns([0.48, 0.52], gap="large")
        with left2:
            st.markdown("**2) A simple dividing line**")
            st.markdown(
                "A straight line will separate spam from safe. Before training, we show where the **current configuration** "
                "would place that dividing line conceptually. After training, you’ll see the actual learned boundary in a "
                "zoomed-in view that spotlights borderline emails and those closest to each class center."
            )
            show_centers = bool(has_model and has_split)
            if has_model and has_split:
                if st.button(
                    "Refresh chart",
                    key=f"refresh_meaning_map_section2_{story_run_id}",
                ):
                    request_meaning_map_refresh("section2")

        with right2:
            if meaning_map_error:
                st.info(meaning_map_error)
            elif chart_ready:
                zoom_df = _meaning_map_zoom_subset(meaning_map_df, meaning_map_meta)
                chart2 = _build_meaning_map_chart(
                    zoom_df if zoom_df is not None else meaning_map_df,
                    meaning_map_meta,
                    show_examples=False,
                    show_class_centers=show_centers,
                    highlight_borderline=False,
                )
                if chart2 is not None:
                    st.altair_chart(
                        chart2,
                        use_container_width=True,
                        key=f"meaning_map_chart2_live_{story_run_id}",
                    )
                else:
                    st.info("Meaning Map unavailable for this run.")
            else:
                st.altair_chart(
                    _ghost_meaning_map_enhanced(
                        ss,
                        title="Dividing line (schematic)",
                        show_divider=True,
                        show_band=False,
                    ),
                    use_container_width=True,
                    key=f"ghost_live_map2_{story_run_id}",
                )

        guard_params = ss.get("guard_params", {}) if hasattr(ss, "get") else {}
        try:
            assist_center = float(guard_params.get("assist_center", ss.get("threshold", 0.6)))
        except Exception:
            assist_center = float(ss.get("threshold", 0.6))
        try:
            uncertainty_band = float(guard_params.get("uncertainty_band", 0.08))
        except Exception:
            uncertainty_band = 0.08

        left3, right3 = st.columns([0.48, 0.52], gap="large")
        with left3:
            st.markdown("**3) Extra clues when unsure**")
            st.markdown(
                f"When the text score is near **τ≈{assist_center:.2f}**, numeric guardrails help out within a small window "
                f"(**±{uncertainty_band:.2f}**) — we look at links, ALL-CAPS, money or urgency hints."
            )
            guard_low = None
            guard_high = None
            if isinstance(meaning_map_meta, dict):
                guard_low = meaning_map_meta.get("guard_window_low")
                guard_high = meaning_map_meta.get("guard_window_high")
            if chart_ready and isinstance(guard_low, (int, float)) and isinstance(guard_high, (int, float)):
                st.caption(
                    f"Showing emails where the spam probability falls between {guard_low:.2f} and {guard_high:.2f}."
                )
            else:
                st.caption("Train the model to see which emails triggered these numeric clues.")
            if has_model and has_split:
                if st.button(
                    "Refresh chart",
                    key=f"refresh_meaning_map_section3_{story_run_id}",
                ):
                    request_meaning_map_refresh("section3")

        with right3:
            if meaning_map_error:
                st.info(meaning_map_error)
            elif chart_ready:
                _render_numeric_clue_cards(meaning_map_df)
            else:
                _render_numeric_clue_preview(assist_center, uncertainty_band)

        st.caption(_numeric_guardrails_caption_text(ss))
