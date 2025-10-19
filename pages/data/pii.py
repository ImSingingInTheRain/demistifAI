"""PII cleanup helpers for the Prepare stage preview."""

from __future__ import annotations

from typing import Callable, ContextManager, Optional

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from demistifai.ui.components import (
    _ensure_pii_state,
    _highlight_spans_html,
    render_pii_cleanup_banner,
    render_pii_indicators,
)
from demistifai.core.pii import format_pii_summary, summarize_pii_counts
from demistifai.core.utils import streamlit_rerun
from demistifai.dataset import lint_dataset, lint_dataset_detailed, lint_text_spans

from pages.data.dataset_io import SessionState


def render_pii_cleanup(
    *,
    section_surface: Callable[[Optional[str]], ContextManager[DeltaGenerator]],
    ss: SessionState,
) -> None:
    """Render the PII cleanup mini-game when a preview exists."""

    if not ss.get("dataset_preview"):
        return

    _ensure_pii_state()
    preview_rows = ss["dataset_preview"]
    detailed_hits = lint_dataset_detailed(preview_rows)
    counts = summarize_pii_counts(detailed_hits)
    ss["pii_hits_map"] = detailed_hits
    flagged_ids = sorted(detailed_hits.keys())
    ss["pii_queue"] = flagged_ids
    ss["pii_total_flagged"] = len(flagged_ids)

    with section_surface():
        banner_clicked = render_pii_cleanup_banner(counts)
        if banner_clicked:
            ss["pii_open"] = True

    if not ss.get("pii_open"):
        return

    with section_surface():
        st.markdown("### 🔐 Personal Data Cleanup")
        remaining_to_clean = len(flagged_ids)
        pii_counts_dict = {
            "Score": int(ss.get("pii_score", 0) or 0),
            "Cleaned": int(ss.get("pii_cleaned_count", 0) or 0),
            "PII to be cleaned": remaining_to_clean,
        }
        render_pii_indicators(counts=pii_counts_dict)

        if not flagged_ids:
            st.success("No PII left to clean in the preview. 🎉")
            return

        idx = int(ss.get("pii_queue_idx", 0))
        if idx >= len(flagged_ids):
            idx = 0
        ss["pii_queue_idx"] = idx
        row_id = flagged_ids[idx]
        row_data = dict(preview_rows[row_id])

        col_editor, col_tokens = st.columns([2.6, 1.2], gap="large")

        with col_editor:
            st.caption("Edit & highlight")
            st.caption(
                f"Cleaning email {idx + 1} of {remaining_to_clean} flagged entries."
            )
            title_spans = ss["pii_hits_map"].get(row_id, {}).get("title", [])
            body_spans = ss["pii_hits_map"].get(row_id, {}).get("body", [])
            edited_values = ss.get("pii_edits", {}).get(row_id, {})
            title_default = edited_values.get("title", row_data.get("title", ""))
            body_default = edited_values.get("body", row_data.get("body", ""))

            title_key = f"pii_title_{row_id}"
            body_key = f"pii_body_{row_id}"
            if title_key not in st.session_state:
                st.session_state[title_key] = title_default
            if body_key not in st.session_state:
                st.session_state[body_key] = body_default

            pending_token = ss.pop("pii_pending_token", None)
            if pending_token:
                if pending_token.get("row_id") == row_id:
                    target_field = pending_token.get("field", "body")
                    target_key = title_key if target_field == "title" else body_key
                    current_text = st.session_state.get(
                        target_key,
                        row_data.get(target_field, ""),
                    )
                    token_text = pending_token.get("token", "")
                    if token_text:
                        spacer = " " if current_text and not current_text.endswith(" ") else ""
                        st.session_state[target_key] = f"{current_text}{spacer}{token_text}"
                else:
                    ss["pii_pending_token"] = pending_token

            st.markdown("**Title (highlighted)**", help="Highlights show detected PII.")
            st.markdown(
                _highlight_spans_html(st.session_state[title_key], title_spans),
                unsafe_allow_html=True,
            )
            st.markdown("**Body (highlighted)**")
            st.markdown(
                _highlight_spans_html(st.session_state[body_key], body_spans),
                unsafe_allow_html=True,
            )

            title_value = st.text_input("✏️ Title (editable)", key=title_key)
            body_value = st.text_area("✏️ Body (editable)", key=body_key, height=180)

        with col_tokens:
            st.caption("Replacements")
            st.write("Click to insert tokens at cursor or paste them:")
            token_columns = st.columns(2)

            def _queue_token(token: str, field: str = "body") -> None:
                ss["pii_pending_token"] = {
                    "row_id": row_id,
                    "field": field,
                    "token": token,
                }
                streamlit_rerun()

            with token_columns[0]:
                if st.button("{{EMAIL}}", key=f"pii_token_email_{row_id}"):
                    _queue_token("{{EMAIL}}")
                if st.button("{{IBAN}}", key=f"pii_token_iban_{row_id}"):
                    _queue_token("{{IBAN}}")
                if st.button("{{CARD_16}}", key=f"pii_token_card_{row_id}"):
                    _queue_token("{{CARD_16}}")
            with token_columns[1]:
                if st.button("{{PHONE}}", key=f"pii_token_phone_{row_id}"):
                    _queue_token("{{PHONE}}")
                if st.button("{{OTP_6}}", key=f"pii_token_otp_{row_id}"):
                    _queue_token("{{OTP_6}}")
                if st.button("{{URL_SUSPICIOUS}}", key=f"pii_token_url_{row_id}"):
                    _queue_token("{{URL_SUSPICIOUS}}")

            st.divider()
            action_columns = st.columns([1.2, 1, 1.2])
            apply_and_next = action_columns[0].button(
                "✅ Apply & Next", key="pii_apply_next", type="primary"
            )
            skip_row = action_columns[1].button("Skip", key="pii_skip")
            finish_cleanup = action_columns[2].button("Finish cleanup", key="pii_finish")

            if apply_and_next:
                ss.setdefault("pii_edits", {}).setdefault(row_id, {})
                ss["pii_edits"][row_id]["title"] = title_value
                ss["pii_edits"][row_id]["body"] = body_value
                relinted_title = lint_text_spans(title_value)
                relinted_body = lint_text_spans(body_value)
                preview_rows[row_id]["title"] = title_value
                preview_rows[row_id]["body"] = body_value
                ss["dataset_preview_lint"] = lint_dataset(preview_rows)
                if not relinted_title and not relinted_body:
                    ss["pii_cleaned_count"] = ss.get("pii_cleaned_count", 0) + 1
                    points = 10
                    ss["pii_score"] = ss.get("pii_score", 0) + points
                    st.toast(f"Clean! +{points} points", icon="🎯")
                    ss["pii_hits_map"].pop(row_id, None)
                    ss["pii_queue"] = [rid for rid in flagged_ids if rid != row_id]
                    flagged_ids = ss["pii_queue"]
                    ss["pii_queue_idx"] = min(idx, max(0, len(flagged_ids) - 1))
                    ss["pii_total_flagged"] = len(flagged_ids)
                else:
                    ss["pii_hits_map"][row_id] = {"title": relinted_title, "body": relinted_body}
                    ss["pii_score"] = max(0, ss.get("pii_score", 0) - 2)
                    st.toast("Still detecting PII — try replacing with tokens.", icon="⚠️")
                streamlit_rerun()

            if skip_row:
                if flagged_ids:
                    ss["pii_queue_idx"] = (idx + 1) % len(flagged_ids)
                streamlit_rerun()

            if finish_cleanup:
                updated_detailed = lint_dataset_detailed(preview_rows)
                new_counts = summarize_pii_counts(updated_detailed)
                ss["pii_hits_map"] = updated_detailed
                ss["pii_queue"] = sorted(updated_detailed.keys())
                ss["pii_total_flagged"] = len(ss["pii_queue"])
                ss["dataset_preview_lint"] = lint_dataset(preview_rows)
                st.success(
                    "Cleanup finished. Remaining hits — {summary}.".format(
                        summary=format_pii_summary(new_counts)
                    )
                )
                ss["pii_open"] = False
                return
