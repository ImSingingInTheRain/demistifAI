"""Review, snapshot, and diagnostics helpers for the Prepare stage."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from datetime import datetime
from typing import Any, Callable, ContextManager, Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from demistifai.constants import CLASSES
from demistifai.core.dataset import _evaluate_dataset_health
from demistifai.core.embeddings import _compute_cached_embeddings
from demistifai.core.pii import format_pii_summary, pii_chip_row_html
from demistifai.core.state import _push_data_stage_flash, _set_advanced_knob_state
from demistifai.core.utils import (
    _caps_ratio,
    _count_money_mentions,
    _count_suspicious_links,
    streamlit_rerun,
)
from demistifai.core.validation import VALID_LABELS
from demistifai.dataset import (
    DEFAULT_DATASET_CONFIG,
    build_dataset_from_config,
    compute_dataset_hash,
    compute_dataset_summary,
    dataset_delta_story,
    dataset_summary_delta,
    lint_dataset,
)
from demistifai.modeling import combine_text
from demistifai.ui.components.data_review import (
    data_review_styles,
    dataset_balance_bar_html,
    dataset_snapshot_active_badge,
    dataset_snapshot_card_html,
    dataset_snapshot_styles,
    edge_case_pairs_html,
    stratified_sample_cards_html,
)

from pages.data.dataset_io import (
    SessionState,
    clear_dataset_preview_state,
    discard_preview,
    prepare_uploaded_csv,
)


SectionSurface = Callable[[Optional[str]], ContextManager[DeltaGenerator]]
PrepareRecordsCallback = Callable[[Any], None]


def render_dataset_health_section(
    *,
    section_surface: SectionSurface,
    compare_panel_html: str,
    preview_summary_for_health: Optional[Dict[str, Any]],
    lint_counts_preview: Optional[Dict[str, int]],
    dataset_generated_once: bool,
) -> None:
    """Render the dataset health summary and comparison panel."""

    if not (dataset_generated_once or preview_summary_for_health is not None):
        return

    spam_pct: Optional[float] = None
    total_rows: Optional[float] = None
    lint_label = ""
    badge_text = ""
    lint_chip_html = ""
    dataset_health_available = False

    if preview_summary_for_health is not None:
        health = _evaluate_dataset_health(preview_summary_for_health, lint_counts_preview)
        spam_pct = health["spam_pct"]
        total_rows = health["total_rows"]
        lint_label = health["lint_label"]
        badge_text = health["badge_text"]
        lint_flags_total = health["lint_flags"]
        if lint_label == "Unknown":
            lint_icon = "ℹ️"
        elif lint_flags_total:
            lint_icon = "⚠️"
        else:
            lint_icon = "🛡️"
        lint_chip_html = (
            "<span class='lint-chip'><span class='lint-chip__icon'>{icon}</span>"
            "<span class='lint-chip__text'>Personal data alert: {label}</span></span>"
        ).format(icon=lint_icon, label=html.escape(lint_label or "Unknown"))
        dataset_health_available = True

    with section_surface():
        health_col, compare_col = st.columns([1.4, 1], gap="large")
        with health_col:
            st.markdown(
                "#### Dataset health",
                help="Quick pulse on balance, volume, and lint signals for the generated preview.",
            )
            if dataset_health_available:
                if spam_pct is not None and total_rows is not None:
                    safe_pct = max(0.0, min(100.0, 100.0 - spam_pct))
                    total_display = (
                        int(total_rows) if isinstance(total_rows, (int, float)) else total_rows
                    )
                    st.markdown(
                        """
                        <div class="dataset-health-card">
                            <div class="dataset-health-panel">
                                <div class="dataset-health-panel__status">
                                    <div class="dataset-health-panel__status-copy">
                                        <small>Preview summary</small>
                                        <h5>Balance &amp; lint</h5>
                                    </div>
                                    <div class="dataset-health-status dataset-health-status--{status_class}">
                                        <span class="dataset-health-status__dot"></span>
                                        <span>{status_text}</span>
                                    </div>
                                </div>
                                <div class="dataset-health-panel__row dataset-health-panel__row--bar">
                                    <div class="dataset-health-panel__bar">
                                        <span class="dataset-health-panel__bar-spam" style="width: {spam_width}%"></span>
                                        <span class="dataset-health-panel__bar-safe" style="width: {safe_width}%"></span>
                                    </div>
                                </div>
                                <div class="dataset-health-panel__row dataset-health-panel__row--meta">
                                    <div class="dataset-health-panel__meta-primary">
                                        <span>Spam {spam_pct:.0f}%</span>
                                        <span>Safe {safe_pct:.0f}%</span>
                                        <span>Rows {total_rows}</span>
                                    </div>
                                    {lint_section}
                                </div>
                            </div>
                        </div>
                        """.format(
                            status_class="healthy" if lint_label == "Good" else "warning",
                            status_text=html.escape(badge_text or "Healthy mix"),
                            spam_width=f"{spam_pct:.1f}" if spam_pct is not None else "0",
                            safe_width=f"{safe_pct:.1f}" if spam_pct is not None else "0",
                            spam_pct=spam_pct or 0.0,
                            safe_pct=safe_pct,
                            total_rows=html.escape(str(total_display)),
                            lint_section=lint_chip_html,
                        ),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Dataset summary not available.")
            else:
                st.caption("Generate a preview to evaluate dataset health.")
        with compare_col:
            st.markdown(compare_panel_html, unsafe_allow_html=True)


def render_preview_and_commit(
    *,
    section_surface: SectionSurface,
    ss: SessionState,
    prepare_records_callback: PrepareRecordsCallback,
) -> None:
    """Render the preview review cards and commit actions."""

    if not ss.get("dataset_preview"):
        return

    with section_surface():
        st.markdown("### Review & approve")
        preview_summary = ss.get("dataset_preview_summary") or compute_dataset_summary(
            ss["dataset_preview"]
        )
        lint_counts = ss.get("dataset_preview_lint") or {
            "credit_card": 0,
            "iban": 0,
            "email": 0,
            "phone": 0,
            "otp6": 0,
            "url": 0,
        }

        st.markdown(data_review_styles(), unsafe_allow_html=True)

        kpi_col, sample_col, edge_col = st.columns([1.1, 1.2, 1.1], gap="large")

        with kpi_col:
            st.write("**KPIs**")
            st.metric("Preview rows", preview_summary.get("total", 0))
            spam_ratio = preview_summary.get("spam_ratio", 0.0)
            st.metric("Spam share", f"{spam_ratio * 100:.1f}%")
            st.metric(
                "Avg suspicious links (spam)",
                f"{preview_summary.get('avg_susp_links', 0.0):.2f}",
            )

            st.markdown(dataset_balance_bar_html(spam_ratio), unsafe_allow_html=True)

            chip_html = pii_chip_row_html(lint_counts, extra_class="pii-chip-row--compact")
            if chip_html:
                st.markdown(chip_html, unsafe_allow_html=True)
            st.caption("Guardrail: no live link fetching, HTML escaped, duplicates dropped.")

        with sample_col:
            st.write("**Stratified sample**")
            preview_rows = ss.get("dataset_preview", [])
            spam_examples = [row for row in preview_rows if row.get("label") == "spam"]
            safe_examples = [row for row in preview_rows if row.get("label") == "safe"]
            max_cards = 4
            cards: List[Dict[str, str]] = []
            idx_spam = idx_safe = 0
            for i in range(max_cards):
                if i % 2 == 0:
                    if idx_spam < len(spam_examples):
                        cards.append(spam_examples[idx_spam])
                        idx_spam += 1
                    elif idx_safe < len(safe_examples):
                        cards.append(safe_examples[idx_safe])
                        idx_safe += 1
                else:
                    if idx_safe < len(safe_examples):
                        cards.append(safe_examples[idx_safe])
                        idx_safe += 1
                    elif idx_spam < len(spam_examples):
                        cards.append(spam_examples[idx_spam])
                        idx_spam += 1
                if len(cards) >= max_cards:
                    break
            if not cards:
                st.info("Preview examples will appear here once generated.")
            else:
                st.markdown(
                    stratified_sample_cards_html(cards),
                    unsafe_allow_html=True,
                )

        with edge_col:
            st.write("**Edge-case pairs**")
            preview_config = ss.get("dataset_preview_config", {})
            if preview_config.get("edge_cases", 0) <= 0:
                st.caption("Add edge cases in the builder to surface look-alike contrasts here.")
            else:
                by_title: Dict[str, Dict[str, Dict[str, str]]] = {}
                for row in ss.get("dataset_preview", []):
                    title = (row.get("title", "") or "").strip()
                    label = row.get("label", "")
                    if not title or label not in VALID_LABELS:
                        continue
                    by_title.setdefault(title, {})[label] = row
                pairs = [
                    (data.get("spam"), data.get("safe"))
                    for data in by_title.values()
                    if data.get("spam") and data.get("safe")
                ]
                if not pairs:
                    st.info("No contrasting pairs surfaced yet — regenerate to refresh examples.")
                else:
                    st.markdown(
                        edge_case_pairs_html(pairs[:3]),
                        unsafe_allow_html=True,
                    )

        preview_rows_full: List[Dict[str, Any]] = ss.get("dataset_preview") or []
        manual_df = ss.get("dataset_manual_queue")
        if not isinstance(manual_df, pd.DataFrame) or len(manual_df) != len(preview_rows_full):
            manual_df = pd.DataFrame(preview_rows_full)
        else:
            manual_df = manual_df.copy()
        if not manual_df.empty:
            if "include" not in manual_df.columns:
                manual_df.insert(0, "include", True)
            else:
                manual_df["include"] = manual_df["include"].fillna(True)
        elif preview_rows_full:
            manual_df = pd.DataFrame(preview_rows_full)
            if not manual_df.empty and "include" not in manual_df.columns:
                manual_df.insert(0, "include", True)
        with st.expander(
            "Expand this section if you want to manually review and edit individual emails part of the dataset"
        ):
            edited_df = st.data_editor(
                manual_df,
                width="stretch",
                hide_index=True,
                key="dataset_manual_editor",
                column_config={
                    "include": st.column_config.CheckboxColumn(
                        "Include?", help="Uncheck to drop before committing."
                    ),
                    "label": st.column_config.SelectboxColumn(
                        "Label", options=sorted(VALID_LABELS)
                    ),
                },
            )
        ss["dataset_manual_queue"] = edited_df
        st.caption(
            "Manual queue covers the entire preview — re-run the builder to generate more variations."
        )

    edited_df_for_commit = ss.get("dataset_manual_queue")

    with section_surface():
        st.markdown("### Commit dataset")
        st.markdown("<div class='cta-sticky'>", unsafe_allow_html=True)
        commit_col, discard_col, _ = st.columns([1, 1, 2])

        if commit_col.button("Commit dataset", type="primary", use_container_width=True):
            preview_rows_commit = ss.get("dataset_preview")
            config = ss.get(
                "dataset_preview_config",
                ss.get("dataset_config", DEFAULT_DATASET_CONFIG),
            )
            if not preview_rows_commit:
                st.error("Generate a preview before committing.")
            else:
                edited_records: List[Dict[str, Any]] = []
                if isinstance(edited_df_for_commit, pd.DataFrame):
                    edited_records = edited_df_for_commit.to_dict(orient="records")
                preview_copy = [dict(row) for row in preview_rows_commit]
                for idx, record in enumerate(edited_records):
                    if idx >= len(preview_copy):
                        break
                    preview_copy[idx]["title"] = str(
                        record.get("title", preview_copy[idx].get("title", ""))
                    )
                    preview_copy[idx]["body"] = str(
                        record.get("body", preview_copy[idx].get("body", ""))
                    )
                    preview_copy[idx]["label"] = record.get(
                        "label", preview_copy[idx].get("label", "spam")
                    )
                    preview_copy[idx]["include"] = bool(
                        record.get("include", True)
                    )
                final_rows: List[Dict[str, str]] = []
                for idx, row in enumerate(preview_copy):
                    include_flag = row.pop("include", True)
                    if idx < len(edited_records):
                        include_flag = bool(edited_records[idx].get("include", include_flag))
                    if not include_flag:
                        continue
                    final_rows.append(
                        {
                            "title": row.get("title", "").strip(),
                            "body": row.get("body", "").strip(),
                            "label": row.get("label", "spam"),
                        }
                    )
                if len(final_rows) < 10:
                    st.warning("Need at least 10 rows to maintain a meaningful dataset.")
                else:
                    previous_summary = ss.get("dataset_summary", {})
                    previous_config = ss.get("dataset_config", DEFAULT_DATASET_CONFIG)
                    lint_counts_commit = lint_dataset(final_rows) or {}
                    new_summary = compute_dataset_summary(final_rows)
                    delta = dataset_summary_delta(previous_summary, new_summary)
                    ss["previous_dataset_summary"] = previous_summary
                    ss["dataset_summary"] = new_summary
                    ss["dataset_config"] = config
                    ss["dataset_compare_delta"] = delta
                    ss["last_dataset_delta_story"] = dataset_delta_story(delta)
                    ss["labeled"] = final_rows
                    ss["active_dataset_snapshot"] = None
                    ss["dataset_snapshot_name"] = ""
                    ss["dataset_last_built_at"] = datetime.now().isoformat(timespec="seconds")
                    clear_dataset_preview_state()
                    health_evaluation = _evaluate_dataset_health(
                        new_summary, lint_counts_commit
                    )
                    spam_ratio_commit = new_summary.get("spam_ratio") or 0.0
                    spam_share_pct_commit = max(
                        0.0, min(100.0, float(spam_ratio_commit) * 100.0)
                    )
                    committed_rows = new_summary.get("total", len(final_rows))
                    try:
                        committed_display = int(committed_rows)
                    except (TypeError, ValueError):
                        committed_display = len(final_rows)
                    health_token = health_evaluation.get("health_emoji") or "—"
                    summary_line = (
                        f"Committed {committed_display} rows • "
                        f"Spam share {spam_share_pct_commit:.1f}% • Health: {health_token}"
                    )

                    prev_spam_ratio = (
                        previous_summary.get("spam_ratio") if previous_summary else None
                    )
                    spam_delta_pp = 0.0
                    if prev_spam_ratio is not None:
                        try:
                            spam_delta_pp = (
                                float(spam_ratio_commit) - float(prev_spam_ratio)
                            ) * 100.0
                        except (TypeError, ValueError):
                            spam_delta_pp = 0.0
                    prev_edge_cases = previous_config.get("edge_cases", 0)
                    new_edge_cases = config.get("edge_cases", prev_edge_cases)
                    if spam_delta_pp >= 10.0:
                        hint_line = "Expect recall ↑, precision may ↓; tune threshold in Evaluate."
                    elif new_edge_cases > prev_edge_cases:
                        hint_line = (
                            "Boundary likely sharpened; training may need more iterations."
                        )
                    else:
                        hint_line = (
                            "Mix steady; train to refresh the model, then tune the threshold in Evaluate."
                        )

                    _push_data_stage_flash("success", f"{summary_line}\n{hint_line}")
                    if any(lint_counts_commit.values()):
                        _push_data_stage_flash(
                            "warning",
                            "Lint warnings persist after commit ({}).".format(
                                format_pii_summary(lint_counts_commit)
                            ),
                        )

                    _set_advanced_knob_state(config, force=True)
                    prepare_records_callback(final_rows)
                    if ss.get("_needs_advanced_knob_rerun"):
                        streamlit_rerun()

        if discard_col.button("Discard preview", type="secondary", use_container_width=True):
            discard_preview()
            st.info("Preview cleared. The active labeled dataset remains unchanged.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_dataset_snapshot_section(
    *,
    section_surface: SectionSurface,
    ss: SessionState,
    dataset_generated_once: bool,
    preview_summary_for_health: Optional[Dict[str, Any]],
    prepare_records_callback: PrepareRecordsCallback,
) -> None:
    """Render dataset snapshot management controls."""

    if not (dataset_generated_once or preview_summary_for_health is not None):
        return

    with section_surface():
        st.markdown("### Dataset snapshot")
        current_config = ss.get("dataset_config", DEFAULT_DATASET_CONFIG)
        config_json = json.dumps(current_config, indent=2, sort_keys=True)
        current_summary = ss.get("dataset_summary") or compute_dataset_summary(
            ss.get("labeled", [])
        )
        st.caption("Save immutable snapshots to reference in the model card and audits.")

        st.markdown(dataset_snapshot_styles(), unsafe_allow_html=True)

        total_rows = current_summary.get("total") if isinstance(current_summary, dict) else None
        if total_rows is None:
            total_rows = len(ss.get("labeled", []))
        rows_display = total_rows
        try:
            rows_display = int(total_rows)
        except (TypeError, ValueError):
            rows_display = total_rows or "—"

        spam_ratio_val = None
        if isinstance(current_summary, dict):
            spam_ratio_val = current_summary.get("spam_ratio")
        if spam_ratio_val is None:
            spam_share_display = "—"
        else:
            try:
                spam_share_display = f"{float(spam_ratio_val) * 100:.1f}%"
            except (TypeError, ValueError):
                spam_share_display = "—"

        edge_cases = current_config.get("edge_cases") if isinstance(current_config, dict) else None
        edge_display = None
        if edge_cases is not None:
            try:
                edge_display = int(edge_cases)
            except (TypeError, ValueError):
                edge_display = edge_cases

        labeled_rows = ss.get("labeled", [])
        try:
            snapshot_hash = compute_dataset_hash(labeled_rows)[:10]
        except Exception:
            snapshot_hash = "—"
        timestamp_display = ss.get("dataset_last_built_at") or "—"
        seed_value = current_config.get("seed") if isinstance(current_config, dict) else None
        seed_display = seed_value if seed_value is not None else "—"

        summary_rows = [
            ("Rows", rows_display),
            ("Spam share", spam_share_display),
        ]
        if edge_display is not None:
            summary_rows.append(("Edge cases", edge_display))
        fingerprint_rows = [
            ("Short hash", snapshot_hash),
            ("Timestamp", timestamp_display),
            ("Seed", seed_display),
        ]

        card_html = dataset_snapshot_card_html(summary_rows, fingerprint_rows)
        st.markdown(card_html, unsafe_allow_html=True)

        with st.expander("View JSON", expanded=False):
            st.json(json.loads(config_json))
        ss["dataset_snapshot_name"] = st.text_input(
            "Snapshot name",
            value=ss.get("dataset_snapshot_name", ""),
            help="Describe the scenario (e.g., 'High links, 5% noise').",
        )
        if st.button("Save dataset snapshot", key="save_dataset_snapshot"):
            snapshot_id = compute_dataset_hash(ss["labeled"])
            entry = {
                "id": snapshot_id,
                "name": ss.get("dataset_snapshot_name")
                or f"snapshot-{len(ss['datasets'])+1}",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "config": ss.get("dataset_config", DEFAULT_DATASET_CONFIG),
                "config_json": config_json,
                "rows": len(ss["labeled"]),
            }
            existing = next((snap for snap in ss["datasets"] if snap.get("id") == snapshot_id), None)
            if existing:
                existing.update(entry)
            else:
                ss["datasets"].append(entry)
            ss["active_dataset_snapshot"] = snapshot_id
            st.success(f"Snapshot saved with id `{snapshot_id[:10]}…`. Use it in the model card.")

        if ss.get("datasets"):
            st.markdown("#### Saved snapshots")
            header_cols = st.columns([3, 2.2, 2.2, 1.4, 1.2])
            for col, label in zip(header_cols, ["Name", "Fingerprint", "Saved", "Rows", ""]):
                col.markdown(f"**{label}**")

            datasets_sorted = sorted(
                ss["datasets"],
                key=lambda snap: snap.get("timestamp", ""),
                reverse=True,
            )
            for idx, snap in enumerate(datasets_sorted):
                name_col, fp_col, ts_col, rows_col, action_col = st.columns([3, 2.2, 2.2, 1.4, 1.2])
                is_active = snap.get("id") == ss.get("active_dataset_snapshot")
                name_value = snap.get("name") or "(unnamed snapshot)"
                badge_html = dataset_snapshot_active_badge(is_active)
                name_col.markdown(
                    "<div class='dataset-snapshot-name'><span class='dataset-snapshot-name__text'>{}</span>{}</div>".format(
                        html.escape(name_value),
                        badge_html,
                    ),
                    unsafe_allow_html=True,
                )

                snapshot_id = snap.get("id", "")
                short_id = "—"
                if snapshot_id:
                    short_id = snapshot_id if len(snapshot_id) <= 10 else f"{snapshot_id[:10]}…"
                fp_col.markdown(f"`{short_id}`")
                ts_col.markdown(snap.get("timestamp", "—"))
                rows_col.markdown(str(snap.get("rows", "—")))

                button_label = "Set active"
                button_disabled = is_active
                button_key = f"set_active_snapshot_{idx}_{snapshot_id[:6]}"
                if action_col.button(button_label, key=button_key, disabled=button_disabled):
                    config = snap.get("config")
                    if isinstance(config, str) and config:
                        try:
                            config = json.loads(config)
                        except json.JSONDecodeError:
                            config = None
                    if not isinstance(config, dict):
                        st.error("Snapshot is missing a valid configuration.")
                    else:
                        dataset_rows = build_dataset_from_config(config)
                        summary = compute_dataset_summary(dataset_rows)
                        lint_counts = lint_dataset(dataset_rows) or {}
                        ss["labeled"] = dataset_rows
                        ss["dataset_config"] = config
                        ss["dataset_summary"] = summary
                        ss["previous_dataset_summary"] = None
                        ss["dataset_compare_delta"] = None
                        ss["last_dataset_delta_story"] = None
                        ss["dataset_snapshot_name"] = snap.get("name", "")
                        ss["active_dataset_snapshot"] = snap.get("id")
                        ss["dataset_last_built_at"] = datetime.now().isoformat(timespec="seconds")
                        clear_dataset_preview_state()
                        _push_data_stage_flash(
                            "success",
                            f"Snapshot '{snap.get('name', 'snapshot')}' activated. Dataset rebuilt with {len(dataset_rows)} rows.",
                        )
                        if any(lint_counts.values()):
                            _push_data_stage_flash(
                                "warning",
                                "Lint warnings present in restored snapshot ({}).".format(
                                    format_pii_summary(lint_counts)
                                ),
                            )
                        _set_advanced_knob_state(config, force=True)
                        prepare_records_callback(dataset_rows)
                        if ss.get("_needs_advanced_knob_rerun"):
                            streamlit_rerun()
        else:
            st.caption("No snapshots yet. Save one after curating your first dataset.")


def render_nerd_mode_insights(
    *,
    section_surface: SectionSurface,
    ss: SessionState,
    enabled: bool,
) -> None:
    """Render diagnostics available when Nerd Mode is enabled."""

    if not enabled:
        return

    with section_surface():
        st.markdown("### Nerd Mode insights")
        df_lab = pd.DataFrame(ss["labeled"])
        if df_lab.empty:
            st.info("Label some emails or import data to unlock diagnostics.")
            return

        diagnostics_df = df_lab.head(500).copy()
        st.caption(f"Diagnostics sample: {len(diagnostics_df)} emails (cap 500).")

        st.markdown("#### Feature distributions by class")
        feature_records: List[Dict[str, Any]] = []
        for _, row in diagnostics_df.iterrows():
            label = row.get("label")
            if label not in VALID_LABELS:
                continue
            title = row.get("title", "") or ""
            body = row.get("body", "") or ""
            feature_records.append(
                {
                    "label": label,
                    "suspicious_links": float(_count_suspicious_links(body)),
                    "caps_ratio": float(_caps_ratio(f"{title} {body}")),
                    "money_mentions": float(_count_money_mentions(body)),
                }
            )
        feature_df = pd.DataFrame(feature_records)
        if feature_df.empty or feature_df["label"].nunique() < 2 or feature_df.groupby("label").size().min() < 3:
            st.caption("Need at least 3 emails per class to chart numeric feature distributions.")
        else:
            feature_specs = [
                ("suspicious_links", "Suspicious links per email"),
                ("caps_ratio", "ALL-CAPS ratio"),
                ("money_mentions", "Money & urgency mentions"),
            ]
            for feature_key, feature_label in feature_specs:
                sub_df = feature_df.loc[:, ["label", feature_key]].rename(columns={feature_key: "value"})
                sub_df["value"] = pd.to_numeric(sub_df["value"], errors="coerce")
                base_chart = (
                    alt.Chart(sub_df.dropna())
                    .mark_bar(opacity=0.75)
                    .encode(
                        alt.X("value:Q", bin=alt.Bin(maxbins=10), title="Feature value"),
                        alt.Y("count()", title="Count"),
                        alt.Color(
                            "label:N",
                            scale=alt.Scale(domain=tuple(CLASSES), range=["#ef4444", "#1d4ed8"]),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("label:N", title="Class"),
                            alt.Tooltip("value:Q", bin=alt.Bin(maxbins=10), title="Feature value"),
                            alt.Tooltip("count()", title="Count"),
                        ],
                    )
                    .properties(height=220)
                )
                chart = (
                    base_chart
                    .facet(column=alt.Column("label:N", title=None))
                    .resolve_scale(y="independent")
                    .properties(title=feature_label)
                )
                st.altair_chart(chart, use_container_width=True)

        st.markdown("#### Near-duplicate check")
        embed_sample = diagnostics_df.head(min(len(diagnostics_df), 200)).copy()
        if embed_sample.empty or embed_sample["label"].nunique() < 2:
            st.caption("Need both classes present to run the near-duplicate check.")
            embeddings = np.empty((0, 0), dtype=np.float32)
            embed_error = None
        else:
            embed_sample = embed_sample.sample(frac=1.0, random_state=42).reset_index(drop=True)
            sample_records = embed_sample.to_dict(orient="records")
            sample_hash = compute_dataset_hash(sample_records)
            texts = tuple(
                combine_text(rec.get("title", ""), rec.get("body", ""))
                for rec in sample_records
            )
            embed_error = None
            try:
                embeddings = _compute_cached_embeddings(sample_hash, texts)
            except Exception as exc:  # pragma: no cover - defensive for encoder availability
                embeddings = np.empty((0, 0), dtype=np.float32)
                embed_error = str(exc)

        if embed_error:
            st.warning(f"Embedding diagnostics unavailable: {embed_error}")
        elif embeddings.size < 4:
            st.caption("Not enough samples for similarity analysis (need at least two per class).")
        else:
            st.caption(f"Near-duplicate scan on {embeddings.shape[0]} emails (cap 200).")
            sims = embeddings @ embeddings.T
            np.fill_diagonal(sims, -1.0)
            labels = embed_sample["label"].tolist()
            top_pairs: List[tuple[float, int, int]] = []
            n_rows = sims.shape[0]
            for i in range(n_rows):
                for j in range(i + 1, n_rows):
                    if labels[i] == labels[j]:
                        continue
                    sim_val = float(sims[i, j])
                    if sim_val > 0.9:
                        top_pairs.append((sim_val, i, j))
            top_pairs.sort(key=lambda tup: tup[0], reverse=True)
            top_pairs = top_pairs[:5]
            if not top_pairs:
                st.caption("No high-similarity cross-label pairs detected in the sampled set.")
            else:
                for sim_val, idx_a, idx_b in top_pairs:
                    row_a = embed_sample.iloc[idx_a]
                    row_b = embed_sample.iloc[idx_b]
                    st.markdown(
                        f"**Similarity {sim_val:.2f}** — {row_a.get('label', '').title()} vs {row_b.get('label', '').title()}"
                    )
                    pair_cols = st.columns(2)
                    for col, row in zip(pair_cols, [row_a, row_b]):
                        with col:
                            st.caption(row.get("label", "").title() or "Unknown")
                            st.write(f"**{row.get('title', '(untitled)')}**")
                            body_text = (row.get("body", "") or "").replace("\n", " ")
                            excerpt = body_text[:160] + ("…" if len(body_text) > 160 else "")
                            st.write(excerpt)

        st.markdown("#### Class prototypes")
        if embed_error:
            st.caption("Embeddings unavailable — prototypes skipped.")
        elif embeddings.size == 0:
            st.caption("Need labeled examples to compute prototypes.")
        else:
            proto_cols = st.columns(2)
            for col, label in zip(proto_cols, CLASSES):
                with col:
                    mask = embed_sample["label"] == label
                    idxs = np.where(mask.to_numpy())[0]
                    if idxs.size == 0:
                        st.caption(f"No {label} examples in the sample.")
                        continue
                    centroid = embeddings[idxs].mean(axis=0)
                    norm = float(np.linalg.norm(centroid))
                    if norm == 0.0:
                        st.caption("Centroid not informative for this class.")
                        continue
                    centroid /= norm
                    sims_label = embeddings[idxs] @ centroid
                    top_local = idxs[np.argsort(-sims_label)[:3]]
                    st.markdown(f"**{label.title()} archetype**")
                    for rank, idx_point in enumerate(top_local, start=1):
                        row = embed_sample.iloc[idx_point]
                        body_text = (row.get("body", "") or "").replace("\n", " ")
                        excerpt = body_text[:140] + ("…" if len(body_text) > 140 else "")
                        st.caption(f"{rank}. {row.get('title', '(untitled)')}")
                        st.write(excerpt)

        st.divider()
        st.markdown("#### Quick lexical snapshot")
        tokens_spam = Counter()
        tokens_safe = Counter()
        for _, row in df_lab.iterrows():
            text = f"{row.get('title', '')} {row.get('body', '')}".lower()
            tokens = re.findall(r"[a-zA-Z']+", text)
            if row.get("label") == "spam":
                tokens_spam.update(tokens)
            else:
                tokens_safe.update(tokens)
        top_spam = tokens_spam.most_common(12)
        top_safe = tokens_safe.most_common(12)
        col_tok1, col_tok2 = st.columns(2)
        with col_tok1:
            st.markdown("**Class token cloud — Spam**")
            st.write(", ".join(f"{w} ({c})" for w, c in top_spam) or "—")
        with col_tok2:
            st.markdown("**Class token cloud — Safe**")
            st.write(", ".join(f"{w} ({c})" for w, c in top_safe) or "—")

        title_groups: Dict[str, set] = {}
        for _, row in df_lab.iterrows():
            title = row.get("title", "").strip().lower()
            label = row.get("label")
            title_groups.setdefault(title, set()).add(label)
        leakage_titles = [title for title, labels in title_groups.items() if len(labels) > 1 and title]
        if leakage_titles:
            st.caption(
                "Duplicate subject lines with conflicting labels: {}".format(
                    ", ".join(sorted(leakage_titles[:6]))
                )
            )

        strat_df = df_lab.groupby("label").size().reset_index(name="count")
        st.dataframe(strat_df, hide_index=True, width="stretch")


def render_csv_upload(
    *,
    ss: SessionState,
    prepare_records_callback: PrepareRecordsCallback,
    enabled: bool,
) -> None:
    """Render the CSV upload expander when Nerd Mode is enabled."""

    if not enabled:
        return

    with st.expander("📤 Upload CSV of labeled emails (strict schema)", expanded=False):
        st.caption(
            "Schema: title, body, label (spam|safe). Limits: ≤2,000 rows, title ≤200 chars, body ≤2,000 chars."
        )
        st.caption("Uploaded data stays in this session only. No emails are sent or fetched.")
        up = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader_labeled")
        if up is not None:
            result = prepare_uploaded_csv(up, existing_rows=ss["labeled"], max_rows=2000)
            if result.error:
                st.error(result.error)
            elif result.dataframe is None:
                st.warning("Uploaded file contains no rows after validation.")
            else:
                df_up = result.dataframe
                lint_counts = result.lint_counts
                st.caption(
                    "Rows dropped: {} (reason: {})".format(
                        result.dropped_total, result.reason_text
                    )
                )
                st.dataframe(df_up.head(20), hide_index=True, width="stretch")
                st.caption(
                    "Rows passing validation: {} | Lint -> {}".format(
                        len(df_up), format_pii_summary(lint_counts)
                    )
                )
                if len(df_up) > 0 and st.button(
                    "Import into labeled dataset", key="btn_import_csv"
                ):
                    ss["labeled"].extend(df_up.to_dict(orient="records"))
                    ss["dataset_summary"] = compute_dataset_summary(ss["labeled"])
                    prepare_records_callback(ss.get("labeled"))
                    st.success(
                        "Imported {} rows into labeled dataset. Revisit builder to rebalance if needed.".format(
                            len(df_up)
                        )
                    )
