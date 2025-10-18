"""Dataset manipulation helpers for the Prepare stage.

This module centralises operations that mutate dataset-related session
state or transform uploaded data prior to rendering within the Streamlit
UI. The orchestration layer in :mod:`pages.data` delegates to these
helpers so that the main page module can focus on layout and user
interactions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, MutableMapping, Optional

import pandas as pd
import streamlit as st

from demistifai.core.cache import cached_prepare
from demistifai.core.state import hash_dataframe, hash_dict, validate_invariants
from demistifai.core.validation import VALID_LABELS, _normalize_label, _validate_csv_schema
from demistifai.dataset import (
    DEFAULT_DATASET_CONFIG,
    DatasetConfig,
    build_dataset_from_config,
    compute_dataset_summary,
    dataset_delta_story,
    dataset_summary_delta,
    explain_config_change,
    lint_dataset,
    starter_dataset_copy,
)


SessionState = MutableMapping[str, Any]


def _coerce_session_state(ss: Optional[SessionState] = None) -> SessionState:
    """Return a mutable mapping representing Streamlit's session state."""

    if ss is None:
        ss = st.session_state
    return ss


def clear_dataset_preview_state(ss: Optional[SessionState] = None) -> None:
    """Remove transient dataset preview keys from session state."""

    state = _coerce_session_state(ss)
    state["dataset_preview"] = None
    state["dataset_preview_config"] = None
    state["dataset_preview_summary"] = None
    state["dataset_preview_lint"] = None
    state["dataset_manual_queue"] = None
    state["dataset_controls_open"] = False


def discard_preview(ss: Optional[SessionState] = None) -> None:
    """Discard the generated dataset preview and reset comparison context."""

    state = _coerce_session_state(ss)
    clear_dataset_preview_state(state)
    state["dataset_compare_delta"] = None
    state["last_dataset_delta_story"] = explain_config_change(
        state.get("dataset_config", DEFAULT_DATASET_CONFIG)
    )


def prepare_records(
    records: Any,
    *,
    data_state: MutableMapping[str, Any],
    state_root: MutableMapping[str, Any],
    drain_toasts: Callable[[], None],
    invalidate_callback: Optional[Callable[[], None]] = None,
    invalidate: bool = True,
) -> None:
    """Prepare raw dataset records and update cached hashes.

    Parameters mirror the previous inline helper to keep side effects scoped to
    the calling context. Any UI feedback (toasts, errors) is still surfaced via
    Streamlit, but state mutation and validation occur here to keep the page
    module lean.
    """

    params = dict(data_state.get("params") or {})
    data_state["params"] = params

    if invalidate and invalidate_callback:
        invalidate_callback()

    if records is None:
        data_state["raw"] = None
        data_state["prepared"] = pd.DataFrame()
        data_state["n_rows"] = 0
        data_state["hash"] = ""
        validate_invariants(state_root)
        drain_toasts()
        return

    if isinstance(records, pd.DataFrame):
        raw_df = records.copy(deep=True)
    else:
        try:
            raw_df = pd.DataFrame(records)
        except Exception as exc:  # pragma: no cover - defensive conversion
            st.error(f"Failed to coerce dataset into a dataframe: {exc}")
            return

    try:
        prepared_df = cached_prepare(
            raw_df,
            params,
            lib_versions={"pandas": pd.__version__},
        )
    except Exception as exc:  # pragma: no cover - defensive for preprocessing issues
        st.error(f"Failed to prepare dataset: {exc}")
        return

    data_state["raw"] = raw_df.copy(deep=True)
    data_state["prepared"] = prepared_df.copy(deep=True)
    data_state["n_rows"] = int(prepared_df.shape[0])
    data_state["hash"] = hash_dict(
        {
            "prep": hash_dataframe(prepared_df),
            "params": params,
        }
    )
    validate_invariants(state_root)
    drain_toasts()


def generate_preview_from_config(
    config: DatasetConfig,
    *,
    current_summary: Dict[str, Any],
    ss: Optional[SessionState] = None,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], str]:
    """Build a dataset preview for the provided configuration."""

    state = _coerce_session_state(ss)
    dataset_rows = build_dataset_from_config(config)
    preview_summary = compute_dataset_summary(dataset_rows)
    lint_counts = lint_dataset(dataset_rows)

    manual_df = pd.DataFrame(dataset_rows[: min(len(dataset_rows), 200)])
    if not manual_df.empty:
        manual_df.insert(0, "include", True)

    state["dataset_preview"] = dataset_rows
    state["dataset_preview_config"] = config
    state["dataset_preview_summary"] = preview_summary
    state["dataset_preview_lint"] = lint_counts
    state["dataset_manual_queue"] = manual_df
    delta_summary = dataset_summary_delta(current_summary, preview_summary)
    state["dataset_compare_delta"] = delta_summary
    delta_story = dataset_delta_story(delta_summary)
    state["last_dataset_delta_story"] = delta_story
    state["dataset_has_generated_once"] = True

    st.success(
        "Dataset generated — scroll to **Review** and curate data before committing."
    )
    return preview_summary, delta_summary, delta_story


def reset_dataset_to_baseline(ss: Optional[SessionState] = None) -> Dict[str, Any]:
    """Reset the labeled dataset and related session state to the starter set."""

    state = _coerce_session_state(ss)
    state["labeled"] = starter_dataset_copy()
    state["dataset_config"] = DEFAULT_DATASET_CONFIG.copy()
    baseline_summary = compute_dataset_summary(state["labeled"])
    state["dataset_summary"] = baseline_summary
    state["previous_dataset_summary"] = None
    state["dataset_compare_delta"] = None
    state["last_dataset_delta_story"] = None
    state["active_dataset_snapshot"] = None
    state["dataset_snapshot_name"] = ""
    state["dataset_last_built_at"] = datetime.now().isoformat(timespec="seconds")
    clear_dataset_preview_state(state)
    return baseline_summary


@dataclass
class CsvUploadResult:
    """Outcome of preparing a CSV upload for import."""

    dataframe: Optional[pd.DataFrame]
    lint_counts: Dict[str, int]
    dropped_total: int
    drop_reasons: List[str]
    initial_rows: int
    error: Optional[str] = None

    @property
    def reason_text(self) -> str:
        """Human-readable summary of drop reasons."""

        if not self.drop_reasons:
            return "—"
        return "; ".join(self.drop_reasons)


def prepare_uploaded_csv(
    uploaded_file: Any,
    *,
    existing_rows: List[Dict[str, Any]],
    max_rows: int = 2000,
) -> CsvUploadResult:
    """Validate and clean a CSV upload for incorporation into the dataset."""

    if uploaded_file is None:
        return CsvUploadResult(
            dataframe=None,
            lint_counts={},
            dropped_total=0,
            drop_reasons=[],
            initial_rows=0,
            error="No file provided",
        )

    try:
        df_up = pd.read_csv(uploaded_file)
    except Exception as exc:  # pragma: no cover - file parsing can vary
        return CsvUploadResult(
            dataframe=None,
            lint_counts={},
            dropped_total=0,
            drop_reasons=[],
            initial_rows=0,
            error=f"Failed to read CSV: {exc}",
        )

    df_up.columns = [c.strip().lower() for c in df_up.columns]
    schema_ok, schema_message = _validate_csv_schema(df_up)
    if not schema_ok:
        return CsvUploadResult(
            dataframe=None,
            lint_counts={},
            dropped_total=0,
            drop_reasons=[],
            initial_rows=len(df_up),
            error=schema_message,
        )

    if len(df_up) > max_rows:
        return CsvUploadResult(
            dataframe=None,
            lint_counts={},
            dropped_total=0,
            drop_reasons=[],
            initial_rows=len(df_up),
            error=f"Too many rows (max {max_rows:,}). Trim the file and retry.",
        )

    initial_rows = len(df_up)
    df_up["label"] = df_up["label"].apply(_normalize_label)
    invalid_mask = ~df_up["label"].isin(VALID_LABELS)
    dropped_invalid = int(invalid_mask.sum())
    df_up = df_up[~invalid_mask]

    for col in ["title", "body"]:
        df_up[col] = df_up[col].fillna("").astype(str).str.strip()

    length_mask = (df_up["title"].str.len() <= 200) & (df_up["body"].str.len() <= 2000)
    dropped_length = int(len(df_up) - length_mask.sum())
    df_up = df_up[length_mask]

    nonempty_mask = (df_up["title"] != "") | (df_up["body"] != "")
    dropped_empty = int(len(df_up) - nonempty_mask.sum())
    df_up = df_up[nonempty_mask]

    dropped_dupes = 0
    if existing_rows:
        df_existing = pd.DataFrame(existing_rows)
        if not df_existing.empty:
            len_before_duplicates = len(df_up)
            merged = df_up.merge(
                df_existing,
                on=["title", "body", "label"],
                how="left",
                indicator=True,
            )
            df_up = merged[merged["_merge"] == "left_only"].loc[
                :, ["title", "body", "label"]
            ]
            dropped_dupes = int(max(0, len_before_duplicates - len(df_up)))

    total_dropped = max(0, initial_rows - len(df_up))
    drop_reasons: List[str] = []
    if dropped_invalid:
        drop_reasons.append(
            f"{dropped_invalid} invalid label{'s' if dropped_invalid != 1 else ''}"
        )
    if dropped_length:
        drop_reasons.append(f"{dropped_length} over length limit")
    if dropped_empty:
        drop_reasons.append(f"{dropped_empty} blank title/body")
    if dropped_dupes:
        drop_reasons.append(f"{dropped_dupes} duplicates vs session")

    lint_counts = lint_dataset(df_up.to_dict(orient="records")) if len(df_up) else {}

    return CsvUploadResult(
        dataframe=df_up,
        lint_counts=lint_counts,
        dropped_total=total_dropped,
        drop_reasons=drop_reasons,
        initial_rows=initial_rows,
    )


__all__ = [
    "CsvUploadResult",
    "clear_dataset_preview_state",
    "discard_preview",
    "generate_preview_from_config",
    "prepare_records",
    "prepare_uploaded_csv",
    "reset_dataset_to_baseline",
]

