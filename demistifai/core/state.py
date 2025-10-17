from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
import hashlib
import json
from typing import Any, Dict

import streamlit as st
from streamlit.errors import StreamlitAPIException

from demistifai.dataset import (
    ATTACHMENT_MIX_PRESETS,
    DEFAULT_ATTACHMENT_MIX,
    DEFAULT_DATASET_CONFIG,
    DatasetConfig,
)

import pandas as pd


def _set_advanced_knob_state(
    config: Dict[str, Any] | DatasetConfig,
    *,
    force: bool = False,
) -> None:
    """Ensure Nerd Mode advanced knob widgets reflect the active configuration."""

    if not isinstance(config, dict):
        config = DEFAULT_DATASET_CONFIG

    try:
        links_level = int(str(config.get("susp_link_level", "1")))
    except (TypeError, ValueError):
        links_level = 1
    tld_level = str(config.get("susp_tld_level", "med"))
    caps_level = str(config.get("caps_intensity", "med"))
    money_level = str(config.get("money_urgency", "low"))
    current_mix = config.get("attachments_mix", DEFAULT_ATTACHMENT_MIX)
    attachment_choice = next(
        (name for name, mix in ATTACHMENT_MIX_PRESETS.items() if mix == current_mix),
        "Balanced",
    )
    try:
        noise_pct = float(config.get("label_noise_pct", 0.0))
    except (TypeError, ValueError):
        noise_pct = 0.0
    try:
        seed_value = int(config.get("seed", 42))
    except (TypeError, ValueError):
        seed_value = 42
    poison_demo = bool(config.get("poison_demo", False))

    adv_state = {
        "adv_links_level": links_level,
        "adv_tld_level": tld_level,
        "adv_caps_level": caps_level,
        "adv_money_level": money_level,
        "adv_attachment_choice": attachment_choice,
        "adv_label_noise_pct": noise_pct,
        "adv_seed": seed_value,
        "adv_poison_demo": poison_demo,
    }

    for key, value in adv_state.items():
        if force or key not in st.session_state:
            try:
                st.session_state[key] = value
            except StreamlitAPIException:
                pending = st.session_state.setdefault("_pending_advanced_knob_state", {})
                pending[key] = value
                st.session_state["_needs_advanced_knob_rerun"] = True


def _apply_pending_advanced_knob_state() -> None:
    """Apply any queued advanced knob updates before widgets instantiate."""

    pending = st.session_state.pop("_pending_advanced_knob_state", None)
    if not pending:
        return

    for key, value in pending.items():
        st.session_state[key] = value

    st.session_state.pop("_needs_advanced_knob_rerun", None)


def _push_data_stage_flash(level: str, message: str) -> None:
    """Queue a flash message to render at the top of the data stage."""

    if not message:
        return

    queue = st.session_state.setdefault("data_stage_flash_queue", [])
    queue.append({"level": level, "message": message})


def ensure_state(schema_ver: int = 1) -> dict:
    """Return the demAI session container, creating/migrating it in ``st.session_state``."""

    defaults = _build_default_state(schema_ver)
    state = st.session_state.get("demai")
    if not isinstance(state, dict):
        state = {}

    _merge_state_defaults(state, defaults)

    current_ver = state.get("schema_ver")
    state["schema_ver"] = schema_ver if current_ver is None else max(schema_ver, current_ver)

    st.session_state["demai"] = state
    return state


def validate_invariants(s: dict) -> None:
    """Enforce demAI state invariants, mutating dependent sections and raising UI notices."""

    if not isinstance(s, dict):
        raise TypeError("demai state must be a dict")

    data_state = s.setdefault("data", {})
    split_state = s.setdefault("split", {})
    model_state = s.setdefault("model", {})
    ui_state = s.setdefault("ui", {})
    toasts = ui_state.setdefault("toasts", [])

    data_hash = data_state.get("hash")
    model_hash = model_state.get("data_hash")
    if data_hash and model_hash and data_hash != model_hash:
        if model_state.get("status") != "stale":
            model_state["status"] = "stale"
        _append_toast(toasts, "Model marked stale because the dataset changed.")

    split_hash = split_state.get("hash")
    if split_hash and data_hash and split_hash != data_hash:
        default_split = _build_default_state(s.get("schema_ver", 1))["split"]
        split_state.clear()
        split_state.update(deepcopy(default_split))
        _append_toast(
            toasts,
            "Existing train/validate split was cleared because it no longer matches the dataset.",
        )


def hash_dict(obj: dict) -> str:
    """Produce a deterministic SHA-256 digest for dictionary content."""

    normalized = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def hash_dataframe(df, *, sample_rows: int | None = 1000) -> str:
    """Hash dataframe schema plus a stable sample without mutating the dataframe."""

    if df is None:
        return hashlib.sha256(b"<null-dataframe>").hexdigest()
    if not isinstance(df, pd.DataFrame):
        raise TypeError("hash_dataframe expects a pandas.DataFrame instance")

    total_rows = len(df)
    indices = _select_sample_indices(total_rows, sample_rows)
    sample_df = df.iloc[indices] if indices else df.iloc[[]]

    sample_payload = []
    for idx, row in sample_df.iterrows():
        normalized_row = {"__index__": _normalize_scalar(idx)}
        for col, value in row.items():
            normalized_row[col] = _normalize_scalar(value)
        sample_payload.append(normalized_row)

    payload = {
        "shape": [total_rows, len(df.columns)],
        "columns": list(df.columns),
        "sample_index_count": len(indices),
        "sample": sample_payload,
    }

    return hash_dict(payload)


def _build_default_state(schema_ver: int) -> dict:
    """Create a fresh copy of the default demAI session schema."""

    return {
        "schema_ver": schema_ver,
        "data": {
            "frame": None,
            "hash": None,
            "source": None,
            "metadata": {},
        },
        "split": {
            "hash": None,
            "payload": None,
            "meta": {},
        },
        "model": {
            "status": "idle",
            "artifact": None,
            "data_hash": None,
            "metrics": {},
            "params": {},
        },
        "ui": {
            "toasts": [],
        },
    }


def _merge_state_defaults(target: dict, template: dict) -> None:
    """Recursively populate missing keys from the template without clobbering values."""

    for key, default_value in template.items():
        if key not in target:
            target[key] = deepcopy(default_value)
        else:
            current_value = target[key]
            if isinstance(current_value, dict) and isinstance(default_value, dict):
                _merge_state_defaults(current_value, default_value)


def _append_toast(toasts: list, message: str) -> None:
    """Append a toast message once per session lifecycle."""

    if not message:
        return
    if message not in toasts:
        toasts.append(message)


def _select_sample_indices(total_rows: int, sample_rows: int | None) -> list[int]:
    """Return stable positional indices for dataframe hashing."""

    if total_rows <= 0:
        return []
    if sample_rows is None or sample_rows <= 0 or total_rows <= sample_rows:
        return list(range(total_rows))
    if sample_rows == 1:
        return [0]

    max_index = total_rows - 1
    step = max_index / (sample_rows - 1)
    positions = {0, max_index}
    for i in range(1, sample_rows - 1):
        positions.add(int(round(i * step)))
    return sorted(positions)


def _normalize_scalar(value: Any) -> Any:
    """Convert scalars to JSON-serialisable primitives while preserving information."""

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - fall back to repr for exotic types
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover
            pass
    return value


def _json_default(value: Any) -> Any:
    """Fallback normaliser for json.dumps used in hashing helpers."""

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover
            pass
    return str(value)
