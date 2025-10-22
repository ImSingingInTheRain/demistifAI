"""Session state initialization helpers for the Streamlit app."""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any, Callable, Iterable, MutableMapping, Set

from demistifai.constants import AUTONOMY_LEVELS
from demistifai.core.state import _set_advanced_knob_state
from demistifai.dataset import (
    DEFAULT_DATASET_CONFIG,
    compute_dataset_summary,
    generate_incoming_batch,
    starter_dataset_copy,
)
from demistifai.modeling import FEATURE_ORDER


def _ensure_key(
    ss: MutableMapping[str, Any],
    key: str,
    value: Any,
    created: Set[str],
) -> Any:
    if key not in ss:
        ss[key] = value
        created.add(key)
    return ss[key]


def _ensure_from_factory(
    ss: MutableMapping[str, Any],
    key: str,
    factory: Callable[[], Any],
    created: Set[str],
) -> Any:
    if key not in ss:
        ss[key] = factory()
        created.add(key)
    return ss[key]


def initialize_session_defaults(ss: MutableMapping[str, Any]) -> Iterable[str]:
    """Populate Streamlit session state with baseline defaults."""

    created: Set[str] = set()

    _ensure_key(ss, "viewport_is_mobile", False, created)
    _ensure_key(ss, "nerd_mode", False, created)
    _ensure_key(ss, "autonomy", AUTONOMY_LEVELS[0], created)
    _ensure_key(ss, "threshold", 0.6, created)
    _ensure_key(ss, "nerd_mode_eval", False, created)
    _ensure_key(ss, "eval_timestamp", None, created)
    _ensure_key(ss, "eval_temp_threshold", float(ss.get("threshold", 0.6)), created)
    adaptive_default = bool(ss.get("adaptive", True))
    _ensure_key(ss, "adaptive", adaptive_default, created)
    _ensure_key(ss, "use_adaptiveness", adaptive_default, created)

    labeled = _ensure_from_factory(ss, "labeled", starter_dataset_copy, created)

    _ensure_key(ss, "incoming_seed", None, created)
    if not ss.get("incoming"):
        seed = random.randint(1, 1_000_000)
        ss["incoming_seed"] = seed
        ss["incoming"] = generate_incoming_batch(n=30, seed=seed, spam_ratio=0.32)
        created.add("incoming")

    _ensure_key(ss, "model", None, created)
    _ensure_key(ss, "split_cache", None, created)
    _ensure_key(ss, "mail_inbox", [], created)
    _ensure_key(ss, "mail_spam", [], created)
    _ensure_key(ss, "metrics", {"TP": 0, "FP": 0, "TN": 0, "FN": 0}, created)
    _ensure_key(ss, "last_classification", None, created)
    _ensure_key(
        ss,
        "numeric_adjustments",
        {feat: 0.0 for feat in FEATURE_ORDER},
        created,
    )
    _ensure_key(ss, "nerd_mode_data", False, created)
    _ensure_key(ss, "nerd_mode_train", False, created)
    _ensure_key(ss, "intro_show_mission", False, created)
    _ensure_key(ss, "calibrate_probabilities", False, created)
    _ensure_key(ss, "calibration_result", None, created)
    _ensure_key(
        ss,
        "train_params",
        {"test_size": 0.30, "random_state": 42, "max_iter": 1000, "C": 1.0},
        created,
    )
    _ensure_key(
        ss,
        "guard_params",
        {
            "assist_center": float(ss.get("threshold", 0.6)),
            "uncertainty_band": 0.08,
            "numeric_scale": 0.5,
            "numeric_logit_cap": 1.0,
            "combine_strategy": "blend",
            "shift_suspicious_tld": -0.04,
            "shift_many_links": -0.03,
            "shift_calm_text": 0.02,
        },
        created,
    )
    _ensure_key(ss, "use_high_autonomy", ss.get("autonomy", AUTONOMY_LEVELS[0]).startswith("High"), created)
    _ensure_key(ss, "train_story_run_id", None, created)
    _ensure_key(ss, "use_batch_results", [], created)
    _ensure_key(ss, "use_audit_log", [], created)
    _ensure_key(ss, "nerd_mode_use", False, created)

    dataset_config = _ensure_key(ss, "dataset_config", DEFAULT_DATASET_CONFIG.copy(), created)
    _set_advanced_knob_state(dataset_config, force=False)

    if "dataset_summary" not in ss:
        ss["dataset_summary"] = compute_dataset_summary(labeled)
        created.add("dataset_summary")

    _ensure_key(
        ss,
        "dataset_last_built_at",
        datetime.now().isoformat(timespec="seconds"),
        created,
    )
    _ensure_key(ss, "previous_dataset_summary", None, created)
    _ensure_key(ss, "dataset_preview", None, created)
    _ensure_key(ss, "dataset_preview_config", None, created)
    _ensure_key(ss, "dataset_preview_summary", None, created)
    _ensure_key(ss, "dataset_manual_queue", None, created)
    _ensure_key(ss, "dataset_controls_open", False, created)
    _ensure_key(ss, "dataset_has_generated_once", False, created)
    _ensure_key(ss, "datasets", [], created)
    _ensure_key(ss, "active_dataset_snapshot", None, created)
    _ensure_key(ss, "dataset_snapshot_name", "", created)
    _ensure_key(ss, "last_dataset_delta_story", None, created)
    _ensure_key(ss, "dataset_compare_delta", None, created)
    _ensure_key(ss, "dataset_preview_lint", None, created)
    _ensure_key(ss, "last_eval_results", None, created)

    return sorted(created)
