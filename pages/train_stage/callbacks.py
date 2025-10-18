"""Session and modeling callbacks for the Train stage."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import train_test_split

from demistifai.core.cache import cached_features, cached_train
from demistifai.core.state import hash_dict
from demistifai.modeling import cache_train_embeddings, combine_text, encode_texts


def callable_or_attr(target: Any, attr: str | None = None) -> bool:
    """Return ``True`` when ``target`` (or one of its attributes) is callable."""

    try:
        value = getattr(target, attr) if attr else target
    except Exception:
        return False
    return callable(value)


def prime_session_state_from_store(
    ss: st.session_state.__class__,
    model_state: Dict[str, Any],
    split_state: Dict[str, Any],
) -> None:
    """Populate Streamlit session_state with cached model artifacts."""

    model_obj = model_state.get("clf")
    if model_obj is not None:
        current_model = ss.get("model")
        if current_model is None or current_model is model_obj:
            ss["model"] = model_obj

    vectorizer = model_state.get("vectorizer")
    if vectorizer is not None:
        ss["model_vectorizer"] = vectorizer

    x_train_payload = split_state.get("X_train")
    x_test_payload = split_state.get("X_test")
    y_train_payload = split_state.get("y_train")
    y_test_payload = split_state.get("y_test")
    if (
        isinstance(x_train_payload, dict)
        and isinstance(x_test_payload, dict)
        and y_train_payload is not None
        and y_test_payload is not None
        and ss.get("split_cache") is None
    ):
        ss["split_cache"] = (
            list(x_train_payload.get("titles") or []),
            list(x_test_payload.get("titles") or []),
            list(x_train_payload.get("bodies") or []),
            list(x_test_payload.get("bodies") or []),
            list(y_train_payload or []),
            list(y_test_payload or []),
        )


def _as_list(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if hasattr(payload, "tolist"):
        try:
            return payload.tolist()
        except Exception:
            pass
    return list(payload)


def sync_training_artifacts(
    prepared: pd.DataFrame,
    *,
    ss: st.session_state.__class__,
    run_state: Dict[str, Any],
    data_state: Dict[str, Any],
    split_state: Dict[str, Any],
    model_state: Dict[str, Any],
) -> None:
    """Refresh cached splits and model artifacts after training."""

    model_obj = ss.get("model")
    if model_obj is None:
        return

    raw_train_params = dict(ss.get("train_params") or {})
    try:
        test_size = float(raw_train_params.get("test_size", 0.30))
    except (TypeError, ValueError):
        test_size = 0.30
    try:
        random_state = int(raw_train_params.get("random_state", 42))
    except (TypeError, ValueError):
        random_state = 42

    split_params = {
        "test_size": test_size,
        "random_state": random_state,
        "stratify": True,
    }

    try:
        max_iter = int(raw_train_params.get("max_iter", 1000))
    except (TypeError, ValueError):
        max_iter = 1000
    try:
        c_value = float(raw_train_params.get("C", 1.0))
    except (TypeError, ValueError):
        c_value = 1.0

    guard_params = dict(ss.get("guard_params") or {})
    default_center = float(ss.get("threshold", 0.6))
    model_params = {
        "max_iter": max_iter,
        "C": c_value,
        "random_state": random_state,
        "numeric_assist_center": float(guard_params.get("assist_center", default_center)),
        "uncertainty_band": float(guard_params.get("uncertainty_band", 0.08)),
        "numeric_scale": float(guard_params.get("numeric_scale", 0.5)),
        "numeric_logit_cap": float(guard_params.get("numeric_logit_cap", 1.0)),
        "combine_strategy": str(guard_params.get("combine_strategy", "blend")),
        "shift_suspicious_tld": float(guard_params.get("shift_suspicious_tld", -0.04)),
        "shift_many_links": float(guard_params.get("shift_many_links", -0.03)),
        "shift_calm_text": float(guard_params.get("shift_calm_text", +0.02)),
    }

    numeric_adjustments = dict(ss.get("numeric_adjustments", {}))
    model_params_for_cache = dict(model_params)
    model_params_for_cache["numeric_adjustments"] = numeric_adjustments

    data_hash = data_state.get("hash", "")
    split_hash = hash_dict({"data": data_hash, "split_params": split_params})

    stored_model = model_state.get("clf")
    already_synced = (
        split_state.get("hash") == split_hash
        and model_state.get("status") == "trained"
        and model_state.get("data_hash") == data_hash
        and model_state.get("params") == model_params
        and stored_model is not None
        and model_obj is stored_model
    )
    if already_synced:
        prime_session_state_from_store(ss, model_state, split_state)
        return

    has_new_model = stored_model is None or model_obj is not stored_model
    if not has_new_model:
        if model_state.get("status") != "trained":
            return
        return

    run_state["busy"] = True
    try:
        feature_payload, labels = cached_features(
            prepared,
            {
                "title_column": "title",
                "body_column": "body",
                "target_column": "label",
            },
            lib_versions={"pandas": pd.__version__, "numpy": np.__version__},
        )

        if not labels:
            raise ValueError("No labeled examples available for training.")

        (
            train_titles,
            test_titles,
            train_bodies,
            test_bodies,
            train_texts,
            test_texts,
            train_numeric,
            test_numeric,
            y_train,
            y_test,
        ) = train_test_split(
            feature_payload["titles"],
            feature_payload["bodies"],
            feature_payload["texts"],
            feature_payload["numeric"],
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        model = cached_train(
            {"titles": list(train_titles), "bodies": list(train_bodies)},
            list(y_train),
            model_params_for_cache,
            lib_versions={"numpy": np.__version__, "sklearn": sklearn_version},
        )

        if callable_or_attr(encode_texts):
            try:
                cache_train_embeddings(
                    [combine_text(t, b) for t, b in zip(train_titles, train_bodies)]
                )
            except Exception:
                pass

        split_state["X_train"] = {
            "titles": list(train_titles),
            "bodies": list(train_bodies),
            "texts": list(train_texts),
            "numeric": _as_list(train_numeric),
        }
        split_state["X_test"] = {
            "titles": list(test_titles),
            "bodies": list(test_bodies),
            "texts": list(test_texts),
            "numeric": _as_list(test_numeric),
        }
        split_state["y_train"] = list(y_train)
        split_state["y_test"] = list(y_test)
        split_state["hash"] = split_hash
        split_state["params"] = dict(split_params)

        model_state["clf"] = model
        model_state["vectorizer"] = getattr(model, "vectorizer", None) or getattr(
            model, "vectorizer_", None
        )
        model_state["params"] = dict(model_params)
        model_state["data_hash"] = data_hash
        model_state["status"] = "trained"

        ss["model"] = model
        ss["split_cache"] = (
            list(train_titles),
            list(test_titles),
            list(train_bodies),
            list(test_bodies),
            list(y_train),
            list(y_test),
        )
    except Exception as exc:
        raise RuntimeError("Failed to update cached training artifacts") from exc
    finally:
        run_state["busy"] = False


def request_meaning_map_refresh(
    ss: st.session_state.__class__, section_key: str | None, rerun_fn
) -> None:
    """Trigger a meaning map rebuild and rerun the app."""

    new_run_id = uuid4().hex
    if section_key:
        new_run_id = f"{section_key}_{new_run_id}"
    ss["train_story_run_id"] = new_run_id
    ss["train_refresh_expected"] = True
    attempts = int(ss.get("train_refresh_attempts", 0) or 0)
    ss["train_refresh_attempts"] = max(1, attempts)
    ss["train_storyboard_payload"] = {}
    rerun_fn()


def parse_split_cache(cache: Iterable[Any] | None) -> tuple[list, list, list, list, list, list]:
    """Return train/test splits from cached payloads."""

    if cache is None:
        raise ValueError("Missing split cache.")
    cache_list = list(cache)
    if len(cache_list) == 4:
        X_tr, X_te, y_tr, y_te = cache_list
        train_bodies = ["" for _ in range(len(X_tr))]
        test_bodies = ["" for _ in range(len(X_te))]
        return (
            list(X_tr),
            list(X_te),
            train_bodies,
            test_bodies,
            list(y_tr),
            list(y_te),
        )
    if len(cache_list) == 6:
        X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = cache_list
        return (
            list(X_tr_t),
            list(X_te_t),
            list(X_tr_b),
            list(X_te_b),
            list(y_tr),
            list(y_te),
        )
    y_tr = list(cache_list[-2]) if len(cache_list) >= 2 else []
    y_te = list(cache_list[-1]) if len(cache_list) >= 1 else []
    return [], [], [], [], y_tr, y_te


def label_balance_status(labeled: list[dict] | None) -> dict[str, Any]:
    """Return counts, total, ratio, and OK flag for balance."""

    labeled = labeled or []
    counts = Counter(
        [
            (r.get("label") or "").strip().lower()
            for r in labeled
            if isinstance(r, dict)
        ]
    )
    for key in ("spam", "safe"):
        counts.setdefault(key, 0)
    total = counts["spam"] + counts["safe"]
    big = max(counts["spam"], counts["safe"])
    small = min(counts["spam"], counts["safe"])
    ratio = (small / big) if big else 0.0
    ok = (total >= 12) and (counts["spam"] >= 6) and (counts["safe"] >= 6) and (ratio >= 0.60)
    return {"counts": counts, "total": total, "ratio": ratio, "ok": ok}


def pii_status(ss: st.session_state.__class__) -> dict[str, Any]:
    """Read PII scan summary saved during Prepare (if any)."""

    pii = ss.get("pii_scan") or {}
    status = pii.get("status", "unknown")
    counts = pii.get("counts", {})
    return {"status": status, "counts": counts}


def go_to_prepare(ss: st.session_state.__class__) -> None:
    """Jump to Prepare stage using Streamlit's rerun API."""

    ss["stage"] = "prepare"
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun is None:
        raise RuntimeError("Streamlit rerun API is unavailable")
    rerun()


__all__ = [
    "callable_or_attr",
    "go_to_prepare",
    "label_balance_status",
    "parse_split_cache",
    "pii_status",
    "prime_session_state_from_store",
    "request_meaning_map_refresh",
    "sync_training_artifacts",
]
