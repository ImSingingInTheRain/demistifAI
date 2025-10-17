from __future__ import annotations

import html
import logging
import math
import random
import hashlib
import re
import textwrap
import time
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from demistifai.ui.components.arch_demai import render_demai_architecture
from demistifai.constants import STAGE_BY_KEY, STAGE_INDEX, StageMeta
from demistifai.core.cache import cached_features, cached_train
from demistifai.core.language import HAS_LANGDETECT
from demistifai.core.nav import render_stage_top_grid
from demistifai.core.state import ensure_state, hash_dict, validate_invariants
from demistifai.core.utils import streamlit_rerun
from demistifai.dataset import (
    DEFAULT_DATASET_CONFIG,
    EDGE_CASE_TEMPLATES,
    DatasetConfig,
    STARTER_LABELED,
    compute_dataset_hash,
    compute_dataset_summary,
    dataset_delta_story,
    dataset_summary_delta,
    explain_config_change,
    generate_labeled_dataset,
    lint_dataset,
    lint_dataset_detailed,
    lint_text_spans,
)
from demistifai.modeling import (
    FEATURE_DISPLAY_NAMES,
    FEATURE_ORDER,
    FEATURE_PLAIN_LANGUAGE,
    HybridEmbedFeatsLogReg,
    PlattProbabilityCalibrator,
    URGENCY_TERMS,
    _combine_text,
    _counts,
    _fmt_delta,
    _fmt_pct,
    _pr_acc_cm,
    _predict_proba_batch,
    _y01,
    assess_performance,
    cache_train_embeddings,
    combine_text,
    compute_confusion,
    compute_numeric_features,
    df_confusion,
    encode_texts,
    embedding_backend_info,
    extract_urls,
    features_matrix,
    get_domain_tld,
    get_encoder,
    get_nearest_training_examples,
    make_after_eval_story,
    make_after_training_story,
    model_kind_string,
    numeric_feature_contributions,
    plot_threshold_curves,
    predict_spam_probability,
    threshold_presets,
    top_token_importances,
    verdict_label,
)

from stages.train_helpers import (
    _build_meaning_map_chart,
    _guardrail_window_values,
    _meaning_map_zoom_subset,
    _numeric_guardrails_caption_text,
    _prepare_meaning_map,
    _render_numeric_clue_cards,
    _render_numeric_clue_preview,
    _render_training_examples_preview,
    _render_unified_training_storyboard,
)

from demistifai.ui.components.mac_window import render_mac_window
from demistifai.ui.components.terminal.train import render_train_terminal
from demistifai.ui.components.train_animation import build_training_animation_column


logger = logging.getLogger(__name__)


def _callable_or_attr(target: Any, attr: str | None = None) -> bool:
    """Return True if ``target`` (or one of its attributes) is callable."""

    try:
        value = getattr(target, attr) if attr else target
    except Exception:
        return False
    return callable(value)


def render_train_stage_page(
    *,
    set_active_stage: Callable[[str], None],
    render_eu_ai_quote,
    render_language_mix_chip_rows,
    render_nerd_mode_toggle,
    section_surface,
    summarize_language_mix,
):
    """Entry point for the Train stage when called from ``streamlit_app``."""

    s = ensure_state()
    validate_invariants(s)

    ss = st.session_state

    run_state = s.setdefault("run", {})
    run_state.setdefault("busy", False)
    data_state = s.setdefault("data", {})
    split_state = s.setdefault("split", {})
    model_state = s.setdefault("model", {})
    stage = STAGE_BY_KEY["train"]

    prepared_df = data_state.get("prepared")

    def _prime_session_state_from_store() -> None:
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

    def _sync_training_artifacts(prepared: pd.DataFrame) -> None:
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
            _prime_session_state_from_store()
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

            if _callable_or_attr(encode_texts):
                try:
                    cache_train_embeddings(
                        [combine_text(t, b) for t, b in zip(train_titles, train_bodies)]
                    )
                except Exception:
                    pass

            def _as_list(payload: Any) -> list[Any]:
                if payload is None:
                    return []
                if hasattr(payload, "tolist"):
                    try:
                        return payload.tolist()
                    except Exception:
                        pass
                return list(payload)

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
            logger.exception("Failed to update cached training artifacts")
            st.error(f"Failed to update training artifacts: {exc}")
        finally:
            run_state["busy"] = False

    def _render_train_terminal(slot: DeltaGenerator) -> None:
        with slot:
            render_train_terminal(
                speed_type_ms=22,
                pause_between_lines_ms=320,
            )

    render_stage_top_grid("train", left_renderer=_render_train_terminal)

    if prepared_df is None:
        with section_surface():
            st.subheader(
                f"{stage.icon} {stage.title} — How does the spam detector learn from examples?"
            )
            st.info("First prepare and validate your dataset in **📊 Data**.")
            if st.button("Go to Data stage", type="primary"):
                set_active_stage("data")
                streamlit_rerun()
        return

    _prime_session_state_from_store()

    animation_column = build_training_animation_column()

    training_notes_html = textwrap.dedent(
        """
        <style>
          .train-animation__notes {
            display: grid;
            gap: 0.75rem;
          }
          .train-animation__notes h4 {
            margin: 0;
            font-size: 1.0rem;
            font-weight: 700;
            color: #0f172a;
          }
          .train-animation__notes ul {
            margin: 0;
            padding-left: 1.15rem;
            display: grid;
            gap: 0.45rem;
            font-size: 0.95rem;
            line-height: 1.45;
            color: rgba(15, 23, 42, 0.78);
          }
          .train-animation__notes li strong {
            color: #1d4ed8;
          }
        </style>
        <div class="train-animation__notes">
          <h4>Training loop snapshot</h4>
          <ul>
            <li><strong>MiniLM embeddings</strong> map emails into a 384D space.</li>
            <li><strong>Logistic regression</strong> learns a separating boundary.</li>
            <li><strong>Epoch playback</strong> shows how spam/work clusters settle.</li>
            <li>Toggle Nerd Mode to inspect features and weights in detail.</li>
          </ul>
        </div>
        """
    ).strip()

    render_mac_window(
        st,
        title="Training dynamics monitor",
        subtitle="MiniLM organising emails by meaning",
        columns=2,
        ratios=(0.3, 0.7),
        col_html=[training_notes_html, animation_column.html],
        id_suffix="train-animation",
        fallback_height=animation_column.fallback_height,
    )

    has_embed = _callable_or_attr(encode_texts)

    render_train_stage(
        ss,
        streamlit_rerun=streamlit_rerun,
        has_embed=has_embed,
        has_langdetect=HAS_LANGDETECT,
        render_eu_ai_quote=render_eu_ai_quote,
        render_language_mix_chip_rows=render_language_mix_chip_rows,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        section_surface=section_surface,
        summarize_language_mix=summarize_language_mix,
    )

    if isinstance(prepared_df, pd.DataFrame) and not prepared_df.empty:
        _sync_training_artifacts(prepared_df)


def _request_meaning_map_refresh(ss, section_key: str | None, rerun_fn) -> None:
    new_run_id = uuid4().hex
    if section_key:
        new_run_id = f"{section_key}_{new_run_id}"
    ss["train_story_run_id"] = new_run_id
    ss["train_refresh_expected"] = True
    attempts = int(ss.get("train_refresh_attempts", 0) or 0)
    ss["train_refresh_attempts"] = max(1, attempts)
    ss["train_storyboard_payload"] = {}
    rerun_fn()


def _parse_split_cache(cache):
    if cache is None:
        raise ValueError("Missing split cache.")
    if len(cache) == 4:
        X_tr, X_te, y_tr, y_te = cache
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
    if len(cache) == 6:
        X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = cache
        return (
            list(X_tr_t),
            list(X_te_t),
            list(X_tr_b),
            list(X_te_b),
            list(y_tr),
            list(y_te),
        )
    y_tr = list(cache[-2]) if len(cache) >= 2 else []
    y_te = list(cache[-1]) if len(cache) >= 1 else []
    return [], [], [], [], y_tr, y_te

def _label_balance_status(labeled: list[dict] | None) -> dict:
    """Return counts, total, ratio, and OK flag for balance."""

    labeled = labeled or []
    counts = _counts(
        [
            (r.get("label") or "").strip().lower()
            for r in labeled
            if isinstance(r, dict)
        ]
    )
    total = counts["spam"] + counts["safe"]
    big = max(counts["spam"], counts["safe"])
    small = min(counts["spam"], counts["safe"])
    ratio = (small / big) if big else 0.0
    ok = (total >= 12) and (counts["spam"] >= 6) and (counts["safe"] >= 6) and (ratio >= 0.60)
    return {"counts": counts, "total": total, "ratio": ratio, "ok": ok}

def _pii_status(ss) -> dict:
    """Read PII scan summary saved during Prepare (if any)."""

    pii = ss.get("pii_scan") or {}
    status = pii.get("status", "unknown")
    counts = pii.get("counts", {})
    return {"status": status, "counts": counts}


def _go_to_prepare(ss):
    """Jump to Prepare stage (match your stage switching mechanism)."""

    ss["stage"] = "prepare"
    st.experimental_rerun()

def render_train_stage(
    ss,
    *,
    streamlit_rerun,
    has_embed,
    has_langdetect,
    render_eu_ai_quote,
    render_language_mix_chip_rows,
    render_nerd_mode_toggle,
    section_surface,
    summarize_language_mix,
):


    stage = STAGE_BY_KEY["train"]
    ss.setdefault("token_budget_cache", {})
    ss.setdefault("guard_params", {})
    ss.setdefault("train_in_progress", False)
    ss.setdefault("train_refresh_expected", False)
    ss.setdefault("train_refresh_attempts", 0)
    ss["guard_params"].setdefault("assist_center_mode", "manual")  # "manual" | "auto"
    ss["guard_params"].setdefault("assist_center", float(ss.get("threshold", 0.6)))
    ss["guard_params"].setdefault("uncertainty_band", 0.08)
    ss["guard_params"].setdefault("numeric_scale", 0.5)
    ss["guard_params"].setdefault("numeric_logit_cap", 1.0)
    ss["guard_params"].setdefault("combine_strategy", "blend")

    st.markdown(
        """
        <style>
        .train-intro-card {
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.16), rgba(14, 165, 233, 0.16));
            border-radius: 1.25rem;
            padding: 1.28rem 1.6rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 18px 38px rgba(15, 23, 42, 0.1);
            margin-bottom: 1.8rem;
        }
        .train-intro-card__header {
            display: flex;
            gap: 1rem;
            align-items: flex-start;
            margin-bottom: 0.85rem;
        }
        .train-intro-card__icon {
            font-size: 1.75rem;
            line-height: 1;
            background: rgba(15, 23, 42, 0.08);
            border-radius: 1rem;
            padding: 0.55rem 0.95rem;
            box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.05);
        }
        .train-intro-card__eyebrow {
            font-size: 0.75rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 700;
            color: rgba(15, 23, 42, 0.55);
            display: inline-block;
        }
        .train-intro-card__title {
            margin: 0;
            font-size: 1.4rem;
            font-weight: 700;
            color: #0f172a;
        }
        .train-intro-card__body {
            margin: 0 0 1.15rem 0;
            color: rgba(15, 23, 42, 0.82);
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .train-intro-card__steps {
            display: grid;
            gap: 0.75rem;
        }
        .train-intro-card__step {
            display: flex;
            gap: 0.75rem;
            align-items: flex-start;
        }
        .train-intro-card__step-index {
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: rgba(79, 70, 229, 0.18);
            color: rgba(30, 64, 175, 0.9);
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
        }
        .train-intro-card__step-body {
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.8);
            line-height: 1.55;
        }
        .train-launchpad-card {
            border-radius: 1.1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: rgba(255, 255, 255, 0.92);
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.12);
            padding: 0.96rem 1.35rem;
            margin-bottom: 1rem;
        }
        .train-launchpad-card__title {
            margin: 0 0 0.65rem 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
        }
        .train-launchpad-card__list {
            list-style: none;
            margin: 0;
            padding: 0;
            display: grid;
            gap: 0.55rem;
        }
        .train-launchpad-card__list li {
            display: grid;
            grid-template-columns: 1.8rem 1fr;
            gap: 0.55rem;
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.78);
            line-height: 1.4;
        }
        .train-launchpad-card__bullet {
            width: 1.8rem;
            height: 1.8rem;
            border-radius: 0.75rem;
            background: rgba(79, 70, 229, 0.12);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
        }
        .train-launchpad-card--secondary {
            background: rgba(30, 64, 175, 0.06);
            border: 1px dashed rgba(30, 64, 175, 0.35);
        }
        .train-how-card {
            border-radius: 1.1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 0.96rem 1.35rem;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.1);
        }
        .train-how-card__header {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            margin-bottom: 0.75rem;
        }
        .train-how-card__icon {
            font-size: 1.35rem;
        }
        .train-how-card__title {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: #0f172a;
        }
        .train-how-card__body {
            margin: 0 0 0.85rem 0;
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.78);
            line-height: 1.55;
        }
        .train-how-card__steps {
            margin: 0 0 0.9rem 1.15rem;
            padding: 0;
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.82);
        }
        .train-how-card__grid {
            display: grid;
            gap: 0.9rem;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            margin-bottom: 0.6rem;
        }
        .train-how-card__panel {
            position: relative;
            border-radius: 1rem;
            border: 1px solid rgba(79, 70, 229, 0.16);
            background: rgba(79, 70, 229, 0.06);
            padding: 0.85rem 1rem;
            display: grid;
            gap: 0.35rem;
        }
        .train-how-card__panel-icon {
            font-size: 1.1rem;
        }
        .train-how-card__panel-title {
            margin: 0;
            font-size: 0.9rem;
            font-weight: 700;
            color: rgba(30, 64, 175, 0.95);
        }
        .train-how-card__panel-body {
            margin: 0;
            font-size: 0.85rem;
            color: rgba(15, 23, 42, 0.75);
            line-height: 1.5;
        }
        .train-how-card__divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(15, 23, 42, 0), rgba(15, 23, 42, 0.25), rgba(15, 23, 42, 0));
            margin: 0.75rem 0 0.9rem 0;
        }
        .train-how-card__body--muted {
            color: rgba(15, 23, 42, 0.6);
            font-size: 0.88rem;
            margin-bottom: 0.65rem;
        }
        .train-how-card__step-grid {
            display: grid;
            gap: 0.75rem;
        }
        .train-how-card__step-box {
            border-radius: 1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: linear-gradient(135deg, rgba(14, 116, 144, 0.08), rgba(59, 130, 246, 0.08));
            padding: 0.95rem 1.1rem;
            display: grid;
            gap: 0.5rem;
        }
        .train-how-card__step-label {
            display: flex;
            align-items: center;
            gap: 0.55rem;
        }
        .train-how-card__step-number {
            width: 2rem;
            height: 2rem;
            border-radius: 999px;
            background: rgba(14, 116, 144, 0.18);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.95rem;
            color: rgba(12, 74, 110, 0.95);
        }
        .train-how-card__step-icon {
            font-size: 1.25rem;
        }
        .train-how-card__step-title {
            font-weight: 700;
            font-size: 0.95rem;
            color: rgba(15, 23, 42, 0.9);
        }
        .train-how-card__step-list {
            margin: 0;
            padding-left: 1.2rem;
            font-size: 0.88rem;
            color: rgba(15, 23, 42, 0.78);
            line-height: 1.55;
        }
        .train-how-card__step-sublist {
            margin: 0.45rem 0 0 1.1rem;
            padding-left: 1rem;
            font-size: 0.85rem;
            color: rgba(15, 23, 42, 0.75);
            line-height: 1.5;
        }
        .train-how-card__step-example {
            margin: 0;
            font-size: 0.82rem;
            color: rgba(15, 23, 42, 0.68);
            background: rgba(255, 255, 255, 0.6);
            border-radius: 0.75rem;
            padding: 0.55rem 0.75rem;
            border: 1px dashed rgba(12, 74, 110, 0.25);
        }
        .train-launchpad-card__grid {
            display: grid;
            gap: 0.65rem;
        }
        .train-launchpad-card__item {
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 0.6rem;
            padding: 0.6rem 0.75rem;
            border-radius: 0.85rem;
            background: rgba(15, 23, 42, 0.04);
            align-items: flex-start;
        }
        .train-launchpad-card__badge {
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }
        .train-launchpad-card__badge-num {
            width: 1.8rem;
            height: 1.8rem;
            border-radius: 999px;
            background: rgba(79, 70, 229, 0.16);
            color: rgba(30, 64, 175, 0.92);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.85rem;
        }
        .train-launchpad-card__badge-icon {
            font-size: 1.05rem;
        }
        .train-launchpad-card__item-title {
            margin: 0;
            font-size: 0.9rem;
            font-weight: 600;
            color: #0f172a;
        }
        .train-launchpad-card__item-body {
            margin: 0.2rem 0 0 0;
            font-size: 0.83rem;
            color: rgba(15, 23, 42, 0.75);
            line-height: 1.45;
        }
        .train-token-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 999px;
            padding: 0.4rem 0.85rem;
            background: rgba(59, 130, 246, 0.12);
            color: rgba(30, 64, 175, 0.9);
            font-size: 0.8rem;
            font-weight: 600;
        }
        .numeric-clue-card-grid {
            display: grid;
            gap: 0.85rem;
        }
        .numeric-clue-card {
            border-radius: 1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(14, 165, 233, 0.08));
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            padding: 0.85rem 1rem;
        }
        .numeric-clue-card__header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.75rem;
        }
        .numeric-clue-card__subject {
            font-size: 0.95rem;
            font-weight: 700;
            color: #0f172a;
            flex: 1 1 auto;
        }
        .numeric-clue-card__tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            justify-content: flex-end;
        }
        .numeric-clue-tag {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            font-size: 0.72rem;
            font-weight: 600;
            background: rgba(15, 23, 42, 0.06);
            color: rgba(15, 23, 42, 0.7);
        }
        .numeric-clue-tag--truth {
            background: rgba(15, 23, 42, 0.08);
            color: rgba(15, 23, 42, 0.65);
        }
        .numeric-clue-tag--spam {
            background: rgba(239, 68, 68, 0.18);
            color: #b91c1c;
        }
        .numeric-clue-tag--safe {
            background: rgba(59, 130, 246, 0.18);
            color: #1d4ed8;
        }
        .numeric-clue-tag--unknown {
            background: rgba(148, 163, 184, 0.22);
            color: rgba(30, 41, 59, 0.72);
        }
        .numeric-clue-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.28rem 0.65rem;
            font-size: 0.76rem;
            font-weight: 600;
            background: rgba(15, 23, 42, 0.08);
            color: rgba(15, 23, 42, 0.75);
        }
        .numeric-clue-chip--spam {
            background: rgba(239, 68, 68, 0.18);
            color: #b91c1c;
        }
        .numeric-clue-chip--safe {
            background: rgba(34, 197, 94, 0.18);
            color: #15803d;
        }
        .numeric-clue-card__chips {
            margin-top: 0.65rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .numeric-clue-card__reason {
            margin-top: 0.55rem;
            font-size: 0.82rem;
            color: rgba(15, 23, 42, 0.78);
        }
        .numeric-clue-card__meta {
            margin-top: 0.45rem;
            font-size: 0.72rem;
            color: rgba(15, 23, 42, 0.62);
        }
        .numeric-clue-preview {
            border-radius: 1rem;
            border: 1px dashed rgba(37, 99, 235, 0.35);
            background: rgba(191, 219, 254, 0.28);
            padding: 1rem 1.1rem;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.4);
        }
        .numeric-clue-preview__header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            gap: 0.5rem;
            font-size: 0.85rem;
            font-weight: 600;
            color: rgba(30, 64, 175, 0.85);
            margin-bottom: 0.6rem;
        }
        .numeric-clue-preview__band {
            border-radius: 0.85rem;
            background: linear-gradient(90deg, rgba(59, 130, 246, 0.14), rgba(14, 165, 233, 0.16));
            border: 1px dashed rgba(37, 99, 235, 0.45);
            padding: 0.9rem 0.9rem 0.95rem;
        }
        .numeric-clue-preview__ticks {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            color: rgba(30, 58, 138, 0.78);
            margin-bottom: 0.7rem;
        }
        .numeric-clue-preview__chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .numeric-clue-preview__chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.3rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(255, 255, 255, 0.78);
            color: rgba(30, 41, 59, 0.8);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.18);
        }
        .numeric-clue-preview__note {
            margin: 0.8rem 0 0 0;
            font-size: 0.78rem;
            color: rgba(30, 41, 59, 0.72);
            line-height: 1.4;
        }
        .train-inline-note {
            margin-top: 0.35rem;
            font-size: 0.8rem;
            color: rgba(30, 64, 175, 0.82);
            background: rgba(191, 219, 254, 0.35);
            border-radius: 0.6rem;
            padding: 0.45rem 0.75rem;
            display: inline-block;
        }
        .train-action-card {
            border-radius: 1.1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 1rem 1.35rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(14, 165, 233, 0.12));
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.1);
            margin-bottom: 0.75rem;
        }
        .train-action-card__eyebrow {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-weight: 700;
            color: rgba(30, 64, 175, 0.75);
        }
        .train-action-card__title {
            margin: 0.35rem 0 0.6rem 0;
            font-size: 1.25rem;
            font-weight: 700;
            color: #0f172a;
        }
        .train-action-card__body {
            margin: 0;
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.78);
            line-height: 1.5;
        }
        .train-context-card {
            border-radius: 1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 0.8rem 1.1rem;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
            margin-bottom: 0.9rem;
        }
        .train-context-card h5 {
            margin: 0 0 0.5rem 0;
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
        }
        .train-context-card ul {
            margin: 0;
            padding-left: 1.1rem;
            font-size: 0.88rem;
            color: rgba(15, 23, 42, 0.78);
            line-height: 1.5;
        }
        .train-context-card--tip {
            background: rgba(236, 233, 254, 0.6);
            border: 1px dashed rgba(79, 70, 229, 0.35);
            color: rgba(55, 48, 163, 0.95);
        }
        .train-band-card {
            border-radius: 1rem;
            border: 1px solid rgba(79, 70, 229, 0.25);
            padding: 0.95rem 1.1rem 1.05rem 1.1rem;
            background: rgba(79, 70, 229, 0.08);
            box-shadow: inset 0 0 0 1px rgba(79, 70, 229, 0.08);
        }
        .train-band-card__title {
            margin: 0 0 0.55rem 0;
            font-size: 0.95rem;
            font-weight: 700;
            color: rgba(49, 46, 129, 0.9);
        }
        .train-band-card__bar {
            position: relative;
            width: 100%;
            height: 16px;
            border-radius: 999px;
            background: rgba(30, 64, 175, 0.12);
        }
        .train-band-card__band {
            position: absolute;
            top: 0;
            bottom: 0;
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.35);
        }
        .train-band-card__threshold {
            position: absolute;
            top: -4px;
            bottom: -4px;
            width: 2px;
            border-radius: 2px;
            background: rgba(30, 64, 175, 0.95);
        }
        .train-band-card__scale {
            display: flex;
            justify-content: space-between;
            font-size: 0.68rem;
            color: rgba(30, 64, 175, 0.8);
            margin-top: 0.35rem;
        }
        .train-band-card__caption {
            margin-top: 0.45rem;
            font-size: 0.75rem;
            color: rgba(49, 46, 129, 0.85);
        }
        .train-band-card__hint {
            margin-top: 0.35rem;
            font-size: 0.78rem;
            color: rgba(49, 46, 129, 0.75);
        }
        .train-nerd-intro {
            border-radius: 1rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            padding: 1rem 1.1rem;
            background: rgba(14, 165, 233, 0.08);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }
        .train-nerd-intro h4 {
            margin: 0 0 0.35rem 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: rgba(7, 89, 133, 0.9);
        }
        .train-nerd-intro p {
            margin: 0;
            font-size: 0.88rem;
            color: rgba(7, 89, 133, 0.78);
            line-height: 1.5;
        }
        .train-nerd-hint {
            margin-top: 0.85rem;
            font-size: 0.8rem;
            color: rgba(15, 23, 42, 0.7);
            background: rgba(191, 219, 254, 0.45);
            border-radius: 0.75rem;
            padding: 0.55rem 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    stage_number = STAGE_INDEX.get(stage.key, 0) - STAGE_INDEX.get("data", 0) + 1
    if stage_number < 1:
        stage_number = STAGE_INDEX.get(stage.key, 0) + 1

    nerd_mode_flag = bool(ss.get("nerd_mode_train") or ss.get("nerd_mode"))


    nerd_mode_train_enabled = bool(ss.get("nerd_mode_train"))

    stage_title_html = html.escape(stage.title)
    stage_desc_html = html.escape(stage.description)
    stage_icon_html = html.escape(stage.icon)

    with section_surface():
        
        render_eu_ai_quote("An AI system “infers, from the input it receives…”.")
        
        intro_col, launchpad_col = st.columns([0.58, 0.42], gap="large")
        
        with intro_col:
            st.markdown(
                """
                <div class="train-intro-card">
                    <div class="train-intro-card__header">
                        <span class="train-intro-card__icon">{icon}</span>
                        <div>
                            <span class="train-intro-card__eyebrow">Stage {num}</span>
                            <h4 class="train-intro-card__title">{title}</h4>
                        </div>
                    </div>
                    <div class="train-how-card__header">
                            <div>
                                <h5 class="train-how-card__title">Now your AI system will learn how to achieve an objective</h5>
                                <p class="train-how-card__body">Think of this as teaching your AI assistant how to tell the difference between spam and safe emails.</p>
                            </div>
                        </div>
                        <div class="train-how-card__grid">
                            <div class="train-how-card__panel">
                                <div class="train-how-card__panel-icon">🫶</div>
                                <h6 class="train-how-card__panel-title">Your part</h6>
                                <p class="train-how-card__panel-body">Provide examples with clear labels (“This one is spam, that one is safe”).</p>
                            </div>
                            <div class="train-how-card__panel">
                                <div class="train-how-card__panel-icon">🤖</div>
                                <h6 class="train-how-card__panel-title">The system’s part</h6>
                                <p class="train-how-card__panel-body">Spot patterns that generalize to emails it hasn’t seen yet.</p>
                            </div>
                        </div>
                </div>
                """.format(
                    icon=stage_icon_html,
                    num=stage_number,
                    title=stage_title_html,
                    desc=stage_desc_html,
                ),
                unsafe_allow_html=True,
            )
        with launchpad_col:
            st.markdown(
                """
                <div class="train-launchpad-card">
                  <div class="train-launchpad-card__title">🧭 Training Launchpad — readiness & controls</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            bal = _label_balance_status(ss.get("labeled"))
            bal_ok = bal["ok"]
            bal_chip = "✅ OK" if bal_ok else "⚠️ Need work"
            chip_style = (
                "display:inline-block;background:rgba(14,165,233,0.12);color:#0369a1;"
                "border-radius:999px;padding:0.1rem 0.5rem;font-size:0.75rem;font-weight:600;"
            )
            st.markdown(
                f"""
                <div class="train-launchpad-card__item">
                  <p class="train-launchpad-card__item-title">Balanced labels <span style=\"{chip_style}\">{bal_chip}</span></p>
                  <p class="train-launchpad-card__item-body">Spam: {bal['counts']['spam']} • Safe: {bal['counts']['safe']} (ratio ~{bal['ratio']:.2f})</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if not bal_ok:
                if st.button("Fix in Prepare", key="launchpad_fix_balance", use_container_width=True):
                    _go_to_prepare(ss)

            pii = _pii_status(ss)
            tag = {
                "clean": "✅ OK",
                "found": "⚠️ Need work",
                "unknown": "ⓘ Not scanned",
            }.get(pii["status"], "ⓘ Not scanned")
            counts_str = ", ".join(f"{k} {v}" for k, v in (pii["counts"] or {}).items()) or "—"
            st.markdown(
                f"""
                <div class="train-launchpad-card__item">
                  <p class="train-launchpad-card__item-title">Data hygiene <span style=\"{chip_style}\">{tag}</span></p>
                  <p class="train-launchpad-card__item-body">PII in preview: {counts_str}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if pii["status"] in {"found", "unknown"}:
                if st.button("Review in Prepare", key="launchpad_fix_pii", use_container_width=True):
                    _go_to_prepare(ss)

            nerd_mode_train_active = bool(ss.get("nerd_mode_train"))
            if not nerd_mode_train_active:
                ss["train_params"]["test_size"] = (
                    st.slider(
                        "Hold-out for honest test (%)",
                        min_value=10,
                        max_value=50,
                        step=5,
                        value=int(float(ss["train_params"].get("test_size", 0.30)) * 100),
                        help="Set aside some labeled emails for a fair check. More hold-out = fairer exam, fewer examples to learn from.",
                    )
                    / 100.0
                )
                if ss["train_params"]["test_size"] < 0.15 or ss["train_params"]["test_size"] > 0.40:
                    st.caption("Tip: 20–30% is a good range for most datasets.")

            gp = ss["guard_params"]
            auto_mode = st.toggle(
                "Implicit strategy mode (Auto)",
                value=(gp.get("assist_center_mode") == "auto"),
                key="launchpad_auto_mode",
            )
            gp["assist_center_mode"] = "auto" if auto_mode else "manual"
            if gp["assist_center_mode"] == "manual" and not nerd_mode_train_active:
                gp["assist_center"] = st.slider(
                    "Where ‘borderline’ lives (0–1)",
                    0.30,
                    0.90,
                    float(gp.get("assist_center", ss.get("threshold", 0.6))),
                    0.01,
                )
                gp["uncertainty_band"] = st.slider(
                    "Uncertainty band (±)",
                    0.0,
                    0.20,
                    float(gp.get("uncertainty_band", 0.08)),
                    0.01,
                )
            elif gp["assist_center_mode"] == "auto":
                st.caption(
                    "We’ll pick the center from the hold-out set after training so numeric clues trigger where they help most."
                )
            ss["guard_params"] = gp

            nerd_mode_train_enabled = render_nerd_mode_toggle(
                key="nerd_mode_train",
                title="Nerd Mode — advanced controls",
                description="Tweak the train/test split, solver iterations, and regularization strength.",
                icon="🔬",
            )

    if nerd_mode_train_enabled:
        with section_surface():
            st.markdown(
                """
                <div class="train-nerd-intro">
                    <h4>🛡️ Advanced split & guardrail controls</h4>
                    <p>Fine-tune how much data we hold out, solver behaviour, and the numeric assist rules that complement the text model.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            ss.setdefault("guard_params", {})
            gp = ss["guard_params"]
            assist_mode = gp.get("assist_center_mode", "auto")
            colA, colB = st.columns(2)
            with colA:
                ss["train_params"]["test_size"] = st.slider(
                    "🧪 Hold-out test fraction (advanced)",
                    min_value=0.10,
                    max_value=0.50,
                    value=float(ss["train_params"]["test_size"]),
                    step=0.05,
                    help="How much labeled data to keep aside as a mini 'exam' (not used for learning).",
                )
                st.caption("🧪 More hold-out = more honest testing but fewer examples for learning.")
                ss["train_params"]["random_state"] = st.number_input(
                    "Random seed",
                    min_value=0,
                    value=int(ss["train_params"]["random_state"]),
                    step=1,
                    help="Fix this to make your train/test split reproducible.",
                )
                st.caption("Keeps your split and results repeatable.")
                if assist_mode == "manual":
                    gp["assist_center"] = st.slider(
                        "🛡️ Numeric assist center (text score)",
                        min_value=0.30,
                        max_value=0.90,
                        step=0.01,
                        value=float(gp.get("assist_center", float(ss.get("threshold", 0.6)))),
                        help=(
                            "Center of the borderline region. When the text-only spam probability is near this "
                            "value, numeric guardrails are allowed to lend a hand."
                        ),
                    )
                    st.caption(
                        "🛡️ Where ‘borderline’ lives on the 0–1 scale; most emails away from here won’t use numeric cues."
                    )
                    gp["uncertainty_band"] = st.slider(
                        "🛡️ Uncertainty band (± around threshold)",
                        min_value=0.0,
                        max_value=0.20,
                        step=0.01,
                        value=float(gp.get("uncertainty_band", 0.08)),
                        help="Only consult numeric cues when the text score falls inside this band.",
                    )
                    st.caption("🛡️ Wider band = numeric cues help more often; narrower = trust text more.")
                    gp["numeric_scale"] = st.slider(
                        "🛡️ Numeric blend weight (when consulted)",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.05,
                        value=float(gp.get("numeric_scale", 0.5)),
                        help="How much numeric probability counts in the blend within the band.",
                    )
                    st.caption("🛡️ Higher = numeric cues have a stronger say when consulted.")
                else:
                    st.caption("Controlled by Implicit strategy mode")
                    assist_center = float(gp.get("assist_center", float(ss.get("threshold", 0.6))))
                    uncertainty_band = float(gp.get("uncertainty_band", 0.08))
                    numeric_scale = float(gp.get("numeric_scale", 0.5))
                    st.markdown(
                        f"<div class='train-token-chip'>Center: {assist_center:.2f}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='train-token-chip'>Band ±{uncertainty_band:.2f}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='train-token-chip'>Blend weight: {numeric_scale:.2f}</div>",
                        unsafe_allow_html=True,
                    )
            with colB:
                ss["train_params"]["max_iter"] = st.number_input(
                    "Max iterations (solver)",
                    min_value=200,
                    value=int(ss["train_params"]["max_iter"]),
                    step=100,
                    help="How many optimization steps the classifier can take before stopping.",
                )
                st.caption("Higher values let the solver search longer; use if it says ‘didn’t converge’.")
                ss["train_params"]["C"] = st.number_input(
                    "Regularization strength C (inverse of regularization)",
                    min_value=0.01,
                    value=float(ss["train_params"]["C"]),
                    step=0.25,
                    format="%.2f",
                    help="Higher C fits training data more tightly; lower C adds regularization to reduce overfitting.",
                )
                st.caption("Higher C hugs the training data (risk overfit). Lower C smooths (better generalization).")
                if assist_mode == "manual":
                    gp["numeric_logit_cap"] = st.slider(
                        "🛡️ Cap numeric logit (absolute)",
                        min_value=0.2,
                        max_value=3.0,
                        step=0.1,
                        value=float(gp.get("numeric_logit_cap", 1.0)),
                        help="Limits how strongly numeric cues can push toward Spam/Safe.",
                    )
                    st.caption("🛡️ A safety cap so numeric cues can’t overpower the text score.")
                    gp["combine_strategy"] = st.radio(
                        "🛡️ Numeric combination strategy",
                        options=["blend", "threshold_shift"],
                        index=0 if gp.get("combine_strategy", "blend") == "blend" else 1,
                        horizontal=True,
                        help="Blend = mix text & numeric probs; Threshold shift = keep text prob, adjust effective threshold slightly.",
                    )
                else:
                    numeric_logit_cap = float(gp.get("numeric_logit_cap", 1.0))
                    combine_strategy = str(gp.get("combine_strategy", "blend"))
                    st.markdown(
                        f"<div class='train-token-chip'>Logit cap: {numeric_logit_cap:.2f}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='train-token-chip'>Strategy: {combine_strategy.replace('_', ' ').title()}</div>",
                        unsafe_allow_html=True,
                    )

            if assist_mode == "manual" and gp.get("combine_strategy", "blend") == "threshold_shift":
                st.markdown("**🛡️ Threshold-shift micro-rules** (applied only within the uncertainty band)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    gp["shift_suspicious_tld"] = st.number_input(
                        "🛡️ Shift for suspicious TLD",
                        value=float(gp.get("shift_suspicious_tld", -0.04)),
                        step=0.01,
                        format="%.2f",
                        help="Negative shift lowers the threshold (be stricter) when a suspicious domain is present.",
                    )
                    st.caption("🛡️ Tweaks the cut-off in specific situations (e.g., suspicious domains → stricter).")
                with col2:
                    gp["shift_many_links"] = st.number_input(
                        "🛡️ Shift for many external links",
                        value=float(gp.get("shift_many_links", -0.03)),
                        step=0.01,
                        format="%.2f",
                        help="Negative shift lowers the threshold when multiple external links are detected.",
                    )
                    st.caption("🛡️ Tweaks the cut-off in specific situations (e.g., suspicious domains → stricter).")
                with col3:
                    gp["shift_calm_text"] = st.number_input(
                        "🛡️ Shift for calm text",
                        value=float(gp.get("shift_calm_text", +0.02)),
                        step=0.01,
                        format="%.2f",
                        help="Positive shift raises the threshold when text looks calm (very low ALL-CAPS).",
                    )
                    st.caption("🛡️ Tweaks the cut-off in specific situations (e.g., suspicious domains → stricter).")
            elif assist_mode != "manual" and gp.get("combine_strategy", "blend") == "threshold_shift":
                st.markdown("**🛡️ Threshold-shift micro-rules** (read-only)")
                shifts = {
                    "Suspicious TLD": float(gp.get("shift_suspicious_tld", -0.04)),
                    "Many external links": float(gp.get("shift_many_links", -0.03)),
                    "Calm text": float(gp.get("shift_calm_text", +0.02)),
                }
                cols = st.columns(len(shifts))
                for (label, value), col in zip(shifts.items(), cols):
                    with col:
                        st.markdown(
                            f"<div class='train-token-chip'>{label}: {value:+.2f}</div>",
                            unsafe_allow_html=True,
                        )

            st.markdown(
                """
                <div class="train-nerd-hint">
                    <strong>🎯 Guide:</strong> Hold-out keeps an honest exam set, the seed makes runs reproducible, <em>max iter</em> and <em>C</em> steer the solver, and the numeric guardrails define when structured cues can override the text score.
                </div>
                """,
                unsafe_allow_html=True,
            )

    token_budget_text = "Token budget: —"
    avg_tokens_estimate: Optional[float] = None
    show_trunc_tip = False
    try:
        labeled_rows = ss.get("labeled", [])
        if labeled_rows:
            dataset_hash = compute_dataset_hash(labeled_rows)
            cache = ss.get("token_budget_cache", {})
            stats = cache.get(dataset_hash)
            if stats is None:
                titles = [str(row.get("title", "")) for row in labeled_rows]
                bodies = [str(row.get("body", "")) for row in labeled_rows]
                stats = _estimate_token_stats(titles, bodies, max_tokens=384)
                cache[dataset_hash] = stats
                ss["token_budget_cache"] = cache
            if stats and stats.get("n"):
                avg_tokens = float(stats.get("avg_tokens", 0.0))
                if math.isfinite(avg_tokens) and avg_tokens > 0.0:
                    avg_tokens_estimate = avg_tokens
                pct_trunc = float(stats.get("p_truncated", 0.0)) * 100.0
                token_budget_text = f"Token budget: avg ~{avg_tokens:.0f} • truncated: {pct_trunc:.1f}%"
                show_trunc_tip = float(stats.get("p_truncated", 0.0)) > 0.05
        else:
            token_budget_text = "Token budget: —"
    except Exception:
        token_budget_text = "Token budget: —"
        show_trunc_tip = False

    if show_trunc_tip:
        st.markdown(
            f"<div class='train-token-chip'>{html.escape(token_budget_text)}</div>",
            unsafe_allow_html=True,
        )
    if show_trunc_tip:
        st.markdown(
            "<div class='train-inline-note'>Tip: long emails will be clipped; summaries help.</div>",
            unsafe_allow_html=True,
        )

    should_rerun_after_training = False
    training_successful = False

    with section_surface():
        action_col, context_col = st.columns([0.55, 0.45], gap="large")
        with action_col:
            st.markdown(
                """
                <div class="train-action-card">
                    <div class="train-action-card__eyebrow">Run training</div>
                    <div class="train-action-card__title">Teach the spam detector</div>
                    <p class="train-action-card__body">When you’re ready, launch the training run. We’ll automatically evaluate on the hold-out split.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            train_disabled = bool(ss.get("train_in_progress"))
            trigger_train = st.button(
                "🚀 Train model",
                type="primary",
                use_container_width=True,
                disabled=train_disabled,
            )
            if train_disabled:
                st.caption("Training in progress… hang tight while we refresh the charts.")
            if ss.get("train_flash_finished"):
                st.success("Training finished.")
                ss["train_flash_finished"] = False
        with context_col:
            
            if not ss.get("nerd_mode_train"):
                center, band, low, high = _guardrail_window_values(ss)
                low_pct = low * 100.0
                high_pct = high * 100.0
                threshold_pct = max(0.0, min(100.0, center * 100.0))

                band_left = max(0.0, min(100.0, low_pct))
                band_right = max(0.0, min(100.0, high_pct))
                band_width = max(0.0, band_right - band_left)


    has_model = ss.get("model") is not None
    has_split_cache = ss.get("split_cache") is not None

    def _request_refresh(section: str | None) -> None:
        _request_meaning_map_refresh(ss, section, streamlit_rerun)

    _render_unified_training_storyboard(
        ss,
        has_model=has_model,
        has_split=has_split_cache,
        has_embed=bool(has_embed),
        section_surface=section_surface,
        request_meaning_map_refresh=_request_refresh,
        parse_split_cache=_parse_split_cache,
        rerun=streamlit_rerun,
        logger=logger,
    )

    if trigger_train:
        if len(ss["labeled"]) < 6:
            st.warning("Please label a few more emails first (≥6 examples).")
        else:
            df = pd.DataFrame(ss["labeled"])
            if len(df["label"].unique()) < 2:
                st.warning("You need both classes (spam and safe) present to train.")
            else:
                params = ss.get("train_params", {})
                test_size = float(params.get("test_size", 0.30))
                random_state = int(params.get("random_state", 42))
                max_iter = int(params.get("max_iter", 1000))
                C_value = float(params.get("C", 1.0))

                titles = df["title"].fillna("").tolist()
                bodies = df["body"].fillna("").tolist()
                y = df["label"].tolist()
                ss["train_in_progress"] = True
                try:
                    with st.spinner("Training the model and refreshing charts…"):
                        X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = train_test_split(
                            titles,
                            bodies,
                            y,
                            test_size=test_size,
                            random_state=random_state,
                            stratify=y,
                        )

                        gp = ss.get("guard_params", {})
                        model = HybridEmbedFeatsLogReg(
                            max_iter=max_iter,
                            C=C_value,
                            random_state=random_state,
                            numeric_assist_center=float(
                                gp.get("assist_center", float(ss.get("threshold", 0.6)))
                            ),
                            uncertainty_band=float(gp.get("uncertainty_band", 0.08)),
                            numeric_scale=float(gp.get("numeric_scale", 0.5)),
                            numeric_logit_cap=float(gp.get("numeric_logit_cap", 1.0)),
                            combine_strategy=str(gp.get("combine_strategy", "blend")),
                            shift_suspicious_tld=float(gp.get("shift_suspicious_tld", -0.04)),
                            shift_many_links=float(gp.get("shift_many_links", -0.03)),
                            shift_calm_text=float(gp.get("shift_calm_text", +0.02)),
                        )
                        model = model.fit(X_tr_t, X_tr_b, y_tr)
                        model.apply_numeric_adjustments(ss["numeric_adjustments"])
                        ss["model"] = model
                        ss["split_cache"] = (X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te)
                        try:
                            gp = ss.get("guard_params", {}) or {}
                            if (
                                gp.get("assist_center_mode") == "auto"
                                and len(X_te_t)
                                and len(X_te_b)
                            ):
                                probs_te_raw = model.predict_proba(X_te_t, X_te_b)
                                probs_te = np.asarray(probs_te_raw, dtype=float)
                                if probs_te.ndim != 2 or probs_te.shape[1] < 2:
                                    raise ValueError(
                                        "predict_proba must return a 2D array with at least two columns"
                                    )
                                spam_idx = int(getattr(model, "_i_spam", 1))
                                if spam_idx < 0 or spam_idx >= probs_te.shape[1]:
                                    raise IndexError(
                                        "Spam index is out of bounds for predict_proba output"
                                    )
                                p_spam_te = np.clip(probs_te[:, spam_idx], 1e-6, 1 - 1e-6)
                                y_true = np.asarray(y_te) == "spam"
                                fallback_center = float(
                                    gp.get("assist_center", float(ss.get("threshold", 0.6)))
                                )
                                best_tau, best_f1 = fallback_center, -1.0
                                if y_true.size:
                                    for tau in np.linspace(0.30, 0.90, 61):
                                        y_pred = p_spam_te >= tau
                                        f1 = f1_score(y_true, y_pred)
                                        if f1 > best_f1:
                                            best_f1, best_tau = float(f1), float(tau)
                                gp["assist_center"] = float(best_tau)
                                ss["guard_params"] = gp
                        except Exception:
                            logger.exception(
                                "Failed to auto-select assist center from hold-out set"
                            )
                        ss["eval_timestamp"] = datetime.now().isoformat(timespec="seconds")
                        ss["eval_temp_threshold"] = float(ss.get("threshold", 0.6))
                        ss["train_story_run_id"] = uuid4().hex
                        ss["train_flash_finished"] = True
                        ss["train_refresh_expected"] = True
                        ss["train_refresh_attempts"] = 1
                        should_rerun_after_training = True
                        training_successful = True
                        for key in (
                            "meaning_map_show_examples",
                            "meaning_map_show_centers",
                            "meaning_map_highlight_borderline",
                            "meaning_map_show_pair_trigger",
                        ):
                            ss.pop(key, None)
                        if has_embed:
                            try:
                                train_texts_cache = [
                                    combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)
                                ]
                                cache_train_embeddings(train_texts_cache)
                            except Exception:
                                pass
                finally:
                    ss["train_in_progress"] = False

    if should_rerun_after_training and training_successful:
        streamlit_rerun()
        return

    parsed_split = None
    y_tr_labels = None
    y_te_labels = None
    train_texts_combined_cache: list[str] = []
    test_texts_combined_cache: list[str] = []
    lang_mix_train: Optional[Dict[str, Any]] = None
    lang_mix_test: Optional[Dict[str, Any]] = None
    lang_mix_error: Optional[str] = None
    has_model = ss.get("model") is not None
    has_split_cache = ss.get("split_cache") is not None
    if has_model and has_split_cache:
        try:
            parsed_split = _parse_split_cache(ss["split_cache"])
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr_labels, y_te_labels = parsed_split

            if X_tr_t is not None and X_tr_b is not None:
                train_texts_combined_cache = [
                    combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)
                ]
            if X_te_t is not None and X_te_b is not None:
                test_texts_combined_cache = [
                    combine_text(t, b) for t, b in zip(X_te_t, X_te_b)
                ]

            if has_langdetect:
                try:
                    lang_mix_train = summarize_language_mix(train_texts_combined_cache)
                    lang_mix_test = summarize_language_mix(test_texts_combined_cache)
                except Exception as exc:
                    lang_mix_error = str(exc) or exc.__class__.__name__
                    lang_mix_train = None
                    lang_mix_test = None
            else:
                lang_mix_error = "language detector unavailable"

            # Existing success + story (kept)
            story = make_after_training_story(y_tr_labels, y_te_labels)
            st.markdown(story)
        except Exception as exc:
            st.caption(f"Training storyboard unavailable ({exc}).")
            logger.exception("Training storyboard failed")

    if ss.get("nerd_mode_train") and ss.get("model") is not None and parsed_split:
        with st.expander("Nerd Mode — what just happened (technical)", expanded=True):
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr_labels, y_te_labels = parsed_split
            train_texts_combined: list[str] = list(train_texts_combined_cache)
            train_embeddings: Optional[np.ndarray] = None
            train_embeddings_error: Optional[str] = None
            if not train_texts_combined and X_tr_t and X_tr_b:
                train_texts_combined = [combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)]
            if train_texts_combined:
                if has_embed:
                    try:
                        train_embeddings = cache_train_embeddings(train_texts_combined)
                        if getattr(train_embeddings, "size", 0) == 0:
                            train_embeddings = None
                    except Exception as exc:
                        train_embeddings = None
                        train_embeddings_error = str(exc) or exc.__class__.__name__
                    if train_embeddings is None:
                        try:
                            train_embeddings = encode_texts(train_texts_combined)
                            if getattr(train_embeddings, "size", 0) == 0:
                                train_embeddings = None
                        except Exception as exc:
                            train_embeddings = None
                            train_embeddings_error = str(exc) or exc.__class__.__name__
                        else:
                            train_embeddings_error = None
                else:
                    train_embeddings_error = "text encoder unavailable"
            try:
                st.markdown("**Data split**")
                st.markdown(
                    f"- Train set size: {len(y_tr_labels)}  \n"
                    f"- Test set size: {len(y_te_labels)}  \n"
                    f"- Class balance (train): {_counts(list(y_tr_labels))}  \n"
                    f"- Class balance (test): {_counts(list(y_te_labels))}"
                )
            except Exception:
                st.caption("Split details unavailable.")

            if lang_mix_error:
                st.caption(f"Language mix unavailable ({lang_mix_error}).")
            elif has_langdetect:
                try:
                    render_language_mix_chip_rows(lang_mix_train, lang_mix_test)
                except Exception as exc:
                    msg = str(exc) or exc.__class__.__name__
                    st.caption(f"Language mix unavailable ({msg}).")

            centroid_distance: Optional[float] = None
            centroid_message: Optional[str] = None
            try:
                if not train_texts_combined:
                    centroid_message = "Centroid distance unavailable (no training texts)."
                elif not has_embed:
                    centroid_message = "Centroid distance unavailable (text encoder unavailable)."
                elif train_embeddings is None or getattr(train_embeddings, "size", 0) == 0:
                    detail = train_embeddings_error or "embeddings missing"
                    centroid_message = f"Centroid distance unavailable ({detail})."
                elif not y_tr_labels:
                    centroid_message = "Centroid distance unavailable (labels missing)."
                else:
                    y_train_arr = np.asarray(y_tr_labels)
                    if train_embeddings.shape[0] != y_train_arr.shape[0]:
                        centroid_message = "Centroid distance unavailable (embedding count mismatch)."
                    else:
                        spam_mask = y_train_arr == "spam"
                        safe_mask = y_train_arr == "safe"
                        if not np.any(spam_mask) or not np.any(safe_mask):
                            centroid_message = "Centroid distance requires at least one spam and one safe email."
                        else:
                            spam_centroid = train_embeddings[spam_mask].mean(axis=0)
                            safe_centroid = train_embeddings[safe_mask].mean(axis=0)
                            spam_norm = float(np.linalg.norm(spam_centroid))
                            safe_norm = float(np.linalg.norm(safe_centroid))
                            if spam_norm == 0.0 or safe_norm == 0.0:
                                centroid_message = "Centroid distance unavailable (zero-length centroid)."
                            else:
                                cosine_similarity = float(
                                    np.clip(
                                        np.dot(spam_centroid, safe_centroid)
                                        / (spam_norm * safe_norm),
                                        -1.0,
                                        1.0,
                                    )
                                )
                                centroid_distance = 1.0 - cosine_similarity
            except Exception:
                centroid_message = "Centroid distance unavailable."

            if centroid_distance is not None:
                st.metric("Centroid cosine distance", f"{centroid_distance:.2f}")
                meter_width = float(np.clip(centroid_distance, 0.0, 1.0)) * 100.0
                meter_html = f"""
                <div style="margin-top:-0.5rem; margin-bottom:0.75rem;">
                    <div style="background:rgba(49, 51, 63, 0.1); border-radius:999px; height:10px; width:100%;">
                        <div style="background:linear-gradient(90deg, #4ade80, #22c55e); border-radius:999px; height:100%; width:{meter_width:.0f}%;"></div>
                    </div>
                    <div style="font-size:0.7rem; color:rgba(49,51,63,0.6); margin-top:0.25rem;">0 = identical • 1 = orthogonal</div>
                </div>
                """
                st.markdown(meter_html, unsafe_allow_html=True)
                st.caption(
                    "Bigger distance means spam and safe live farther apart in meaning space—good separation."
                )
            elif centroid_message:
                st.caption(centroid_message)

            st.markdown("#### Decision margin spread (text head)")
            margins: Optional[np.ndarray] = None
            margin_error = False
            model_obj = ss.get("model")
            if not train_texts_combined:
                st.caption("Margin distribution unavailable (no training texts).")
            else:
                logits: Optional[np.ndarray] = None
                try:
                    if hasattr(model_obj, "predict_logit"):
                        logits = np.asarray(model_obj.predict_logit(train_texts_combined), dtype=float)
                    if logits is None or logits.size == 0:
                        probs = model_obj.predict_proba(X_tr_t, X_tr_b)[:, getattr(model_obj, "_i_spam", 1)]
                        probs = np.clip(probs, 1e-6, 1 - 1e-6)
                        logits = np.log(probs / (1.0 - probs))
                    logits = np.asarray(logits, dtype=float).reshape(-1)
                    logits = logits[np.isfinite(logits)]
                    if logits.size > 0:
                        margins = np.abs(logits)
                except Exception as exc:
                    st.caption(f"Margin distribution unavailable: {exc}")
                    margins = None
                    margin_error = True

            if margins is not None and margins.size > 0:
                try:
                    bins = min(12, max(5, int(np.ceil(np.log2(margins.size + 1)))))
                    counts, edges = np.histogram(margins, bins=bins)
                    labels = [
                        f"{edges[i]:.2f}–{edges[i + 1]:.2f}" for i in range(len(edges) - 1)
                    ]
                    hist_df = pd.DataFrame({"margin": labels, "count": counts})
                    st.bar_chart(hist_df.set_index("margin"), width="stretch")
                    st.caption(
                        "Higher margins = clearer decisions; lots of small margins means many borderline emails."
                    )
                except Exception as exc:
                    st.caption(f"Could not render margin histogram: {exc}")
            elif train_texts_combined and not margin_error:
                st.caption("Margin distribution unavailable (no valid logit values).")

            params = ss.get("train_params", {})
            st.markdown("**Parameters used**")
            st.markdown(
                f"- Hold-out fraction: {params.get('test_size', '—')}  \n"
                f"- Random seed: {params.get('random_state', '—')}  \n"
                f"- Max iterations: {params.get('max_iter', '—')}  \n"
                f"- C (inverse regularization): {params.get('C', '—')}"
            )

            calibrate_default = bool(ss.get("calibrate_probabilities", False))
            calib_toggle = st.toggle(
                "Calibrate probabilities (test set)",
                value=calibrate_default,
                key="train_calibrate_toggle",
                help="Platt scaling if test size ≥ 30, else isotonic disabled.",
                disabled=not has_calibration,
            )
            calib_active = bool(calib_toggle and has_calibration)
            ss["calibrate_probabilities"] = bool(calib_active)

            calibration_details = None
            if not has_calibration:
                st.caption("Unavailable")
                if hasattr(model_obj, "set_calibration"):
                    try:
                        model_obj.set_calibration(None)
                    except Exception:
                        pass
                ss["calibration_result"] = None
            elif calib_active:
                test_size = len(y_te_labels) if y_te_labels is not None else 0
                if test_size < 30:
                    st.caption("Unavailable")
                    if hasattr(model_obj, "set_calibration"):
                        try:
                            model_obj.set_calibration(None)
                        except Exception:
                            pass
                    calibration_details = {"status": "too_small", "test_size": test_size}
                    ss["calibration_result"] = calibration_details
                elif model_obj is None:
                    st.caption("Unavailable")
                else:
                    try:
                        spam_index = getattr(model_obj, "_i_spam", 1)
                        if hasattr(model_obj, "predict_proba_base"):
                            base_matrix = model_obj.predict_proba_base(X_te_t, X_te_b)
                        else:
                            base_matrix = model_obj.predict_proba(X_te_t, X_te_b)
                        base_matrix = np.asarray(base_matrix, dtype=float)
                        if base_matrix.ndim != 2 or base_matrix.shape[0] == 0:
                            raise ValueError("Empty probability matrix from model.")
                        base_probs = base_matrix[:, spam_index]
                        base_probs = np.clip(base_probs, 1e-6, 1 - 1e-6)
                        y_true01 = np.asarray(_y01(list(y_te_labels)), dtype=float)
                        base_logits = np.log(base_probs / (1.0 - base_probs))
                        calibrator = PlattProbabilityCalibrator(
                            random_state=int(params.get("random_state", 42))
                        )
                        calibrator.fit(base_logits, list(y_te_labels))
                        if hasattr(model_obj, "set_calibration"):
                            model_obj.set_calibration(calibrator)
                        calibrated_matrix = model_obj.predict_proba(X_te_t, X_te_b)
                        calibrated_matrix = np.asarray(calibrated_matrix, dtype=float)
                        calibrated_probs = calibrated_matrix[:, spam_index]
                        calibrated_probs = np.clip(calibrated_probs, 1e-6, 1 - 1e-6)
                        brier_before = float(np.mean((base_probs - y_true01) ** 2))
                        brier_after = float(np.mean((calibrated_probs - y_true01) ** 2))
                        bins = np.linspace(0.0, 1.0, 11)
                        reliability_rows: List[Dict[str, object]] = []
                        stages = [
                            ("Before calibration", base_probs),
                            ("After calibration", calibrated_probs),
                        ]
                        for stage_label, probs in stages:
                            bin_ids = np.digitize(probs, bins, right=True) - 1
                            bin_ids = np.clip(bin_ids, 0, len(bins) - 2)
                            for b in range(len(bins) - 1):
                                mask = bin_ids == b
                                if not np.any(mask):
                                    continue
                                reliability_rows.append(
                                    {
                                        "stage": stage_label,
                                        "bin": b,
                                        "expected": float(np.mean(probs[mask])),
                                        "observed": float(np.mean(y_true01[mask])),
                                        "count": int(mask.sum()),
                                    }
                                )
                        reliability_df = pd.DataFrame(reliability_rows)
                        reliability_df["bin_label"] = reliability_df["bin"].map(
                            lambda b: f"{bins[b]:.1f}–{bins[b + 1]:.1f}"
                        )
                        calibration_details = {
                            "status": "ok",
                            "brier_before": brier_before,
                            "brier_after": brier_after,
                            "test_size": test_size,
                            "reliability": reliability_df,
                        }
                        ss["calibration_result"] = calibration_details
                    except Exception:
                        st.caption("Unavailable")
                        if hasattr(model_obj, "set_calibration"):
                            try:
                                model_obj.set_calibration(None)
                            except Exception:
                                pass
                        calibration_details = {"status": "error"}
                        ss["calibration_result"] = calibration_details
            else:
                if hasattr(model_obj, "set_calibration"):
                    try:
                        model_obj.set_calibration(None)
                    except Exception:
                        pass
                ss["calibration_result"] = None

            if calibration_details and calibration_details.get("status") == "ok":
                brier_before = calibration_details["brier_before"]
                brier_after = calibration_details["brier_after"]
                delta = brier_after - brier_before
                col_b1, col_b2 = st.columns(2)
                col_b1.metric("Brier score (uncalibrated)", f"{brier_before:.4f}")
                col_b2.metric(
                    "Brier score (calibrated)",
                    f"{brier_after:.4f}",
                    delta=f"{delta:+.4f}",
                    delta_color="inverse",
                )
                st.caption(
                    f"Calibrated on {calibration_details['test_size']} hold-out examples using Platt scaling."
                )
                reliability_df = calibration_details.get("reliability")
                if reliability_df is not None and not reliability_df.empty:
                    chart = (
                        alt.Chart(reliability_df)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("expected:Q", title="Mean predicted probability"),
                            y=alt.Y("observed:Q", title="Observed spam rate"),
                            color=alt.Color("stage:N", title=""),
                            tooltip=[
                                alt.Tooltip("stage:N", title="Series"),
                                alt.Tooltip("bin_label:N", title="Bin"),
                                alt.Tooltip("expected:Q", title="Predicted", format=".2f"),
                                alt.Tooltip("observed:Q", title="Observed", format=".2f"),
                                alt.Tooltip("count:Q", title="Count"),
                            ],
                        )
                        .properties(height=260)
                    )
                    diagonal = alt.Chart(
                        pd.DataFrame({"expected": [0.0, 1.0], "observed": [0.0, 1.0]})
                    ).mark_line(strokeDash=[4, 4], color="gray")
                    st.altair_chart(chart + diagonal, use_container_width=True)
                    st.caption(
                        "We align predicted probabilities to reality. If the curve hugs the diagonal, scores are trustworthy."
                    )

            st.markdown(f"**Model object**: `{model_kind_string(ss['model'])}`")

            st.markdown("### Interpretability & tuning")
            try:
                coef_details = ss["model"].numeric_feature_details().copy()
                coef_details["friendly_name"] = coef_details["feature"].map(
                    FEATURE_DISPLAY_NAMES
                )
                st.caption(
                    "Positive weights push toward the **spam** class; negative toward **safe**. "
                    "Values are log-odds after standardization."
                )

                chart_data = (
                    coef_details.sort_values("weight_per_std", ascending=True)
                    .set_index("friendly_name")["weight_per_std"]
                )
                st.bar_chart(chart_data, width="stretch")
                st.caption(
                    "Bars to the right push toward 'Spam'; left bars push toward 'Safe'. Longer bar = stronger nudge."
                )

                display_df = coef_details.assign(
                    odds_multiplier_plus_1sigma=coef_details["odds_multiplier_per_std"],
                    approx_pct_change_odds=(coef_details["odds_multiplier_per_std"] - 1.0) * 100.0,
                )[
                    [
                        "friendly_name",
                        "base_weight_per_std",
                        "user_adjustment",
                        "weight_per_std",
                        "odds_multiplier_plus_1sigma",
                        "approx_pct_change_odds",
                        "train_mean",
                        "train_std",
                    ]
                ]

                st.dataframe(
                    display_df.rename(
                        columns={
                            "friendly_name": "Feature",
                            "base_weight_per_std": "Learned log-odds (+1σ)",
                            "user_adjustment": "Your adjustment (+1σ)",
                            "weight_per_std": "Adjusted log-odds (+1σ)",
                            "odds_multiplier_plus_1sigma": "Adjusted odds multiplier (+1σ)",
                            "approx_pct_change_odds": "%Δ odds from adjustment (+1σ)",
                            "train_mean": "Train mean",
                            "train_std": "Train std",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "Base weights come from training. Use the sliders below to nudge each cue if your domain knowledge "
                    "suggests it should count more or less. Adjustments apply per standard deviation of the raw feature."
                )

                st.markdown("#### What influenced the score (span knockout demo)")
                if not train_texts_combined:
                    st.caption("Span influence demo unavailable (no training emails).")
                elif model_obj is None or not hasattr(model_obj, "predict_logit"):
                    st.caption("Span influence demo unavailable (text-head logits missing).")
                else:
                    options = list(range(len(train_texts_combined)))

                    def _format_train_option(i: int) -> str:
                        label = (
                            y_tr_labels[i]
                            if y_tr_labels and 0 <= i < len(y_tr_labels)
                            else "?"
                        )
                        subject = X_tr_t[i] if X_tr_t and 0 <= i < len(X_tr_t) else ""
                        if not isinstance(subject, str) or not subject.strip():
                            subject = train_texts_combined[i][:80]
                        subject_short = _shorten_text(str(subject).strip(), limit=80)
                        return f"{i + 1}. [{label.upper()}] {subject_short}" if label else subject_short

                    selected_idx = st.selectbox(
                        "Pick a training email",
                        options,
                        format_func=_format_train_option,
                        key="nerd_span_email_index",
                    )

                    selected_text = train_texts_combined[selected_idx]
                    candidate_spans = extract_candidate_spans(selected_text)

                    if not candidate_spans:
                        st.info("No influential spans detected for this email.")
                    else:
                        cache_bucket = ss.setdefault("nerd_span_cache", {})
                        text_hash = hashlib.sha1(selected_text.encode("utf-8")).hexdigest()
                        cache_key = f"{id(model_obj)}:{text_hash}"
                        cached = cache_bucket.get(cache_key)

                        if not cached:
                            try:
                                base_logit = float(model_obj.predict_logit([selected_text])[0])
                            except Exception as exc:
                                base_logit = None
                                cached = {"error": str(exc)}
                            else:
                                influence_rows: list[dict[str, float | str]] = []
                                for span_text, (start, end) in candidate_spans:
                                    modified = selected_text[:start] + selected_text[end:]
                                    try:
                                        new_logit = float(
                                            model_obj.predict_logit([modified])[0]
                                        )
                                    except Exception:
                                        continue
                                    delta = base_logit - new_logit
                                    influence_rows.append(
                                        {
                                            "span": span_text,
                                            "delta": delta,
                                        }
                                    )

                                influence_rows.sort(
                                    key=lambda row: row.get("delta", 0.0), reverse=True
                                )
                                cached = {
                                    "base_logit": base_logit,
                                    "rows": influence_rows,
                                }
                            cache_bucket[cache_key] = cached

                        if cached.get("error"):
                            st.caption(
                                "Could not compute span influence: "
                                f"{cached['error']}"
                            )
                        else:
                            base_logit = cached.get("base_logit")
                            rows = cached.get("rows", [])
                            positive_rows = [
                                row
                                for row in rows
                                if isinstance(row.get("delta"), (int, float))
                                and float(row["delta"]) > 0.0
                            ]

                            if base_logit is not None:
                                st.caption(
                                    f"Base text-head logit: {float(base_logit):+.3f}"
                                )

                            if not positive_rows:
                                st.info(
                                    "Removing detected spans did not lower the score."
                                )
                            else:
                                top_rows = positive_rows[:8]
                                display_rows = []
                                for row in top_rows:
                                    span_text = str(row.get("span", "")).replace("\n", " ")
                                    span_text = " ".join(span_text.split())
                                    if len(span_text) > 120:
                                        span_text = span_text[:117] + "…"
                                    display_rows.append(
                                        {
                                            "Span": span_text,
                                            "Δ logit (drop)": round(float(row["delta"]), 4),
                                        }
                                    )

                                df_spans = pd.DataFrame(display_rows)
                                st.dataframe(df_spans, hide_index=True, width="stretch")
                                st.caption(
                                    "We remove phrases and see how the score drops; bigger drops = more influence."
                                )

            except Exception as e:
                st.caption(f"Coefficients unavailable: {e}")
