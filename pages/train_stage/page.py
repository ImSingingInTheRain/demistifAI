from __future__ import annotations

import logging
import math
import random
import hashlib
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
import streamlit as st

from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from demistifai.ui.components.arch_demai import render_demai_architecture
from demistifai.constants import STAGE_BY_KEY, STAGE_INDEX, StageMeta
from demistifai.core.language import HAS_LANGDETECT
from demistifai.core.nav import render_stage_top_grid
from demistifai.core.state import ensure_state, validate_invariants
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

from pages.train_helpers import (
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
from demistifai.ui.components.train_animation import build_training_animation_column
from demistifai.ui.components.train_intro import (
    build_inline_note,
    build_launchpad_card,
    build_launchpad_status_item,
    build_token_chip,
    build_train_intro_card,
    build_training_notes_column,
    build_nerd_intro_card,
    training_stage_stylesheet,
)
from demistifai.ui.primitives import shorten_text

from .callbacks import (
    callable_or_attr,
    go_to_prepare,
    label_balance_status,
    parse_split_cache,
    pii_status,
    prime_session_state_from_store,
    request_meaning_map_refresh,
    sync_training_artifacts,
)
from .navigation import render_prepare_dataset_prompt, render_train_terminal_slot
from .visualizations import build_calibration_chart


logger = logging.getLogger(__name__)
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

    render_stage_top_grid("train", left_renderer=render_train_terminal_slot)

    if prepared_df is None:
        render_prepare_dataset_prompt(stage, section_surface, set_active_stage)
        return

    prime_session_state_from_store(ss, model_state, split_state)

    animation_column = build_training_animation_column()
    notes_column = build_training_notes_column()

    render_mac_window(
        st,
        title="Training dynamics monitor",
        subtitle="MiniLM organising emails by meaning",
        columns=2,
        ratios=(0.3, 0.7),
        col_html=[notes_column.html, animation_column.html],
        id_suffix="train-animation",
        fallback_height=animation_column.fallback_height,
        scoped_css=notes_column.css,
    )

    has_embed = callable_or_attr(encode_texts)

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
        try:
            sync_training_artifacts(
                prepared_df,
                ss=ss,
                run_state=run_state,
                data_state=data_state,
                split_state=split_state,
                model_state=model_state,
            )
        except RuntimeError as exc:
            logger.exception("Failed to update cached training artifacts")
            detail = exc.__cause__ or exc
            st.error(f"Failed to update training artifacts: {detail}")


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

    st.markdown(training_stage_stylesheet(), unsafe_allow_html=True)

    stage_number = STAGE_INDEX.get(stage.key, 0) - STAGE_INDEX.get("data", 0) + 1
    if stage_number < 1:
        stage_number = STAGE_INDEX.get(stage.key, 0) + 1

    nerd_mode_flag = bool(ss.get("nerd_mode_train") or ss.get("nerd_mode"))


    nerd_mode_train_enabled = bool(ss.get("nerd_mode_train"))

    with section_surface():

        render_eu_ai_quote("An AI system “infers, from the input it receives…”.")

        intro_col, launchpad_col = st.columns([0.58, 0.42], gap="large")

        with intro_col:
            st.markdown(
                build_train_intro_card(
                    stage_number=stage_number,
                    icon=stage.icon,
                    title=stage.title,
                ),
                unsafe_allow_html=True,
            )
        with launchpad_col:
            st.markdown(build_launchpad_card(), unsafe_allow_html=True)

            bal = label_balance_status(ss.get("labeled"))
            bal_ok = bal["ok"]
            bal_chip = "✅ OK" if bal_ok else "⚠️ Need work"
            balance_body = (
                f"Spam: {bal['counts']['spam']} • Safe: {bal['counts']['safe']} (ratio ~{bal['ratio']:.2f})"
            )
            st.markdown(
                build_launchpad_status_item(
                    title="Balanced labels",
                    status=bal_chip,
                    body=balance_body,
                ),
                unsafe_allow_html=True,
            )
            if not bal_ok:
                if st.button("Fix in Prepare", key="launchpad_fix_balance", use_container_width=True):
                    go_to_prepare(ss)

            pii = pii_status(ss)
            tag = {
                "clean": "✅ OK",
                "found": "⚠️ Need work",
                "unknown": "ⓘ Not scanned",
            }.get(pii["status"], "ⓘ Not scanned")
            counts_str = ", ".join(f"{k} {v}" for k, v in (pii["counts"] or {}).items()) or "—"
            st.markdown(
                build_launchpad_status_item(
                    title="Data hygiene",
                    status=tag,
                    body=f"PII in preview: {counts_str}",
                ),
                unsafe_allow_html=True,
            )
            if pii["status"] in {"found", "unknown"}:
                if st.button("Review in Prepare", key="launchpad_fix_pii", use_container_width=True):
                    go_to_prepare(ss)

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
            st.markdown(build_nerd_intro_card(), unsafe_allow_html=True)
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
            build_token_chip(token_budget_text),
            unsafe_allow_html=True,
        )
        st.markdown(
            build_inline_note("Tip: long emails will be clipped; summaries help."),
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
        request_meaning_map_refresh(ss, section, streamlit_rerun)

    _render_unified_training_storyboard(
        ss,
        has_model=has_model,
        has_split=has_split_cache,
        has_embed=bool(has_embed),
        section_surface=section_surface,
        request_meaning_map_refresh=_request_refresh,
        parse_split_cache=parse_split_cache,
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
            parsed_split = parse_split_cache(ss["split_cache"])
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
                chart = build_calibration_chart(reliability_df)
                if chart is not None:
                    st.altair_chart(chart, use_container_width=True)
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
                        subject_short = shorten_text(str(subject).strip(), limit=80)
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
