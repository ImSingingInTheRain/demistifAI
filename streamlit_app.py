from __future__ import annotations

import html
import json
import logging
import random
from collections import Counter
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from uuid import uuid4

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.delta_generator import DeltaGenerator

from demistifai.constants import STAGES
from demistifai.core.state import ensure_state, validate_invariants, hash_dict
from demistifai.core.utils import streamlit_rerun

from demistifai.constants import (
    APP_THEME_CSS,
    AUTONOMY_LEVELS,
    EMAIL_INBOX_TABLE_CSS,
    STAGE_BY_KEY,
    STAGE_TEMPLATE_CSS,
)
from demistifai.core.guardrails import (
    GUARDRAIL_LABEL_ICONS,
    _guardrail_signals,
    _guardrail_badges_html,
)
from demistifai.core.audit import _append_audit
from demistifai.core.export import _export_batch_df
from demistifai.core.language import (
    summarize_language_mix,
    render_language_mix_chip_rows,
)
from demistifai.core.nav import render_stage_top_grid
from demistifai.core.state import (
    _set_advanced_knob_state,
    _apply_pending_advanced_knob_state,
)
from demistifai.dataset import (
    DEFAULT_DATASET_CONFIG,
    compute_dataset_summary,
    generate_incoming_batch,
    starter_dataset_copy,
)
from demistifai.modeling import (
    FEATURE_ORDER,
    HybridEmbedFeatsLogReg,
    PlattProbabilityCalibrator,
    _fmt_delta,
    _fmt_pct,
    _pr_acc_cm,
    _predict_proba_batch,
    _y01,
    _counts,
    cache_train_embeddings,
    combine_text,
    encode_texts,
    compute_confusion,
    get_nearest_training_examples,
    make_after_eval_story,
    numeric_feature_contributions,
    plot_threshold_curves,
    threshold_presets,
    top_token_importances,
    verdict_label,
)
from demistifai.components.guardrail_panel import render_guardrail_panel
from demistifai.core.downloads import download_text

from pages.data import render_data_stage as render_data_stage_content
from stages.overview import render_overview_stage as render_overview_stage_content
from stages.train_stage import render_train_stage_page
from demistifai.ui.custom_header import mount_demai_header
from demistifai.ui.components.terminal.evaluate import render_evaluate_terminal
from demistifai.ui.components.terminal.use import render_use_terminal
from welcome import render_intro_stage as render_intro_stage_content
logger = logging.getLogger(__name__)

st.set_page_config(page_title="demistifAI", page_icon="📧", layout="wide")

state = ensure_state()
s = state
ss = st.session_state

ss.setdefault("viewport_is_mobile", False)

from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


def _shorten_text(text: str, limit: int = 120) -> str:
    """Return a shortened version of *text* capped at *limit* characters."""

    if len(text) <= limit:
        return text
    return f"{text[: limit - 1]}…"


def _safe_subject(row: dict) -> str:
    return str(row.get("title", "") or "").strip()

st.markdown(APP_THEME_CSS, unsafe_allow_html=True)
st.markdown(STAGE_TEMPLATE_CSS, unsafe_allow_html=True)
st.markdown(EMAIL_INBOX_TABLE_CSS, unsafe_allow_html=True)
mount_demai_header()


@contextmanager
def section_surface(extra_class: Optional[str] = None):
    """Render a consistently styled section surface container."""

    base_class = "section-surface"
    classes = f"{base_class} {extra_class}" if extra_class else base_class

    st.markdown(f'<div class="{classes}">', unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


def guidance_popover(title: str, text: str):
    with st.popover(f"❓ {title}"):
        st.write(text)


def eu_ai_quote_box(text: str, label: str = "EU AI Act") -> str:
    escaped_text = html.escape(text)
    escaped_label = html.escape(label)
    return (
        """
        <div class="ai-quote-box">
            <div class="ai-quote-box__icon">⚖️</div>
            <div class="ai-quote-box__content">
                <span class="ai-quote-box__source">{label}</span>
                <p>{text}</p>
            </div>
        </div>
        """
        .format(label=escaped_label, text=escaped_text)
    )


def render_eu_ai_quote(text: str, label: str = "From the EU AI Act, Article 3") -> None:
    st.markdown(eu_ai_quote_box(text, label), unsafe_allow_html=True)


def render_nerd_mode_toggle(
    *,
    key: str,
    title: str,
    description: Optional[str] = None,
    icon: Optional[str] = "🧠",
    target: DeltaGenerator | None = None,
) -> bool:
    """Render a consistently styled Nerd Mode toggle block."""

    toggle_label = f"{icon} {title}" if icon else title
    wrapper = target.container() if target is not None else st.container()
    default_state = bool(ss.get(key, False))
    icon_html = f"<span class='nerd-toggle__icon'>{html.escape(icon)}</span>" if icon else ""
    safe_title = html.escape(title)
    safe_description = html.escape(description) if description else ""

    with wrapper:
        content_col, toggle_col = st.columns([1, 0.32], gap="large")
        with content_col:
            st.markdown(
                f"<div class='nerd-toggle__title'>{icon_html}<span class='nerd-toggle__title-text'>{safe_title}</span></div>",
                unsafe_allow_html=True,
            )
            if description:
                st.markdown(
                    f"<div class='nerd-toggle__description'>{safe_description}</div>",
                    unsafe_allow_html=True,
                )
        with toggle_col:
            value = st.toggle(
                toggle_label,
                key=key,
                value=default_state,
                label_visibility="collapsed",
            )

    return value


def render_email_inbox_table(
    df: pd.DataFrame,
    *,
    title: str,
    subtitle: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> None:
    """Display a small email-centric table with shared styling."""

    with st.container(border=True):
        st.markdown(f"**{title}**")
        if subtitle:
            st.caption(subtitle)

        if df is None or df.empty:
            st.caption("No emails to display.")
            return

        display_df = df.copy()
        if columns:
            existing = [col for col in columns if col in display_df.columns]
            if existing:
                display_df = display_df[existing]

        st.dataframe(display_df, hide_index=True, width="stretch")


def render_mailbox_panel(
    messages: Optional[List[Dict[str, Any]]],
    *,
    mailbox_title: str,
    filled_subtitle: str,
    empty_subtitle: str,
) -> None:
    """Render a mailbox tab with consistent styling and fallbacks."""

    with st.container(border=True):
        st.markdown(f"**{mailbox_title}**")
        records = messages or []
        if not records:
            st.caption(empty_subtitle)
            return

        st.caption(filled_subtitle)
        df_box = pd.DataFrame(records)
        column_order = ["title", "pred", "p_spam", "body"]
        rename_map = {
            "title": "Title",
            "pred": "Predicted",
            "p_spam": "P(spam)",
            "body": "Body",
        }
        existing = [col for col in column_order if col in df_box.columns]
        if existing:
            df_display = df_box[existing].rename(columns=rename_map)
        else:
            df_display = df_box
        st.dataframe(df_display, hide_index=True, width="stretch")


_apply_pending_advanced_knob_state()
requested_stage_values = st.query_params.get_all("stage")
requested_stage = requested_stage_values[0] if requested_stage_values else None
run_state = s.setdefault("run", {})
if requested_stage in STAGE_BY_KEY and requested_stage != run_state.get("active_stage"):
    run_state["active_stage"] = requested_stage

if not run_state.get("active_stage") and STAGES:
    run_state["active_stage"] = STAGES[0].key

default_stage_key = run_state.get("active_stage")
selected_stage_key = ss.get("active_stage", default_stage_key)

if selected_stage_key not in STAGE_BY_KEY:
    if STAGES:
        selected_stage_key = STAGES[0].key
    else:
        selected_stage_key = None

if default_stage_key not in STAGE_BY_KEY:
    default_stage_key = selected_stage_key

if (
    run_state.get("busy")
    and default_stage_key in STAGE_BY_KEY
    and selected_stage_key not in (None, default_stage_key)
):
    selected_stage_key = default_stage_key
    ss["active_stage"] = default_stage_key
    toast_message = "Training in progress — navigation disabled."
    ui_state = s.setdefault("ui", {})
    toasts = ui_state.setdefault("toasts", [])
    if toast_message not in toasts:
        toasts.append(toast_message)
        st.toast(toast_message, icon="⏳")

if selected_stage_key is not None:
    if selected_stage_key != run_state.get("active_stage"):
        ss["stage_scroll_to_top"] = True
    run_state["active_stage"] = selected_stage_key
    ss["active_stage"] = selected_stage_key
    if st.query_params.get_all("stage") != [selected_stage_key]:
        st.query_params["stage"] = selected_stage_key

ss.setdefault("nerd_mode", False)
ss.setdefault("autonomy", AUTONOMY_LEVELS[0])
ss.setdefault("threshold", 0.6)
ss.setdefault("nerd_mode_eval", False)
ss.setdefault("eval_timestamp", None)
ss.setdefault("eval_temp_threshold", float(ss["threshold"]))
ss.setdefault("adaptive", True)
ss.setdefault("labeled", starter_dataset_copy())      # list of dicts: title, body, label
if "incoming_seed" not in ss:
    ss["incoming_seed"] = None
if not ss.get("incoming"):
    seed = random.randint(1, 1_000_000)
    ss["incoming_seed"] = seed
    ss["incoming"] = generate_incoming_batch(n=30, seed=seed, spam_ratio=0.32)
ss.setdefault("model", None)
ss.setdefault("split_cache", None)
ss.setdefault("mail_inbox", [])  # list of dicts: title, body, pred, p_spam
ss.setdefault("mail_spam", [])
ss.setdefault("metrics", {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
ss.setdefault("last_classification", None)
ss.setdefault("numeric_adjustments", {feat: 0.0 for feat in FEATURE_ORDER})
ss.setdefault("nerd_mode_data", False)
ss.setdefault("nerd_mode_train", False)
ss.setdefault("calibrate_probabilities", False)
ss.setdefault("calibration_result", None)
ss.setdefault(
    "train_params",
    {"test_size": 0.30, "random_state": 42, "max_iter": 1000, "C": 1.0},
)
ss.setdefault(
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
)
ss.setdefault("use_high_autonomy", ss.get("autonomy", AUTONOMY_LEVELS[0]).startswith("High"))
ss.setdefault("train_story_run_id", None)
ss.setdefault("use_batch_results", [])
ss.setdefault("use_adaptiveness", bool(ss.get("adaptive", True)))
ss.setdefault("use_audit_log", [])
ss.setdefault("nerd_mode_use", False)
ss.setdefault("dataset_config", DEFAULT_DATASET_CONFIG.copy())
_set_advanced_knob_state(ss["dataset_config"], force=False)
if "dataset_summary" not in ss:
    ss["dataset_summary"] = compute_dataset_summary(ss["labeled"])
ss.setdefault("dataset_last_built_at", datetime.now().isoformat(timespec="seconds"))
ss.setdefault("previous_dataset_summary", None)
ss.setdefault("dataset_preview", None)
ss.setdefault("dataset_preview_config", None)
ss.setdefault("dataset_preview_summary", None)
ss.setdefault("dataset_manual_queue", None)
ss.setdefault("dataset_controls_open", False)
ss.setdefault("dataset_has_generated_once", False)
ss.setdefault("datasets", [])
ss.setdefault("active_dataset_snapshot", None)
ss.setdefault("dataset_snapshot_name", "")
ss.setdefault("last_dataset_delta_story", None)
ss.setdefault("dataset_compare_delta", None)
ss.setdefault("dataset_preview_lint", None)
ss.setdefault("last_eval_results", None)


def set_active_stage(stage_key: str) -> None:
    """Update the active stage and synchronize related navigation state."""

    if stage_key not in STAGE_BY_KEY:
        return

    run_state = s.setdefault("run", {})
    current_stage = run_state.get("active_stage")
    if current_stage == stage_key:
        return

    run_state["active_stage"] = stage_key
    ss["active_stage"] = stage_key
    ss["stage_scroll_to_top"] = True

    # Mirror the active stage in the URL query parameter for deep-linking and
    # to support refresh persistence.
    if st.query_params.get_all("stage") != [stage_key]:
        st.query_params["stage"] = stage_key


def _set_adaptive_state(new_value: bool, *, source: str) -> None:
    """Synchronize adaptiveness settings across UI controls."""

    current_value = bool(ss.get("adaptive", False))
    desired_value = bool(new_value)
    if desired_value == current_value:
        return

    ss["adaptive"] = desired_value
    ss["use_adaptiveness"] = desired_value

    if source != "stage":
        ss.pop("adaptive_stage", None)


ss["use_adaptiveness"] = bool(ss.get("adaptive", False))


def render_intro_stage():

    render_intro_stage_content(section_surface=section_surface)


def render_overview_stage():
    render_overview_stage_content(
        ss,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        section_surface=section_surface,
    )



def render_data_stage():
    render_data_stage_content(
        section_surface=section_surface,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
    )


def render_evaluate_stage():
    s = ensure_state()
    validate_invariants(s)

    stage = STAGE_BY_KEY["evaluate"]

    def _render_evaluate_terminal(slot: DeltaGenerator) -> None:
        with slot:
            render_evaluate_terminal()

    render_stage_top_grid("evaluate", left_renderer=_render_evaluate_terminal)

    model_state = s.setdefault("model", {})
    data_state = s.setdefault("data", {})
    split_state = s.setdefault("split", {})

    if model_state.get("status") != "trained":
        with section_surface():
            st.subheader(f"{stage.icon} {stage.title} — How well does your spam detector perform?")
            st.info("Model not trained or stale — please (re)train.")
        return

    model_obj = model_state.get("clf")
    if model_obj is not None and ss.get("model") is None:
        ss["model"] = model_obj

    if not ss.get("split_cache"):
        x_train_payload = split_state.get("X_train") or {}
        x_test_payload = split_state.get("X_test") or {}
        y_train_payload = split_state.get("y_train") or []
        y_test_payload = split_state.get("y_test") or []
        if any(
            [
                x_train_payload,
                x_test_payload,
                y_train_payload,
                y_test_payload,
            ]
        ):
            ss["split_cache"] = (
                list(x_train_payload.get("titles") or []),
                list(x_test_payload.get("titles") or []),
                list(x_train_payload.get("bodies") or []),
                list(x_test_payload.get("bodies") or []),
                list(y_train_payload or []),
                list(y_test_payload or []),
            )

    if not (ss.get("model") and ss.get("split_cache")):
        with section_surface():
            st.subheader(f"{stage.icon} {stage.title} — How well does your spam detector perform?")
            st.info("Train a model first in the **Train** tab.")
        return

    nerd_flag = bool(ss.get("nerd_mode_eval") or ss.get("nerd_mode"))

    cache = ss["split_cache"]
    if len(cache) == 4:
        X_tr, X_te, y_tr, y_te_raw = cache
        texts_test = X_te
        X_te_t = X_te_b = None
    else:
        X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te_raw = cache
        texts_test = [(t or "") + "\n" + (b or "") for t, b in zip(X_te_t, X_te_b)]

    y_test_labels = list(y_te_raw)

    metrics_state = model_state.get("metrics")
    if not isinstance(metrics_state, dict):
        metrics_state = {}
        model_state["metrics"] = metrics_state

    model_params = dict(model_state.get("params") or {})
    data_hash = data_state.get("hash", "") or ""
    lib_versions = {"numpy": np.__version__, "sklearn": sklearn_version}
    metrics_key = hash_dict(
        {
            "data_hash": data_hash,
            "model_params": model_params,
            "lib_versions": lib_versions,
        }
    )

    cached_p_spam = metrics_state.get("p_spam")
    cached_y_true = metrics_state.get("y_true")
    metrics_valid = (
        metrics_state.get("cache_key") == metrics_key
        and cached_p_spam is not None
        and cached_y_true is not None
    )

    if not metrics_valid:
        model_for_eval = ss["model"]
        try:
            if len(cache) == 6:
                probs = model_for_eval.predict_proba(X_te_t, X_te_b)
            else:
                probs = model_for_eval.predict_proba(texts_test)
        except TypeError:
            probs = model_for_eval.predict_proba(texts_test)

        classes = list(getattr(model_for_eval, "classes_", []))
        if classes and "spam" in classes:
            idx_spam = classes.index("spam")
        else:
            idx_spam = 1 if probs.shape[1] > 1 else 0
        p_spam_arr = probs[:, idx_spam]

        metrics_state = {
            "cache_key": metrics_key,
            "computed_at": datetime.now().isoformat(timespec="seconds"),
            "p_spam": p_spam_arr.tolist(),
            "y_true": list(y_test_labels),
            "classes": classes,
            "data_hash": data_hash,
            "model_params": model_params,
            "lib_versions": lib_versions,
        }
        model_state["metrics"] = metrics_state
    else:
        classes = list(metrics_state.get("classes", []))

    p_spam = np.asarray(metrics_state.get("p_spam", []), dtype=float)
    y_te = list(metrics_state.get("y_true", y_test_labels))
    y_true01 = _y01(list(y_te))

    current_thr = float(ss.get("threshold", 0.5))
    cm = compute_confusion(y_true01, p_spam, current_thr)
    acc = (cm["TP"] + cm["TN"]) / max(1, len(y_true01))
    emoji, verdict = verdict_label(acc, len(y_true01))
    prev_eval = ss.get("last_eval_results") or {}
    acc_cur, p_cur, r_cur, f1_cur, cm_cur = _pr_acc_cm(y_true01, p_spam, current_thr)

    with section_surface():
        narrative_col, metrics_col = st.columns([3, 2], gap="large")
        with narrative_col:
            st.subheader(f"{stage.icon} {stage.title} — How well does your spam detector perform?")
            st.write(
                "Now that your model has learned from examples, it’s time to test how well it works. "
                "During training, we kept some emails aside — the **test set**. The model hasn’t seen these before. "
                "By checking its guesses against the true labels, we get a fair measure of performance."
            )
            st.markdown("### What do these results say?")
            st.markdown(make_after_eval_story(len(y_true01), cm))
        with metrics_col:
            st.markdown("### Snapshot")
            st.success(f"**Accuracy:** {acc:.2%}  |  {emoji} {verdict}")
            st.caption(f"Evaluated on {len(y_true01)} unseen emails at threshold {current_thr:.2f}.")
            st.markdown(
                "- ✅ Spam caught: **{tp}**\n"
                "- ❌ Spam missed: **{fn}**\n"
                "- ⚠️ Safe mis-flagged: **{fp}**\n"
                "- ✅ Safe passed: **{tn}**"
            .format(tp=cm["TP"], fn=cm["FN"], fp=cm["FP"], tn=cm["TN"]))
            dataset_story = ss.get("last_dataset_delta_story")
            metric_deltas: list[str] = []
            if prev_eval:
                metric_deltas.append(f"Δaccuracy {acc_cur - prev_eval.get('accuracy', acc_cur):+.2%}")
                metric_deltas.append(f"Δprecision {p_cur - prev_eval.get('precision', p_cur):+.2%}")
                metric_deltas.append(f"Δrecall {r_cur - prev_eval.get('recall', r_cur):+.2%}")
            extra_caption = " | ".join(part for part in [dataset_story, " · ".join(metric_deltas) if metric_deltas else ""] if part)
            if extra_caption:
                st.caption(f"📂 {extra_caption}")

    with section_surface():
        st.markdown("### Spam threshold")
        presets = threshold_presets(y_true01, p_spam)

        if "eval_temp_threshold" not in ss:
            ss["eval_temp_threshold"] = current_thr

        controls_col, slider_col = st.columns([2, 3], gap="large")
        with controls_col:
            if st.button("Balanced (max F1)", use_container_width=True):
                ss["eval_temp_threshold"] = float(presets["balanced_f1"])
                st.toast(f"Suggested threshold (max F1): {ss['eval_temp_threshold']:.2f}", icon="✅")
            if st.button("Protect inbox (≥95% precision)", use_container_width=True):
                ss["eval_temp_threshold"] = float(presets["precision_95"])
                st.toast(
                    f"Suggested threshold (precision≥95%): {ss['eval_temp_threshold']:.2f}",
                    icon="✅",
                )
            if st.button("Catch spam (≥90% recall)", use_container_width=True):
                ss["eval_temp_threshold"] = float(presets["recall_90"])
                st.toast(
                    f"Suggested threshold (recall≥90%): {ss['eval_temp_threshold']:.2f}",
                    icon="✅",
                )
            if st.button("Adopt this threshold", use_container_width=True):
                ss["threshold"] = float(ss.get("eval_temp_threshold", current_thr))
                st.success(
                    f"Adopted new operating threshold: **{ss['threshold']:.2f}**. This will be used in Classify and Full Autonomy."
                )
        with slider_col:
            temp_threshold = float(
                st.slider(
                    "Adjust threshold (temporary)",
                    0.1,
                    0.9,
                    value=float(ss.get("eval_temp_threshold", current_thr)),
                    step=0.01,
                    key="eval_temp_threshold",
                    help="Lower values catch more spam (higher recall) but risk more false alarms. Higher values protect the inbox (higher precision) but may miss some spam.",
                )
            )

            cm_temp = compute_confusion(y_true01, p_spam, temp_threshold)
            acc_temp = (cm_temp["TP"] + cm_temp["TN"]) / max(1, len(y_true01))
            st.caption(
                f"At {temp_threshold:.2f}, accuracy would be **{acc_temp:.2%}** (TP {cm_temp['TP']}, FP {cm_temp['FP']}, TN {cm_temp['TN']}, FN {cm_temp['FN']})."
            )

        acc_new, p_new, r_new, f1_new, cm_new = _pr_acc_cm(y_true01, p_spam, temp_threshold)

        with st.container(border=True):
            st.markdown("#### What changes when I move the threshold?")
            st.caption("Comparing your **adopted** threshold vs. the **temporary** slider value above:")

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Current (adopted)**")
                st.write(f"- Threshold: **{current_thr:.2f}**")
                st.write(f"- Accuracy: {_fmt_pct(acc_cur)}")
                st.write(f"- Precision (spam): {_fmt_pct(p_cur)}")
                st.write(f"- Recall (spam): {_fmt_pct(r_cur)}")
                st.write(f"- False positives (safe→spam): **{cm_cur['FP']}**")
                st.write(f"- False negatives (spam→safe): **{cm_cur['FN']}**")

            with col_right:
                st.markdown("**If you adopt the slider value**")
                st.write(f"- Threshold: **{temp_threshold:.2f}**")
                st.write(f"- Accuracy: {_fmt_pct(acc_new)} ({_fmt_delta(acc_new, acc_cur)})")
                st.write(f"- Precision (spam): {_fmt_pct(p_new)} ({_fmt_delta(p_new, p_cur)})")
                st.write(f"- Recall (spam): {_fmt_pct(r_new)} ({_fmt_delta(r_new, r_cur)})")
                st.write(
                    f"- False positives: **{cm_new['FP']}** ({_fmt_delta(cm_new['FP'], cm_cur['FP'], pct=False)})"
                )
                st.write(
                    f"- False negatives: **{cm_new['FN']}** ({_fmt_delta(cm_new['FN'], cm_cur['FN'], pct=False)})"
                )

            if temp_threshold > current_thr:
                st.info(
                    "Raising the threshold makes the model **more cautious**: usually **fewer false positives** (protects inbox) but **more spam may slip through**."
                )
            elif temp_threshold < current_thr:
                st.info(
                    "Lowering the threshold makes the model **more aggressive**: it **catches more spam** (higher recall) but may **flag more legit emails**."
                )
            else:
                st.info("Same threshold as adopted — metrics unchanged.")

    with section_surface():
        with st.expander("📌 Suggestions to improve your model"):
            st.markdown(
                """
  - Add more labeled emails, especially tricky edge cases
  - Balance the dataset between spam and safe
  - Use diverse wording in your examples
  - Tune the spam threshold for your needs
  - Review the confusion matrix to spot mistakes
  - Ensure emails have enough meaningful content
  """
            )

    with section_surface():
        st.markdown("### Guardrail audit (borderline emails)")

        guardrail_cards: list[dict[str, str]] = []
        guardrail_counts: Counter[str] = Counter()

        if len(p_spam) == 0:
            st.caption("Evaluation set empty — guardrail audit unavailable.")
        else:
            margin_window = 0.15
            max_cards = 8
            sorted_indices = sorted(
                range(len(p_spam)),
                key=lambda i: abs(float(p_spam[i]) - current_thr),
            )

            for idx in sorted_indices:
                if len(guardrail_cards) >= max_cards:
                    break

                prob_val = float(p_spam[idx])
                margin = abs(prob_val - current_thr)
                if margin > margin_window:
                    continue

                subject_raw = ""
                body_raw = ""
                if X_te_t is not None and X_te_b is not None:
                    if idx < len(X_te_t):
                        subject_raw = X_te_t[idx]
                    if idx < len(X_te_b):
                        body_raw = X_te_b[idx]
                else:
                    text_val = texts_test[idx] if idx < len(texts_test) else ""
                    text_str = text_val if isinstance(text_val, str) else str(text_val or "")
                    parts = text_str.split("\n", 1)
                    subject_raw = parts[0] if parts else ""
                    body_raw = parts[1] if len(parts) > 1 else ""

                subject_raw = str(subject_raw or "").strip()
                body_raw = str(body_raw or "").strip()

                signals = _guardrail_signals(subject_raw, body_raw)
                active_keys = [key for key, flag in signals.items() if flag]
                if not active_keys:
                    continue

                guardrail_counts.update(active_keys)

                subject_display = html.escape(
                    _shorten_text(subject_raw or "(no subject)", limit=100)
                )
                pred_label = "spam" if prob_val >= current_thr else "safe"
                icon = GUARDRAIL_LABEL_ICONS.get(pred_label, "✉️")

                try:
                    true_label_raw = y_te[idx]
                except Exception:
                    true_label_raw = ""
                true_label_clean = str(true_label_raw or "").strip().title()

                meta_left = html.escape(
                    f"P(spam) {prob_val:.2f} • Δ {prob_val - current_thr:+.2f}"
                )
                meta_right_parts = [f"{icon} {pred_label.title()}"]
                if true_label_clean:
                    meta_right_parts.append(f"True: {true_label_clean}")
                meta_right = html.escape(" • ".join(meta_right_parts))

                excerpt = (
                    _shorten_text(body_raw or "", limit=220)
                    .replace("\n", " ")
                    .strip()
                )
                excerpt_html = html.escape(excerpt) if excerpt else "No body text."
                badges_html = _guardrail_badges_html(signals)
                body_html = "".join(
                    [
                        badges_html,
                        f"<div style=\"margin-top:0.4rem; color: rgba(55, 65, 81, 0.85);\">{excerpt_html}</div>",
                    ]
                )

                guardrail_cards.append(
                    {
                        "subject": subject_display,
                        "meta_left": meta_left,
                        "meta_right": meta_right,
                        "body": body_html,
                    }
                )

            if not guardrail_cards:
                st.caption(
                    "No borderline emails triggered guardrail signals in the test set."
                )
            else:
                chart_obj = None
                counts_sorted = guardrail_counts.most_common()
                if counts_sorted:
                    guardrail_df = pd.DataFrame(
                        counts_sorted, columns=["guardrail", "count"]
                    )
                    guardrail_chart = (
                        alt.Chart(guardrail_df)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                        .encode(
                            x=alt.X("count:Q", title="Emails flagged"),
                            y=alt.Y(
                                "guardrail:N",
                                sort="-x",
                                title="Signal",
                            ),
                            color=alt.Color("guardrail:N", legend=None),
                            tooltip=[
                                alt.Tooltip("guardrail:N", title="Signal"),
                                alt.Tooltip("count:Q", title="Emails"),
                            ],
                        )
                        .properties(height=220)
                    )
                    chart_obj = st.altair_chart(
                        guardrail_chart, use_container_width=True
                    )

                render_guardrail_panel(chart=chart_obj, cards=guardrail_cards)

    ss["last_eval_results"] = {
        "accuracy": acc_cur,
        "precision": p_cur,
        "recall": r_cur,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    nerd_mode_eval_enabled = render_nerd_mode_toggle(
        key="nerd_mode_eval",
        title="Nerd Mode — technical details",
        description="Inspect precision/recall tables, interpretability cues, and governance notes.",
        icon="🔬",
    )

    if nerd_mode_eval_enabled:
        with section_surface():
            temp_threshold = float(ss.get("eval_temp_threshold", current_thr))
            y_hat_temp = (p_spam >= temp_threshold).astype(int)
            prec_spam, rec_spam, f1_spam, sup_spam = precision_recall_fscore_support(
                y_true01, y_hat_temp, average="binary", zero_division=0
            )
            y_true_safe = 1 - y_true01
            y_hat_safe = 1 - y_hat_temp
            prec_safe, rec_safe, f1_safe, sup_safe = precision_recall_fscore_support(
                y_true_safe, y_hat_safe, average="binary", zero_division=0
            )

            st.markdown("### Detailed metrics (at current threshold)")

            def _as_int(value, fallback):
                if value is None:
                    return int(fallback)
                try:
                    return int(value)
                except TypeError:
                    return int(fallback)

            spam_support = _as_int(sup_spam, np.sum(y_true01))
            safe_support = _as_int(sup_safe, np.sum(1 - y_true01))

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "class": "spam",
                            "precision": prec_spam,
                            "recall": rec_spam,
                            "f1": f1_spam,
                            "support": spam_support,
                        },
                        {
                            "class": "safe",
                            "precision": prec_safe,
                            "recall": rec_safe,
                            "f1": f1_safe,
                            "support": safe_support,
                        },
                    ]
                ).round(3),
                width="stretch",
                hide_index=True,
            )

            st.markdown("### Precision & Recall vs Threshold (validation)")
            fig = plot_threshold_curves(y_true01, p_spam)
            st.pyplot(fig)

            st.markdown("### Interpretability")
            try:
                if hasattr(ss["model"], "named_steps"):
                    clf = ss["model"].named_steps.get("clf")
                    vec = ss["model"].named_steps.get("tfidf")
                    if hasattr(clf, "coef_") and vec is not None:
                        vocab = np.array(vec.get_feature_names_out())
                        coefs = clf.coef_[0]
                        top_spam = vocab[np.argsort(coefs)[-10:]][::-1]
                        top_safe = vocab[np.argsort(coefs)[:10]]
                        col_i1, col_i2 = st.columns(2)
                        with col_i1:
                            st.write("Top signals → **Spam**")
                            st.write(", ".join(top_spam))
                        with col_i2:
                            st.write("Top signals → **Safe**")
                            st.write(", ".join(top_safe))
                    else:
                        st.caption("Coefficients unavailable for this classifier.")
                elif hasattr(ss["model"], "numeric_feature_coefs"):
                    coef_map = ss["model"].numeric_feature_coefs()
                    st.caption("Numeric feature weights (positive → Spam, negative → Safe):")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "feature": k,
                                    "weight_toward_spam": v,
                                }
                                for k, v in coef_map.items()
                            ]
                        ).sort_values("weight_toward_spam", ascending=False),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.caption("Interpretability: no compatible inspector for this model.")
            except Exception as e:
                st.caption(f"Interpretability view unavailable: {e}")

            st.markdown("### Governance & reproducibility")
            try:
                if len(cache) == 4:
                    n_tr, n_te = len(y_tr), len(y_te)
                else:
                    n_tr, n_te = len(y_tr), len(y_te)
                split = ss.get("train_params", {}).get("test_size", "—")
                seed = ss.get("train_params", {}).get("random_state", "—")
                ts = ss.get("eval_timestamp", "—")
                st.write(f"- Train set: {n_tr}  |  Test set: {n_te}  |  Hold-out fraction: {split}")
                st.write(f"- Random seed: {seed}")
                st.write(f"- Training time: {ts}")
                st.write(f"- Adopted threshold: {ss.get('threshold', 0.5):.2f}")
            except Exception:
                st.caption("Governance info unavailable.")


def render_classify_stage():

    stage = STAGE_BY_KEY["classify"]

    def _render_use_terminal(slot: DeltaGenerator) -> None:
        with slot:
            render_use_terminal()

    render_stage_top_grid("classify", left_renderer=_render_use_terminal)

    nerd_flag = bool(ss.get("nerd_mode_use") or ss.get("nerd_mode"))


    with section_surface():
        overview_col, guidance_col = st.columns([3, 2], gap="large")
        with overview_col:
            st.subheader(f"{stage.icon} {stage.title} — Run the spam detector")
            render_eu_ai_quote(
                "The EU AI Act says “an AI system infers, from the input it receives, how to generate outputs such as content, predictions, recommendations or decisions.”"
            )
            st.write(
                "In this step, the system takes each email (title + body) as **input** and produces an **output**: "
                "a **prediction** (*Spam* or *Safe*) with a confidence score. By default, it also gives a **recommendation** "
                "about where to place the email (Spam or Inbox)."
            )
        with guidance_col:
            st.markdown("### Operating tips")
            st.markdown(
                "- Monitor predictions before enabling full autonomy.\n"
                "- Keep an eye on confidence values to decide when to intervene."
            )

    with section_surface():
        st.markdown("### Autonomy")
        default_high_autonomy = ss.get("autonomy", AUTONOMY_LEVELS[0]).startswith("High")
        auto_col, explain_col = st.columns([2, 3], gap="large")
        with auto_col:
            use_high_autonomy = st.toggle(
                "High autonomy (auto-move emails)", value=default_high_autonomy, key="use_high_autonomy"
            )
        with explain_col:
            if use_high_autonomy:
                ss["autonomy"] = AUTONOMY_LEVELS[1]
                st.success("High autonomy ON — the system will **move** emails to Spam or Inbox automatically.")
            else:
                ss["autonomy"] = AUTONOMY_LEVELS[0]
                st.warning("High autonomy OFF — review recommendations before moving emails.")
        if not ss.get("model"):
            st.warning("Train a model first in the **Train** tab.")
            st.stop()

    st.markdown("### Incoming preview")
    if not ss.get("incoming"):
        st.caption("No incoming emails. Add or import more in **📊 Prepare Data**, or paste a custom email below.")
        with st.expander("Add a custom email to process"):
            title_val = st.text_input("Title", key="use_custom_title", placeholder="Subject…")
            body_val = st.text_area("Body", key="use_custom_body", height=100, placeholder="Email body…")
            if st.button("Add to incoming", key="btn_add_to_incoming"):
                if title_val.strip() or body_val.strip():
                    ss["incoming"].append({"title": title_val.strip(), "body": body_val.strip()})
                    st.success("Added to incoming.")
                    _append_audit("incoming_added", {"title": title_val[:64]})
                else:
                    st.warning("Please provide at least a title or a body.")
    else:
        preview_n = min(10, len(ss["incoming"]))
        preview_df = pd.DataFrame(ss["incoming"][:preview_n])
        if not preview_df.empty:
            subtitle = f"Showing the first {preview_n} incoming emails (unlabeled)."
            render_email_inbox_table(preview_df, title="Incoming emails", subtitle=subtitle, columns=["title", "body"])
        else:
            render_email_inbox_table(pd.DataFrame(), title="Incoming emails", subtitle="No incoming emails available.")

        if st.button(f"Process {preview_n} email(s)", type="primary", key="btn_process_batch"):
            batch = ss["incoming"][:preview_n]
            y_hat, p_spam, p_safe = _predict_proba_batch(ss["model"], batch)
            thr = float(ss.get("threshold", 0.5))

            batch_rows: list[dict] = []
            moved_spam = moved_inbox = 0
            for idx, item in enumerate(batch):
                pred = y_hat[idx]
                prob_spam = float(p_spam[idx])
                prob_safe = float(p_safe[idx]) if hasattr(p_safe, "__len__") else float(1.0 - prob_spam)
                action = "Recommend: Spam" if prob_spam >= thr else "Recommend: Inbox"
                routed_to = None
                if ss["use_high_autonomy"]:
                    routed_to = "Spam" if prob_spam >= thr else "Inbox"
                    mailbox_record = {
                        "title": item.get("title", ""),
                        "body": item.get("body", ""),
                        "pred": pred,
                        "p_spam": round(prob_spam, 3),
                    }
                    if routed_to == "Spam":
                        ss["mail_spam"].append(mailbox_record)
                        moved_spam += 1
                    else:
                        ss["mail_inbox"].append(mailbox_record)
                        moved_inbox += 1
                    action = f"Moved: {routed_to}"
                row = {
                    "title": item.get("title", ""),
                    "body": item.get("body", ""),
                    "pred": pred,
                    "p_spam": round(prob_spam, 3),
                    "p_safe": round(prob_safe, 3),
                    "action": action,
                    "routed_to": routed_to,
                }
                batch_rows.append(row)

            ss["use_batch_results"] = batch_rows
            ss["incoming"] = ss["incoming"][preview_n:]
            if ss["use_high_autonomy"]:
                st.success(
                    f"Processed {preview_n} emails — decisions applied (Inbox: {moved_inbox}, Spam: {moved_spam})."
                )
                _append_audit(
                    "batch_processed_auto", {"n": preview_n, "inbox": moved_inbox, "spam": moved_spam}
                )
            else:
                st.info(f"Processed {preview_n} emails — recommendations ready.")
                _append_audit("batch_processed_reco", {"n": preview_n})

    if ss.get("use_batch_results"):
        with section_surface():
            st.markdown("### Results")
            df_res = pd.DataFrame(ss["use_batch_results"])
            show_cols = ["title", "pred", "p_spam", "action", "routed_to"]
            existing_cols = [col for col in show_cols if col in df_res.columns]
            display_df = df_res[existing_cols].rename(
                columns={"pred": "Prediction", "p_spam": "P(spam)", "action": "Action", "routed_to": "Routed"}
            )
            render_email_inbox_table(display_df, title="Batch results", subtitle="Predictions and actions just taken.")
            st.caption(
                "Each row shows the predicted label, confidence (P(spam)), and the recommendation or action taken."
            )

        nerd_mode_enabled = render_nerd_mode_toggle(
            key="nerd_mode_use",
            title="Nerd Mode — details for this batch",
            description="Inspect raw probabilities, distributions, and the session audit trail.",
            icon="🔬",
        )
        if nerd_mode_enabled:
            df_res = pd.DataFrame(ss["use_batch_results"])
            with section_surface():
                st.markdown("### Nerd Mode — batch diagnostics")
                col_nm1, col_nm2 = st.columns([2, 1])
                with col_nm1:
                    st.markdown("**Raw probabilities (per email)**")
                    detail_cols = ["title", "p_spam", "p_safe", "pred", "action", "routed_to"]
                    det_existing = [col for col in detail_cols if col in df_res.columns]
                    st.dataframe(df_res[det_existing], width="stretch", hide_index=True)
                with col_nm2:
                    st.markdown("**Batch metrics**")
                    n_items = len(df_res)
                    mean_conf = float(df_res["p_spam"].mean()) if "p_spam" in df_res else 0.0
                    n_spam = int((df_res["pred"] == "spam").sum()) if "pred" in df_res else 0
                    n_safe = n_items - n_spam
                    st.write(f"- Items: {n_items}")
                    st.write(f"- Predicted Spam: {n_spam} | Safe: {n_safe}")
                    st.write(f"- Mean P(spam): {mean_conf:.2f}")

                    if "p_spam" in df_res:
                        fig, ax = plt.subplots()
                        ax.hist(df_res["p_spam"], bins=10)
                        ax.set_xlabel("P(spam)")
                        ax.set_ylabel("Count")
                        ax.set_title("Spam score distribution")
                        st.pyplot(fig)

            with section_surface():
                st.markdown("### Why did it decide this way? (per email)")

                split_cache = ss.get("split_cache")
                train_texts: list[str] = []
                train_labels: list[str] = []
                train_emb: Optional[np.ndarray] = None
                if split_cache:
                    try:
                        if len(split_cache) == 6:
                            X_tr_t, _, X_tr_b, _, y_tr_vals, _ = split_cache
                            train_texts = [combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)]
                            train_labels = list(y_tr_vals)
                        else:
                            X_tr_texts, _, y_tr_vals, _ = split_cache
                            train_texts = list(X_tr_texts)
                            train_labels = list(y_tr_vals)
                        if train_texts:
                            train_emb = cache_train_embeddings(train_texts)
                    except Exception:
                        train_texts = []
                        train_labels = []
                        train_emb = None

                threshold_val = float(ss.get("threshold", 0.5))
                model_obj = ss.get("model")

                for email_idx, row in enumerate(ss["use_batch_results"]):
                    title = row.get("title", "")
                    body = row.get("body", "")
                    pred_label = row.get("pred", "")
                    p_spam_val = row.get("p_spam")
                    try:
                        p_spam_float = float(p_spam_val)
                    except (TypeError, ValueError):
                        p_spam_float = None

                    header = title or "(no subject)"
                    with st.container(border=True):
                        st.markdown(f"#### {header}")
                        st.caption(f"Predicted **{pred_label or '—'}**")

                        if p_spam_float is not None:
                            margin = p_spam_float - threshold_val
                            decision = "Spam" if p_spam_float >= threshold_val else "Safe"
                            st.markdown(
                                f"**Decision summary:** P(spam) = {p_spam_float:.2f} vs threshold {threshold_val:.2f} → **{decision}** "
                                f"(margin {margin:+.2f})"
                            )
                            if hasattr(ss.get("model"), "last_thr_eff"):
                                idxs_thr = getattr(ss.get("model"), "last_thr_eff", None)
                                if idxs_thr:
                                    idxs, thr_eff = idxs_thr
                                    try:
                                        pos = list(idxs).index(email_idx)
                                        st.caption(
                                            f"Effective threshold (with numeric micro-rules): {thr_eff[pos]:.2f}"
                                        )
                                    except Exception:
                                        pass
                        else:
                            st.caption("Probability not available for this email.")

                        if train_texts and train_labels:
                            try:
                                nn_examples = get_nearest_training_examples(
                                    combine_text(title, body), train_texts, train_labels, train_emb, k=3
                                )
                            except Exception:
                                nn_examples = []
                        else:
                            nn_examples = []

                        if nn_examples:
                            st.markdown("**Similar training emails (semantic evidence):**")
                            for example in nn_examples:
                                text_full = example["text"]
                                title_example = text_full.split("\n", 1)[0]
                                st.write(
                                    f"- *{title_example.strip() or '(no subject)'}* — label: **{example['label']}** "
                                    f"(sim {example['similarity']:.2f})"
                                )
                        else:
                            st.caption("No similar training emails available.")

                        if hasattr(model_obj, "scaler") and (
                            hasattr(model_obj, "lr_num") or hasattr(model_obj, "lr")
                        ):
                            contribs = numeric_feature_contributions(model_obj, title, body)
                            if contribs:
                                contribs_sorted = sorted(contribs, key=lambda x: x[2], reverse=True)
                                st.markdown("**Numeric cues (how they nudged the decision):**")
                                st.dataframe(
                                    pd.DataFrame(
                                        [
                                            {
                                                "feature": feat,
                                                "standardized": val,
                                                "toward_spam_logit": contrib,
                                            }
                                            for feat, val, contrib in contribs_sorted
                                        ]
                                    ).round(3),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                st.caption("Positive values push toward **Spam**; negative toward **Safe**.")
                            else:
                                st.caption("Numeric feature contributions unavailable for this email.")
                        else:
                            st.caption("Numeric cue breakdown requires the hybrid model.")

                        if model_obj is not None:
                            with st.expander("🖍️ Highlight influential words (experimental)", expanded=False):
                                st.caption(
                                    "Runs extra passes to see which words reduce P(spam) the most when removed."
                                )
                                if st.checkbox(
                                    "Compute highlights for this email", key=f"hl_{email_idx}", value=False
                                ):
                                    base_prob, rows_imp = top_token_importances(model_obj, title, body)
                                    if base_prob is None:
                                        st.caption("Unable to compute token importances for this model/email.")
                                    else:
                                        st.caption(
                                            f"Base P(spam) = {base_prob:.2f}. Higher importance means removing the word lowers P(spam) more."
                                        )
                                        if rows_imp:
                                            df_imp = pd.DataFrame(rows_imp[:10])
                                            st.dataframe(df_imp, use_container_width=True, hide_index=True)
                                        else:
                                            st.caption("No influential words found among the sampled tokens.")
                        else:
                            st.caption("Word highlights require a trained model.")

            with section_surface():
                st.markdown("### Audit trail (this session)")
                if ss.get("use_audit_log"):
                    st.dataframe(pd.DataFrame(ss["use_audit_log"]), width="stretch", hide_index=True)
                else:
                    st.caption("No events recorded yet.")

            exp_df = _export_batch_df(ss["use_batch_results"])
            csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
            json_bytes = json.dumps(ss["use_batch_results"], ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                "⬇️ Download results (CSV)", data=csv_bytes, file_name="batch_results.csv", mime="text/csv"
            )
            st.download_button(
                "⬇️ Download results (JSON)", data=json_bytes, file_name="batch_results.json", mime="application/json"
            )

    st.markdown("### Adaptiveness — learn from your corrections")
    render_eu_ai_quote(
        "The EU AI Act says “AI systems may exhibit adaptiveness.” Enable adaptiveness to confirm or correct results; the model can retrain on your feedback."
    )
    def _handle_stage_adaptive_change() -> None:
        _set_adaptive_state(ss.get("adaptive_stage", ss.get("adaptive", False)), source="stage")

    st.toggle(
        "Enable adaptiveness (learn from feedback)",
        value=bool(ss.get("adaptive", False)),
        key="adaptive_stage",
        on_change=_handle_stage_adaptive_change,
    )
    use_adaptiveness = bool(ss.get("adaptive", False))

    if use_adaptiveness and ss.get("use_batch_results"):
        st.markdown("#### Review and give feedback")
        for i, row in enumerate(ss["use_batch_results"]):
            with st.container(border=True):
                st.markdown(f"**Title:** {row.get('title', '')}")
                pspam_value = row.get("p_spam")
                if isinstance(pspam_value, (int, float)):
                    pspam_text = f"{pspam_value:.2f}"
                else:
                    pspam_text = pspam_value
                action_display = row.get("action", "")
                pred_display = row.get("pred", "")
                st.markdown(
                    f"**Predicted:** {pred_display}  •  **P(spam):** {pspam_text}  •  **Action:** {action_display}"
                )
                col_a, col_b, col_c = st.columns(3)
                if col_a.button("Confirm", key=f"use_confirm_{i}"):
                    _append_audit("confirm_label", {"i": i, "pred": pred_display})
                    st.toast("Thanks — recorded your confirmation.", icon="✅")
                if col_b.button("Correct → Spam", key=f"use_correct_spam_{i}"):
                    ss["labeled"].append(
                        {"title": row.get("title", ""), "body": row.get("body", ""), "label": "spam"}
                    )
                    _append_audit("correct_label", {"i": i, "new": "spam"})
                    st.toast("Recorded correction → Spam.", icon="✍️")
                if col_c.button("Correct → Safe", key=f"use_correct_safe_{i}"):
                    ss["labeled"].append(
                        {"title": row.get("title", ""), "body": row.get("body", ""), "label": "safe"}
                    )
                    _append_audit("correct_label", {"i": i, "new": "safe"})
                    st.toast("Recorded correction → Safe.", icon="✍️")

        if st.button("🔁 Retrain now with feedback", key="btn_retrain_feedback"):
            df_all = pd.DataFrame(ss["labeled"])
            if not df_all.empty and len(df_all["label"].unique()) >= 2:
                params = ss.get("train_params", {})
                test_size = float(params.get("test_size", 0.30))
                random_state = int(params.get("random_state", 42))
                max_iter = int(params.get("max_iter", 1000))
                C_value = float(params.get("C", 1.0))

                titles = df_all["title"].fillna("").tolist()
                bodies = df_all["body"].fillna("").tolist()
                labels = df_all["label"].tolist()
                X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = train_test_split(
                    titles,
                    bodies,
                    labels,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=labels,
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
                try:
                    pass
                except Exception:
                    pass
                model = model.fit(X_tr_t, X_tr_b, y_tr)
                model.apply_numeric_adjustments(ss["numeric_adjustments"])
                ss["model"] = model
                ss["split_cache"] = (X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te)
                ss["eval_timestamp"] = datetime.now().isoformat(timespec="seconds")
                ss["eval_temp_threshold"] = float(ss.get("threshold", 0.6))
                ss["train_story_run_id"] = uuid4().hex
                for key in (
                    "meaning_map_show_examples",
                    "meaning_map_show_centers",
                    "meaning_map_highlight_borderline",
                    "meaning_map_show_pair_trigger",
                ):
                    ss.pop(key, None)
                st.success("Adaptive learning: model retrained with your feedback.")
                _append_audit("retrain_feedback", {"n_labeled": len(df_all)})
            else:
                st.warning("Need both classes (spam & safe) in labeled data to retrain.")

    st.markdown("### 📥 Mailboxes")
    inbox_tab, spam_tab = st.tabs(
        [
            f"Inbox (safe) — {len(ss['mail_inbox'])}",
            f"Spam — {len(ss['mail_spam'])}",
        ]
    )
    with inbox_tab:
        render_mailbox_panel(
            ss.get("mail_inbox"),
            mailbox_title="Inbox (safe)",
            filled_subtitle="Messages the system kept in your inbox.",
            empty_subtitle="Inbox is empty so far.",
        )
    with spam_tab:
        render_mailbox_panel(
            ss.get("mail_spam"),
            mailbox_title="Spam",
            filled_subtitle="What the system routed away from the inbox.",
            empty_subtitle="No emails have been routed to spam yet.",
        )

    st.caption(
        f"Threshold used for routing: **{float(ss.get('threshold', 0.5)):.2f}**. "
        "Adjust it in **🧪 Evaluate** to change how cautious/aggressive the system is."
    )

def render_model_card_stage():

    render_stage_top_grid("model_card")


    with section_surface():
        st.subheader("Model Card — transparency")
        guidance_popover("Transparency", """
Model cards summarize intended purpose, data, metrics, autonomy & adaptiveness settings.
They help teams reason about risks and the appropriate oversight controls.
""")
        algo = "Sentence embeddings (MiniLM) + standardized numeric cues + Logistic Regression"
        n_samples = len(ss["labeled"])
        labels_present = sorted({row["label"] for row in ss["labeled"]}) if ss["labeled"] else []
        metrics_text = ""
        holdout_n = 0
        if ss.get("model") and ss.get("split_cache"):
            _, X_te_t, _, X_te_b, _, y_te = ss["split_cache"]
            y_pred = ss["model"].predict(X_te_t, X_te_b)
            holdout_n = len(y_te)
            metrics_text = f"Accuracy on hold‑out: {accuracy_score(y_te, y_pred):.2%} (n={holdout_n})"
        snapshot_id = ss.get("active_dataset_snapshot")
        snapshot_entry = None
        if snapshot_id:
            snapshot_entry = next((snap for snap in ss.get("datasets", []) if snap.get("id") == snapshot_id), None)
        dataset_config_for_card = (snapshot_entry or {}).get("config", ss.get("dataset_config", DEFAULT_DATASET_CONFIG))
        dataset_config_json = json.dumps(dataset_config_for_card, indent=2, sort_keys=True)
        snapshot_label = snapshot_id if snapshot_id else "— (save one in Prepare Data)"

        card_md = f"""
# Model Card — demistifAI (Spam Detector)
**Intended purpose**: Educational demo to illustrate the AI Act definition of an **AI system** via a spam classifier.

**Algorithm**: {algo}
**Features**: Sentence embeddings (MiniLM) concatenated with small, interpretable numeric features:
- num_links_external, has_suspicious_tld, punct_burst_ratio, money_symbol_count, urgency_terms_count.
These are standardized and combined with the embedding before a linear classifier.

**Classes**: spam, safe
**Dataset size**: {n_samples} labeled examples
**Classes present**: {', '.join(labels_present) if labels_present else '[not trained]'}

**Key metrics**: {metrics_text or 'Train a model to populate metrics.'}

**Autonomy**: {ss['autonomy']} (threshold={ss['threshold']:.2f})
**Adaptiveness**: {'Enabled' if ss['adaptive'] else 'Disabled'} (learn from user corrections).

**Data**: user-augmented seed set (title + body); session-only.
**Dataset snapshot ID**: {snapshot_label}
**Dataset config**:
```
{dataset_config_json}
```
**Known limitations**: tiny datasets; vocabulary sensitivity; no MIME/URL/metadata features.

**AI Act mapping**
- **Machine-based system**: Streamlit app (software) running on cloud runtime (hardware).
- **Inference**: model learns patterns from labeled examples.
- **Output generation**: predictions + confidence; used to recommend/route emails.
    - **Varying autonomy**: user selects autonomy level; at high autonomy, the system acts.
- **Adaptiveness**: optional feedback loop that updates the model.
"""
        content_col, highlight_col = st.columns([3, 2], gap="large")
        with content_col:
            st.markdown(card_md)
            download_text(card_md, "model_card.md", "Download model_card.md")
        with highlight_col:
            st.markdown(
                """
                <div class="info-metric-grid">
                    <div class="info-metric-card">
                        <div class="label">Labeled dataset</div>
                        <div class="value">{samples}</div>
                    </div>
                    <div class="info-metric-card">
                        <div class="label">Hold-out size</div>
                        <div class="value">{holdout}</div>
                    </div>
                    <div class="info-metric-card">
                        <div class="label">Autonomy</div>
                        <div class="value">{autonomy}</div>
                    </div>
                    <div class="info-metric-card">
                        <div class="label">Adaptiveness</div>
                        <div class="value">{adaptive}</div>
                    </div>
                </div>
                """.format(
                    samples=n_samples,
                    holdout=holdout_n or "—",
                    autonomy=html.escape(ss.get("autonomy", AUTONOMY_LEVELS[0])),
                    adaptive="On" if ss.get("adaptive") else "Off",
                ),
                unsafe_allow_html=True,
            )

        with highlight_col:
            st.markdown("#### Dataset provenance")
            if snapshot_id:
                st.write(f"Snapshot ID: `{snapshot_id}`")
            else:
                st.write("Snapshot ID: — (save one in Prepare Data → Snapshot & provenance).")
            st.code(dataset_config_json, language="json")


def render_train_stage() -> None:
    render_train_stage_page(
        set_active_stage=set_active_stage,
        render_eu_ai_quote=render_eu_ai_quote,
        render_language_mix_chip_rows=render_language_mix_chip_rows,
        render_nerd_mode_toggle=render_nerd_mode_toggle,
        section_surface=section_surface,
        summarize_language_mix=summarize_language_mix,
    )


STAGE_RENDERERS = {
    'intro': render_intro_stage,
    'overview': render_overview_stage,
    'data': render_data_stage,
    'train': render_train_stage,
    'evaluate': render_evaluate_stage,
    'classify': render_classify_stage,
    'model_card': render_model_card_stage,
}


validate_invariants(s)
active_stage = s.get("run", {}).get("active_stage") or STAGES[0].key
ss["active_stage"] = active_stage
renderer = STAGE_RENDERERS.get(active_stage, render_intro_stage)

if ss.pop("stage_scroll_to_top", False):
    components.html(
        """
        <script>
        (function() {
            const main = window.parent.document.querySelector('section.main');
            if (main && typeof main.scrollTo === 'function') {
                main.scrollTo({ top: 0, behavior: 'smooth' });
            }
            if (window.parent && typeof window.parent.scrollTo === 'function') {
                window.parent.scrollTo({ top: 0, behavior: 'smooth' });
            }
        })();
        </script>
        """,
        height=0,
    )

renderer()

st.markdown("---")
st.caption("© demistifAI — Built for interactive learning and governance discussions.")
