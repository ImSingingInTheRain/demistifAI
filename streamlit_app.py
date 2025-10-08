from __future__ import annotations

import base64
import html
import json
import random
import string
import hashlib
import re
from collections import Counter
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from demistifai.constants import (
    APP_THEME_CSS,
    AUTONOMY_LEVELS,
    BRANDS,
    CLASSES,
    COURIERS,
    DATASET_LEGIT_DOMAINS,
    DATASET_SUSPICIOUS_TLDS,
    EMAIL_INBOX_TABLE_CSS,
    LIFECYCLE_CYCLE_CSS,
    STAGE_BY_KEY,
    STAGE_INDEX,
    STAGE_TEMPLATE_CSS,
    STAGES,
    URGENCY,
    StageMeta,
)
from demistifai.dataset import (
    ATTACHMENT_MIX_PRESETS,
    ATTACHMENT_TYPES,
    DEFAULT_ATTACHMENT_MIX,
    DEFAULT_DATASET_CONFIG,
    EDGE_CASE_TEMPLATES,
    DatasetConfig,
    STARTER_LABELED,
    STARTER_INCOMING,
    SUSPICIOUS_TLD_SUFFIXES,
    starter_dataset_copy,
    build_dataset_from_config,
    compute_dataset_hash,
    compute_dataset_summary,
    dataset_delta_story,
    dataset_summary_delta,
    explain_config_change,
    generate_labeled_dataset,
    lint_dataset,
    _caps_ratio,
    _count_money_mentions,
    _count_suspicious_links,
    _has_suspicious_tld,
)
from demistifai.modeling import (
    FEATURE_DISPLAY_NAMES,
    FEATURE_ORDER,
    FEATURE_PLAIN_LANGUAGE,
    HybridEmbedFeatsLogReg,
    _combine_text,
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
    extract_urls,
    _counts,
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

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="demistifAI", page_icon="📧", layout="wide")

st.markdown(APP_THEME_CSS, unsafe_allow_html=True)
st.markdown(STAGE_TEMPLATE_CSS, unsafe_allow_html=True)
st.markdown(EMAIL_INBOX_TABLE_CSS, unsafe_allow_html=True)
st.markdown(LIFECYCLE_CYCLE_CSS, unsafe_allow_html=True)


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


VALID_LABELS = {"spam", "safe"}


def _normalize_label(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.strip().lower()
    if x in {"ham", "legit", "legitimate"}:
        return "safe"
    return x


def _validate_csv_schema(df: pd.DataFrame) -> tuple[bool, str]:
    required = {"title", "body", "label"}
    missing = required - set(map(str.lower, df.columns))
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    return True, ""


def route_decision(autonomy: str, y_hat: str, pspam: Optional[float], threshold: float):
    routed = None
    if pspam is not None:
        to_spam = pspam >= threshold
    else:
        to_spam = y_hat == "spam"

    if autonomy.startswith("High"):
        routed = "Spam" if to_spam else "Inbox"
        action = f"Auto-routed to **{routed}** (threshold={threshold:.2f})"
    else:
        action = f"Recommend: {'Spam' if to_spam else 'Inbox'} (threshold={threshold:.2f})"
    return action, routed

def download_text(text: str, filename: str, label: str = "Download"):
    b64 = base64.b64encode(text.encode("utf-8")).decode()
    st.markdown(f'<a href="data:text/plain;base64,{b64}" download="{filename}">{label}</a>', unsafe_allow_html=True)


def render_nerd_mode_toggle(
    *,
    key: str,
    title: str,
    description: str,
    icon: Optional[str] = "🧠",
) -> bool:
    """Render a consistently styled Nerd Mode toggle block."""

    toggle_label = f"{icon} {title}" if icon else title
    value = st.toggle(toggle_label, key=key, value=bool(ss.get(key, False)))
    if description:
        st.caption(description)
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


def _append_audit(event: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Record an audit log entry for the current session."""

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "event": event,
    }
    if details:
        entry["details"] = details

    log = ss.setdefault("use_audit_log", [])
    log.append(entry)


def _export_batch_df(rows: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    """Normalize batch result rows into a consistent DataFrame for export."""

    base_cols = ["title", "body", "pred", "p_spam", "p_safe", "action", "routed_to"]
    if not rows:
        return pd.DataFrame(columns=base_cols)

    df_rows = pd.DataFrame(rows)
    for col in base_cols:
        if col not in df_rows.columns:
            df_rows[col] = None
    return df_rows[base_cols]


ss = st.session_state
requested_stage_values = st.query_params.get_all("stage")
requested_stage = requested_stage_values[0] if requested_stage_values else None
default_stage = STAGES[0].key
ss.setdefault("active_stage", default_stage)
if requested_stage in STAGE_BY_KEY:
    if requested_stage != ss["active_stage"]:
        ss["active_stage"] = requested_stage
else:
    if st.query_params.get_all("stage") != [ss["active_stage"]]:
        st.query_params["stage"] = ss["active_stage"]
ss.setdefault("nerd_mode", False)
ss.setdefault("autonomy", AUTONOMY_LEVELS[0])
ss.setdefault("threshold", 0.6)
ss.setdefault("nerd_mode_eval", False)
ss.setdefault("eval_timestamp", None)
ss.setdefault("eval_temp_threshold", float(ss["threshold"]))
ss.setdefault("adaptive", True)
ss.setdefault("labeled", starter_dataset_copy())      # list of dicts: title, body, label
ss.setdefault("incoming", STARTER_INCOMING.copy())    # list of dicts: title, body
ss.setdefault("model", None)
ss.setdefault("split_cache", None)
ss.setdefault("mail_inbox", [])  # list of dicts: title, body, pred, p_spam
ss.setdefault("mail_spam", [])
ss.setdefault("metrics", {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
ss.setdefault("last_classification", None)
ss.setdefault("numeric_adjustments", {feat: 0.0 for feat in FEATURE_ORDER})
ss.setdefault("nerd_mode_data", False)
ss.setdefault("nerd_mode_train", False)
ss.setdefault(
    "train_params",
    {"test_size": 0.30, "random_state": 42, "max_iter": 1000, "C": 1.0},
)
ss.setdefault("use_high_autonomy", ss.get("autonomy", AUTONOMY_LEVELS[0]).startswith("High"))
ss.setdefault("use_batch_results", [])
ss.setdefault("use_adaptiveness", bool(ss.get("adaptive", True)))
ss.setdefault("use_audit_log", [])
ss.setdefault("nerd_mode_use", False)
ss.setdefault("dataset_config", DEFAULT_DATASET_CONFIG.copy())
if "dataset_summary" not in ss:
    ss["dataset_summary"] = compute_dataset_summary(ss["labeled"])
ss.setdefault("previous_dataset_summary", None)
ss.setdefault("dataset_preview", None)
ss.setdefault("dataset_preview_config", None)
ss.setdefault("dataset_preview_summary", None)
ss.setdefault("dataset_manual_queue", None)
ss.setdefault("dataset_controls_open", False)
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

    if ss.get("active_stage") != stage_key:
        ss["active_stage"] = stage_key

    # Keep the sidebar radio selection aligned with the active stage so the
    # UI immediately reflects navigation triggered by buttons elsewhere.
    if ss.get("sidebar_stage_nav") != stage_key:
        ss["sidebar_stage_nav"] = stage_key

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

    if source != "sidebar":
        ss.pop("adaptive_sidebar", None)
    if source != "stage":
        ss.pop("adaptive_stage", None)


def _handle_sidebar_adaptive_change() -> None:
    _set_adaptive_state(ss.get("adaptive_sidebar", ss.get("adaptive", False)), source="sidebar")


ss["use_adaptiveness"] = bool(ss.get("adaptive", False))

with st.sidebar:
    st.markdown("<div class='sidebar-shell'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-brand">
            <p class="sidebar-title">demistifAI control room</p>
            <p class="sidebar-subtitle">Navigate the lifecycle, review guidance, and manage your session without losing progress.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stage_keys = [stage.key for stage in STAGES]
    active_index = STAGE_INDEX.get(ss.get("active_stage", STAGES[0].key), 0)

    st.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
    selected_stage = st.radio(
        "Navigate demistifAI",
        stage_keys,
        index=active_index,
        key="sidebar_stage_nav",
        label_visibility="collapsed",
        format_func=lambda key: f"{STAGE_BY_KEY[key].icon} {STAGE_BY_KEY[key].title}",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if selected_stage != ss.get("active_stage"):
        set_active_stage(selected_stage)

    current_stage = STAGE_BY_KEY.get(ss.get("active_stage", selected_stage))
    if current_stage is not None:
        st.markdown(
            """
            <div class="sidebar-stage-card">
                <div class="sidebar-stage-card__icon">{icon}</div>
                <div class="sidebar-stage-card__meta">
                    <span class="sidebar-stage-card__eyebrow">Current stage</span>
                    <p class="sidebar-stage-card__title">{title}</p>
                    <p class="sidebar-stage-card__description">{description}</p>
                </div>
            </div>
            """.format(
                icon=html.escape(current_stage.icon),
                title=html.escape(current_stage.title),
                description=html.escape(current_stage.description),
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div class='sidebar-section-title'>Session controls</div>", unsafe_allow_html=True)
    st.toggle(
        "Learn from my corrections (adaptiveness)",
        value=ss.get("adaptive", True),
        key="adaptive_sidebar",
        help="When enabled, your corrections in the Use stage will update the model during the session.",
    )
    _handle_sidebar_adaptive_change()

    if st.button("🔄 Reset demo data", use_container_width=True):
        ss["labeled"] = starter_dataset_copy()
        ss["incoming"] = STARTER_INCOMING.copy()
        ss["model"] = None
        ss["split_cache"] = None
        ss["mail_inbox"].clear(); ss["mail_spam"].clear()
        ss["metrics"] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        ss["last_classification"] = None
        ss["numeric_adjustments"] = {feat: 0.0 for feat in FEATURE_ORDER}
        ss["use_batch_results"] = []
        ss["use_audit_log"] = []
        ss["nerd_mode_use"] = False
        ss["use_high_autonomy"] = ss.get("autonomy", AUTONOMY_LEVELS[0]).startswith("High")
        ss["adaptive"] = True
        ss["use_adaptiveness"] = True
        ss.pop("adaptive_sidebar", None)
        ss.pop("adaptive_stage", None)
        st.success("Reset complete.")

    st.caption(
        "Need a refresher? Use the navigation above to revisit any step without restarting your scenario."
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.title("📧 demistifAI")



def render_intro_stage():

    next_stage_key: Optional[str] = None
    intro_index = STAGE_INDEX.get("intro")
    if intro_index is not None and intro_index < len(STAGES) - 1:
        next_stage_key = STAGES[intro_index + 1].key

    with section_surface("section-surface--hero"):
        hero_left, hero_right = st.columns([3, 2], gap="large")
        with hero_left:
            st.subheader("Welcome to demistifAI! 🎉")
            st.markdown(
                "demistifAI is an interactive experience where you will build, evaluate, and operate an AI system—"
                "applying key concepts from the EU AI Act."
            )
            st.markdown(
                "Along the way you’ll see:\n"
                "- how an AI system works end-to-end,\n"
                "- how it infers using AI models,\n"
                "- how models learn from data to achieve an explicit objective,\n"
                "- how autonomy levels affect you as a user, and how optional adaptiveness feeds your feedback back into training."
            )
            render_eu_ai_quote(
                "“AI system means a machine-based system that is designed to operate with varying levels of autonomy and that may exhibit "
                "adaptiveness after deployment, and that, for explicit or implicit objectives, infers, from the input it "
                "receives, how to generate outputs such as predictions, content, recommendations, or decisions that can "
                "influence physical or virtual environments.”"
            )
            
        with hero_right:
            hero_info_html = """
            <div class="hero-info-grid">
                <div class="hero-info-card">
                    <h3>What you’ll do</h3>
                    <p>
                        Build an email spam detector that identifies patterns in messages. You’ll set how strict the filter is
                        (threshold), choose the autonomy level, and optionally enable adaptiveness to learn from your feedback.
                    </p>
                </div>
                <div class="hero-info-card">
                    <h3>Why demistifAI</h3>
                    <p>
                        AI systems are often seen as black boxes, and the EU AI Act can feel too abstract. This experience demystifies
                        both—showing how everyday AI works in practice.
                    </p>
                </div>
            </div>
            """
            st.markdown(hero_info_html, unsafe_allow_html=True)

            if next_stage_key:
                 st.button(
                    "🚀 Start your machine",
                    key="flow_start_machine_hero",
                    type="primary",
                    on_click=set_active_stage,
                    args=(next_stage_key,),
                    use_container_width=True
                )

               
    with section_surface():
        st.markdown(
            """
            <div>
                <h4>Your AI system lifecycle at a glance</h4>
                <p>These are the core stages you will navigate. They flow into one another — it’s a continuous loop you can revisit.</p>
                <div class="lifecycle-flow">
                    <div class="lifecycle-step">
                            <span class="lifecycle-icon">📊</span>
                            <span class="lifecycle-label">Prepare Data</span>
                        </div>
                        <span class="lifecycle-arrow">➝</span>
                        <div class="lifecycle-step">
                            <span class="lifecycle-icon">🧠</span>
                            <span class="lifecycle-label">Train</span>
                        </div>
                        <span class="lifecycle-arrow">➝</span>
                        <div class="lifecycle-step">
                            <span class="lifecycle-icon">🧪</span>
                            <span class="lifecycle-label">Evaluate</span>
                        </div>
                        <span class="lifecycle-arrow">➝</span>
                        <div class="lifecycle-step">
                            <span class="lifecycle-icon">📬</span>
                            <span class="lifecycle-label">Use</span>
                        </div>
                        <span class="lifecycle-loop">↺</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


    with section_surface():
        ready_left, ready_right = st.columns([3, 2], gap="large")
        with ready_left:
            st.markdown("### Ready to make a machine learn?")
            st.markdown("No worries — you don’t need to be a developer or data scientist to follow along.")
        with ready_right:
            if next_stage_key:
                st.button(
                    "🚀 Start your machine",
                    key="flow_start_machine_ready",
                    type="primary",
                    on_click=set_active_stage,
                    args=(next_stage_key,),
                )


def render_overview_stage():
    # --- Intro: EU AI Act quote + context card ---
    with section_surface():
        intro_left, intro_right = st.columns(2, gap="large")
        with intro_left:
            # Fix grammar: machine-based
            render_eu_ai_quote(
                "The EU AI Act says that “An AI system is a machine-based system”."
            )
        with intro_right:
            st.markdown(
                """
                <div class="callout callout--info">
                    <h4>🧭 Start your machine</h4>
                    <p>You are already inside a <strong>machine-based system</strong>: the Streamlit UI (software) running in the cloud (hardware).</p>
                    <p>Use this simple interface to <strong>build, evaluate, and operate</strong> a small email spam detector.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Nerd Mode toggle (copy kept, text refined slightly) ---
    with section_surface():
        nerd_enabled = render_nerd_mode_toggle(
            key="nerd_mode",
            title="Nerd Mode",
            icon="🧠",
            description="Toggle to see technical details and extra functionality. You can enable it at any stage to look under the hood.",
        )

    # --- Meet the machine (left) + Mission & Inbox preview (right) ---
    with section_surface():
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown("### Meet the machine")

            # 3 callouts: User interface, AI model, Inbox interface
            components_html = """
            <div class="callout-grid">
                <div class="callout callout--info">
                    <h5>🖥️ User interface</h5>
                    <p>The control panel for your AI system. Step through <strong>Prepare data</strong>, <strong>Train</strong>, <strong>Evaluate</strong>, and <strong>Use</strong>. Tooltips and short explainers guide you; <em>Nerd Mode</em> reveals more.</p>
                </div>
                <div class="callout callout--info">
                    <h5>🧠 AI model (how it learns & infers)</h5>
                    <p>The model learns from <strong>labeled examples</strong> you provide to tell <strong>Spam</strong> from <strong>Safe</strong>. For each new email it produces a <strong>spam score</strong> (P(spam)); your <strong>threshold</strong> turns that score into a recommendation or decision.</p>
                </div>
                <div class="callout callout--info">
                    <h5>📥 Inbox interface</h5>
                    <p>A simulated inbox feeds emails into the system. Preview items, process a batch or review one by one, and optionally enable <strong>adaptiveness</strong> so your confirmations/corrections help the model improve.</p>
                </div>
                <div class="callout callout--info">
                    <h5>🎯 Your mission</h5>
                    <p>Keep unwanted email out while letting the important messages through.</p>
                </div>
            </div>
            """
            st.markdown(components_html, unsafe_allow_html=True)

        with right:
            st.markdown(
                """
                <div class="mission-preview-stack">
                    <div class="inbox-preview-card">
                        <div class="preview-header">
                            <span class="preview-header-icon">📥</span>
                            <div>
                                <h4>Inbox preview</h4>
                                <p>A snapshot of the next messages waiting to be classified.</p>
                            </div>
                        </div>
                """,
                unsafe_allow_html=True,
            )

            if not ss["incoming"]:
                render_email_inbox_table(pd.DataFrame(), title="Inbox", subtitle="Inbox stream is empty.")
            else:
                df_incoming = pd.DataFrame(ss["incoming"])
                preview = df_incoming.head(5)
                render_email_inbox_table(preview, title="Inbox", columns=["title", "body"])

            st.markdown(
                """
                        <p class="preview-note">Preview only — you'll process batches in <strong>Use</strong> once your system is ready.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Nerd Mode details (mirrors the 3 components; adds governance/packages/limits) ---
    if nerd_enabled:
        with section_surface():
            st.markdown("### 🔬 Nerd Mode — technical details")

            nerd_details_html = """
            <div class="callout-grid">
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">🖥️</div>
                    <div class="callout-body">
                        <h5>User interface (software &amp; runtime)</h5>
                        <ul>
                            <li>You’re using a simple Streamlit (Python) web app running in the cloud.</li>
                            <li>The app remembers your session choices — data, model, threshold, autonomy — so you can move around without losing progress.</li>
                            <li>Short tips and popovers appear where helpful; toggle <em>Nerd Mode</em> any time to dive deeper.</li>
                        </ul>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">🧠</div>
                    <div class="callout-body">
                        <h5>AI model (how it works, without the math)</h5>
                        <ul>
                            <li><strong>What’s inside:</strong>
                                <ul>
                                    <li>A MiniLM sentence-transformer turns each email’s title + body into meaning-rich numbers.</li>
                                    <li>A Logistic Regression layer draws the boundary between Spam and Safe.</li>
                                </ul>
                            </li>
                            <li><strong>How it learns (training):</strong>
                                <ul>
                                    <li>You supply labeled examples (Spam/Safe).</li>
                                    <li>The app trains on most of them and holds out a slice for fair evaluation later.</li>
                                    <li>Training is repeatable via a fixed random seed; class weights rebalance skewed datasets.</li>
                                </ul>
                            </li>
                            <li><strong>How it predicts (inference):</strong>
                                <ul>
                                    <li>For a new email, the model outputs a spam score between 0 and 1.</li>
                                    <li>A threshold converts that score into action: below = Safe, above = Spam.</li>
                                    <li>In <em>Evaluate</em>, tune the threshold with presets such as Balanced, Protect inbox, or Catch spam.</li>
                                </ul>
                            </li>
                            <li><strong>Why it decided that (interpretability):</strong>
                                <ul>
                                    <li>View similar training emails and simple clues (urgent tone, suspicious links, ALL-CAPS bursts).</li>
                                    <li>Enable numeric signals to see which features nudged the call toward Spam or Safe.</li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">📥</div>
                    <div class="callout-body">
                        <h5>Inbox interface (your data in and out)</h5>
                        <ul>
                            <li>The app manages incoming (unlabeled) emails, labeled training emails, and the routed Inbox/Spam buckets.</li>
                            <li>Process emails in small batches (e.g., the first 10) or handle them one by one.</li>
                            <li><strong>Autonomy levels:</strong>
                                <ul>
                                    <li>Moderate (default): the system recommends a route; you decide.</li>
                                    <li>High autonomy: the system routes automatically using your chosen threshold.</li>
                                </ul>
                            </li>
                            <li><strong>Adaptiveness (optional):</strong> confirm or correct outcomes to add feedback, then retrain to personalize the model.</li>
                        </ul>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">🛡️</div>
                    <div class="callout-body">
                        <h5>Governance &amp; transparency</h5>
                        <ul>
                            <li>A model card records purpose, data summary, metrics, chosen threshold, autonomy, adaptiveness, seed, and timestamps.</li>
                            <li>We track risks: false positives (legit to Spam) and false negatives (Spam to Inbox).</li>
                            <li>An optional audit log lists batch actions, corrections, and retraining events for the session.</li>
                        </ul>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">🧩</div>
                    <div class="callout-body">
                        <h5>Packages (what powers this)</h5>
                        <p>streamlit (UI), pandas/numpy (data), scikit-learn (training &amp; evaluation), optional sentence-transformers + torch/transformers (embeddings), matplotlib (plots)</p>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">📏</div>
                    <div class="callout-body">
                        <h5>Limits (demo scope)</h5>
                        <ul>
                            <li>Uses synthetic or curated text — there’s no live mailbox connection.</li>
                            <li>Designed for learning clarity rather than production-grade email security.</li>
                        </ul>
                    </div>
                </div>
            </div>
            """
            st.markdown(nerd_details_html, unsafe_allow_html=True)


def render_data_stage():

    stage = STAGE_BY_KEY["data"]

    current_summary = compute_dataset_summary(ss["labeled"])
    ss["dataset_summary"] = current_summary

    with section_surface():
        lead_col, side_col = st.columns([3, 2], gap="large")
        with lead_col:
            st.subheader(f"{stage.icon} {stage.title} — curate the objective-aligned dataset")
            st.markdown(
                """
                You define the purpose (**filter spam**) and you curate the evidence the model will learn from.
                Adjust class balance, feature prevalence, and data quality to see how governance choices change performance.
                """
            )
        with side_col:
            st.markdown(
                """
                <div class="callout callout--mission">
                    <h4>EU AI Act tie-ins</h4>
                    <ul>
                        <li><strong>Objective &amp; data:</strong> You set the purpose and align the dataset to it.</li>
                        <li><strong>Risk controls:</strong> Class balance, noise limits, and validation guardrails manage bias &amp; quality.</li>
                        <li><strong>Transparency:</strong> Snapshots capture config + hash for provenance.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    delta_text = ""
    if ss.get("dataset_compare_delta"):
        delta_text = dataset_delta_story(ss["dataset_compare_delta"])
    if not delta_text and ss.get("last_dataset_delta_story"):
        delta_text = ss["last_dataset_delta_story"]
    if not delta_text:
        delta_text = explain_config_change(ss.get("dataset_config", DEFAULT_DATASET_CONFIG))

    with section_surface():
        st.markdown("### 1 · Prepare data → Dataset builder")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4, gap="large")
        with col_m1:
            st.metric("Examples", current_summary.get("total", 0))
        with col_m2:
            st.metric("Spam share", f"{current_summary.get('spam_ratio', 0)*100:.1f}%")
        with col_m3:
            st.metric("Suspicious TLD hits", current_summary.get("suspicious_tlds", 0))
        with col_m4:
            st.metric("Avg suspicious links (spam)", f"{current_summary.get('avg_susp_links', 0.0):.2f}")

        st.caption(
            "Class balance and feature prevalence are governance controls — tweak them to see how they shape learning."
        )
        if delta_text:
            st.info(f"🧭 {delta_text}")

        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4, gap="small")
        with btn_col1:
            if st.button("Adjust dataset", key="open_dataset_builder"):
                ss["dataset_controls_open"] = True
        with btn_col2:
            if st.button("Reset to baseline", key="reset_dataset_baseline"):
                ss["labeled"] = starter_dataset_copy()
                ss["dataset_config"] = DEFAULT_DATASET_CONFIG.copy()
                baseline_summary = compute_dataset_summary(ss["labeled"])
                ss["dataset_summary"] = baseline_summary
                ss["previous_dataset_summary"] = None
                ss["dataset_compare_delta"] = None
                ss["last_dataset_delta_story"] = None
                ss["active_dataset_snapshot"] = None
                ss["dataset_snapshot_name"] = ""
                ss["dataset_preview"] = None
                ss["dataset_preview_config"] = None
                ss["dataset_preview_summary"] = None
                ss["dataset_preview_lint"] = None
                ss["dataset_manual_queue"] = None
                ss["dataset_controls_open"] = False
                st.success(
                    f"Dataset reset to starter baseline ({len(STARTER_LABELED)} rows)."
                )
        with btn_col3:
            if st.button("Compare to last dataset", key="compare_dataset_button"):
                if ss.get("previous_dataset_summary"):
                    ss["dataset_compare_delta"] = dataset_summary_delta(
                        ss["previous_dataset_summary"], current_summary
                    )
                    st.toast("Comparison updated below the builder.", icon="📊")
                else:
                    st.toast("No previous dataset stored yet — save a snapshot after your first tweak.", icon="ℹ️")
        with btn_col4:
            st.button(
                "Clear preview",
                key="clear_dataset_preview",
                disabled=ss.get("dataset_preview") is None,
                on_click=_discard_preview,
            )

    if ss.get("dataset_controls_open"):
        with section_surface():
            st.markdown("#### Adjust dataset knobs (guardrails enforced)")
            st.caption(
                "Cap: first 200 rows per apply for manual review • Noise slider max 5% • Synthetic poisoning is contained."
            )

            def _clear_preview_state_inline():
                _clear_dataset_preview_state()
                ss["dataset_controls_open"] = False

            top_cols = st.columns([3, 1])
            with top_cols[1]:
                st.button("Close controls", key="close_dataset_builder", on_click=_clear_preview_state_inline)

            cfg = ss.get("dataset_config", DEFAULT_DATASET_CONFIG)
            with st.form("dataset_builder_form"):
                col_a, col_b = st.columns(2, gap="large")
                with col_a:
                    n_total_choice = st.radio(
                        "Training volume (N emails)",
                        options=[100, 300, 500],
                        index=[100, 300, 500].index(int(cfg.get("n_total", 500))) if int(cfg.get("n_total", 500)) in [100, 300, 500] else 2,
                        help="Preset sizes illustrate how data volume influences learning (guarded ≤500).",
                    )
                    spam_ratio = st.slider(
                        "Class balance (spam share)",
                        min_value=0.20,
                        max_value=0.80,
                        value=float(cfg.get("spam_ratio", 0.5)),
                        step=0.05,
                        help="Adjust prevalence to explore bias/recall trade-offs.",
                    )
                    links_level = st.slider(
                        "Suspicious links per spam email",
                        min_value=0,
                        max_value=2,
                        value=int(str(cfg.get("susp_link_level", "1"))),
                        help="Controls how many sketchy URLs appear in spam examples (0–2).",
                    )
                    edge_cases = st.slider(
                        "Edge-case near-duplicate pairs",
                        min_value=0,
                        max_value=len(EDGE_CASE_TEMPLATES),
                        value=int(cfg.get("edge_cases", 0)),
                        help="Inject similar-looking spam/safe pairs to stress the model.",
                    )
                    noise_pct = st.slider(
                        "Label noise (%)",
                        min_value=0.0,
                        max_value=5.0,
                        value=float(cfg.get("label_noise_pct", 0.0)),
                        step=1.0,
                        help="Flip a small share of labels to demonstrate noise impact (2–5% suggested).",
                    )
                with col_b:
                    tld_level = st.select_slider(
                        "Suspicious TLD frequency",
                        options=["low", "med", "high"],
                        value=cfg.get("susp_tld_level", "med"),
                    )
                    caps_level = st.select_slider(
                        "ALL-CAPS / urgency intensity",
                        options=["low", "med", "high"],
                        value=cfg.get("caps_intensity", "med"),
                    )
                    money_level = st.select_slider(
                        "Money symbols & urgency",
                        options=["off", "low", "high"],
                        value=cfg.get("money_urgency", "low"),
                    )
                    attachment_keys = list(ATTACHMENT_MIX_PRESETS.keys())
                    current_mix = cfg.get("attachments_mix", DEFAULT_ATTACHMENT_MIX)
                    current_choice = next(
                        (name for name, mix in ATTACHMENT_MIX_PRESETS.items() if mix == current_mix),
                        "Balanced",
                    )
                    attachment_choice = st.selectbox(
                        "Attachment lure mix",
                        options=attachment_keys,
                        index=attachment_keys.index(current_choice) if current_choice in attachment_keys else 1,
                        help="Choose how often risky attachments (HTML/ZIP/XLSM/EXE) appear vs. safer PDFs.",
                    )
                    seed_value = st.number_input(
                        "Random seed",
                        min_value=0,
                        value=int(cfg.get("seed", 42)),
                        help="Keep this fixed for reproducibility.",
                    )
                    poison_demo = st.toggle(
                        "Data poisoning demo (synthetic)",
                        value=bool(cfg.get("poison_demo", False)),
                        help="Adds a tiny malicious distribution shift labeled as safe to show metric degradation.",
                    )

                submitted = st.form_submit_button("Apply tweaks and preview", type="primary")

            if submitted:
                attachment_mix = ATTACHMENT_MIX_PRESETS.get(attachment_choice, DEFAULT_ATTACHMENT_MIX).copy()
                config: DatasetConfig = {
                    "seed": int(seed_value),
                    "n_total": int(n_total_choice),
                    "spam_ratio": float(spam_ratio),
                    "susp_link_level": str(int(links_level)),
                    "susp_tld_level": tld_level,
                    "caps_intensity": caps_level,
                    "money_urgency": money_level,
                    "attachments_mix": attachment_mix,
                    "edge_cases": int(edge_cases),
                    "label_noise_pct": float(noise_pct),
                    "poison_demo": bool(poison_demo),
                }
                dataset_rows = build_dataset_from_config(config)
                preview_summary = compute_dataset_summary(dataset_rows)
                lint_counts = lint_dataset(dataset_rows)
                ss["dataset_preview"] = dataset_rows
                ss["dataset_preview_config"] = config
                ss["dataset_preview_summary"] = preview_summary
                ss["dataset_preview_lint"] = lint_counts
                ss["dataset_manual_queue"] = pd.DataFrame(dataset_rows[: min(len(dataset_rows), 200)])
                if ss["dataset_manual_queue"] is not None and not ss["dataset_manual_queue"].empty:
                    ss["dataset_manual_queue"].insert(0, "include", True)
                ss["dataset_compare_delta"] = dataset_summary_delta(current_summary, preview_summary)
                ss["last_dataset_delta_story"] = dataset_delta_story(ss["dataset_compare_delta"])
                st.success("Preview ready — scroll to **Review & approve** to curate rows before committing.")
                explanation = explain_config_change(config, ss.get("dataset_config", DEFAULT_DATASET_CONFIG))
                if explanation:
                    st.caption(explanation)
                if lint_counts and any(lint_counts.values()):
                    st.warning(
                        "PII lint flags — sensitive-looking patterns detected (credit cards: {} | IBAN: {})."
                        .format(lint_counts.get("credit_card", 0), lint_counts.get("iban", 0))
                    )
                if len(dataset_rows) > 200:
                    st.caption("Manual queue shows the first 200 items per guardrail. Full dataset size: {}.".format(len(dataset_rows)))

    if ss.get("dataset_preview"):
        with section_surface():
            st.markdown("### 2 · Review & approve")
            preview_summary = ss.get("dataset_preview_summary") or compute_dataset_summary(ss["dataset_preview"])
            sum_col, lint_col = st.columns([2, 2], gap="large")
            with sum_col:
                st.metric("Preview rows", preview_summary.get("total", 0))
                st.metric("Spam share", f"{preview_summary.get('spam_ratio', 0)*100:.1f}%")
                st.metric("Avg suspicious links (spam)", f"{preview_summary.get('avg_susp_links', 0.0):.2f}")
            with lint_col:
                lint_counts = ss.get("dataset_preview_lint") or {"credit_card": 0, "iban": 0}
                st.write("**Validation checks**")
                st.write(f"- Credit card-like patterns: {lint_counts.get('credit_card', 0)}")
                st.write(f"- IBAN-like patterns: {lint_counts.get('iban', 0)}")
                st.caption("Guardrail: no live link fetching, HTML escaped, duplicates dropped.")

            manual_df = ss.get("dataset_manual_queue")
            if manual_df is None or manual_df.empty:
                manual_df = pd.DataFrame(ss["dataset_preview"][: min(len(ss["dataset_preview"]), 200)])
                if not manual_df.empty:
                    manual_df.insert(0, "include", True)
            edited_df = st.data_editor(
                manual_df,
                width="stretch",
                hide_index=True,
                key="dataset_manual_editor",
                column_config={
                    "include": st.column_config.CheckboxColumn("Include?", help="Uncheck to drop before committing."),
                    "label": st.column_config.SelectboxColumn("Label", options=sorted(VALID_LABELS)),
                },
            )
            ss["dataset_manual_queue"] = edited_df
            st.caption("Manual queue covers up to 200 rows per apply — re-run the builder to generate more variations.")

            commit_col, discard_col, _ = st.columns([1, 1, 2])

            if commit_col.button("Commit dataset tweaks", type="primary"):
                preview_rows = ss.get("dataset_preview")
                config = ss.get("dataset_preview_config", ss.get("dataset_config", DEFAULT_DATASET_CONFIG))
                if not preview_rows:
                    st.error("Generate a preview before committing.")
                else:
                    edited_records = []
                    if isinstance(edited_df, pd.DataFrame):
                        edited_records = edited_df.to_dict(orient="records")
                    preview_copy = [dict(row) for row in preview_rows]
                    for idx, record in enumerate(edited_records):
                        if idx >= len(preview_copy):
                            break
                        preview_copy[idx]["title"] = str(record.get("title", preview_copy[idx].get("title", "")))
                        preview_copy[idx]["body"] = str(record.get("body", preview_copy[idx].get("body", "")))
                        preview_copy[idx]["label"] = record.get("label", preview_copy[idx].get("label", "spam"))
                        preview_copy[idx]["include"] = bool(record.get("include", True))
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
                        lint_counts = lint_dataset(final_rows)
                        new_summary = compute_dataset_summary(final_rows)
                        delta = dataset_summary_delta(ss.get("dataset_summary", {}), new_summary)
                        ss["previous_dataset_summary"] = ss.get("dataset_summary", {})
                        ss["dataset_summary"] = new_summary
                        ss["dataset_config"] = config
                        ss["dataset_compare_delta"] = delta
                        ss["last_dataset_delta_story"] = dataset_delta_story(delta)
                        ss["labeled"] = final_rows
                        ss["active_dataset_snapshot"] = None
                        ss["dataset_snapshot_name"] = ""
                        _clear_dataset_preview_state()
                        st.success(f"Dataset updated with {len(final_rows)} curated rows.")
                        if any(lint_counts.values()):
                            st.warning(
                                "Lint warnings persist after commit (credit cards: {} | IBAN: {})."
                                .format(lint_counts.get("credit_card", 0), lint_counts.get("iban", 0))
                            )

            if discard_col.button("Discard preview"):
                _discard_preview()
                st.info("Preview cleared. The active labeled dataset remains unchanged.")

    with section_surface():
        st.markdown("### 3 · Snapshot & provenance")
        config_json = json.dumps(ss.get("dataset_config", DEFAULT_DATASET_CONFIG), indent=2, sort_keys=True)
        st.caption("Save immutable snapshots to reference in the model card and audits.")
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
                "name": ss.get("dataset_snapshot_name") or f"snapshot-{len(ss['datasets'])+1}",
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
            df_snap = pd.DataFrame(ss["datasets"])
            st.dataframe(df_snap[["name", "id", "timestamp", "rows"]], hide_index=True, width="stretch")
        else:
            st.caption("No snapshots yet. Save one after curating your first dataset.")

    nerd_data = render_nerd_mode_toggle(
        key="nerd_mode_data",
        title="Nerd Mode — diagnostics & CSV import",
        description="Inspect token clouds, feature histograms, leakage checks, and ingest custom CSVs.",
    )

    if nerd_data:
        with section_surface():
            st.markdown("### 4 · Nerd Mode extras")
            df_lab = pd.DataFrame(ss["labeled"])
            if df_lab.empty:
                st.info("Label some emails or import data to unlock diagnostics.")
            else:
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
                    st.write(", ".join(f"{w} ({c})" for w, c in top_spam))
                with col_tok2:
                    st.markdown("**Class token cloud — Safe**")
                    st.write(", ".join(f"{w} ({c})" for w, c in top_safe))

                spam_link_counts = [
                    _count_suspicious_links(row.get("body", ""))
                    for _, row in df_lab.iterrows()
                    if row.get("label") == "spam"
                ]
                link_series = pd.Series(spam_link_counts, name="Suspicious links")
                if not link_series.empty:
                    st.bar_chart(link_series.value_counts().sort_index(), height=200)
                else:
                    st.caption("No spam samples yet to chart suspicious link frequency.")

                title_groups: Dict[str, set] = {}
                leakage_titles = []
                for _, row in df_lab.iterrows():
                    title = row.get("title", "").strip().lower()
                    label = row.get("label")
                    title_groups.setdefault(title, set()).add(label)
                for title, labels in title_groups.items():
                    if len(labels) > 1 and title:
                        leakage_titles.append(title)
                if leakage_titles:
                    st.warning("Potential leakage: identical subjects across labels -> " + ", ".join(leakage_titles[:5]))
                else:
                    st.caption("Leakage check: no identical subjects across labels in the active dataset.")

                strat_df = df_lab.groupby("label").size().reset_index(name="count")
                st.dataframe(strat_df, hide_index=True, width="stretch")

        with st.expander("📤 Upload CSV of labeled emails (strict schema)", expanded=False):
            st.caption(
                "Schema: title, body, label (spam|safe). Limits: ≤2,000 rows, title ≤200 chars, body ≤2,000 chars."
            )
            st.caption("Uploaded data stays in this session only. No emails are sent or fetched.")
            up = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader_labeled")
            if up is not None:
                try:
                    df_up = pd.read_csv(up)
                    df_up.columns = [c.strip().lower() for c in df_up.columns]
                    ok, msg = _validate_csv_schema(df_up)
                    if not ok:
                        st.error(msg)
                    else:
                        if len(df_up) > 2000:
                            st.error("Too many rows (max 2,000). Trim the file and retry.")
                        else:
                            df_up["label"] = df_up["label"].apply(_normalize_label)
                            df_up = df_up[df_up["label"].isin(VALID_LABELS)]
                            for col in ["title", "body"]:
                                df_up[col] = df_up[col].fillna("").astype(str).str.strip()
                            df_up = df_up[(df_up["title"].str.len() <= 200) & (df_up["body"].str.len() <= 2000)]
                            df_up = df_up[(df_up["title"] != "") | (df_up["body"] != "")]
                            df_existing = pd.DataFrame(ss["labeled"])
                            if not df_existing.empty:
                                merged = df_up.merge(df_existing, on=["title", "body", "label"], how="left", indicator=True)
                                df_up = merged[merged["_merge"] == "left_only"].loc[:, ["title", "body", "label"]]
                            lint_counts = lint_dataset(df_up.to_dict(orient="records"))
                            st.dataframe(df_up.head(20), hide_index=True, width="stretch")
                            st.caption(f"Rows passing validation: {len(df_up)} | Lint -> credit cards: {lint_counts['credit_card']}, IBAN: {lint_counts['iban']}")
                            if len(df_up) > 0 and st.button("Import into labeled dataset", key="btn_import_csv"):
                                ss["labeled"].extend(df_up.to_dict(orient="records"))
                                ss["dataset_summary"] = compute_dataset_summary(ss["labeled"])
                                st.success(f"Imported {len(df_up)} rows into labeled dataset. Revisit builder to rebalance if needed.")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")


def _clear_dataset_preview_state() -> None:
    ss["dataset_preview"] = None
    ss["dataset_preview_config"] = None
    ss["dataset_preview_summary"] = None
    ss["dataset_preview_lint"] = None
    ss["dataset_manual_queue"] = None
    ss["dataset_controls_open"] = False


def _discard_preview() -> None:
    _clear_dataset_preview_state()
    ss["dataset_compare_delta"] = None
    ss["last_dataset_delta_story"] = explain_config_change(ss.get("dataset_config", DEFAULT_DATASET_CONFIG))


def render_train_stage():

    stage = STAGE_BY_KEY["train"]

    with section_surface():
        main_col, aside_col = st.columns([3, 2], gap="large")
        with main_col:
            st.subheader(f"{stage.icon} {stage.title} — teach the model to infer")
            render_eu_ai_quote("The EU AI Act says: “An AI system infers from the input it receives…”")
            st.write(
                "We’ll train the spam detector so it can **infer** whether each new email is **Spam** or **Safe**."
            )
            st.markdown(
                "- In the previous step, you prepared **labeled examples** (emails marked as spam or safe).  \n"
                "- The model now **looks for patterns** in those examples.  \n"
                "- With enough clear examples, it learns to **generalize** to new emails."
            )
        with aside_col:
            st.markdown("### Training checklist")
            st.markdown(
                "- Ensure both **spam** and **safe** emails are labeled.\n"
                "- Aim for a balanced mix of examples.\n"
                "- Use Nerd Mode to tune advanced parameters when you’re ready."
            )

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

    nerd_mode_train_enabled = render_nerd_mode_toggle(
        key="nerd_mode_train",
        title="Nerd Mode — advanced controls",
        description="Tweak the train/test split, solver iterations, and regularization strength.",
        icon="🔬",
    )
    if nerd_mode_train_enabled:
        with section_surface():
            colA, colB = st.columns(2)
            with colA:
                ss["train_params"]["test_size"] = st.slider(
                    "Hold-out test fraction",
                    min_value=0.10,
                    max_value=0.50,
                    value=float(ss["train_params"]["test_size"]),
                    step=0.05,
                    help="How much labeled data to keep aside as a mini 'exam' (not used for learning).",
                )
                ss["train_params"]["random_state"] = st.number_input(
                    "Random seed",
                    min_value=0,
                    value=int(ss["train_params"]["random_state"]),
                    step=1,
                    help="Fix this to make your train/test split reproducible.",
                )
            with colB:
                ss["train_params"]["max_iter"] = st.number_input(
                    "Max iterations (solver)",
                    min_value=200,
                    value=int(ss["train_params"]["max_iter"]),
                    step=100,
                    help="How many optimization steps the classifier can take before stopping.",
                )
                ss["train_params"]["C"] = st.number_input(
                    "Regularization strength C (inverse of regularization)",
                    min_value=0.01,
                    value=float(ss["train_params"]["C"]),
                    step=0.25,
                    format="%.2f",
                    help="Higher C fits training data more tightly; lower C adds regularization to reduce overfitting.",
                )

            st.info(
                "• **Hold-out fraction**: keeps part of the data for an honest test.  \\\n"
                "• **Random seed**: makes results repeatable.  \\\n"
                "• **Max iterations / C**: learning dials—defaults are fine; feel free to experiment."
            )

    with section_surface():
        action_col, context_col = st.columns([2, 3], gap="large")
        with action_col:
            st.markdown("### Train the model")
            st.markdown("👉 When you’re ready, click **Train**.")
            trigger_train = st.button("🚀 Train model", type="primary")
        with context_col:
            st.markdown(
                "- Uses the labeled dataset curated in the previous stage.\n"
                "- Applies the hyperparameters you set above."
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
                X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = train_test_split(
                    titles,
                    bodies,
                    y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y,
                )

                model = HybridEmbedFeatsLogReg()
                try:
                    model.lr.set_params(max_iter=max_iter, C=C_value)
                except Exception:
                    pass
                model = model.fit(X_tr_t, X_tr_b, y_tr)
                model.apply_numeric_adjustments(ss["numeric_adjustments"])
                ss["model"] = model
                ss["split_cache"] = (X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te)
                ss["eval_timestamp"] = datetime.now().isoformat(timespec="seconds")
                ss["eval_temp_threshold"] = float(ss.get("threshold", 0.6))
                try:
                    train_texts_cache = [combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)]
                    cache_train_embeddings(train_texts_cache)
                except Exception:
                    pass

    parsed_split = None
    y_tr_labels = None
    y_te_labels = None
    if ss.get("model") is not None and ss.get("split_cache") is not None:
        try:
            parsed_split = _parse_split_cache(ss["split_cache"])
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr_labels, y_te_labels = parsed_split

            # Existing success + story (kept)
            story = make_after_training_story(y_tr_labels, y_te_labels)
            st.success("Training finished.")
            st.markdown(story)

            # --- New: Training Storyboard (plain language, for everyone) ---
            with section_surface():
                st.markdown("### Training story — what just happened")

                # 1) What data was used
                ct = _counts(list(y_tr_labels))
                st.markdown(
                    f"- The system trained on **{len(y_tr_labels)} emails** "
                    f"(Spam: {ct['spam']}, Safe: {ct['safe']}).\n"
                    "- It looked for patterns that distinguish **Spam** from **Safe**.\n"
                    "- It saved these patterns as simple rules (weights) it can use later to decide."
                )

                # Mini class-balance chart
                try:
                    bal_df = pd.DataFrame(
                        {"class": ["spam", "safe"], "count": [ct["spam"], ct["safe"]]}
                    ).set_index("class")
                    st.caption("Training set balance")
                    st.bar_chart(bal_df, width="stretch")
                except Exception:
                    pass

                # 2) Top signals the model noticed (plain list)
                shown_any_signals = False
                try:
                    # Prefer numeric-feature view if available (Hybrid model)
                    if hasattr(ss["model"], "numeric_feature_details"):
                        nfd = ss["model"].numeric_feature_details().copy()
                        nfd["friendly_name"] = nfd["feature"].map(FEATURE_DISPLAY_NAMES)
                        # Positive weights → Spam, Negative → Safe
                        top_spam = (
                            nfd.sort_values("weight_per_std", ascending=False)
                            .head(3)["friendly_name"].tolist()
                        )
                        top_safe = (
                            nfd.sort_values("weight_per_std", ascending=True)
                            .head(3)["friendly_name"].tolist()
                        )
                        st.markdown("**Top signals the model picked up**")
                        st.write(f"• Toward **Spam**: {', '.join(top_spam) if top_spam else '—'}")
                        st.write(f"• Toward **Safe**: {', '.join(top_safe) if top_safe else '—'}")
                        st.caption(
                            "These are simple cues (e.g., links, ALL-CAPS bursts, money/urgency hints) that nudged decisions."
                        )
                        shown_any_signals = True
                except Exception:
                    pass

                if not shown_any_signals:
                    # Fallback wording if coefficients aren’t available
                    st.markdown("**What it learned**")
                    st.write(
                        "The model pays more attention to words and cues that frequently appear in spam (e.g., urgent offers, suspicious links) "
                        "and learns to ignore everyday business phrases that tend to be safe."
                    )

                # 3) A couple of concrete examples the model saw (subjects only)
                try:
                    if X_tr_t and X_tr_b and y_tr_labels:
                        train_subjects = list(X_tr_t)
                        y_arr = list(y_tr_labels)
                        # pick first spam + first safe subject line available
                        spam_subj = next((s for s, y in zip(train_subjects, y_arr) if y == "spam"), None)
                        safe_subj = next((s for s, y in zip(train_subjects, y_arr) if y == "safe"), None)
                        if spam_subj or safe_subj:
                            st.markdown("**Examples it learned from**")
                            if spam_subj:
                                st.write(f"• Spam example: *{spam_subj[:100]}{'…' if len(spam_subj)>100 else ''}*")
                            if safe_subj:
                                st.write(f"• Safe example: *{safe_subj[:100]}{'…' if len(safe_subj)>100 else ''}*")
                except Exception:
                    pass

                # 4) What this means / next step
                st.markdown(
                    "Your model now has a simple **mental map** of what Spam vs. Safe looks like. "
                    "Next, we’ll check how well this map works on emails it hasn’t seen before."
                )
                st.info("Go to **3) Evaluate** to test performance and choose a spam threshold.")

        except Exception as e:
            st.info(f"Training complete. (Details unavailable: {e})")
            parsed_split = None
            y_tr_labels = None
            y_te_labels = None

    if ss.get("nerd_mode_train") and ss.get("model") is not None and parsed_split:
        with st.expander("Nerd Mode — what just happened (technical)", expanded=True):
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr_labels, y_te_labels = parsed_split
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

            params = ss.get("train_params", {})
            st.markdown("**Parameters used**")
            st.markdown(
                f"- Hold-out fraction: {params.get('test_size', '—')}  \n"
                f"- Random seed: {params.get('random_state', '—')}  \n"
                f"- Max iterations: {params.get('max_iter', '—')}  \n"
                f"- C (inverse regularization): {params.get('C', '—')}"
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

                st.markdown("#### Plain-language explanations & manual tweaks")
                for row in coef_details.itertuples():
                    feat = row.feature
                    friendly = FEATURE_DISPLAY_NAMES.get(feat, feat)
                    explanation = FEATURE_PLAIN_LANGUAGE.get(feat, "")
                    st.markdown(f"**{friendly}** — {explanation}")
                    slider_key = f"adj_slider_{feat}"
                    current_setting = ss["numeric_adjustments"][feat]
                    if slider_key in st.session_state and st.session_state[slider_key] != current_setting:
                        st.session_state[slider_key] = current_setting
                    new_adj = st.slider(
                        f"Adjustment for {friendly} (log-odds per +1σ)",
                        min_value=-1.5,
                        max_value=1.5,
                        value=float(current_setting),
                        step=0.1,
                        key=slider_key,
                    )
                    if new_adj != ss["numeric_adjustments"][feat]:
                        ss["numeric_adjustments"][feat] = new_adj
                        if ss.get("model"):
                            ss["model"].apply_numeric_adjustments(ss["numeric_adjustments"])
            except Exception as e:
                st.caption(f"Coefficients unavailable: {e}")

            st.markdown("#### Embedding prototypes & nearest neighbors")
            try:
                if X_tr_t and X_tr_b:
                    X_train_texts = [combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)]
                    X_train_emb = encode_texts(X_train_texts)
                    y_train_arr = np.array(y_tr_labels)

                    def prototype_for(cls):
                        mask = y_train_arr == cls
                        if not np.any(mask):
                            return None
                        return X_train_emb[mask].mean(axis=0, keepdims=True)

                    def top_nearest(query_vec, k=5):
                        if query_vec is None:
                            return np.array([]), np.array([])
                        sims = (X_train_emb @ query_vec.T).ravel()
                        order = np.argsort(-sims)
                        top_k = order[: min(k, len(order))]
                        return top_k, sims[top_k]

                    for cls in CLASSES:
                        proto = prototype_for(cls)
                        if proto is None:
                            st.write(f"No training emails for {cls} yet.")
                            continue
                        idx, sims = top_nearest(proto, k=5)
                        st.markdown(f"**{cls.capitalize()} prototype — most similar training emails**")
                        for i, (ix, sc) in enumerate(zip(idx, sims), 1):
                            text_full = X_train_texts[ix]
                            parts = text_full.split("\n", 1)
                            title_i = parts[0]
                            body_i = parts[1] if len(parts) > 1 else ""
                            st.write(f"{i}. *{title_i}*  — sim={sc:.2f}")
                            preview = body_i[:200]
                            st.caption(preview + ("..." if len(body_i) > 200 else ""))
                else:
                    st.caption("Embedding details unavailable (no training texts).")
            except Exception as e:
                st.caption(f"Interpretability view unavailable: {e}")



def render_evaluate_stage():

    stage = STAGE_BY_KEY["evaluate"]

    if not (ss.get("model") and ss.get("split_cache")):
        with section_surface():
            st.subheader(f"{stage.icon} {stage.title} — How well does your spam detector perform?")
            st.info("Train a model first in the **Train** tab.")
        return

    cache = ss["split_cache"]
    if len(cache) == 4:
        X_tr, X_te, y_tr, y_te = cache
        texts_test = X_te
        X_te_t = X_te_b = None
    else:
        X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = cache
        texts_test = [(t or "") + "\n" + (b or "") for t, b in zip(X_te_t, X_te_b)]

    try:
        if len(cache) == 6:
            probs = ss["model"].predict_proba(X_te_t, X_te_b)
        else:
            probs = ss["model"].predict_proba(texts_test)
    except TypeError:
        probs = ss["model"].predict_proba(texts_test)

    classes = list(getattr(ss["model"], "classes_", []))
    if classes and "spam" in classes:
        idx_spam = classes.index("spam")
    else:
        idx_spam = 1 if probs.shape[1] > 1 else 0
    p_spam = probs[:, idx_spam]
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

                        if hasattr(model_obj, "scaler") and hasattr(model_obj, "lr"):
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

                model = HybridEmbedFeatsLogReg()
                try:
                    model.lr.set_params(max_iter=max_iter, C=C_value)
                except Exception:
                    pass
                model = model.fit(X_tr_t, X_tr_b, y_tr)
                model.apply_numeric_adjustments(ss["numeric_adjustments"])
                ss["model"] = model
                ss["split_cache"] = (X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te)
                ss["eval_timestamp"] = datetime.now().isoformat(timespec="seconds")
                ss["eval_temp_threshold"] = float(ss.get("threshold", 0.6))
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


def render_stage_navigation_controls(active_stage_key: str) -> None:
    """Display previous/next controls for the staged experience."""

    stage_keys = [stage.key for stage in STAGES]
    if active_stage_key not in stage_keys:
        return

    stage_index = stage_keys.index(active_stage_key)
    current_stage = STAGE_BY_KEY[active_stage_key]
    previous_stage = STAGE_BY_KEY.get(stage_keys[stage_index - 1]) if stage_index > 0 else None
    next_stage = (
        STAGE_BY_KEY.get(stage_keys[stage_index + 1])
        if stage_index < len(stage_keys) - 1
        else None
    )

    prev_col, info_col, next_col = st.columns([1, 2, 1], gap="large")

    with prev_col:
        if previous_stage is not None:
            st.button(
                f"⬅️ {previous_stage.icon} {previous_stage.title}",
                key=f"stage_nav_prev_{previous_stage.key}",
                on_click=set_active_stage,
                args=(previous_stage.key,),
                use_container_width=True,
            )

    with info_col:
        st.markdown(
            f"""
            <div class="stage-navigation-info">
                <div class="stage-navigation-step">Stage {stage_index + 1} of {len(stage_keys)}</div>
                <div class="stage-navigation-title">{html.escape(current_stage.icon)} {html.escape(current_stage.title)}</div>
                <p class="stage-navigation-description">{html.escape(current_stage.description)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with next_col:
        if next_stage is not None:
            st.button(
                f"{next_stage.icon} {next_stage.title} ➡️",
                key=f"stage_nav_next_{next_stage.key}",
                on_click=set_active_stage,
                args=(next_stage.key,),
                use_container_width=True,
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


active_stage = ss['active_stage']
renderer = STAGE_RENDERERS.get(active_stage, render_intro_stage)
renderer()
render_stage_navigation_controls(active_stage)

st.markdown("---")
st.caption("© demistifAI — Built for interactive learning and governance discussions.")
