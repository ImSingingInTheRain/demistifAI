"""Overview stage rendering helpers."""

from __future__ import annotations

import html
import re
import textwrap
from datetime import datetime
from typing import Any, Callable, ContextManager, Dict, List, MutableMapping, Optional

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from demistifai.constants import AUTONOMY_LEVELS, STAGE_BY_KEY, STAGE_INDEX, STAGES
from demistifai.ui.components.arch_demai import demai_architecture_markup, demai_architecture_styles
from demistifai.ui.components.mac_window import render_mac_window
from demistifai.ui.components.terminal.boot_sequence import (
    _DEFAULT_DEMAI_LINES,
    render_ai_act_terminal as render_boot_sequence_terminal,
)
from demistifai.core.nav import render_stage_top_grid


def render_overview_stage(
    ss: MutableMapping[str, Any],
    *,
    render_nerd_mode_toggle: Callable[..., Any],
    section_surface: Callable[..., ContextManager[DeltaGenerator]],
) -> None:
    """Render the overview stage.

    Parameters
    ----------
    ss:
        Streamlit session state mapping.
    render_nerd_mode_toggle:
        Callback to render the Nerd Mode toggle control.
    section_surface:
        Context manager that wraps stage sections with shared styling.
    """

    next_stage_key: Optional[str] = None
    overview_index = STAGE_INDEX.get("overview")
    if overview_index is not None and overview_index < len(STAGES) - 1:
        next_stage_key = STAGES[overview_index + 1].key

    labeled_examples = ss.get("labeled") or []
    dataset_rows = len(labeled_examples)
    dataset_last_built = ss.get("dataset_last_built_at")
    if dataset_last_built:
        try:
            built_dt = datetime.fromisoformat(dataset_last_built)
            dataset_timestamp = f"Dataset build: {built_dt.strftime('%b %d, %H:%M')}"
        except ValueError:
            dataset_timestamp = f"Dataset build: {dataset_last_built}"
    else:
        dataset_timestamp = "Dataset build: pending"

    incoming_records = ss.get("incoming") or []
    incoming_count = len(incoming_records)
    incoming_seed = ss.get("incoming_seed")
    autonomy_label = str(ss.get("autonomy", AUTONOMY_LEVELS[0]))
    adaptiveness_enabled = bool(ss.get("adaptive", False))
    nerd_enabled = bool(ss.get("nerd_mode_train") or ss.get("nerd_mode"))

    def _render_overview_terminal(slot: DeltaGenerator) -> None:
        with slot:
            render_boot_sequence_terminal(
                demai_lines=_DEFAULT_DEMAI_LINES,
                speed_type_ms=20,
                pause_between_ops_ms=360,
            )

    def _render_overview_nerd_toggle(slot: DeltaGenerator) -> None:
        render_nerd_mode_toggle(
            key="nerd_mode",
            title="Nerd Mode",
            target=slot,
        )

    render_stage_top_grid(
        "overview",
        left_renderer=_render_overview_terminal,
        right_first_renderer=_render_overview_nerd_toggle,
    )

    st.markdown(
        """
        <style>
        .overview-intro-card {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.16), rgba(14, 116, 144, 0.16));
            border-radius: 1.25rem;
            padding: 1.6rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.4rem;
        }
        .overview-intro-card__header {
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .overview-intro-card__icon {
            font-size: 1.9rem;
            line-height: 1;
            background: rgba(15, 23, 42, 0.08);
            border-radius: 0.9rem;
            padding: 0.55rem 0.95rem;
            box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
        }
        .overview-intro-card__eyebrow {
            font-size: 0.75rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-weight: 700;
            color: rgba(15, 23, 42, 0.6);
            display: inline-block;
            margin-bottom: 0.35rem;
        }
        .overview-intro-card__title {
            margin: 0;
            font-size: 1.5rem;
            font-weight: 700;
            color: #0f172a;
        }
        .overview-intro-card__body {
            margin: 0;
            color: rgba(15, 23, 42, 0.82);
            font-size: 0.98rem;
            line-height: 1.65;
        }
        .overview-checklist {
            margin: 1.1rem 0 0;
            padding: 0;
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 0.6rem;
        }
        .overview-checklist li {
            display: flex;
            gap: 0.6rem;
            align-items: flex-start;
            color: rgba(15, 23, 42, 0.78);
            font-size: 0.95rem;
            line-height: 1.6;
        }
        .overview-checklist li::before {
            content: "✔";
            font-weight: 700;
            color: #1d4ed8;
            margin-top: 0.1rem;
        }
        .overview-intro-actions {
            margin-top: 1.35rem;
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }
        .overview-intro-actions__hint {
            font-size: 0.85rem;
            color: rgba(15, 23, 42, 0.62);
        }
        .overview-cta {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.7rem 1.1rem;
            border-radius: 999px;
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: white;
            font-weight: 600;
            text-decoration: none;
            border: none;
        }
        .overview-cta__next {
            background: rgba(37, 99, 235, 0.12);
            color: #1d4ed8;
        }
        .overview-intro-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.2rem;
        }
        .status-card {
            padding: 1.1rem 1.3rem;
            border-radius: 1.1rem;
            background: rgba(248, 250, 252, 0.92);
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.08);
            display: grid;
            gap: 0.45rem;
        }
        .status-card__eyebrow {
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            font-weight: 700;
            color: rgba(15, 23, 42, 0.55);
        }
        .status-card__value {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0f172a;
        }
        .status-card__body {
            margin: 0;
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.7);
        }
        .status-card__footnote {
            margin: 0;
            font-size: 0.8rem;
            color: rgba(15, 23, 42, 0.55);
        }
        .mission-brief {
            background: rgba(255, 255, 255, 0.96);
            border-radius: 1.4rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 28px 46px rgba(15, 23, 42, 0.12);
            padding: 1.75rem 1.9rem;
            display: grid;
            gap: 1.4rem;
        }
        .mission-brief__header {
            display: flex;
            gap: 1rem;
            align-items: center;
        }
        .mission-brief__icon {
            font-size: 2.1rem;
        }
        .mission-brief__eyebrow {
            font-size: 0.78rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: rgba(37, 99, 235, 0.7);
            font-weight: 700;
        }
        .mission-brief__title {
            margin: 0;
            font-size: 1.45rem;
            font-weight: 700;
            color: #0f172a;
        }
        .mission-brief__bridge {
            margin: 0;
            font-size: 0.98rem;
            color: rgba(15, 23, 42, 0.75);
        }
        .mission-brief__grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 1.8rem;
        }
        .mission-brief__objective {
            display: grid;
            gap: 0.95rem;
        }
        .mission-brief__list {
            margin: 0;
            padding-left: 1.2rem;
            display: grid;
            gap: 0.55rem;
            font-size: 0.92rem;
            color: rgba(15, 23, 42, 0.72);
        }
        .mission-brief__preview {
            display: grid;
            gap: 0.9rem;
        }
        .mission-brief__preview-card {
            background: rgba(37, 99, 235, 0.08);
            border-radius: 1.1rem;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.12);
            display: grid;
            gap: 0.65rem;
        }
        .mission-brief__preview-card--mailbox {
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.16), rgba(59, 130, 246, 0.14));
        }
        .mission-brief__preview-content {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
            width: 100%;
        }
        .mission-brief__preview-eyebrow {
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: rgba(37, 99, 235, 0.8);
            font-weight: 700;
        }
        .mission-brief__preview-title {
            margin: 0;
            font-size: 1.1rem;
            color: #1d4ed8;
            font-weight: 600;
        }
        .mission-brief__preview-intro {
            margin: 0;
            font-size: 0.92rem;
            color: rgba(15, 23, 42, 0.7);
        }
        .mission-brief__inbox-list {
            list-style: none;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
        }
        .mission-brief__inbox-item {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        .mission-brief__subject {
            font-weight: 600;
            font-size: 0.96rem;
            color: #0f172a;
        }
        .mission-brief__snippet {
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.68);
            line-height: 1.4;
        }
        .mission-brief__empty {
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.65);
        }
        .mission-brief__preview-note {
            margin: 0;
            font-size: 0.85rem;
            color: rgba(15, 23, 42, 0.6);
        }
        .mission-brief__highlights {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
        }
        .mission-highlight {
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            background: rgba(255, 255, 255, 0.85);
            border-radius: 999px;
            padding: 0.55rem 0.95rem;
            font-size: 0.9rem;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.12);
            border: 1px solid rgba(37, 99, 235, 0.18);
        }
        .mission-highlight__icon {
            font-size: 1.1rem;
        }
        .mailbox-preview {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 1.35rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 20px 44px rgba(15, 23, 42, 0.1);
            overflow: hidden;
        }
        .mailbox-preview__header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.1rem 1.4rem;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(59, 130, 246, 0.1));
            border-bottom: 1px solid rgba(37, 99, 235, 0.16);
        }
        .mailbox-preview__header h4 {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 600;
            color: #1e3a8a;
        }
        .mailbox-preview__header span {
            font-size: 0.85rem;
            color: rgba(15, 23, 42, 0.65);
        }
        .mail-rows {
            display: flex;
            flex-direction: column;
        }
        .mail-row {
            display: grid;
            grid-template-columns: auto 1fr auto;
            align-items: start;
            gap: 1rem;
            padding: 1rem 1.4rem;
            border-bottom: 1px solid rgba(15, 23, 42, 0.06);
            background: rgba(248, 250, 252, 0.7);
        }
        .mail-row:nth-child(even) {
            background: rgba(255, 255, 255, 0.92);
        }
        .mail-row__status {
            width: 12px;
            height: 12px;
            border-radius: 999px;
            background: linear-gradient(135deg, #1d4ed8, #2563eb);
            margin-top: 0.35rem;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.18);
        }
        .mail-row__details {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }
        .mail-row__subject {
            margin: 0;
            font-size: 0.98rem;
            font-weight: 600;
            color: #0f172a;
        }
        .mail-row__snippet {
            margin: 0;
            font-size: 0.9rem;
            color: rgba(15, 23, 42, 0.68);
            line-height: 1.45;
        }
        .mail-row__meta {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 0.4rem;
            font-size: 0.8rem;
            color: rgba(15, 23, 42, 0.6);
        }
        .mail-row__tag {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            background: rgba(37, 99, 235, 0.14);
            color: #1d4ed8;
            font-weight: 600;
            font-size: 0.75rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .mail-empty {
            padding: 1.35rem 1.4rem;
            font-size: 0.92rem;
            color: rgba(15, 23, 42, 0.65);
        }
        @media (max-width: 960px) {
            .mission-brief__grid {
                grid-template-columns: 1fr;
                gap: 1.4rem;
            }
            .mission-brief {
                padding: 1.35rem 1.4rem;
            }
            .mission-brief__preview {
                margin-top: 0.4rem;
            }
            .mail-row {
                grid-template-columns: auto 1fr;
            }
            .mail-row__meta {
                align-items: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    dataset_value = f"{dataset_rows:,} labeled emails ready" if dataset_rows else "Starter dataset loading"
    incoming_value = f"{incoming_count:,} emails queued"
    if incoming_seed is not None:
        incoming_value = f"{incoming_value} • seed {incoming_seed}"

    adaptiveness_value = (
        "Adaptiveness on — confirmations feed future training"
        if adaptiveness_enabled
        else "Adaptiveness off — corrections stay manual"
    )

    render_mac_window(
        st,
        title="System snapshot",
        columns=1,
        ratios=(1,),
        col_html=[
            demai_architecture_markup(
                nerd_mode=nerd_enabled,
                active_stage="overview",
            )
        ],
        id_suffix="overview-mac-placeholder",
        scoped_css=demai_architecture_styles(),
    )

    preview_records: List[Dict[str, Any]] = []
    if incoming_records:
        df_incoming = pd.DataFrame(incoming_records)
        preview_records = df_incoming.head(5).to_dict("records")

    def _format_snippet(text: Optional[str], *, limit: int = 110) -> str:
        snippet = (text or "").strip()
        snippet = re.sub(r"\s+", " ", snippet)
        if len(snippet) > limit:
            snippet = snippet[: limit - 1].rstrip() + "…"
        return snippet

    inbox_rows_html = []
    for record in preview_records:
        subject = html.escape(record.get("title", "Untitled email"))
        snippet = html.escape(_format_snippet(record.get("body")))
        inbox_rows_html.append(
            textwrap.dedent(
                """
                <div class="mail-row">
                    <div class="mail-row__status"></div>
                    <div class="mail-row__details">
                        <p class="mail-row__subject">{subject}</p>
                        <p class="mail-row__snippet">{snippet}</p>
                    </div>
                    <div class="mail-row__meta">
                        <span class="mail-row__tag">Queued</span>
                    </div>
                </div>
                """
            ).format(subject=subject, snippet=snippet).strip()
        )

    if not inbox_rows_html:
        inbox_rows_html.append(
            """
            <div class="mail-empty">
                Inbox feed warming up — generate incoming emails in the Use stage to populate this preview.
            </div>
            """.strip()
        )

    next_stage_cta_html = ""
    if next_stage_key and next_stage_key in STAGE_BY_KEY:
        next_stage_cta_html = (
            f"<a class=\"overview-cta overview-cta__next\" href=\"#\">Next: {html.escape(STAGE_BY_KEY[next_stage_key].title)}</a>"
        )

    overview_intro_html = textwrap.dedent(
        f"""
        <div class="overview-intro-card">
            <div class="overview-intro-card__header">
                <div class="overview-intro-card__icon">🛰️</div>
                <div>
                    <span class="overview-intro-card__eyebrow">Lifecycle orientation</span>
                    <h2 class="overview-intro-card__title">Welcome to the demistifAI control room</h2>
                </div>
            </div>
            <p class="overview-intro-card__body">
                This guided workspace links every stage of your AI lifecycle. Start with a mission briefing,
                keep an eye on the system snapshot, then follow the checklist to build, evaluate, and govern your spam detector.
            </p>
            <div class="overview-intro-actions">
                <div class="overview-intro-actions__hint">First time here? Move through the stages from left to right.</div>
                {next_stage_cta_html}
            </div>
        </div>
        """
    ).strip()

    with section_surface():
        st.markdown(overview_intro_html, unsafe_allow_html=True)

        status_cards_html = textwrap.dedent(
            f"""
            <div class="overview-intro-grid">
                <div class="status-card">
                    <span class="status-card__eyebrow">Dataset status</span>
                    <span class="status-card__value">{dataset_value}</span>
                    <p class="status-card__body">Labeled emails power model training and evaluation.</p>
                    <p class="status-card__footnote">{dataset_timestamp}</p>
                </div>
                <div class="status-card">
                    <span class="status-card__eyebrow">Inbox feed</span>
                    <span class="status-card__value">{incoming_value}</span>
                    <p class="status-card__body">Generate synthetic emails to stress-test routing decisions.</p>
                    <p class="status-card__footnote">Use stage controls to refresh the queue.</p>
                </div>
                <div class="status-card">
                    <span class="status-card__eyebrow">Operating mode</span>
                    <span class="status-card__value">{autonomy_label}</span>
                    <p class="status-card__body">Adjust autonomy and adaptiveness to balance control and scale.</p>
                    <p class="status-card__footnote">{adaptiveness_value}</p>
                </div>
            </div>
            """
        ).strip()
        st.markdown(status_cards_html, unsafe_allow_html=True)

    mailbox_html = textwrap.dedent(
        """
        <div class="mailbox-preview">
            <div class="mailbox-preview__header">
                <h4>Live inbox preview</h4>
                <span>First {count} messages waiting for triage</span>
            </div>
            <div class="mail-rows">{rows}</div>
        </div>
        """
    ).format(count=len(preview_records) or 0, rows="".join(inbox_rows_html)).strip()

    mission_html = textwrap.dedent(
        f"""
        <div class="mission-brief">
            <div class="mission-brief__header">
                <span class="mission-brief__icon">🎯</span>
                <div>
                    <span class="mission-brief__eyebrow">Mission briefing</span>
                    <h3 class="mission-brief__title">Your mission</h3>
                </div>
            </div>
            <p class="mission-brief__bridge">You’re stepping into the control room of an email triage machine. The inbox snapshot on the right matches the live preview you’ll work from in a moment.</p>
            <div class="mission-brief__grid">
                <div class="mission-brief__objective">
                    <p>Keep unwanted email out while letting the important messages through. You’ll steer the controls, set the operating thresholds, and verify the system’s choices.</p>
                    <ul class="mission-brief__list">
                        <li>Scan the inbox feed and spot risky patterns early.</li>
                        <li>Decide how strict the spam filter should be and when autonomy applies.</li>
                        <li>Confirm or correct decisions so the system learns your judgement.</li>
                    </ul>
                </div>
                <div class="mission-brief__preview">
                    <div class="mission-brief__preview-card mission-brief__preview-card--mailbox">
                        <span class="mission-brief__preview-eyebrow">Inbox stream</span>
                        {mailbox_html}
                    </div>
                </div>
            </div>
        </div>
        """
    ).strip()

    with section_surface():
        st.markdown(mission_html, unsafe_allow_html=True)

    if nerd_enabled:
        with section_surface():
            st.markdown(
                """
                <div class="overview-subheading">
                    <span class="overview-subheading__eyebrow">Deep dive</span>
                    <h3>🔬 Nerd Mode — technical details</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

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


