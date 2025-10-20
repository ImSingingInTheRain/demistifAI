"""Overview stage rendering helpers."""

from __future__ import annotations

from typing import Any, Callable, ContextManager, Dict, List, MutableMapping

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from demistifai.ui.components import render_stage_top_grid
from demistifai.ui.components.overview import (
    demai_architecture_markup,
    demai_architecture_styles,
    mailbox_preview_markup,
    mission_brief_styles,
    mission_overview_column_markup,
)
from demistifai.ui.components.terminal.boot_sequence import (
    _DEFAULT_DEMAI_LINES,
    render_ai_act_terminal as render_boot_sequence_terminal,
)
from demistifai.ui.theme.macos_window import macos_window_markup, macos_window_styles


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

    incoming_records = ss.get("incoming") or []
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

    macos_theme_key = "macos_window_theme_injected"
    if not ss.get(macos_theme_key):
        st.markdown(macos_window_styles(), unsafe_allow_html=True)
        ss[macos_theme_key] = True

    architecture_window = macos_window_markup(
        "System snapshot",
        columns=1,
        column_blocks=(
            demai_architecture_markup(
                nerd_mode=nerd_enabled,
                active_stage="overview",
            ),
        ),
        id_suffix="overview-mac-placeholder",
        column_variant="flush",
    )
    architecture_styles = demai_architecture_styles()
    if architecture_styles:
        architecture_window = f"{architecture_styles}\n{architecture_window}"
    st.markdown(architecture_window, unsafe_allow_html=True)

    preview_records: List[Dict[str, Any]] = []
    if incoming_records:
        df_incoming = pd.DataFrame(incoming_records)
        preview_records = df_incoming.head(5).to_dict("records")

    mailbox_html = mailbox_preview_markup(preview_records)
    mission_html = mission_overview_column_markup()

    mission_window = macos_window_markup(
        "Mission briefing",
        subtitle="Control room orientation",
        columns=2,
        ratios=(1.1, 0.9),
        id_suffix="overview-mission-brief",
        column_blocks=(mission_html, mailbox_html),
    )
    mission_styles = mission_brief_styles()
    if mission_styles:
        mission_window = f"{mission_styles}\n{mission_window}"
    st.markdown(mission_window, unsafe_allow_html=True)

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


