"""Dataset builder panel helpers for the Prepare stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Dict, Optional

import html
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from demistifai.dataset import (
    ATTACHMENT_MIX_PRESETS,
    DEFAULT_ATTACHMENT_MIX,
    DEFAULT_DATASET_CONFIG,
    DatasetConfig,
    STARTER_LABELED,
    compute_dataset_summary,
    dataset_delta_story,
    dataset_summary_delta,
    explain_config_change,
)
from demistifai.core.state import _push_data_stage_flash, _set_advanced_knob_state
from demistifai.core.utils import streamlit_rerun

from pages.data.dataset_io import (
    SessionState,
    generate_preview_from_config,
    reset_dataset_to_baseline,
)
from pages.data.ui import build_compare_panel_html, cli_command_line, cli_comment


@dataclass
class DatasetBuilderResult:
    """Outcome of rendering the dataset builder surface."""

    current_summary: Dict[str, Any]
    compare_panel_html: str


def render_prepare_panel(
    *,
    slot: DeltaGenerator,
    section_surface: Callable[[Optional[str]], ContextManager[DeltaGenerator]],
    render_nerd_mode_toggle: Callable[..., bool],
) -> bool:
    """Render the Nerd Mode toggle panel and return its state."""

    with slot:
        with section_surface():
            nerd_mode_enabled = render_nerd_mode_toggle(
                key="nerd_mode_data",
                title="Nerd Mode",
                description="Expose feature prevalence, randomness, diagnostics, and CSV import when you need them.",
            )
    return bool(nerd_mode_enabled)


def render_dataset_builder(
    *,
    slot: DeltaGenerator,
    section_surface: Callable[[Optional[str]], ContextManager[DeltaGenerator]],
    ss: SessionState,
    current_summary: Dict[str, Any],
    nerd_mode_data_enabled: bool,
    prepare_records_callback: Callable[[Any], None],
) -> DatasetBuilderResult:
    """Render the dataset builder form and return the resulting context."""

    delta_summary: Optional[Dict[str, Any]] = ss.get("dataset_compare_delta")
    delta_text = ""
    if delta_summary:
        delta_text = dataset_delta_story(delta_summary)
    if not delta_text and ss.get("last_dataset_delta_story"):
        delta_text = ss["last_dataset_delta_story"]
    if not delta_text:
        delta_text = explain_config_change(ss.get("dataset_config", DEFAULT_DATASET_CONFIG))

    base_summary_for_delta: Optional[Dict[str, Any]] = None
    target_summary_for_delta: Optional[Dict[str, Any]] = None
    compare_panel_html = ""

    with slot:
        with section_surface("dataset-builder-surface"):
            st.markdown("<div class='dataset-builder'>", unsafe_allow_html=True)
            st.markdown(
                cli_command_line("dataset.builder --interactive"),
                unsafe_allow_html=True,
            )
            st.markdown(
                cli_comment(
                    "Tune dataset volume, class balance, and adversarial knobs before generating a preview."
                ),
                unsafe_allow_html=True,
            )

            if ss.get("dataset_preview_summary"):
                base_summary_for_delta = current_summary
                target_summary_for_delta = ss["dataset_preview_summary"]
            elif ss.get("previous_dataset_summary") and delta_summary:
                base_summary_for_delta = ss.get("previous_dataset_summary")
                target_summary_for_delta = ss.get("dataset_summary", current_summary)
            else:
                base_summary_for_delta = compute_dataset_summary(STARTER_LABELED)
                target_summary_for_delta = current_summary
                if delta_summary is None:
                    delta_summary = dataset_summary_delta(
                        base_summary_for_delta, target_summary_for_delta
                    )

            preview_clicked = False
            reset_clicked = False

            cfg = ss.get("dataset_config", DEFAULT_DATASET_CONFIG)
            _set_advanced_knob_state(cfg)

            spam_ratio_default = float(cfg.get("spam_ratio", 0.5))
            spam_share_default = int(round(spam_ratio_default * 100))
            spam_share_default = min(max(spam_share_default, 20), 80)
            if spam_share_default % 5 != 0:
                spam_share_default = int(5 * round(spam_share_default / 5))

            with st.form("dataset_builder_form"):
                st.markdown(
                    "<div class='dataset-builder__form-shell'>",
                    unsafe_allow_html=True,
                )

                base_command_col, base_control_col = st.columns(
                    [0.6, 0.4], gap="large"
                )
                with base_control_col:
                    dataset_size = st.radio(
                        "Dataset size",
                        options=[100, 300, 500],
                        index=[100, 300, 500].index(int(cfg.get("n_total", 500)))
                        if int(cfg.get("n_total", 500)) in [100, 300, 500]
                        else 2,
                        help="Preset sizes illustrate how data volume influences learning (guarded ≤500).",
                        label_visibility="collapsed",
                    )
                with base_command_col:
                    base_command_col.markdown(
                        cli_command_line(
                            "dataset.config --size <span class='dataset-builder__value'>{}</span>".format(
                                html.escape(str(dataset_size))
                            ),
                            unsafe=True,
                        ),
                        unsafe_allow_html=True,
                    )
                    base_command_col.markdown(
                        cli_comment(
                            "Preset sizes illustrate how data volume influences learning (guarded ≤500)."
                        ),
                        unsafe_allow_html=True,
                    )

                spam_command_col, spam_control_col = st.columns(
                    [0.6, 0.4], gap="large"
                )
                with spam_control_col:
                    spam_share_pct = st.slider(
                        "Spam share",
                        min_value=20,
                        max_value=80,
                        value=spam_share_default,
                        step=5,
                        help="Adjust prevalence to explore bias/recall trade-offs.",
                        label_visibility="collapsed",
                    )
                with spam_command_col:
                    spam_command_col.markdown(
                        cli_command_line(
                            "dataset.config --spam-share <span class='dataset-builder__value'>{}%</span>".format(
                                html.escape(str(spam_share_pct))
                            ),
                            unsafe=True,
                        ),
                        unsafe_allow_html=True,
                    )
                    spam_command_col.markdown(
                        cli_comment(
                            "Adjust prevalence to explore bias/recall trade-offs."
                        ),
                        unsafe_allow_html=True,
                    )

                edge_command_col, edge_control_col = st.columns(
                    [0.6, 0.4], gap="large"
                )
                with edge_control_col:
                    edge_cases = st.slider(
                        "Edge cases",
                        min_value=0,
                        max_value=10,
                        value=int(cfg.get("edge_cases", 2)),
                        help="Surface tricky look-alikes to test your preview set.",
                        label_visibility="collapsed",
                    )
                with edge_command_col:
                    edge_command_col.markdown(
                        cli_command_line(
                            "dataset.config --edge-cases <span class='dataset-builder__value'>{}</span>".format(
                                html.escape(str(edge_cases))
                            ),
                            unsafe=True,
                        ),
                        unsafe_allow_html=True,
                    )
                    edge_command_col.markdown(
                        cli_comment(
                            "Surface tricky look-alikes to test your preview set."
                        ),
                        unsafe_allow_html=True,
                    )

                if nerd_mode_data_enabled:
                    st.markdown(
                        "<div class='dataset-builder__divider'><span>nerd mode</span></div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        cli_comment(
                            "Fine-tune suspicious links, domains, tone, attachments, randomness, and demos before generating a preview."
                        ),
                        unsafe_allow_html=True,
                    )
                    attachment_keys = list(ATTACHMENT_MIX_PRESETS.keys())
                    adv_col_a, adv_col_b = st.columns(2, gap="large")
                    with adv_col_a:
                        adv_links = st.slider(
                            "Suspicious links per spam email",
                            min_value=0,
                            max_value=2,
                            value=int(st.session_state.get("adv_links_level", 1)),
                            help="Controls how many sketchy URLs appear in spam examples (0–2).",
                            key="adv_links_level",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            cli_command_line(
                                "dataset.advanced --links <span class='dataset-builder__value'>{}</span>".format(
                                    html.escape(str(adv_links))
                                ),
                                unsafe=True,
                            ),
                            unsafe_allow_html=True,
                        )

                        adv_tld = st.select_slider(
                            "Suspicious TLD frequency",
                            options=["low", "med", "high"],
                            value=str(
                                st.session_state.get(
                                    "adv_tld_level", cfg.get("susp_tld_level", "med")
                                )
                            ),
                            key="adv_tld_level",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            cli_command_line(
                                "dataset.advanced --tld <span class='dataset-builder__value'>{}</span>".format(
                                    html.escape(str(adv_tld))
                                ),
                                unsafe=True,
                            ),
                            unsafe_allow_html=True,
                        )

                        adv_caps = st.select_slider(
                            "ALL-CAPS / urgency intensity",
                            options=["low", "med", "high"],
                            value=str(
                                st.session_state.get(
                                    "adv_caps_level", cfg.get("caps_intensity", "med")
                                )
                            ),
                            key="adv_caps_level",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            cli_command_line(
                                "dataset.advanced --caps <span class='dataset-builder__value'>{}</span>".format(
                                    html.escape(str(adv_caps))
                                ),
                                unsafe=True,
                            ),
                            unsafe_allow_html=True,
                        )
                    with adv_col_b:
                        adv_money = st.select_slider(
                            "Money symbols & urgency",
                            options=["off", "low", "high"],
                            value=str(
                                st.session_state.get(
                                    "adv_money_level", cfg.get("money_urgency", "low")
                                )
                            ),
                            key="adv_money_level",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            cli_command_line(
                                "dataset.advanced --money <span class='dataset-builder__value'>{}</span>".format(
                                    html.escape(str(adv_money))
                                ),
                                unsafe=True,
                            ),
                            unsafe_allow_html=True,
                        )

                        attachment_choice_cli = st.selectbox(
                            "Attachment lure mix",
                            options=attachment_keys,
                            index=attachment_keys.index(
                                st.session_state.get("adv_attachment_choice", "Balanced")
                            )
                            if st.session_state.get("adv_attachment_choice", "Balanced") in attachment_keys
                            else 1,
                            help="Choose how often risky attachments (HTML/ZIP/XLSM/EXE) appear vs. safer PDFs.",
                            key="adv_attachment_choice",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            cli_command_line(
                                "dataset.advanced --attachments <span class='dataset-builder__value'>{}</span>".format(
                                    html.escape(str(attachment_choice_cli))
                                ),
                                unsafe=True,
                            ),
                            unsafe_allow_html=True,
                        )

                        adv_noise = st.slider(
                            "Label noise (%)",
                            min_value=0.0,
                            max_value=5.0,
                            step=1.0,
                            value=float(
                                st.session_state.get(
                                    "adv_label_noise_pct", float(cfg.get("label_noise_pct", 0.0))
                                )
                            ),
                            key="adv_label_noise_pct",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            cli_command_line(
                                "dataset.advanced --label-noise <span class='dataset-builder__value'>{}%</span>".format(
                                    html.escape(str(int(adv_noise)))
                                ),
                                unsafe=True,
                            ),
                            unsafe_allow_html=True,
                        )

                        adv_seed = st.number_input(
                            "Random seed",
                            min_value=0,
                            value=int(st.session_state.get("adv_seed", int(cfg.get("seed", 42)))),
                            key="adv_seed",
                            help="Keep this fixed for reproducibility.",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            cli_command_line(
                                "dataset.advanced --seed <span class='dataset-builder__value'>{}</span>".format(
                                    html.escape(str(adv_seed))
                                ),
                                unsafe=True,
                            ),
                            unsafe_allow_html=True,
                        )

                        adv_poison = st.toggle(
                            "Data poisoning demo (synthetic)",
                            value=bool(
                                st.session_state.get(
                                    "adv_poison_demo", bool(cfg.get("poison_demo", False))
                                )
                            ),
                            key="adv_poison_demo",
                            help="Adds a tiny malicious distribution shift labeled as safe to show metric degradation.",
                            label_visibility="collapsed",
                        )
                        st.markdown(
                            cli_command_line(
                                "dataset.advanced --poison-demo <span class='dataset-builder__value'>{}</span>".format(
                                    html.escape("on" if adv_poison else "off")
                                ),
                                unsafe=True,
                            ),
                            unsafe_allow_html=True,
                        )

                st.markdown("<div class='cta-sticky'>", unsafe_allow_html=True)
                btn_primary, btn_secondary = st.columns([1, 1])
                with btn_primary:
                    preview_clicked = st.form_submit_button(
                        "Generate preview", type="primary"
                    )
                with btn_secondary:
                    reset_clicked = st.form_submit_button(
                        "Reset to baseline", type="secondary"
                    )
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            spam_ratio = float(spam_share_pct) / 100.0

            if reset_clicked:
                baseline_summary = reset_dataset_to_baseline(ss)
                _push_data_stage_flash(
                    "success", f"Dataset reset to starter baseline ({len(STARTER_LABELED)} rows)."
                )
                _set_advanced_knob_state(ss["dataset_config"], force=True)
                if ss.get("_needs_advanced_knob_rerun"):
                    streamlit_rerun()
                current_summary = baseline_summary
                delta_summary = ss.get("dataset_compare_delta")
                delta_text = explain_config_change(
                    ss.get("dataset_config", DEFAULT_DATASET_CONFIG)
                )
                base_summary_for_delta = compute_dataset_summary(STARTER_LABELED)
                target_summary_for_delta = current_summary
                prepare_records_callback(ss.get("labeled"))

            if preview_clicked:
                attachment_choice = st.session_state.get(
                    "adv_attachment_choice",
                    next(
                        (
                            name
                            for name, mix in ATTACHMENT_MIX_PRESETS.items()
                            if mix == cfg.get("attachments_mix", DEFAULT_ATTACHMENT_MIX)
                        ),
                        "Balanced",
                    ),
                )
                attachment_mix = ATTACHMENT_MIX_PRESETS.get(
                    attachment_choice, DEFAULT_ATTACHMENT_MIX
                ).copy()
                links_level_value = int(
                    st.session_state.get(
                        "adv_links_level",
                        int(str(cfg.get("susp_link_level", "1"))),
                    )
                )
                tld_level_value = str(
                    st.session_state.get("adv_tld_level", cfg.get("susp_tld_level", "med"))
                )
                caps_level_value = str(
                    st.session_state.get("adv_caps_level", cfg.get("caps_intensity", "med"))
                )
                money_level_value = str(
                    st.session_state.get("adv_money_level", cfg.get("money_urgency", "low"))
                )
                noise_pct_value = float(
                    st.session_state.get(
                        "adv_label_noise_pct", float(cfg.get("label_noise_pct", 0.0))
                    )
                )
                seed_value = int(
                    st.session_state.get("adv_seed", int(cfg.get("seed", 42)))
                )
                poison_demo_value = bool(
                    st.session_state.get("adv_poison_demo", bool(cfg.get("poison_demo", False)))
                )
                config: DatasetConfig = {
                    "seed": int(seed_value),
                    "n_total": int(dataset_size),
                    "spam_ratio": float(spam_ratio),
                    "susp_link_level": str(int(links_level_value)),
                    "susp_tld_level": tld_level_value,
                    "caps_intensity": caps_level_value,
                    "money_urgency": money_level_value,
                    "attachments_mix": attachment_mix,
                    "edge_cases": int(edge_cases),
                    "label_noise_pct": float(noise_pct_value),
                    "poison_demo": bool(poison_demo_value),
                }
                preview_summary_local, delta_summary_generated, delta_story = generate_preview_from_config(
                    config,
                    current_summary=current_summary,
                    ss=ss,
                )
                delta_summary = delta_summary_generated
                if delta_summary:
                    delta_text = delta_story
                base_summary_for_delta = current_summary
                target_summary_for_delta = preview_summary_local

            st.markdown("</div>", unsafe_allow_html=True)

    compare_panel_html = build_compare_panel_html(
        base_summary_for_delta,
        target_summary_for_delta,
        delta_summary,
        delta_text,
    )

    return DatasetBuilderResult(
        current_summary=current_summary,
        compare_panel_html=compare_panel_html,
    )
