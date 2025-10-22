"""Dataset builder panel helpers for the Prepare stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Dict, Optional

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
from demistifai.ui.components.data import build_compare_panel_html


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

    cfg = ss.get("dataset_config", DEFAULT_DATASET_CONFIG)
    live_cfg_seed = ss.get("dataset_builder_live_config")
    if live_cfg_seed is None:
        live_cfg_seed = ss.get("dataset_preview_config")
    if live_cfg_seed is None:
        live_cfg_seed = cfg
    pending_config: DatasetConfig = {
        **DEFAULT_DATASET_CONFIG,
        **dict(live_cfg_seed or {}),
    }

    dataset_total_display = int(
        pending_config.get("n_total", DEFAULT_DATASET_CONFIG.get("n_total", 500))
    )

    with slot:
        with section_surface("dataset-builder-surface"):
            st.markdown("<div class='dataset-builder'>", unsafe_allow_html=True)
            st.markdown(
                "#### Configure the preview dataset",
            )
            st.caption(
                "Set how many emails to synthesize, balance spam vs. safe traffic, and decide how many tricky look-alikes to include before generating a preview."
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

            spam_ratio_default = float(pending_config.get("spam_ratio", 0.5))
            spam_share_default = int(round(spam_ratio_default * 100))
            spam_share_default = min(max(spam_share_default, 20), 80)
            if spam_share_default % 5 != 0:
                spam_share_default = int(5 * round(spam_share_default / 5))

            st.markdown(
                "<div class='dataset-builder__form-shell'>",
                unsafe_allow_html=True,
            )

            st.markdown("##### Base recipe")
            st.caption(
                "Start with the core knobs before flipping on Nerd Mode."
            )

            size_cols = st.columns([0.6, 0.4], gap="large")
            with size_cols[0]:
                st.markdown("**Dataset size**")
                dataset_size = st.radio(
                    "Dataset size",
                    options=[100, 300, 500],
                    index=[100, 300, 500].index(int(pending_config.get("n_total", 500)))
                    if int(pending_config.get("n_total", 500)) in [100, 300, 500]
                    else 2,
                    label_visibility="collapsed",
                    key="dataset_builder_size",
                )
            with size_cols[1]:
                st.caption(
                    "Preview runs synthesize this many emails. Smaller sets generate faster; 500 rows show the full mix."
                )

            spam_cols = st.columns([0.6, 0.4], gap="large")
            with spam_cols[0]:
                st.markdown("**Spam share**")
                spam_share_pct = st.slider(
                    "Spam share",
                    min_value=20,
                    max_value=80,
                    value=spam_share_default,
                    step=5,
                    label_visibility="collapsed",
                    key="dataset_builder_spam_share",
                )
            with spam_cols[1]:
                st.caption(
                    "Shift the class balance to see how recall and precision trade off."
                )

            edge_cols = st.columns([0.6, 0.4], gap="large")
            with edge_cols[0]:
                st.markdown("**Edge cases**")
                edge_cases = st.slider(
                    "Edge cases",
                    min_value=0,
                    max_value=10,
                    value=int(pending_config.get("edge_cases", 2)),
                    label_visibility="collapsed",
                    key="dataset_builder_edge_cases",
                )
            with edge_cols[1]:
                st.caption(
                    "Add tricky look-alikes to stress-test the preview set before you commit changes."
                )

            pending_config["n_total"] = int(dataset_size)
            pending_config["spam_ratio"] = float(spam_share_pct) / 100.0
            pending_config["edge_cases"] = int(edge_cases)

            _set_advanced_knob_state(pending_config)

            if nerd_mode_data_enabled:
                st.markdown(
                    "<div class='dataset-builder__divider'><span>nerd mode</span></div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div class='dataset-builder__nerd-callout'>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "Fine-tune suspicious links, domains, tone, attachments, randomness, and demos before generating a preview.",
                )
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("##### Nerd Mode controls")
                st.caption(
                    "Dial in adversarial signals and reproducibility extras when you need finer control."
                )
                attachment_keys = list(ATTACHMENT_MIX_PRESETS.keys())
                adv_col_a, adv_col_b = st.columns(2, gap="large")
                with adv_col_a:
                    st.markdown("**Suspicious links per spam email**")
                    adv_links = st.slider(
                        "Suspicious links per spam email",
                        min_value=0,
                        max_value=2,
                        value=int(st.session_state.get("adv_links_level", 1)),
                        key="adv_links_level",
                        label_visibility="collapsed",
                    )
                    st.caption(
                        "Raise this to stress phishing-heavy traffic (0–2 URLs per spam email)."
                    )

                    st.markdown("**Suspicious TLD frequency**")
                    adv_tld = st.select_slider(
                        "Suspicious TLD frequency",
                        options=["low", "med", "high"],
                        value=str(
                            st.session_state.get(
                                "adv_tld_level", pending_config.get("susp_tld_level", "med")
                            )
                        ),
                        key="adv_tld_level",
                        label_visibility="collapsed",
                    )
                    st.caption(
                        "Lean on sketchy domains to mimic brand impersonation attempts."
                    )

                    st.markdown("**ALL-CAPS / urgency intensity**")
                    adv_caps = st.select_slider(
                        "ALL-CAPS / urgency intensity",
                        options=["low", "med", "high"],
                        value=str(
                            st.session_state.get(
                                "adv_caps_level", pending_config.get("caps_intensity", "med")
                            )
                        ),
                        key="adv_caps_level",
                        label_visibility="collapsed",
                    )
                    st.caption(
                        "Increase to test how the model handles shouting scams and urgent tone."
                    )
                with adv_col_b:
                    st.markdown("**Money symbols & urgency**")
                    adv_money = st.select_slider(
                        "Money symbols & urgency",
                        options=["off", "low", "high"],
                        value=str(
                            st.session_state.get(
                                "adv_money_level", pending_config.get("money_urgency", "low")
                            )
                        ),
                        key="adv_money_level",
                        label_visibility="collapsed",
                    )
                    st.caption(
                        "Emphasize payment hooks and jackpot bait when you want finance-themed spam."
                    )

                    st.markdown("**Attachment lure mix**")
                    attachment_choice = st.selectbox(
                        "Attachment lure mix",
                        options=attachment_keys,
                        index=attachment_keys.index(
                            st.session_state.get("adv_attachment_choice", "Balanced")
                        )
                        if st.session_state.get("adv_attachment_choice", "Balanced") in attachment_keys
                        else 1,
                        key="adv_attachment_choice",
                        label_visibility="collapsed",
                    )
                    st.caption(
                        "Choose how often risky attachments (HTML/ZIP/XLSM/EXE) appear vs. safer PDFs."
                    )

                    st.markdown("**Label noise (%)**")
                    adv_noise = st.slider(
                        "Label noise (%)",
                        min_value=0.0,
                        max_value=5.0,
                        step=1.0,
                        value=float(
                            st.session_state.get(
                                "adv_label_noise_pct",
                                float(pending_config.get("label_noise_pct", 0.0)),
                            )
                        ),
                        key="adv_label_noise_pct",
                        label_visibility="collapsed",
                    )
                    st.caption(
                        "Inject a tiny fraction of flipped labels to see how robust the preview set is."
                    )

                    st.markdown("**Random seed**")
                    adv_seed = st.number_input(
                        "Random seed",
                        min_value=0,
                        value=int(
                            st.session_state.get(
                                "adv_seed", int(pending_config.get("seed", 42))
                            )
                        ),
                        key="adv_seed",
                        label_visibility="collapsed",
                    )
                    st.caption("Keep this fixed for reproducibility.")

                    adv_poison = st.toggle(
                        "Data poisoning demo (synthetic)",
                        value=bool(
                            st.session_state.get(
                                "adv_poison_demo", bool(pending_config.get("poison_demo", False))
                            )
                        ),
                        key="adv_poison_demo",
                    )
                    st.caption(
                        "Adds a tiny malicious distribution shift labeled as safe to show metric degradation."
                    )

            st.markdown("<div class='cta-sticky'>", unsafe_allow_html=True)
            btn_primary, btn_secondary = st.columns([1, 1])
            with btn_primary:
                preview_clicked = st.button(
                    "Generate preview", type="primary", use_container_width=True
                )
            with btn_secondary:
                reset_clicked = st.button(
                    "Reset to baseline", type="secondary", use_container_width=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            ss["dataset_builder_live_config"] = pending_config.copy()

            spam_ratio = pending_config.get("spam_ratio", float(spam_share_pct) / 100.0)

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
                ss["dataset_builder_live_config"] = ss.get(
                    "dataset_config", DEFAULT_DATASET_CONFIG
                ).copy()

            if preview_clicked:
                attachment_choice = st.session_state.get(
                    "adv_attachment_choice",
                    next(
                        (
                            name
                            for name, mix in ATTACHMENT_MIX_PRESETS.items()
                            if mix == pending_config.get("attachments_mix", DEFAULT_ATTACHMENT_MIX)
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
                        int(str(pending_config.get("susp_link_level", "1"))),
                    )
                )
                tld_level_value = str(
                    st.session_state.get(
                        "adv_tld_level", pending_config.get("susp_tld_level", "med")
                    )
                )
                caps_level_value = str(
                    st.session_state.get(
                        "adv_caps_level", pending_config.get("caps_intensity", "med")
                    )
                )
                money_level_value = str(
                    st.session_state.get(
                        "adv_money_level", pending_config.get("money_urgency", "low")
                    )
                )
                noise_pct_value = float(
                    st.session_state.get(
                        "adv_label_noise_pct",
                        float(pending_config.get("label_noise_pct", 0.0)),
                    )
                )
                seed_value = int(
                    st.session_state.get(
                        "adv_seed", int(pending_config.get("seed", 42))
                    )
                )
                poison_demo_value = bool(
                    st.session_state.get(
                        "adv_poison_demo", bool(pending_config.get("poison_demo", False))
                    )
                )
                config: DatasetConfig = {
                    "seed": int(seed_value),
                    "n_total": int(pending_config.get("n_total", dataset_total_display)),
                    "spam_ratio": float(spam_ratio),
                    "susp_link_level": str(int(links_level_value)),
                    "susp_tld_level": tld_level_value,
                    "caps_intensity": caps_level_value,
                    "money_urgency": money_level_value,
                    "attachments_mix": attachment_mix,
                    "edge_cases": int(pending_config.get("edge_cases", edge_cases)),
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
                ss["dataset_builder_live_config"] = config.copy()

            st.markdown("</div>", unsafe_allow_html=True)

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
