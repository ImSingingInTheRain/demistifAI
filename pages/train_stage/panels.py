from __future__ import annotations

from typing import Any, Dict, MutableMapping

import streamlit as st

from demistifai.constants import StageMeta
from demistifai.ui.components.train_intro import (
    build_launchpad_card,
    build_launchpad_status_item,
    build_nerd_intro_card,
    build_train_intro_card,
)

from .guardrails import render_guardrail_controls
from .callbacks import go_to_prepare, label_balance_status, pii_status


def render_launchpad_panel(
    ss: MutableMapping[str, Any],
    *,
    stage: StageMeta,
    stage_number: int,
    render_eu_ai_quote,
    section_surface,
    render_nerd_mode_toggle,
    guard_params: Dict[str, Any],
) -> bool:
    """Render the introductory launchpad with dataset checks and guardrail toggles."""

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

            updated_guard_params = render_guardrail_controls(
                ss,
                guard_params=guard_params,
                nerd_mode_train_active=nerd_mode_train_active,
            )

            nerd_mode_train_enabled = render_nerd_mode_toggle(
                key="nerd_mode_train",
                title="Nerd Mode — advanced controls",
                description="Tweak the train/test split, solver iterations, and regularization strength.",
                icon="🔬",
            )

    ss["guard_params"] = updated_guard_params
    return bool(nerd_mode_train_enabled)


def render_nerd_mode_panels(ss: MutableMapping[str, Any], *, section_surface) -> None:
    """Render the advanced Nerd Mode configuration panels."""

    with section_surface():
        st.markdown(build_nerd_intro_card(), unsafe_allow_html=True)
        ss.setdefault("guard_params", {})
        guard_params = ss["guard_params"]
        assist_mode = guard_params.get("assist_center_mode", "auto")

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
                guard_params["assist_center"] = st.slider(
                    "🛡️ Numeric assist center (text score)",
                    min_value=0.30,
                    max_value=0.90,
                    step=0.01,
                    value=float(guard_params.get("assist_center", float(ss.get("threshold", 0.6)))),
                    help=(
                        "Center of the borderline region. When the text-only spam probability is near this "
                        "value, numeric guardrails are allowed to lend a hand."
                    ),
                )
                st.caption(
                    "🛡️ Where ‘borderline’ lives on the 0–1 scale; most emails away from here won’t use numeric cues."
                )
                guard_params["uncertainty_band"] = st.slider(
                    "🛡️ Uncertainty band (± around threshold)",
                    min_value=0.0,
                    max_value=0.20,
                    step=0.01,
                    value=float(guard_params.get("uncertainty_band", 0.08)),
                    help="Only consult numeric cues when the text score falls inside this band.",
                )
                st.caption("🛡️ Wider band = numeric cues help more often; narrower = trust text more.")
                guard_params["numeric_scale"] = st.slider(
                    "🛡️ Numeric blend weight (when consulted)",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    value=float(guard_params.get("numeric_scale", 0.5)),
                    help="How much numeric probability counts in the blend within the band.",
                )
                st.caption("🛡️ Higher = numeric cues have a stronger say when consulted.")
            else:
                st.caption("Controlled by Implicit strategy mode")
                assist_center = float(guard_params.get("assist_center", float(ss.get("threshold", 0.6))))
                uncertainty_band = float(guard_params.get("uncertainty_band", 0.08))
                numeric_scale = float(guard_params.get("numeric_scale", 0.5))
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
                guard_params["numeric_logit_cap"] = st.slider(
                    "🛡️ Cap numeric logit (absolute)",
                    min_value=0.2,
                    max_value=3.0,
                    step=0.1,
                    value=float(guard_params.get("numeric_logit_cap", 1.0)),
                    help="Limits how strongly numeric cues can push toward Spam/Safe.",
                )
                st.caption("🛡️ A safety cap so numeric cues can’t overpower the text score.")
                guard_params["combine_strategy"] = st.radio(
                    "🛡️ Numeric combination strategy",
                    options=["blend", "threshold_shift"],
                    index=0 if guard_params.get("combine_strategy", "blend") == "blend" else 1,
                    horizontal=True,
                    help="Blend = mix text & numeric probs; Threshold shift = keep text prob, adjust effective threshold slightly.",
                )
            else:
                numeric_logit_cap = float(guard_params.get("numeric_logit_cap", 1.0))
                combine_strategy = str(guard_params.get("combine_strategy", "blend"))
                st.markdown(
                    f"<div class='train-token-chip'>Logit cap: {numeric_logit_cap:.2f}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='train-token-chip'>Strategy: {combine_strategy.replace('_', ' ').title()}</div>",
                    unsafe_allow_html=True,
                )

        if assist_mode == "manual" and guard_params.get("combine_strategy", "blend") == "threshold_shift":
            st.markdown("**🛡️ Threshold-shift micro-rules** (applied only within the uncertainty band)")
            col1, col2, col3 = st.columns(3)
            with col1:
                guard_params["shift_suspicious_tld"] = st.number_input(
                    "🛡️ Shift for suspicious TLD",
                    value=float(guard_params.get("shift_suspicious_tld", -0.04)),
                    step=0.01,
                    format="%.2f",
                    help="Negative shift lowers the threshold (be stricter) when a suspicious domain is present.",
                )
                st.caption("🛡️ Tweaks the cut-off in specific situations (e.g., suspicious domains → stricter).")
            with col2:
                guard_params["shift_many_links"] = st.number_input(
                    "🛡️ Shift for many external links",
                    value=float(guard_params.get("shift_many_links", -0.03)),
                    step=0.01,
                    format="%.2f",
                    help="Negative shift lowers the threshold when multiple external links are detected.",
                )
                st.caption("🛡️ Tweaks the cut-off in specific situations (e.g., suspicious domains → stricter).")
            with col3:
                guard_params["shift_calm_text"] = st.number_input(
                    "🛡️ Shift for calm text",
                    value=float(guard_params.get("shift_calm_text", +0.02)),
                    step=0.01,
                    format="%.2f",
                    help="Positive shift raises the threshold when text looks calm (very low ALL-CAPS).",
                )
                st.caption("🛡️ Tweaks the cut-off in specific situations (e.g., suspicious domains → stricter).")
        elif assist_mode != "manual" and guard_params.get("combine_strategy", "blend") == "threshold_shift":
            st.markdown("**🛡️ Threshold-shift micro-rules** (read-only)")
            shifts = {
                "Suspicious TLD": float(guard_params.get("shift_suspicious_tld", -0.04)),
                "Many external links": float(guard_params.get("shift_many_links", -0.03)),
                "Calm text": float(guard_params.get("shift_calm_text", +0.02)),
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

    ss["guard_params"] = guard_params
