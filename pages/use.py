"""Render the Use/Classify stage UI."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
import streamlit as st

from demistifai.constants import AUTONOMY_LEVELS, STAGE_BY_KEY
from demistifai.core.audit import _append_audit
from demistifai.core.export import _export_batch_df
from demistifai.ui.components import (
    render_batch_results_panel,
    render_incoming_preview,
    render_stage_top_grid,
)
from demistifai.modeling import (
    HybridEmbedFeatsLogReg,
    cache_train_embeddings,
    combine_text,
    get_nearest_training_examples,
    numeric_feature_contributions,
    top_token_importances,
    _predict_proba_batch,
)

from demistifai.ui.components.terminal.use import (
    render_ai_act_terminal as render_use_terminal,
)

try:
    from sklearn.model_selection import train_test_split
except Exception:  # pragma: no cover - sklearn should be present but stay resilient.
    train_test_split = None  # type: ignore


def render_classify_stage(
    *,
    ss: Any,
    section_surface: Callable[..., Any],
    render_eu_ai_quote: Callable[[str, str], None],
    render_email_inbox_table: Callable[..., None],
    render_nerd_mode_toggle: Callable[..., bool],
    render_mailbox_panel: Callable[..., None],
    set_adaptive_state: Callable[..., None],
) -> None:
    """Render the Use stage, handling batch processing and adaptiveness."""

    stage = STAGE_BY_KEY["classify"]

    def _render_use_terminal(slot):
        with slot:
            render_use_terminal()

    render_stage_top_grid("classify", left_renderer=_render_use_terminal)

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
    render_incoming_preview(
        ss=ss,
        render_email_inbox_table=render_email_inbox_table,
        predict_batch=_predict_proba_batch,
        use_high_autonomy=use_high_autonomy,
        append_audit=_append_audit,
    )

    if ss.get("use_batch_results"):
        batch_df = pd.DataFrame(ss["use_batch_results"])
        render_batch_results_panel(
            batch_df=batch_df,
            batch_rows=ss["use_batch_results"],
            ss=ss,
            section_surface=section_surface,
            render_email_inbox_table=render_email_inbox_table,
            render_nerd_mode_toggle=render_nerd_mode_toggle,
            export_batch_df=_export_batch_df,
            combine_text=combine_text,
            cache_train_embeddings=cache_train_embeddings,
            get_nearest_training_examples=get_nearest_training_examples,
            numeric_feature_contributions=numeric_feature_contributions,
            top_token_importances=top_token_importances,
        )
    st.markdown("### Adaptiveness — learn from your corrections")
    render_eu_ai_quote(
        "The EU AI Act says “AI systems may exhibit adaptiveness.” Enable adaptiveness to confirm or correct results; the model can retrain on your feedback."
    )

    def _handle_stage_adaptive_change() -> None:
        set_adaptive_state(ss.get("adaptive_stage", ss.get("adaptive", False)), source="stage")

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
                    ss["labeled"].append({"title": row.get("title", ""), "body": row.get("body", ""), "label": "spam"})
                    _append_audit("correct_label", {"i": i, "new": "spam"})
                    st.toast("Recorded correction → Spam.", icon="✍️")
                if col_c.button("Correct → Safe", key=f"use_correct_safe_{i}"):
                    ss["labeled"].append({"title": row.get("title", ""), "body": row.get("body", ""), "label": "safe"})
                    _append_audit("correct_label", {"i": i, "new": "safe"})
                    st.toast("Recorded correction → Safe.", icon="✍️")

        if st.button("🔁 Retrain now with feedback", key="btn_retrain_feedback"):
            df_all = pd.DataFrame(ss["labeled"])
            if not df_all.empty and len(df_all["label"].unique()) >= 2 and train_test_split is not None:
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
                    numeric_assist_center=float(gp.get("assist_center", float(ss.get("threshold", 0.6)))),
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
    inbox_tab, spam_tab = st.tabs([
        f"Inbox (safe) — {len(ss['mail_inbox'])}",
        f"Spam — {len(ss['mail_spam'])}",
    ])
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

