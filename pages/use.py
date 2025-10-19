"""Render the Use/Classify stage UI."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Optional
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from demistifai.constants import AUTONOMY_LEVELS, STAGE_BY_KEY
from demistifai.core.audit import _append_audit
from demistifai.core.export import _export_batch_df
from demistifai.ui.components import render_stage_top_grid
from demistifai.modeling import (
    HybridEmbedFeatsLogReg,
    cache_train_embeddings,
    combine_text,
    get_nearest_training_examples,
    numeric_feature_contributions,
    top_token_importances,
    _predict_proba_batch,
)

from demistifai.ui.components.terminal.use import render_use_terminal

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
                _append_audit("batch_processed_auto", {"n": preview_n, "inbox": moved_inbox, "spam": moved_spam})
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
            st.caption("Each row shows the predicted label, confidence (P(spam)), and the recommendation or action taken.")

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
                                if st.checkbox("Compute highlights for this email", key=f"hl_{email_idx}", value=False):
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

