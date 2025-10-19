"""Batch results panel for the Use/Classify stage."""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt


def render_batch_results_panel(
    *,
    batch_df: pd.DataFrame,
    batch_rows: Sequence[dict[str, Any]],
    ss: Any,
    section_surface: Callable[..., Any],
    render_email_inbox_table: Callable[..., None],
    render_nerd_mode_toggle: Callable[..., bool],
    export_batch_df: Callable[[Sequence[dict[str, Any]]], pd.DataFrame],
    combine_text: Callable[[str, str], str],
    cache_train_embeddings: Callable[[Sequence[str]], Any],
    get_nearest_training_examples: Callable[..., Sequence[dict[str, Any]]],
    numeric_feature_contributions: Callable[..., Sequence[tuple[str, float, float]]],
    top_token_importances: Callable[..., tuple[Any, Sequence[dict[str, Any]]]],
) -> None:
    """Render the batch results table, diagnostics, and exports."""

    show_cols = ["title", "pred", "p_spam", "action", "routed_to"]
    existing_cols = [col for col in show_cols if col in batch_df.columns]
    display_df = batch_df[existing_cols].rename(
        columns={"pred": "Prediction", "p_spam": "P(spam)", "action": "Action", "routed_to": "Routed"}
    )

    with section_surface():
        st.markdown("### Results")
        render_email_inbox_table(display_df, title="Batch results", subtitle="Predictions and actions just taken.")
        st.caption("Each row shows the predicted label, confidence (P(spam)), and the recommendation or action taken.")

    nerd_mode_enabled = render_nerd_mode_toggle(
        key="nerd_mode_use",
        title="Nerd Mode — details for this batch",
        description="Inspect raw probabilities, distributions, and the session audit trail.",
        icon="🔬",
    )

    if nerd_mode_enabled:
        with section_surface():
            st.markdown("### Nerd Mode — batch diagnostics")
            col_nm1, col_nm2 = st.columns([2, 1])
            with col_nm1:
                st.markdown("**Raw probabilities (per email)**")
                detail_cols = ["title", "p_spam", "p_safe", "pred", "action", "routed_to"]
                det_existing = [col for col in detail_cols if col in batch_df.columns]
                st.dataframe(batch_df[det_existing], width="stretch", hide_index=True)
            with col_nm2:
                st.markdown("**Batch metrics**")
                n_items = len(batch_df)
                mean_conf = float(batch_df["p_spam"].mean()) if "p_spam" in batch_df else 0.0
                n_spam = int((batch_df["pred"] == "spam").sum()) if "pred" in batch_df else 0
                n_safe = n_items - n_spam
                st.write(f"- Items: {n_items}")
                st.write(f"- Predicted Spam: {n_spam} | Safe: {n_safe}")
                st.write(f"- Mean P(spam): {mean_conf:.2f}")

                if "p_spam" in batch_df:
                    fig, ax = plt.subplots()
                    ax.hist(batch_df["p_spam"], bins=10)
                    ax.set_xlabel("P(spam)")
                    ax.set_ylabel("Count")
                    ax.set_title("Spam score distribution")
                    st.pyplot(fig)

        split_cache = ss.get("split_cache")
        train_texts: list[str] = []
        train_labels: list[str] = []
        train_emb: Any = None
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

        with section_surface():
            st.markdown("### Why did it decide this way? (per email)")

            for email_idx, row in enumerate(batch_rows):
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
                            f"**Decision summary:** P(spam) = {p_spam_float:.2f} vs threshold {threshold_val:.2f} → "
                            f"**{decision}** (margin {margin:+.2f})"
                        )
                        if hasattr(model_obj, "last_thr_eff"):
                            idxs_thr = getattr(model_obj, "last_thr_eff", None)
                            if idxs_thr:
                                idxs, thr_eff = idxs_thr
                                try:
                                    pos = list(idxs).index(email_idx)
                                    st.caption(f"Effective threshold (with numeric micro-rules): {thr_eff[pos]:.2f}")
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

                    if hasattr(model_obj, "scaler") and (hasattr(model_obj, "lr_num") or hasattr(model_obj, "lr")):
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
                            st.caption("Runs extra passes to see which words reduce P(spam) the most when removed.")
                            if st.checkbox("Compute highlights for this email", key=f"hl_{email_idx}", value=False):
                                base_prob, rows_imp = top_token_importances(model_obj, title, body)
                                if base_prob is None:
                                    st.caption("Unable to compute token importances for this model/email.")
                                else:
                                    st.caption(
                                        f"Base P(spam) = {base_prob:.2f}. Higher importance means removing the word lowers "
                                        "P(spam) more."
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

        exp_df = export_batch_df(batch_rows)
        csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
        json_bytes = json.dumps(batch_rows, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("⬇️ Download results (CSV)", data=csv_bytes, file_name="batch_results.csv", mime="text/csv")
        st.download_button(
            "⬇️ Download results (JSON)", data=json_bytes, file_name="batch_results.json", mime="application/json"
        )
