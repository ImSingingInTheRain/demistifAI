"""Incoming preview panel for the Use stage."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import streamlit as st


IncomingRecord = Mapping[str, Any]


def render_incoming_preview(
    *,
    ss: Any,
    render_email_inbox_table: Callable[..., None],
    predict_batch: Callable[[Any, Sequence[IncomingRecord]], tuple[Any, Any, Any]],
    use_high_autonomy: bool,
    append_audit: Callable[[str, Mapping[str, Any]], None],
) -> None:
    """Render the incoming preview list, batch processor, and mutations."""

    incoming_list = ss.setdefault("incoming", [])

    if not incoming_list:
        st.caption("No incoming emails. Add or import more in **📊 Prepare Data**, or paste a custom email below.")
        with st.expander("Add a custom email to process"):
            title_val = st.text_input("Title", key="use_custom_title", placeholder="Subject…")
            body_val = st.text_area("Body", key="use_custom_body", height=100, placeholder="Email body…")
            if st.button("Add to incoming", key="btn_add_to_incoming"):
                if title_val.strip() or body_val.strip():
                    incoming_list.append({"title": title_val.strip(), "body": body_val.strip()})
                    st.success("Added to incoming.")
                    append_audit("incoming_added", {"title": title_val[:64]})
                else:
                    st.warning("Please provide at least a title or a body.")
        return

    preview_n = min(10, len(incoming_list))
    preview_df = pd.DataFrame(incoming_list[:preview_n])
    if not preview_df.empty:
        subtitle = f"Showing the first {preview_n} incoming emails (unlabeled)."
        render_email_inbox_table(
            preview_df,
            title="Incoming emails",
            subtitle=subtitle,
            columns=["title", "body"],
        )
    else:
        render_email_inbox_table(
            pd.DataFrame(),
            title="Incoming emails",
            subtitle="No incoming emails available.",
        )

    if not st.button(f"Process {preview_n} email(s)", type="primary", key="btn_process_batch"):
        return

    batch = incoming_list[:preview_n]
    y_hat, p_spam, p_safe = predict_batch(ss["model"], batch)
    threshold = float(ss.get("threshold", 0.5))

    batch_rows: list[dict[str, Any]] = []
    moved_spam = moved_inbox = 0

    for idx, item in enumerate(batch):
        pred = y_hat[idx]
        prob_spam = float(p_spam[idx])
        prob_safe = float(p_safe[idx]) if hasattr(p_safe, "__len__") else float(1.0 - prob_spam)
        action = "Recommend: Spam" if prob_spam >= threshold else "Recommend: Inbox"
        routed_to = None

        if use_high_autonomy:
            routed_to = "Spam" if prob_spam >= threshold else "Inbox"
            mailbox_record = {
                "title": item.get("title", ""),
                "body": item.get("body", ""),
                "pred": pred,
                "p_spam": round(prob_spam, 3),
            }
            if routed_to == "Spam":
                ss.setdefault("mail_spam", []).append(mailbox_record)
                moved_spam += 1
            else:
                ss.setdefault("mail_inbox", []).append(mailbox_record)
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
    ss["incoming"] = incoming_list[preview_n:]

    if use_high_autonomy:
        st.success(
            f"Processed {preview_n} emails — decisions applied (Inbox: {moved_inbox}, Spam: {moved_spam})."
        )
        append_audit("batch_processed_auto", {"n": preview_n, "inbox": moved_inbox, "spam": moved_spam})
    else:
        st.info(f"Processed {preview_n} emails — recommendations ready.")
        append_audit("batch_processed_reco", {"n": preview_n})
