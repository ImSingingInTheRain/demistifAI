import base64
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="demistifAI — Spam Detector", page_icon="📧", layout="wide")

CLASSES = ["spam", "safe"]
AUTONOMY_LEVELS = [
    "Low autonomy (predict + confidence)",
    "Moderate autonomy (recommendation)",
    "Full autonomy (auto-route)",
]

STARTER_LABELED: List[Dict] = [
    {"title": "WIN a FREE iPhone!!!", "body": "Click now to claim your prize. Offer ends today.", "label": "spam"},
    {"title": "You have been selected", "body": "Congratulations! Verify your account to get the reward.", "label": "spam"},
    {"title": "Earn $$$ fast", "body": "No experience required. Work from home and make money quickly.", "label": "spam"},
    {"title": "Exclusive offer for you", "body": "Limited time 90% discount on premium subscription!!!", "label": "spam"},
    {"title": "URGENT: Account suspended", "body": "Your account will be closed. Confirm your password here.", "label": "spam"},
    {"title": "Gift card inside", "body": "Open this attachment to redeem your bonus.", "label": "spam"},
    {"title": "Team meeting moved to 14:00", "body": "Please join on the usual Teams link. Agenda attached.", "label": "safe"},
    {"title": "Invoice for August", "body": "Hi, please find the invoice attached. Let me know if any questions.", "label": "safe"},
    {"title": "Draft for review", "body": "Could you review the policy draft by Friday? Thanks!", "label": "safe"},
    {"title": "Lunch tomorrow?", "body": "Are you free at 12:30 near the canteen?", "label": "safe"},
    {"title": "Compliance workshop minutes", "body": "Attached minutes and action items from today's session.", "label": "safe"},
    {"title": "Reminder: Q4 budget review", "body": "Please upload your worksheets by Tuesday EOD.", "label": "safe"},
]

STARTER_INCOMING: List[Dict] = [
    {"title": "Your parcel is waiting", "body": "Provide your details to schedule delivery."},
    {"title": "Act now for 90% discount!!!", "body": "This exclusive link expires in 2 hours. Don’t miss out!"},
    {"title": "Minutes from the compliance workshop", "body": "See attached for decisions and next steps."},
    {"title": "Password will expire soon", "body": "Update your password using the corporate portal."},
    {"title": "Coffee catch‑up?", "body": "Fancy a coffee on Thursday afternoon?"},
    {"title": "CONGRATULATIONS! You’ve won", "body": "Claim your gift card by entering your bank details here."},
    {"title": "Travel itinerary update", "body": "Your train has changed platforms; ticket attached."},
    {"title": "Security alert", "body": "We detected a sign‑in from a new device. If this wasn’t you, reset your password."},
]

def guidance_popover(title: str, text: str):
    with st.popover(f"❓ {title}"):
        st.write(text)

def df_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"True: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels])

def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

def combine_text(title: str, body: str) -> str:
    return (title or "") + "\n" + (body or "")

def predict_with_prob(model, title: str, body: str):
    text = combine_text(title, body)
    y_hat = model.predict([text])[0]
    classes = list(model.named_steps["clf"].classes_)
    pspam = model.predict_proba([text])[0][classes.index("spam")] if "spam" in classes else None
    return y_hat, float(pspam) if pspam is not None else None

def route_decision(autonomy: str, y_hat: str, pspam: float, threshold: float):
    routed = None
    if autonomy.startswith("Low"):
        action = f"Prediction only. Confidence P(spam) ≈ {pspam:.2f}" if pspam is not None else "Prediction only."
    elif autonomy.startswith("Moderate"):
        to_spam = (pspam is not None and pspam >= threshold) or y_hat == "spam"
        action = f"Recommend: {'Spam' if to_spam else 'Inbox'} (threshold={threshold:.2f})"
    else:
        to_spam = (pspam is not None and pspam >= threshold) or y_hat == "spam"
        routed = "Spam" if to_spam else "Inbox"
        action = f"Auto-routed to **{routed}** (threshold={threshold:.2f})"
    return action, routed

def download_text(text: str, filename: str, label: str = "Download"):
    b64 = base64.b64encode(text.encode("utf-8")).decode()
    st.markdown(f'<a href="data:text/plain;base64,{b64}" download="{filename}">{label}</a>', unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("autonomy", AUTONOMY_LEVELS[0])
ss.setdefault("threshold", 0.6)
ss.setdefault("adaptive", True)
ss.setdefault("labeled", STARTER_LABELED.copy())      # list of dicts: title, body, label
ss.setdefault("incoming", STARTER_INCOMING.copy())    # list of dicts: title, body
ss.setdefault("model", None)
ss.setdefault("split_cache", None)
ss.setdefault("mail_inbox", [])  # list of dicts: title, body, pred, p_spam
ss.setdefault("mail_spam", [])
ss.setdefault("metrics", {"TP": 0, "FP": 0, "TN": 0, "FN": 0})

st.sidebar.header("⚙️ Settings")
ss["autonomy"] = st.sidebar.selectbox("Autonomy level", AUTONOMY_LEVELS, index=AUTONOMY_LEVELS.index(ss["autonomy"]))
guidance_popover("Varying autonomy", """
**Low**: system only *predicts* with a confidence score.  
**Moderate**: system *recommends* routing (Spam vs Inbox) but waits for you.  
**Full**: system *acts* — it routes the email automatically based on the threshold.
""")
ss["threshold"] = st.sidebar.slider("Spam threshold (P(spam))", 0.1, 0.9, ss["threshold"], 0.05)
st.sidebar.checkbox("Adaptive learning (learn from corrections)", value=ss["adaptive"], key="adaptive")
st.sidebar.caption("When enabled, your corrections are added to the dataset and the model retrains.")
if st.sidebar.button("🔄 Reset demo data"):
    ss["labeled"] = STARTER_LABELED.copy()
    ss["incoming"] = STARTER_INCOMING.copy()
    ss["model"] = None
    ss["split_cache"] = None
    ss["mail_inbox"].clear(); ss["mail_spam"].clear()
    ss["metrics"] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    st.sidebar.success("Reset complete.")

st.title("📧 demistifAI — Spam Detector")
col_title, col_help = st.columns([5,2])
with col_help:
    guidance_popover("Machine‑based system", """
This app is the **software** component; Streamlit Cloud is the **hardware** (cloud runtime).  
You define the intended purpose (classify emails as **spam** or **safe**).
""")
st.caption("Two classes are available: **spam** and **safe**. Preloaded labeled dataset + unlabeled inbox stream.")

tab_data, tab_train, tab_eval, tab_classify, tab_card = st.tabs(
    ["1) Data", "2) Train", "3) Evaluate", "4) Classify", "5) Model Card"]
)

with tab_data:
    st.subheader("1) Data — curate and expand")
    guidance_popover("Inference inputs (training)", """
During **training**, inputs are example emails (title + body) paired with the **objective** (label: spam/safe).  
The model **infers** patterns that correlate with your labels — including **implicit objectives** such as click‑bait terms.
""")

    st.write("### ✅ Labeled dataset")
    if ss["labeled"]:
        df_lab = pd.DataFrame(ss["labeled"])
        st.dataframe(df_lab, use_container_width=True, hide_index=True)
        st.caption(f"Size: {len(df_lab)} | Classes present: {sorted(df_lab['label'].unique().tolist())}")
    else:
        st.caption("No labeled data yet.")

    with st.expander("➕ Add a labeled example"):
        title = st.text_input("Title", key="add_l_title", placeholder="Subject: ...")
        body = st.text_area("Body", key="add_l_body", height=100, placeholder="Email body...")
        label = st.radio("Label", CLASSES, index=1, horizontal=True, key="add_l_label")
        if st.button("Add to labeled dataset", key="btn_add_labeled"):
            if not (title.strip() or body.strip()):
                st.warning("Provide at least a title or a body.")
            else:
                ss["labeled"].append({"title": title.strip(), "body": body.strip(), "label": label})
                st.success("Added to labeled dataset.")

    st.markdown("---")
    st.write("### 📥 Unlabeled inbox — label emails inline")
    guidance_popover("Hands‑on labeling", """
Click **Mark as spam** or **Mark as safe** next to each email.  
Once labeled, the email moves to the **labeled dataset** above.
""")
    if not ss["incoming"]:
        st.caption("Inbox stream is empty.")
    else:
        for i, item in enumerate(list(ss["incoming"])):
            with st.container(border=True):
                c1, c2 = st.columns([4,1])
                with c1:
                    st.markdown(f"**Title:** {item['title']}")
                    st.markdown(f"**Body:** {item['body']}")
                with c2:
                    col_btn1, col_btn2 = st.columns(2)
                    if col_btn1.button("Mark as spam", key=f"mark_spam_{i}"):
                        ss["labeled"].append({"title": item["title"], "body": item["body"], "label": "spam"})
                        ss["incoming"].pop(i)
                        st.success("Labeled as spam and moved to dataset.")
                        st.experimental_rerun()
                    if col_btn2.button("Mark as safe", key=f"mark_safe_{i}"):
                        ss["labeled"].append({"title": item["title"], "body": item["body"], "label": "safe"})
                        ss["incoming"].pop(i)
                        st.success("Labeled as safe and moved to dataset.")
                        st.experimental_rerun()

with tab_train:
    st.subheader("2) Train — make the model learn")
    guidance_popover("How training works", """
You set the **objective** (spam vs safe). The algorithm adjusts its **parameters** to reduce mistakes on your labeled examples.  
We use **TF‑IDF** features and **Logistic Regression** so we can show calibrated confidence (P(spam)).
""")
    test_size = st.slider("Hold‑out test fraction", 0.1, 0.5, 0.3, 0.05)
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
    if st.button("🚀 Train model", type="primary"):
        if len(ss["labeled"]) < 6:
            st.warning("Please label a few more emails first (≥6 examples).")
        else:
            df = pd.DataFrame(ss["labeled"])
            if len(df["label"].unique()) < 2:
                st.warning("You need both classes (spam and safe) present to train.")
            else:
                X = (df["title"].fillna("") + "\n" + df["body"].fillna("")).tolist()
                y = df["label"].tolist()
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
                model = make_pipeline().fit(X_train, y_train)
                ss["model"] = model
                ss["split_cache"] = (X_train, X_test, y_train, y_test)
                st.success("Model trained.")
                with st.expander("See learned indicators (top features)", expanded=False):
                    clf = model.named_steps["clf"]
                    tfidf = model.named_steps["tfidf"]
                    if hasattr(clf, "coef_"):
                        feature_names = np.array(tfidf.get_feature_names_out())
                        coefs = clf.coef_[0]  # spam vs safe
                        top_spam_idx = np.argsort(coefs)[-10:][::-1]
                        top_safe_idx = np.argsort(coefs)[:10]
                        st.write("**Top spam indicators**")
                        st.write(", ".join(feature_names[top_spam_idx]))
                        st.write("**Top safe indicators**")
                        st.write(", ".join(feature_names[top_safe_idx]))
                    else:
                        st.caption("Weights not available.")

with tab_eval:
    st.subheader("3) Evaluate — check generalization")
    guidance_popover("Why evaluate?", """
Hold‑out evaluation estimates how well the model generalizes to **new** emails.  
It helps detect overfitting, bias, and areas needing more data.
""")
    if ss.get("model") and ss.get("split_cache"):
        X_train, X_test, y_train, y_test = ss["split_cache"]
        y_pred = ss["model"].predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Test accuracy: **{acc:.2%}** on {len(y_test)} samples.")
        st.dataframe(df_confusion(y_test, y_pred, CLASSES), use_container_width=True)
        st.text("Classification report")
        st.code(classification_report(y_test, y_pred, labels=CLASSES), language="text")
    else:
        st.info("Label some emails and train a model in the **Train** tab.")

with tab_classify:
    st.subheader("4) Classify — generate outputs & route")
    col_help1, col_help2 = st.columns(2)
    with col_help1:
        guidance_popover("Generate outputs", """
At **inference** time, the model uses its learned parameters to output a **prediction** and a **confidence** (P(spam)).
""")
    with col_help2:
        guidance_popover("Autonomy in action", """
Depending on the autonomy level, the system may: only **predict**, **recommend**, or **auto‑route** the email to Spam/Inbox.
""")
    if not ss.get("model"):
        st.info("Train a model first in the **Train** tab.")
    else:
        src = st.radio("Classify from", ["Next incoming email", "Custom input"], horizontal=True)
        if src == "Next incoming email":
            if ss["incoming"]:
                current = ss["incoming"][0]
                st.text_input("Title", value=current["title"], key="cur_title", disabled=True)
                st.text_area("Body", value=current["body"], key="cur_body", height=120, disabled=True)
            else:
                st.warning("No incoming emails left. Add more in the Data tab or use Custom input.")
                current = {"title": "", "body": ""}
        else:
            cur_t = st.text_input("Title", key="custom_title", placeholder="Subject: ...")
            cur_b = st.text_area("Body", key="custom_body", height=120, placeholder="Email body...")
            current = {"title": cur_t, "body": cur_b}

        col_pred, col_thresh = st.columns([2,1])
        with col_thresh:
            thr = st.slider("Threshold (P(spam))", 0.1, 0.9, ss["threshold"], 0.05, help="Used for recommendation/auto‑route")
        with col_pred:
            if st.button("Predict / Route", type="primary"):
                if not (current["title"].strip() or current["body"].strip()):
                    st.warning("Please provide an email to classify.")
                else:
                    y_hat, pspam = predict_with_prob(ss["model"], current["title"], current["body"])
                    st.success(f"Prediction: **{y_hat}** — P(spam) ≈ **{pspam:.2f}**")
                    action, routed = route_decision(ss["autonomy"], y_hat, pspam, thr)
                    st.info(f"Autonomy action: {action}")
                    if routed is not None:
                        record = {"title": current["title"], "body": current["body"], "pred": y_hat, "p_spam": round(pspam,3)}
                        if routed == "Spam":
                            ss["mail_spam"].append(record)
                        else:
                            ss["mail_inbox"].append(record)
                        if src == "Next incoming email" and ss["incoming"]:
                            ss["incoming"].pop(0)

        st.markdown("### 📥 Mailboxes")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Inbox (safe)** — {len(ss['mail_inbox'])}")
            st.dataframe(pd.DataFrame(ss["mail_inbox"]), use_container_width=True, hide_index=True) if ss["mail_inbox"] else st.caption("Empty")
        with c2:
            st.write(f"**Spam** — {len(ss['mail_spam'])}")
            st.dataframe(pd.DataFrame(ss["mail_spam"]), use_container_width=True, hide_index=True) if ss["mail_spam"] else st.caption("Empty")

        st.markdown("### ✅ Record ground truth & (optionally) learn")
        gt = st.selectbox("What is the **actual** label of the last routed email?", ["", "spam", "safe"], index=0)
        if st.button("Record ground truth"):
            last_box = None
            if ss["mail_spam"] and (len(ss["mail_spam"]) >= len(ss["mail_inbox"])):
                last_box = ("spam", ss["mail_spam"][-1])
            elif ss["mail_inbox"]:
                last_box = ("safe", ss["mail_inbox"][-1])

            if last_box is None:
                st.warning("Route an email first (Full autonomy).")
            else:
                system_label, last_rec = last_box
                if gt == "":
                    st.warning("Choose a ground-truth label.")
                else:
                    if gt == "spam" and system_label == "spam": ss["metrics"]["TP"] += 1
                    elif gt == "spam" and system_label == "safe": ss["metrics"]["FN"] += 1
                    elif gt == "safe" and system_label == "spam": ss["metrics"]["FP"] += 1
                    elif gt == "safe" and system_label == "safe": ss["metrics"]["TN"] += 1
                    st.success("Recorded. Running metrics updated.")
                    if ss["adaptive"]:
                        ss["labeled"].append({"title": last_rec["title"], "body": last_rec["body"], "label": gt})
                        df = pd.DataFrame(ss["labeled"])
                        if len(df["label"].unique()) >= 2:
                            X = (df["title"].fillna("") + "\n" + df["body"].fillna("")).tolist()
                            y = df["label"].tolist()
                            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
                            ss["model"] = make_pipeline().fit(X_tr, y_tr)
                            ss["split_cache"] = (X_tr, X_te, y_tr, y_te)
                            st.info("🔁 Adaptive learning: model retrained with your correction.")

        m = ss["metrics"]; total = sum(m.values()) or 1
        acc = (m["TP"] + m["TN"]) / total
        st.write(f"**Running accuracy** (from your recorded ground truths): {acc:.2%} | TP {m['TP']} • FP {m['FP']} • TN {m['TN']} • FN {m['FN']}")

with tab_card:
    st.subheader("5) Model Card — transparency")
    guidance_popover("Transparency", """
Model cards summarize intended purpose, data, metrics, autonomy & adaptiveness settings.  
They help teams reason about risks and the appropriate oversight controls.
""")
    algo = "TF‑IDF + Logistic Regression"
    n_samples = len(ss["labeled"])
    labels_present = sorted({row["label"] for row in ss["labeled"]}) if ss["labeled"] else []
    metrics_text = ""
    if ss.get("model") and ss.get("split_cache"):
        X_train, X_test, y_train, y_test = ss["split_cache"]
        y_pred = ss["model"].predict(X_test)
        metrics_text = f"Accuracy on hold‑out: {accuracy_score(y_test, y_pred):.2%} (n={len(y_test)})"
    card_md = f"""
# Model Card — demistifAI (Spam Detector)
**Intended purpose**: Educational demo to illustrate the AI Act definition of an **AI system** via a spam classifier.

**Algorithm**: {algo}  
**Classes**: spam, safe  
**Dataset size**: {n_samples} labeled examples  
**Classes present**: {', '.join(labels_present) if labels_present else '[not trained]'}

**Key metrics**: {metrics_text or 'Train a model to populate metrics.'}

**Autonomy**: {ss['autonomy']} (threshold={ss['threshold']:.2f})  
**Adaptiveness**: {'Enabled' if ss['adaptive'] else 'Disabled'} (learn from user corrections).

**Data**: user-augmented seed set (title + body); session-only.  
**Known limitations**: tiny datasets; vocabulary sensitivity; no MIME/URL/metadata features.

**AI Act mapping**  
- **Machine-based system**: Streamlit app (software) running on cloud runtime (hardware).  
- **Inference**: model learns patterns from labeled examples.  
- **Output generation**: predictions + confidence; used to recommend/route emails.  
- **Varying autonomy**: user selects autonomy level; at full autonomy, the system acts.  
- **Adaptiveness**: optional feedback loop that updates the model.
"""
    st.markdown(card_md)
    download_text(card_md, "model_card.md", "Download model_card.md")

st.markdown("---")
st.caption("© demistifAI — Built for interactive learning and governance discussions.")
