import io
import json
import base64
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.special import expit  # future-proofing

st.set_page_config(page_title="demistifAI — Spam Detector", page_icon="📧", layout="wide")

CLASSES = ["spam", "safe"]
AUTONOMY_LEVELS = [
    "Low autonomy (predict + confidence)",
    "Moderate autonomy (recommendation)",
    "Full autonomy (auto-route)",
]

STARTER_LABELED = [
    ("WIN a FREE iPhone!!! Click now", "spam"),
    ("You have been selected for a prize", "spam"),
    ("Earn $$$ fast!!! Limited time", "spam"),
    ("Exclusive offer for you!!!", "spam"),
    ("Subject: Team meeting moved to 14:00", "safe"),
    ("Invoice for August attached", "safe"),
    ("Could you review the draft by Friday?", "safe"),
    ("Lunch tomorrow?", "safe"),
]

STARTER_INCOMING = [
    "Subject: Your parcel is waiting — provide details",
    "🔥 Act now for a 90% discount!!!",
    "Minutes from the compliance workshop",
    "Your password will expire soon, update here",
    "Reminder: Q4 budget review on Tuesday",
    "CONGRATULATIONS! You’ve won a gift card",
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

def predict_with_prob(model, text: str):
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
ss.setdefault("labeled", STARTER_LABELED.copy())
ss.setdefault("incoming", STARTER_INCOMING.copy())
ss.setdefault("model", None)
ss.setdefault("split_cache", None)
ss.setdefault("mail_inbox", [])
ss.setdefault("mail_spam", [])
ss.setdefault("metrics", {"TP": 0, "FP": 0, "TN": 0, "FN": 0})

st.sidebar.header("⚙️ Settings")
ss["autonomy"] = st.sidebar.selectbox("Autonomy level", AUTONOMY_LEVELS, index=AUTONOMY_LEVELS.index(ss["autonomy"]))
guidance_popover("Varying autonomy", """
**Low**: system only *predicts* with a confidence score.  
**Moderate**: system *recommends* routing (Spam vs Inbox).  
**Full**: system *acts* — it routes the email automatically based on the threshold.
""")
ss["threshold"] = st.sidebar.slider("Spam threshold (P(spam))", 0.1, 0.9, ss["threshold"], 0.05)
st.sidebar.checkbox("Adaptive learning (learn from corrections)", value=ss["adaptive"], key="adaptive")
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
st.caption("Two classes: **spam** and **safe**. Preloaded dataset is included; you can expand it.")

tab_data, tab_train, tab_eval, tab_classify, tab_card = st.tabs(
    ["1) Data", "2) Train", "3) Evaluate", "4) Classify", "5) Model Card"]
)

with tab_data:
    st.subheader("1) Data — curate and expand")
    guidance_popover("Inference inputs (training)", """
During **training**, inputs are example emails paired with the **objective** (label: spam/safe).  
The model **infers** patterns (words/phrases) that correlate with your labels — including **implicit objectives** such as click‑bait terms.
""")
    df_lab = pd.DataFrame(ss["labeled"], columns=["text", "label"])
    st.write("**Current labeled dataset**")
    st.dataframe(df_lab, use_container_width=True, hide_index=True)
    st.caption(f"Size: {len(df_lab)} | Classes present: {sorted(df_lab['label'].unique().tolist())}")

    with st.expander("➕ Add more labeled examples"):
        col1, col2 = st.columns([3,1])
        with col1:
            new_text = st.text_area("Email text", key="new_text", height=100, placeholder="Subject: ...\nBody: ...")
        with col2:
            new_label = st.radio("Label", CLASSES, index=1, key="new_label")
            st.write("")
            if st.button("Add example"):
                if new_text.strip():
                    ss["labeled"].append((new_text.strip(), new_label))
                    st.success("Added.")
                else:
                    st.warning("Please provide some text.")

    st.markdown("---")
    st.write("**Incoming emails to classify** (unlabeled stream)")
    if ss["incoming"]:
        st.dataframe(pd.DataFrame(ss["incoming"], columns=["email"]), use_container_width=True, hide_index=True)
    else:
        st.caption("Inbox stream is empty.")

with tab_train:
    st.subheader("2) Train — make the model learn")
    guidance_popover("How training works", """
You set the **objective** (spam vs safe). The algorithm adjusts its **parameters** to reduce mistakes on your labeled examples.  
We use **TF‑IDF** features and **Logistic Regression** so we can show calibrated confidence (P(spam)).
""")
    test_size = st.slider("Hold‑out test fraction", 0.1, 0.5, 0.3, 0.05)
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
    if st.button("🚀 Train model", type="primary"):
        if len(ss["labeled"]) < 4:
            st.warning("Please provide at least 4 labeled examples (with both classes).")
        else:
            df = pd.DataFrame(ss["labeled"], columns=["text", "label"])
            if len(df["label"].unique()) < 2:
                st.warning("You need both classes (spam and safe) present to train.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    df["text"].tolist(), df["label"].tolist(),
                    test_size=test_size, random_state=random_state, stratify=df["label"]
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
        st.info("Train a model first in the **Train** tab.")

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
                st.text_area("Email to classify", value=current, height=140, key="current_incoming")
            else:
                st.warning("No incoming emails left. Add more in the Data tab or use Custom input.")
                current = ""
        else:
            current = st.text_area("Paste an email (subject + body)", key="custom_email", height=140, placeholder="Subject: ...\nBody: ...")

        col_pred, col_thresh = st.columns([2,1])
        with col_thresh:
            thr = st.slider("Threshold (P(spam))", 0.1, 0.9, ss["threshold"], 0.05, help="Used for recommendation/auto‑route")
        with col_pred:
            if st.button("Predict / Route", type="primary"):
                if not current.strip():
                    st.warning("Please provide an email to classify.")
                else:
                    y_hat, pspam = predict_with_prob(ss["model"], current)
                    st.success(f"Prediction: **{y_hat}** — P(spam) ≈ **{pspam:.2f}**")
                    action, routed = route_decision(ss["autonomy"], y_hat, pspam, thr)
                    st.info(f"Autonomy action: {action}")
                    if routed is not None:
                        if routed == "Spam":
                            ss["mail_spam"].append({"email": current, "pred": y_hat, "p_spam": round(pspam,3)})
                        else:
                            ss["mail_inbox"].append({"email": current, "pred": y_hat, "p_spam": round(pspam,3)})
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
        gt = st.selectbox("What is the **actual** label of the last classified email?", ["", "spam", "safe"], index=0)
        if st.button("Record ground truth"):
            last = None
            if ss["mail_spam"]:
                last = ("spam", ss["mail_spam"][-1]["pred"])
            if ss["mail_inbox"]:
                last = ("safe", ss["mail_inbox"][-1]["pred"])
            if last is None:
                st.warning("Classify an email first.")
            else:
                true_label, pred_label = gt, last[1]
                if true_label == "":
                    st.warning("Choose a ground-truth label.")
                else:
                    if true_label == "spam" and pred_label == "spam": ss["metrics"]["TP"] += 1
                    elif true_label == "spam" and pred_label == "safe": ss["metrics"]["FN"] += 1
                    elif true_label == "safe" and pred_label == "spam": ss["metrics"]["FP"] += 1
                    elif true_label == "safe" and pred_label == "safe": ss["metrics"]["TN"] += 1
                    st.success("Recorded. Running metrics updated.")
                    if ss["adaptive"]:
                        text = None
                        if ss["mail_spam"] and (len(ss["mail_spam"]) >= len(ss["mail_inbox"])):
                            text = ss["mail_spam"][-1]["email"]
                        elif ss["mail_inbox"]:
                            text = ss["mail_inbox"][-1]["email"]
                        if text:
                            ss["labeled"].append((text, true_label))
                            df = pd.DataFrame(ss["labeled"], columns=["text", "label"])
                            if len(df["label"].unique()) >= 2:
                                X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["label"], test_size=0.3, random_state=42, stratify=df["label"])
                                ss["model"] = make_pipeline().fit(X_tr, y_tr)
                                ss["split_cache"] = (X_tr, X_te, list(y_tr), list(y_te))
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
    labels_present = sorted({lbl for _, lbl in ss["labeled"]})
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
**Classes present**: {', '.join(labels_present)}

**Key metrics**: {metrics_text or 'Train a model to populate metrics.'}

**Autonomy**: {ss['autonomy']} (threshold={ss['threshold']:.2f})  
**Adaptiveness**: {'Enabled' if ss['adaptive'] else 'Disabled'} (learn from user corrections).

**Data**: user-augmented small seed set; session-only.  
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
