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
    # -------- SPAM (14) --------
    {
        "title": "URGENT: Verify your payroll information today",
        "body": "Your salary deposit is on hold. Confirm your bank details via this external link to avoid delay.",
        "label": "spam",
    },
    {
        "title": "WIN a FREE iPhone — final round eligibility",
        "body": "Congratulations! Complete the short survey to claim your iPhone. Offer expires in 2 hours.",
        "label": "spam",
    },
    {
        "title": "Password will expire — action required",
        "body": "Reset your password here: http://accounts-security.example-reset.com. Failure to act may lock your account.",
        "label": "spam",
    },
    {
        "title": "Delivery notice: package waiting for customs clearance",
        "body": "Pay a small fee to release your parcel. Use our quick checkout link to avoid return.",
        "label": "spam",
    },
    {
        "title": "Your account was suspended",
        "body": "Unusual login detected. Open the attached HTML file and confirm your credentials to restore access.",
        "label": "spam",
    },
    {
        "title": "Invoice discrepancy for March",
        "body": "We overcharged your account. Provide your card number for an immediate refund.",
        "label": "spam",
    },
    {
        "title": "Corporate survey: guaranteed €100 voucher",
        "body": "Finish this 1-minute survey and receive a voucher instantly. No employee ID needed.",
        "label": "spam",
    },
    {
        "title": "Security alert from IT desk",
        "body": "We’ve updated our security policy. Download the attached ZIP to review the changes.",
        "label": "spam",
    },
    {
        "title": "Limited-time premium subscription at 90% off",
        "body": "Upgrade now to unlock executive insights. Click the promotional link and pay today.",
        "label": "spam",
    },
    {
        "title": "Payment overdue: settle immediately",
        "body": "Your service will be interrupted. Wire funds to the account in the attachment to avoid penalties.",
        "label": "spam",
    },
    {
        "title": "HR update: bonus eligibility check",
        "body": "Confirm your identity by entering your national ID on our verification page to receive your bonus.",
        "label": "spam",
    },
    {
        "title": "Conference invite: free registration + gift",
        "body": "Register via the link to receive a €50 gift card. Limited seats; confirm with your credit card.",
        "label": "spam",
    },
    {
        "title": "DocuSign: You received a secure document",
        "body": "Open this third-party portal to review and log in with your email password for access.",
        "label": "spam",
    },
    {
        "title": "Crypto opportunity: double your balance",
        "body": "Transfer your funds to our wallet and we’ll return 2× within 24 hours. Trusted by leaders.",
        "label": "spam",
    },

    # -------- SAFE (14) --------
    {
        "title": "Password reset confirmation for corporate portal",
        "body": "You requested a password change via the **internal** IT portal. If this wasn’t you, contact IT on Teams.",
        "label": "safe",
    },
    {
        "title": "DHL tracking: package out for delivery",
        "body": "Your parcel is scheduled today. Track via the official DHL portal with your tracking ID (no payment required).",
        "label": "safe",
    },
    {
        "title": "March invoice attached — Accounts Payable",
        "body": "Hi, please find the March invoice attached in PDF. No payment details requested; PO referenced below.",
        "label": "safe",
    },
    {
        "title": "Minutes from the compliance workshop",
        "body": "Attached are the minutes and action items agreed during today’s workshop. Feedback welcome by Friday.",
        "label": "safe",
    },
    {
        "title": "Security advisory — internal policy update",
        "body": "Please review the new password guidelines on the **intranet**. No external links or attachments.",
        "label": "safe",
    },
    {
        "title": "Reminder: Q4 budget review on Tuesday",
        "body": "Please upload your cost worksheets to the internal SharePoint before 16:00 CEST.",
        "label": "safe",
    },
    {
        "title": "Travel itinerary update",
        "body": "Your train platform changed. Updated PDF itinerary is attached; no action required.",
        "label": "safe",
    },
    {
        "title": "Team meeting moved to 14:00",
        "body": "Join via the regular Teams link. Agenda: quarterly KPIs, risk register, and roadmap.",
        "label": "safe",
    },
    {
        "title": "Draft for review — policy document",
        "body": "Could you review the attached draft and add comments in Word track changes by EOD Thursday?",
        "label": "safe",
    },
    {
        "title": "Onboarding checklist for new starters",
        "body": "HR checklist attached; forms to be submitted via Workday. Reach out if anything is unclear.",
        "label": "safe",
    },
    {
        "title": "Canteen menu and wellness events",
        "body": "This week’s healthy menu and yoga session schedule are included. No RSVP needed.",
        "label": "safe",
    },
    {
        "title": "Customer feedback summary — Q3",
        "body": "Please see the attached slide deck with survey trends and next steps for CX improvements.",
        "label": "safe",
    },
    {
        "title": "Security alert — new device sign-in (confirmed)",
        "body": "You signed in from a new laptop. If recognized, no action needed. Audit log available on the intranet.",
        "label": "safe",
    },
    {
        "title": "Coffee catch-up?",
        "body": "Are you free on Thursday afternoon for a quick chat about the training roadmap?",
        "label": "safe",
    },
]

STARTER_INCOMING: List[Dict] = [
    # Edge-case pairs similar to labeled set
    {
        "title": "Password will expire — update required",
        "body": "To keep access, update your password using our internal portal link on the intranet page.",
    },
    {
        "title": "Password will expire — action needed",
        "body": "Update here: http://it-support-reset.example-login.com to prevent account lock.",
    },
    {
        "title": "Delivery notice: confirm your address",
        "body": "We couldn’t deliver your parcel. Click the link to pay a small redelivery fee.",
    },
    {
        "title": "DHL update: parcel delayed due to weather",
        "body": "No action required. Track using your official tracking number on dhl.com.",
    },
    {
        "title": "Invoice correction required",
        "body": "We can refund you today if you send your card number and CVV for verification.",
    },
    {
        "title": "AP: invoice posted to SharePoint",
        "body": "Your invoice has been posted to the finance SharePoint. PO and cost center included.",
    },

    # Varied business communications
    {
        "title": "Security bulletin — phishing simulation next week",
        "body": "IT will run a phishing simulation. Do not click unknown links; report suspicious emails via the button.",
    },
    {
        "title": "Corporate survey — help improve the office",
        "body": "Share your thoughts in a 3-minute internal survey on the intranet (no incentives offered).",
    },
    {
        "title": "Join our external webinar: limited seats",
        "body": "Reserve your seat using the registration link. A small deposit is required to confirm attendance.",
    },
    {
        "title": "Quarterly planning session",
        "body": "Please add your slides to the shared folder before the meeting. No external links.",
    },

    # Marketing/promotional vs genuine notices
    {
        "title": "Premium research access at 85% off — today only",
        "body": "Unlock exclusive reports now. Secure payment through our partner portal.",
    },
    {
        "title": "Policy update: remote work guidelines",
        "body": "The updated policy is available on the intranet. Please acknowledge by Friday.",
    },

    # “Looks real” security/invoice tones
    {
        "title": "Security alert — verify identity",
        "body": "Download the attached file and log in to validate your account. Immediate action required.",
    },
    {
        "title": "Security alert — new device sign-in",
        "body": "Was this you? If recognized, ignore. Otherwise, reset password from the internal portal.",
    },
    {
        "title": "Overdue payment — settle now",
        "body": "Service interruption imminent. Transfer funds to the following wallet to avoid fees.",
    },
    {
        "title": "AP reminder — PO mismatch on ticket #4923",
        "body": "Please correct the PO reference in the invoice metadata on SharePoint; no payment info needed.",
    },

    # Neutral miscellany
    {
        "title": "Team offsite: dietary requirements",
        "body": "Please submit your preferences by Wednesday; vegetarian and vegan options available.",
    },
    {
        "title": "Mentorship program enrollment",
        "body": "Sign up via the HR portal. Matching will occur next month; no external forms.",
    },
]

def guidance_popover(title: str, text: str):
    with st.popover(f"❓ {title}"):
        st.write(text)

def df_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"True: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels])


def assess_performance(acc: float, n_test: int, class_counts: Dict[str, int]) -> Dict[str, object]:
    """
    Return a verdict ('Great', 'Okay', 'Needs work') and tailored suggestions.
    Heuristics:
      - acc >= 0.90 and n_test >= 10 -> Great
      - 0.75 <= acc < 0.90 or n_test < 10 -> Okay
      - acc < 0.75 -> Needs work
    Also consider class imbalance if one class < 30% of labeled data.
    """
    verdict = "Okay"
    if n_test >= 10 and acc >= 0.90:
        verdict = "Great"
    elif acc < 0.75:
        verdict = "Needs work"

    tips: List[str] = []
    if verdict != "Great":
        tips.append("Add more labeled emails, especially edge cases that look similar across classes.")
        tips.append("Balance the dataset (roughly comparable counts of 'spam' and 'safe').")
        tips.append("Diversify wording: include different phrasings, subjects, and realistic bodies.")
    tips.append("Tune the spam threshold in the Classify tab to trade off false positives vs false negatives.")
    tips.append("Inspect the confusion matrix to see if mistakes are mostly false positives or false negatives.")
    tips.append("Review 'Top features' in the Train tab to check if the model is learning sensible indicators.")
    tips.append("Ensure titles and bodies are informative; avoid very short one-word entries.")

    total_labeled = sum(class_counts.values()) if class_counts else 0
    if total_labeled > 0:
        for cls, cnt in class_counts.items():
            share = cnt / total_labeled
            if share < 0.30:
                tips.insert(0, f"Label more '{cls}' examples (currently ~{share:.0%}), the model may be biased.")
                break

    return {"verdict": verdict, "tips": tips}

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
                        st.rerun()
                    if col_btn2.button("Mark as safe", key=f"mark_safe_{i}"):
                        ss["labeled"].append({"title": item["title"], "body": item["body"], "label": "safe"})
                        ss["incoming"].pop(i)
                        st.success("Labeled as safe and moved to dataset.")
                        st.rerun()

with tab_train:
    st.subheader("2) Train — make the model learn")
    guidance_popover("How training works", """
You set the **objective** (spam vs safe). The algorithm adjusts its **parameters** to reduce mistakes on your labeled examples.  
We use **TF‑IDF** features and **Logistic Regression** so we can show calibrated confidence (P(spam)).
""")
    test_size = st.slider("Hold‑out test fraction", 0.1, 0.5, 0.3, 0.05)
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
    st.info(
        "• **Hold-out test fraction**: we keep this percentage of your labeled emails aside as a mini 'exam'. "
        "The model never sees them during training, so the test score reflects how well it might handle new emails.\n"
        "• **Random seed**: this fixes the 'shuffle order' so you (and others) can get the same split and the same results when re-running the demo."
    )
    guidance_popover("Hold-out & Random seed", """
**Hold-out test fraction**  
We split your labeled emails into a training set and a test set. The test set is like a mini exam: the model hasn’t seen those emails before, so its score is a better proxy for real-world performance.

**Random seed**  
Controls the randomness of the split. By fixing the seed, you can reproduce the same results later (useful for learning and comparison).
""")
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

        try:
            df_all = pd.DataFrame(ss["labeled"])
            class_counts = df_all["label"].value_counts().to_dict() if not df_all.empty else {}
        except Exception:
            class_counts = {}

        assessment = assess_performance(acc, n_test=len(y_test), class_counts=class_counts)

        if assessment["verdict"] == "Great":
            st.success(
                f"Verdict: **{assessment['verdict']}** — This test accuracy ({acc:.2%}) looks strong for a small demo dataset."
            )
        elif assessment["verdict"] == "Okay":
            st.info(f"Verdict: **{assessment['verdict']}** — Decent, but there’s room to improve.")
        else:
            st.warning(
                f"Verdict: **{assessment['verdict']}** — The model likely needs more/better data or tuning."
            )

        with st.expander("How to improve"):
            st.markdown("\n".join([f"- {tip}" for tip in assessment["tips"]]))
            st.caption(
                "Tip: In **Full autonomy**, false positives hide legit mail and false negatives let spam through — tune the threshold accordingly."
            )

        st.caption("Confusion matrix (rows: ground truth, columns: model prediction).")
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
