import io
import json
import base64
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.special import expit  # sigmoid for mapping decision scores

# ---------------------------
# App configuration
# ---------------------------
st.set_page_config(
    page_title="demistifAI — Demystify ML & the EU AI Act",
    page_icon="🧩",
    layout="wide"
)

# ---------------------------
# Utilities
# ---------------------------

@dataclass
class ImageExample:
    label: str
    image_b64: str  # base64-encoded bytes

def _b64_from_bytes(b: bytes) -> str:
    return base64.b64encode(b).decode("utf-8")

def _bytes_from_b64(s: str) -> bytes:
    return base64.b64decode(s.encode("utf-8"))

def _read_image(img_bytes: bytes, max_side: int = 128) -> np.ndarray:
    """Open image, convert to RGB, resize so longest side = max_side, then make simple features."""
    with Image.open(io.BytesIO(img_bytes)) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
        new_w, new_h = int(round(w / scale)), int(round(h / scale))
        im = im.resize((new_w, new_h))
        arr = np.asarray(im).astype(np.float32) / 255.0  # [H,W,3] ∈ [0,1]
    # Features: color hist (24) + downsampled 32x32x3 (3072)
    hist_features = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=8, range=(0.0, 1.0), density=True)
        hist_features.append(hist)
    hist_features = np.concatenate(hist_features, axis=0)
    im_small = Image.fromarray((arr * 255).astype(np.uint8)).resize((32, 32))
    arr_small = np.asarray(im_small).astype(np.float32) / 255.0
    pixel_features = arr_small.flatten()
    feats = np.concatenate([hist_features, pixel_features], axis=0)  # (3096,)
    return feats

def _confusion_matrix_df(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"True: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels])

def _chat_explain(role: str, content: str):
    """Use chat-style messages as guided narration."""
    with st.chat_message(role):
        st.write(content)

def _binary_tip():
    st.info(
        "Tip: Start with **2–3 classes** and ~5–10 examples per class. "
        "Small datasets are perfect to visualize overfitting and improvements."
    )

def _download_bytes(obj_bytes: bytes, filename: str, label: str):
    b64 = base64.b64encode(obj_bytes).decode()
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def _download_text(text: str, filename: str, label: str):
    _download_bytes(text.encode("utf-8"), filename, label)

def _guard_min_examples(n_classes: int, n_samples: int) -> bool:
    return n_classes >= 2 and n_samples >= 4

# ---------------------------
# Session state
# ---------------------------
ss = st.session_state

ss.setdefault("mode", "Image classification")
ss.setdefault("step", 0)  # wizard step
ss.setdefault("adaptive_mode", False)  # continuous learning toggle

ss.setdefault("image_examples", [])  # list[ImageExample as dict]
ss.setdefault("text_examples", [])   # list[{"text":..., "label":...}]
ss.setdefault("csv_df", None)        # numeric dataset as DataFrame
ss.setdefault("csv_target", "")
ss.setdefault("task_type", "Classification")  # for CSV: Classification or Regression

ss.setdefault("image_model", None)
ss.setdefault("text_model", None)
ss.setdefault("csv_model", None)

ss.setdefault("labels_image", [])
ss.setdefault("labels_text", [])

# Spam detector simulation state
ss.setdefault("spam_mode", False)
ss.setdefault("routing_mode", "Manual review")  # Manual review, Suggest-only, Auto-route
ss.setdefault("threshold", 0.5)
ss.setdefault("mail_inbox", [])  # list of dicts with {"text","pred","prob"}
ss.setdefault("mail_spam", [])
ss.setdefault("spam_metrics", {"TP":0, "FP":0, "TN":0, "FN":0})

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Setup")
    mode = st.radio("Choose demo type", ["Image classification", "Text classification", "Numeric (CSV)"],
                    index=["Image classification", "Text classification", "Numeric (CSV)"].index(ss["mode"]))
    ss["mode"] = mode

    if mode == "Image classification":
        algo = st.selectbox("Algorithm", ["k-Nearest Neighbors", "Logistic Regression"])
    elif mode == "Text classification":
        algo = st.selectbox("Algorithm", ["Logistic Regression", "Linear SVM"])
        st.markdown("---")
        st.subheader("📧 Spam detector simulation")
        ss["spam_mode"] = st.toggle("Enable spam detector mode", value=ss["spam_mode"],
                                    help="Prepares labels 'spam' and 'inbox' and simulates autonomy via auto-routing.")
        if ss["spam_mode"]:
            ss["routing_mode"] = st.selectbox("Autonomy level", ["Manual review", "Suggest-only", "Auto-route"],
                                              index=["Manual review","Suggest-only","Auto-route"].index(ss["routing_mode"]))
            ss["threshold"] = st.slider("Spam threshold (probability of 'spam')", 0.1, 0.9, ss["threshold"], 0.05,
                                        help="Above this threshold, messages are routed to Spam in Auto-route mode.")
            st.caption("Tip: Logistic Regression provides calibrated probabilities; SVM uses a score mapped via a sigmoid approximation.")
    else:
        algo = st.selectbox("Algorithm", ["Logistic Regression", "Random Forest (clf)", "Linear Regression (reg)"])

    test_size = st.slider("Hold-out test size", min_value=0.1, max_value=0.5, value=0.3, step=0.05,
                          help="Portion of data held out for unbiased evaluation.")
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1, help="For reproducible splits.")
    ss["adaptive_mode"] = st.toggle("Adaptive learning (continuous updates from new labeled inputs)", value=ss["adaptive_mode"])

    st.markdown("---")
    st.caption("Session-scoped only; refresh to reset.")

# ---------------------------
# EU AI Act definition guidance
# ---------------------------
st.title("🧩 demistifAI — Demystify Machine Learning & the EU AI Act")
with st.expander("What is an **AI system** in the EU AI Act? (Simplified) 🏛️", expanded=True):
    st.markdown("""
- **Machine-based system**: software running on hardware. *In this app: the Streamlit UI (software) on a cloud host (hardware).*
- **Designed to operate with varying levels of autonomy**: it can act within boundaries without step-by-step instructions. *Here, the model makes predictions automatically once trained.*
- **Uses machine-learning approaches** to **infer** patterns from data during training. *You set an objective (labels); the model also discovers implicit objectives, e.g., 'color/word cues' correlated with labels.*
- **Generates outputs** (predictions/recommendations/decisions) that **influence** what a user does next. *Here, outputs are predicted labels or numeric values.*
- **May exhibit adaptiveness** over its lifecycle. *Toggle **Adaptive learning** to allow the model to update itself as you add new labeled cases during use.*
""")
    _chat_explain("assistant", "Think of the app as a small, transparent **machine-based system**: you control purpose and data; the model learns patterns and produces outputs.")

# ---------------------------
# Lifecycle helper
# ---------------------------
LIFECYCLE_TIPS = {
    0: ("Collect data (Machine-based system)", "Define intended purpose and the components: **software** (this UI + models) and **hardware** (cloud runtime). Prepare labeled examples with clear semantics."),
    1: ("Split data", "Plan for testing & validation. Keep a hold-out set to measure generalization."),
    2: ("Train (Inference learning)", "The model **infers** from data toward your objective (labels). Inputs during training are examples with labels; the algorithm adjusts parameters to minimize errors."),
    3: ("Evaluate", "Check accuracy/robustness; note limitations. Document results and known biases."),
    4: ("Predict (Generate output)", "Given new inputs, **inference** uses learned parameters to produce outputs (labels/values) without human step-by-step rules."),
    5: ("Model card (Autonomy & Adaptiveness)", "Describe intended use, autonomy boundaries, and whether **adaptive learning** is enabled.")
}

def eu_ai_act_box(step_idx: int):
    title, tip = LIFECYCLE_TIPS.get(step_idx, ("", ""))
    with st.expander(f"📘 AI Act lens — {title}", expanded=False):
        st.write(tip)
        st.caption("Educational mapping; adapt obligations to your role (provider/deployer) and risk level.")

# ---------------------------
# Wizard
# ---------------------------
steps = ["Collect data", "Split", "Train", "Evaluate", "Predict", "Model card"]
st.markdown("### Stepper")
cols = st.columns(len(steps))
for i, name in enumerate(steps):
    with cols[i]:
        st.button(f"{i+1}. {name}", key=f"goto_{i}", on_click=lambda idx=i: ss.update(step=idx))
st.progress((ss["step"]+1)/len(steps))

def do_split(X, y, stratify=None):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

def can_proceed_collect() -> bool:
    if ss["mode"] == "Image classification":
        return _guard_min_examples(len(ss["labels_image"]), len(ss["image_examples"]))
    if ss["mode"] == "Text classification":
        return _guard_min_examples(len(ss["labels_text"]), len(ss["text_examples"]))
    if ss["mode"] == "Numeric (CSV)":
        return ss["csv_df"] is not None and ss["csv_target"] in (ss["csv_df"].columns if ss["csv_df"] is not None else [])
    return False

def nav_buttons(disable_next: bool = False):
    c1, c2, _ = st.columns([1,1,6])
    with c1:
        st.button("⬅️ Back", disabled=ss["step"]==0, on_click=lambda: ss.update(step=max(0, ss["step"]-1)))
    with c2:
        st.button("Next ➡️", disabled=disable_next or ss["step"]>=len(steps)-1, on_click=lambda: ss.update(step=min(len(steps)-1, ss["step"]+1)))

def retrain_if_adaptive(train_fn):
    """Helper: if adaptive mode is on, call the provided train_fn() and message the user."""
    if ss.get("adaptive_mode", False):
        train_fn()
        _chat_explain("assistant", "🔁 **Adaptive learning** is enabled: the model has been updated with your new labeled example.")

# ---------------------------
# Modes
# ---------------------------
if ss["mode"] == "Image classification":
    # Step 0: Collect
    if ss["step"] == 0:
        st.subheader("📷 Collect labeled image examples")
        eu_ai_act_box(0)
        _binary_tip()

        c1, c2 = st.columns(2)
        with c1:
            new_label = st.text_input("Class label (e.g., 'cat', 'dog')", key="img_label", help="You set the **objective** by naming classes.")
            files = st.file_uploader("Upload images (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True, key="img_files", help="Multiple examples per class help learning.")
            if st.button("➕ Add to dataset", type="primary"):
                if not new_label or not files:
                    st.warning("Provide a class label and at least one image.")
                else:
                    for f in files:
                        ss["image_examples"].append({"label": new_label.strip(), "image_b64": _b64_from_bytes(f.getvalue())})
                    ss["labels_image"] = sorted(list({ex["label"] for ex in ss["image_examples"]}))
                    _chat_explain("assistant", f"You've **set an objective** via labels and provided training inputs. The model will try to discover **implicit objectives** (e.g., colors/shapes) that correlate with your labels.")
        with c2:
            st.write("Current dataset")
            if len(ss["image_examples"]) == 0:
                st.caption("No examples yet.")
            else:
                df_prev = pd.DataFrame([{"label": ex["label"], "bytes": len(_bytes_from_b64(ex["image_b64"]))} for ex in ss["image_examples"]])
                st.dataframe(df_prev, use_container_width=True, hide_index=True)

            # Export/Import
            st.markdown("**Export/Import**")
            if st.button("⬇️ Export dataset (.json)"):
                payload = json.dumps({"type": "image", "examples": ss["image_examples"]}, ensure_ascii=False)
                _download_text(payload, "image_dataset.json", "Download image_dataset.json")
            imp = st.file_uploader("Import dataset (.json)", type=["json"], key="img_import")
            if imp is not None:
                try:
                    data = json.loads(imp.getvalue().decode("utf-8"))
                    if data.get("type") == "image":
                        ss["image_examples"] = data.get("examples", [])
                        ss["labels_image"] = sorted(list({ex["label"] for ex in ss["image_examples"]}))
                        st.success("Imported image dataset.")
                except Exception as e:
                    st.error(f"Failed to import: {e}")

        nav_buttons(disable_next=not can_proceed_collect())

    # Step 1: Split
    if ss["step"] == 1 and can_proceed_collect():
        st.subheader("✂️ Split into train/test")
        eu_ai_act_box(1)
        X = np.stack([_read_image(_bytes_from_b64(ex["image_b64"])) for ex in ss["image_examples"]])
        y = np.array([ex["label"] for ex in ss["image_examples"]])
        X_train, X_test, y_train, y_test = do_split(X, y, stratify=y)
        st.session_state["img_split"] = (X_train, X_test, y_train, y_test)
        st.write(f"Train size: **{len(y_train)}** | Test size: **{len(y_test)}** | Classes: {sorted(list(set(y)))}")
        nav_buttons()

    # Step 2: Train (Inference learning)
    if ss["step"] == 2 and "img_split" in ss:
        st.subheader("🧠 Train (learn to **infer** from data)")
        eu_ai_act_box(2)
        X_train, X_test, y_train, y_test = st.session_state["img_split"]
        if algo == "k-Nearest Neighbors":
            model = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", KNeighborsClassifier(n_neighbors=min(3, len(X_train))))])
        else:
            model = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", LogisticRegression(max_iter=200))])

        def train_model():
            model.fit(X_train, y_train)
            ss["image_model"] = model

        if st.button("🚀 Train", type="primary"):
            train_model()
            st.success("Model trained.")
            _chat_explain("assistant", "During training, the algorithm **learns parameters** that capture patterns linked to your labels (your explicit objective). It may also capture **implicit objectives** (shortcuts) present in the data.")

        nav_buttons(disable_next=ss["image_model"] is None)

    # Step 3: Evaluate
    if ss["step"] == 3 and ss.get("image_model") is not None and "img_split" in ss:
        st.subheader("📊 Evaluate")
        eu_ai_act_box(3)
        X_train, X_test, y_train, y_test = ss["img_split"]
        y_pred = ss["image_model"].predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Test accuracy: **{acc:.2%}** on {len(y_test)} samples.")
        st.dataframe(_confusion_matrix_df(y_test, y_pred, ss["labels_image"]), use_container_width=True)
        st.text("Classification report:")
        st.code(classification_report(y_test, y_pred), language="text")
        nav_buttons()

    # Step 4: Predict (Generate output) + Adaptiveness
    if ss["step"] == 4 and ss.get("image_model") is not None:
        st.subheader("🔮 Predict (generate output)")
        eu_ai_act_box(4)
        test_img = st.file_uploader("Upload an image to classify", type=["png","jpg","jpeg"], key="predict_img")
        if test_img is not None:
            feats = _read_image(test_img.getvalue()).reshape(1, -1)
            pred = ss["image_model"].predict(feats)[0]
            st.success(f"Predicted label: **{pred}**")
            st.image(test_img, caption="Your image", use_container_width=False)
            _chat_explain("assistant", "In **inference**, the model applies learned parameters to new inputs to **generate outputs** (predicted labels) without step-by-step rules.")

            # Adaptive learning controls
            if ss.get("adaptive_mode", False):
                st.markdown("**Adaptive learning:** Add this case to the dataset and update the model.")
                new_label = st.text_input("Label for this image", value=str(pred), key="img_new_label")
                if st.button("Learn from this example", key="img_adapt_btn"):
                    ss["image_examples"].append({"label": new_label.strip(), "image_b64": _b64_from_bytes(test_img.getvalue())})
                    ss["labels_image"] = sorted(list({ex["label"] for ex in ss["image_examples"]}))
                    # Recompute split & retrain
                    X = np.stack([_read_image(_bytes_from_b64(ex["image_b64"])) for ex in ss["image_examples"]])
                    y = np.array([ex["label"] for ex in ss["image_examples"]])
                    X_train, X_test, y_train, y_test = do_split(X, y, stratify=y)
                    ss["img_split"] = (X_train, X_test, y_train, y_test)
                    def train_model():
                        if algo == "k-Nearest Neighbors":
                            m = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", KNeighborsClassifier(n_neighbors=min(3, len(X_train))))])
                        else:
                            m = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", LogisticRegression(max_iter=200))])
                        m.fit(X_train, y_train)
                        ss["image_model"] = m
                    retrain_if_adaptive(train_model)

        # Autonomy explanation
        st.info("**Autonomy** in this demo is **limited**: the model predicts automatically, but humans define purpose, supply data, and decide whether to update the model (adaptive learning).")
        nav_buttons()

    # Step 5: Model card (incl. autonomy/adaptiveness)
    if ss["step"] == 5:
        st.subheader("🪪 Model card")
        eu_ai_act_box(5)
        labels = ", ".join(ss["labels_image"])
        n_samples = len(ss["image_examples"])
        algo_used = algo
        metrics = ""
        if "img_split" in ss and ss.get("image_model") is not None:
            X_train, X_test, y_train, y_test = ss["img_split"]
            y_pred = ss["image_model"].predict(X_test)
            metrics = f"Accuracy: {accuracy_score(y_test, y_pred):.2%} (n={len(y_test)})"
        card_md = f"""
# Model Card — Image Classifier (demistifAI)
**Intended purpose**: Educational demo.

**Algorithm**: {algo_used}  
**Labels**: {labels or '[none]'}  
**Dataset size**: {n_samples} images

**Training/testing split**: {int((1-test_size)*100)}% / {int(test_size*100)}%  
**Key metrics**: {metrics or 'Train a model to populate metrics.'}

**Autonomy**: Limited. The system predicts automatically but humans set goals, supply data, and approve updates.  
**Adaptiveness**: {'Enabled' if ss.get('adaptive_mode', False) else 'Disabled'} (users may add labeled cases during use).  

**Data**: user-uploaded, session-scoped. Features: color histograms + downsampled pixels.  
**Known limitations**: tiny datasets, simple features.  
**AI Act mapping**: Machine-based system; Inference; Output; Limited autonomy; Optional adaptiveness.
"""
        st.markdown(card_md)
        _download_text(card_md, "model_card_image.md", "Download model_card_image.md")
        nav_buttons(disable_next=True)

elif ss["mode"] == "Text classification":
    # Step 0: Collect
    if ss["step"] == 0:
        st.subheader("✍️ Collect labeled text examples")
        eu_ai_act_box(0)
        if ss["spam_mode"]:
            st.info("**Spam detector mode**: Use labels like **spam** and **inbox** (ham). You can start with a few examples per class.")
            samples_col1, samples_col2 = st.columns(2)
            with samples_col1:
                if st.button("Add sample SPAM lines"):
                    for line in ["WIN a FREE iPhone!!!", "You have been selected for a prize", "Earn $$$ fast!!!", "Exclusive offer, limited time!!!"]:
                        ss["text_examples"].append({"text": line, "label": "spam"})
            with samples_col2:
                if st.button("Add sample INBOX lines"):
                    for line in ["Team meeting moved to 14:00", "Invoice for August attached", "Could you review the draft?", "Lunch tomorrow?"]:
                        ss["text_examples"].append({"text": line, "label": "inbox"})
            ss["labels_text"] = sorted(list({ex["label"] for ex in ss["text_examples"]}))

        _chat_explain("assistant", "You define the **objective** via class labels; each line is a training input. The model will try to infer patterns (words/phrases) correlated with labels.")

        c1, c2 = st.columns(2)
        with c1:
            label_t = st.text_input("Class label", key="txt_label")
            text_area = st.text_area("Paste one example per line", key="txt_examples", height=160, placeholder="Great product!\nReally loved it.\nWould buy again.")
            if st.button("➕ Add to dataset", type="primary", key="add_txt"):
                if not label_t or not text_area.strip():
                    st.warning("Provide a class label and at least one line.")
                else:
                    for line in text_area.splitlines():
                        line = line.strip()
                        if line:
                            ss["text_examples"].append({"text": line, "label": label_t.strip()})
                    ss["labels_text"] = sorted(list({ex["label"] for ex in ss["text_examples"]}))
                    _chat_explain("assistant", "Inputs (text) + objective (label) form the training data for inference learning.")
        with c2:
            st.write("Current dataset")
            if len(ss["text_examples"]) == 0:
                st.caption("No examples yet.")
            else:
                st.dataframe(pd.DataFrame(ss["text_examples"]), use_container_width=True, hide_index=True)

            # Export/Import
            st.markdown("**Export/Import**")
            if st.button("⬇️ Export dataset (.json)", key="export_text"):
                payload = json.dumps({"type": "text", "examples": ss["text_examples"]}, ensure_ascii=False)
                _download_text(payload, "text_dataset.json", "Download text_dataset.json")
            imp = st.file_uploader("Import dataset (.json)", type=["json"], key="txt_import")
            if imp is not None:
                try:
                    data = json.loads(imp.getvalue().decode("utf-8"))
                    if data.get("type") == "text":
                        ss["text_examples"] = data.get("examples", [])
                        ss["labels_text"] = sorted(list({ex["label"] for ex in ss["text_examples"]}))
                        st.success("Imported text dataset.")
                except Exception as e:
                    st.error(f"Failed to import: {e}")

        nav_buttons(disable_next=not can_proceed_collect())

    # Step 1: Split
    if ss["step"] == 1 and can_proceed_collect():
        st.subheader("✂️ Split into train/test")
        eu_ai_act_box(1)
        df = pd.DataFrame(ss["text_examples"])
        X = df["text"].tolist()
        y = df["label"].values
        X_train, X_test, y_train, y_test = do_split(X, y, stratify=y)
        ss["txt_split"] = (X_train, X_test, y_train, y_test)
        st.write(f"Train size: **{len(y_train)}** | Test size: **{len(y_test)}** | Classes: {sorted(list(set(y)))}")
        nav_buttons()

    # Step 2: Train
    if ss["step"] == 2 and "txt_split" in ss:
        st.subheader("🧠 Train (learn to **infer** from text)")
        eu_ai_act_box(2)
        X_train, X_test, y_train, y_test = ss["txt_split"]

        if algo == "Logistic Regression":
            clf = LogisticRegression(max_iter=1000)
        else:
            clf = LinearSVC()

        model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)), ("clf", clf)])

        def train_model():
            model.fit(X_train, y_train)
            ss["text_model"] = model

        if st.button("🚀 Train", type="primary", key="train_txt_btn"):
            train_model()
            st.success("Model trained.")
            _chat_explain("assistant", "The model inferred weights for words/phrases (features) to distinguish your classes — that's **inference learning**.")

        nav_buttons(disable_next=ss["text_model"] is None)

    # Step 3: Evaluate
    if ss["step"] == 3 and ss.get("text_model") is not None and "txt_split" in ss:
        st.subheader("📊 Evaluate")
        eu_ai_act_box(3)
        X_train, X_test, y_train, y_test = ss["txt_split"]
        y_pred = ss["text_model"].predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Test accuracy: **{acc:.2%}** on {len(y_test)} samples.")
        st.dataframe(_confusion_matrix_df(y_test, y_pred, ss["labels_text"]), use_container_width=True)
        st.text("Classification report:")
        st.code(classification_report(y_test, y_pred), language="text")
        nav_buttons()

    # Step 4: Predict + Adaptiveness + Spam routing
    if ss["step"] == 4 and ss.get("text_model") is not None:
        st.subheader("🔮 Predict (generate output)")
        eu_ai_act_box(4)
        new_text = st.text_area("Paste an email (subject + body)", key="predict_text", height=160, placeholder="Subject: Invoice for August\nHi, please find the invoice attached...")
        pred_label = None
        prob_spam = None

        def _predict_with_prob(model, text):
            if not text.strip():
                return None, None
            y_hat = model.predict([text])[0]
            # Probability estimate if available
            p = None
            if hasattr(model.named_steps["clf"], "predict_proba") and "spam" in ss["labels_text"]:
                idx = list(model.named_steps["clf"].classes_).index("spam")
                p = model.predict_proba([text])[0][idx]
            elif hasattr(model.named_steps["clf"], "decision_function") and "spam" in ss["labels_text"]:
                scores = model.named_steps["clf"].decision_function([text])
                p = float(expit(scores[0])) if np.ndim(scores)==1 else None
            return y_hat, p

        if st.button("Predict", key="predict_btn"):
            if not new_text.strip():
                st.warning("Please enter text.")
            else:
                pred_label, prob_spam = _predict_with_prob(ss["text_model"], new_text)
                if ss["spam_mode"] and prob_spam is not None:
                    st.success(f"Prediction: **{pred_label}** — Estimated P(spam) ≈ **{prob_spam:.2f}**")
                else:
                    st.success(f"Prediction: **{pred_label}**")

                # Routing simulation
                decision = "No action"
                if ss["spam_mode"]:
                    if ss["routing_mode"] == "Manual review":
                        decision = "Hold for human review"
                    elif ss["routing_mode"] == "Suggest-only":
                        decision = f"Suggest: route to {'Spam' if (prob_spam or 0) >= ss['threshold'] or pred_label=='spam' else 'Inbox'}"
                    else:  # Auto-route
                        route_spam = False
                        if prob_spam is not None:
                            route_spam = prob_spam >= ss["threshold"]
                        else:
                            route_spam = (pred_label == "spam")
                        if route_spam:
                            ss["mail_spam"].append({"text": new_text, "pred": pred_label, "prob": prob_spam})
                            decision = "Auto-routed to **Spam**"
                        else:
                            ss["mail_inbox"].append({"text": new_text, "pred": pred_label, "prob": prob_spam})
                            decision = "Auto-routed to **Inbox**"
                st.info(f"Autonomy action: {decision}")

        # Ground truth & learning
        if ss["spam_mode"] and new_text.strip():
            gt = st.selectbox("Mark ground truth for this email", ["", "spam", "inbox"], index=0, help="Record actual label to track errors and (optionally) learn.")
            if st.button("Record ground truth"):
                if gt:
                    # Determine system decision in auto-route; otherwise use predicted label
                    system_label = pred_label or ""
                    if ss["routing_mode"] == "Auto-route":
                        if ss["mail_spam"] and ss["mail_spam"][-1]["text"] == new_text:
                            system_label = "spam"
                        elif ss["mail_inbox"] and ss["mail_inbox"][-1]["text"] == new_text:
                            system_label = "inbox"

                    # Update metrics
                    if   gt == "spam"  and system_label == "spam":  ss["spam_metrics"]["TP"] += 1
                    elif gt == "spam"  and system_label == "inbox": ss["spam_metrics"]["FN"] += 1
                    elif gt == "inbox" and system_label == "spam":  ss["spam_metrics"]["FP"] += 1
                    elif gt == "inbox" and system_label == "inbox": ss["spam_metrics"]["TN"] += 1
                    st.success("Recorded. Metrics updated.")

                    # Adaptive learning from ground truth
                    if ss.get("adaptive_mode", False):
                        ss["text_examples"].append({"text": new_text.strip(), "label": gt})
                        ss["labels_text"] = sorted(list({ex["label"] for ex in ss["text_examples"]}))
                        # Re-split and retrain
                        df = pd.DataFrame(ss["text_examples"])
                        X = df["text"].tolist()
                        y = df["label"].values
                        X_train, X_test, y_train, y_test = do_split(X, y, stratify=y)
                        ss["txt_split"] = (X_train, X_test, y_train, y_test)
                        # Rebuild model consistent with chosen algo
                        if algo == "Logistic Regression":
                            clf = LogisticRegression(max_iter=1000)
                        else:
                            clf = LinearSVC()
                        m = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)), ("clf", clf)])
                        m.fit(X_train, y_train)
                        ss["text_model"] = m
                        _chat_explain("assistant", "🔁 Adaptive learning: model updated with the ground-truth label you provided.")

        # Show simulated mailboxes
        if ss["spam_mode"]:
            st.markdown("### 📥 Simulated Mailboxes")
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Inbox** — {len(ss['mail_inbox'])} message(s)")
                if ss["mail_inbox"]:
                    st.dataframe(pd.DataFrame(ss["mail_inbox"]), use_container_width=True, hide_index=True)
                else:
                    st.caption("Empty")
            with c2:
                st.write(f"**Spam** — {len(ss['mail_spam'])} message(s)")
                if ss["mail_spam"]:
                    st.dataframe(pd.DataFrame(ss["mail_spam"]), use_container_width=True, hide_index=True)
                else:
                    st.caption("Empty")

            # Running metrics
            st.markdown("### 📊 Running metrics (based on recorded ground truth)")
            m = ss["spam_metrics"]
            total = sum(m.values()) or 1
            accuracy = (m["TP"]+m["TN"])/total
            st.write(f"TP: {m['TP']} | FP: {m['FP']} | TN: {m['TN']} | FN: {m['FN']} — Accuracy: **{accuracy:.2%}**")

        st.info("**Autonomy in spam detection**: at higher autonomy (Auto-route), misclassifications have direct user impact (e.g., lost emails). Use thresholds, oversight, and feedback loops to manage risk.")
        nav_buttons()

    # Step 5: Model card
    if ss["step"] == 5:
        st.subheader("🪪 Model card")
        eu_ai_act_box(5)
        labels = ", ".join(ss["labels_text"])
        n_samples = len(ss["text_examples"])
        metrics = ""
        if "txt_split" in ss and ss.get("text_model") is not None:
            X_train, X_test, y_train, y_test = ss["txt_split"]
            y_pred = ss["text_model"].predict(X_test)
            metrics = f"Accuracy: {accuracy_score(y_test, y_pred):.2%} (n={len(y_test)})"
        autonomy = "Auto-route" if (ss.get("spam_mode", False) and ss.get("routing_mode")=="Auto-route") else "Limited / manual"
        card_md = f"""
# Model Card — Text Classifier (demistifAI)
**Intended purpose**: Educational demo {('(spam detector simulation)') if ss.get('spam_mode', False) else ''}.

**Algorithm**: {algo}  
**Labels**: {labels or '[none]'}  
**Dataset size**: {n_samples} examples

**Training/testing split**: {int((1-test_size)*100)}% / {int(test_size*100)}%  
**Key metrics**: {metrics or 'Train a model to populate metrics.'}

**Autonomy**: {autonomy}.  
**Adaptiveness**: {'Enabled' if ss.get('adaptive_mode', False) else 'Disabled'}.  

**Data**: user-provided, session-scoped. Features: TF‑IDF.  
**Known limitations**: small datasets; vocabulary sensitivity.  
**AI Act mapping**: Machine-based system; Inference; Output; Limited autonomy (or higher with auto-routing); Optional adaptiveness.
"""
        st.markdown(card_md)
        _download_text(card_md, "model_card_text.md", "Download model_card_text.md")
        nav_buttons(disable_next=True)

else:  # Numeric (CSV)
    # Step 0: Collect
    if ss["step"] == 0:
        st.subheader("🧮 Upload CSV (numeric)")
        eu_ai_act_box(0)
        st.caption("This mode accepts a CSV with a header. Choose **task type** and **target column**.")

        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_file")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                ss["csv_df"] = df
                st.success(f"Loaded CSV with shape {df.shape}.")
                st.dataframe(df.head(), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

        ss["task_type"] = st.selectbox("Task type", ["Classification", "Regression"])
        if ss["csv_df"] is not None:
            target = st.selectbox("Target column", list(ss["csv_df"].columns))
            ss["csv_target"] = target

        nav_buttons(disable_next=not can_proceed_collect())

    # Step 1: Split
    if ss["step"] == 1 and can_proceed_collect():
        st.subheader("✂️ Split into train/test")
        eu_ai_act_box(1)
        df = ss["csv_df"].dropna()
        y = df[ss["csv_target"]]
        X = df.drop(columns=[ss["csv_target"]])
        if ss["task_type"] == "Classification":
            stratify = y if len(pd.unique(y))>1 else None
        else:
            stratify = None
        X_train, X_test, y_train, y_test = do_split(X, y, stratify=stratify)
        ss["csv_split"] = (X_train, X_test, y_train, y_test)
        st.write(f"Train size: **{len(y_train)}** | Test size: **{len(y_test)}** | Features: {list(X.columns)[:6]}{'...' if X.shape[1]>6 else ''}")
        nav_buttons()

    # Step 2: Train
    if ss["step"] == 2 and "csv_split" in ss:
        st.subheader("🧠 Train (learn to **infer** from tabular data)")
        eu_ai_act_box(2)
        X_train, X_test, y_train, y_test = ss["csv_split"]
        if ss["task_type"] == "Classification":
            if algo == "Random Forest (clf)":
                model = RandomForestClassifier(n_estimators=200, random_state=random_state)
            else:
                model = LogisticRegression(max_iter=1000)
        else:  # Regression
            model = LinearRegression()

        def train_model():
            model.fit(X_train, y_train)
            ss["csv_model"] = model

        if st.button("🚀 Train", type="primary", key="train_csv_btn"):
            train_model()
            st.success("Model trained.")
            _chat_explain("assistant", "Training tunes parameters to minimize error on your labeled examples — the essence of **inference learning**.")

        nav_buttons(disable_next=ss["csv_model"] is None)

    # Step 3: Evaluate
    if ss["step"] == 3 and ss.get("csv_model") is not None and "csv_split" in ss:
        st.subheader("📊 Evaluate")
        eu_ai_act_box(3)
        X_train, X_test, y_train, y_test = ss["csv_split"]
        y_pred = ss["csv_model"].predict(X_test)
        if ss["task_type"] == "Classification":
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Test accuracy: **{acc:.2%}** on {len(y_test)} samples.")
            labels = sorted(list(pd.unique(y_test)))
            try:
                st.dataframe(_confusion_matrix_df(y_test, y_pred, labels), use_container_width=True)
            except Exception:
                pass
        else:
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            st.success(f"Regression — MAE: **{mae:.4f}**, RMSE: **{rmse:.4f}**, R²: **{r2:.3f}**")
        nav_buttons()

    # Step 4: Predict + Adaptiveness
    if ss["step"] == 4 and ss.get("csv_model") is not None and "csv_split" in ss:
        st.subheader("🔮 Predict (generate output)")
        eu_ai_act_box(4)
        st.caption("Upload a single-row CSV with the **same columns** as training features (excluding the target).")
        new_csv = st.file_uploader("Upload single-row CSV for prediction", type=["csv"], key="predict_csv")
        if new_csv is not None:
            try:
                new_df = pd.read_csv(new_csv)
                if ss["csv_target"] in new_df.columns:
                    new_df = new_df.drop(columns=[ss["csv_target"]])
                pred = ss["csv_model"].predict(new_df)[0]
                st.success(f"Prediction: **{pred}**")
                # Adaptive learning controls
                if ss.get("adaptive_mode", False):
                    st.markdown("**Adaptive learning:** Provide the true target to learn from this example and update the model.")
                    true_y = st.text_input("True target value", key="csv_true_y")
                    if st.button("Learn from this example", key="csv_adapt_btn"):
                        try:
                            row = new_df.iloc[0].to_dict()
                            # Do not cast types aggressively; keep simple for demo
                            row[ss["csv_target"]] = true_y
                            ss["csv_df"] = pd.concat([ss["csv_df"], pd.DataFrame([row])], ignore_index=True)
                            # Re-split and retrain
                            df = ss["csv_df"].dropna()
                            y = df[ss["csv_target"]]
                            X = df.drop(columns=[ss["csv_target"]])
                            stratify = y if (ss["task_type"] == "Classification" and len(pd.unique(y))>1) else None
                            X_train, X_test, y_train, y_test = do_split(X, y, stratify=stratify)
                            ss["csv_split"] = (X_train, X_test, y_train, y_test)
                            def train_model():
                                if ss["task_type"] == "Classification":
                                    if algo == "Random Forest (clf)":
                                        m = RandomForestClassifier(n_estimators=200, random_state=random_state)
                                    else:
                                        m = LogisticRegression(max_iter=1000)
                                else:
                                    m = LinearRegression()
                                m.fit(X_train, y_train)
                                ss["csv_model"] = m
                            retrain_if_adaptive(train_model)
                        except Exception as e:
                            st.error(f"Failed to learn from example: {e}")
            except Exception as e:
                st.error(f"Failed to predict: {e}")
        st.info("**Autonomy** is limited: predictions are automatic within boundaries, but humans remain in control of objectives and updates.")
        nav_buttons()

    # Step 5: Model card
    if ss["step"] == 5:
        st.subheader("🪪 Model card")
        eu_ai_act_box(5)
        n_rows = int(ss["csv_df"].shape[0]) if ss["csv_df"] is not None else 0
        target = ss["csv_target"]
        metrics = ""
        if "csv_split" in ss and ss.get("csv_model") is not None:
            X_train, X_test, y_train, y_test = ss["csv_split"]
            y_pred = ss["csv_model"].predict(X_test)
            if ss["task_type"] == "Classification":
                metrics = f"Accuracy: {accuracy_score(y_test, y_pred):.2%} (n={len(y_test)})"
            else:
                mae = mean_absolute_error(y_test, y_pred); rmse = mean_squared_error(y_test, y_pred, squared=False); r2 = r2_score(y_test, y_pred)
                metrics = f"MAE {mae:.4f}, RMSE {rmse:.4f}, R² {r2:.3f} (n={len(y_test)})"
        card_md = f"""
# Model Card — Numeric ({ss['task_type']}) (demistifAI)
**Intended purpose**: Educational demo.

**Algorithm**: {algo}  
**Target**: {target or '[not set]'}  
**Dataset size**: {n_rows} rows

**Training/testing split**: {int((1-test_size)*100)}% / {int(test_size*100)}%  
**Key metrics**: {metrics or 'Train a model to populate metrics.'}

**Autonomy**: Limited; predictions within defined boundaries.  
**Adaptiveness**: {'Enabled' if ss.get('adaptive_mode', False) else 'Disabled'}.  

**Data**: user-provided CSV; missing values dropped.  
**Known limitations**: basic preprocessing; no categorical encoding in this demo.  
**AI Act mapping**: Machine-based system; Inference; Output; Limited autonomy; Optional adaptiveness.
"""
        st.markdown(card_md)
        _download_text(card_md, "model_card_tabular.md", "Download model_card_tabular.md")
        nav_buttons(disable_next=True)

# ---------------------------
# Footer: autonomy & adaptiveness recap
# ---------------------------
st.markdown("---")
with st.expander("⚖️ Autonomy & Adaptiveness — in this demo context"):
    st.markdown("""
- **Autonomy (levels)**: Manual review → Suggest-only → Auto-route. Higher autonomy moves decisions without human approval.
- **Adaptiveness (optional)**: if enabled, you can add ground-truthed cases during use; the model updates accordingly.
- **Risk**: false positives may hide important emails; thresholds, oversight, and feedback loops mitigate risk.
""")
