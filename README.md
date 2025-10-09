# demistifAI — Guided EU AI Act Spam Detector Lab

DemistifAI is an interactive Streamlit experience that walks teams through the EU AI Act definition of an **AI system** while they actually build, evaluate, and operate a spam detector. Every stage of the workflow ties legal concepts (machine-based system, inference, outputs, autonomy, adaptiveness) to hands-on steps so non-technical stakeholders can explore how governance meets machine learning.

## What you learn
- **Machine-based system** – Understand how a Streamlit front-end, Python models, and a cloud runtime come together as one system.
- **Inference & objectives** – See how the model learns the explicit objective (*keep spam out of the inbox*) from labeled examples and how implicit objectives can creep in via the data you feed it.
- **Outputs & autonomy** – Observe predictions, confidence scores, and mailbox routing with both recommendation-only and high-autonomy modes.
- **Adaptiveness** – Optionally enable adaptive learning to feed back user corrections and retrain the model during use.

## Stage-by-stage tour
1. **🚀 Welcome** – Mission briefing, inbox preview, and quick-start controls.
2. **🧭 Start your machine** – EU AI Act primer, lifecycle overview, and "Nerd Mode" toggle for architecture/package details.
3. **📊 Prepare Data** – Inspect the 502-item seed dataset (balanced spam/safe emails), add single examples, or bulk-import labeled CSVs in the expected schema.
4. **🧠 Train** – Configure train/test split, solver iterations, and regularization (C). Kick off training, inspect feature weights, adjust numeric cues (links, suspicious TLDs, punctuation bursts, etc.), and review nearest-neighbour prototypes from the MiniLM sentence embeddings.
5. **🧪 Evaluate** – Review narratives, accuracy snapshot, and confusion matrix on the hold-out set. Experiment with spam-threshold presets (max F1, ≥95% precision, ≥90% recall), compare outcomes, and dive into precision/recall tables, threshold curves, and governance metadata in Nerd Mode.
6. **📬 Use** – Process incoming emails in recommendation or high-autonomy mode. Export batch results (CSV/JSON), review inbox/spam mailboxes, and (when adaptiveness is on) confirm/correct predictions then trigger a feedback-driven retrain.
7. **📄 Model Card** – Auto-generate a Markdown model card summarizing purpose, data, metrics, autonomy settings, and AI Act mapping, with an instant download link.

## Detailed walkthrough of each build stage

The following sections walk through the five core build stages—**Start your machine**, **Prepare Data**, **Train**, **Evaluate**, and **Use**—highlighting the user experience in normal and Nerd Mode alongside the underlying code that powers each step. Use this as a guide when extending functionality or refactoring the Streamlit app.

### 🧭 Start your machine

**Normal mode experience.** The `render_overview_stage()` function opens with an EU AI Act quote, a mission briefing, and an inbox preview so participants see the system context before building anything. The hero callout explains that they are already operating inside a "machine-based system" composed of the Streamlit UI, Python backend, and cloud runtime. A mission panel outlines the expected outcomes (working spam detector, model card, and regulation literacy) while the right column shows up to five incoming email previews pulled from session state.

**Nerd Mode enhancements.** Toggling Nerd Mode (via `render_nerd_mode_toggle`) adds a technical rundown of the architecture, key Python packages, data flow, and caching approach, grounding the legal framing in concrete implementation details.

```python
def render_overview_stage():
    with section_surface():
        intro_left, intro_right = st.columns(2, gap="large")
        with intro_left:
            render_eu_ai_quote("The EU AI Act says that “An AI system is a machine based system”.")
        with intro_right:
            st.markdown("""<div class="callout callout--info"> ... </div>""", unsafe_allow_html=True)

    with section_surface():
        nerd_enabled = render_nerd_mode_toggle(
            key="nerd_mode",
            title="Nerd Mode",
            icon="🧠",
            description="At every stage you can activate a Nerd Mode …"
        )
    if nerd_enabled:
        with section_surface():
            st.markdown("### Nerd Mode — technical details")
            st.markdown("- **Architecture:** Streamlit app (Python) …")
```

Flip the switch when you need a deeper system-level explanation; leave it off to keep the facilitation narrative front-and-centre.

### 📊 Prepare Data

**Normal mode experience.** `render_data_stage()` opens with EU AI Act text about explicit objectives and reminds participants that their goal is to label emails as spam or safe. The default view shows the in-memory labeled dataset and quick stats—total examples, class balance, and practical tips. Users who have not yet added data see helper messages steering them toward the labeling tools.

**Nerd Mode enhancements.** Enabling Nerd Mode reveals schema documentation, an example CSV, and two expanders: one for adding single labeled examples and another for importing a CSV. Import logic validates column names, normalises labels, drops empty rows, skips duplicates, and surfaces counts before writing back to `ss["labeled"]`.

```python
def render_data_stage():
    stage = STAGE_BY_KEY["data"]
    …
    nerd_data = render_nerd_mode_toggle(
        key="nerd_mode_data",
        title="Nerd Mode",
        description="Peek into schema expectations and options to extend the dataset.",
    )
    …
    with st.expander("📤 Upload a CSV of labeled emails", expanded=False):
        up = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader_labeled")
        if up is not None:
            df_up = pd.read_csv(up)
            df_up.columns = [c.strip().lower() for c in df_up.columns]
            ok, msg = _validate_csv_schema(df_up)
            if ok:
                df_clean = df_up.loc[mask_valid, ["title", "body", "label"]].copy()
                …
                if st.button("✅ Import into dataset", key="btn_import_csv"):
                    ss["labeled"].extend(df_merge.to_dict(orient="records"))
```

This stage is the home of all data curation logic, making it the best place to extend schema support or plug in data quality checks.

### 🧠 Train

**Normal mode experience.** `render_train_stage()` reinforces the EU AI Act concept of inference and provides a practical checklist (balanced classes, adequate examples) before exposing a primary "🚀 Train model" button. Pressing the button triggers validation (minimum six examples and both labels present) and, if successful, trains a `HybridEmbedFeatsLogReg` classifier using sentence embeddings plus engineered numeric features.

**🔬 Nerd Mode enhancements.** The Nerd Mode toggle exposes sliders for hold-out size and threshold, numeric inputs for random seed, solver iterations, and regularisation (`C`). After training, the stage also surfaces interpretable outputs—feature weight tables, adjustable sliders for numeric cues, and embedding prototype explorers.

```python
nerd_mode_train_enabled = render_nerd_mode_toggle(
    key="nerd_mode_train",
    title="Nerd Mode — advanced controls",
    description="Tweak the train/test split, solver iterations, and regularization strength.",
    icon="🔬",
)
…
if trigger_train:
    if len(ss["labeled"]) < 6:
        st.warning("Please label a few more emails first (≥6 examples).")
    else:
        df = pd.DataFrame(ss["labeled"])
        …
        model = HybridEmbedFeatsLogReg().fit(X_tr_t, X_tr_b, y_tr)
        model.apply_numeric_adjustments(ss["numeric_adjustments"])
        ss["model"] = model
        ss["split_cache"] = (X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te)
```

Post-training UIs—including coefficient charts and the per-feature adjustment sliders—live in this function, so add new interpretability or tuning affordances here.

### 🧪 Evaluate

**Normal mode experience.** `render_evaluate_stage()` first checks for a trained model and cached split. When available, it computes hold-out probabilities, builds a confusion matrix, summarises accuracy, and narrates what the numbers mean. Threshold presets (balanced F1, ≥95% precision, ≥90% recall) and a slider let participants test different operating points before adopting one for production use.

**Nerd Mode enhancements.** The advanced panel adds per-class precision/recall/F1 tables, threshold trade-off curves, coefficient summaries (for linear models), and governance metadata such as split sizes, random seeds, and evaluation timestamps.

```python
cm = compute_confusion(y_true01, p_spam, current_thr)
acc = (cm["TP"] + cm["TN"]) / max(1, len(y_true01))
…
if st.button("Adopt this threshold", use_container_width=True):
    ss["threshold"] = float(ss.get("eval_temp_threshold", current_thr))
    st.success("Adopted new operating threshold …")

nerd_mode_eval_enabled = render_nerd_mode_toggle(
    key="nerd_mode_eval",
    title="Nerd Mode — technical details",
    description="Inspect precision/recall tables, interpretability cues, and governance notes.",
    icon="🔬",
)
if nerd_mode_eval_enabled:
    st.dataframe(pd.DataFrame([...]).round(3), width="stretch", hide_index=True)
    fig = plot_threshold_curves(y_true01, p_spam)
    st.pyplot(fig)
```

Use this stage when you need to extend evaluation metrics, add governance artefacts, or surface new explainability widgets.

### 📬 Use

**Normal mode experience.** `render_classify_stage()` reiterates the EU AI Act framing that AI systems generate outputs (predictions and recommendations) from inputs (emails). Users can toggle between recommendation-only and high-autonomy routing. Processing a batch of incoming emails yields predictions, recommended actions, mailbox moves (if autonomy is enabled), and exportable CSV/JSON summaries.

**Nerd Mode enhancements.** Instead of a single toggle, the stage layers in adaptive learning controls that let advanced users confirm/correct predictions, append the feedback to the labeled dataset, and retrain on demand—all while logging audit events. The mailboxes panel displays routed messages for ongoing monitoring.

```python
use_high_autonomy = st.toggle(
    "High autonomy (auto-move emails)", value=default_high_autonomy, key="use_high_autonomy"
)
…
if st.button(f"Process {preview_n} email(s)", type="primary", key="btn_process_batch"):
    batch = ss["incoming"][:preview_n]
    y_hat, p_spam, p_safe = _predict_proba_batch(ss["model"], batch)
    …
    if ss["use_high_autonomy"]:
        ss["mail_spam"].append(mailbox_record)
…
st.toggle(
    "Enable adaptiveness (learn from feedback)",
    value=bool(ss.get("adaptive", False)),
    key="adaptive_stage",
    on_change=_handle_stage_adaptive_change,
)
if use_adaptiveness and ss.get("use_batch_results"):
    if st.button("🔁 Retrain now with feedback", key="btn_retrain_feedback"):
        model = HybridEmbedFeatsLogReg().fit(X_tr_t, X_tr_b, y_tr)
        ss["model"] = model
```

The Use stage is where you would add new deployment surfaces (e.g., API endpoints) or richer feedback workflows.

## Key capabilities
- **Rich starter data** – 502 labeled emails (spam vs safe) seeded for rapid experimentation.
- **Sentence embeddings + numeric cues** – Combines MiniLM sentence embeddings with interpretable numeric features before a logistic regression classifier.
- **Threshold management** – Adopt thresholds tuned for balanced F1, inbox protection, or spam capture, and see the trade-offs before applying them.
- **Autonomy controls** – Switch between recommendation-only and auto-routing modes; audit mailbox movements in real time.
- **Adaptive learning loop** – Capture user confirmations/corrections, append them to the labeled dataset, and retrain without leaving the UI.
- **Governance guidance** – Contextual EU AI Act quotations, transparency popovers, and a downloadable model card for documentation.

## Running locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

> **Tip:** The app downloads MiniLM sentence-transformer weights the first time you train; keep the virtual environment around to avoid re-downloading.

## Data & privacy notes
- Seed data and any examples you add live in Streamlit session state only; no server-side persistence is included.
- CSV imports must include `title`, `body`, and `label` columns (labels: `spam` or `safe`). Empty rows and invalid labels are automatically filtered out.
- Adaptive retraining stays within the session — export the batch predictions or model card if you need an audit trail.

## License
See [LICENSE](LICENSE) for details.
