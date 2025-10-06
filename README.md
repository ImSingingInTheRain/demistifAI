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
