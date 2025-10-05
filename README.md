# demistifAI — Demystify Machine Learning & the EU AI Act

An interactive learning experience that walks users through ML **and** the AI Act’s definition of an **AI system** — while they actually train and use small models.

## What learners will experience
- **Machine-based system**: understand software (Streamlit UI + models) and hardware (cloud runtime) components.
- **Inference & objectives**: set explicit objectives (labels) and see how models also pick up **implicit** objectives in data.
- **Generating outputs**: watch inference produce predictions on new inputs.
- **Varying levels of autonomy**: see “limited autonomy” in practice and, in spam mode, raise autonomy to **auto-route**.
- **Adaptiveness**: optional **Adaptive learning** lets the model update when learners add new labeled cases during use.

## Features
- Wizard UI: **Collect → Split → Train → Evaluate → Predict → Model Card**
- Three modalities: **Images**, **Text**, **Numeric (CSV)**
- EU AI Act callouts at each step
- Export/Import datasets for images/text (JSON)
- Auto-generated **Model Cards** (downloadable Markdown)
- Optional **Adaptive learning** (continuous updates from newly labeled cases)

## Spam detector simulation (Text)
Toggle **Spam detector mode** to:
- Choose autonomy: **Manual review → Suggest-only → Auto-route**
- Set **P(spam)** threshold (Logistic Regression calibrated; SVM uses sigmoid on decision score)
- See live **Inbox/Spam** bins and **TP/FP/TN/FN** running metrics
- Record ground truth and, if **Adaptive learning** is on, re-train on the fly

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud via GitHub
1. Push `app.py`, `requirements.txt`, and `README.md` to your GitHub repo (e.g., `demistifAI/`).
2. In Streamlit Cloud: **New app → Connect to GitHub → select repo**.
3. Set **Main file path** to `app.py` and deploy.
4. (Optional) Pin Python version (e.g., 3.11) in advanced settings.

### Notes
- No secrets are required.
- Data lives only in the user session; for classroom presets, preload tiny datasets in a `data/` folder and extend the app code.
- Adaptive learning uses simple re-training on the accumulated dataset after corrections.
