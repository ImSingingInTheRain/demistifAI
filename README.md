# demistifAI — Guided EU AI Act Spam Detector Lab

DemistifAI is an interactive Streamlit lab that walks teams through the EU AI Act definition of an **AI system** while they actually build, evaluate, and operate a spam detector. Every stage in the app links the regulation’s language (machine-based system, autonomy, adaptiveness, inference, outputs) to hands-on controls so that non-technical and technical stakeholders can reason about governance together.

## What you learn
- **Machine-based system** – Experience how the Streamlit UI, Python model, and runtime work together as a single system.
- **Inference & objectives** – See how labeled data expresses the explicit goal (*block spam*) and how dataset choices alter the implicit objective.
- **Outputs, autonomy & guardrails** – Compare recommendation-only vs. high-autonomy routing, explore numeric guardrails around the text model, and set the operating threshold.
- **Adaptiveness & provenance** – Toggle feedback loops, retrain on user corrections, capture audit events, and stamp model cards with dataset snapshots.

## Stage-by-stage tour
1. **🚀 Welcome** – Mission briefing, lifecycle visual, and EU AI Act framing with a one-click launch into the build.
2. **🧭 Start your machine** – Meet the system components, check dataset/incoming status, and decide when to flip on Nerd Mode for technical deep dives.
3. **📊 Prepare Data** – Generate synthetic datasets (size, spam share, edge cases), tune advanced knobs for suspicious links/TLDs/attachments, inspect health + PII linting, clean flagged personal data, manually edit rows, commit the dataset, and save provenance snapshots or import CSVs.
4. **🧠 Train** – Configure train/test splits, solver parameters, and numeric assist guardrails; launch training for the MiniLM+logistic hybrid; review storytelling panels, meaning maps, paraphrase demos, and adjust numeric feature weights or inspect prototypes in Nerd Mode.
5. **🧪 Evaluate** – Measure hold-out performance with narratives, confusion matrix, and threshold presets; in Nerd Mode dig into per-class precision/recall, threshold trade-off curves, interpretability tables, and governance metadata.
6. **📬 Use** – Process incoming batches with autonomy toggles, inspect results, download CSV/JSON, review audit logs, and (optionally) enable adaptiveness to confirm/correct predictions, see nearest neighbours/numeric contributions, and retrain on feedback.
7. **📄 Model Card** – Auto-generate a Markdown model card capturing purpose, metrics, autonomy, adaptiveness, dataset snapshot ID, and configuration for transparency.

## Detailed walkthrough of each stage

### 🚀 Welcome
- **Normal mode highlights.** Hero surface explains the mission, quotes Article 3 of the EU AI Act, and previews the lifecycle loop you’ll repeat.
- **Guided hand-off.** Large call-to-action buttons jump straight into *Start your machine* so facilitators can keep momentum.

### 🧭 Start your machine
- **Normal mode highlights.** Mission brief panels recap what you’ll orchestrate (UI, model, inbox) and show live status for labeled data, incoming emails, autonomy selection, and adaptiveness. A compact inbox preview keeps the objective tangible.
- **Nerd Mode extras.** Deep-dive callouts detail the architecture, model pipeline, packages, autonomy levels, adaptiveness loop, governance artefacts, and the optional session audit log so technical readers can map the “machine-based system” to concrete components.

### 📊 Prepare Data
- **Normal mode highlights.**
  - **Dataset builder** lets you choose dataset size (100/300/500), spam share (20–80%), and the number of “edge case” lookalike pairs before generating a preview.
  - **Health & deltas** summarise class balance, total rows, suspicious link/TLD/money/attachment prevalence, and how the preview differs from the current dataset.
  - **Personal data linting** runs automatically; a cleanup game highlights spans and offers replacement tokens such as `{{EMAIL}}`, `{{IBAN}}`, `{{CARD_16}}`, `{{PHONE}}`, `{{OTP_6}}`, and `{{URL_SUSPICIOUS}}` before you approve the dataset.
  - **Manual review** provides a spreadsheet-like editor where you can tweak titles/bodies/labels, include or drop rows, and then commit the curated dataset (with a success summary and hints about expected precision/recall shifts).
  - **Snapshots & imports** let you save immutable dataset hashes/configs for provenance, reload past snapshots, or bring in labeled CSVs (validated for schema, length, empties, duplicates, and lint results).
- **Nerd Mode extras.**
  - An “Advanced knobs” expander exposes controls for suspicious link density, suspicious TLD frequency, caps intensity, money mentions, attachment lure mix, random seeds, label noise %, and a poisoning demo flag.
  - Additional diagnostics compare dataset summaries, explain configuration changes, and show lint status badges.

### 🧠 Train
- **Normal mode highlights.**
  - Pre-flight checks ensure you have enough balanced examples and recap the EU AI Act concept of inference.
  - The primary **Train model** button fits a `HybridEmbedFeatsLogReg` classifier that blends MiniLM sentence embeddings with interpretable numeric cues.
  - Post-training storyboards narrate what just happened, including a meaning map of the training data, a paraphrase similarity demo, and pointers to move on to evaluation.
- **Nerd Mode extras.**
  - Advanced panels expose train/test split sliders, random seed, solver iterations, regularisation strength, and the numeric assist guardrail parameters (center, uncertainty band, blend weight, logit cap, strategy).
  - Diagnostics surface centroid distances, language-mix chips (when `langdetect` is available), numeric guardrail audit stats, coefficient tables with plain-language explanations, adjustable sliders for each numeric cue, and prototype/nearest-neighbour explorers.
  - Experimental interpretability widgets (span influence, token highlights) and embedding caches are wired in for deeper analysis.

### 🧪 Evaluate
- **Normal mode highlights.**
  - Narrative panels interpret accuracy, confusion matrix counts, and trends compared to your last run (including dataset delta stories).
  - Threshold controls offer one-click presets (balanced F1, ≥95% precision, ≥90% recall), a fine-grained slider, and an “Adopt this threshold” action that feeds the chosen value into Use/Model Card stages.
- **Nerd Mode extras.**
  - Per-class precision/recall/F1 tables, precision–recall vs. threshold plots, interpretability summaries (top coefficients or numeric weights), and governance metadata (split sizes, seed, timestamp, adopted threshold) underpin audit conversations.

### 📬 Use
- **Normal mode highlights.**
  - Autonomy toggles let you run in recommendation-only or high-autonomy modes while processing batches of incoming emails (with a quick preview grid).
  - Results tables show predictions, confidence, and actions; mailboxes display what landed in Inbox vs. Spam, and download buttons export CSV/JSON summaries.
- **Nerd Mode extras.**
  - Diagnostics list raw probabilities, batch aggregates, histogram of spam scores, and—per email—nearest training examples, numeric feature contributions, optional token-removal highlights, and the session audit trail.
  - Adaptiveness mode adds confirmation/correction buttons, appends feedback to the labeled dataset, and offers an inline retrain that respects your guardrail settings.

### 📄 Model Card
- **Normal mode highlights.**
  - A ready-to-download Markdown model card documents purpose, algorithm, feature set, metrics, autonomy/adaptiveness choices, dataset snapshot ID, configuration JSON, and AI Act mapping points.
  - Highlight cards emphasise dataset size, hold-out count, autonomy level, and adaptiveness state for quick reporting.

## Key capabilities
- **Rich synthetic seed + configurable generator** with guardrails for spam share, edge cases, links, TLDs, urgency, attachments, and optional noise/poisoning demos.
- **Personal data hygiene workflow** that detects IBANs, card numbers, phone/email/OTP strings, and suspicious URLs, then guides replacements before committing datasets.
- **Hybrid MiniLM + numeric feature model** with adjustable numeric assist guardrails, coefficient sliders, and audit stats explaining when the hybrid kicks in.
- **Meaning map & interpretability suite** featuring embedding visualisations, paraphrase demos, nearest neighbours, numeric contributions, and experimental token influence analysis.
- **Threshold management & autonomy controls** spanning evaluation presets, adoption buttons, high-autonomy routing, and downloadable batch artefacts.
- **Adaptiveness & auditability** through feedback capture, in-app retraining, dataset snapshots, session audit logs, and a comprehensive model card.

## Running locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

> **Tip:** The first training run downloads MiniLM sentence-transformer weights; keep the virtual environment around to avoid re-downloading.

### Streamlit navigation bar requirements

The `streamlit-navigation-bar` component used by the app expects the following runtime environment:

- **Python 3.8 or newer.** The project is developed against Python 3.12, so any supported interpreter above 3.8 works.
- **Streamlit 1.33 or newer.** The `requirements.txt` file pins Streamlit at 1.38+, satisfying the component’s minimum version.
- **`streamlit-theme` 1.2.3 or newer.** This dependency ships the theme utilities that the navigation bar relies on.
- **Browser support for the CSS `:has()` pseudo-class.** The navigation bar styling tweaks depend on `:has()`, so use a modern browser (recent Chromium, Firefox, or Safari releases) that implements it.

## Data & privacy notes
- Seed data, generated previews, committed datasets, and audit logs live only in Streamlit session state — there is no server-side persistence.
- CSV imports must include `title`, `body`, and `label` (`spam`/`safe`), stay under 2,000 rows, and pass lint checks; invalid labels, long/empty rows, and duplicates are dropped automatically.
- PII cleanup tokens (e.g., `{{EMAIL}}`, `{{PHONE}}`, `{{URL_SUSPICIOUS}}`) help redact personal data while preserving model-relevant structure.
- Dataset snapshots are stored per session; export the model card or batch results if you need long-term provenance.

## License
See [LICENSE](LICENSE) for details.
