# demistifAI Agent Guide

Welcome! This repository contains the **demistifAI** Streamlit lab that walks users through the EU AI Act framing while they build and operate a spam detector. This guide captures the practices that keep the experience coherent and maintainable. It applies to the entire repository.

## Architectural quick facts
- The Streamlit entry point is [`streamlit_app.py`](./streamlit_app.py). Most UI surfaces live there in dedicated helper functions (e.g., `render_*_stage`) that map directly to the lifecycle stages defined in [`demistifai/constants.py`](./demistifai/constants.py).
- Stage boundaries and a component placement map live in [`docs/stage_component_reference.md`](./docs/stage_component_reference.md); keep it current when stages move or code is reorganised.
- Core data, modeling, and simulation helpers live in the [`demistifai`](./demistifai) package:
  - `dataset.py` owns labeled data generation, linting, and CSV import/export helpers.
  - `modeling.py` contains feature engineering, training, calibration, and interpretability utilities.
  - `incoming_generator.py` produces synthetic email batches for the "Use" stage.
- UI polish (CSS, iconography, stage metadata) is centralized in `constants.py`; reuse or extend these constants instead of hard-coding copies in the app.

## Streamlit & UI practices
- Prefer composing new UI sections with the existing helpers:
  - Wrap multi-element panels in `with section_surface(...):` to inherit consistent styling.
  - Use `render_nerd_mode_toggle` patterns when adding advanced controls so they slot into the Nerd Mode experience.
- Keep copy changes regulation-aware. When altering text that interprets the EU AI Act, cross-check the README narrative and keep tone consistent (mission brief, stage framing, etc.).
- Avoid mixing raw HTML unless necessary; when you must use HTML, sanitize user-provided content (e.g., `html.escape`) just as the current code does.
- Leverage `st.session_state` (`ss = st.session_state`) for cross-stage state. Reuse existing keys when possible to prevent desynchronisation between stages.
- When introducing new stages or major cards, update `STAGES`, `STAGE_BY_KEY`, and related metadata in `constants.py` so navigation and progress indicators stay accurate.

## Data & modeling guidelines
- Reuse dataset utilities (`lint_dataset`, `compute_dataset_summary`, `generate_labeled_dataset`, etc.) rather than reimplementing validation or statistics in the UI. This keeps business rules in one place.
- Respect the existing optional dependency guards. When adding libraries like `sentence-transformers`, wrap imports in a try/except block that sets an availability flag instead of failing hard at import time.
- `HybridEmbedFeatsLogReg` combines embedding and numeric features; extend or modify it within `modeling.py` so feature engineering remains testable outside of Streamlit callbacks.
- When adding numeric cues or feature weight adjustments, keep display labels in sync with `FEATURE_DISPLAY_NAMES` / `FEATURE_PLAIN_LANGUAGE` to avoid mismatches in Nerd Mode tables.
- Dataset schemas require `title`, `body`, and `label` (`spam`/`safe`). Update the README and any validation helpers if this changes.

## Testing & QA expectations
- Automated tests are not yet configured. At a minimum run `python -m compileall streamlit_app.py demistifai` before committing to catch syntax errors.
- For functional changes, smoke test with `streamlit run streamlit_app.py` and walk through the stage(s) you touched. Capture screenshots for meaningful visual changes.
- Watch for large downloads the first time the sentence-transformer encoder is used; cache reuse is expected during local testing.

## Documentation & release notes
- Reflect new capabilities or workflow shifts in the README (especially in the stage-by-stage walkthrough) when they materially change user experience.
- Keep terminology (e.g., "High autonomy", "Nerd Mode") consistent across UI strings, README, and any new docs you add.
- Follow repository-wide communication norms: concise commit messages, and PR descriptions that summarise user-facing impact plus any testing performed.
- Update [`docs/stage_component_reference.md`](./docs/stage_component_reference.md) whenever stage line ranges change or components move to new files, and call out the refresh in your PR summary.

Following these conventions keeps the guided lab cohesive and saves future you from UI or state-management surprises. Happy shipping!
