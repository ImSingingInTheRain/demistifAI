# demistifAI Agent Guide

Welcome! This repository contains the **demistifAI** Streamlit lab that walks users through the EU AI Act framing while they build and operate a spam detector. This guide captures the practices that keep the experience coherent and maintainable. It applies to the entire repository.

## Architectural quick facts
- [`streamlit_app.py`](./streamlit_app.py) is the Streamlit entry point. It wires the stage registry (`STAGES` in [`demistifai/constants.py`](./demistifai/constants.py)) to the renderer map in `STAGE_RENDERERS` and imports the shared UI primitives from `demistifai/ui`.
- Stage-specific views now live in the [`pages/`](./pages) package (for example `pages/data/page.py`, `pages/train_stage/page.py`, `pages/evaluate.py`; the package re-exports `render_train_stage_page`). Keep new stage logic in this package and surface it through the renderer map.
- Core utilities are organised under [`demistifai/core`](./demistifai/core): navigation and session management (`navigation.py`, `state.py`, `session_defaults.py`), dataset health (`dataset.py`, `validation.py`), guardrails (`guardrails.py`, `pii.py`), routing (`routing.py`), and download/export helpers.
- Modeling, dataset generation, and simulated inbox traffic remain in [`demistifai/modeling.py`](./demistifai/modeling.py), [`demistifai/dataset.py`](./demistifai/dataset.py), and [`demistifai/incoming_generator.py`](./demistifai/incoming_generator.py). Extend these modules rather than duplicating logic in pages.
- Visual primitives (mailbox tables, Nerd Mode toggles, EU AI quotes, section wrappers) live in [`demistifai/ui/primitives`](./demistifai/ui/primitives). Compose new UI from these helpers to keep styling consistent.
- Stage boundaries and component placements are documented in [`docs/stage_component_reference.md`](./docs/stage_component_reference.md). Update it when you move stages, shuffle key components, or adjust line references.

## Streamlit & UI practices
- Wrap multi-element panels in `with section_surface(...):` (from `demistifai.ui.primitives.sections`) so they inherit consistent spacing, headings, and Nerd Mode treatments.
- Reuse `render_nerd_mode_toggle`, `guidance_popover`, and the other primitives instead of recreating bespoke Streamlit widgets. The `pages/` modules demonstrate the current composition patterns.
- For global navigation tweaks, modify `demistifai/ui/custom_header.py` and the helpers in `demistifai/core/navigation.py` so the header buttons, URL state, and `STAGES` metadata stay in sync.
- Keep copy changes regulation-aware. When altering text that interprets the EU AI Act, cross-check the narrative in [`README.md`](./README.md) and the stage walkthroughs to keep tone and terminology consistent.
- Avoid mixing raw HTML unless necessary. If you must, sanitize user-provided content (e.g., with `html.escape`) as done in the existing inbox and guidance surfaces.
- Use `st.session_state` (`ss = st.session_state`) for cross-stage state. Reuse or update the keys defined in `demistifai/core/session_defaults.py` to prevent desynchronisation between stages.
- When introducing new stages or major cards, update `STAGES`, `STAGE_BY_KEY`, and the header controls so navigation, progress indicators, and keyboard shortcuts remain correct.

## Data & modeling guidelines
- Reuse the dataset helpers (`lint_dataset`, `compute_dataset_summary`, `generate_labeled_dataset`, etc.) from `demistifai/dataset.py` and `demistifai/core/dataset.py` rather than reimplementing validation or metrics in the UI.
- Respect the existing optional dependency guards in `demistifai/modeling.py` and `demistifai/core/language.py`. When adding heavy libraries (e.g., `sentence-transformers`, Plotly), wrap imports in try/except blocks that expose availability flags.
- `HybridEmbedFeatsLogReg` combines embedding and numeric features. Extend or modify it within `demistifai/modeling.py` so feature engineering stays testable outside Streamlit callbacks.
- Keep feature display labels (`FEATURE_DISPLAY_NAMES`, `FEATURE_PLAIN_LANGUAGE`) in sync with any Nerd Mode tables or explanations you add to pages.
- Dataset schemas require `title`, `body`, and `label` (`spam`/`safe`). Update the README, dataset linting, and validation helpers if this changes.

## Testing & QA expectations
- Automated tests are not yet configured. Run `python -m compileall streamlit_app.py demistifai pages` before committing to catch syntax errors across the main entry point, package, and stage modules.
- For functional changes, smoke test with `streamlit run streamlit_app.py` and walk through the stage(s) you touched. Capture screenshots for meaningful visual changes.
- When using the screenshot workflow, wait at least 30 seconds after opening the app before capturing so the interface can finish loading.
- The first call into the MiniLM encoder (used via `sentence-transformers`) may trigger a large download. Expect cache reuse on subsequent runs.

## Documentation & release notes
- Reflect new capabilities or workflow shifts in [`README.md`](./README.md) (especially the stage-by-stage walkthrough) when they materially change user experience.
- Keep terminology (e.g., "High autonomy", "Nerd Mode", "Control Room") consistent across UI strings, README, and any new docs you add.
- Follow repository-wide communication norms: concise commit messages and PR descriptions that summarise user-facing impact plus any testing performed.
- Update [`docs/stage_component_reference.md`](./docs/stage_component_reference.md) whenever stage line ranges change or components move to new files, and call out the refresh in your PR summary.

Following these conventions keeps the guided lab cohesive and saves future you from UI or state-management surprises. Happy shipping!
