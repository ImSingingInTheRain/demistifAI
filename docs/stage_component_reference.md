# demistifAI Stage & Component Reference

This document anchors the stage layout inside `streamlit_app.py` and where to find related helpers across the repository. Line
numbers use `nl -ba` numbering (1-indexed) for quick cross-checks.

## Stage boundaries in `streamlit_app.py`
| Stage key | Function | Line range | Notes |
| --- | --- | --- | --- |
| `intro` | `render_intro_stage` | 376–378 | Wrapper that delegates to `pages.welcome.render_intro_stage`; see `pages/welcome.py` for the full UI. |
| `overview` | `render_overview_stage` | 381–386 | Wrapper that forwards to `pages.overview.render_overview_stage` for the full UI. |
| `data` | `render_data_stage` | 390–394 | Wrapper that delegates to `pages/data.render_data_stage` where the full Prepare UI now lives. |
| `evaluate` | `render_evaluate_stage` | 397–402 | Wrapper delegating to `pages.evaluate.render_evaluate_stage_page`. |
| `classify` | `render_classify_stage` | 404–413 | Wrapper delegating to `pages/use.render_classify_stage` for the full Use console UI. |
| `model_card` | `render_model_card_stage` | 416–420 | Wrapper that calls `stages.model_card.render_model_card_stage` to render the transparency card experience. |
| `train` | `render_train_stage` | 423–431 | Delegates to `pages.train_stage.render_train_stage_page` where the full UI lives. |

> **Tip:** Re-run `nl -ba streamlit_app.py | sed -n 'START,ENDp'` after edits to confirm updated line ranges.

## Stage implementation outside the main app
| Stage key | File | Line range | Purpose |
| --- | --- | --- | --- |
| `intro` | `pages/welcome.py` | 22–336 | Full intro stage UI including lifecycle hero, EU AI Act framing, and launch controls. |
| `overview` | `pages/overview.py` | 25–753 | Stage Control Room with EU AI Act framing, system snapshot/status, and mission walkthrough of the pipeline. |
| `data` | `pages/data.py` | 84–1849 | Full Prepare stage UI covering dataset builder, linting feedback, PII cleanup, diagnostics, and CSV workflows. |
| `classify` | `pages/use.py` | 37–489 | Live classification console, autonomy controls, adaptiveness, and routing copy. |
| `evaluate` | `pages/evaluate.py` | 44–566 | Evaluation metrics, ROC / confusion matrix views, and governance summary. |
| `model_card` | `stages/model_card.py` | 21–141 | Transparency summary, dataset snapshot details, and download affordances. |
| `train` | `pages/train_stage.py` | 116–2214 | Full training UI, entrypoint wrapper, nerd mode tooling, interpretability widgets, and background tasks. |

Supporting helpers for training live alongside the stage:
- `pages/train_helpers.py` – shared callbacks and utilities for the training workflow.
- `pages/__init__.py` – exposes training exports to the Streamlit entry point.

## Component placement map
- **`streamlit_app.py`** – Entry point; orchestrates layout, stage switching, and cross-stage session state.
- **`pages/data.py`** – Prepare/Data stage UI, including dataset builder, PII cleanup, and diagnostics tooling.
- **`pages/use.py`** – Use/Classify stage UI handling batch processing, autonomy, and adaptiveness workflows.
- **`demistifai/constants.py`** – Stage metadata, icons, copy blocks, and shared CSS snippets.
- **`demistifai/dataset.py`** – Dataset generation, CSV I/O, and linting utilities for Prepare/Data stages.
- **`demistifai/modeling.py`** – Feature engineering, model training, calibration, and interpretability helpers.
- **`demistifai/incoming_generator.py`** – Synthetic incoming email batches used in the Classify stage.
- **`pages/overview.py`** – Overview stage UI, system snapshot, and mission briefing components.
- **`stages/model_card.py`** – Model card transparency stage UI and download helpers.
- **`pages/train_stage.py` & `pages/train_helpers.py`** – Dedicated training UI and supporting logic.

## Maintenance expectations
- When stage content moves or grows, update the line ranges above so future contributors land in the right place.
- If you relocate components (e.g., move dataset helpers to another module), revise the placement map accordingly.
- Mention updates to this reference in your PR summaries when the stage flow or component layout shifts.
