# demistifAI Stage & Component Reference

This document anchors the stage layout inside `streamlit_app.py` and where to find related helpers across the repository. Line
numbers use `nl -ba` numbering (1-indexed) for quick cross-checks.

## Stage boundaries in `streamlit_app.py`
| Stage key | Function | Line range | Notes |
| --- | --- | --- | --- |
| `intro` | `render_intro_stage` | 410–412 | Wrapper that delegates to `welcome.render_intro_stage`; see `welcome.py` for the full UI. |
| `overview` | `render_overview_stage` | 415–420 | Wrapper that forwards to `stages.overview.render_overview_stage` for the full UI. |
| `data` | `render_data_stage` | 424–428 | Wrapper that delegates to `pages/data.render_data_stage` where the full Prepare UI now lives. |
| `evaluate` | `render_evaluate_stage` | 431–944 | Evaluation metrics, ROC / confusion matrix views, and governance summary. |
| `classify` | `render_classify_stage` | 947–1404 | Live classification console, governance tools, and routing copy. |
| `model_card` | `render_model_card_stage` | 1406–1510 | Transparency summary, dataset snapshot details, and download affordances. |
| `train` | `render_train_stage` | 1511–1519 | Delegates to `stages/train_stage.render_train_stage_page` where the full UI lives. |

> **Tip:** Re-run `nl -ba streamlit_app.py | sed -n 'START,ENDp'` after edits to confirm updated line ranges.

## Stage implementation outside the main app
| Stage key | File | Line range | Purpose |
| --- | --- | --- | --- |
| `intro` | `welcome.py` | 22–336 | Full intro stage UI including lifecycle hero, EU AI Act framing, and launch controls. |
| `overview` | `stages/overview.py` | 25–753 | Stage Control Room with EU AI Act framing, system snapshot/status, and mission walkthrough of the pipeline. |
| `data` | `pages/data.py` | 84–1849 | Full Prepare stage UI covering dataset builder, linting feedback, PII cleanup, diagnostics, and CSV workflows. |
| `train` | `stages/train_stage.py` | 116–2214 | Full training UI, entrypoint wrapper, nerd mode tooling, interpretability widgets, and background tasks. |

Supporting helpers for training live alongside the stage:
- `stages/train_helpers.py` – shared callbacks and utilities for the training workflow.
- `stages/__init__.py` – exposes training exports to the Streamlit entry point.

## Component placement map
- **`streamlit_app.py`** – Entry point; orchestrates layout, stage switching, and cross-stage session state.
- **`pages/data.py`** – Prepare/Data stage UI, including dataset builder, PII cleanup, and diagnostics tooling.
- **`demistifai/constants.py`** – Stage metadata, icons, copy blocks, and shared CSS snippets.
- **`demistifai/dataset.py`** – Dataset generation, CSV I/O, and linting utilities for Prepare/Data stages.
- **`demistifai/modeling.py`** – Feature engineering, model training, calibration, and interpretability helpers.
- **`demistifai/incoming_generator.py`** – Synthetic incoming email batches used in the Classify stage.
- **`stages/overview.py`** – Overview stage UI, system snapshot, and mission briefing components.
- **`stages/train_stage.py` & `stages/train_helpers.py`** – Dedicated training UI and supporting logic.

## Maintenance expectations
- When stage content moves or grows, update the line ranges above so future contributors land in the right place.
- If you relocate components (e.g., move dataset helpers to another module), revise the placement map accordingly.
- Mention updates to this reference in your PR summaries when the stage flow or component layout shifts.
