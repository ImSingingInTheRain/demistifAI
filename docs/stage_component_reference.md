# demistifAI Stage & Component Reference

This document anchors the stage layout inside `streamlit_app.py` and where to find related helpers across the repository. Line
numbers use `nl -ba` numbering (1-indexed) for quick cross-checks.

## Stage boundaries in `streamlit_app.py`
| Stage key | Function | Line range | Notes |
| --- | --- | --- | --- |
| `intro` | `render_intro_stage` | 1240–1714 | Animated title, refreshed hero, "Why demAI" carousel, EU AI Act spotlight, lifecycle primer, and call-to-action buttons. |
| `overview` | `render_overview_stage` | 1715–2410 | EU AI Act framing, autonomy sliders, and walkthrough of the full pipeline. |
| `data` | `render_data_stage` | 2411–4179 | Dataset import/generation, linting feedback, and label review utilities. |
| `evaluate` | `render_evaluate_stage` | 4180–4477 | Evaluation metrics, ROC / confusion matrix views, and governance summary. |
| `classify` | `render_classify_stage` | 4478–4927 | Live classification console, governance tools, and routing copy. |
| `model_card` | `render_model_card_stage` | 4928–5081 | Transparency summary, dataset snapshot details, and download affordances. |
| `train` | `_render_train_stage_wrapper` | 5082–5093 | Streamlit-side wrapper; full UI lives in `stages/train_stage.py`. |

> **Tip:** Re-run `nl -ba streamlit_app.py | sed -n 'START,ENDp'` after edits to confirm updated line ranges.

## Stage implementation outside the main app
| Stage key | File | Line range | Purpose |
| --- | --- | --- | --- |
| `train` | `stages/train_stage.py` | 167–1867 | Full training UI, nerd mode tooling, interpretability widgets, and background tasks. |

Supporting helpers for training live alongside the stage:
- `stages/train_helpers.py` – shared callbacks and utilities for the training workflow.
- `stages/__init__.py` – exposes training exports to the Streamlit entry point.

## Component placement map
- **`streamlit_app.py`** – Entry point; orchestrates layout, stage switching, and cross-stage session state.
- **`demistifai/constants.py`** – Stage metadata, icons, copy blocks, and shared CSS snippets.
- **`demistifai/dataset.py`** – Dataset generation, CSV I/O, and linting utilities for Prepare/Data stages.
- **`demistifai/modeling.py`** – Feature engineering, model training, calibration, and interpretability helpers.
- **`demistifai/incoming_generator.py`** – Synthetic incoming email batches used in the Classify stage.
- **`stages/train_stage.py` & `stages/train_helpers.py`** – Dedicated training UI and supporting logic.

## Maintenance expectations
- When stage content moves or grows, update the line ranges above so future contributors land in the right place.
- If you relocate components (e.g., move dataset helpers to another module), revise the placement map accordingly.
- Mention updates to this reference in your PR summaries when the stage flow or component layout shifts.
