# demistifAI Stage & Component Reference

This document anchors the stage layout inside `streamlit_app.py` and where to find related helpers across the repository. Line
numbers use `nl -ba` numbering (1-indexed) for quick cross-checks.

## Stage boundaries in `streamlit_app.py`
| Stage key | Function | Line range | Notes |
| --- | --- | --- | --- |
| `intro` | `render_intro_stage` | 493–807 | Command grid with inline EU AI Act definition, lifecycle primer, and launch controls. |
| `overview` | `render_overview_stage` | 808–1461 | Stage Control Room with EU AI Act framing, system snapshot/status, and mission walkthrough of the pipeline. |
| `data` | `render_data_stage` | 1462–3184 | Dataset import/generation, linting feedback, and label review utilities. |
| `evaluate` | `render_evaluate_stage` | 3185–3618 | Evaluation metrics, ROC / confusion matrix views, and governance summary. |
| `classify` | `render_classify_stage` | 3619–4077 | Live classification console, governance tools, and routing copy. |
| `model_card` | `render_model_card_stage` | 4078–4182 | Transparency summary, dataset snapshot details, and download affordances. |
| `train` | `_render_train_stage_wrapper` | 4183–4254 | Streamlit-side wrapper; full UI lives in `stages/train_stage.py`. |

> **Tip:** Re-run `nl -ba streamlit_app.py | sed -n 'START,ENDp'` after edits to confirm updated line ranges.

## Stage implementation outside the main app
| Stage key | File | Line range | Purpose |
| --- | --- | --- | --- |
| `train` | `stages/train_stage.py` | 168–1871 | Full training UI, nerd mode tooling, interpretability widgets, and background tasks. |

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
