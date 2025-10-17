# demistifAI Stage & Component Reference

This document anchors the stage layout inside `streamlit_app.py` and where to find related helpers across the repository. Line
numbers use `nl -ba` numbering (1-indexed) for quick cross-checks.

## Stage boundaries in `streamlit_app.py`
| Stage key | Renderer registration | Line range | Notes |
| --- | --- | --- | --- |
| `intro` | `STAGE_RENDERERS["intro"]` | 87 | Partial wrapper that injects the shared `section_surface` before dispatching to `pages.welcome.render_intro_stage`. |
| `overview` | `STAGE_RENDERERS["overview"]` | 88–93 | Passes session state and shared helpers into `pages.overview.render_overview_stage`. |
| `data` | `STAGE_RENDERERS["data"]` | 94–98 | Provides the shared `section_surface`/`render_nerd_mode_toggle` helpers to `pages.data.render_data_stage`. |
| `train` | `STAGE_RENDERERS["train"]` | 99–107 | Supplies navigation callbacks, EU AI quote, and language mix helpers to `pages.train_stage.render_train_stage_page`. |
| `evaluate` | `STAGE_RENDERERS["evaluate"]` | 108–113 | Registers Evaluate with shared surfaces and copy shortening helpers. |
| `classify` | `STAGE_RENDERERS["classify"]` | 114–123 | Wires in session state, inbox table, mailbox panel, and adaptiveness synchronisation for `pages.use.render_classify_stage`. |
| `model_card` | `STAGE_RENDERERS["model_card"]` | 124–128 | Injects surface and guidance popover helpers for `pages.model_card.render_model_card_stage`. |

> **Tip:** Re-run `nl -ba streamlit_app.py | sed -n 'START,ENDp'` after edits to confirm updated line ranges.

## Stage implementation outside the main app
| Stage key | File | Line range | Purpose |
| --- | --- | --- | --- |
| `intro` | `pages/welcome.py` | 22–337 | Full intro stage UI including lifecycle hero, EU AI Act framing, and launch controls. |
| `overview` | `pages/overview.py` | 25–755 | Stage Control Room with EU AI Act framing, system snapshot/status, and mission walkthrough of the pipeline. |
| `data` | `pages/data.py` | 84–1849 | Full Prepare stage UI covering dataset builder, linting feedback, PII cleanup, diagnostics, and CSV workflows. |
| `train` | `pages/train_stage.py` | 117–2215 | Full training UI, entrypoint wrapper, nerd mode tooling, interpretability widgets, and background tasks. |
| `evaluate` | `pages/evaluate.py` | 44–566 | Evaluation metrics, ROC / confusion matrix views, and governance summary. |
| `classify` | `pages/use.py` | 37–489 | Live classification console, autonomy controls, adaptiveness, and routing copy. |
| `model_card` | `pages/model_card.py` | 21–141 | Transparency summary, dataset snapshot details, and download affordances. |

Supporting helpers for training live alongside the stage:
- `pages/train_helpers.py` – shared callbacks and utilities for the training workflow.
- `pages/__init__.py` – central exports for stage renderers consumed by `streamlit_app.py`.

## Component placement map
- **`streamlit_app.py`** – Entry point; orchestrates layout, stage switching, and cross-stage session state.
- **`demistifai/core/navigation.py`** – Synchronizes stage selection across state containers and exposes `activate_stage` for programmatic jumps.
- **`pages/data.py`** – Prepare/Data stage UI, including dataset builder, PII cleanup, and diagnostics tooling.
- **`pages/use.py`** – Use/Classify stage UI handling batch processing, autonomy, and adaptiveness workflows.
- **`demistifai/constants.py`** – Stage metadata, icons, copy blocks, and shared CSS snippets.
- **`demistifai/dataset.py`** – Dataset generation, CSV I/O, and linting utilities for Prepare/Data stages.
- **`demistifai/modeling.py`** – Feature engineering, model training, calibration, and interpretability helpers.
- **`demistifai/incoming_generator.py`** – Synthetic incoming email batches used in the Classify stage.
- **`pages/overview.py`** – Overview stage UI, system snapshot, and mission briefing components.
- **`pages/model_card.py`** – Model card transparency stage UI and download helpers.
- **`pages/train_stage.py` & `pages/train_helpers.py`** – Dedicated training UI and supporting logic.

## Maintenance expectations
- When stage content moves or grows, update the line ranges above so future contributors land in the right place.
- If you relocate components (e.g., move dataset helpers to another module), revise the placement map accordingly.
- Mention updates to this reference in your PR summaries when the stage flow or component layout shifts.
