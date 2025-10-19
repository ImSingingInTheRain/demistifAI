# Train Stage Author Notes

This guide covers the `pages/train_stage/` package. Follow it whenever you touch files under this directory tree.

## Page orchestration
- `render_train_stage_page` in `page.py` bootstraps core state (`ensure_state`, `validate_invariants`) and fans out to the package modules. Keep the entry point lean: surface-level layout belongs in `page.py`, but callbacks, panels, and visualisations must stay in their dedicated modules.
- The render flow is intentionally modular:
  - **Callbacks** (`callbacks.py`) keep Streamlit command wiring isolated; request/resume logic should land here.
  - **Panels** (`panels.py`) owns launchpad and Nerd Mode surfaces; avoid reimplementing UI scaffolding inside `page.py`.
  - **Results** (`results.py`) owns post-training surfaces, including rerun triggers.
  - **Visualisations** (`visualizations.py`) encapsulate chart-building utilities (e.g., calibration, meaning map); expose only finished Streamlit-ready components to the page layer.
  - Any new surface should extend the relevant module and be imported by `page.py` similar to existing helpers.

## Session-state expectations (`state.py`)
- Reuse the utilities in `state.py` rather than mutating `st.session_state` manually. They document the canonical keys and keep rerun logic predictable.
- `ensure_train_stage_state(ss, threshold=…)` must be called before reading guardrail or token-budget settings; it seeds keys such as `train_in_progress`, `train_refresh_expected`, `train_params`, and the guard-parameter bundle (`assist_center_mode`, `uncertainty_band`, etc.).
- Guardrail parameters live under `ss["guard_params"]`; modify them through `ensure_train_stage_state` / `persist_training_outcome` to keep defaults in sync.
- `guard_params` are consumed by `train_model_on_split` and may be auto-adjusted by `auto_select_assist_center`. Never handcraft parallel structures in the page layer.
- Use `compute_token_budget_summary` to derive token hints. It depends on the cache keyed by dataset hash; do not bypass it with ad-hoc calculations.
- `parse_train_params` is the canonical normaliser for `ss["train_params"]`. Call it before invoking the training pipeline.
- Training refresh bookkeeping relies on:
  - `register_training_refresh` — marks completion (`train_story_run_id`, `train_flash_finished`, `train_refresh_expected`, `eval_timestamp`, `eval_temp_threshold`).
  - `reset_meaning_map_flags` — clears meaning-map toggles defined in `MEANING_MAP_STATE_KEYS` so charts recompute.
  - `guard_params` updates happen via `persist_training_outcome`, which also stores the latest model and split cache.
- Meaning map flags, guardrail state, and storyboard cues are shared across modules. Touch the keys defined in `state.py` only through these helpers.

## Training pipeline flow
- The happy path is: `execute_training_pipeline` → `persist_training_outcome` → `register_training_refresh` → `reset_meaning_map_flags` → `render_training_results`.
- `execute_training_pipeline` wraps split generation, model fitting, guardrail auto-tuning, and embedding cache warm-up. Pass in `parse_train_params` output plus the guard-params dict from `ensure_train_stage_state`.
- After a successful run, always persist via `persist_training_outcome`; downstream panels, callbacks, and results expect `ss["model"]`, `ss["split_cache"]`, and `ss["guard_params"]` to exist.
- Call `register_training_refresh` immediately after persisting so metrics panels and meaning-map requests know to refresh.
- Reset the meaning-map toggles via `reset_meaning_map_flags` to ensure the visualization modules rebuild when the user returns to Nerd Mode.
- Finally delegate to `render_training_results` for the post-training surfaces. It owns rerun triggers and refresh countdowns; do not bypass it from the page.

## Helper packages
- Reach into `helpers/` (meaning map, guardrail, storyboard) instead of reimplementing logic inline. `page.py` already imports:
  - `helpers.meaning_map` for preparation, zooming, and chart-building.
  - `helpers.storyboard` for training example previews and unified storyboard rendering.
- When adding new meaning-map or guardrail behaviours, extend the respective helper module so state management and memoisation stay centralised.
- Storyboard narratives should live in the storyboard helper; keep `page.py` focused on sequencing, not content management.
- Before introducing new helper code, check the existing packages to avoid duplicating features (for example, `helpers.guardrail` or `helpers.meaning_map`).

Following these patterns keeps Train-stage reruns, refresh triggers, and shared state predictable while letting the helper modules evolve independently.
