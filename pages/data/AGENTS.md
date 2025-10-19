# Data stage contributor notes

- `render_data_stage` owns the stage layout: it instantiates the stage grid scaffolding, flushes any pending flashes from `data_stage_flash_queue`, and routes control to the builder, review, and PII helpers so the cards stay coordinated.
- Mutate datasets only through `prepare_records` and `generate_preview_from_config` in `pages/data/dataset_io.py`. These helpers recompute hashes, summaries, and trigger `_invalidate_downstream`, so bypassing them leaves stale state behind.
- When builder or review callbacks fire, push short-lived notifications with `_push_data_stage_flash` (which feeds `data_stage_flash_queue`) and refresh the shared derived values—`dataset_summary`, `dataset_compare_delta`, and Nerd Mode toggles—so the UI matches the latest edits.
- Prefer extending the existing sections (`render_dataset_health_section`, `render_preview_and_commit`, `render_dataset_snapshot_section`, `render_csv_upload`) instead of adding bespoke Streamlit blocks; the shared panels already bundle styling, copy, and state wiring for the stage.
