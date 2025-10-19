# Pages package authoring notes

These guidelines apply to every module under `pages/`, unless a deeper `AGENTS.md` refines them.

## Page composition
- Keep Streamlit entry functions named `render_<stage>_stage` (or `render_<name>_page` for ancillary views) and export them from the package `__init__.py` so `streamlit_app.py` stays declarative.
- Each stage page should delegate data fetching, callbacks, and heavy computation to helpers in `demistifai/`. Restrict Streamlit calls to layout and user interaction wiring.
- Reuse shared UI primitives from `demistifai/ui/primitives` and `demistifai/ui/custom_header` to preserve styling consistency. Avoid inline HTML unless an existing helper does so.

## State management
- Interact with `st.session_state` through helpers in `demistifai/core/state.py` or the stage-specific state modules within `pages/<stage>/`. Do not introduce new global keys without documenting them next to their default initialisation.
- When adding callbacks, place them in a sibling `callbacks.py` (or extend the existing one) and call them from the page module to keep rerun behaviour predictable.

## Copy & documentation
- Align user-facing copy with the tone set in `README.md` and the existing stage descriptions. For regulatory references, mirror the phrasing already used in the app to stay accurate to the EU AI Act framing.
- Update any stage-specific README or docs when adding new steps or toggles that change the walkthrough.
