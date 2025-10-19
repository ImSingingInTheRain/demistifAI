# demistifai package authoring notes

These conventions apply to all Python modules under `demistifai/` unless an AGENTS.md deeper in the tree overrides them.

## General structure
- Keep modules importable without Streamlit. Delay optional imports (for example, `sentence_transformers`) until runtime and guard them with availability checks rather than try/except around the module import itself.
- Prefer extending existing subpackages (for example, `core/`, `ui/`, `metrics/`) instead of adding new top-level modules. Introduce new folders only when multiple modules share a coherent theme.
- Centralise cross-stage state helpers in `core/` and reuse constants from `constants.py` to prevent drift between the entry point and the page packages.

## Coding style
- Match the repository's prevailing style: module-level functions first, followed by classes, then private helpers.
- Type annotate public functions and dataclasses. Use `typing.Protocol` when adding pluggable behaviours (e.g., model adapters) so Streamlit callbacks stay discoverable by static tooling.
- Keep logging routed through the `logging` module; configure it in `core/logging.py` rather than ad-hoc `print` statements.

## Documentation & tests
- Update docstrings when adding parameters or return types; surface user-facing narrative in the README or stage docs if behaviour changes meaningfully.
- When you touch logic that pages rely on, run `python -m compileall streamlit_app.py demistifai pages` before committing to catch syntax issues across the app entry points.
