"""Interactive terminal Streamlit component registration."""

from __future__ import annotations

import atexit
import os
from contextlib import ExitStack
from importlib import resources
from pathlib import Path
from typing import Any, Optional, Sequence

from streamlit.components.v1 import declare_component

from .shared_renderer import TerminalRenderBundle, build_terminal_render_bundle

_COMPONENT_NAME = "interactive_terminal_component"
_DEV_COMPONENT_URL_ENV_VAR = "DEMISTIFAI_TERMINAL_COMPONENT_DEV_URL"
_RESOURCE_STACK = ExitStack()
atexit.register(_RESOURCE_STACK.close)


def _resolve_component_path() -> Path:
    """Return the path to the packaged frontend assets.

    Using ``importlib.resources`` keeps the lookup compatible with packaged
    distributions where the module location may not be directly on disk.
    """

    frontend_resource = resources.files(__package__).joinpath("frontend")
    try:
        extracted_path = _RESOURCE_STACK.enter_context(
            resources.as_file(frontend_resource)
        )
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("Interactive terminal frontend assets are unavailable") from exc

    resolved_path = Path(extracted_path)
    if not resolved_path.exists():  # pragma: no cover - defensive guard
        raise RuntimeError(
            f"Interactive terminal frontend missing at {resolved_path!s}"
        )
    return resolved_path


_COMPONENT_DIR = _resolve_component_path()

_DEV_COMPONENT_URL = os.getenv(_DEV_COMPONENT_URL_ENV_VAR)
if _DEV_COMPONENT_URL:
    _interactive_terminal = declare_component(
        _COMPONENT_NAME,
        url=_DEV_COMPONENT_URL,
    )
else:
    _interactive_terminal = declare_component(
        _COMPONENT_NAME,
        path=str(_COMPONENT_DIR),
    )


def render_interactive_terminal(
    *,
    suffix: str,
    lines: Sequence[str],
    speed_type_ms: int,
    speed_delete_ms: int,
    pause_between_ops_ms: int,
    key: str,
    placeholder: str = "",
    accept_keystrokes: bool = False,
    show_caret: bool = True,
    secondary_inputs: Optional[Sequence[str]] = None,
) -> Optional[dict[str, Any]]:
    """Render the interactive terminal custom component.

    Returns the latest interaction payload ``{"text": str, "submitted": bool, "ready": bool}``
    when Streamlit provides component values, otherwise ``None``.
    """

    placeholder_text = str(placeholder or "")
    aria_label = placeholder_text.strip() or "Interactive terminal input"
    bundle: TerminalRenderBundle = build_terminal_render_bundle(
        suffix=suffix,
        lines=list(lines),
        speed_type_ms=speed_type_ms,
        speed_delete_ms=speed_delete_ms,
        pause_between_ops_ms=pause_between_ops_ms,
        key=key,
        show_caret=show_caret,
        placeholder=placeholder_text,
        input_aria_label=aria_label,
        accept_keystrokes=accept_keystrokes,
        secondary_inputs=secondary_inputs,
    )

    typing_config = {
        "speedType": bundle.payload["speedType"],
        "speedDelete": bundle.payload["speedDelete"],
        "pauseBetween": bundle.payload["pauseBetween"],
    }

    component_value = _interactive_terminal(
        key=key,
        markup=bundle.markup,
        payload=bundle.payload,
        serializedLines=bundle.serializable_segments,
        typingConfig=typing_config,
        placeholder=placeholder_text,
        acceptKeystrokes=bool(accept_keystrokes),
    )

    if isinstance(component_value, dict):
        value = component_value.get("value")
        if isinstance(value, dict):
            text_value = value.get("text")
            submitted_value = value.get("submitted")
            ready_value = value.get("ready")
            return {
                "text": "" if text_value is None else str(text_value),
                "submitted": bool(submitted_value),
                "ready": bool(ready_value),
            }

    return None


__all__ = ["render_interactive_terminal"]
