"""Session-state helper for the intro terminal component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import streamlit as st

SessionMapping = MutableMapping[str, Any]


@dataclass
class IntroTerminalSession:
    """Wrap the Streamlit session keys used by the intro terminal."""

    command_key: str
    component_key: str = "intro_inline_terminal"
    state: SessionMapping[str, Any] = field(default_factory=lambda: st.session_state)

    def __post_init__(self) -> None:
        self._lines_key = f"{self.command_key}_lines"
        self._lines_signature_key = f"{self.command_key}_lines_signature"
        self._ready_key = f"{self.command_key}_ready"
        self._ready_at_key = f"{self.command_key}_ready_at"
        self._append_pending_key = f"{self.command_key}_append_pending"
        self._prefill_line_count_key = f"{self.command_key}_prefill_line_count"
        self._preserve_state_key = f"{self.command_key}_preserve_state"
        self._keep_input_active_key = f"{self.command_key}_keep_input_active"

    def ensure_lines(self, default_lines: Sequence[str]) -> List[str]:
        """Return the intro terminal lines, initialising them when absent."""

        lines = self.state.get(self._lines_key)
        if not isinstance(lines, list) or not lines:
            lines = list(default_lines)
            self.state[self._lines_key] = lines
        return lines

    @property
    def lines_signature(self) -> Optional[Tuple[str, ...]]:
        signature = self.state.get(self._lines_signature_key)
        if signature is None:
            return None
        if isinstance(signature, tuple):
            return signature
        if isinstance(signature, list):
            normalised = tuple(str(item) for item in signature)
            self.state[self._lines_signature_key] = normalised
            return normalised
        return None

    def set_lines_signature(self, signature: Sequence[str]) -> None:
        normalised_signature = tuple(str(item) for item in signature)
        self.state[self._lines_signature_key] = normalised_signature

    def clear_lines_signature(self) -> None:
        self.state.pop(self._lines_signature_key, None)

    @property
    def append_pending(self) -> bool:
        return bool(self.state.get(self._append_pending_key, False))

    def mark_append_pending(self) -> None:
        self.state[self._append_pending_key] = True

    def clear_append_pending(self) -> None:
        self.state.pop(self._append_pending_key, None)

    def append_lines(
        self,
        new_lines: Sequence[str],
        *,
        prefill_line_count: Optional[int] = None,
        keep_input_active: bool = False,
    ) -> None:
        """Extend the rendered lines and flag that a replay is pending."""

        if not new_lines:
            return
        lines = self.state.get(self._lines_key)
        if not isinstance(lines, list):
            lines = []
            self.state[self._lines_key] = lines
        lines.extend(list(new_lines))
        self.mark_append_pending()
        self.clear_lines_signature()
        if prefill_line_count is not None:
            self.set_prefilled_line_count(prefill_line_count)
        if keep_input_active:
            self.request_input_focus()

    @property
    def ready(self) -> bool:
        return bool(self.state.get(self._ready_key, False))

    def set_ready(self, ready: bool) -> None:
        self.state[self._ready_key] = bool(ready)

    @property
    def ready_deadline(self) -> Optional[float]:
        value = self.state.get(self._ready_at_key)
        if isinstance(value, (float, int)):
            return float(value)
        return None

    def set_ready_deadline(self, timestamp: float) -> None:
        self.state[self._ready_at_key] = float(timestamp)

    def clear_ready_deadline(self) -> None:
        self.state.pop(self._ready_at_key, None)

    @property
    def input_text(self) -> str:
        value = self.state.get(self.command_key, "")
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    def set_input_text(self, text: str) -> None:
        self.state[self.command_key] = text

    def clear_input_text(self) -> None:
        self.state[self.command_key] = ""

    @property
    def prefilled_line_count(self) -> int:
        value = self.state.get(self._prefill_line_count_key)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def set_prefilled_line_count(self, count: int) -> None:
        self.state[self._prefill_line_count_key] = int(count)

    def clear_prefilled_line_count(self) -> None:
        self.state.pop(self._prefill_line_count_key, None)

    def preserve_input(self, text: Optional[str] = None) -> None:
        """Persist the current input text so reruns keep the typed command."""

        if text is None:
            text = self.input_text
        self.set_input_text(text)
        self.state[self._preserve_state_key] = True
        self.normalise_component_payload()

    def request_input_focus(self) -> None:
        """Flag that the terminal input should remain focused."""

        self.state[self._keep_input_active_key] = True
        self.normalise_component_payload()

    def consume_input_focus_request(self) -> bool:
        """Return whether input focus should be preserved for the next render."""

        keep_active = bool(self.state.get(self._keep_input_active_key, False))
        if keep_active:
            self.state[self._keep_input_active_key] = False
        else:
            self.state.pop(self._keep_input_active_key, None)
        self.normalise_component_payload()
        return keep_active

    def consume_preserve_flag(self) -> bool:
        """Return whether input preservation is requested and clear the flag."""

        preserve_flag = bool(self.state.get(self._preserve_state_key, False))
        if preserve_flag:
            self.state[self._preserve_state_key] = False
        else:
            self.state.pop(self._preserve_state_key, None)
        return preserve_flag

    def reset_for_animation(
        self,
        *,
        drop_lines: bool = False,
        clear_input: bool = True,
        keep_input_active: bool = False,
        preserve_ready: bool = False,
    ) -> None:
        """Prepare the session so the intro animation replays on the next run."""

        preserve_text = self.consume_preserve_flag()
        if drop_lines:
            self.state.pop(self._lines_key, None)
            self.clear_prefilled_line_count()
        self.clear_lines_signature()
        self.clear_ready_deadline()
        if keep_input_active:
            self.request_input_focus()
        else:
            self.state.pop(self._keep_input_active_key, None)
        if preserve_ready:
            ready_value = self.ready
        else:
            ready_value = False
            self.set_ready(False)
        if clear_input and not preserve_text:
            self.clear_input_text()
        if drop_lines:
            self.clear_append_pending()
        self.normalise_component_payload(ready=ready_value)

    def normalise_component_payload(self, *, ready: Optional[bool] = None) -> None:
        """Normalise the embedded component payload for the next rerun."""

        component_state = self.state.get(self.component_key)
        new_state: Dict[str, Any] = {}
        if isinstance(component_state, dict):
            new_state.update(component_state)
        payload: Dict[str, Any] = {}
        if isinstance(new_state.get("value"), dict):
            payload.update(new_state["value"])
        payload["text"] = self.input_text
        payload["ready"] = self.ready if ready is None else bool(ready)
        payload["submitted"] = False
        payload["keepInput"] = bool(
            self.state.get(self._keep_input_active_key, False)
        )
        new_state["value"] = payload
        self.state[self.component_key] = new_state

