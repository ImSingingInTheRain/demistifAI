"""Regression tests for terminal segment serialisation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demistifai.ui.components.terminal import intro_terminal, train
from demistifai.ui.components.terminal.shared_renderer import (
    TerminalRenderBundle,
    build_terminal_render_bundle,
)


def _build_bundle(
    *,
    suffix: str,
    lines: Sequence[str],
    line_delta: dict[str, object],
) -> TerminalRenderBundle:
    return build_terminal_render_bundle(
        suffix=suffix,
        lines=lines,
        speed_type_ms=20,
        pause_between_ops_ms=360,
        key="terminal-test",
        show_caret=True,
        line_delta=line_delta,
    )


def _assert_segment_classes(
    *,
    lines: Sequence[str],
    segments: Sequence[Sequence[dict[str, str | None]]],
    suffix: str,
) -> None:
    expected_cmd = f"cmd-{suffix}"
    expected_err = f"err-{suffix}"
    assert len(segments) == len(lines)
    for line, segment_list in zip(lines, segments):
        assert segment_list, "each line must include at least one serialised segment"
        classes = {segment.get("c") for segment in segment_list}
        if line.startswith("$ "):
            assert classes == {expected_cmd}
        elif line.strip().upper().startswith("ERROR"):
            assert classes == {expected_err}
        else:
            assert classes <= {None}


def test_intro_terminal_segments_are_serialised_with_classes() -> None:
    lines = intro_terminal._DEFAULT_DEMAI_LINES
    bundle = _build_bundle(
        suffix=intro_terminal._TERMINAL_SUFFIX,
        lines=lines,
        line_delta={"action": "replace", "lines": list(lines)},
    )

    assert bundle.payload["segments"] == bundle.serializable_segments
    assert bundle.payload["totalLineCount"] == len(lines)

    delta = bundle.payload.get("lineDelta")
    assert delta and delta["action"] == "replace"
    assert delta["segments"] == bundle.serializable_segments
    assert delta["totalLineCount"] == len(lines)

    _assert_segment_classes(
        lines=lines,
        segments=bundle.serializable_segments,
        suffix=intro_terminal._TERMINAL_SUFFIX,
    )


def test_stage_terminal_append_segments_include_classes() -> None:
    lines = train._DEFAULT_DEMAI_LINES
    appended: Sequence[str] = lines[-3:]
    bundle = _build_bundle(
        suffix=train._TERMINAL_SUFFIX,
        lines=lines,
        line_delta={"action": "append", "lines": list(appended)},
    )

    assert bundle.payload["segments"] == bundle.serializable_segments
    assert bundle.payload["totalLineCount"] == len(lines)

    delta = bundle.payload.get("lineDelta")
    assert delta and delta["action"] == "append"
    assert delta["totalLineCount"] == len(lines)

    delta_segments = delta["segments"]
    assert isinstance(delta_segments, Iterable)
    assert len(delta_segments) == len(appended)

    _assert_segment_classes(
        lines=lines,
        segments=bundle.serializable_segments,
        suffix=train._TERMINAL_SUFFIX,
    )
    _assert_segment_classes(
        lines=appended,
        segments=delta_segments,  # type: ignore[arg-type]
        suffix=train._TERMINAL_SUFFIX,
    )
