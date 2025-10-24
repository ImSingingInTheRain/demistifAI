"""Shared helpers for building the animated terminal renderer."""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from textwrap import dedent
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple
import html
import re

from streamlit.components.v1 import html as components_html

Segment = Tuple[str, Optional[str]]
SegmentPayload = List[Segment]
SerializableSegment = List[dict[str, Optional[str]]]


@dataclass(frozen=True)
class TerminalRenderBundle:
    """Container with render-time data for terminal surfaces."""

    markup: str
    payload: dict[str, Any]
    serializable_segments: List[SerializableSegment]
    terminal_style: str


def _build_terminal_style(suffix: str) -> str:
    return dedent(
        f"""
        <style>
          .terminal-{suffix} {{
            width: min(100%, 680px);
            height: auto;
            background: #0d1117;
            color: #ffffff;
            font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            box-shadow: 0 22px 45px rgba(2, 6, 23, 0.45);
            padding: 1.5rem 1.2rem 1.35rem;
            position: relative;
            overflow: hidden;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 1.15rem;
            box-sizing: border-box;
          }}
          .terminal-{suffix}::before {{
            content: '●  ●  ●';
            position: absolute; top: 8px; left: 12px;
            color: #ef4444cc; letter-spacing: 6px; font-size: .9rem;
          }}
          .term-body-{suffix} {{
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
            line-height: 1.6;
            font-size: .96rem;
            min-height: 1.5rem;
          }}
          .term-input-wrap-{suffix} {{
            display: flex;
            align-items: center;
            gap: .4rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 8px;
            background: rgba(13, 17, 23, 0.9);
            padding: .55rem .75rem;
            transition: border-color .2s ease, box-shadow .2s ease, background .2s ease, opacity .2s ease;
            pointer-events: none;
            opacity: .55;
          }}
          .term-input-wrap-{suffix}[data-active="true"] {{
            pointer-events: auto;
            opacity: 1;
          }}
          .term-input-wrap-{suffix}[data-active="true"]:focus-within {{
            border-color: rgba(34, 211, 238, 0.85);
            box-shadow: 0 0 0 2px rgba(14, 116, 144, 0.35);
            background: rgba(13, 17, 23, 0.98);
          }}
          .term-input-{suffix} {{
            width: 100%;
            background: transparent;
            border: 0;
            color: #f8fafc;
            padding: 0;
            font-size: .94rem;
            line-height: 1.4;
            font-family: inherit;
            letter-spacing: .01em;
            caret-color: #22d3ee;
            outline: none;
          }}
          .term-input-{suffix}::placeholder {{
            color: rgba(226, 232, 240, 0.6);
          }}
          .term-input-{suffix}:disabled {{
            opacity: 1;
            cursor: not-allowed;
          }}
          .term-input-wrap-{suffix}[data-active="false"] .term-input-{suffix} {{
            cursor: not-allowed;
          }}
          .sr-only-{suffix} {{
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
          }}
          .caret-{suffix} {{
            margin-left: 2px;
            display:inline-block; width:6px; height:1rem;
            background:#22d3ee; vertical-align:-0.18rem;
            animation: blink-{suffix} .85s steps(1,end) infinite;
          }}
          .cmd-{suffix} {{ color: #93c5fd; }}
          .err-{suffix} {{ color: #fb7185; font-weight: 700; }}
          @keyframes blink-{suffix} {{ 50% {{ opacity: 0; }} }}

          .terminal-wrap-{suffix} {{
            opacity: 0; transform: translateY(6px);
            animation: fadein-{suffix} .6s ease forwards;
          }}
          @keyframes fadein-{suffix} {{
            to {{ opacity: 1; transform: translateY(0) }}
          }}

          @media (prefers-reduced-motion: reduce) {{
            .caret-{suffix} {{ animation: none; }}
            .terminal-wrap-{suffix} {{ animation: none; opacity:1; transform:none; }}
          }}

          @media (max-width: 640px) {{
            .terminal-{suffix} {{
              border-radius: 10px;
              padding: clamp(1rem, 5vw, 1.35rem);
              gap: 1rem;
            }}
            .term-body-{suffix} {{
              font-size: .9rem;
            }}
            .term-input-{suffix} {{
              font-size: .88rem;
            }}
            .term-input-wrap-{suffix} {{
              border-radius: 7px;
            }}
          }}

          @media (max-width: 600px) {{
            .terminal-wrap-{suffix} {{
              margin: 0;
              padding: 0;
            }}
            .terminal-{suffix} {{
              width: 100vw;
              max-width: none;
              margin-block: 0;
              margin-inline: calc(50% - 50vw);
              border-radius: 0;
              padding: clamp(1.2rem, 7vw, 1.8rem);
              box-sizing: border-box;
              display: flex;
              flex-direction: column;
              gap: 1.25rem;
            }}
            .terminal-{suffix}::before {{
              top: 14px;
              left: 50%;
              transform: translateX(-50%);
            }}
            .term-body-{suffix} {{
              font-size: 1.05rem;
              line-height: 1.72;
              width: 100%;
              overflow-wrap: anywhere;
              word-break: break-word;
              overflow-x: hidden;
            }}
            .caret-{suffix} {{
              align-self: flex-start;
              margin-top: -.2rem;
              flex-shrink: 0;
            }}
            .term-input-{suffix} {{
              font-size: 1rem;
            }}
            .term-input-wrap-{suffix} {{
              border-radius: 6px;
            }}
          }}
        </style>
        """
    )


def _normalize_lines(lines: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for raw in lines:
        line = str(raw)
        normalized.append(line if line.endswith("\n") else f"{line}\n")
    return normalized


def _escape_text(text: str) -> str:
    return html.escape(text, quote=False)


def _escape_attr(text: str) -> str:
    return html.escape(text, quote=True)


def _split_segments(line: str, suffix: str) -> List[Segment]:
    stripped = line.strip()
    if not line:
        return [(line, None)]
    if line.startswith("$ "):
        return [(line, f"cmd-{suffix}")]
    if re.match(r"^ERROR\b", stripped, flags=re.IGNORECASE):
        return [(line, f"err-{suffix}")]
    return [(line, None)]


def _segments_to_html(segments: Sequence[Segment]) -> str:
    rendered: List[str] = []
    for text, css in segments:
        escaped = _escape_text(text)
        if css:
            rendered.append(f'<span class="{css}">{escaped}</span>')
        else:
            rendered.append(escaped)
    return "".join(rendered)


def _highlight_line(line: str, suffix: str) -> str:
    return _segments_to_html(_split_segments(line, suffix))


def _compute_full_html(lines: Sequence[str], suffix: str) -> str:
    normalized = _normalize_lines(lines)
    segments = [_split_segments(line, suffix) for line in normalized]
    return "".join(_segments_to_html(parts) for parts in segments)


def _compute_segment_payload(lines: Sequence[str], suffix: str) -> List[SegmentPayload]:
    normalized = _normalize_lines(lines)
    return [_split_segments(line, suffix) for line in normalized]


def _build_terminal_markup(
    *,
    payload: dict[str, Any],
    suffix: str,
    terminal_style: str,
    normalized_lines: Sequence[str],
    show_caret: bool,
    secondary_inputs: Optional[Sequence[dict[str, str]]] = None,
) -> str:
    final_text = "".join(normalized_lines)
    caret_display = "inline-block" if show_caret else "none"
    accept_keystrokes = "true" if payload.get("acceptKeystrokes") else "false"
    input_placeholder = _escape_attr(str(payload.get("placeholder", "")))
    input_aria_label = _escape_attr(str(payload.get("inputAriaLabel", "")))
    input_label = _escape_text(str(payload.get("inputLabel", "")))
    input_hint = _escape_text(str(payload.get("inputHint", "")))
    input_id = _escape_attr(str(payload.get("inputId", "")))
    input_hint_id = _escape_attr(str(payload.get("inputHintId", "")))
    secondary_markup_parts: List[str] = []
    for secondary in secondary_inputs or ():
        secondary_placeholder = _escape_attr(str(secondary.get("placeholder", "")))
        secondary_aria_label = _escape_attr(str(secondary.get("inputAriaLabel", "")))
        secondary_label = _escape_text(str(secondary.get("inputLabel", "")))
        secondary_hint = _escape_text(str(secondary.get("inputHint", "")))
        secondary_input_id = _escape_attr(str(secondary.get("inputId", "")))
        secondary_hint_id = _escape_attr(str(secondary.get("inputHintId", "")))
        secondary_markup_parts.append(
            Template(
                dedent(
                    """
                        <div class="term-input-wrap-$suffix" data-active="false" aria-hidden="true" data-secondary="true">
                          <label class="sr-only-$suffix" for="$input_id">$input_label</label>
                          <input
                            id="$input_id"
                            class="term-input-$suffix"
                            type="text"
                            inputmode="text"
                            autocomplete="off"
                            spellcheck="false"
                            aria-label="$input_aria_label"
                            aria-describedby="$input_hint_id"
                            placeholder="$input_placeholder"
                            disabled
                            data-secondary="true"
                          />
                          <span id="$input_hint_id" class="sr-only-$suffix">$input_hint</span>
                        </div>
                    """
                )
            ).substitute(
                suffix=suffix,
                input_placeholder=secondary_placeholder,
                input_aria_label=secondary_aria_label,
                input_label=secondary_label,
                input_hint=secondary_hint,
                input_id=secondary_input_id,
                input_hint_id=secondary_hint_id,
            )
        )
    secondary_markup = "".join(secondary_markup_parts)
    template = Template(
        dedent(
            """
            $terminal_style<div class="terminal-wrap-$suffix">
              <div id="$dom_id" class="terminal-$suffix" role="region" aria-label="EU AI Act terminal animation" data-ready="false" data-accept-keystrokes="$accept_keystrokes">
                <pre class="term-body-$suffix"></pre>
                <span class="caret-$suffix" style="display:$caret_display"></span>
                <div class="term-input-wrap-$suffix" data-active="false" aria-hidden="true">
                  <label class="sr-only-$suffix" for="$input_id">$input_label</label>
                  <input
                    id="$input_id"
                    class="term-input-$suffix"
                    type="text"
                    inputmode="text"
                    autocomplete="off"
                    spellcheck="false"
                    aria-label="$input_aria_label"
                    aria-describedby="$input_hint_id"
                    placeholder="$input_placeholder"
                  />
                  <span id="$input_hint_id" class="sr-only-$suffix">$input_hint</span>
                </div>
                $secondary_markup
              </div>
            </div>

            <noscript>
              <div class="terminal-$suffix">
                <pre class="term-body-$suffix">$final_text</pre>
              </div>
            </noscript>
            """
        )
    )
    return template.substitute(
        terminal_style=terminal_style,
        suffix=suffix,
        caret_display=caret_display,
        final_text=final_text,
        dom_id=payload["domId"],
        accept_keystrokes=accept_keystrokes,
        input_placeholder=input_placeholder,
        input_aria_label=input_aria_label,
        input_label=input_label,
        input_hint=input_hint,
        input_id=input_id,
        input_hint_id=input_hint_id,
        secondary_markup=secondary_markup,
    )


def build_terminal_render_bundle(
    *,
    suffix: str,
    lines: Sequence[str],
    speed_type_ms: int,
    pause_between_ops_ms: int,
    key: str,
    show_caret: bool,
    terminal_style: Optional[str] = None,
    placeholder: str = "",
    input_aria_label: Optional[str] = None,
    accept_keystrokes: bool = False,
    debounce_ms: int = 150,
    secondary_inputs: Optional[Sequence[str]] = None,
    input_text: Optional[str] = None,
    prefilled_line_count: Optional[int] = None,
) -> TerminalRenderBundle:
    style = terminal_style or _build_terminal_style(suffix)
    normalized_lines = _normalize_lines(lines)
    segments = _compute_segment_payload(lines, suffix)
    full_html = "".join(_segments_to_html(parts) for parts in segments)
    serializable_segments: List[SerializableSegment] = [
        [{"t": text, "c": css} for text, css in parts]
        for parts in segments
    ]
    total_lines = len(normalized_lines)
    clamped_prefilled_count = 0
    if prefilled_line_count is not None:
        try:
            candidate = int(prefilled_line_count)
        except (TypeError, ValueError):
            candidate = 0
        clamped_prefilled_count = max(0, min(candidate, total_lines))

    payload = {
        "lines": list(lines),
        "fullHtml": full_html,
        "speedType": max(0, int(speed_type_ms)),
        "pauseBetween": max(0, int(pause_between_ops_ms)),
        "showCaret": bool(show_caret),
        "suffix": suffix,
        "domId": f"term-{key}",
        "segments": serializable_segments,
    }
    placeholder_text = str(placeholder or "")
    placeholder_trimmed = placeholder_text.strip()
    aria_label_text = (input_aria_label or placeholder_trimmed or "Interactive terminal input").strip()
    if not aria_label_text:
        aria_label_text = "Interactive terminal input"
    hint_text = placeholder_trimmed or "Type your command and press Enter to submit."
    input_id = f"{payload['domId']}-input"
    input_hint_id = f"{payload['domId']}-hint"
    secondary_payloads: List[dict[str, str]] = []
    for index, secondary_placeholder in enumerate(secondary_inputs or ()): 
        secondary_text = str(secondary_placeholder or "")
        secondary_trimmed = secondary_text.strip()
        secondary_label = secondary_trimmed or "Secondary terminal input"
        secondary_hint = (
            f"{secondary_trimmed} command field is currently disabled."
            if secondary_trimmed
            else "This command field is currently disabled."
        )
        secondary_payloads.append(
            {
                "placeholder": secondary_text,
                "inputAriaLabel": secondary_label,
                "inputLabel": secondary_label,
                "inputHint": secondary_hint,
                "inputId": f"{input_id}-secondary-{index + 1}",
                "inputHintId": f"{input_hint_id}-secondary-{index + 1}",
            }
        )
    payload.update(
        {
            "placeholder": placeholder_text,
            "inputAriaLabel": aria_label_text,
            "inputLabel": aria_label_text,
            "inputHint": hint_text,
            "inputId": input_id,
            "inputHintId": input_hint_id,
            "acceptKeystrokes": bool(accept_keystrokes),
            "debounceMs": max(0, int(debounce_ms)),
        }
    )
    if secondary_payloads:
        payload["secondaryInputs"] = secondary_payloads
    if input_text is not None:
        payload["inputValue"] = str(input_text)
    if prefilled_line_count is not None:
        payload["prefilledLineCount"] = clamped_prefilled_count
    markup = _build_terminal_markup(
        payload=payload,
        suffix=suffix,
        terminal_style=style,
        normalized_lines=normalized_lines,
        show_caret=show_caret,
        secondary_inputs=secondary_payloads,
    )
    return TerminalRenderBundle(
        markup=markup,
        payload=payload,
        serializable_segments=serializable_segments,
        terminal_style=style,
    )


TerminalRenderer = Callable[
    [Optional[Iterable[str]], int, int, str, bool],
    None,
]


def make_terminal_renderer(
    suffix: str,
    default_lines: Sequence[str],
    default_key: str = "ai_act_terminal",
) -> TerminalRenderer:
    """Create a terminal renderer configured for a stage."""

    terminal_style = _build_terminal_style(suffix)
    default_lines_cache = tuple(default_lines)

    def render_ai_act_terminal(
        demai_lines: Optional[Iterable[str]] = None,
        speed_type_ms: int = 20,
        pause_between_ops_ms: int = 360,
        key: str = default_key,
        show_caret: bool = True,
    ) -> None:
        lines = list(demai_lines) if demai_lines is not None else list(default_lines_cache)
        bundle = build_terminal_render_bundle(
            suffix=suffix,
            lines=lines,
            speed_type_ms=speed_type_ms,
            pause_between_ops_ms=pause_between_ops_ms,
            key=key,
            show_caret=show_caret,
            terminal_style=terminal_style,
        )

        components_html(bundle.markup, height=800)

    render_ai_act_terminal.__name__ = "render_ai_act_terminal"
    render_ai_act_terminal.__doc__ = (
        "Render the animated EU AI Act terminal sequence using a client-side typing loop with auto-resize.\n\n"
        "- Non-blocking: the rest of the Streamlit page renders immediately.\n"
        "- Auto-resizing: iframe height grows with content while typing.\n"
        "- Honors 'prefers-reduced-motion': final state is shown if motion is reduced.\n"
        "- f-string safe: avoids backslashes inside f-string expressions."
    )

    return render_ai_act_terminal


__all__ = [
    "make_terminal_renderer",
    "build_terminal_render_bundle",
    "TerminalRenderBundle",
]
