"""Shared helpers for building the animated terminal renderer."""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from textwrap import dedent
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple
import html
import json
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
) -> str:
    final_text = "".join(normalized_lines)
    caret_display = "inline-block" if show_caret else "none"
    payload_json = json.dumps(payload)
    accept_keystrokes = "true" if payload.get("acceptKeystrokes") else "false"
    input_placeholder = _escape_attr(str(payload.get("placeholder", "")))
    input_aria_label = _escape_attr(str(payload.get("inputAriaLabel", "")))
    input_label = _escape_text(str(payload.get("inputLabel", "")))
    input_hint = _escape_text(str(payload.get("inputHint", "")))
    input_id = _escape_attr(str(payload.get("inputId", "")))
    input_hint_id = _escape_attr(str(payload.get("inputHintId", "")))
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
              </div>
            </div>

            <noscript>
              <div class="terminal-$suffix">
                <pre class="term-body-$suffix">$final_text</pre>
              </div>
            </noscript>

            <script>
            (function() {{
              const cfg = $payload_json;
              const root  = document.getElementById(cfg.domId);
              if(!root) return;

              const pre   = root.querySelector(".term-body-" + cfg.suffix);
              const caret = root.querySelector(".caret-" + cfg.suffix);
              const inputWrap = root.querySelector(".term-input-wrap-" + cfg.suffix);
              const input = root.querySelector(".term-input-" + cfg.suffix);
              const acceptKeystrokes = Boolean(cfg.acceptKeystrokes);
              const debounceMs = Number.isFinite(cfg.debounceMs) ? Math.max(0, cfg.debounceMs) : 150;

              const streamlitAvailable =
                typeof window.Streamlit === "object" &&
                window.Streamlit !== null &&
                typeof window.Streamlit.setComponentValue === "function";

              const state = {
                text: input ? String(input.value || "") : "",
                submitted: false,
                ready: false,
              };
              let wasReady = false;
              let debounceHandle = null;

              const copyState = () => ({ text: state.text, submitted: state.submitted, ready: state.ready });

              const sendState = (immediate) => {
                if (!streamlitAvailable) return;
                if (immediate) {
                  window.Streamlit.setComponentValue({ value: copyState() });
                  return;
                }
                window.clearTimeout(debounceHandle);
                debounceHandle = window.setTimeout(() => {
                  window.Streamlit.setComponentValue({ value: copyState() });
                }, debounceMs);
              };

              const syncReadyUi = () => {
                root.setAttribute("data-ready", state.ready ? "true" : "false");
                if (inputWrap) {
                  const active = state.ready && acceptKeystrokes;
                  inputWrap.setAttribute("data-active", active ? "true" : "false");
                  inputWrap.setAttribute("aria-hidden", active ? "false" : "true");
                }
                if (input) {
                  if (state.ready && acceptKeystrokes) {
                    input.disabled = false;
                    input.setAttribute("aria-disabled", "false");
                    if (!wasReady) {
                      wasReady = true;
                      window.requestAnimationFrame(() => {
                        try { input.focus({ preventScroll: true }); }
                        catch (err) { input.focus(); }
                      });
                    }
                  } else {
                    input.disabled = true;
                    input.setAttribute("aria-disabled", "true");
                  }
                }
              };

              const setState = (updates, options) => {
                const immediate = options && options.immediate === true;
                const prevReady = state.ready;
                if (updates && typeof updates === "object") {
                  if (Object.prototype.hasOwnProperty.call(updates, "text") && typeof updates.text === "string") {
                    state.text = updates.text;
                  }
                  if (Object.prototype.hasOwnProperty.call(updates, "submitted") && typeof updates.submitted === "boolean") {
                    state.submitted = updates.submitted;
                  }
                  if (Object.prototype.hasOwnProperty.call(updates, "ready") && typeof updates.ready === "boolean") {
                    state.ready = updates.ready;
                  }
                }
                if (prevReady !== state.ready) {
                  syncReadyUi();
                }
                sendState(immediate);
              };

              syncReadyUi();
              sendState(true);

              if (input) {
                input.addEventListener("input", () => {
                  setState({ text: input.value, submitted: false });
                });
                input.addEventListener("keydown", (event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    setState({ text: input.value, submitted: true }, { immediate: true });
                    window.setTimeout(() => {
                      setState({ submitted: false });
                    }, debounceMs);
                  }
                });
                input.addEventListener("blur", () => {
                  setState({ submitted: false });
                });
              }

              const rawLines = (cfg.lines || []).map((l) => (l == null ? "" : String(l)));
              const toLinesWithNL = (arr) => arr.map((l) => (l.endsWith("\\n") ? l : l + "\\n"));
              const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

              const normaliseSegments = (segments) =>
                segments.map((line) =>
                  Array.isArray(line)
                    ? line.map((seg) => ({
                        t: typeof seg.t === "string" ? seg.t : "",
                        c: typeof seg.c === "string" && seg.c ? seg.c : null,
                      }))
                    : []
                );

              const splitSegments = (line) => {{
                const trimmed = line.trim();
                if (line.startsWith("$$ ")) return [{{ t: line, c: "cmd-" + cfg.suffix }}];
                if (/^ERROR\b/i.test(trimmed)) return [{{ t: line, c: "err-" + cfg.suffix }}];
                return [{{ t: line, c: null }}];
              }};

              const normalisedLines = toLinesWithNL(rawLines);
              const perLineSegs = Array.isArray(cfg.segments)
                ? normaliseSegments(cfg.segments)
                : normalisedLines.map(splitSegments);

              const perLineSegmentHtml = perLineSegs.map((segs) =>
                segs.map((seg) => (seg.c ? `<span class="$${{seg.c}}">$${{esc(seg.t)}}</span>` : esc(seg.t)))
              );
              const perLineHtml = perLineSegmentHtml.map((parts) => parts.join(""));
              const perLineRaw = perLineSegs.map((segs) => segs.map((seg) => seg.t).join(""));

              const finalRaw = perLineRaw.join("");
              const computedFinalHtml = perLineHtml.join("");
              const finalHtml = typeof cfg.fullHtml === "string" ? cfg.fullHtml : computedFinalHtml;
              cfg.fullHtml = finalHtml;

              const ensureMeasuredHeight = () => {{
                const measurement = root.cloneNode(true);
                measurement.removeAttribute("id");
                measurement.style.position = "absolute";
                measurement.style.visibility = "hidden";
                measurement.style.pointerEvents = "none";
                measurement.style.opacity = "0";
                measurement.style.left = "-9999px";
                measurement.style.top = "0";
                measurement.style.height = "auto";
                measurement.style.minHeight = "auto";
                measurement.style.maxHeight = "none";
                const width = root.getBoundingClientRect().width || root.offsetWidth || root.clientWidth;
                if (width) {{
                  measurement.style.width = width + "px";
                }}
                const measurePre = measurement.querySelector(".term-body-" + cfg.suffix);
                if (measurePre) {{
                  measurePre.innerHTML = finalHtml;
                }}
                const measureCaret = measurement.querySelector(".caret-" + cfg.suffix);
                if (measureCaret) {{
                  measureCaret.style.display = "none";
                }}
                const measureWrap = measurement.querySelector(".term-input-wrap-" + cfg.suffix);
                const readyActive = root.getAttribute("data-ready") === "true" && acceptKeystrokes;
                if (measureWrap) {{
                  measureWrap.setAttribute("data-active", readyActive ? "true" : "false");
                  measureWrap.setAttribute("aria-hidden", readyActive ? "false" : "true");
                }}
                const measureInput = measurement.querySelector(".term-input-" + cfg.suffix);
                if (measureInput) {{
                  measureInput.disabled = !readyActive;
                }}
                (root.parentElement || document.body).appendChild(measurement);
                const height = measurement.scrollHeight;
                measurement.remove();
                if (height) {{
                  root.style.minHeight = `$${{height}}px`;
                  root.style.height = `$${{height}}px`;
                }}
                return height;
              }};

              const notifyResize = (height) => {{
                if (Number.isFinite(height) && height > 0) {{
                  window.parent.postMessage({{ "type": "streamlit:resize", "height": height + 24 }}, "*");
                }}
              }};

              const measuredHeight = ensureMeasuredHeight();
              notifyResize(measuredHeight);

              const prefersReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

              let finished = false;
              const finalizeTerminal = () => {{
                if (finished) {{
                  return;
                }}
                finished = true;
                setState({{ ready: true }}, {{ immediate: true }});
                const finalHeight = ensureMeasuredHeight();
                notifyResize(finalHeight);
              }};

              if (prefersReduced || cfg.speedType === 0) {{
                pre.innerHTML = finalHtml;
                if (caret) caret.style.display = "none";
                finalizeTerminal();
                return;
              }}

              // --- Typing engine (client-side, non-blocking) ---
              const TYPE_DELAY = Math.max(0, cfg.speedType);
              const BETWEEN_LINES = Math.max(0, cfg.pauseBetween);

              const doneHtmlParts = [];
              let activeNode = null;
              let lineIndex = 0;
              let segmentIndex = 0;
              let charIndex = 0;

              const syncDoneHtml = () => {{
                pre.innerHTML = doneHtmlParts.join("");
              }};

              const ensureActiveNode = () => {{
                if (activeNode) {{
                  return activeNode;
                }}
                const segments = perLineSegs[lineIndex] || [];
                const current = segments[segmentIndex];
                if (!current) {{
                  return null;
                }}
                if (current.c) {{
                  const span = document.createElement("span");
                  span.className = current.c;
                  span.textContent = "";
                  pre.appendChild(span);
                  activeNode = span;
                }} else {{
                  activeNode = document.createTextNode("");
                  pre.appendChild(activeNode);
                }}
                return activeNode;
              }};

              const commitActiveSegment = () => {{
                if (!activeNode) {{
                  return;
                }}
                if (activeNode.parentNode === pre) {{
                  pre.removeChild(activeNode);
                }}
                const segmentHtml = (perLineSegmentHtml[lineIndex] || [])[segmentIndex] || "";
                doneHtmlParts.push(segmentHtml);
                activeNode = null;
                syncDoneHtml();
              }};

              syncDoneHtml();

              function step() {{
                if (lineIndex >= perLineSegs.length) {{
                  syncDoneHtml();
                  if (caret) {{
                    caret.style.display = "none";
                  }}
                  finalizeTerminal();
                  return;
                }}

                const segments = perLineSegs[lineIndex] || [];
                const current = segments[segmentIndex];

                if (!current) {{
                  segmentIndex = 0;
                  lineIndex += 1;
                  setTimeout(step, BETWEEN_LINES);
                  return;
                }}

                const target = current.t;
                if (charIndex < target.length) {{
                  const node = ensureActiveNode();
                  if (node) {{
                    const char = target.charAt(charIndex);
                    node.textContent = (node.textContent || "") + char;
                  }}
                  charIndex += 1;
                  setTimeout(step, TYPE_DELAY);
                  return;
                }}

                commitActiveSegment();
                segmentIndex += 1;
                charIndex = 0;

                if (segmentIndex >= segments.length) {{
                  segmentIndex = 0;
                  lineIndex += 1;
                  setTimeout(step, BETWEEN_LINES);
                  return;
                }}

                setTimeout(step, TYPE_DELAY);
              }}

              requestAnimationFrame(step);
            })();
            </script>
            """
        )
    )
    return template.substitute(
        terminal_style=terminal_style,
        suffix=suffix,
        caret_display=caret_display,
        final_text=final_text,
        payload_json=payload_json,
        dom_id=payload["domId"],
        accept_keystrokes=accept_keystrokes,
        input_placeholder=input_placeholder,
        input_aria_label=input_aria_label,
        input_label=input_label,
        input_hint=input_hint,
        input_id=input_id,
        input_hint_id=input_hint_id,
    )


def build_terminal_render_bundle(
    *,
    suffix: str,
    lines: Sequence[str],
    speed_type_ms: int,
    speed_delete_ms: int,
    pause_between_ops_ms: int,
    key: str,
    show_caret: bool,
    terminal_style: Optional[str] = None,
    placeholder: str = "",
    input_aria_label: Optional[str] = None,
    accept_keystrokes: bool = False,
    debounce_ms: int = 150,
) -> TerminalRenderBundle:
    style = terminal_style or _build_terminal_style(suffix)
    normalized_lines = _normalize_lines(lines)
    segments = _compute_segment_payload(lines, suffix)
    full_html = "".join(_segments_to_html(parts) for parts in segments)
    serializable_segments: List[SerializableSegment] = [
        [{"t": text, "c": css} for text, css in parts]
        for parts in segments
    ]
    payload = {
        "lines": list(lines),
        "fullHtml": full_html,
        "speedType": max(0, int(speed_type_ms)),
        "pauseBetween": max(0, int(pause_between_ops_ms)),
        "speedDelete": max(0, int(speed_delete_ms)),
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
    markup = _build_terminal_markup(
        payload=payload,
        suffix=suffix,
        terminal_style=style,
        normalized_lines=normalized_lines,
        show_caret=show_caret,
    )
    return TerminalRenderBundle(
        markup=markup,
        payload=payload,
        serializable_segments=serializable_segments,
        terminal_style=style,
    )


TerminalRenderer = Callable[
    [Optional[Iterable[str]], int, int, int, str, bool],
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
        speed_delete_ms: int = 14,
        pause_between_ops_ms: int = 360,
        key: str = default_key,
        show_caret: bool = True,
    ) -> None:
        lines = list(demai_lines) if demai_lines is not None else list(default_lines_cache)
        bundle = build_terminal_render_bundle(
            suffix=suffix,
            lines=lines,
            speed_type_ms=speed_type_ms,
            speed_delete_ms=speed_delete_ms,
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
