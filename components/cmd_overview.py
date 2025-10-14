from __future__ import annotations

import html
import json
import re
from textwrap import dedent
from typing import Iterable, List
from uuid import uuid4

import streamlit as st
from streamlit.components.v1 import html as components_html


_DEFAULT_DEMAI_LINES: List[str] = [
    "> What is an AI system?",
    "LOADING EU AI ACT, Article 3 \n",
    "0% ■■■■■■■■■■■■■■■■■■■■■■■■ 100% \n",
    "AI system means a machine-based system...",
    "> Wait, but what that actually means? \n",
    "...",
    "STARING demAI.machine",
    "0% ■■■■■■■■■■■■■■■■■■■■■■■■ 100%",
    "You are already inside a machine-based system: a user interface (software) running in the cloud (hardware). You will be guided you through each stage with interactive prompts like this. Browse this page to discover more information about the demAI machine. Use the control room to advance to different stages and enable a Nerd Mode when you are thirsty for more details.",
]

_TERMINAL_SUFFIX = "ai_act_fullterm"
_FINAL_STATE_KEY = "_ai_act_terminal_final_raw"

_TERMINAL_STYLE = dedent(
    f"""
    <style>
      .terminal-{_TERMINAL_SUFFIX} {{
        width: 100%;
        background: #0d1117;
        color: #e5e7eb;
        font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        border-radius: 12px;
        padding: 1.5rem 1rem 1.3rem;
        box-shadow: 0 14px 34px rgba(0,0,0,.25);
        position: relative;
        overflow: hidden;
        min-height: 260px;
      }}
      .terminal-{_TERMINAL_SUFFIX}::before {{
        content: '●  ●  ●';
        position: absolute; top: 8px; left: 12px;
        color: #ef4444cc; letter-spacing: 6px; font-size: .9rem;
      }}
      .term-body-{_TERMINAL_SUFFIX} {{
        margin-top: .8rem;
        white-space: pre-wrap;
        word-wrap: break-word;
        line-height: 1.6;
        font-size: .96rem;
      }}
      .caret-{_TERMINAL_SUFFIX} {{
        margin-left: 2px;
        display:inline-block; width:6px; height:1rem;
        background:#22d3ee; vertical-align:-0.18rem;
        animation: blink-{_TERMINAL_SUFFIX} .85s steps(1,end) infinite;
      }}
      .cmdline-{_TERMINAL_SUFFIX} {{ color: #93c5fd; }}
      .hl-{_TERMINAL_SUFFIX}     {{ color: #a5f3fc; font-weight: 600; }}
      @keyframes blink-{_TERMINAL_SUFFIX} {{ 50% {{ opacity: 0; }} }}
      @media (prefers-reduced-motion: reduce) {{
        .caret-{_TERMINAL_SUFFIX} {{ animation: none; }}
      }}
    </style>
    """
)


def _highlight_line(line: str) -> str:
    stripped = line.strip()
    safe = html.escape(line)
    if re.fullmatch(r"dem[a-z]*ai", stripped, flags=re.IGNORECASE):
        return f'<span class="hl-{_TERMINAL_SUFFIX}">{safe}</span>'
    if line.startswith("$ "):
        return f'<span class="cmdline-{_TERMINAL_SUFFIX}">{safe}</span>'
    return safe


def _highlight_raw(raw: str) -> str:
    highlighted_parts: List[str] = []
    for segment in raw.splitlines(keepends=True):
        if segment.endswith("\n"):
            content = segment[:-1]
            suffix = "\n"
        else:
            content = segment
            suffix = ""
        highlighted_parts.append(_highlight_line(content) + suffix)
    return "".join(highlighted_parts)


def _build_terminal_shell(pre_inner: str, caret_visible: bool, mount_id: str | None = None) -> str:
    caret_style = "display:inline-block;" if caret_visible else "display:none;"
    mount_attr = f' id="{mount_id}"' if mount_id else ""
    return dedent(
        f"""
        <div{mount_attr} class="terminal-{_TERMINAL_SUFFIX}">
          <pre class="term-body-{_TERMINAL_SUFFIX}">{pre_inner}</pre>
          <span class="caret-{_TERMINAL_SUFFIX}" style="{caret_style}"></span>
        </div>
        """
    )


def _prepare_lines(lines: Iterable[str]) -> List[str]:
    return ["" if line is None else str(line) for line in lines]


def _compute_final_state(demai_lines: Iterable[str]) -> str:
    """Final raw content once the animation completes (for caching)."""
    raw = ""
    for line in demai_lines:
        raw += line
        if not line.endswith("\n"):
            raw += "\n"
    return raw


def _estimate_terminal_height(raw: str) -> int:
    """Return an approximate terminal height for the rendered content."""

    if not raw:
        return 320

    line_count = raw.count("\n") + 1
    min_height = 300
    max_height = 680
    per_line = 22
    padding = 140

    estimated = padding + line_count * per_line
    return max(min_height, min(max_height, estimated))


def render_ai_act_terminal(
    demai_lines=None,
    speed_type_ms: int = 20,
    speed_delete_ms: int = 14,         # kept for API compatibility (unused)
    pause_between_ops_ms: int = 360,   # used as per-line pause
):
    """Render the animated EU AI Act terminal sequence using native Streamlit primitives."""

    if demai_lines is None:
        demai_lines = _DEFAULT_DEMAI_LINES

    prepared_lines = _prepare_lines(demai_lines)

    final_state_key = f"{_FINAL_STATE_KEY}:{hash(tuple(prepared_lines))}"
    final_state = st.session_state.get(final_state_key)

    final_raw = final_state or _compute_final_state(prepared_lines)
    target_height = _estimate_terminal_height(final_raw)

    if final_state:
        highlighted = _highlight_raw(final_state)
        static_html = (
            dedent(
                """
                __STYLE__
                __SHELL__
                """
            )
            .replace("__STYLE__", _TERMINAL_STYLE)
            .replace("__SHELL__", _build_terminal_shell(highlighted, caret_visible=False))
        )
        components_html(static_html, height=target_height)
        return
    mount_id = f"terminal-{_TERMINAL_SUFFIX}-{uuid4().hex}"
    config = {
        "lines": prepared_lines,
        "typeDelay": max(speed_type_ms, 0),
        "pauseDelay": max(pause_between_ops_ms, 0),
    }
    config_json = json.dumps(config)
    animated_html = (
        dedent(
            """
            __STYLE__
            __SHELL__
            <script>
              (function() {
                const mount = document.getElementById('__MOUNT_ID__');
                if (!mount || mount.dataset.animated === '1') return;
                mount.dataset.animated = '1';

                const config = __CONFIG_JSON__;
                const lines = Array.isArray(config.lines) ? config.lines : [];
                const typeDelay = Math.max(config.typeDelay || 0, 0);
                const pauseDelay = Math.max(config.pauseDelay || 0, 0);

                const pre = mount.querySelector('.term-body-__SUFFIX__');
                const caret = mount.querySelector('.caret-__SUFFIX__');
                if (!pre || !caret) return;

                let raw = '';

                const esc = (value) => value
                  .replace(/&/g, '&amp;')
                  .replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;')
                  .replace(/"/g, '&quot;')
                  .replace(/'/g, '&#x27;');

                const highlightChunk = (text) => {
                  const parts = text.split(/(\n)/);
                  let out = '';
                  for (const part of parts) {
                    if (part === '\n') { out += '\n'; continue; }
                    const stripped = part.trim();
                    let safe = esc(part);
                    if (/^dem[a-z]*ai$/i.test(stripped)) {
                      safe = '<span class="hl-__SUFFIX__">' + safe + '</span>';
                    } else if (part.indexOf('$ ') === 0) {
                      safe = '<span class="cmdline-__SUFFIX__">' + safe + '</span>';
                    }
                    out += safe;
                  }
                  return out;
                };

                const render = (showCaret) => {
                  pre.innerHTML = highlightChunk(raw);
                  caret.style.display = showCaret ? 'inline-block' : 'none';
                  pre.scrollTop = pre.scrollHeight;
                };

                const wait = (ms) => ms ? new Promise((resolve) => setTimeout(resolve, ms)) : Promise.resolve();

                const typeText = async (text) => {
                  for (let i = 0; i < text.length; i += 1) {
                    raw += text.charAt(i);
                    render(true);
                    await wait(typeDelay);
                  }
                };

                const appendNewline = async () => {
                  raw += '\n';
                  render(true);
                  await wait(typeDelay);
                };

                const run = async () => {
                  render(true);
                  for (const original of lines) {
                    const line = typeof original === 'string' ? original : '';
                    await typeText(line);
                    if (!line.endsWith('\n')) { await appendNewline(); }
                    await wait(pauseDelay);
                  }
                  render(false);
                };

                requestAnimationFrame(run);
              })();
            </script>
            """
        )
        .replace("__STYLE__", _TERMINAL_STYLE)
        .replace("__SHELL__", _build_terminal_shell("", caret_visible=True, mount_id=mount_id))
        .replace("__MOUNT_ID__", mount_id)
        .replace("__CONFIG_JSON__", config_json)
        .replace("__SUFFIX__", _TERMINAL_SUFFIX)
    )

    components_html(animated_html, height=target_height)

    # Cache the fully-materialized final text for quick re-renders
    st.session_state[final_state_key] = final_raw
