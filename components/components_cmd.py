from __future__ import annotations

import html
import json
import re
from textwrap import dedent
from typing import Iterable, List
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components


_DEFAULT_DEMAI_LINES: List[str] = [
    "Welcome to demAI — an interactive experience where you will build and operate an AI system, while discovering and applying key concepts from the EU AI Act.\n",
    "",
    "demonstrateAI",
    "Experience how an AI system actually works, step by step — from data preparation to predictions — through an interactive, hands-on journey.\n",
    "",
    "demistifyAI",
    "Break down complex AI concepts into clear, tangible actions so that anyone can understand what’s behind the model’s decisions.\n",
    "",
    "democratizeAI",
    "Empower everyone to engage responsibly with AI, making transparency and trust accessible to all.",
]

_DEFAULT_OPS = [
    {"kind": "type", "text": "$ get EU-AI-Act.definition\n\n"},
    {"kind": "type", "text": " ‘AI system’ means a machine-based system that is designed to operate with varying levels of autonomy and that may exhibit adaptiveness after deployment, and that, for explicit or implicit objectives, infers, from the input it receives, how to generate outputs such as predictions, content, recommendations, or decisions that can influence physical or virtual environments; \n\n"},
    {"kind": "type", "text": "...confused?\n\n"},
    {"kind": "type", "text": "$ pip install demAI\n\n"},
]

_TERMINAL_SUFFIX = "ai_act_fullterm"
_CSS_KEY = "_ai_act_terminal_css_injected"
_FINAL_STATE_KEY = "_ai_act_terminal_final_raw"


def _ensure_terminal_css() -> None:
    if st.session_state.get(_CSS_KEY):
        return

    css = dedent(
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
        </style>
        """
    )

    st.markdown(css, unsafe_allow_html=True)
    st.session_state[_CSS_KEY] = True


def _highlight_line(line: str) -> str:
    stripped = line.strip()
    safe = html.escape(line)
    if re.fullmatch(r"dem[a-z]*ai", stripped, flags=re.IGNORECASE):
        return f'<span class="hl-{_TERMINAL_SUFFIX}">{safe}</span>'
    if line.startswith("$ "):
        return f'<span class="cmdline-{_TERMINAL_SUFFIX}">{safe}</span>'
    return safe


def _render_terminal_html(placeholder, raw: str, show_caret: bool) -> None:
    highlighted_parts = []
    for segment in raw.splitlines(keepends=True):
        if segment.endswith("\n"):
            content = segment[:-1]
            suffix = "\n"
        else:
            content = segment
            suffix = ""
        highlighted_parts.append(_highlight_line(content) + suffix)

    highlighted = "".join(highlighted_parts)
    caret_style = "display:inline-block;" if show_caret else "display:none;"
    html_payload = dedent(
        f"""
        <div class="terminal-{_TERMINAL_SUFFIX}">
          <pre class="term-body-{_TERMINAL_SUFFIX}">{highlighted}</pre>
          <span class="caret-{_TERMINAL_SUFFIX}" style="{caret_style}"></span>
        </div>
        """
    )
    placeholder.markdown(html_payload, unsafe_allow_html=True)


def _prepare_lines(lines: Iterable[str]) -> List[str]:
    return ["" if line is None else str(line) for line in lines]


def _compute_final_state(ops: Iterable[dict], demai_lines: Iterable[str]) -> str:
    raw = ""

    for op in ops:
        kind = op.get("kind")
        text = op.get("text", "")
        if kind == "type":
            raw += text
        elif kind == "delete" and text and raw.endswith(text):
            raw = raw[: -len(text)]

    raw += "\n"

    for line in demai_lines:
        raw += line
        if not line.endswith("\n"):
            raw += "\n"

    return raw


def _build_inline_mount_markup(config: dict, mount_id: str) -> str:
    config_attr = html.escape(json.dumps(config), quote=True)
    return dedent(
        f"""
        <div id="{mount_id}" class="terminal-{_TERMINAL_SUFFIX}" data-config='{config_attr}' data-animated="0">
          <pre class="term-body-{_TERMINAL_SUFFIX}"></pre>
          <span class="caret-{_TERMINAL_SUFFIX}"></span>
        </div>
        """
    )


def _build_inline_bootstrap(mount_id: str) -> str:
    script = dedent(
        """
        <script>
          (function() {
            var doc = document;
            try {
              if (window.parent && window.parent.document) {
                doc = window.parent.document;
              }
            } catch (err) {
              // ignore cross-origin access issues
            }

            if (!doc) {
              return;
            }

            var mount = doc.getElementById("__MOUNT_ID__");
            if (!mount) {
              return;
            }

            if (mount.dataset.animated === "1") {
              return;
            }
            mount.dataset.animated = "1";

            var config = {};
            try {
              config = JSON.parse(mount.dataset.config || "{}");
            } catch (err) {
              config = {};
            }

            var ops = Array.isArray(config.ops) ? config.ops : [];
            var manifesto = Array.isArray(config.lines) ? config.lines : [];
            var typeDelay = Math.max(config.speed_type || 0, 0);
            var deleteDelay = Math.max(config.speed_delete || 0, 0);
            var pauseDelay = Math.max(config.pause_between_ops || 0, 0);

            var pre = mount.querySelector(".term-body-__SUFFIX__");
            var caret = mount.querySelector(".caret-__SUFFIX__");
            if (!pre || !caret) {
              return;
            }

            var raw = "";

            function escapeHtml(value) {
              return value
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#x27;");
            }

            function highlight(text) {
              var parts = text.split(/(\n)/);
              var result = "";
              for (var i = 0; i < parts.length; i += 1) {
                var part = parts[i];
                if (part === "\n") {
                  result += "\n";
                  continue;
                }
                var stripped = part.trim();
                var safe = escapeHtml(part);
                if (/^dem[a-z]*ai$/i.test(stripped)) {
                  safe = '<span class="hl-__SUFFIX__">' + safe + "</span>";
                } else if (part.indexOf("$ ") === 0) {
                  safe = '<span class="cmdline-__SUFFIX__">' + safe + "</span>";
                }
                result += safe;
              }
              return result;
            }

            function render(showCaret) {
              pre.innerHTML = highlight(raw);
              caret.style.display = showCaret ? "inline-block" : "none";
            }

            function wait(ms) {
              if (!ms) {
                return Promise.resolve();
              }
              return new Promise(function(resolve) {
                setTimeout(resolve, ms);
              });
            }

            async function typeText(text) {
              if (!text) {
                return;
              }
              for (var idx = 0; idx < text.length; idx += 1) {
                raw += text.charAt(idx);
                render(true);
                await wait(typeDelay);
              }
            }

            async function deleteText(text) {
              if (!text) {
                render(true);
                return;
              }
              var target = String(text);
              if (!raw.endsWith(target)) {
                render(true);
                return;
              }
              for (var j = 0; j < target.length; j += 1) {
                raw = raw.slice(0, -1);
                render(true);
                await wait(deleteDelay);
              }
            }

            async function run() {
              render(true);
              for (var k = 0; k < ops.length; k += 1) {
                var op = ops[k] || {};
                var kind = op.kind;
                var text = typeof op.text === "string" ? op.text : "";
                if (kind === "type") {
                  await typeText(text);
                } else if (kind === "delete") {
                  await deleteText(text);
                } else {
                  render(true);
                }
                await wait(pauseDelay);
              }

              await typeText("\n");
              for (var m = 0; m < manifesto.length; m += 1) {
                var line = typeof manifesto[m] === "string" ? manifesto[m] : "";
                await typeText(line);
                if (!line.endsWith("\n")) {
                  await typeText("\n");
                }
              }

              render(false);
            }

            run();
          })();
        </script>
        """
    )

    return script.replace("__MOUNT_ID__", mount_id).replace("__SUFFIX__", _TERMINAL_SUFFIX)


def render_ai_act_terminal(
    demai_lines=None,
    speed_type_ms: int = 20,
    speed_delete_ms: int = 14,
    pause_between_ops_ms: int = 360,
):
    """Render the animated EU AI Act terminal sequence using native Streamlit primitives."""

    if demai_lines is None:
        demai_lines = _DEFAULT_DEMAI_LINES

    _ensure_terminal_css()
    container = st.container()

    final_state_key = f"{_FINAL_STATE_KEY}:{hash(tuple(demai_lines))}"
    final_state = st.session_state.get(final_state_key)
    if final_state:
        _render_terminal_html(container, final_state, show_caret=False)
        return

    prepared_lines = _prepare_lines(demai_lines)
    config = {
        "ops": list(_DEFAULT_OPS),
        "lines": prepared_lines,
        "speed_type": max(speed_type_ms, 0),
        "speed_delete": max(speed_delete_ms, 0),
        "pause_between_ops": max(pause_between_ops_ms, 0),
    }
    mount_id = f"terminal-{_TERMINAL_SUFFIX}-{uuid4().hex}"
    mount_markup = _build_inline_mount_markup(config, mount_id)
    bootstrap = _build_inline_bootstrap(mount_id)

    container.markdown(mount_markup, unsafe_allow_html=True)
    components.html(bootstrap, height=0)

    st.session_state[final_state_key] = _compute_final_state(_DEFAULT_OPS, prepared_lines)
