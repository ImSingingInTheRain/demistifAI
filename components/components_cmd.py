from __future__ import annotations

import json
from textwrap import dedent

import streamlit as st
import streamlit.components.v1 as components


def render_ai_act_terminal(
    demai_lines=None,
    speed_type_ms: int = 20,
    speed_delete_ms: int = 14,
    pause_between_ops_ms: int = 360,
):
    """Render the animated EU AI Act terminal sequence client-side."""

    if demai_lines is None:
        demai_lines = [
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

    ops = [
        {"kind": "type", "text": "$ get EU AI Act AI system definition\n\n"},
        {"kind": "type", "text": "An AI system is a machine-based system"},
        {"kind": "delete", "text": " a machine-based system"},
        {"kind": "type", "text": " designed to operate with varying levels of autonomy"},
        {"kind": "delete", "text": " designed to operate with varying levels of autonomy"},
        {"kind": "type", "text": " may exhibit adaptiveness after deployment"},
        {"kind": "delete", "text": " may exhibit adaptiveness after deployment"},
        {"kind": "type", "text": " for explicit or implicit objectives, infers"},
        {"kind": "delete", "text": " for explicit or implicit objectives, infers"},
        {"kind": "type", "text": " infers from the input it receives"},
        {"kind": "delete", "text": " from the input it receives"},
        {"kind": "type", "text": " how to generate outputs such as predictions, content, recommendations, or decisions"},
        {"kind": "delete", "text": " such as predictions, content, recommendations, or decisions"},
        {"kind": "type", "text": " that can influence physical or virtual environments.\n\n"},
        {"kind": "type", "text": "confused?\n\n"},
        {"kind": "type", "text": "$ pip install demAI\n\n"},
    ]

    demai_lines_json = json.dumps(demai_lines)
    ops_json = json.dumps(ops)

    suf = "ai_act_fullterm"

    html_payload = dedent(
        f"""
            <style>
              .terminal-{suf} {{
                width: 100%;
                background: #0d1117;
                color: #e5e7eb;
                font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
                border-radius: 12px;
                padding: 1.1rem 1rem 1.3rem;
                box-shadow: 0 14px 34px rgba(0,0,0,.25);
                position: relative;
                overflow: hidden;
                min-height: 260px;
              }}
              .terminal-{suf}::before {{
                content: '●  ●  ●';
                position: absolute; top: 8px; left: 12px;
                color: #ef4444cc; letter-spacing: 6px; font-size: .9rem;
              }}
              .term-body-{suf} {{
                margin-top: .8rem;
                white-space: pre-wrap;
                word-wrap: break-word;
                line-height: 1.6;
                font-size: .96rem;
              }}
              .cmdline-{suf} {{ color: #93c5fd; }}
              .hl-{suf}     {{ color: #a5f3fc; font-weight: 600; }}
              .caret-{suf} {{
                display:inline-block; width:6px; height:1rem;
                background:#22d3ee; vertical-align:-0.18rem;
                animation: blink-{suf} .85s steps(1,end) infinite;
              }}
              @keyframes blink-{suf} {{ 50% {{ opacity: 0; }} }}
            </style>

            <div class="terminal-{suf}">
              <pre id="term-{suf}" class="term-body-{suf}"></pre>
              <span id="caret-{suf}" class="caret-{suf}"></span>
            </div>

            <script>
              (function(){{
                const term   = document.getElementById("term-{suf}");
                const caret  = document.getElementById("caret-{suf}");
                const ops    = {ops_json};
                const extra  = {demai_lines_json};
                const typeDelay   = {int(speed_type_ms)};
                const deleteDelay = {int(speed_delete_ms)};
                const pauseDelay  = {int(pause_between_ops_ms)};

                function sanitize(line) {{
                  return line.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
                }}

                function highlightIfDem(line) {{
                  const safe = sanitize(line);
                  if (/^\\s*dem[a-z]*ai\\s*$/i.test(line.trim())) {{
                    return '<span class="hl-{suf}">' + safe + '</span>';
                  }}
                  if (line.startsWith("$ ")) {{
                    return '<span class="cmdline-{suf}">' + safe + '</span>';
                  }}
                  return safe;
                }}

                function setHTML(html) {{
                  term.innerHTML = html;
                  term.scrollTop = term.scrollHeight;
                }}

                let raw = "";

                function renderRaw() {{
                  const parts = raw.split(/(\n)/);
                  let out = "";
                  for (let i = 0; i < parts.length; i++) {{
                    const seg = parts[i];
                    if (seg === "\n") {{
                      out += "\n";
                      continue;
                    }}
                    out += highlightIfDem(seg);
                  }}
                  setHTML(out);
                }}

                function appendChar(ch) {{
                  raw += ch;
                  renderRaw();
                }}

                function typeText(text, done) {{
                  let i = 0;
                  (function loop(){{
                    if (i >= text.length) return done();
                    appendChar(text.charAt(i++));
                    setTimeout(loop, typeDelay);
                  }})();
                }}

                function deleteText(text, done) {{
                  const target = String(text);
                  if (!target) return done();
                  const end = raw.endsWith(target);
                  const count = end ? target.length : 0;
                  let removed = 0;
                  (function loop(){{
                    if (removed >= count) {{
                      renderRaw();
                      return done();
                    }}
                    raw = raw.slice(0, -1);
                    renderRaw();
                    removed++;
                    setTimeout(loop, deleteDelay);
                  }})();
                }}

                function runOps(list, idx, done) {{
                  if (idx >= list.length) return done();
                  const op = list[idx];
                  const goNext = () => setTimeout(() => runOps(list, idx+1, done), pauseDelay);
                  if (op.kind === "type") return typeText(op.text, goNext);
                  if (op.kind === "delete") return deleteText(op.text, goNext);
                  goNext();
                }}

                function typeLines(lines, i, done) {{
                  if (i >= lines.length) return done();
                  const line = lines[i];
                  const prompt = (i === 0) ? "" : "";
                  typeText(prompt + line + (line.endsWith("\n") ? "" : "\n"), () => typeLines(lines, i+1, done));
                }}

                runOps(ops, 0, () => {{
                  typeLines([""], 0, () => {{
                    typeText("", () => {{
                      typeLines(extra, 0, () => {{
                        caret.style.display = "none";
                      }});
                    }});
                  }});
                }});
              }})();
            </script>
            """
    )

    render_html = getattr(st, "html", None)
    if callable(render_html):
        render_html(html_payload)
        return

    components.html(html_payload, height=310, scrolling=False)

