import json
from textwrap import dedent
from string import Template
from html import escape

import streamlit.components.v1 as components

EU_AI_ACT_DEF = (
    "AI system means a machine-based system that is designed to operate with varying levels of autonomy "
    "and that may exhibit adaptiveness after deployment, and that, for explicit or implicit objectives, "
    "infers, from the input it receives, how to generate outputs such as predictions, content, "
    "recommendations, or decisions that can influence physical or virtual environments."
)

# map of substrings to highlight when first fully typed
HLS = [
    "machine-based",
    "input",
    "infers",
    "outputs",
    "predictions",
    "recommendations",
    "decisions",
    "influence",
]


def render_eu_ai_act_typing():
    """Renders the animated, typed definition with progressive highlights."""
    definition_js = json.dumps(EU_AI_ACT_DEF)
    highlights_js = json.dumps(HLS)
    noscript_text = escape(EU_AI_ACT_DEF)

    html = Template(dedent("""
    <style>
      .eu-typing {
        --fg: #0f172a; --muted:#42506b; --accent:#4f46e5;
        --bg: linear-gradient(135deg, rgba(79,70,229,.06), rgba(14,165,233,.06));
        position: relative; padding: 1.25rem 1.4rem; border-radius: 1.1rem;
        background: var(--bg); border: 1px solid rgba(15,23,42,.08);
        box-shadow: 0 20px 44px rgba(15,23,42,.12);
      }
      .eu-typing__eyebrow {
        font-size:.78rem; letter-spacing:.14em; text-transform:uppercase;
        font-weight:800; color:#3b5bcc; margin: .1rem 0 .6rem;
      }
      .eu-typing__row { display:flex; gap:1rem; align-items:flex-start; }
      .eu-typing__icon {
        width:2.2rem; height:2.2rem; border-radius:.7rem;
        background:#fff; display:grid; place-items:center;
        box-shadow: inset 0 0 0 1px rgba(15,23,42,.06);
        flex:0 0 auto; font-size:1.15rem;
      }
      .eu-typing__text {
        line-height:1.6; font-size:1.02rem; color:var(--fg);
        min-height: 8.75rem; /* reserve space */
        position:relative;
      }
      .caret {
        display:inline-block; width:1px; background: var(--fg);
        animation: blink .9s steps(1) infinite;
        vertical-align:-3px; height:1.1em; margin-left:1px;
      }
      @keyframes blink { 50% { opacity:0; } }

      /* highlight effect */
      .hl { background: linear-gradient(0deg, rgba(79,70,229,.22), rgba(79,70,229,.22));
             border-radius:.25rem; padding:.05rem .2rem; }
      .hl-pop { animation: pop .28s ease-out; }
      @keyframes pop { 0%{transform:scale(.96); box-shadow:0 0 0 rgba(79,70,229,0);}
                       70%{transform:scale(1.02); box-shadow:0 8px 18px rgba(79,70,229,.25);}
                       100%{transform:scale(1); box-shadow:0 0 0 rgba(79,70,229,0);} }

      /* small screens */
      @media (max-width:520px){
        .eu-typing { padding: 1.1rem; }
        .eu-typing__text { font-size:1rem; min-height: 7.5rem; }
        .eu-typing__icon { width:2rem; height:2rem; font-size:1rem; }
      }
    </style>

    <div class="eu-typing" aria-live="polite">
      <div class="eu-typing__eyebrow">From the EU AI Act, Article&nbsp;3</div>
      <div class="eu-typing__row">
        <div class="eu-typing__icon" aria-hidden="true">⚖️</div>
        <p id="eu-typing-text" class="eu-typing__text">
          <noscript>$noscript</noscript>
          <span id="typed"></span><span class="caret" id="caret"></span>
        </p>
      </div>
    </div>

    <script>
      (function() {
        const full = $definition;
        const highlights = $highlights;  // substrings to highlight when they appear
        const typedEl = document.getElementById('typed');
        const caret = document.getElementById('caret');

        if (!typedEl || !caret) return;

        let i = 0;
        const baseDelay = 14;     // typing speed (ms per char) desktop
        const slowAfterComma = 280;
        const mobile = window.matchMedia('(max-width:520px)').matches;
        const delay = mobile ? 18 : baseDelay;

        function applyHighlights() {
          let html = typedEl.textContent;
          // Apply in priority order; use non-overlapping simple replace
          highlights.forEach(h => {
            const rx = new RegExp('(\\b' + h.replace(/[-/\\\\^$*+?.()|[\\\\]{}]/g,'\\$$&') + '\\b)', 'i');
            html = html.replace(rx, '<span class="hl hl-pop">$$1</span>');
          });
          typedEl.innerHTML = html;
        }

        function tick() {
          if (i >= full.length) {
            caret.style.display = 'none';
            return;
          }
          const ch = full.charAt(i++);
          typedEl.textContent += ch;
          // update highlights only when we finish a word boundary (space, punctuation)
          if (/[\\s,.–—;:)/]/.test(ch)) applyHighlights();

          let next = delay;
          if (ch === ',' || ch === '—') next = slowAfterComma;
          if (ch === '.' ) next = slowAfterComma + 120;
          window.setTimeout(tick, next);
        }
        // start after a tiny pause for visual polish
        setTimeout(tick, 350);
      })();
    </script>
    """)).safe_substitute(
        definition=definition_js,
        highlights=highlights_js,
        noscript=noscript_text,
    )
    components.html(html, height=260, scrolling=False)


def render_machine_definition_typing():
    """Animated quote for the Start Your Machine stage with delete + highlight effect."""

    intro_text = (
        "AI system means a machine-based system that is designed to operate with varying "
        "levels of autonomy and that"
    )
    focus_text = "AI system means a machine-based system"

    html = Template(dedent("""
    <style>
      .eu-typing {
        --fg: #0f172a; --muted:#42506b; --accent:#4f46e5;
        --bg: linear-gradient(135deg, rgba(79,70,229,.06), rgba(14,165,233,.06));
        position: relative; padding: 1.25rem 1.4rem; border-radius: 1.1rem;
        background: var(--bg); border: 1px solid rgba(15,23,42,.08);
        box-shadow: 0 20px 44px rgba(15,23,42,.12);
      }
      .eu-typing__eyebrow {
        font-size:.78rem; letter-spacing:.14em; text-transform:uppercase;
        font-weight:800; color:#3b5bcc; margin: .1rem 0 .6rem;
      }
      .eu-typing__row { display:flex; gap:1rem; align-items:flex-start; }
      .eu-typing__icon {
        width:2.2rem; height:2.2rem; border-radius:.7rem;
        background:#fff; display:grid; place-items:center;
        box-shadow: inset 0 0 0 1px rgba(15,23,42,.06);
        flex:0 0 auto; font-size:1.15rem;
      }
      .eu-typing__text {
        line-height:1.6; font-size:1.02rem; color:var(--fg);
        min-height: 5.6rem;
        position:relative;
      }
      .caret {
        display:inline-block; width:1px; background: var(--fg);
        animation: blink .9s steps(1) infinite;
        vertical-align:-3px; height:1.1em; margin-left:1px;
      }
      @keyframes blink { 50% { opacity:0; } }

      .hl {
        background: linear-gradient(0deg, rgba(79,70,229,.22), rgba(79,70,229,.22));
        border-radius:.25rem; padding:.05rem .2rem;
      }
      .hl-strong { font-weight:600; color:#1d4ed8; }

      @media (max-width:520px){
        .eu-typing { padding: 1.1rem; }
        .eu-typing__text { font-size:1rem; min-height: 5.2rem; }
        .eu-typing__icon { width:2rem; height:2rem; font-size:1rem; }
      }
    </style>

    <div class="eu-typing" aria-live="polite">
      <div class="eu-typing__eyebrow">From the EU AI Act, Article&nbsp;3</div>
      <div class="eu-typing__row">
        <div class="eu-typing__icon" aria-hidden="true">⚖️</div>
        <p id="machine-typing-text" class="eu-typing__text">
          <span id="machine-typed"></span><span class="caret" id="machine-caret"></span>
        </p>
      </div>
    </div>

    <script>
      (function() {
        const full = $full_text;
        const keep = $focus_text;
        const typedEl = document.getElementById('machine-typed');
        const caret = document.getElementById('machine-caret');

        if (!typedEl || !caret) return;

        let index = 0;
        const typeDelay = 18;
        const deleteDelay = 26;

        function typeForward() {
          if (index >= full.length) {
            setTimeout(deleteBack, 650);
            return;
          }

          const ch = full.charAt(index++);
          typedEl.textContent += ch;

          let next = typeDelay;
          if (ch === ',' || ch === ';') next = 160;
          if (ch === ' ') next = 22;
          window.setTimeout(typeForward, next);
        }

        function deleteBack() {
          const current = typedEl.textContent || '';
          if (current.length <= keep.length) {
            typedEl.textContent = keep;
            setTimeout(finish, 420);
            return;
          }

          typedEl.textContent = current.slice(0, -1);
          window.setTimeout(deleteBack, deleteDelay);
        }

        function finish() {
          const finalText = keep + '…';
          const highlighted = finalText.replace(
            /machine-based system/i,
            '<span class="hl hl-strong">$&</span>'
          );
          typedEl.innerHTML = highlighted;
          caret.style.display = 'none';
        }

        setTimeout(typeForward, 320);
      })();
    </script>
    """)).safe_substitute(
        full_text=json.dumps(intro_text),
        focus_text=json.dumps(focus_text),
    )

    components.html(html, height=200, scrolling=False)
