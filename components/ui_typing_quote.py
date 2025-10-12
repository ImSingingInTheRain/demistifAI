import base64
import json
from textwrap import dedent
from string import Template
from html import escape

import streamlit.components.v1 as components

EU_AI_ACT_DEF = (
    "An AI system infers how to generate outputs that can influence physical or virtual environments."
)

# map of substrings to highlight when first fully typed
HLS = [
    "infers",
    "outputs",
    "physical",
    "virtual",
    "environments",
]


def _eu_ai_act_typing_markup(
    *,
    padding: str = "1.25rem 1.4rem",
    background: str = "linear-gradient(135deg, rgba(79,70,229,.06), rgba(14,165,233,.06))",
    box_shadow: str = "0 20px 44px rgba(15,23,42,.12)",
    min_height: str = "8.75rem",
    eyebrow_text: str = "From the EU AI Act, Article&nbsp;3",
    icon: str = "⚖️",
) -> str:
    definition_js = json.dumps(EU_AI_ACT_DEF)
    highlights_js = json.dumps(HLS)
    noscript_text = escape(EU_AI_ACT_DEF)

    return Template(dedent(r"""
    <style>
      .eu-typing {
        --fg: #0f172a; --muted:#42506b; --accent:#4f46e5;
        position: relative; padding: $padding; border-radius: 1.1rem;
        background: $background; border: 1px solid rgba(15,23,42,.08);
        box-shadow: $box_shadow;
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
        min-height: $min_height; /* reserve space */
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
        .eu-typing__text { font-size:1rem; min-height: calc($min_height - .9rem); }
        .eu-typing__icon { width:2rem; height:2rem; font-size:1rem; }
      }
    </style>

    <div class="eu-typing" aria-live="polite">
      <div class="eu-typing__eyebrow">$eyebrow</div>
      <div class="eu-typing__row">
        <div class="eu-typing__icon" aria-hidden="true">$icon</div>
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
            const rx = new RegExp('(\\\b' + h.replace(/[-/\\\\^$*+?.()|[\\\\]{}]/g,'\\$$&') + '\\b)', 'i');
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
          if (/[\\\s,.–—;:)/]/.test(ch)) applyHighlights();

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
        eyebrow=eyebrow_text,
        icon=icon,
        padding=padding,
        background=background,
        box_shadow=box_shadow,
        min_height=min_height,
    )


def render_eu_ai_act_typing():
    """Renders the animated, typed definition with progressive highlights."""
    html = _eu_ai_act_typing_markup()
    components.html(html, height=260, scrolling=False)


def get_eu_ai_act_typing_iframe_src() -> str:
    """Returns a data URI suitable for embedding in an iframe within the hero card."""

    html = _eu_ai_act_typing_markup(
        padding="0.9rem 1rem",
        background="linear-gradient(135deg, rgba(79,70,229,.04), rgba(14,165,233,.04))",
        box_shadow="0 12px 30px rgba(79,70,229,.14)",
        min_height="7.9rem",
        eyebrow_text="Definition • EU AI Act",
    )
    encoded = base64.b64encode(html.encode("utf-8")).decode("ascii")
    return f"data:text/html;base64,{encoded}"


def get_eu_ai_act_typing_inline_markup(
    *,
    id_prefix: str = "eu-typing-hero",
    eyebrow_text: str = "Definition • EU AI Act",
    icon: str = "⚖️",
) -> str:
    """Return markup for the inline hero animation container.

    The returned HTML carries data attributes so a bootstrap script can animate
    the definition without relying on an iframe wrapper.
    """

    definition_attr = escape(EU_AI_ACT_DEF, quote=True)
    highlights_attr = escape(json.dumps(HLS), quote=True)
    eyebrow_attr = escape(eyebrow_text, quote=True)
    icon_attr = escape(icon, quote=True)
    noscript_text = escape(EU_AI_ACT_DEF)

    container_id = f"{id_prefix}-container"
    text_id = f"{id_prefix}-text"
    typed_id = f"{id_prefix}-typed"
    caret_id = f"{id_prefix}-caret"

    return Template(
        dedent(
            """
            <div class="eu-typing" id="$container_id" data-definition="$definition" data-highlights="$highlights" aria-live="polite">
              <div class="eu-typing__eyebrow">$eyebrow</div>
              <div class="eu-typing__row">
                <div class="eu-typing__icon" aria-hidden="true">$icon</div>
                <p id="$text_id" class="eu-typing__text" data-role="text">
                  <span id="$typed_id" data-role="typed"></span><span class="caret" id="$caret_id" data-role="caret"></span>
                  <noscript>$noscript</noscript>
                </p>
              </div>
            </div>
            """
        )
    ).safe_substitute(
        container_id=container_id,
        definition=definition_attr,
        highlights=highlights_attr,
        eyebrow=eyebrow_attr,
        icon=icon_attr,
        text_id=text_id,
        typed_id=typed_id,
        caret_id=caret_id,
        noscript=noscript_text,
    )


def get_eu_ai_act_typing_inline_bootstrap(*, id_prefix: str = "eu-typing-hero") -> str:
    """Return a script snippet that animates the inline hero quote.

    The script is intended to be rendered via ``components.html(..., height=0)``
    so it can reach into the Streamlit document and progressively type the text.
    """

    container_id = f"{id_prefix}-container"
    typed_id = f"{id_prefix}-typed"
    caret_id = f"{id_prefix}-caret"

    return Template(
        dedent(
            r"""
            <script>
              (function() {
                var rootWindow = window;
                var doc = document;
                try {
                  if (window.parent && window.parent.document) {
                    rootWindow = window.parent;
                    doc = window.parent.document;
                  }
                } catch (err) {
                  // ignore cross-origin access issues
                }

                if (!doc) {
                  return;
                }

                var container = doc.getElementById('$container_id');
                var typedEl = doc.getElementById('$typed_id');
                var caret = doc.getElementById('$caret_id');
                if (!container || !typedEl || !caret) {
                  return;
                }

                if (container.dataset.animated === '1') {
                  return;
                }
                container.dataset.animated = '1';

                var highlights = [];
                try {
                  highlights = JSON.parse(container.dataset.highlights || '[]');
                } catch (err) {
                  highlights = [];
                }

                typedEl.textContent = '';

                function escapeRegex(str) {
                  return str.replace(/[-/\\^$*+?.()|[\\]{}]/g, '\\$&');
                }

                function applyHighlights() {
                  var html = typedEl.textContent;
                  highlights.forEach(function(h) {
                    if (!h) {
                      return;
                    }
                    var rx = new RegExp('(\\\b' + escapeRegex(h) + '\\b)', 'i');
                    html = html.replace(rx, '<span class="hl hl-pop">$$1</span>');
                  });
                  typedEl.innerHTML = html;
                }

                var typeDelay = 18;
                var deleteDelay = 26;
                var pauseAfterType = 360;
                var pauseAfterDelete = 240;

                if (rootWindow.matchMedia && rootWindow.matchMedia('(max-width:520px)').matches) {
                  typeDelay = 22;
                  deleteDelay = 30;
                }

                var steps = [
                  { kind: 'type', text: 'An AI system is a machine-based system' },
                  { kind: 'delete', text: ' a machine-based system' },
                  { kind: 'type', text: ' designed to operate with varying levels of autonomy' },
                  { kind: 'delete', text: ' is designed to operate with varying levels of autonomy' },
                  { kind: 'type', text: ' may exhibit adaptiveness after deployment' },
                  { kind: 'delete', text: ' may exhibit adaptiveness after deployment' },
                  { kind: 'type', text: ' for explicit or implicit objectives, infers' },
                  { kind: 'delete', text: ' for explicit or implicit objectives,' },
                  { kind: 'type', text: ' from the input it receives' },
                  { kind: 'delete', text: ' from the input it receives' },
                  { kind: 'type', text: ' how to generate outputs such as predictions, content, recommendations, or decisions' },
                  { kind: 'delete', text: ' such as predictions, content, recommendations, or decisions' },
                  { kind: 'type', text: ' that can influence physical or virtual environments.' }
                ];

                function typeForward(text, index, done) {
                  if (index >= text.length) {
                    return done();
                  }
                  var ch = text.charAt(index);
                  typedEl.textContent += ch;
                  applyHighlights();

                  var nextDelay = typeDelay;
                  if (ch === ',' || ch === '—') {
                    nextDelay = Math.max(typeDelay, 220);
                  } else if (ch === '.') {
                    nextDelay = Math.max(typeDelay, 260);
                  } else if (ch === ' ') {
                    nextDelay = Math.max(typeDelay, 22);
                  }

                  rootWindow.setTimeout(function() {
                    typeForward(text, index + 1, done);
                  }, nextDelay);
                }

                function deleteBackward(text, done) {
                  var length = text.length;
                  var current = typedEl.textContent;
                  if (!current.endsWith(text)) {
                    length = Math.min(length, current.length);
                  }

                  function stepDelete(remaining) {
                    if (remaining <= 0) {
                      return done();
                    }
                    typedEl.textContent = typedEl.textContent.slice(0, -1);
                    applyHighlights();
                    rootWindow.setTimeout(function() {
                      stepDelete(remaining - 1);
                    }, deleteDelay);
                  }

                  stepDelete(length);
                }

                function runStep(idx) {
                  if (idx >= steps.length) {
                    applyHighlights();
                    caret.style.display = 'none';
                    return;
                  }

                  var step = steps[idx];
                  if (step.kind === 'type') {
                    typeForward(step.text, 0, function() {
                      rootWindow.setTimeout(function() {
                        runStep(idx + 1);
                      }, step.pauseAfter || pauseAfterType);
                    });
                  } else {
                    deleteBackward(step.text, function() {
                      rootWindow.setTimeout(function() {
                        runStep(idx + 1);
                      }, step.pauseAfter || pauseAfterDelete);
                    });
                  }
                }

                rootWindow.setTimeout(function() {
                  runStep(0);
                }, 320);
              })();
            </script>
            """
        )
    ).safe_substitute(
        container_id=container_id,
        typed_id=typed_id,
        caret_id=caret_id,
    )


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
