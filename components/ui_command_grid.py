from textwrap import dedent
import streamlit.components.v1 as components


def render_command_grid(lines=None, title=""):
    """
    Renders a 2/3–1/3 responsive grid:
      - Left: animated command terminal (typewriter)
      - Right: placeholder card with equal height to the terminal
    The right card's height is auto-synced to the left via CSS Grid + stretch and
    a tiny ResizeObserver for stubborn browsers.
    """
    if lines is None:
        lines = [
            "$ pip install demAI",
            "Welcome to demAI — an interactive experience where you will build and operate an AI system, while discovering and applying key concepts from the EU AI Act.\n",
            "",
            "demonstrateAI",
            "Experience how an AI system actually works, step by step — from data preparation to predictions — through an interactive, hands-on journey.\n",
            "",
            "demistifyAI",
            "Break down complex AI concepts into clear, tangible actions so that anyone can understand what’s behind the model’s decisions.\n",
            "",
            "democratizeAI",
            "Empower everyone to engage responsibly with AI, making transparency and trust accessible to all."
        ]

    # Unique suffix prevents ID clashes if rendered multiple times
    suf = "welcome_cmd"

    html = dedent(f"""
    <style>
      /* ---- Scoped grid -------------------------------------------------- */
      .cmdgrid-{suf} {{
        display: grid;
        grid-template-columns: 2fr 1fr;        /* 2/3 : 1/3 */
        gap: 1.1rem;
        align-items: stretch;                  /* equal row height */
      }}
      @media (max-width: 860px){{
        .cmdgrid-{suf} {{ grid-template-columns: 1fr; }}
      }}

      /* ---- Terminal (left) --------------------------------------------- */
      .terminal-{suf} {{
        background: #0d1117;
        color: #e5e7eb;
        font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        border-radius: 12px;
        padding: 1.1rem 1rem 1.3rem;
        box-shadow: 0 14px 34px rgba(0,0,0,.25);
        position: relative;
        overflow: hidden;
        min-height: 240px;
        display: flex;
        flex-direction: column;
      }}
      .terminal-{suf}::before {{
        content: '●  ●  ●';
        position: absolute; top: 8px; left: 12px;
        color: #ef4444cc; letter-spacing: 6px; font-size: .9rem;
      }}
      .term-body-{suf} {{
        margin-top: 0.6rem;
        white-space: pre-wrap; word-wrap: break-word;
        line-height: 1.55; font-size: .95rem;
        flex: 1 1 auto;
      }}
      .cmdline-{suf} {{ color: #93c5fd; }}
      .hl-{suf} {{ color: #a5f3fc; }}
      .caret-{suf} {{
        display:inline-block; width:6px; height:1rem;
        background:#22d3ee; vertical-align:-0.18rem;
        animation: blink-{suf} .85s steps(1,end) infinite;
      }}
      @keyframes blink-{suf} {{ 50% {{ opacity: 0; }} }}

      /* ---- Placeholder card (right) ------------------------------------ */
      .placeholder-{suf} {{
        background: linear-gradient(155deg, rgba(248,250,252,.95), rgba(226,232,240,.6));
        border-radius: 12px;
        box-shadow: 0 12px 28px rgba(15,23,42,.08), inset 0 0 0 1px rgba(148,163,184,.25);
        padding: 1rem 1.1rem;
        display: flex; align-items: center; justify-content: center;
        min-height: 240px;          /* matches terminal minimum */
        height: 100%;               /* stretch to grid row height */
        color: #334155;
        text-align: center;
      }}
      .placeholder-{suf} .ph-label {{
        font-weight: 700; color:#0f172a; margin-bottom:.25rem;
      }}
      .placeholder-{suf} .ph-sub {{
        font-size:.9rem; color: rgba(15,23,42,.7);
      }}
    </style>

    <div class="cmdgrid-{suf}">
      <!-- LEFT: terminal -->
      <div class="terminal-{suf}">
        <pre id="term-content-{suf}" class="term-body-{suf}"></pre>
        <div id="term-caret-{suf}" class="caret-{suf}"></div>
      </div>

      <!-- RIGHT: placeholder (equal height) -->
      <div id="placeholder-{suf}" class="placeholder-{suf}">
        <div>
          <div class="ph-label">Placeholder</div>
          <div class="ph-sub">This panel matches the terminal’s height and will host future UI.</div>
        </div>
      </div>
    </div>

    <script>
      // Typewriter for terminal lines (robust & simple)
      const LINES_{suf} = {lines!r};
      const container_{suf} = document.getElementById("term-content-{suf}");
      const caret_{suf} = document.getElementById("term-caret-{suf}");
      container_{suf}.textContent = '';

      const lineNodes_{suf} = [];
      for(let idx = 0; idx < LINES_{suf}.length; idx++){{
        const span = document.createElement('span');
        const candidateLine = LINES_{suf}[idx];
        const rawLine = (candidateLine === undefined || candidateLine === null) ? '' : candidateLine;
        if(idx === 0){{
          span.classList.add('cmdline-{suf}');
        }} else {{
          const trimmed = rawLine.trim();
          if(trimmed && /^dem[a-z]*ai$/i.test(trimmed)){{
            span.classList.add('hl-{suf}');
          }}
        }}
        container_{suf}.appendChild(span);
        lineNodes_{suf}.push(span);
      }}

      let i_{suf} = 0;
      let j_{suf} = 0;

      function typeNext_{suf}(){{
        if(i_{suf} >= LINES_{suf}.length){{
          caret_{suf}.style.display = 'none';
          return;
        }}
        const candidateActiveLine = LINES_{suf}[i_{suf}];
        const line = (candidateActiveLine === undefined || candidateActiveLine === null) ? '' : candidateActiveLine;
        const target = lineNodes_{suf}[i_{suf}];
        if(j_{suf} < line.length){{
          target.textContent += line[j_{suf}++];
          setTimeout(typeNext_{suf}, 24);
        }} else {{
          if(i_{suf} < LINES_{suf}.length - 1 && !line.endsWith('\n')){{
            target.textContent += '\n';
          }}
          i_{suf}++;
          j_{suf} = 0;
          setTimeout(typeNext_{suf}, 360);
        }}
      }}
      setTimeout(typeNext_{suf}, 320);

      // Height sync fallback (most browsers stretch via CSS Grid already)
      try {{
        const grid = document.querySelector('.cmdgrid-{suf}');
        const ph  = document.getElementById('placeholder-{suf}');
        const ro = new ResizeObserver(()=>{{ ph.style.height = grid.offsetHeight + 'px'; }});
        ro.observe(grid);
      }} catch(e){{ /* no-op */ }}
    </script>
    """)
    components.html(html, height=520, scrolling=False)
