from textwrap import dedent
import html
import streamlit as st


def render_demai_architecture(nerd_mode: bool = False, active_stage: str | None = None):
    """
    Simple, robust 'demAI machine' diagram:
    - 3 responsive flip-cards laid out with CSS Grid
    - nerd_mode=True shows backs (details) without flipping
    - active_stage highlights a card: 'overview'/'data' -> UI, 'use' -> Inbox, 'train'/'evaluate' -> Model
    """

    # --- Content ---------------------------------------------------------
    MODEL_TITLE = "🧠 AI model"
    MODEL_DESC  = ("Learns from examples you provide to distinguish Spam vs Safe. "
                   "At use-time it analyzes each new email and emits a spam score.")
    MODEL_NERD  = "Text encoder + classifier; optional numeric guardrails; thresholding → spam/safe."

    INBOX_TITLE = "📥 Inbox interface"
    INBOX_DESC  = "A simulated inbox feeds emails into the system and sends them to the AI model."
    INBOX_NERD  = "Batch/stream ingestion; metadata extraction; replay for evaluation."

    UI_TITLE    = "🖥️ User interface"
    UI_DESC     = "Your control panel for building and using the system. Tooltips and explainers guide each step."
    UI_NERD     = "Stages: Prepare • Train • Evaluate • Use; debug panels; parameter sliders; model insights."

    # Map stage to highlight target
    stage_glow = {
        "overview": "ui",
        "data":     "ui",
        "train":    "model",
        "evaluate": "model",
        "use":      "inbox",
    }.get((active_stage or "").lower(), "")

    glow_ui    = " is-highlight" if stage_glow == "ui"    else ""
    glow_inbox = " is-highlight" if stage_glow == "inbox" else ""
    glow_model = " is-highlight" if stage_glow == "model" else ""

    # --- Styles (no absolute positioning; pure grid) ---------------------
    css = dedent("""
    <style>
      .arch-surface{
        --ink:#0f172a; --muted:rgba(15,23,42,.78);
        --card:#fff; --stroke:rgba(15,23,42,.08);
        --brand:rgba(59,130,246,.65);
        border-radius:16px; padding:clamp(12px,2.4vw,20px);
        background: radial-gradient(120% 100% at 50% 0%, rgba(99,102,241,.08), rgba(14,165,233,.06));
        box-shadow: inset 0 0 0 1px rgba(15,23,42,.06);
      }

      /* Subtle “frame” to nod to layers without layout complexity */
      .arch-surface__frame{
        border-radius:14px;
        background: linear-gradient(180deg, rgba(255,255,255,.92), rgba(248,250,252,.82));
        box-shadow: inset 0 0 0 1px var(--stroke);
        padding: clamp(8px, 1.6vw, 14px);
      }

      .arch-grid{
        display:grid;
        gap: clamp(12px, 1.6vw, 16px);
        grid-template-columns: 1fr;              /* mobile first */
      }

      /* Desktop: 3 columns, center card slightly wider */
      @media (min-width: 900px){
        .arch-grid{
          grid-template-columns: 1fr 1.15fr 1fr;
          align-items: stretch;
        }
      }

      .arch-card{
        border-radius:16px; perspective:1000px; position:relative;
      }

      .arch-card__inner{
        position:relative; border-radius:16px; height:100%;
        transform-style: preserve-3d; transition: transform .6s ease;
        box-shadow: 0 16px 36px rgba(15,23,42,.14), inset 0 0 0 1px var(--stroke);
        background: var(--card);
        cursor: pointer;
      }

      .arch-card__face{
        position:absolute; inset:0; border-radius:16px;
        display:grid; place-content:center; gap:.55rem;
        padding: clamp(16px, 2.4vw, 22px);
        backface-visibility:hidden;
        text-align:center;
      }

      .arch-card__front{ }
      .arch-card__back{ transform: rotateY(180deg); }

      .arch-card__title{
        color: var(--ink); font-weight:800;
        font-size: clamp(1.05rem, 1.1vw + .95rem, 1.25rem);
        display:inline-flex; align-items:center; gap:.6rem; justify-content:center;
      }
      .arch-card__icon{ font-size: clamp(1.2rem, 1.4vw + 1rem, 1.5rem); }
      .arch-card__desc{ color:var(--muted); line-height:1.6; font-size: clamp(.95rem, .3vw + .9rem, 1rem); }
      .arch-card__extra{
        margin-top:.45rem; font-size:.9rem; color:rgba(15,23,42,.9);
        background: rgba(226,232,240,.6); border-radius:10px; padding:.55rem .7rem;
      }

      /* Flip on toggle */
      .arch-card.is-flipped .arch-card__inner{ transform: rotateY(180deg); }

      /* Nerd mode: always show details; no flipping cursor */
      .arch-surface.nerd-on .arch-card__inner{
        transform: none !important; cursor: default;
      }
      .arch-surface.nerd-on .arch-card__front{ display:none; }
      .arch-surface.nerd-on .arch-card__back{ transform: none; }

      /* Highlight ring */
      .arch-card.is-highlight .arch-card__inner{
        box-shadow:
          0 18px 38px rgba(59,130,246,.15),
          inset 0 0 0 2px var(--brand),
          inset 0 0 0 1px rgba(255,255,255,.3);
      }

      /* Focus accessibility */
      .arch-card__inner:focus-visible{ outline: 2px solid rgba(59,130,246,.7); outline-offset: 4px; }

    </style>
    """)

    # --- Markup ----------------------------------------------------------
    html_block = dedent(f"""
    <div class="arch-surface{' nerd-on' if nerd_mode else ''}" aria-label="demAI architecture diagram">
      <div class="arch-surface__frame">
        <div class="arch-grid">

          <!-- UI card -->
          <article class="arch-card{' is-highlight' if glow_ui else ''}" data-arch="ui">
            <div class="arch-card__inner" tabindex="0" aria-label="User interface card">
              <div class="arch-card__face arch-card__front">
                <div class="arch-card__title"><span class="arch-card__icon">🖥️</span>{html.escape(UI_TITLE)}</div>
              </div>
              <div class="arch-card__face arch-card__back">
                <div class="arch-card__title"><span class="arch-card__icon">🖥️</span>{html.escape(UI_TITLE)}</div>
                <p class="arch-card__desc">{html.escape(UI_DESC)}</p>
                <div class="arch-card__extra">{html.escape(UI_NERD)}</div>
              </div>
            </div>
          </article>

          <!-- Model card (center column on desktop) -->
          <article class="arch-card{' is-highlight' if glow_model else ''}" data-arch="model">
            <div class="arch-card__inner" tabindex="0" aria-label="AI model card">
              <div class="arch-card__face arch-card__front">
                <div class="arch-card__title"><span class="arch-card__icon">🧠</span>{html.escape(MODEL_TITLE)}</div>
              </div>
              <div class="arch-card__face arch-card__back">
                <div class="arch-card__title"><span class="arch-card__icon">🧠</span>{html.escape(MODEL_TITLE)}</div>
                <p class="arch-card__desc">{html.escape(MODEL_DESC)}</p>
                <div class="arch-card__extra">{html.escape(MODEL_NERD)}</div>
              </div>
            </div>
          </article>

          <!-- Inbox card -->
          <article class="arch-card{' is-highlight' if glow_inbox else ''}" data-arch="inbox">
            <div class="arch-card__inner" tabindex="0" aria-label="Inbox interface card">
              <div class="arch-card__face arch-card__front">
                <div class="arch-card__title"><span class="arch-card__icon">📥</span>{html.escape(INBOX_TITLE)}</div>
              </div>
              <div class="arch-card__face arch-card__back">
                <div class="arch-card__title"><span class="arch-card__icon">📥</span>{html.escape(INBOX_TITLE)}</div>
                <p class="arch-card__desc">{html.escape(INBOX_DESC)}</p>
                <div class="arch-card__extra">{html.escape(INBOX_NERD)}</div>
              </div>
            </div>
          </article>

        </div>
      </div>
    </div>
    """)

    # --- Script (click/keyboard flip; no flip in nerd mode) --------------
    js = dedent("""
    <script>
      (function(){
        const root = document.currentScript.closest('.block-container') || document.body;
        const host = root.querySelector('.arch-surface');
        if(!host) return;

        if(host.classList.contains('nerd-on')) return; // always open; no flipping

        host.querySelectorAll('.arch-card').forEach(card=>{
          const inner = card.querySelector('.arch-card__inner');
          // click
          inner.addEventListener('click', ()=> card.classList.toggle('is-flipped'));
          // keyboard
          inner.addEventListener('keydown', (e)=>{
            if(e.key === 'Enter' || e.key === ' '){
              e.preventDefault();
              card.classList.toggle('is-flipped');
            }
          });
        });
      })();
    </script>
    """)

    payload = css + html_block + js

    # Native render (no iframe)
    if hasattr(st, "html"):
        st.html(payload)
    else:
        st.markdown(payload, unsafe_allow_html=True)
