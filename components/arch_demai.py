from textwrap import dedent
import html
import streamlit as st


def render_demai_architecture(nerd_mode: bool = False, active_stage: str | None = None):
    """
    Renders the interactive 'demAI machine' architecture diagram used across stages.
    (Same architecture: style_block + html_block + script_block)
    """

    # Content
    MODEL_TITLE = "🧠 AI model"
    MODEL_DESC  = ("Learns from examples you provide to distinguish Spam vs Safe. "
                   "At use-time it analyzes each new email and emits a spam score.")
    MODEL_NERD  = "Text encoder + classifier; optional numeric guardrails; thresholding → spam/safe."

    INBOX_TITLE = "📥 Inbox interface"
    INBOX_DESC  = ("A simulated inbox feeds emails into the system and sends them to the AI model.")
    INBOX_NERD  = "Batch/stream ingestion; metadata extraction; replay for evaluation."

    UI_TITLE    = "🖥️ User interface"
    UI_DESC     = ("Your control panel for building and using the system. Tooltips and explainers guide each step.")
    UI_NERD     = "Stages: Prepare • Train • Evaluate • Use; debug panels; parameter sliders; model insights."

    # Optional highlight
    stage_glow = {
        "overview": "ui",
        "data":     "ui",
        "train":    "model",
        "evaluate": "model",
        "use":      "inbox",
    }.get((active_stage or "").lower(), "")

    glow_ui    = " is-glow" if stage_glow == "ui"    else ""
    glow_inbox = " is-glow" if stage_glow == "inbox" else ""
    glow_model = " is-glow" if stage_glow == "model" else ""

    # ---------- Styles (updated) ----------
    style_block = dedent("""
    <style>
      .demai-arch {
        --ink: #0f172a; --muted: rgba(15,23,42,.78);
        --ring1: rgba(99,102,241,.10); --ring2: rgba(14,165,233,.10);
        --card: #fff; --shadow: 0 16px 36px rgba(15,23,42,.14), inset 0 0 0 1px rgba(15,23,42,.06);
        position: relative; border-radius: 16px;
        background: radial-gradient(90% 90% at 50% 0%, rgba(2,132,199,.10), rgba(99,102,241,.06));
        padding: clamp(12px, 2.6vw, 20px); margin: .25rem 0 1rem 0; isolation: isolate;
      }

      .demai-arch__canvas {
        position: relative; width: min(980px, 100%); margin: 0 auto;
        aspect-ratio: 16 / 9; border-radius: 14px;
        background: linear-gradient(180deg, rgba(255,255,255,.85), rgba(248,250,252,.70));
        box-shadow: inset 0 0 0 1px rgba(15,23,42,.06); overflow: hidden;
      }
      @supports not (aspect-ratio: 16 / 9) {
        .demai-arch__canvas::before { content:""; display:block; padding-top:56.25%; }
      }

      /* Rings: give more space so cards can size up nicely */
      .demai-arch__ring { position:absolute; border-radius:18px; display:grid; place-items:center; }
      .demai-arch__ring--outer { inset: 6%; box-shadow: inset 0 0 0 2px rgba(15,23,42,.06); }
      .demai-arch__ring--mid   { inset: 16%; box-shadow: inset 0 0 0 2px rgba(15,23,42,.08); }
      .demai-arch__ring--inner { inset: 32%; box-shadow: inset 0 0 0 2px rgba(15,23,42,.10); }

      .demai-arch__ring.is-glow {
        box-shadow:
          inset 0 0 0 2px rgba(59,130,246,.45),
          0 0 0 6px rgba(59,130,246,.12),
          0 8px 28px rgba(59,130,246,.10);
      }

      .demai-arch__slot { display:grid; place-items:center; width:100%; height:100%; padding: 3.5%; }

      /* Cards: give them a real height so absolute faces don't collapse the parent */
      .demai-arch .arch-card {
        width: min(560px, 88%); max-width: 560px;
        min-height: clamp(140px, 24vh, 190px);   /* <- critical fix */
        background: var(--card);
        border-radius: 16px; box-shadow: var(--shadow);
        transform-style: preserve-3d; perspective: 1000px;
        cursor: pointer; position: relative;
        transition: transform .18s ease;
      }
      .demai-arch .arch-card:hover { transform: translateY(-2px); }

      .demai-arch .arch-card__face {
        position: absolute; inset: 0;
        padding: clamp(16px, 2.8vw, 22px) clamp(18px, 3vw, 26px);
        display:grid; gap:.5rem; align-content:center; text-align:center;
        border-radius: 16px;
        backface-visibility: hidden;
        transition: transform .6s ease, opacity .6s ease;
      }
      .demai-arch .arch-card__front { transform: rotateY(0deg); }
      .demai-arch .arch-card__back  { transform: rotateY(180deg); opacity: 0; }

      .demai-arch .arch-card.is-flipped .arch-card__front { transform: rotateY(180deg); opacity: 0; }
      .demai-arch .arch-card.is-flipped .arch-card__back  { transform: rotateY(360deg); opacity: 1; }

      .demai-arch .arch-card__title {
        font-weight: 800; color: var(--ink);
        display:inline-flex; gap:.55rem; align-items:center; justify-content:center;
        font-size: clamp(1.06rem, 1.1vw + .95rem, 1.25rem);
      }
      .demai-arch .arch-card__icon { font-size: clamp(1.2rem, 1.4vw + 1rem, 1.5rem); }
      .demai-arch .arch-card__desc { color: var(--muted); line-height:1.58; font-size: clamp(.95rem, .35vw + .9rem, 1rem); }
      .demai-arch .arch-card__extra {
        margin-top:.4rem; font-size: .9rem; color: rgba(15,23,42,.85);
        background: rgba(226,232,240,.6); border-radius:10px; padding:.55rem .7rem;
      }

      .demai-arch.nerd-on .arch-card { cursor: default; }
      .demai-arch.nerd-on .arch-card .arch-card__front { display:none; }
      .demai-arch.nerd-on .arch-card .arch-card__back  { position:relative; transform:none; opacity:1; }

      .demai-arch .arch-card:focus-visible { outline:2px solid rgba(59,130,246,.7); outline-offset:4px; }

      /* Mobile */
      @media (max-width: 680px){
        .demai-arch__canvas { aspect-ratio: 4 / 5; }
        .demai-arch__ring--outer { inset: 4%; }
        .demai-arch__ring--mid   { inset: 12%; }
        .demai-arch__ring--inner { inset: 28%; }
        .demai-arch .arch-card { min-height: clamp(150px, 30vh, 220px); }
      }
    </style>
    """)

    # ---------- Markup ----------
    html_block = dedent(f"""
    <div class="demai-arch{' nerd-on' if nerd_mode else ''}" aria-label="demAI architecture diagram">
      <div class="demai-arch__canvas" role="presentation">
        <div class="demai-arch__ring demai-arch__ring--outer{glow_ui}">
          <div class="demai-arch__slot">
            <article class="arch-card" tabindex="0" data-arch="ui" aria-label="User interface card">
              <div class="arch-card__face arch-card__front">
                <div class="arch-card__title"><span class="arch-card__icon">🖥️</span> {html.escape(UI_TITLE)}</div>
              </div>
              <div class="arch-card__face arch-card__back">
                <div class="arch-card__title"><span class="arch-card__icon">🖥️</span> {html.escape(UI_TITLE)}</div>
                <p class="arch-card__desc">{html.escape(UI_DESC)}</p>
                <div class="arch-card__extra">{html.escape(UI_NERD)}</div>
              </div>
            </article>
          </div>
        </div>

        <div class="demai-arch__ring demai-arch__ring--mid{glow_inbox}">
          <div class="demai-arch__slot">
            <article class="arch-card" tabindex="0" data-arch="inbox" aria-label="Inbox interface card">
              <div class="arch-card__face arch-card__front">
                <div class="arch-card__title"><span class="arch-card__icon">📥</span> {html.escape(INBOX_TITLE)}</div>
              </div>
              <div class="arch-card__face arch-card__back">
                <div class="arch-card__title"><span class="arch-card__icon">📥</span> {html.escape(INBOX_TITLE)}</div>
                <p class="arch-card__desc">{html.escape(INBOX_DESC)}</p>
                <div class="arch-card__extra">{html.escape(INBOX_NERD)}</div>
              </div>
            </article>
          </div>
        </div>

        <div class="demai-arch__ring demai-arch__ring--inner{glow_model}">
          <div class="demai-arch__slot">
            <article class="arch-card" tabindex="0" data-arch="model" aria-label="AI model card">
              <div class="arch-card__face arch-card__front">
                <div class="arch-card__title"><span class="arch-card__icon">🧠</span> {html.escape(MODEL_TITLE)}</div>
              </div>
              <div class="arch-card__face arch-card__back">
                <div class="arch-card__title"><span class="arch-card__icon">🧠</span> {html.escape(MODEL_TITLE)}</div>
                <p class="arch-card__desc">{html.escape(MODEL_DESC)}</p>
                <div class="arch-card__extra">{html.escape(MODEL_NERD)}</div>
              </div>
            </article>
          </div>
        </div>
      </div>
    </div>
    """)

    # ---------- Script (unchanged behavior) ----------
    script_block = dedent("""
    <script>
      (function() {
        const root = document.currentScript.closest('.block-container') || document.body;
        const host = root.querySelector('.demai-arch');
        if (!host) return;
        const nerdOn = host.classList.contains('nerd-on');
        if (nerdOn) return; // auto-expanded; no flips

        host.querySelectorAll('.arch-card').forEach(card => {
          card.addEventListener('click', () => card.classList.toggle('is-flipped'));
          card.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); card.classList.toggle('is-flipped'); }
          });
        });
      })();
    </script>
    """)

    full_markup = "\n".join([style_block, html_block, script_block])

    if hasattr(st, "html"):
        st.html(full_markup)
    else:
        st.markdown(full_markup, unsafe_allow_html=True)