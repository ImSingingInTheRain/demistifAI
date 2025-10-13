from textwrap import dedent
import html
import streamlit as st


def render_demai_architecture(nerd_mode: bool = False, active_stage: str | None = None):
    """
    Renders the interactive 'demAI machine' architecture diagram used across stages.

    Uses `st.html` when available (Streamlit ≥ 1.36) so that `<style>`/`<script>`
    tags execute correctly on newer releases; falls back to `st.markdown` for
    older deployments.

    nerd_mode=True  -> auto-expand cards and reveal extra details.
    active_stage    -> optional highlight: 'overview'/'data'/'train'/'evaluate'/'use'
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

    # Highlight ring by stage (optional, subtle glow)
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


    style_block = dedent("""
    <style>
      .demai-arch {
        --ink: #0f172a; --muted: rgba(15,23,42,.72);
        --ring1: rgba(99,102,241,.10); --ring2: rgba(14,165,233,.10); --ring3: rgba(15,23,42,.06);
        --card: #fff; --shadow: 0 14px 30px rgba(15,23,42,.12), inset 0 0 0 1px rgba(15,23,42,.06);
        position: relative; border-radius: 16px;
        background: radial-gradient(90% 90% at 50% 50%, var(--ring1), var(--ring2));
        box-shadow: inset 0 0 0 1px rgba(15,23,42,.06);
        padding: clamp(14px, 3vw, 22px); margin: .25rem 0 1rem 0; isolation: isolate;
      }

      .demai-arch__canvas {
        position: relative; width: min(920px, 100%); margin: 0 auto;
        aspect-ratio: 16 / 9; border-radius: 12px;
        background: linear-gradient(180deg, rgba(255,255,255,.7), rgba(255,255,255,.35));
        box-shadow: inset 0 0 0 1px rgba(15,23,42,.06); overflow: hidden;
      }
      @supports not (aspect-ratio: 16 / 9) {
        .demai-arch__canvas::before { content:""; display:block; padding-top:56.25%; }
      }

      .demai-arch__ring { position:absolute; inset:6%; border-radius:18px; display:grid; place-items:center; }
      .demai-arch__ring--outer { inset:4%;  box-shadow: inset 0 0 0 2px rgba(15,23,42,.06); }
      .demai-arch__ring--mid   { inset:12%; box-shadow: inset 0 0 0 2px rgba(15,23,42,.08); }
      .demai-arch__ring--inner { inset:26%; box-shadow: inset 0 0 0 2px rgba(15,23,42,.10); }

      .demai-arch__ring.is-glow { box-shadow: inset 0 0 0 2px rgba(59,130,246,.35), 0 0 0 6px rgba(59,130,246,.10); }

      .demai-arch__slot { display:grid; place-items:center; width:100%; height:100%; padding:4%; }

      .demai-arch .arch-card {
        width: min(520px, 86%); max-width: 520px; background: var(--card);
        border-radius: 14px; box-shadow: var(--shadow); transform-style: preserve-3d;
        perspective: 1000px; cursor: pointer; position: relative;
      }
      .demai-arch .arch-card__face {
        padding: clamp(14px, 2.4vw, 18px) clamp(16px, 2.8vw, 22px);
        backface-visibility: hidden; transition: transform .6s ease, opacity .6s ease;
        border-radius: 14px; position: absolute; inset: 0; display:grid; gap:.35rem;
        align-content:center; text-align:center;
      }
      .demai-arch .arch-card__front { transform: rotateY(0deg); }
      .demai-arch .arch-card__back  { transform: rotateY(180deg); opacity: 0; }

      .demai-arch .arch-card.is-flipped .arch-card__front { transform: rotateY(180deg); opacity: 0; }
      .demai-arch .arch-card.is-flipped .arch-card__back  { transform: rotateY(360deg); opacity: 1; }

      .demai-arch .arch-card__title {
        font-weight: 800; color: var(--ink);
        display:inline-flex; gap:.5rem; align-items:center; justify-content:center;
        font-size: clamp(1rem, 1.2vw + .8rem, 1.15rem);
      }
      .demai-arch .arch-card__icon { font-size: clamp(1.1rem, 1.4vw + .9rem, 1.35rem); }
      .demai-arch .arch-card__desc { color: var(--muted); line-height:1.55; font-size:.96rem; }
      .demai-arch .arch-card__extra { margin-top:.35rem; font-size:.86rem; color:rgba(15,23,42,.78); background:rgba(226,232,240,.6); border-radius:8px; padding:.5rem .6rem; }

      .demai-arch.nerd-on .arch-card { cursor: default; }
      .demai-arch.nerd-on .arch-card .arch-card__front { display:none; }
      .demai-arch.nerd-on .arch-card .arch-card__back  { position:relative; transform:none; opacity:1; }

      .demai-arch .arch-card:focus-visible { outline:2px solid rgba(59,130,246,.7); outline-offset:4px; }

      @media (max-width: 680px){
        .demai-arch__canvas { aspect-ratio: 4 / 5; }
        .demai-arch__ring--inner { inset: 20%; }
      }
    </style>
    """)

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
