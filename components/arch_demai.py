from textwrap import dedent
import html
import streamlit as st
from streamlit.components.v1 import html as components_html


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
      .demai-arch{
        --ink:#0f172a; --muted:rgba(15,23,42,.78);
        --card:#fff; --shadow:0 16px 36px rgba(15,23,42,.14), inset 0 0 0 1px rgba(15,23,42,.06);
        position:relative; border-radius:16px; isolation:isolate;
        background:radial-gradient(90% 90% at 50% 0%, rgba(2,132,199,.10), rgba(99,102,241,.06));
        padding:clamp(12px,2.6vw,20px); margin:.25rem 0 1rem;
      }
    
      .demai-arch__canvas{
        position:relative; width:min(980px,100%); margin:0 auto;
        aspect-ratio:16/9; border-radius:14px; overflow:hidden;
        background:linear-gradient(180deg, rgba(255,255,255,.88), rgba(248,250,252,.72));
        box-shadow:inset 0 0 0 1px rgba(15,23,42,.06);
      }
      @supports not (aspect-ratio:16/9){
        .demai-arch__canvas::before{content:"";display:block;padding-top:56.25%;}
      }
    
      /* RINGS (desktop) ----------------------------------------------------- */
      .demai-arch__ring{ position:absolute; border-radius:18px; display:grid; place-items:center; }
      .demai-arch__ring--outer{ inset:6%;  box-shadow:inset 0 0 0 2px rgba(15,23,42,.06); }
      .demai-arch__ring--mid  { inset:16%; box-shadow:inset 0 0 0 2px rgba(15,23,42,.08); }
      .demai-arch__ring--inner{ inset:32%; box-shadow:inset 0 0 0 2px rgba(15,23,42,.10); }
    
      .demai-arch__ring.is-glow{
        box-shadow:
          inset 0 0 0 2px rgba(59,130,246,.45),
          0 0 0 6px rgba(59,130,246,.12),
          0 8px 28px rgba(59,130,246,.10);
      }
    
      .demai-arch__slot{ display:grid; place-items:center; width:100%; height:100%; padding:3.2%; }
    
      /* CARDS: different sizes per ring + slight vertical offsets ------------ */
      .arch-card{
        background:var(--card); border-radius:16px; box-shadow:var(--shadow);
        position:relative; transform-style:preserve-3d; perspective:1000px;
        cursor:pointer; transition:transform .18s ease;
      }
      .arch-card:hover{ transform:translateY(-2px); }
    
      .arch-card__face{
        position:absolute; inset:0; backface-visibility:hidden;
        padding:clamp(16px,2.8vw,22px) clamp(18px,3vw,26px);
        display:grid; gap:.5rem; align-content:center; text-align:center; border-radius:16px;
        transition:transform .6s ease, opacity .6s ease;
      }
      .arch-card__front{ transform:rotateY(0deg); }
      .arch-card__back { transform:rotateY(180deg); opacity:0; }
    
      .arch-card.is-flipped .arch-card__front{ transform:rotateY(180deg); opacity:0; }
      .arch-card.is-flipped .arch-card__back { transform:rotateY(360deg); opacity:1; }
    
      .arch-card__title{
        font-weight:800; color:var(--ink);
        display:inline-flex; gap:.55rem; align-items:center; justify-content:center;
        font-size:clamp(1.06rem,1.1vw + .95rem,1.25rem);
      }
      .arch-card__icon{ font-size:clamp(1.2rem,1.4vw + 1rem,1.5rem); }
      .arch-card__desc{ color:var(--muted); line-height:1.58; font-size:clamp(.95rem,.35vw + .9rem,1rem); }
      .arch-card__extra{
        margin-top:.4rem; font-size:.9rem; color:rgba(15,23,42,.85);
        background:rgba(226,232,240,.6); border-radius:10px; padding:.55rem .7rem;
      }
    
      /* Per-ring sizing & offsets so cards don’t collide */
      .demai-arch__ring--outer .arch-card{ width:min(64%,620px); min-height:clamp(140px,22vh,190px); transform:translateY(-8%); }
      .demai-arch__ring--mid   .arch-card{ width:min(56%,560px); min-height:clamp(140px,22vh,185px); transform:translateY(3%); }
      .demai-arch__ring--inner .arch-card{ width:min(48%,520px); min-height:clamp(150px,24vh,200px); transform:translateY(14%); }
    
      /* Nerd mode = always expanded backs */
      .demai-arch.nerd-on .arch-card{ cursor:default; }
      .demai-arch.nerd-on .arch-card .arch-card__front{ display:none; }
      .demai-arch.nerd-on .arch-card .arch-card__back{ position:relative; transform:none; opacity:1; }
    
      .arch-card:focus-visible{ outline:2px solid rgba(59,130,246,.7); outline-offset:4px; }
    
      /* MOBILE LAYOUT (≤680px): switch to stacked list ---------------------- */
      @media (max-width:680px){
        .demai-arch__canvas{
          aspect-ratio:auto; padding:10px 10px 14px; display:grid; gap:12px;
          background:linear-gradient(180deg, rgba(255,255,255,.92), rgba(248,250,252,.82));
        }
        /* stack rings; hide decorative borders */
        .demai-arch__ring{
          position:relative; inset:auto; border-radius:14px; box-shadow:none !important;
        }
        .demai-arch__slot{ padding:0; }
        .demai-arch__ring::before{ content:""; display:none; }
    
        /* full-width cards, comfortable height */
        .arch-card{ width:100%; min-height: clamp(150px, 28vh, 240px); transform:none !important; }
        .demai-arch__ring--outer .arch-card,
        .demai-arch__ring--mid   .arch-card,
        .demai-arch__ring--inner .arch-card{ transform:none; }
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

    st_html = getattr(st, "html", None)
    if callable(st_html):
        st_html(full_markup)
        return

    components_html(full_markup, height=560, scrolling=False)
