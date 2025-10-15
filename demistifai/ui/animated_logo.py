"""Animated demAI logo helpers."""

from __future__ import annotations

from streamlit.components.v1 import html as components_html

_DEMAI_LOGO_HTML = '''
<div class="demai-hero" role="banner" aria-label="demAI animated title">
  <h1 class="demai-title">
    <span id="demai-base">dem</span><span id="demai-dyn"></span><span class="demai-caret" aria-hidden="true"></span>
  </h1>
  <noscript><h1 class="demai-title noscript">demAI</h1></noscript>
</div>

<style>
  .demai-hero{
    display:flex;align-items:center;justify-content:flex-start;
    padding:0; margin:0;
  }
  .demai-title{
    --grad: linear-gradient(90deg,#111827 0%,#0f172a 30%,#1e3a8a 60%,#0ea5e9 100%);
    font-size: clamp(2.0rem, 4.8vw, 3.4rem);
    line-height:1.05;margin:0;font-weight:800;letter-spacing:.5px;
    background: var(--grad);
    -webkit-background-clip:text;background-clip:text;color:transparent;
    position:relative;white-space:nowrap;
  }
  /* gentle moving sheen */
  .demai-title{background-size:200% 100%;animation:de-sheen 5s ease-in-out infinite;}
  @keyframes de-sheen{
    0%,20% {background-position:0% 0%}
    50%    {background-position:100% 0%}
    100%   {background-position:0% 0%}
  }
  /* caret */
  .demai-caret{width:.08em;height:1.05em;background:currentColor;display:inline-block;
    transform:translateY(.12em);margin-left:.08em;animation:blink 1.05s steps(1) infinite;}
  @keyframes blink{0%,60%{opacity:1}61%,100%{opacity:0}}
  /* dark mode support: inherit color from parent if gradients disabled */
  @media (prefers-reduced-motion: reduce){
    .demai-title{animation:none}
    .demai-caret{animation:none}
  }
  .noscript{color:inherit;background:none}
  .demai-hero{opacity:0;transform:translateY(6px);animation:fadein .6s ease forwards}
  @keyframes fadein{to{opacity:1;transform:translateY(0)}}
</style>

<script>
(function(){
  const FRAME_MARKER = "__FRAME_MARKER__";
  try {
    const frame = window.frameElement;
    if (frame && FRAME_MARKER) {
      frame.setAttribute('data-demai-logo-marker', FRAME_MARKER);
    }
  } catch (err) {
    /* ignore */
  }

  const dyn = document.getElementById('demai-dyn');
  const base = document.getElementById('demai-base');
  if(!dyn || !base) return;

  const steps = [
    { type: "onstrateAI", hold: 800, backToDem: true  },
    { type: "istifyAI",   hold: 800, backToDem: true  },
    { type: "ocratizeAI", hold: 900, backToDem: true  },
    { type: "AI",         hold: 900, backToDem: false }
  ];

  const TYPE_DELAY = 55;     // ms per char typed
  const ERASE_DELAY = 35;    // ms per char deleted
  const BETWEEN_STEPS = 280; // ms pause between steps

  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  async function typeText(t){
    for(const ch of t){ dyn.textContent += ch; await sleep(TYPE_DELAY); }
  }
  async function backToDem(){
    while(dyn.textContent.length){
      dyn.textContent = dyn.textContent.slice(0,-1);
      await sleep(ERASE_DELAY);
    }
  }
  async function run(){
    for(const step of steps){
      await typeText(step.type);
      await sleep(step.hold);
      if(step.backToDem){ await backToDem(); await sleep(BETWEEN_STEPS); }
    }
  }
  requestAnimationFrame(run);
})();
</script>
'''


def render_demai_logo(height: int = 50, *, frame_marker: str = "") -> None:
    """Render the animated demAI title as a Streamlit component."""

    html = _DEMAI_LOGO_HTML.replace("__FRAME_MARKER__", frame_marker)
    components_html(html, height=height)


