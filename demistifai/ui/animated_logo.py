"""Animated demAI logo helpers."""

from __future__ import annotations

import streamlit as st
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


_HEADER_LOGO_CSS = """
<style>
:root {
    --demai-header-logo-gap: clamp(0.6rem, 2.2vw, 1.05rem);
    --demai-header-logo-width: clamp(9.8rem, 14vw, 13.8rem);
    --demai-header-logo-height: clamp(3.2rem, 4.4vw, 4.5rem);
    --demai-header-logo-width-sm: clamp(8.4rem, 26vw, 11.4rem);
    --demai-header-logo-height-sm: clamp(2.8rem, 6vw, 3.6rem);
}

/* Remove layout gap for the header-mounted iframe wrapper */
div[data-demai-logo-wrapper="true"] {
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
}

iframe[data-demai-logo-marker="header"].demai-header-logo-frame {
    width: var(--demai-header-logo-width) !important;
    height: var(--demai-header-logo-height) !important;
    pointer-events: none;
    border: none !important;
    background: transparent !important;
    flex: 0 0 auto;
    z-index: 1200;
}

header[data-testid="stHeader"] > iframe[data-demai-logo-marker="header"].demai-header-logo-frame:first-child {
    margin-left: 0;
}

header[data-testid="stHeader"] {
    position: relative;
    display: flex;
    align-items: center;
    gap: var(--demai-header-logo-gap);
    padding: clamp(0.5rem, 1.6vw, 1.05rem) clamp(1rem, 3vw, 1.8rem);
    min-height: 72px;
}

header[data-testid="stHeader"] [data-testid="collapsedControl"] {
    display: flex;
    align-items: center;
}

header[data-testid="stHeader"] .stAppToolbar {
    flex: 1 1 auto;
}

header[data-testid="stHeader"] .stAppToolbar > div {
    justify-content: flex-end;
}

@media (max-width: 992px) {
    :root {
        --demai-header-logo-gap: clamp(0.45rem, 2vw, 0.8rem);
    }
    iframe[data-demai-logo-marker="header"].demai-header-logo-frame {
        width: var(--demai-header-logo-width-sm) !important;
        height: var(--demai-header-logo-height-sm) !important;
    }
    header[data-testid="stHeader"] {
        min-height: 68px;
    }
}

@media (max-width: 640px) {
    :root {
        --demai-header-logo-gap: clamp(0.32rem, 2.6vw, 0.6rem);
    }
    iframe[data-demai-logo-marker="header"].demai-header-logo-frame {
        transform: scale(0.82);
        transform-origin: top left;
    }
    header[data-testid="stHeader"] {
        min-height: 64px;
    }
}
</style>
<script>
(function () {
  const STATE_KEY = "__demaiHeaderLogoState";
  const FRAME_SELECTOR = 'iframe[data-demai-logo-marker="header"]';
  const HEADER_SELECTOR = 'header[data-testid="stHeader"]';
  const TOGGLE_SELECTOR = `${HEADER_SELECTOR} [data-testid="collapsedControl"]`;
  const state = window[STATE_KEY] || (window[STATE_KEY] = {});

  function ensureWrapperMarked(state, frame) {
    if (state.wrapper && document.body.contains(state.wrapper)) {
      return;
    }

    const wrapper = frame?.parentElement?.closest('div[data-testid="stVerticalBlock"]');
    if (wrapper && wrapper.getAttribute('data-demai-logo-wrapper') !== 'true') {
      wrapper.setAttribute('data-demai-logo-wrapper', 'true');
      state.wrapper = wrapper;
    }
  }

  function mountFrame() {
    const frame = document.querySelector(FRAME_SELECTOR);
    const header = document.querySelector(HEADER_SELECTOR);

    if (!frame || !header) {
      return false;
    }

    ensureWrapperMarked(state, frame);

    if (!frame.classList.contains('demai-header-logo-frame')) {
      frame.classList.add('demai-header-logo-frame');
      frame.style.position = 'relative';
      frame.style.top = 'auto';
      frame.style.left = 'auto';
    }

    const toggle = header.querySelector(TOGGLE_SELECTOR);
    if (toggle) {
      if (toggle.nextSibling !== frame) {
        toggle.insertAdjacentElement('afterend', frame);
      }
    } else if (header.firstChild !== frame) {
      header.insertAdjacentElement('afterbegin', frame);
    }

    return true;
  }

  function scheduleMount() {
    if (!mountFrame()) {
      window.requestAnimationFrame(scheduleMount);
    }
  }

  state.mount = mountFrame;

  if (!state.observer) {
    state.observer = new MutationObserver(mountFrame);
    state.observer.observe(document.body, { childList: true, subtree: true });
  }

  scheduleMount();
})();
</script>
"""


def mount_demai_header_logo(height: int = 96) -> None:
    """Render the animated demAI logo fixed to the Streamlit header."""

    render_demai_logo(height=height, frame_marker="header")
    st.markdown(_HEADER_LOGO_CSS, unsafe_allow_html=True)
