"""Minimal animated demAI logo (returns raw HTML)."""

from __future__ import annotations

from streamlit.components.v1 import html as components_html


_LOGO_HTML_TEMPLATE = """
<div class="demai-hero" role="banner" aria-label="demAI animated title">
  <h1 class="demai-title">
    <span id="demai-base">dem</span><span id="demai-dyn"></span><span class="demai-caret" aria-hidden="true"></span>
  </h1>
  <noscript><h1 class="demai-title noscript">demAI</h1></noscript>
</div>

<style>
  .demai-hero {
    display: flex;
    align-items: center;
    padding: 0;
    margin: 0;
  }
  .demai-title {
    font-size: clamp(1.6rem, 4vw, 2.4rem);
    line-height: 1.1;
    margin: 0;
    font-weight: 800;
    letter-spacing: .5px;
    background: linear-gradient(
      90deg,
      #ffffff 0%,
      #e0f2fe 15%,
      #1e3a8a 55%,
      #0ea5e9 100%
    );
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    background-size: 200% 100%;
    animation: sheen 5s ease-in-out infinite;
    white-space: nowrap;
  }
  @keyframes sheen {
    0%, 20% { background-position: 0% 0% }
    50%      { background-position: 100% 0% }
    100%     { background-position: 0% 0% }
  }
  .demai-caret {
    width: .08em;
    height: 1.05em;
    background: currentColor;
    display: inline-block;
    transform: translateY(.12em);
    margin-left: .08em;
    animation: blink 1.05s steps(1) infinite;
  }
  @keyframes blink { 0%,60%{opacity:1} 61%,100%{opacity:0} }
  @media (prefers-reduced-motion: reduce) {
    .demai-title, .demai-caret { animation: none; }
  }
  .noscript { color: inherit; background: none; }
</style>

<script>
(function() {
  try {
    if (window.frameElement && "__FRAME_MARKER__") {
      window.frameElement.setAttribute("data-demai-logo-marker", "__FRAME_MARKER__");
    }
  } catch (e) {}

  const dyn = document.getElementById('demai-dyn');
  if (!dyn) return;

  const steps = [
    { t: "onstrateAI", hold: 700, back: true },
    { t: "istifyAI",   hold: 700, back: true },
    { t: "ocratizeAI", hold: 800, back: true },
    { t: "AI",         hold: 900, back: false }
  ];

  const typeDelay = 55, eraseDelay = 35, between = 220;
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  async function type(s) {
    for (const ch of s) {
      dyn.textContent += ch;
      await sleep(typeDelay);
    }
  }

  async function erase() {
    while (dyn.textContent) {
      dyn.textContent = dyn.textContent.slice(0, -1);
      await sleep(eraseDelay);
    }
  }

  async function run() {
    for (const step of steps) {
      await type(step.t);
      await sleep(step.hold);
      if (step.back) {
        await erase();
        await sleep(between);
      }
    }
  }

  run();
})();
</script>
"""


def demai_logo_html(frame_marker: str = "demai-header") -> str:
    """Return raw HTML for the animated demAI logo."""
    return _LOGO_HTML_TEMPLATE.replace("__FRAME_MARKER__", frame_marker)


def render_demai_logo(height: int = 56, *, frame_marker: str = "demai-inline") -> None:
    """Optional: render the logo as a Streamlit component (if you ever need it inline)."""
    components_html(demai_logo_html(frame_marker), height=height)


