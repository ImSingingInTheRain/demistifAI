"""Intro stage hero layout helpers."""

from __future__ import annotations

from textwrap import dedent

from demistifai.constants import LIFECYCLE_RING_HTML

__all__ = [
    "intro_hero_scoped_css",
    "intro_lifecycle_columns",
    "intro_ai_act_quote_wrapper_open",
    "intro_ai_act_quote_wrapper_close",
    "render_intro_hero",
]


def intro_hero_scoped_css() -> str:
    """Return scoped CSS for the intro lifecycle hero."""

    return dedent(
        """
        <style>
            .section-surface.section-surface--hero > div[data-testid="stVerticalBlock"] {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: clamp(1.4rem, 3vw, 2.4rem);
                padding: 0;
            }
            .section-surface.section-surface--hero > div[data-testid="stVerticalBlock"] > div:first-child {
                width: min(100%, 1040px);
            }
            .mw-intro-lifecycle iframe[srcdoc*="demai-hero"] {
                display: block;
                margin: 0 auto;
            }
            .mw-intro-lifecycle {
                position: relative;
                margin: 0 auto clamp(1.8rem, 4vw, 2.8rem);
                max-width: min(1200px, 100%);
                min-height: clamp(420px, 56vh, 520px);
                border-radius: 22px;
                overflow: hidden;
                isolation: isolate;
                border: 1px solid rgba(15, 23, 42, 0.08);
                box-shadow: 0 30px 70px rgba(15, 23, 42, 0.16);
            }
            .mw-intro-lifecycle::before {
                content: "";
                position: absolute;
                inset: 0;
                pointer-events: none;
                background:
                    radial-gradient(circle at top left, rgba(96, 165, 250, 0.24), transparent 58%),
                    radial-gradient(circle at bottom right, rgba(129, 140, 248, 0.2), transparent 62%);
                opacity: 0.9;
            }
            .mw-intro-lifecycle__body {
                position: relative;
                z-index: 1;
                background: rgba(248, 250, 252, 0.96);
                backdrop-filter: blur(18px);
                padding: clamp(1.1rem, 2vw, 1.6rem);
            }
            .mw-intro-lifecycle__grid {
                gap: clamp(1rem, 2.6vw, 1.9rem);
                padding-top: 10px;
                padding-left: 10px;
                padding-right: clamp(1.2rem, 3.6vw, 2.4rem);
                padding-bottom: 10px;
                align-items: stretch;
            }
            .mw-intro-lifecycle__col {
                position: relative;
                display: flex;
                flex-direction: column;
                gap: clamp(0.75rem, 1.6vw, 1.1rem);
                border-radius: 18px;
                padding: clamp(1.1rem, 2.4vw, 1.75rem);
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(226, 232, 240, 0.68));
                border: 1px solid rgba(148, 163, 184, 0.22);
                box-shadow: 0 20px 44px rgba(15, 23, 42, 0.12);
                overflow: hidden;
                min-height: clamp(320px, 48vh, 420px);
            }
            .mw-intro-lifecycle__col::before {
                content: "";
                position: absolute;
                inset: 0;
                border-radius: inherit;
                background: linear-gradient(135deg, rgba(59, 130, 246, 0.18), transparent 65%);
                opacity: 0.75;
                pointer-events: none;
            }
            .mw-intro-lifecycle__col > * {
                position: relative;
                z-index: 1;
                width: 100%;
            }
            .mw-intro-lifecycle__col:has(> .intro-lifecycle-map) {
                padding: clamp(0.65rem, 2vw, 1.1rem);
                background: linear-gradient(180deg, rgba(37, 99, 235, 0.18), rgba(14, 116, 144, 0.1));
                border: 1px solid rgba(37, 99, 235, 0.32);
                box-shadow: 0 26px 56px rgba(37, 99, 235, 0.22);
                align-items: center;
            }
            .mw-intro-lifecycle__col:has(> .intro-lifecycle-map)::before {
                background: radial-gradient(circle at center, rgba(96, 165, 250, 0.42), transparent 72%);
                opacity: 0.68;
            }
            .mw-intro-lifecycle .intro-lifecycle-sidecar {
                display: flex;
                flex-direction: column;
                gap: clamp(0.8rem, 1.8vw, 1.2rem);
                padding-top: 10px;
                padding-left: 10px;
                padding-right: 10px;
                padding-bottom: 10px;
            }
            .mw-intro-lifecycle .intro-lifecycle-sidecar__eyebrow {
                font-size: 0.72rem;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                font-weight: 700;
                color: rgba(15, 23, 42, 0.58);
            }
            .mw-intro-lifecycle .intro-lifecycle-sidecar__title {
                margin: 0;
                font-size: clamp(1.2rem, 2.4vw, 1.45rem);
                font-weight: 700;
                color: #0f172a;
            }
            .mw-intro-lifecycle .intro-lifecycle-sidecar__body {
                margin: 0;
                font-size: 0.98rem;
                line-height: 1.65;
                color: rgba(15, 23, 42, 0.78);
            }
            .mw-intro-lifecycle .intro-lifecycle-sidecar__list {
                margin: 0;
                padding: 0;
                list-style: none;
                display: grid;
                gap: 0.6rem;
            }
            .mw-intro-lifecycle .intro-lifecycle-sidecar__list li {
                display: grid;
                gap: 0.2rem;
            }
            .mw-intro-lifecycle .intro-lifecycle-sidecar__list strong {
                font-weight: 700;
                color: #1d4ed8;
            }
            .mw-intro-lifecycle .intro-start-button-slot {
                margin-top: auto;
                display: flex;
                flex-direction: column;
                gap: 0.6rem;
            }
            .mw-intro-lifecycle .intro-start-button-source {
                display: none;
            }
            .mw-intro-lifecycle .intro-start-button-source.intro-start-button-source--mounted {
                display: block;
            }
            .mw-intro-lifecycle .intro-start-button-source--mounted div[data-testid="stButton"] {
                margin: 0;
                width: 100%;
            }
            .mw-intro-lifecycle .intro-start-button-source--mounted div[data-testid="stButton"] > button {
                margin-top: 0.35rem;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 0.4rem;
                font-weight: 600;
                font-size: 0.95rem;
                border-radius: 999px;
                padding: 0.68rem 1.6rem;
                background: linear-gradient(135deg, #2563eb, #4338ca);
                color: #fff;
                border: none;
                box-shadow: 0 18px 36px rgba(37, 99, 235, 0.3);
                width: 100%;
                text-align: center;
                transition: transform 0.18s ease, box-shadow 0.18s ease;
            }
            .mw-intro-lifecycle .intro-start-button-source--mounted div[data-testid="stButton"] > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 24px 40px rgba(37, 99, 235, 0.34);
            }
            .mw-intro-lifecycle .intro-start-button-source--mounted div[data-testid="stButton"] > button:focus-visible {
                outline: 3px solid rgba(59, 130, 246, 0.45);
                outline-offset: 3px;
            }
            .mw-intro-lifecycle .intro-lifecycle-map {
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
                min-height: 100%;
                align-items: center;
                width: 100%;
            }
            .mw-intro-lifecycle .intro-lifecycle-map #demai-lifecycle.dlc {
                flex: 1;
                width: min(100%, 520px);
                margin-inline: auto;
            }
            @media (max-width: 1024px) {
                .section-surface.section-surface--hero > div[data-testid="stVerticalBlock"] {
                    gap: clamp(1.2rem, 4vw, 2rem);
                }
                .mw-intro-lifecycle {
                    margin-bottom: clamp(1.4rem, 4vw, 2.2rem);
                }
            }
            @media (max-width: 920px) {
                .mw-intro-lifecycle__body {
                    padding: clamp(1rem, 5vw, 1.4rem);
                }
                .mw-intro-lifecycle__col {
                    padding: clamp(1rem, 4vw, 1.45rem);
                }
            }
            @media (max-width: 680px) {
                .section-surface.section-surface--hero > div[data-testid="stVerticalBlock"] > div:first-child {
                    width: 100%;
                }
                .mw-intro-lifecycle .intro-lifecycle-sidecar {
                    text-align: center;
                }
                .mw-intro-lifecycle .intro-lifecycle-sidecar__list {
                    gap: 0.75rem;
                }
            }
        </style>
        """
    )


def _intro_left_column_html() -> str:
    return dedent(
        """
        <div class="intro-lifecycle-sidecar" role="complementary" aria-label="Lifecycle guidance">
            <div class="intro-lifecycle-sidecar__eyebrow">What you'll do</div>
            <h5 class="intro-lifecycle-sidecar__title">Build and use an AI spam detector</h5>
            <p class="intro-lifecycle-sidecar__body">
                In this interactive journey, you’ll build and use your own AI system, an email spam detector. You will experience the key steps of a development lifecycle, step by step: no technical skills are needed.
                Along the way, you’ll uncover how AI systems learn, make predictions, while applying in practice key concepts from the EU AI Act.
            </p>
            <ul class="intro-lifecycle-sidecar__list">
                <li>
                    <strong>Discover your journey</strong>
                    To your right, you’ll find an interactive map showing the full lifecycle of your AI system— this is your guide through this hands-on exploration of responsible and transparent AI.
                </li>
                <li>
                    <strong>Are you ready to make a machine learn?</strong>
                    Click the button below to start your demAI journey!
                </li>
            </ul>

        </div>
        """
    ).strip()


def _intro_right_column_html() -> str:
    return dedent(
        f"""
        <div class="intro-lifecycle-map" role="presentation">
            {LIFECYCLE_RING_HTML}
        </div>
        """
    ).strip()


def intro_lifecycle_columns() -> tuple[str, str]:
    """Return HTML for the lifecycle hero columns."""

    return _intro_left_column_html(), _intro_right_column_html()


def intro_ai_act_quote_wrapper_open() -> str:
    """Return the wrapper HTML (opening fragment) for the Article 3 quote."""

    return dedent(
        """
        <style>
            .ai-act-quote-block {
                position: relative;
                display: flex;
                justify-content: center;
                padding: clamp(1.3rem, 2.8vw, 2.1rem);
                border-radius: 1.75rem;
                background: linear-gradient(135deg, rgba(59, 130, 246, 0.16), rgba(236, 72, 153, 0.12));
                border: 1px solid rgba(37, 99, 235, 0.2);
                box-shadow: 0 24px 52px rgba(37, 99, 235, 0.2);
                overflow: hidden;
            }

            .ai-act-quote-block::before {
                content: "";
                position: absolute;
                inset: 0;
                pointer-events: none;
                background: radial-gradient(circle at top right, rgba(59, 130, 246, 0.2), transparent 55%),
                    radial-gradient(circle at bottom left, rgba(236, 72, 153, 0.2), transparent 60%);
                opacity: 0.9;
            }

            .ai-act-quote-block > div[data-testid="stComponent"] {
                width: 100%;
                margin: 0;
                position: relative;
                z-index: 1;
            }
        </style>
        <div class="ai-act-quote-block" role="region" aria-label="From the EU AI Act, Article 3">
        """
    ).strip()


def intro_ai_act_quote_wrapper_close() -> str:
    """Return the closing wrapper HTML for the Article 3 quote."""

    return "</div>"


def render_intro_hero() -> tuple[str, str, str, str]:
    """Return the scoped CSS and markup needed to render the intro hero."""

    scoped_css = intro_hero_scoped_css()
    left_col, right_col = intro_lifecycle_columns()
    quote_wrapper_open = intro_ai_act_quote_wrapper_open()
    return scoped_css, left_col, right_col, quote_wrapper_open
