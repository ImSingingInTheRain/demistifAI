"""Intro stage hero layout helpers that render inside the macOS iframe window."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

from demistifai.ui.components.shared.macos_iframe_window import MacWindowPane

__all__ = [
    "INTRO_HERO_MAP_PANE_ID",
    "INTRO_HERO_SIDECAR_PANE_ID",
    "intro_hero_panes",
    "intro_hero_scoped_css",
    "intro_lifecycle_columns",
    "intro_lifecycle_ring_markup",
    "render_lifecycle_ring_component",
    "render_intro_hero",
]


@dataclass(frozen=True)
class IntroHeroContent:
    """Scoped CSS and pane markup for the welcome hero."""

    css: str
    columns: tuple[str, str]


INTRO_HERO_SIDECAR_PANE_ID = "intro-hero-sidecar"
INTRO_HERO_MAP_PANE_ID = "intro-hero-map"


@dataclass(frozen=True)
class _LifecycleStage:
    """Descriptor for one of the four lifecycle stages."""

    key: str
    icon: str
    title: str
    body: str


_LIFECYCLE_STAGES: tuple[_LifecycleStage, ...] = (
    _LifecycleStage(
        key="prepare",
        icon="📊",
        title="Prepare data",
        body=(
            "Collect a representative dataset, remove sensitive information, and ensure "
            "labels reflect the behaviours you want the system to learn."
        ),
    ),
    _LifecycleStage(
        key="train",
        icon="🧠",
        title="Train",
        body=(
            "Feed the prepared data into your learning pipeline, validate regularly, and "
            "iterate until the model behaves as expected."
        ),
    ),
    _LifecycleStage(
        key="evaluate",
        icon="🧪",
        title="Evaluate",
        body=(
            "Stress-test the model on fresh examples, inspect false positives and negatives, "
            "and adjust thresholds to meet your risk appetite."
        ),
    ),
    _LifecycleStage(
        key="use",
        icon="📬",
        title="Use",
        body=(
            "Deploy to production traffic, monitor outcomes, and collect feedback that feeds "
            "the next training cycle."
        ),
    ),
)


def intro_hero_scoped_css() -> str:
    """Return the scoped stylesheet shared by the hero panes."""

    return dedent(
        """
        <style>
            .intro-hero-pane {
                font-family: "Inter", "SF Pro Text", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                color: #0f172a;
                background: linear-gradient(165deg, rgba(244, 247, 252, 0.92), rgba(219, 234, 254, 0.72));
                border-radius: 28px;
                padding: clamp(1.2rem, 2.4vw, 2rem);
                min-height: 520px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                gap: 1.6rem;
                box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.25);
            }

            .intro-hero-pane--notes {
                background: linear-gradient(165deg, rgba(248, 250, 252, 0.95), rgba(224, 242, 254, 0.75));
            }

            .intro-lifecycle-sidecar {
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }

            .intro-lifecycle-sidecar__eyebrow {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.22em;
                font-weight: 700;
                color: rgba(30, 64, 175, 0.7);
            }

            .intro-lifecycle-sidecar__title {
                font-size: clamp(1.5rem, 2.3vw, 1.9rem);
                margin: 0;
                font-weight: 750;
                line-height: 1.2;
            }

            .intro-lifecycle-sidecar__body {
                margin: 0;
                font-size: 0.95rem;
                line-height: 1.6;
                color: rgba(15, 23, 42, 0.78);
            }

            .intro-lifecycle-sidecar__list {
                margin: 0;
                padding-left: 1.2rem;
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
                color: rgba(15, 23, 42, 0.85);
                font-size: 0.93rem;
            }

            .intro-start-button-slot {
                margin-top: auto;
            }

            .intro-start-button-source {
                display: none;
            }

            .intro-start-button-source.intro-start-button-source--mounted {
                display: block;
            }

            .intro-start-button-source--mounted div[data-testid="stButton"] {
                margin: 0;
                width: 100%;
            }

            .intro-start-button-source--mounted div[data-testid="stButton"] > button {
                width: 100%;
                border-radius: 999px;
                padding: 0.75rem 1.5rem;
                font-weight: 600;
                font-size: 0.95rem;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                gap: 0.4rem;
                background: linear-gradient(140deg, #2563eb, #4338ca);
                color: #fff;
                border: none;
                box-shadow: 0 18px 36px rgba(37, 99, 235, 0.28);
                transition: transform 0.18s ease, box-shadow 0.18s ease;
            }

            .intro-start-button-source--mounted div[data-testid="stButton"] > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 22px 44px rgba(37, 99, 235, 0.32);
            }

            .intro-start-button-source--mounted div[data-testid="stButton"] > button:focus-visible {
                outline: 3px solid rgba(59, 130, 246, 0.65);
                outline-offset: 2px;
            }

            .intro-hero-pane--visual {
                align-items: center;
            }


            .intro-lifecycle {
                width: min(100%, 560px);
                margin: 0 auto;
                display: grid;
                gap: clamp(1.1rem, 2.8vw, 1.6rem);
            }

            .intro-lifecycle__header {
                display: grid;
                gap: 0.35rem;
                text-align: center;
            }

            .intro-lifecycle__title {
                margin: 0;
                font-size: clamp(1.25rem, 2.6vw, 1.4rem);
                font-weight: 750;
            }

            .intro-lifecycle__subtitle {
                margin: 0;
                font-size: clamp(0.86rem, 1.9vw, 0.94rem);
                color: rgba(15, 23, 42, 0.75);
            }

            .intro-lifecycle__ring {
                position: relative;
                aspect-ratio: 1 / 1;
                border-radius: 50%;
                background: radial-gradient(closest-side, rgba(37, 99, 235, 0.08), rgba(56, 189, 248, 0.05));
                box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.08);
                padding: clamp(1.2rem, 4vw, 2.3rem);
                display: grid;
                place-items: center;
            }

            .intro-lifecycle__grid {
                position: relative;
                width: 100%;
                height: 100%;
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                grid-template-rows: repeat(3, minmax(0, 1fr));
                border-radius: 36px;
            }

            .intro-lifecycle__grid::before {
                content: "";
                position: absolute;
                inset: 10%;
                border-radius: 50%;
                box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.24);
                pointer-events: none;
            }

            .intro-lifecycle__stage-input {
                position: absolute;
                opacity: 0;
                pointer-events: none;
            }

            .intro-lifecycle__stage {
                position: relative;
                display: grid;
                gap: 0.35rem;
                align-items: center;
                justify-items: center;
                padding: clamp(0.6rem, 2.2vw, 0.95rem);
                min-width: clamp(104px, 22vw, 148px);
                background: rgba(255, 255, 255, 0.95);
                border-radius: 18px;
                text-align: center;
                font-weight: 600;
                font-size: clamp(0.82rem, 1.8vw, 0.95rem);
                color: rgba(15, 23, 42, 0.82);
                box-shadow: 0 18px 38px rgba(15, 23, 42, 0.14), inset 0 0 0 1px rgba(148, 163, 184, 0.3);
                cursor: pointer;
                transition: transform 0.2s ease, box-shadow 0.2s ease, color 0.2s ease;
                z-index: 1;
            }

            .intro-lifecycle__stage[data-pos="0"] {
                grid-area: 1 / 1 / 2 / 2;
            }

            .intro-lifecycle__stage[data-pos="1"] {
                grid-area: 1 / 3 / 2 / 4;
            }

            .intro-lifecycle__stage[data-pos="2"] {
                grid-area: 3 / 3 / 4 / 4;
            }

            .intro-lifecycle__stage[data-pos="3"] {
                grid-area: 3 / 1 / 4 / 2;
            }

            .intro-lifecycle__icon {
                font-size: clamp(1.4rem, 3.2vw, 1.85rem);
                line-height: 1;
            }

            .intro-lifecycle__label {
                margin: 0;
            }

            .intro-lifecycle__stage:hover,
            .intro-lifecycle__stage-input:checked + .intro-lifecycle__stage {
                transform: translateY(-2px);
                box-shadow: 0 28px 56px rgba(37, 99, 235, 0.22), inset 0 0 0 1px rgba(37, 99, 235, 0.45);
                color: rgba(30, 64, 175, 0.95);
            }

            .intro-lifecycle__stage-input:focus-visible + .intro-lifecycle__stage {
                outline: 3px solid rgba(59, 130, 246, 0.65);
                outline-offset: 4px;
            }

            .intro-lifecycle__details {
                grid-area: 2 / 2 / 3 / 3;
                position: relative;
                display: grid;
                align-items: center;
                justify-items: center;
                padding: clamp(0.4rem, 1.6vw, 0.9rem);
                pointer-events: none;
                z-index: 3;
            }

            .intro-lifecycle__detail {
                position: relative;
                display: none;
                flex-direction: column;
                align-items: flex-start;
                gap: clamp(0.6rem, 1.5vw, 0.9rem);
                padding: clamp(1.05rem, 2.6vw, 1.65rem);
                width: min(100%, clamp(16.5rem, 38vw, 22rem));
                min-height: clamp(10.5rem, 28vw, 13.5rem);
                border-radius: 22px;
                background: rgba(255, 255, 255, 0.97);
                box-shadow: 0 24px 46px rgba(15, 23, 42, 0.2), inset 0 0 0 1px rgba(148, 163, 184, 0.32);
                text-align: left;
                pointer-events: auto;
                z-index: 4;
            }

            .intro-lifecycle__detail-title {
                margin: 0;
                font-size: clamp(1rem, 2.1vw, 1.15rem);
                font-weight: 700;
                color: rgba(15, 23, 42, 0.92);
            }

            .intro-lifecycle__detail-body {
                margin: 0;
                font-size: clamp(0.85rem, 1.9vw, 0.95rem);
                line-height: 1.55;
                color: rgba(15, 23, 42, 0.78);
            }

            .intro-lifecycle__detail-close {
                position: absolute;
                top: 12px;
                right: 12px;
                width: 1px;
                height: 1px;
                padding: 0;
                margin: -1px;
                overflow: hidden;
                clip: rect(0 0 0 0);
                border: 0;
                white-space: nowrap;
                background: none;
                color: inherit;
            }

            #intro-lifecycle-stage-prepare:checked ~ .intro-lifecycle__details .intro-lifecycle__detail[data-stage="prepare"],
            #intro-lifecycle-stage-train:checked ~ .intro-lifecycle__details .intro-lifecycle__detail[data-stage="train"],
            #intro-lifecycle-stage-evaluate:checked ~ .intro-lifecycle__details .intro-lifecycle__detail[data-stage="evaluate"],
            #intro-lifecycle-stage-use:checked ~ .intro-lifecycle__details .intro-lifecycle__detail[data-stage="use"] {
                display: flex;
            }

            @media (max-width: 780px) {
                .intro-hero-pane {
                    padding: clamp(1rem, 4vw, 1.5rem);
                }

                .intro-lifecycle {
                    gap: clamp(1rem, 3.6vw, 1.25rem);
                }

                .intro-lifecycle__title {
                    font-size: clamp(1.1rem, 4.5vw, 1.25rem);
                }

                .intro-lifecycle__ring {
                    padding: clamp(1rem, 7vw, 1.4rem);
                }

                .intro-lifecycle__stage {
                    min-width: clamp(88px, 34vw, 110px);
                    padding: clamp(0.5rem, 3.8vw, 0.75rem);
                    font-size: clamp(0.78rem, 3vw, 0.88rem);
                }

                .intro-lifecycle__icon {
                    font-size: clamp(1.1rem, 5vw, 1.5rem);
                }

                .intro-lifecycle__details {
                    position: absolute;
                    inset: 6%;
                    border-radius: 24px;
                    background: rgba(248, 250, 252, 0.98);
                    box-shadow: 0 28px 52px rgba(15, 23, 42, 0.32);
                    opacity: 0;
                    pointer-events: none;
                    transition: opacity 0.18s ease;
                    z-index: 8;
                }

                .intro-lifecycle__detail {
                    position: absolute;
                    inset: 0;
                    width: auto;
                    min-height: 100%;
                    border-radius: 24px;
                    padding: clamp(1rem, 6vw, 1.8rem);
                    gap: clamp(0.65rem, 4vw, 1rem);
                    display: none;
                    overflow-y: auto;
                    box-shadow: none;
                }

                #intro-lifecycle-stage-prepare:checked ~ .intro-lifecycle__details,
                #intro-lifecycle-stage-train:checked ~ .intro-lifecycle__details,
                #intro-lifecycle-stage-evaluate:checked ~ .intro-lifecycle__details,
                #intro-lifecycle-stage-use:checked ~ .intro-lifecycle__details {
                    opacity: 1;
                    pointer-events: auto;
                }

                .intro-lifecycle__detail-close {
                    position: absolute;
                    width: 36px;
                    height: 36px;
                    margin: 0;
                    clip: auto;
                    background: rgba(15, 23, 42, 0.1);
                    color: rgba(15, 23, 42, 0.85);
                    border-radius: 999px;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1rem;
                    cursor: pointer;
                }

                .intro-lifecycle__detail-close::after {
                    content: "✕";
                    font-weight: 600;
                }

                .intro-lifecycle__detail-title {
                    font-size: clamp(1.05rem, 5vw, 1.25rem);
                }

                .intro-lifecycle__detail-body {
                    font-size: clamp(0.85rem, 4vw, 0.98rem);
                }
            }

            @media (prefers-reduced-motion: reduce) {
                .intro-lifecycle__stage,
                .intro-lifecycle__details {
                    transition: none;
                }
            }
        </style>
        """
    ).strip()



def intro_lifecycle_ring_markup() -> str:
    """Return the lifecycle ring markup rendered in the visual pane."""

    stage_controls: list[str] = []
    detail_items: list[str] = []
    for index, stage in enumerate(_LIFECYCLE_STAGES):
        stage_controls.append(
            dedent(
                f"""
                <input type="radio" class="intro-lifecycle__stage-input" name="intro-lifecycle-stage" id="intro-lifecycle-stage-{stage.key}" value="{stage.key}" aria-labelledby="intro-lifecycle-{stage.key}-title" aria-controls="intro-lifecycle-detail-{stage.key}">
                <label class="intro-lifecycle__stage" data-stage="{stage.key}" data-pos="{index}" for="intro-lifecycle-stage-{stage.key}">
                    <span class="intro-lifecycle__icon" aria-hidden="true">{stage.icon}</span>
                    <span class="intro-lifecycle__label" id="intro-lifecycle-{stage.key}-title">{stage.title}</span>
                </label>
                """
            ).strip()
        )
        detail_items.append(
            dedent(
                f"""
                <article class="intro-lifecycle__detail" data-stage="{stage.key}" id="intro-lifecycle-detail-{stage.key}" aria-labelledby="intro-lifecycle-detail-{stage.key}-title" role="region">
                    <label class="intro-lifecycle__detail-close" for="intro-lifecycle-stage-none" role="button" aria-label="Close stage details"></label>
                    <h5 class="intro-lifecycle__detail-title" id="intro-lifecycle-detail-{stage.key}-title">{stage.title}</h5>
                    <p class="intro-lifecycle__detail-body" id="intro-lifecycle-detail-{stage.key}-body">{stage.body}</p>
                </article>
                """
            ).strip()
        )

    controls_markup = " ".join(stage_controls)
    details_markup = " ".join(detail_items)

    return dedent(
        f"""
        <section class="intro-lifecycle" aria-labelledby="intro-lifecycle-title">
            <div class="intro-lifecycle__header">
                <h4 class="intro-lifecycle__title" id="intro-lifecycle-title">Map the lifecycle of your AI system</h4>
                <p class="intro-lifecycle__subtitle">Each stage flows into the next—complete the loop to operate responsibly.</p>
            </div>
            <div class="intro-lifecycle__ring" role="group" aria-label="AI system lifecycle stages">
                <div class="intro-lifecycle__grid">
                    <input type="radio" class="intro-lifecycle__stage-input" name="intro-lifecycle-stage" id="intro-lifecycle-stage-none" value="" checked aria-label="Hide lifecycle details">
                    {controls_markup}
                    <div class="intro-lifecycle__details" aria-live="polite">
                        {details_markup}
                    </div>
                </div>
            </div>
        </section>
        """
    ).strip()



def intro_lifecycle_columns() -> tuple[str, str]:
    """Return HTML for the lifecycle hero columns."""

    sidecar_html = dedent(
        """
        <div class=\"intro-hero-pane intro-hero-pane--notes\" role=\"complementary\" aria-label=\"Lifecycle guidance\">
            <div class=\"intro-lifecycle-sidecar\">
                <div class=\"intro-lifecycle-sidecar__eyebrow\">What you'll do</div>
                <h5 class=\"intro-lifecycle-sidecar__title\">Build and use an AI spam detector</h5>
                <p class=\"intro-lifecycle-sidecar__body\">
                    In this interactive journey you will assemble a dataset, train a model, evaluate it with fresh signals,
                    and see how it behaves on live emails. Each step links to the next so you can experience the lifecycle
                    end to end without prior machine-learning expertise.
                </p>
                <ul class=\"intro-lifecycle-sidecar__list\">
                    <li><strong>Understand the path ahead.</strong> The lifecycle map shows where you are and what comes next.</li>
                    <li><strong>Experiment safely.</strong> You'll inspect predictions, reflect on risks, and capture lessons for the next loop.</li>
                </ul>
                <div class=\"intro-start-button-slot\">
                    <div class=\"intro-start-button-source\"></div>
                </div>
            </div>
        </div>
        """
    ).strip()

    visual_html = dedent(
        f"""
        <div class=\"intro-hero-pane intro-hero-pane--visual\">
            <div id=\"intro-lifecycle-ring-slot\" aria-live=\"polite\">
                {intro_lifecycle_ring_markup()}
            </div>
        </div>
        """
    ).strip()

    return sidecar_html, visual_html


def intro_hero_panes() -> tuple[MacWindowPane, MacWindowPane]:
    """Return the Mac window panes composing the intro hero."""

    hero_content = render_intro_hero()
    left_html, right_html = hero_content.columns
    scoped_css = hero_content.css
    return (
        MacWindowPane(
            html=left_html.strip(),
            css=scoped_css,
            min_height=520,
            max_width=520,
            pane_id=INTRO_HERO_SIDECAR_PANE_ID,
        ),
        MacWindowPane(
            html=right_html.strip(),
            css=scoped_css,
            min_height=520,
            max_width=720,
            pane_id=INTRO_HERO_MAP_PANE_ID,
        ),
    )


def render_lifecycle_ring_component(*, height: int = 0) -> None:
    """Compatibility shim—ring markup now ships with the hero pane."""

    import streamlit as st

    placeholder = st.empty()
    placeholder.markdown("<!-- Lifecycle ring rendered directly inside the intro hero pane. -->", unsafe_allow_html=True)


def render_intro_hero() -> IntroHeroContent:
    """Return the scoped CSS and HTML columns for the intro hero."""

    return IntroHeroContent(css=intro_hero_scoped_css(), columns=intro_lifecycle_columns())
