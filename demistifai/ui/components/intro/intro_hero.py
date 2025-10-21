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
                max-width: 100%;
                margin: 0 auto;
                display: grid;
                gap: 1.4rem;
            }

            .intro-lifecycle__header {
                display: grid;
                gap: 0.4rem;
                text-align: center;
            }

            .intro-lifecycle__title {
                margin: 0;
                font-size: 1.35rem;
                font-weight: 750;
            }

            .intro-lifecycle__subtitle {
                margin: 0;
                font-size: 0.92rem;
                color: rgba(15, 23, 42, 0.75);
            }

            .intro-lifecycle__ring {
                position: relative;
                aspect-ratio: 1 / 1;
                border-radius: 50%;
                background: radial-gradient(92% 92% at 50% 50%, rgba(37, 99, 235, 0.08), rgba(14, 165, 233, 0.05));
                box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.08);
                display: grid;
                place-items: center;
                padding: clamp(2rem, 6vw, 2.8rem);
            }

            .intro-lifecycle__stages {
                list-style: none;
                margin: 0;
                padding: 0;
                position: relative;
                width: 100%;
                height: 100%;
            }

            .intro-lifecycle__stage {
                position: absolute;
                top: 50%;
                left: 50%;
                --stage-translate-x: 0%;
                --stage-translate-y: 0%;
                transform: translate(-50%, -50%)
                    translate(var(--stage-translate-x), var(--stage-translate-y));
                width: clamp(150px, 44%, 180px);
                min-width: clamp(136px, 36vw, 164px);
                background: #fff;
                border-radius: 20px;
                box-shadow: 0 16px 40px rgba(15, 23, 42, 0.14), inset 0 0 0 1px rgba(148, 163, 184, 0.3);
                padding: 0.8rem 1rem;
                text-align: center;
                display: grid;
                gap: 0.45rem;
                transition: transform 0.18s ease, box-shadow 0.18s ease;
            }

            .intro-lifecycle__stage::after {
                content: "";
                position: absolute;
                inset: auto;
                width: 32px;
                height: 32px;
                border-radius: 999px;
                display: grid;
                place-items: center;
                color: rgba(37, 99, 235, 0.85);
                background: #fff;
                box-shadow: 0 12px 24px rgba(15, 23, 42, 0.12), inset 0 0 0 1px rgba(148, 163, 184, 0.3);
            }

            .intro-lifecycle__stage:hover,
            .intro-lifecycle__stage:focus-visible {
                transform: translate(-50%, -50%)
                    translate(var(--stage-translate-x), var(--stage-translate-y))
                    scale(1.04);
                box-shadow: 0 22px 48px rgba(15, 23, 42, 0.18), inset 0 0 0 1px rgba(59, 130, 246, 0.45);
                outline: none;
            }

            .intro-lifecycle__stage[data-pos="0"] {
                --stage-translate-y: -140%;
            }

            .intro-lifecycle__stage[data-pos="1"] {
                --stage-translate-x: 140%;
            }

            .intro-lifecycle__stage[data-pos="2"] {
                --stage-translate-y: 140%;
            }

            .intro-lifecycle__stage[data-pos="3"] {
                --stage-translate-x: -140%;
            }

            .intro-lifecycle__stage[data-pos="0"]::after {
                content: "➝";
                bottom: -18px;
                left: 50%;
                transform: translate(-50%, 50%);
            }

            .intro-lifecycle__stage[data-pos="1"]::after {
                content: "➝";
                top: 50%;
                right: -18px;
                transform: translate(50%, -50%) rotate(90deg);
            }

            .intro-lifecycle__stage[data-pos="2"]::after {
                content: "➝";
                top: -18px;
                left: 50%;
                transform: translate(-50%, -50%) rotate(180deg);
            }

            .intro-lifecycle__stage[data-pos="3"]::after {
                content: "➝";
                top: 50%;
                left: -18px;
                transform: translate(-50%, -50%) rotate(270deg);
            }

            .intro-lifecycle__stage-index {
                font-size: 0.7rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-weight: 700;
                color: rgba(37, 99, 235, 0.72);
            }

            .intro-lifecycle__icon {
                font-size: 1.3rem;
            }

            .intro-lifecycle__label {
                margin: 0;
                font-size: 1rem;
                font-weight: 700;
                color: #0f172a;
            }

            .intro-lifecycle__copy {
                margin: 0;
                font-size: 0.82rem;
                line-height: 1.45;
                color: rgba(15, 23, 42, 0.7);
            }

            .intro-lifecycle__loop {
                width: 120px;
                height: 120px;
                border-radius: 32px;
                background: linear-gradient(160deg, rgba(59, 130, 246, 0.18), rgba(14, 165, 233, 0.12));
                box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.25);
                display: grid;
                place-items: center;
                font-size: 2rem;
                color: rgba(30, 64, 175, 0.55);
            }

            .intro-lifecycle__legend {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 1rem;
            }

            .intro-lifecycle__legend-item {
                background: rgba(248, 250, 252, 0.9);
                border-radius: 18px;
                padding: 1rem;
                box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.22);
                display: grid;
                gap: 0.4rem;
            }

            .intro-lifecycle__legend-title {
                font-weight: 700;
                margin: 0;
                font-size: 0.92rem;
                color: #0f172a;
            }

            .intro-lifecycle__legend-body {
                margin: 0;
                font-size: 0.84rem;
                line-height: 1.45;
                color: rgba(15, 23, 42, 0.75);
            }

            @media (max-width: 820px) {
                .intro-hero-pane {
                    min-height: auto;
                }

                .intro-lifecycle__ring {
                    padding: clamp(1.6rem, 7vw, 2.2rem);
                }

                .intro-lifecycle__stages {
                    display: grid;
                    gap: 1rem;
                }

                .intro-lifecycle__stage {
                    position: static;
                    transform: none;
                    width: 100%;
                    min-width: 0;
                }

                .intro-lifecycle__stage::after {
                    display: none;
                }

                .intro-lifecycle__legend {
                    grid-template-columns: 1fr;
                }
            }

            @media (prefers-reduced-motion: reduce) {
                .intro-lifecycle__stage {
                    transition: none;
                }
            }
        </style>
        """
    ).strip()


def intro_lifecycle_ring_markup() -> str:
    """Return the lifecycle ring markup rendered in the visual pane."""

    stage_items = []
    legend_items = []
    for index, stage in enumerate(_LIFECYCLE_STAGES):
        stage_items.append(
            dedent(
                f"""
                <li class=\"intro-lifecycle__stage\" data-stage=\"{stage.key}\" data-pos=\"{index}\" tabindex=\"0\">
                    <span class=\"intro-lifecycle__stage-index\">Step {index + 1}</span>
                    <span class=\"intro-lifecycle__icon\" aria-hidden=\"true\">{stage.icon}</span>
                    <p class=\"intro-lifecycle__label\" id=\"intro-lifecycle-{stage.key}-title\">{stage.title}</p>
                    <p class=\"intro-lifecycle__copy\" id=\"intro-lifecycle-{stage.key}-desc\">{stage.body}</p>
                </li>
                """
            ).strip()
        )
        legend_items.append(
            dedent(
                f"""
                <div class=\"intro-lifecycle__legend-item\" data-stage=\"{stage.key}\">
                    <p class=\"intro-lifecycle__legend-title\">{stage.title}</p>
                    <p class=\"intro-lifecycle__legend-body\">{stage.body}</p>
                </div>
                """
            ).strip()
        )

    return dedent(
        f"""
        <section class=\"intro-lifecycle\" aria-labelledby=\"intro-lifecycle-title\">
            <div class=\"intro-lifecycle__header\">
                <h4 class=\"intro-lifecycle__title\" id=\"intro-lifecycle-title\">Map the lifecycle of your AI system</h4>
                <p class=\"intro-lifecycle__subtitle\">Each stage flows into the next—complete the loop to operate responsibly.</p>
            </div>
            <div class=\"intro-lifecycle__ring\" role=\"presentation\">
                <div class=\"intro-lifecycle__loop\" aria-hidden=\"true\">↺</div>
                <ul class=\"intro-lifecycle__stages\" role=\"list\">
                    {' '.join(stage_items)}
                </ul>
            </div>
            <div class=\"intro-lifecycle__legend\">
                {' '.join(legend_items)}
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
