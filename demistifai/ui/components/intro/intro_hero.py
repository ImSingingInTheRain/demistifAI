"""Intro stage hero layout helpers."""

from __future__ import annotations

import inspect
import json
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
    """Container for the intro hero columns and their scoped CSS."""

    css: str
    columns: tuple[str, str]

INTRO_HERO_SIDECAR_PANE_ID = "intro-hero-sidecar"
INTRO_HERO_MAP_PANE_ID = "intro-hero-map"

_LIFECYCLE_RING_HTML = dedent(
    """\
    <!-- demAI Lifecycle Component (self-contained) -->
    <div id="demai-lifecycle" class="dlc" aria-label="AI system lifecycle overview">
        <style>
            /* ===== Scoped to #demai-lifecycle =================================== */
            #demai-lifecycle.dlc {
                --ring-size: clamp(250px, min(40vw, calc(100% - 2rem)), 460px);
                --square-inset: clamp(16%, calc(50% - 180px), 22%);
                --elev: 0 14px 30px rgba(15, 23, 42, 0.12);
                --stroke: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
                --tip-max-width: 240px;
                margin-top: 0.5rem;
                font-family: system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif;
            }
            #demai-lifecycle .dlc-title {
                margin: 0 0 0.4rem 0;
            }
            #demai-lifecycle .dlc-sub {
                margin: 0 0 0.8rem 0;
                color: rgba(15, 23, 42, 0.78);
            }

            /* Ring */
            #demai-lifecycle .ring {
                position: relative;
                width: min(var(--ring-size), 100%);
                max-width: min(var(--ring-size), 100%);
                aspect-ratio: 1 / 1;
                margin: clamp(0.8rem, 2vw, 1.1rem) auto 0;
                border-radius: 50%;
                background: radial-gradient(92% 92% at 50% 50%, rgba(99, 102, 241, 0.10), rgba(14, 165, 233, 0.06));
                box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.06);
                isolation: isolate;
            }
            #demai-lifecycle .ring::after {
                content: "";
                position: absolute;
                inset: var(--square-inset);
                border-radius: 24px;
                pointer-events: none;
            }
            @supports not (aspect-ratio: 1 / 1) {
                #demai-lifecycle .ring::before {
                    content: "";
                    display: block;
                    padding-top: 100%;
                }
            }

            /* Polar placement (keeps arrows upright) */
            #demai-lifecycle .arrow,
            #demai-lifecycle .loop {
                position: absolute;
                transform-origin: center center;
            }

            /* Nodes (tiles) */
            #demai-lifecycle .node {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                display: grid;
                place-items: center;
                gap: 0.25rem;
                padding: 0.65rem 0.9rem;
                min-width: 136px;
                text-align: center;
                background: #fff;
                border-radius: 1rem;
                box-shadow: var(--elev), var(--stroke);
                transition: transform 0.18s ease, box-shadow 0.18s ease;
                cursor: pointer;
            }
            #demai-lifecycle .node .icon {
                font-size: 1.28rem;
            }
            #demai-lifecycle .node .title {
                font-weight: 800;
                color: #0f172a;
                font-size: 0.98rem;
            }
            #demai-lifecycle .node:hover,
            #demai-lifecycle .node:focus-visible,
            #demai-lifecycle .node.active {
                transform: translate(-50%, -50%) scale(1.04);
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18), var(--stroke);
                outline: none;
            }
            #demai-lifecycle .corner-nw {
                top: var(--square-inset);
                left: var(--square-inset);
            }
            #demai-lifecycle .corner-ne {
                top: var(--square-inset);
                left: calc(100% - var(--square-inset));
            }
            #demai-lifecycle .corner-se {
                top: calc(100% - var(--square-inset));
                left: calc(100% - var(--square-inset));
            }
            #demai-lifecycle .corner-sw {
                top: calc(100% - var(--square-inset));
                left: var(--square-inset);
            }

            /* Arrows */
            #demai-lifecycle .arrow {
                --arrow-top: 50%;
                --arrow-left: 50%;
                --angle: 0deg;
                top: var(--arrow-top);
                left: var(--arrow-left);
                transform: translate(-50%, -50%) rotate(var(--angle));
                width: 38px;
                height: 38px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 999px;
                background: #fff;
                color: rgba(30, 64, 175, 0.85);
                box-shadow: 0 12px 22px rgba(15, 23, 42, 0.10), var(--stroke);
                pointer-events: none;
                user-select: none;
            }

            #demai-lifecycle .arrow-top {
                --arrow-top: calc(var(--square-inset));
                --arrow-left: 50%;
                --angle: 0deg;
            }
            #demai-lifecycle .arrow-right {
                --arrow-top: 50%;
                --arrow-left: calc(100% - var(--square-inset));
                --angle: 90deg;
            }
            #demai-lifecycle .arrow-bottom {
                --arrow-top: calc(100% - var(--square-inset));
                --arrow-left: 50%;
                --angle: 180deg;
            }
            #demai-lifecycle .arrow-left {
                --arrow-top: 50%;
                --arrow-left: calc(var(--square-inset));
                --angle: 270deg;
            }

            /* Loop glyph */
            #demai-lifecycle .loop {
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 2rem;
                color: rgba(30, 64, 175, 0.55);
            }

            /* Tooltips */
            #demai-lifecycle .tip {
                position: absolute;
                inset: var(--tip-top, auto) auto var(--tip-bottom, -0.6rem) 50%;
                transform: translate(var(--tip-translate-x, -50%), var(--tip-translate-y, 100%));
                width: min(var(--tip-max-width, 240px), 88vw);
                padding: 0.6rem 0.7rem;
                border-radius: 0.65rem;
                background: #0f172a;
                color: #fff;
                font-size: 0.82rem;
                line-height: 1.35;
                box-shadow: 0 14px 28px rgba(15, 23, 42, 0.25);
                display: none;
                pointer-events: none;
                z-index: 3;
            }
            #demai-lifecycle .node:hover .tip,
            #demai-lifecycle .node:focus .tip,
            #demai-lifecycle .node.active .tip {
                display: block;
            }

            /* Angle utilities */
            #demai-lifecycle .pos-0 {
                --angle: 0deg;
            }
            #demai-lifecycle .pos-45 {
                --angle: 45deg;
            }
            #demai-lifecycle .pos-90 {
                --angle: 90deg;
            }
            #demai-lifecycle .pos-135 {
                --angle: 135deg;
            }
            #demai-lifecycle .pos-180 {
                --angle: 180deg;
            }
            #demai-lifecycle .pos-225 {
                --angle: 225deg;
            }
            #demai-lifecycle .pos-270 {
                --angle: 270deg;
            }
            #demai-lifecycle .pos-315 {
                --angle: 315deg;
            }
            #demai-lifecycle .pos-330 {
                --angle: 330deg;
            }

            /* Legend */
            #demai-lifecycle .legend {
                display: grid;
                gap: 0.9rem;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                margin-top: 1.2rem;
            }
            #demai-lifecycle .legend .item {
                border-radius: 16px;
                padding: 0.85rem 1rem 0.95rem;
                background: linear-gradient(155deg, rgba(248, 250, 252, 0.95), rgba(226, 232, 240, 0.55));
                box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.25);
                display: grid;
                gap: 0.45rem;
                transition: box-shadow 0.18s ease, transform 0.18s ease;
            }
            #demai-lifecycle .legend .item.active {
                box-shadow: 0 12px 26px rgba(15, 23, 42, 0.12), inset 0 0 0 1px rgba(59, 130, 246, 0.45);
                transform: translateY(-2px);
            }
            #demai-lifecycle .legend .head {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            #demai-lifecycle .legend .i {
                font-size: 1.1rem;
            }
            #demai-lifecycle .legend .t {
                font-size: 0.95rem;
                font-weight: 700;
                color: #0f172a;
            }
            #demai-lifecycle .legend .b {
                margin: 0;
                font-size: 0.9rem;
                line-height: 1.55;
                color: rgba(15, 23, 42, 0.78);
            }

            /* Motion & responsiveness */
            @media (max-width: 1100px) {
                #demai-lifecycle .legend {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
            @media (max-width: 860px) {
                #demai-lifecycle {
                    --ring-size: clamp(230px, min(58vw, calc(100% - 1.6rem)), 360px);
                    --square-inset: clamp(18%, calc(50% - 165px), 26%);
                }
            }
            @media (max-width: 720px) {
                #demai-lifecycle {
                    --ring-size: clamp(220px, min(66vw, calc(100% - 1.3rem)), 340px);
                    --square-inset: clamp(20%, calc(50% - 150px), 28%);
                    --tip-max-width: 220px;
                }
                #demai-lifecycle .node {
                    min-width: 120px;
                    padding: 0.56rem 0.72rem;
                    gap: 0.2rem;
                }
                #demai-lifecycle .node .icon {
                    font-size: 1.12rem;
                }
                #demai-lifecycle .node .title {
                    font-size: 0.88rem;
                }
                #demai-lifecycle .arrow {
                    width: 32px;
                    height: 32px;
                }
                #demai-lifecycle .legend {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 0.75rem;
                }
            }
            @media (max-width: 560px) {
                #demai-lifecycle {
                    --ring-size: clamp(205px, min(78vw, calc(100% - 1rem)), 310px);
                    --square-inset: clamp(23%, calc(50% - 130px), 31%);
                    --tip-max-width: 214px;
                }
                #demai-lifecycle .node {
                    min-width: 108px;
                    padding: 0.5rem 0.62rem;
                    border-radius: 0.9rem;
                }
                #demai-lifecycle .node .icon {
                    font-size: 1.02rem;
                }
                #demai-lifecycle .node .title {
                    font-size: 0.8rem;
                }
                #demai-lifecycle .tip {
                    font-size: 0.72rem;
                    line-height: 1.4;
                    border-radius: 0.58rem;
                    padding: 0.5rem 0.6rem;
                }
            }
            @media (max-width: 440px) {
                #demai-lifecycle {
                    --ring-size: clamp(192px, min(88vw, calc(100% - 0.75rem)), 270px);
                    --square-inset: clamp(27%, calc(50% - 112px), 34%);
                    --tip-max-width: 200px;
                }
                #demai-lifecycle .node {
                    min-width: 100px;
                    padding: 0.46rem 0.55rem;
                    gap: 0.16rem;
                }
                #demai-lifecycle .node .icon {
                    font-size: 0.95rem;
                }
                #demai-lifecycle .node .title {
                    font-size: 0.76rem;
                    letter-spacing: -0.01em;
                }
            }
            @media (max-width: 360px) {
                #demai-lifecycle {
                    --tip-max-width: 180px;
                }
                #demai-lifecycle .node {
                    min-width: 96px;
                }
                #demai-lifecycle .legend {
                    grid-template-columns: 1fr;
                }
                #demai-lifecycle .legend .item {
                    transition: none;
                }
            }
            @media (prefers-reduced-motion: reduce) {
                #demai-lifecycle .node {
                    transition: none;
                }
                #demai-lifecycle .legend .item {
                    transition: none;
                }
            }
        </style>

        <h4 class="dlc-title">Map the lifecycle of your AI system</h4>
        <p class="dlc-sub">Progress through the four interconnected stages below. Each phase feeds the next, forming a loop you can revisit as the system evolves.</p>

        <!-- Ring -->
        <div class="ring" role="presentation">
            <!-- Nodes anchored to the guide square corners -->
            <button class="node corner-nw" data-stage="prepare" aria-describedby="desc-prepare">
                <span class="icon" aria-hidden="true">📊</span>
                <span class="title" id="title-prepare">Prepare Data</span>
                <span class="tip" role="tooltip" id="tip-prepare">Gather representative emails, label them carefully, and scrub PII so the model learns from balanced, trustworthy examples.</span>
            </button>

            <div class="arrow arrow-top" aria-hidden="true">➝</div>

            <button class="node corner-ne" data-stage="train" aria-describedby="desc-train">
                <span class="icon" aria-hidden="true">🧠</span>
                <span class="title" id="title-train">Train</span>
                <span class="tip" role="tooltip" id="tip-train">
                    Feed the curated dataset into your learning pipeline, keep a validation split aside, and iterate until performance stabilises.
                </span>
            </button>

            <div class="arrow arrow-right" aria-hidden="true">➝</div>

            <button class="node corner-se" data-stage="evaluate" aria-describedby="desc-evaluate">
                <span class="icon" aria-hidden="true">🧪</span>
                <span class="title" id="title-evaluate">Evaluate</span>
                <span class="tip" role="tooltip" id="tip-evaluate">
                    Inspect precision and recall, review borderline decisions, and tune thresholds to reflect your risk posture.
                </span>
            </button>

            <div class="arrow arrow-bottom" aria-hidden="true">➝</div>

            <button class="node corner-sw" data-stage="use" aria-describedby="desc-use">
                <span class="icon" aria-hidden="true">📬</span>
                <span class="title" id="title-use">Use</span>
                <span class="tip" role="tooltip" id="tip-use">
                    Deploy the model to live traffic, monitor its calls in context, and capture feedback to enrich the next training loop.
                </span>
            </button>

            <div class="arrow arrow-left" aria-hidden="true">➝</div>
            <div class="loop" aria-hidden="true">↺</div>
        </div>

        <script>
            (function () {
                const root = document.getElementById('demai-lifecycle');
                if (!root) return;

                const nodes = Array.from(root.querySelectorAll('.node[data-stage]'));
                const cards = Array.from(root.querySelectorAll('.legend .item[data-stage]'));
                const compactQuery = window.matchMedia('(max-width: 600px)');
                const alreadyBound = root.dataset.lifecycleBound === '1';
                let activeStage = '';

                function resetTip(node) {
                    const tip = node.querySelector('.tip');
                    if (!tip) return;
                    tip.style.removeProperty('--tip-top');
                    tip.style.removeProperty('--tip-bottom');
                    tip.style.removeProperty('--tip-translate-x');
                    tip.style.removeProperty('--tip-translate-y');
                    tip.style.removeProperty('--tip-max-width');
                }

                function positionTip(node) {
                    const tip = node.querySelector('.tip');
                    if (!tip) return;

                    resetTip(node);

                    const nodeRect = node.getBoundingClientRect();
                    const rootRect = root.getBoundingClientRect();
                    const nodeCenterY = nodeRect.top + nodeRect.height / 2;
                    const rootCenterY = rootRect.top + rootRect.height / 2;
                    let orientAbove = nodeCenterY > rootCenterY;
                    const isCompact = compactQuery.matches;

                    if (isCompact) {
                        orientAbove = false;
                        tip.style.setProperty('--tip-bottom', '-0.45rem');
                        tip.style.setProperty('--tip-translate-y', 'calc(100% + 6px)');
                        tip.style.setProperty('--tip-translate-x', '-50%');
                        tip.style.setProperty('--tip-max-width', 'min(200px, 90vw)');
                    } else if (orientAbove) {
                        tip.style.setProperty('--tip-top', '-0.6rem');
                        tip.style.setProperty('--tip-bottom', 'auto');
                        tip.style.setProperty('--tip-translate-y', '-100%');
                    } else {
                        tip.style.removeProperty('--tip-top');
                        tip.style.setProperty('--tip-bottom', '-0.6rem');
                        tip.style.setProperty('--tip-translate-y', '100%');
                    }

                    window.requestAnimationFrame(() => {
                        const rect = tip.getBoundingClientRect();
                        const padding = 16;

                        const rootLeft = rootRect.left + padding;
                        const rootRight = rootRect.right - padding;
                        const overflowLeft = Math.max(0, rootLeft - rect.left);
                        const overflowRight = Math.max(0, rect.right - rootRight);
                        let translateX = '-50%';

                        if (overflowLeft > 0) {
                            translateX = `calc(-50% + ${overflowLeft}px)`;
                        } else if (overflowRight > 0) {
                            translateX = `calc(-50% - ${overflowRight}px)`;
                        }
                        tip.style.setProperty('--tip-translate-x', translateX);

                        const viewportTop = padding;
                        const viewportBottom = window.innerHeight - padding;
                        const overflowTop = Math.max(0, viewportTop - rect.top);
                        const overflowBottom = Math.max(0, rect.bottom - viewportBottom);
                        const baseTranslate = orientAbove ? '-100%' : (isCompact ? 'calc(100% + 6px)' : '100%');
                        let translateY = tip.style.getPropertyValue('--tip-translate-y') || baseTranslate;

                        if (overflowTop > 0) {
                            translateY = orientAbove
                                ? `calc(-100% + ${overflowTop}px)`
                                : (isCompact
                                    ? `calc(100% + 6px + ${overflowTop}px)`
                                    : `calc(100% + ${overflowTop}px)`);
                        } else if (overflowBottom > 0) {
                            translateY = orientAbove
                                ? `calc(-100% - ${overflowBottom}px)`
                                : (isCompact
                                    ? `calc(100% + 6px - ${overflowBottom}px)`
                                    : `calc(100% - ${overflowBottom}px)`);
                        }
                        tip.style.setProperty('--tip-translate-y', translateY);
                    });
                }

                function setActive(stage) {
                    activeStage = stage;
                    nodes.forEach((node) => {
                        const isActive = node.dataset.stage === stage;
                        node.classList.toggle('active', isActive);
                        if (!isActive) {
                            resetTip(node);
                        }
                    });
                    cards.forEach((card) => card.classList.toggle('active', card.dataset.stage === stage));

                    if (stage) {
                        const activeNode = nodes.find((node) => node.dataset.stage === stage);
                        if (activeNode) {
                            positionTip(activeNode);
                        }
                    }
                }

                function positionActiveTip() {
                    if (!activeStage) return;
                    const activeNode = nodes.find((node) => node.dataset.stage === activeStage && node.classList.contains('active'));
                    if (activeNode) {
                        positionTip(activeNode);
                    }
                }

                if (alreadyBound) {
                    positionActiveTip();
                    return;
                }

                root.dataset.lifecycleBound = '1';

                // Hover/focus sync
                nodes.forEach((node) => {
                    const stage = node.dataset.stage;
                    node.addEventListener('mouseenter', () => setActive(stage));
                    node.addEventListener('focus', () => setActive(stage));
                    node.addEventListener('mouseleave', () => setActive(''));
                    node.addEventListener('blur', () => setActive(''));

                    // Tap toggle (mobile)
                    node.addEventListener('click', (event) => {
                        const isActive = node.classList.contains('active');
                        setActive(isActive ? '' : stage);

                        // avoid scrolling on double-tap zoom
                        event.preventDefault();
                    });
                });

                root.addEventListener('mouseleave', () => setActive(''));

                // Allow legend hover to light the ring
                cards.forEach((card) => {
                    const stage = card.dataset.stage;
                    card.addEventListener('mouseenter', () => setActive(stage));
                    card.addEventListener('mouseleave', () => setActive(''));
                    card.addEventListener('click', () => setActive(stage));
                });

                window.addEventListener('resize', positionActiveTip, { passive: true });
                window.addEventListener('scroll', positionActiveTip, { passive: true });
                if (typeof compactQuery.addEventListener === 'function') {
                    compactQuery.addEventListener('change', positionActiveTip);
                } else if (typeof compactQuery.addListener === 'function') {
                    compactQuery.addListener(positionActiveTip);
                }

                // Start with "Prepare" highlighted on large screens; none on small
                const isSmall = window.matchMedia('(max-width:768px)').matches;
                if (!isSmall) setActive('prepare');
            })();
        </script>
    </div>
    """
)


def intro_hero_scoped_css() -> str:
    """Return scoped CSS for the intro lifecycle hero panes."""

    return dedent(
        """
        <style>
            .intro-hero-pane {
                font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                color: #0f172a;
                line-height: 1.6;
                display: flex;
                flex-direction: column;
                gap: clamp(0.85rem, 1.8vw, 1.3rem);
                padding: clamp(1.1rem, 2.4vw, 1.7rem);
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(226, 232, 240, 0.76));
                border: 1px solid rgba(148, 163, 184, 0.22);
                box-shadow: 0 20px 44px rgba(15, 23, 42, 0.12);
                min-height: clamp(320px, 48vh, 420px);
            }
            .intro-hero-pane--visual {
                padding: clamp(0.75rem, 2.1vw, 1.2rem);
                align-items: center;
                justify-content: center;
                background: linear-gradient(180deg, rgba(37, 99, 235, 0.18), rgba(14, 165, 233, 0.12));
                border: 1px solid rgba(37, 99, 235, 0.32);
                box-shadow: 0 26px 56px rgba(37, 99, 235, 0.18);
            }
            .intro-hero-pane--notes {
                gap: clamp(0.75rem, 1.7vw, 1.2rem);
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.94), rgba(226, 232, 240, 0.82));
            }
            .intro-lifecycle-sidecar {
                display: flex;
                flex-direction: column;
                gap: clamp(0.85rem, 1.9vw, 1.35rem);
            }
            .intro-lifecycle-sidecar__eyebrow {
                font-size: 0.72rem;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                font-weight: 700;
                color: rgba(15, 23, 42, 0.6);
            }
            .intro-lifecycle-sidecar__title {
                margin: 0;
                font-size: clamp(1.2rem, 2.4vw, 1.45rem);
                font-weight: 700;
                color: #0f172a;
                line-height: 1.3;
            }
            .intro-lifecycle-sidecar__body {
                margin: 0;
                font-size: 0.98rem;
                line-height: 1.65;
                color: rgba(15, 23, 42, 0.78);
            }
            .intro-lifecycle-sidecar__list {
                margin: 0;
                padding: 0;
                list-style: none;
                display: grid;
                gap: 0.6rem;
            }
            .intro-lifecycle-sidecar__list li {
                display: grid;
                gap: 0.2rem;
            }
            .intro-lifecycle-sidecar__list strong {
                font-weight: 700;
                color: #1d4ed8;
            }
            .intro-start-button-slot {
                margin-top: auto;
                display: flex;
                flex-direction: column;
                gap: 0.6rem;
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
            .intro-start-button-source--mounted div[data-testid="stButton"] > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 22px 42px rgba(37, 99, 235, 0.32);
            }
            .intro-start-button-source--mounted div[data-testid="stButton"] > button:focus-visible {
                outline: 3px solid rgba(59, 130, 246, 0.65);
                outline-offset: 2px;
            }
            .intro-hero-pane--visual .intro-lifecycle-map {
                width: 100%;
                max-width: 520px;
                min-height: clamp(320px, 48vh, 420px);
                display: grid;
                place-items: center;
                background: transparent;
            }
            @media (max-width: 760px) {
                .intro-hero-pane {
                    min-height: auto;
                    padding: clamp(1rem, 3vw, 1.4rem);
                }
                .intro-hero-pane--visual .intro-lifecycle-map {
                    min-height: clamp(260px, 60vw, 380px);
                }
            }
        </style>
        """
    ).strip()


def _intro_left_column_html() -> str:
    return dedent(
        """
        <div class="intro-hero-pane intro-hero-pane--notes" role="complementary" aria-label="Lifecycle guidance">
            <div class="intro-lifecycle-sidecar">
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

                <div class="intro-start-button-slot">
                    <div class="intro-start-button-source"></div>
                </div>
            </div>
        </div>
        """
    ).strip()


def _intro_right_column_html() -> str:
    return dedent(
        f"""
        <div class="intro-hero-pane intro-hero-pane--visual">
            <div class="intro-lifecycle-map" role="presentation" data-intro-lifecycle-slot="1" id="intro-lifecycle-ring-slot"></div>
        </div>
        """
    ).strip()


def intro_lifecycle_columns() -> tuple[str, str]:
    """Return HTML for the lifecycle hero columns."""

    return _intro_left_column_html(), _intro_right_column_html()


def intro_hero_panes() -> tuple[MacWindowPane, MacWindowPane]:
    """Return the MacWindow panes that compose the intro hero."""

    hero_content = render_intro_hero()
    left_html, right_html = hero_content.columns
    scoped_css = hero_content.css
    return (
        MacWindowPane(
            html=left_html,
            css=scoped_css,
            min_height=520,
            pane_id=INTRO_HERO_SIDECAR_PANE_ID,
        ),
        MacWindowPane(
            html=right_html,
            css=scoped_css,
            min_height=520,
            pane_id=INTRO_HERO_MAP_PANE_ID,
        ),
    )


def intro_lifecycle_ring_markup() -> str:
    """Return the raw markup for the lifecycle ring widget."""

    return _LIFECYCLE_RING_HTML


def render_lifecycle_ring_component(*, height: int = 0) -> None:
    """Render the lifecycle ring bundle inside the intro hero placeholder."""

    try:
        import streamlit as st
    except ModuleNotFoundError as err:  # pragma: no cover - streamlit should exist at runtime
        raise RuntimeError("Streamlit is required to render the lifecycle ring component.") from err

    html_renderer = getattr(st, "html", None)
    renderer_supports_height = False
    if html_renderer is None:
        from streamlit.components.v1 import html as components_html

        html_renderer = components_html
        renderer_supports_height = True
    else:
        try:
            renderer_supports_height = "height" in inspect.signature(html_renderer).parameters
        except (TypeError, ValueError):  # pragma: no cover - signature introspection can fail on some callables
            renderer_supports_height = False

    # Prevent ``</script>`` tokens in the lifecycle markup from terminating the
    # embedding script tag prematurely when injected into Streamlit's iframe.
    escaped_markup = _LIFECYCLE_RING_HTML.replace("</script>", "<\\/script>")
    bundle_markup = json.dumps(escaped_markup)
    script_template = dedent(
        """
        <script>
            (function() {
                const parentDoc = window.parent && window.parent.document ? window.parent.document : document;
                const markup = __BUNDLE_MARKUP__;
                const mapPaneId = 'intro-hero-map';

                function getPaneDocument() {
                    const selector = `iframe[data-pane-id="${mapPaneId}"]`;
                    const iframe = parentDoc.querySelector(selector);
                    if (!iframe) {
                        return null;
                    }
                    try {
                        return iframe.contentDocument || (iframe.contentWindow && iframe.contentWindow.document) || null;
                    } catch (error) {
                        return null;
                    }
                }

                function executeScripts(targetDoc, root) {
                    const scripts = root.querySelectorAll('script');
                    scripts.forEach((script) => {
                        const replacement = targetDoc.createElement('script');
                        Array.from(script.attributes).forEach((attr) => {
                            replacement.setAttribute(attr.name, attr.value);
                        });
                        replacement.textContent = script.textContent;
                        script.replaceWith(replacement);
                    });
                }

                function mountLifecycle() {
                    const paneDoc = getPaneDocument();
                    if (!paneDoc) {
                        return false;
                    }
                    const slot = paneDoc.querySelector('#intro-lifecycle-ring-slot');
                    if (!slot) {
                        return false;
                    }
                    if (slot.dataset.introLifecycleMounted === '1') {
                        return true;
                    }

                    const template = paneDoc.createElement('template');
                    template.innerHTML = markup;
                    const content = template.content.cloneNode(true);

                    slot.replaceChildren(content);
                    executeScripts(paneDoc, slot);
                    slot.setAttribute('data-intro-lifecycle-mounted', '1');
                    return true;
                }

                if (!mountLifecycle()) {
                    const maxWaitMs = 4000;
                    const start = Date.now();

                    const interval = setInterval(() => {
                        const mounted = mountLifecycle();
                        const expired = Date.now() - start > maxWaitMs;
                        if (mounted || expired) {
                            clearInterval(interval);
                        }
                    }, 120);
                }
            })();
        </script>
        """
    )
    script = script_template.replace("__BUNDLE_MARKUP__", bundle_markup)

    if renderer_supports_height:
        html_renderer(script, height=height)
    else:
        html_renderer(script)


def render_intro_hero() -> IntroHeroContent:
    """Return the scoped CSS and HTML columns for the intro hero."""

    return IntroHeroContent(
        css=intro_hero_scoped_css(),
        columns=intro_lifecycle_columns(),
    )
