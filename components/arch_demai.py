"""Streamlit-native architecture overview component for the demAI lab."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
import html
from typing import Iterable

import streamlit as st


@dataclass(frozen=True)
class ArchitectureCard:
    """Data model for each building block in the demAI system diagram."""

    key: str
    icon: str
    title: str
    summary: str
    detail: str

    def render(self, highlight: bool, force_open: bool) -> str:
        """Return HTML for a single card."""

        highlight_class = " is-highlight" if highlight else ""
        open_attr = " open" if force_open else ""
        summary_html = html.escape(self.summary)
        detail_html = html.escape(self.detail)
        title_html = html.escape(self.title)

        return dedent(
            f"""
            <details class="arch-card{highlight_class}" data-arch="{self.key}"{open_attr}>
              <summary>
                <div class="arch-card__header">
                  <span class="arch-card__icon" aria-hidden="true">{self.icon}</span>
                  <div class="arch-card__text">
                    <h3 class="arch-card__title">{title_html}</h3>
                    <p class="arch-card__summary">{summary_html}</p>
                  </div>
                  <div class="arch-card__chevron" aria-hidden="true">▾</div>
                </div>
              </summary>
              <div class="arch-card__body" aria-live="polite">
                <p class="arch-card__detail">{detail_html}</p>
              </div>
            </details>
            """
        ).strip()


def _stage_highlight(stage: str | None) -> str:
    """Return which card should glow for the provided stage."""

    mapping = {
        "overview": "ui",
        "data": "ui",
        "train": "model",
        "evaluate": "model",
        "use": "inbox",
    }
    if not stage:
        return ""
    return mapping.get(stage.lower(), "")


def _inject_styles() -> None:
    """Inject the CSS styles that give the cards their personality."""

    style_block = dedent(
        """
        <style>
          .arch-surface{
            --ink:#0f172a;
            --muted:rgba(15,23,42,.76);
            --frame:linear-gradient(180deg, rgba(255,255,255,.95), rgba(248,250,252,.82));
            --highlight:rgba(59,130,246,.22);
            --stroke:rgba(15,23,42,.08);
            --shadow:0 18px 36px rgba(15,23,42,.14);
            --card-bg:rgba(255,255,255,.96);
            position:relative;
            background: radial-gradient(140% 115% at 50% 0%, rgba(99,102,241,.1), rgba(14,165,233,.08));
            border-radius: 18px;
            padding: clamp(16px, 2.6vw, 28px);
            box-shadow: inset 0 0 0 1px rgba(15,23,42,.05);
          }

          .arch-surface__frame{
            border-radius: 16px;
            background: var(--frame);
            box-shadow: inset 0 0 0 1px var(--stroke);
            padding: clamp(10px, 2vw, 18px);
          }

          .arch-grid{
            display:grid;
            gap: clamp(12px, 1.8vw, 18px);
          }

          @media (min-width: 880px){
            .arch-grid{ grid-template-columns: 1fr 1.1fr 1fr; }
          }

          .arch-card{
            list-style:none;
            border-radius: 16px;
            background: var(--card-bg);
            box-shadow: var(--shadow), inset 0 0 0 1px var(--stroke);
            padding: clamp(14px, 2.4vw, 22px);
            transition: box-shadow .25s ease, transform .25s ease;
            position:relative;
            overflow:hidden;
          }

          .arch-card summary{ cursor:pointer; }
          .arch-card summary::-webkit-details-marker{ display:none; }
          .arch-card summary:focus-visible{ outline: 2px solid rgba(59,130,246,.7); outline-offset: 4px; }

          .arch-card__header{
            display:flex;
            align-items:center;
            gap: clamp(12px, 1vw, 16px);
          }

          .arch-card__icon{
            font-size: clamp(1.4rem, 1.8vw, 1.9rem);
            filter: drop-shadow(0 6px 12px rgba(15,23,42,.2));
          }

          .arch-card__text{
            display:flex;
            flex-direction:column;
            gap: .35rem;
          }

          .arch-card__title{
            margin:0;
            font-size: clamp(1.05rem, 1.1vw + .95rem, 1.25rem);
            font-weight:800;
            color: var(--ink);
          }

          .arch-card__summary{
            margin:0;
            color: var(--muted);
            font-size: clamp(.95rem, .3vw + .9rem, 1rem);
            line-height:1.55;
          }

          .arch-card__chevron{
            margin-left:auto;
            color: rgba(15,23,42,.4);
            font-size: 1.2rem;
            transition: transform .25s ease, color .25s ease;
          }

          .arch-card[open] .arch-card__chevron{ transform: rotate(180deg); color: rgba(37,99,235,.8); }

          .arch-card__body{
            margin-top: clamp(12px, 1.2vw, 16px);
            padding-top: clamp(10px, 1vw, 14px);
            border-top: 1px solid rgba(148,163,184,.24);
          }

          .arch-card__detail{
            margin:0;
            color: rgba(15,23,42,.9);
            font-size: clamp(.93rem, .3vw + .9rem, 1rem);
            line-height:1.6;
            background: rgba(226,232,240,.55);
            border-radius: 12px;
            padding: .65rem .85rem;
          }

          .arch-card:hover{ transform: translateY(-4px); box-shadow: 0 22px 40px rgba(15,23,42,.18), inset 0 0 0 1px rgba(148,163,184,.28); }
          .arch-card.is-highlight{ box-shadow: 0 26px 48px rgba(37,99,235,.22), inset 0 0 0 2px rgba(59,130,246,.5); }
          .arch-card.is-highlight .arch-card__title{ color: rgba(37,99,235,.95); }

          .arch-surface.nerd-on .arch-card{ cursor: default; }
          .arch-surface.nerd-on .arch-card summary{ cursor: default; }
        </style>
        """
    )
    st.markdown(style_block, unsafe_allow_html=True)


def render_demai_architecture(*, nerd_mode: bool = False, active_stage: str | None = None) -> None:
    """Render the demAI architecture diagram using Streamlit-native primitives."""

    _inject_styles()

    cards: Iterable[ArchitectureCard] = (
        ArchitectureCard(
            key="ui",
            icon="🖥️",
            title="User interface",
            summary="Your control panel for building and monitoring the system.",
            detail="Stages: Prepare • Train • Evaluate • Use; guided explainers, sliders, and insights dashboards.",
        ),
        ArchitectureCard(
            key="model",
            icon="🧠",
            title="AI model",
            summary="Learns from labeled examples to distinguish spam from safe mail.",
            detail="Text encoder + classifier with optional numeric guardrails. Produces spam scores routed through policy thresholds.",
        ),
        ArchitectureCard(
            key="inbox",
            icon="📥",
            title="Inbox interface",
            summary="Streams new emails into the pipeline and hands them to the model.",
            detail="Batch and streaming ingestion, metadata enrichment, and replay tooling for evaluation batches.",
        ),
    )

    highlighted_key = _stage_highlight(active_stage)

    card_markup = "\n".join(
        card.render(highlight=card.key == highlighted_key, force_open=nerd_mode)
        for card in cards
    )

    container_html = dedent(
        f"""
        <section class="arch-surface{' nerd-on' if nerd_mode else ''}" aria-label="demAI architecture diagram">
          <div class="arch-surface__frame">
            <div class="arch-grid">
              {card_markup}
            </div>
          </div>
        </section>
        """
    ).strip()

    st.markdown(container_html, unsafe_allow_html=True)
