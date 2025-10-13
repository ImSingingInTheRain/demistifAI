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
        open_class = " is-open" if force_open else ""
        summary_html = html.escape(self.summary)
        detail_html = html.escape(self.detail)
        title_html = html.escape(self.title)

        return dedent(
            f"""
            <article class="arch-card{highlight_class}{open_class}" data-arch="{self.key}">
              <div class="arch-card__header" aria-expanded="{'true' if force_open else 'false'}">
                <span class="arch-card__icon" aria-hidden="true">{self.icon}</span>
                <div class="arch-card__text">
                  <h3 class="arch-card__title">{title_html}</h3>
                  <p class="arch-card__summary">{summary_html}</p>
                </div>
              </div>
              <div class="arch-card__body" aria-live="polite" aria-hidden="{'false' if force_open else 'true'}">
                <p class="arch-card__detail">{detail_html}</p>
              </div>
            </article>
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
            --blueprint-bg:#051729;
            --blueprint-grid:rgba(56,118,171,.28);
            --blueprint-border:rgba(120,189,255,.45);
            --card-bg:rgba(7,27,55,.76);
            --card-border:rgba(147,197,253,.4);
            --card-shadow:0 25px 50px rgba(5,15,30,.45);
            --text-primary:#e2f1ff;
            --text-muted:rgba(195,218,247,.88);
            --highlight:rgba(56,189,248,.28);
            position:relative;
            isolation:isolate;
            padding: clamp(18px, 3vw, 32px);
            border-radius: 22px;
            background: radial-gradient(120% 140% at 50% 0%, rgba(56,189,248,.28), rgba(15,118,190,.1) 45%, var(--blueprint-bg) 100%);
            box-shadow: inset 0 0 0 1px rgba(148,197,255,.25), 0 40px 60px rgba(1,8,18,.65);
            overflow:hidden;
          }

          .arch-surface::before{
            content:"";
            position:absolute;
            inset:0;
            background-image:
              repeating-linear-gradient(0deg, transparent 0 30px, var(--blueprint-grid) 30px 31px),
              repeating-linear-gradient(90deg, transparent 0 30px, var(--blueprint-grid) 30px 31px);
            opacity:.6;
            pointer-events:none;
            mix-blend-mode:screen;
            z-index:-1;
          }

          .arch-surface__frame{
            border-radius: 18px;
            padding: clamp(14px, 3vw, 28px);
            background: linear-gradient(180deg, rgba(9,34,66,.65), rgba(7,23,45,.85));
            border: 1px solid var(--blueprint-border);
            box-shadow: inset 0 0 0 1px rgba(120,189,255,.18);
            position:relative;
          }

          .arch-surface__frame::before,
          .arch-surface__frame::after{
            content:"";
            position:absolute;
            border: 1px solid rgba(120,189,255,.32);
            border-radius: 18px;
            pointer-events:none;
          }

          .arch-surface__frame::before{
            inset:14px;
            opacity:.65;
          }

          .arch-surface__frame::after{
            inset:28px;
            opacity:.35;
          }

          .arch-grid{
            display:grid;
            gap: clamp(14px, 2.5vw, 28px);
            grid-template-columns: repeat(auto-fit, minmax(240px,1fr));
            position:relative;
            padding: clamp(8px, 2vw, 14px) clamp(10px, 2.4vw, 18px);
          }

          @media (min-width: 900px){
            .arch-grid{
              grid-template-columns: repeat(3, 1fr);
              grid-auto-rows: 1fr;
            }
          }

          .arch-grid::before{
            content:"";
            position:absolute;
            inset: clamp(12px, 2vw, 24px);
            border: 1px dashed rgba(120,189,255,.35);
            border-radius: 18px;
            pointer-events:none;
            opacity:.7;
          }

          .arch-grid::after{
            content:"";
            position:absolute;
            top: 50%;
            left: clamp(12px, 2vw, 24px);
            right: clamp(12px, 2vw, 24px);
            height: 0;
            border-top: 2px solid rgba(79,172,254,.32);
            filter: drop-shadow(0 0 6px rgba(79,172,254,.6));
            pointer-events:none;
          }

          .arch-card{
            list-style:none;
            position:relative;
            border-radius: 18px;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            box-shadow: var(--card-shadow);
            padding: clamp(18px, 2.6vw, 30px);
            transition: transform .35s ease, box-shadow .35s ease, border-color .35s ease;
            overflow:visible;
            color: var(--text-primary);
            display:flex;
            flex-direction:column;
            gap: clamp(14px, 2vw, 20px);
          }

          .arch-card::before{
            content:"";
            position:absolute;
            inset: 10px;
            border: 1px solid rgba(148,197,255,.25);
            border-radius: 14px;
            opacity:.65;
            pointer-events:none;
          }

          .arch-card::after{
            content:"";
            position:absolute;
            width: 46px;
            height: 46px;
            border-radius:50%;
            border: 1px solid rgba(125,211,252,.6);
            background: radial-gradient(circle at 50% 50%, rgba(148,197,255,.35), transparent 65%);
            top: 12px;
            left: 12px;
            opacity:.45;
            pointer-events:none;
            filter: blur(.4px);
          }

          .arch-card[data-arch="model"]::after{ left: 50%; transform: translateX(-50%); }
          .arch-card[data-arch="inbox"]::after{ left: auto; right: 12px; }

          .arch-card__header{
            display:flex;
            align-items:center;
            gap: clamp(16px, 2vw, 22px);
          }

          .arch-card__icon{
            font-size: clamp(1.75rem, 2.3vw, 2.4rem);
            filter: drop-shadow(0 12px 22px rgba(0,0,0,.45));
            display:flex;
            align-items:center;
            justify-content:center;
            width: clamp(52px, 4.2vw, 60px);
            height: clamp(52px, 4.2vw, 60px);
            border-radius: 16px;
            border: 1px solid rgba(148,197,255,.5);
            background: linear-gradient(145deg, rgba(11,45,86,.95), rgba(5,24,48,.9));
            box-shadow: inset 0 0 0 1px rgba(148,197,255,.2), 0 18px 30px rgba(3,12,26,.55);
          }

          .arch-card__text{
            display:flex;
            flex-direction:column;
            gap: .5rem;
          }

          .arch-card__title{
            margin:0;
            font-size: clamp(1.15rem, 1vw + 1rem, 1.42rem);
            font-weight:800;
            letter-spacing:.02em;
            text-transform:uppercase;
            color: var(--text-primary);
          }

          .arch-card__summary{
            margin:0;
            color: var(--text-muted);
            font-size: clamp(.96rem, .45vw + .9rem, 1.08rem);
            line-height:1.55;
          }

          .arch-card__body{
            margin-top:0;
            padding-top: clamp(14px, 1.6vw, 20px);
            border-top: 1px dashed rgba(148,197,255,.3);
            display:flex;
            flex-direction:column;
            gap: .6rem;
            max-height:0;
            opacity:0;
            overflow:hidden;
            pointer-events:none;
            transition: max-height .35s ease, opacity .3s ease;
          }

          .arch-card__detail{
            margin:0;
            color: rgba(212,233,255,.96);
            font-size: clamp(.92rem, .4vw + .92rem, 1.06rem);
            line-height:1.7;
            background: rgba(3,17,34,.78);
            border-radius: 14px;
            padding: .85rem 1.05rem;
            box-shadow: inset 0 0 0 1px rgba(125,211,252,.18);
          }

          .arch-card:hover{
            transform: translateY(-4px);
            border-color: rgba(191,219,254,.75);
            box-shadow: 0 28px 58px rgba(1,10,26,.68);
          }

          .arch-card.is-highlight{
            border-color: rgba(125,211,252,.95);
            box-shadow: 0 32px 60px rgba(56,189,248,.55);
            background: linear-gradient(180deg, rgba(12,52,95,.9), rgba(7,23,45,.92));
          }

          .arch-card.is-highlight::before{
            border-color: rgba(191,219,254,.85);
            opacity:.9;
          }

          .arch-card.is-highlight .arch-card__title{ color: #f1fbff; }

          .arch-card.is-open .arch-card__body,
          .arch-surface.nerd-on .arch-card.is-open .arch-card__body{
            max-height: 320px;
            opacity:1;
            pointer-events:auto;
          }

          @media (max-width: 640px){
            .arch-grid::after{ display:none; }
            .arch-card::after{ display:none; }
          }
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

    show_details = nerd_mode and (active_stage or "").lower() == "overview"

    card_markup = "\n".join(
        card.render(highlight=card.key == highlighted_key, force_open=show_details)
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
