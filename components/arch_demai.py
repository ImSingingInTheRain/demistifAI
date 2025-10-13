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
            padding: clamp(16px, 2.8vw, 28px);
            transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
            overflow:visible;
            color: var(--text-primary);
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
            width: 14px;
            height: 14px;
            border-radius:50%;
            border: 2px solid rgba(148,197,255,.8);
            background: rgba(5,23,45,.85);
            top: 50%;
            transform: translateY(-50%);
            box-shadow: 0 0 12px rgba(56,189,248,.55);
          }

          .arch-card[data-arch="ui"]::after{ left: -34px; }
          .arch-card[data-arch="model"]::after{ left: 50%; transform: translate(-50%, -50%); }
          .arch-card[data-arch="inbox"]::after{ right: -34px; }

          .arch-card summary{ cursor:pointer; }
          .arch-card summary::-webkit-details-marker{ display:none; }
          .arch-card summary:focus-visible{ outline: 2px solid rgba(148,197,255,.7); outline-offset: 6px; }

          .arch-card__header{
            display:flex;
            align-items:flex-start;
            gap: clamp(14px, 1.8vw, 20px);
          }

          .arch-card__icon{
            font-size: clamp(1.6rem, 2vw, 2.2rem);
            filter: drop-shadow(0 12px 20px rgba(0,0,0,.45));
            transform: translateY(-4px);
          }

          .arch-card__text{
            display:flex;
            flex-direction:column;
            gap: .4rem;
          }

          .arch-card__title{
            margin:0;
            font-size: clamp(1.1rem, .9vw + 1rem, 1.35rem);
            font-weight:800;
            letter-spacing:.02em;
            text-transform:uppercase;
            color: var(--text-primary);
          }

          .arch-card__summary{
            margin:0;
            color: var(--text-muted);
            font-size: clamp(.95rem, .4vw + .9rem, 1.05rem);
            line-height:1.6;
          }

          .arch-card__chevron{
            margin-left:auto;
            color: rgba(148,197,255,.6);
            font-size: 1.2rem;
            transition: transform .3s ease, color .3s ease;
          }

          .arch-card[open] .arch-card__chevron{ transform: rotate(180deg); color: rgba(125,211,252,.95); }

          .arch-card__body{
            margin-top: clamp(16px, 2vw, 22px);
            padding-top: clamp(14px, 1.6vw, 20px);
            border-top: 1px dashed rgba(148,197,255,.35);
          }

          .arch-card__detail{
            margin:0;
            color: rgba(212,233,255,.96);
            font-size: clamp(.92rem, .4vw + .92rem, 1.06rem);
            line-height:1.7;
            background: rgba(3,17,34,.78);
            border-radius: 14px;
            padding: .8rem 1rem;
            box-shadow: inset 0 0 0 1px rgba(125,211,252,.18);
          }

          .arch-card:hover{
            transform: translateY(-6px);
            border-color: rgba(191,219,254,.75);
            box-shadow: 0 32px 60px rgba(1,10,26,.68);
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

          .arch-surface.nerd-on .arch-card{ cursor: default; }
          .arch-surface.nerd-on .arch-card summary{ cursor: default; }

          @media (max-width: 640px){
            .arch-grid::after{ display:none; }
            .arch-card[data-arch="ui"]::after,
            .arch-card[data-arch="model"]::after,
            .arch-card[data-arch="inbox"]::after{ display:none; }
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
