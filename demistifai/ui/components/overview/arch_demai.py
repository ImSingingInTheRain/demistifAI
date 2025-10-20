"""Streamlit-native architecture overview component for the demAI lab."""

from __future__ import annotations

from dataclasses import dataclass
import html
from textwrap import dedent
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


def _style_block() -> str:
    """Return the scoped CSS required for the architecture cards."""

    return dedent(
        """
        <style>
          [data-mw-root] .arch-surface{
            --surface-bg:linear-gradient(180deg, rgba(231,244,255,.9), rgba(219,234,254,.65));
            --surface-grid:rgba(56,118,171,.18);
            --surface-border:rgba(96,165,250,.45);
            --card-bg:rgba(248,250,252,.88);
            --card-border:rgba(148,163,184,.5);
            --card-shadow:0 16px 36px rgba(15,23,42,.18);
            --text-primary:#0f172a;
            --text-muted:rgba(15,23,42,.68);
            --highlight:rgba(59,130,246,.14);
            position:relative;
            isolation:isolate;
            padding: clamp(14px, 2.2vw, 24px);
            border-radius: 18px;
            background: var(--surface-bg);
            box-shadow: inset 0 0 0 1px rgba(148,197,255,.22), 0 26px 50px rgba(15,23,42,.15);
            overflow:hidden;
          }

          [data-mw-root] .arch-surface::before{
            content:"";
            position:absolute;
            inset:0;
            background-image:
              repeating-linear-gradient(0deg, transparent 0 26px, var(--surface-grid) 26px 27px),
              repeating-linear-gradient(90deg, transparent 0 26px, var(--surface-grid) 26px 27px);
            opacity:.55;
            pointer-events:none;
            z-index:-1;
          }

          [data-mw-root] .arch-surface__frame{
            border-radius: 14px;
            padding: clamp(12px, 2.4vw, 22px);
            background: rgba(255,255,255,.72);
            border: 1px solid var(--surface-border);
            box-shadow: inset 0 0 0 1px rgba(148,197,255,.16);
            position:relative;
          }

          [data-mw-root] .arch-surface__frame::before,
          [data-mw-root] .arch-surface__frame::after{
            content:"";
            position:absolute;
            border: 1px solid rgba(148,197,255,.28);
            border-radius: 14px;
            pointer-events:none;
          }

          [data-mw-root] .arch-surface__frame::before{
            inset:12px;
            opacity:.55;
          }

          [data-mw-root] .arch-surface__frame::after{
            inset:24px;
            opacity:.28;
          }

          [data-mw-root] .arch-grid{
            display:grid;
            gap: clamp(12px, 2vw, 20px);
            grid-template-columns: repeat(auto-fit, minmax(210px,1fr));
            position:relative;
            padding: clamp(6px, 1.6vw, 12px) clamp(8px, 2vw, 16px);
          }

          @media (min-width: 900px){
            [data-mw-root] .arch-grid{
              grid-template-columns: repeat(3, 1fr);
              grid-auto-rows: 1fr;
            }
          }

          [data-mw-root] .arch-grid::before{
            content:"";
            position:absolute;
            inset: clamp(10px, 1.8vw, 20px);
            border: 1px dashed rgba(96,165,250,.35);
            border-radius: 14px;
            pointer-events:none;
            opacity:.55;
          }

          [data-mw-root] .arch-grid::after{
            content:"";
            position:absolute;
            top: 50%;
            left: clamp(10px, 1.8vw, 20px);
            right: clamp(10px, 1.8vw, 20px);
            height: 0;
            border-top: 1px solid rgba(59,130,246,.28);
            filter: drop-shadow(0 0 4px rgba(79,172,254,.45));
            pointer-events:none;
          }

          [data-mw-root] .arch-card{
            list-style:none;
            position:relative;
            border-radius: 14px;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            box-shadow: var(--card-shadow);
            padding: clamp(14px, 2vw, 22px);
            transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
            overflow:visible;
            color: var(--text-primary);
            display:flex;
            flex-direction:column;
            gap: clamp(10px, 1.6vw, 16px);
          }

          [data-mw-root] .arch-card::before{
            content:"";
            position:absolute;
            inset: 9px;
            border: 1px solid rgba(148,163,184,.25);
            border-radius: 12px;
            opacity:.45;
            pointer-events:none;
          }

          [data-mw-root] .arch-card::after{
            content:"";
            position:absolute;
            width: 38px;
            height: 38px;
            border-radius:50%;
            border: 1px solid rgba(96,165,250,.5);
            background: radial-gradient(circle at 50% 50%, rgba(148,197,255,.28), transparent 65%);
            top: 10px;
            left: 10px;
            opacity:.4;
            pointer-events:none;
            filter: blur(.4px);
          }

          [data-mw-root] .arch-card[data-arch="model"]::after{ left: 50%; transform: translateX(-50%); }
          [data-mw-root] .arch-card[data-arch="inbox"]::after{ left: auto; right: 10px; }

          [data-mw-root] .arch-card__header{
            display:flex;
            align-items:center;
            gap: clamp(12px, 1.6vw, 18px);
          }

          [data-mw-root] .arch-card__icon{
            font-size: clamp(1.45rem, 1.8vw, 1.9rem);
            filter: drop-shadow(0 10px 18px rgba(15,23,42,.32));
            display:flex;
            align-items:center;
            justify-content:center;
            width: clamp(44px, 3.4vw, 50px);
            height: clamp(44px, 3.4vw, 50px);
            border-radius: 14px;
            border: 1px solid rgba(59,130,246,.45);
            background: linear-gradient(145deg, rgba(224,242,254,.95), rgba(191,219,254,.78));
            box-shadow: inset 0 0 0 1px rgba(59,130,246,.2), 0 14px 26px rgba(59,130,246,.28);
          }

          [data-mw-root] .arch-card__text{
            display:flex;
            flex-direction:column;
            gap: .4rem;
          }

          [data-mw-root] .arch-card__title{
            margin:0;
            font-size: clamp(1.02rem, .65vw + .95rem, 1.18rem);
            font-weight:700;
            letter-spacing:.01em;
            color: var(--text-primary);
          }

          [data-mw-root] .arch-card__summary{
            margin:0;
            color: var(--text-muted);
            font-size: clamp(.88rem, .35vw + .86rem, .98rem);
            line-height:1.45;
          }

          [data-mw-root] .arch-card__body{
            margin-top:0;
            padding-top: clamp(12px, 1.2vw, 16px);
            border-top: 1px dashed rgba(148,163,184,.4);
            display:flex;
            flex-direction:column;
            gap: .55rem;
            max-height:0;
            opacity:0;
            overflow:hidden;
            pointer-events:none;
            transition: max-height .35s ease, opacity .3s ease;
          }

          [data-mw-root] .arch-card__detail{
            margin:0;
            color: rgba(30,41,59,.92);
            font-size: clamp(.86rem, .35vw + .86rem, .98rem);
            line-height:1.55;
            background: rgba(255,255,255,.82);
            border-radius: 12px;
            padding: .75rem .95rem;
            box-shadow: inset 0 0 0 1px rgba(148,163,184,.18);
          }

          [data-mw-root] .arch-card:hover{
            transform: translateY(-3px);
            border-color: rgba(59,130,246,.65);
            box-shadow: 0 20px 38px rgba(59,130,246,.22);
          }

          [data-mw-root] .arch-card.is-highlight{
            border-color: rgba(59,130,246,.75);
            box-shadow: 0 24px 44px rgba(59,130,246,.28);
            background: linear-gradient(180deg, rgba(224,242,254,.92), rgba(191,219,254,.88));
          }

          [data-mw-root] .arch-card.is-highlight::before{
            border-color: rgba(59,130,246,.55);
            opacity:.65;
          }

          [data-mw-root] .arch-card.is-highlight .arch-card__title{ color: #0b1f3a; }

          [data-mw-root] .arch-card.is-open .arch-card__body,
          [data-mw-root] .arch-surface.nerd-on .arch-card.is-open .arch-card__body{
            max-height: 220px;
            opacity:1;
            pointer-events:auto;
          }

          @media (max-width: 640px){
            [data-mw-root] .arch-grid::after{ display:none; }
            [data-mw-root] .arch-card::after{ display:none; }
          }
        </style>
        """
    ).strip()


def demai_architecture_styles() -> str:
    """Expose the architecture style block for external renderers."""

    return _style_block()


def _inject_styles() -> None:
    """Inject the CSS styles that give the cards their personality."""

    st.markdown(_style_block(), unsafe_allow_html=True)


def render_demai_architecture(*, nerd_mode: bool = False, active_stage: str | None = None) -> None:
    """Render the demAI architecture diagram using Streamlit-native primitives."""

    _inject_styles()
    markup = demai_architecture_markup(nerd_mode=nerd_mode, active_stage=active_stage)
    st.markdown(markup, unsafe_allow_html=True)


def demai_architecture_markup(
    *, nerd_mode: bool = False, active_stage: str | None = None, include_styles: bool = False
) -> str:
    """Return HTML markup for the demAI architecture diagram."""

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

    if include_styles:
        return "\n".join([_style_block(), container_html])
    return container_html
