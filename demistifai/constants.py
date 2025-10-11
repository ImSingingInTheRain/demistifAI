"""Application-wide constants and reusable configuration values."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent
from typing import List

# CSS assets used throughout the application.
APP_THEME_CSS = """
<style>
:root {
    --surface-radius: 24px;
    --surface-border: rgba(15, 23, 42, 0.12);
    --surface-shadow: 0 24px 48px rgba(15, 23, 42, 0.14);
    --surface-gradient: linear-gradient(160deg, rgba(226, 232, 240, 0.9), rgba(255, 255, 255, 0.96));
    --accent-primary: #1d4ed8;
    --accent-muted: rgba(30, 64, 175, 0.16);
}

.stApp {
    background: radial-gradient(circle at top left, rgba(148, 163, 184, 0.16), transparent 45%),
        radial-gradient(circle at 85% 10%, rgba(96, 165, 250, 0.16), transparent 50%),
        #f8fafc;
    color: #0f172a;
    font-family: "Inter", "Segoe UI", sans-serif;
}

[data-testid="stMainBlock"] {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

.main .block-container {
    max-width: 1200px;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}

.section-surface {
    display: block;
    width: 100%;
}

.section-surface > div[data-testid="stVerticalBlock"],
.section-surface-block {
    position: relative;
    z-index: 0;
    margin-bottom: clamp(1.2rem, 1.2vw + 0.8rem, 1.8rem);
    border-radius: var(--surface-radius);
    border: 1px solid var(--surface-border);
    background: var(--surface-gradient);
    box-shadow: var(--surface-shadow);
    padding: clamp(1.4rem, 1.4vw + 1rem, 2.1rem);
    overflow: hidden;
    color: #0f172a;
}

.section-surface > div[data-testid="stVerticalBlock"]::before,
.section-surface-block::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(150deg, rgba(59, 130, 246, 0.15), transparent 65%);
    opacity: 0;
    transition: opacity 0.3s ease;
    pointer-events: none;
    z-index: 0;
}

.section-surface > div[data-testid="stVerticalBlock"]:hover::before,
.section-surface-block:hover::before {
    opacity: 1;
}

.section-surface > div[data-testid="stVerticalBlock"] > *,
.section-surface-block > * {
    position: relative;
    z-index: 1;
}

.section-surface > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"],
.section-surface-block > [data-testid="stElementContainer"] {
    margin-bottom: 0.9rem;
}

.section-surface > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:first-child,
.section-surface > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child,
.section-surface-block > [data-testid="stElementContainer"]:first-child,
.section-surface-block > [data-testid="stElementContainer"]:last-child {
    margin-bottom: 0;
}

.section-surface > div[data-testid="stVerticalBlock"] h2,
.section-surface > div[data-testid="stVerticalBlock"] h3,
.section-surface > div[data-testid="stVerticalBlock"] h4,
.section-surface-block h2,
.section-surface-block h3,
.section-surface-block h4 {
    margin-top: 0;
    color: inherit;
}

.section-surface > div[data-testid="stVerticalBlock"] p,
.section-surface > div[data-testid="stVerticalBlock"] ul,
.section-surface-block p,
.section-surface-block ul {
    font-size: 0.98rem;
    line-height: 1.65;
    color: inherit;
}

.section-surface > div[data-testid="stVerticalBlock"] ul,
.section-surface-block ul {
    padding-left: 1.25rem;
}

.section-surface > div[data-testid="stVerticalBlock"] .section-caption,
.section-surface-block .section-caption {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: rgba(15, 23, 42, 0.65);
}

.section-surface > div[data-testid="stVerticalBlock"] .section-caption span,
.section-surface-block .section-caption span {
    display: inline-flex;
    width: 40px;
    height: 2px;
    background: rgba(15, 23, 42, 0.12);
}

.section-surface.section-surface--hero > div[data-testid="stVerticalBlock"],
.section-surface-block.section-surface--hero {
    padding: 2.6rem 2.8rem;
    background: linear-gradient(160deg, #1d4ed8, #312e81);
    color: #f8fafc;
    border: 1px solid rgba(255, 255, 255, 0.28);
    box-shadow: 0 28px 70px rgba(30, 64, 175, 0.35);
}

.section-surface.section-surface--hero > div[data-testid="stVerticalBlock"]::before,
.section-surface-block.section-surface--hero::before {
    display: none;
}

.section-surface.section-surface--hero > div[data-testid="stVerticalBlock"] .section-caption,
.section-surface-block.section-surface--hero .section-caption {
    color: rgba(241, 245, 249, 0.85);
}

.section-surface.section-surface--hero > div[data-testid="stVerticalBlock"] .section-caption span,
.section-surface-block.section-surface--hero .section-caption span {
    background: rgba(241, 245, 249, 0.35);
}

.section-surface > div[data-testid="stVerticalBlock"] .surface-columns,
.section-surface-block .surface-columns {
    display: grid;
    gap: 1.4rem;
}

.section-surface > div[data-testid="stVerticalBlock"] .surface-columns.two,
.section-surface-block .surface-columns.two {
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}

.section-surface > div[data-testid="stVerticalBlock"] .surface-columns.three,
.section-surface-block .surface-columns.three {
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
}

.section-surface--hero [data-testid="column"]:nth-child(2) > div:first-of-type {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.section-surface--hero [data-testid="column"]:nth-child(2) > div:first-of-type > div[data-testid="stVerticalBlock"] {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.6rem;
    width: 100%;
}

.section-surface--hero [data-testid="column"]:nth-child(2) > div:first-of-type > div[data-testid="stVerticalBlock"] > div {
    width: 100%;
}

.section-surface--hero [data-testid="column"]:nth-child(2) > div:first-of-type > div[data-testid="stVerticalBlock"] > div[data-testid="stButton"] {
    display: flex;
    justify-content: center;
}

.section-surface--hero [data-testid="column"]:nth-child(2) > div:first-of-type > div[data-testid="stVerticalBlock"] > div[data-testid="stButton"] > button {
    width: 100%;
    max-width: 260px;
}

.hero-info-grid {
    width: 100%;
    display: grid;
    gap: 1.4rem;
    justify-items: stretch;
}

@media (min-width: 768px) {
    .hero-info-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

.indicator-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
}

.pii-chip-row {
    margin-top: 0.85rem;
}

.pii-chip-row--compact {
    margin-bottom: 0.25rem;
}

.lint-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border-radius: 999px;
    border: 1px solid rgba(15, 23, 42, 0.18);
    padding: 0.2rem 0.65rem;
    font-size: 0.78rem;
    font-weight: 600;
    background: rgba(148, 163, 184, 0.18);
    color: #0f172a;
    letter-spacing: 0.01em;
}

.lint-chip__icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.35rem;
    height: 1.35rem;
    border-radius: 50%;
    background: rgba(15, 23, 42, 0.12);
    font-size: 0.82rem;
}

.lint-chip__text {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
}

.pii-alert-card {
    margin-top: 0.25rem;
    margin-bottom: 0.75rem;
    border-radius: 18px;
    border: 1px solid rgba(217, 119, 6, 0.28);
    background: linear-gradient(135deg, rgba(254, 243, 199, 0.75), rgba(255, 255, 255, 0.92));
    box-shadow: 0 20px 35px rgba(217, 119, 6, 0.18);
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
}

.pii-alert-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at top left, rgba(217, 119, 6, 0.18), transparent 55%);
    opacity: 0.9;
    pointer-events: none;
}

.pii-alert-card > div[data-testid="column"] {
    position: relative;
}

.pii-alert-card__body {
    position: relative;
    z-index: 1;
    color: #7c2d12;
}

.pii-alert-card__title {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #9a3412;
    margin-bottom: 0.4rem;
}

.pii-alert-card__body p {
    margin-bottom: 0.65rem;
    font-size: 0.95rem;
    line-height: 1.6;
}

.pii-alert-card__chips {
    margin-top: 0.35rem;
}

.pii-alert-card__action {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
}

.pii-alert-card__action div[data-testid="stButton"] {
    width: 100%;
}

.pii-alert-card__action button {
    width: 100%;
    min-height: 52px;
    border-radius: 999px;
    font-weight: 700;
    box-shadow: 0 10px 18px rgba(217, 119, 6, 0.25);
}

.sample-card__label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #475569;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.sample-card__label-icon {
    font-size: 0.85rem;
}

.edge-case-card__label {
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.cta-sticky {
    position: absolute;
    bottom: 0;
    padding: 0.75rem 0 0.5rem 0;
    margin-top: 1rem;
    background: linear-gradient(180deg, rgba(248, 250, 252, 0.95), rgba(255, 255, 255, 0.98));
    border-top: 1px solid rgba(15, 23, 42, 0.08);
    box-shadow: 0 -12px 24px rgba(15, 23, 42, 0.08);
    z-index: 5;
}

.cta-sticky > div[data-testid="column"] {
    margin-bottom: 0;
}

.cta-sticky [data-testid="stButton"] > button,
.cta-sticky [data-testid="stFormSubmitButton"] > button {
    width: 100%;
}

.hero-info-card {
    position: relative;
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 1.2rem 1.3rem;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.28);
    background: linear-gradient(135deg, rgba(241, 245, 249, 0.92), rgba(255, 255, 255, 0.98));
    box-shadow: 0 22px 36px rgba(15, 23, 42, 0.08);
}

.hero-info-card::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    border: 1px solid rgba(30, 64, 175, 0.15);
    pointer-events: none;
}

.hero-info-card__icon {
    flex-shrink: 0;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2.35rem;
    height: 2.35rem;
    border-radius: 14px;
    background: rgba(30, 64, 175, 0.12);
    font-size: 1.25rem;
}

.hero-info-card__content h3 {
    margin: 0 0 0.35rem 0;
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
}

.hero-info-card__content p {
    margin: 0;
    font-size: 0.95rem;
    line-height: 1.65;
    color: rgba(15, 23, 42, 0.82);
}

@media (max-width: 767px) {
    .hero-info-card {
        flex-direction: column;
        align-items: stretch;
        text-align: left;
    }

    .hero-info-card__icon {
        width: 2.7rem;
        height: 2.7rem;
        font-size: 1.35rem;
    }
}

.section-surface--hero [data-testid="column"]:nth-child(2) [data-testid="stButton"] {
    width: 100%;
}

.section-surface--hero [data-testid="column"]:nth-child(2) [data-testid="stButton"] > button {
    max-width: 260px;
    margin: 0 auto;
}

.ai-quote-box {
    position: relative;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 1rem;
    padding: 1.25rem 1.4rem;
    margin: 1.2rem 0;
    border-radius: 18px;
    border: 1px solid rgba(30, 64, 175, 0.18);
    background: linear-gradient(145deg, rgba(30, 64, 175, 0.08), rgba(30, 64, 175, 0));
    box-shadow: 0 18px 42px rgba(15, 23, 42, 0.12);
    color: #0f172a;
}

.ai-quote-box::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    border: 1px solid rgba(59, 130, 246, 0.16);
    pointer-events: none;
}

.ai-quote-box__icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: rgba(30, 64, 175, 0.12);
    font-size: 1.6rem;
    line-height: 1;
    color: #1d4ed8;
}

.ai-quote-box__content {
    display: grid;
    gap: 0.35rem;
}

.ai-quote-box__source {
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: rgba(30, 64, 175, 0.82);
}

.ai-quote-box__content p {
    margin: 0;
    font-size: 0.98rem;
    line-height: 1.7;
    color: rgba(15, 23, 42, 0.92);
}

[data-testid="column"] > div[data-testid="stVerticalBlock"] {
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
    height: 100%;
}

[data-testid="column"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

[data-testid="column"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stElementContainer"] {
    margin: 0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem;
    padding: 0.4rem;
    background: rgba(15, 23, 42, 0.05);
    border-radius: 14px;
    border: 1px solid rgba(15, 23, 42, 0.08);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 0.55rem 1.1rem;
    background: transparent;
    color: #1f2937;
    font-weight: 600;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(37, 99, 235, 0.08);
    border-color: rgba(37, 99, 235, 0.18);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(140deg, rgba(37, 99, 235, 0.92), rgba(30, 64, 175, 0.88));
    color: #f8fafc;
    box-shadow: 0 12px 26px rgba(30, 64, 175, 0.22);
    border-color: transparent;
}

[data-testid="stExpander"] details {
    border-radius: 18px;
    border: 1px solid rgba(15, 23, 42, 0.08);
    background: rgba(255, 255, 255, 0.92);
    box-shadow: 0 18px 32px rgba(15, 23, 42, 0.12);
    padding: 0.2rem 0.4rem;
    transition: box-shadow 0.3s ease;
}

[data-testid="stExpander"] details[open] {
    box-shadow: 0 24px 46px rgba(37, 99, 235, 0.16);
    border-color: rgba(37, 99, 235, 0.25);
}

[data-testid="stExpander"] summary {
    font-weight: 600;
    font-size: 1.02rem;
    color: #1f2937;
    padding: 0.85rem 1rem;
}

[data-testid="stExpander"] summary:hover {
    color: #1d4ed8;
}

[data-testid="stExpander"] p {
    margin: 0.5rem 0 1rem;
    color: #334155;
    font-size: 0.95rem;
}

[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 22px 45px rgba(15, 23, 42, 0.15);
    border: 1px solid rgba(15, 23, 42, 0.08);
    background: rgba(255, 255, 255, 0.96);
}

[data-testid="stDataFrame"] div[data-testid="stHorizontalBlock"] {
    margin: 0;
}

.stDataFrame tbody tr:nth-child(odd) {
    background: rgba(226, 232, 240, 0.45);
}

.stDataFrame tbody tr:hover {
    background: rgba(37, 99, 235, 0.12);
}

.stDataFrame th {
    background: linear-gradient(120deg, rgba(37, 99, 235, 0.9), rgba(59, 130, 246, 0.75));
    color: #f8fafc;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    font-weight: 700;
    color: #1f2937;
}

[data-testid="stMetricLabel"] {
    color: rgba(71, 85, 105, 0.85);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.75rem;
}

.stAlert {
    border-radius: 18px;
    border: 1px solid rgba(37, 99, 235, 0.2);
    box-shadow: 0 18px 32px rgba(37, 99, 235, 0.14);
    background: linear-gradient(145deg, rgba(191, 219, 254, 0.65), rgba(59, 130, 246, 0.35));
    color: #1f2937;
}

.stAlert button[kind="secondary"] {
    border-radius: 12px;
    border: 1px solid rgba(37, 99, 235, 0.35);
    background: rgba(255, 255, 255, 0.85);
}

.stAlert button[kind="secondary"]:hover {
    background: rgba(255, 255, 255, 0.95);
    border-color: rgba(37, 99, 235, 0.5);
}

.stButton > button {
    border-radius: 14px;
    padding: 0.65rem 1.4rem;
    font-weight: 600;
    border: none;
    background: linear-gradient(135deg, #1d4ed8, #1e3a8a);
    color: #f8fafc;
    box-shadow: 0 18px 40px rgba(30, 64, 175, 0.28);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 24px 50px rgba(30, 64, 175, 0.32);
}

.stButton > button:focus-visible {
    outline: 3px solid rgba(59, 130, 246, 0.45);
    outline-offset: 2px;
}

.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"],
.stRadio,
.stCheckbox,
.stDateInput,
.stTextInput,
.stNumberInput {
    border-radius: 14px;
    border: 1px solid rgba(15, 23, 42, 0.12);
    padding: 0.65rem 0.85rem;
    background: rgba(255, 255, 255, 0.85);
    box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
}

.stSelectbox div[data-baseweb="select"]:focus-within,
.stMultiSelect div[data-baseweb="select"]:focus-within,
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: rgba(37, 99, 235, 0.45);
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.18);
}

.stTextInput input,
.stNumberInput input {
    background: transparent;
}

@media (max-width: 1024px) {
    .main .block-container {
        padding-left: 1.1rem;
        padding-right: 1.1rem;
    }

    .section-surface > div[data-testid="stVerticalBlock"],
    .section-surface-block {
        padding: 1.6rem 1.8rem;
    }
}

@media (max-width: 768px) {
    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.25rem;
    }

    .section-surface.section-surface--hero > div[data-testid="stVerticalBlock"],
    .section-surface-block.section-surface--hero {
        padding: 2.1rem 2rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        padding: 0.3rem;
        gap: 0.25rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0.9rem;
        font-size: 0.92rem;
    }
}

@media (max-width: 640px) {
    .main .block-container {
        padding-left: 0.85rem;
        padding-right: 0.85rem;
    }

    [data-testid="column"] > div[data-testid="stVerticalBlock"] {
        gap: 0.9rem;
    }

    .section-surface > div[data-testid="stVerticalBlock"],
    .section-surface-block {
        padding: 1.4rem 1.35rem;
    }

    .stButton > button {
        width: 100%;
    }
}

.callout {
    position: relative;
    border-radius: 18px;
    border: 1px solid var(--surface-border);
    background: rgba(255, 255, 255, 0.92);
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
    padding: 1.35rem 1.5rem;
    backdrop-filter: blur(6px);
}

.callout h4,
.callout h5 {
    margin: 0 0 0.65rem 0;
    font-weight: 600;
    color: inherit;
}

.callout p {
    margin: 0;
    font-size: 0.98rem;
    line-height: 1.6;
}

.callout--mission {
    background: linear-gradient(145deg, rgba(59, 130, 246, 0.12), rgba(191, 219, 254, 0.35));
    border-color: rgba(37, 99, 235, 0.3);
}

.callout--info {
    background: linear-gradient(150deg, rgba(191, 219, 254, 0.35), rgba(219, 234, 254, 0.65));
    border-color: rgba(37, 99, 235, 0.25);
}

.callout-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-top: 0.9rem;
}

.callout--outcome {
    display: flex;
    gap: 0.85rem;
    align-items: flex-start;
    border-color: rgba(15, 23, 42, 0.08);
}

.callout-icon {
    flex-shrink: 0;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: rgba(37, 99, 235, 0.12);
    font-size: 1.35rem;
}

.callout-body h5 {
    margin-bottom: 0.35rem;
}

.callout-body p {
    font-size: 0.95rem;
    color: rgba(15, 23, 42, 0.82);
}

.mission-preview-stack {
    display: grid;
    gap: 1.1rem;
}

.mission-card,
.inbox-preview-card {
    position: relative;
    border-radius: 20px;
    border: 1px solid rgba(37, 99, 235, 0.18);
    padding: 1.35rem 1.5rem;
    box-shadow: 0 22px 44px rgba(15, 23, 42, 0.12);
}

.mission-card {
    background: linear-gradient(150deg, rgba(191, 219, 254, 0.45), rgba(59, 130, 246, 0.2));
}

.inbox-preview-card {
    background: linear-gradient(150deg, rgba(248, 250, 252, 0.9), rgba(191, 219, 254, 0.55));
    border-color: rgba(37, 99, 235, 0.16);
}

.mission-card::before,
.inbox-preview-card::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    border: 1px solid rgba(255, 255, 255, 0.35);
    opacity: 0.45;
    pointer-events: none;
}

.mission-header,
.preview-header {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}

.mission-header-icon,
.preview-header-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    border-radius: 16px;
    background: rgba(37, 99, 235, 0.18);
    font-size: 1.5rem;
}

.mission-header h4,
.preview-header h4 {
    margin: 0;
    font-size: 1.15rem;
}

.mission-header p,
.preview-header p {
    margin: 0.2rem 0 0 0;
    font-size: 0.95rem;
    color: rgba(15, 23, 42, 0.75);
}

.mission-points {
    margin: 1rem 0 0 0;
    padding-left: 1.1rem;
    font-size: 0.96rem;
    line-height: 1.6;
    color: rgba(15, 23, 42, 0.86);
}

.preview-note {
    margin: 0.9rem 0 0 0;
    font-size: 0.86rem;
    color: rgba(15, 23, 42, 0.65);
}

.nerd-toggle-card {
    display: contents;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: 1.1rem;
    padding: 1.1rem 1.1rem;
    border-radius: 18px;
    border: 1px solid rgba(59, 130, 246, 0.28);
    background: linear-gradient(150deg, rgba(37, 99, 235, 0.12), rgba(191, 219, 254, 0.22));
    box-shadow: 0 16px 32px rgba(15, 23, 42, 0.12);
    margin-bottom: 0.8rem;
}

.nerd-toggle-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}

.nerd-toggle-description {
    font-size: 0.9rem;
    color: #334155;
    margin-top: 0.2rem;
}

.nerd-toggle-card label[data-testid="stToggle"] {
    display: flex;
    justify-content: flex-end;
}

.nerd-toggle-card label[data-testid="stToggle"] > div[role="switch"] {
    background: rgba(148, 163, 184, 0.45);
    box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.12);
}

.nerd-toggle-card label[data-testid="stToggle"] > div[role="switch"][aria-checked="true"] {
    background: linear-gradient(120deg, rgba(37, 99, 235, 0.85), rgba(59, 130, 246, 0.75));
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.35);
}

.info-metric-grid {
    display: grid;
    gap: 0.9rem;
}

.info-metric-card {
    padding: 0.95rem 1.1rem;
    border-radius: 16px;
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.85), rgba(15, 23, 42, 0.75));
    color: #e2e8f0;
    box-shadow: 0 18px 36px rgba(15, 23, 42, 0.22);
}

.info-metric-card .label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.8;
}

.info-metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: 0.35rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.92), rgba(30, 41, 59, 0.85));
    border-right: 1px solid rgba(30, 41, 59, 0.35);
    color: #f8fafc;
}

[data-testid="stSidebar"] > div:first-child {
    padding: 2.6rem 1.6rem 2.1rem;
}

.sidebar-shell {
    display: flex;
    flex-direction: column;
    gap: 2rem;
}

.sidebar-brand {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
}

.sidebar-brand .sidebar-title {
    margin: 0;
    font-size: 1.18rem;
    font-weight: 700;
    color: #1d4ed8;
}

.sidebar-brand .sidebar-subtitle {
    margin: 0;
    color: rgba(248, 250, 252, 0.8);
    line-height: 1.6;
    font-size: 0.95rem;
}

.sidebar-panel {
    display: grid;
    gap: 1.2rem;
}

.sidebar-panel h4 {
    margin: 0;
    font-size: 1rem;
    color: rgba(248, 250, 252, 0.85);
}

.sidebar-progress {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}

.sidebar-stage-card {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: 0.75rem;
    align-items: center;
    padding: 0.8rem 0.9rem;
    border-radius: 16px;
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.18);
    box-shadow: 0 20px 42px rgba(15, 23, 42, 0.16);
}

.sidebar-stage-card__icon {
    display: grid;
    place-items: center;
    width: 48px;
    height: 48px;
    border-radius: 16px;
    background: rgba(59, 130, 246, 0.14);
    color: #1d4ed8;
    font-size: 1.3rem;
    box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.28);
}

.sidebar-stage-card__meta {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.sidebar-stage-card__eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(248, 250, 252, 0.65);
}

.sidebar-stage-card__title {
    margin: 0;
    font-size: 1.02rem;
    font-weight: 700;
    color: #f8fafc;
}

.sidebar-stage-card__description {
    margin: 0;
    color: rgba(248, 250, 252, 0.75);
    line-height: 1.55;
    font-size: 0.9rem;
}

[data-testid="stSidebar"] .stToggle {
    padding: 0.5rem 0.25rem 0.2rem;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    border-radius: 14px;
    border: none;
    background: linear-gradient(140deg, #1d4ed8, #312e81);
    color: #f8fafc;
    font-weight: 600;
    box-shadow: 0 18px 42px rgba(30, 64, 175, 0.28);
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(140deg, #1e40af, #1d4ed8);
    box-shadow: 0 22px 48px rgba(30, 64, 175, 0.32);
}

[data-testid="stSidebar"] .stButton > button:focus-visible {
    outline: 3px solid rgba(59, 130, 246, 0.55);
    outline-offset: 2px;
}

.metric-highlight {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.85rem 1.1rem;
    border-radius: 16px;
    background: rgba(52, 211, 153, 0.14);
    border: 1px solid rgba(52, 211, 153, 0.28);
    color: #047857;
    font-weight: 600;
    margin: 1rem 0;
}

.metric-highlight svg {
    width: 28px;
    height: 28px;
}

.pill-group {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin: 1rem 0;
}

.pill-group .pill {
    padding: 0.45rem 0.9rem;
    border-radius: 999px;
    background: rgba(59, 130, 246, 0.12);
    color: #1d4ed8;
    font-size: 0.85rem;
    font-weight: 600;
}
</style>
"""

STAGE_TEMPLATE_CSS = """
<style>
:root {
    --stage-card-radius: 18px;
    --stage-card-border: rgba(15, 23, 42, 0.08);
    --stage-card-shadow: 0 18px 40px rgba(15, 23, 42, 0.10);
}

.stage-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.2rem;
    margin-top: 1.2rem;
}

.stage-progress-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin: 1rem 0 0;
    padding: 0.5rem 0.75rem;
    border-radius: 16px;
    background: rgba(15, 23, 42, 0.03);
    border: 1px solid rgba(148, 163, 184, 0.24);
    overflow-x: auto;
}

.stage-progress-grid::-webkit-scrollbar {
    height: 6px;
}

.stage-progress-grid::-webkit-scrollbar-thumb {
    background: rgba(100, 116, 139, 0.35);
    border-radius: 999px;
}

.stage-card {
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.96), rgba(241, 245, 255, 0.9));
    border-radius: var(--stage-card-radius);
    border: 1px solid var(--stage-card-border);
    box-shadow: var(--stage-card-shadow);
    padding: 1.15rem 1.2rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    color: #0f172a;
    position: relative;
    overflow: hidden;
    display: block;
    text-decoration: none;
}

.stage-card:link,
.stage-card:visited {
    color: #0f172a;
    text-decoration: none;
}

.stage-card::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at top right, rgba(74, 108, 247, 0.18), transparent 55%);
    opacity: 0;
    transition: opacity 0.2s ease;
}

.stage-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 22px 50px rgba(15, 23, 42, 0.14);
}

.stage-card:hover::after {
    opacity: 1;
}

.stage-card:focus-visible {
    outline: 3px solid rgba(74, 108, 247, 0.45);
    outline-offset: 3px;
}

.stage-card .stage-icon {
    font-size: 1.85rem;
    line-height: 1;
    margin-bottom: 0.35rem;
}

.stage-card .stage-title {
    font-size: 1.05rem;
    font-weight: 600;
}

.stage-card .stage-summary {
    margin-top: 0.55rem;
    font-size: 0.92rem;
    line-height: 1.5;
    color: #334155;
}

.stage-card.active {
    border-color: rgba(74, 108, 247, 0.55);
    background: linear-gradient(155deg, rgba(74, 108, 247, 0.18), rgba(241, 245, 255, 0.96));
}

.stage-card.active::after {
    opacity: 1;
}

.stage-card__body {
    display: grid;
    gap: 0.65rem;
}

.stage-card .stage-meta {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    font-size: 0.82rem;
    color: rgba(15, 23, 42, 0.65);
}

.stage-card .stage-meta span {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
}

.stage-progress-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.75rem;
    border-radius: 999px;
    background: rgba(59, 130, 246, 0.18);
    color: #1d4ed8;
    font-weight: 600;
    font-size: 0.85rem;
}
</style>
"""

STAGE_TEMPLATE_CSS += """
<style>
.stage-navigation-info {
    text-align: center;
    padding: 1.2rem 1.5rem;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    background: linear-gradient(150deg, rgba(255, 255, 255, 0.96), rgba(226, 232, 240, 0.88));
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
    color: #0f172a;
}

.stage-navigation-step {
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(15, 23, 42, 0.55);
    margin-bottom: 0.35rem;
}

.stage-navigation-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.stage-navigation-description {
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0;
    color: rgba(15, 23, 42, 0.75);
}
</style>
"""

EMAIL_INBOX_TABLE_CSS = """
<style>
.email-inbox-wrapper {
    border: 1px solid rgba(15, 23, 42, 0.12);
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12);
    background: linear-gradient(145deg, rgba(255, 255, 255, 0.96), rgba(241, 245, 255, 0.92));
    margin: 0.75rem 0 1.1rem;
}

.email-inbox-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 1.1rem;
    background: linear-gradient(120deg, rgba(30, 64, 175, 0.9), rgba(59, 130, 246, 0.85));
    color: #f8fafc;
    font-weight: 600;
    font-size: 0.94rem;
}

.email-inbox-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    color: #0f172a;
}

.email-inbox-table thead {
    background: rgba(226, 232, 240, 0.75);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.76rem;
    color: #1e293b;
}

.email-inbox-table th,
.email-inbox-table td {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.35);
    vertical-align: top;
}

.email-inbox-table tbody tr:hover {
    background: rgba(191, 219, 254, 0.45);
}

.email-inbox-table tbody tr:nth-child(even) {
    background: rgba(248, 250, 252, 0.85);
}

.email-inbox-empty {
    padding: 1rem 1.25rem;
    color: #475569;
    font-style: italic;
}
</style>
"""

LIFECYCLE_CYCLE_CSS = dedent(
    """
    <style>
    .lifecycle-wrap {
        margin-top: 0.5rem;
    }

    .lifecycle-wrap h4 {
        margin: 0 0 0.4rem 0;
    }

    .lifecycle-wrap p {
        margin: 0 0 0.8rem 0;
        color: rgba(15, 23, 42, 0.78);
    }

    .lifecycle-cycle {
        display: grid;
        place-items: center;
        padding: 1.1rem 0;
    }

    .cycle-ring {
        position: relative;
        width: clamp(300px, 62vw, 720px);
        aspect-ratio: 1 / 1;
        border-radius: 50%;
        background: radial-gradient(120% 120% at 50% 45%, rgba(59, 130, 246, 0.08), rgba(59, 130, 246, 0.02) 55%, rgba(14, 165, 233, 0.04) 100%);
        box-shadow:
            inset 0 0 0 1px rgba(15, 23, 42, 0.06),
            inset 0 -18px 34px rgba(37, 99, 235, 0.08);
    }

    .cycle-ring:after {
        content: "";
        position: absolute;
        inset: 14%;
        border-radius: 50%;
        background: radial-gradient(ellipse at center, rgba(125, 211, 252, 0.12), transparent 65%);
        pointer-events: none;
    }

    .cycle-node {
        position: absolute;
        display: grid;
        place-items: center;
        gap: 0.3rem;
        width: 96px;
        height: 96px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.96);
        box-shadow:
            0 14px 32px rgba(15, 23, 42, 0.12),
            inset 0 0 0 1px rgba(15, 23, 42, 0.06);
        transition: transform 140ms ease, box-shadow 140ms ease;
        cursor: default;
        --cycle-node-transform: translate(-50%, -50%);
        transform: var(--cycle-node-transform);
    }

    .cycle-node:focus-visible {
        outline: 2px solid rgba(59, 130, 246, 0.7);
        outline-offset: 4px;
    }

    .cycle-node:hover,
    .cycle-node:focus-visible {
        transform: var(--cycle-node-transform) translateY(-4px);
        box-shadow:
            0 18px 36px rgba(15, 23, 42, 0.16),
            inset 0 0 0 1px rgba(15, 23, 42, 0.08);
    }

    .cycle-icon {
        font-size: 1.35rem;
        line-height: 1;
    }

    .cycle-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #0f172a;
        text-align: center;
    }

    .cycle-tip {
        position: absolute;
        left: 50%;
        top: calc(100% + 10px);
        transform: translateX(-50%);
        min-width: 220px;
        max-width: 280px;
        padding: 0.6rem 0.75rem;
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.95);
        color: white;
        font-size: 0.82rem;
        line-height: 1.45;
        box-shadow: 0 18px 38px rgba(15, 23, 42, 0.26);
        opacity: 0;
        visibility: hidden;
        transition: opacity 140ms ease, visibility 140ms ease, transform 140ms ease;
        transform-origin: top center;
        transform: translateX(-50%) translateY(-6px);
        z-index: 5;
        pointer-events: none;
    }

    .cycle-tip:before {
        content: "";
        position: absolute;
        top: -7px;
        left: 50%;
        transform: translateX(-50%) rotate(45deg);
        width: 14px;
        height: 14px;
        background: rgba(15, 23, 42, 0.95);
    }

    .cycle-node:hover .cycle-tip,
    .cycle-node:focus-within .cycle-tip {
        opacity: 1;
        visibility: visible;
        transform: translateX(-50%) translateY(0);
    }

    .cycle-arrow {
        position: absolute;
        font-size: 1.15rem;
        color: rgba(30, 58, 138, 0.65);
        text-shadow: 0 4px 10px rgba(15, 23, 42, 0.18);
        user-select: none;
    }

    .cycle-loop {
        position: absolute;
        right: 8%;
        bottom: 10%;
        font-size: 1.35rem;
        color: rgba(30, 64, 175, 0.65);
        text-shadow: 0 4px 12px rgba(15, 23, 42, 0.16);
    }

    .cycle-node--prepare { top: 6%; left: 50%; --cycle-node-transform: translate(-50%, 0); }
    .cycle-node--train   { top: 30%; left: 88%; --cycle-node-transform: translate(-50%, -50%); }
    .cycle-node--evaluate{ top: 74%; left: 74%; --cycle-node-transform: translate(-50%, -50%); }
    .cycle-node--use     { top: 88%; left: 26%; --cycle-node-transform: translate(-50%, -50%); }

    .cycle-arrow--prepare-train   { top: 18%; left: 72%; }
    .cycle-arrow--train-evaluate  { top: 52%; left: 83%; }
    .cycle-arrow--evaluate-use    { top: 84%; left: 50%; transform: translateX(-50%); }
    .cycle-arrow--use-prepare     { top: 30%; left: 18%; transform: rotate(-12deg); }

    .lifecycle-legend {
        display: grid;
        gap: 0.9rem;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin-top: 1.2rem;
    }

    .lifecycle-legend__item {
        border-radius: 16px;
        padding: 0.85rem 1rem 0.95rem;
        background: linear-gradient(155deg, rgba(248, 250, 252, 0.95), rgba(226, 232, 240, 0.55));
        box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.25);
        display: grid;
        gap: 0.45rem;
    }

    .lifecycle-legend__header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .lifecycle-legend__icon {
        font-size: 1.1rem;
    }

    .lifecycle-legend__title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
    }

    .lifecycle-legend__body {
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.55;
        color: rgba(15, 23, 42, 0.78);
    }

    @media (max-width: 1100px) {
        .lifecycle-legend {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 768px) {
        .cycle-node {
            width: 88px;
            height: 88px;
        }

        .cycle-title {
            font-size: 0.82rem;
        }

        .lifecycle-legend {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }
    }

    @media (max-width: 640px) {
        .cycle-node {
            width: 76px;
            height: 76px;
            gap: 0.2rem;
        }

        .cycle-title {
            font-size: 0.78rem;
        }

        .cycle-tip {
            min-width: 200px;
            max-width: 230px;
        }

        .lifecycle-legend {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """
)


LIFECYCLE_CYCLE_HTML = dedent(
    """
    <section class="lifecycle-wrap" aria-label="AI system lifecycle overview">
        <h4>Map the lifecycle of your AI system</h4>
        <p>Progress through the four interconnected stages below. Each phase feeds the next, forming a loop you can revisit as the system evolves.</p>

        <div class="cycle-ring" role="presentation">
            <div
                class="cycle-node cycle-node--prepare"
                tabindex="0"
                aria-labelledby="cycle-title-prepare"
                aria-describedby="cycle-desc-prepare"
            >
                <span class="cycle-icon" aria-hidden="true">📊</span>
                <span id="cycle-title-prepare" class="cycle-title">Prepare Data</span>
                <div class="cycle-tip" role="tooltip">
                    Gather representative emails, label them carefully, and scrub PII so the model learns from balanced, trustworthy examples.
                </div>
            </div>

            <div class="cycle-arrow cycle-arrow--prepare-train" aria-hidden="true">➝</div>

            <div
                class="cycle-node cycle-node--train"
                tabindex="0"
                aria-labelledby="cycle-title-train"
                aria-describedby="cycle-desc-train"
            >
                <span class="cycle-icon" aria-hidden="true">🧠</span>
                <span id="cycle-title-train" class="cycle-title">Train</span>
                <div class="cycle-tip" role="tooltip">
                    Feed the curated dataset into your learning pipeline, keep a validation split aside, and iterate until performance stabilises.
                </div>
            </div>

            <div class="cycle-arrow cycle-arrow--train-evaluate" aria-hidden="true">➝</div>

            <div
                class="cycle-node cycle-node--evaluate"
                tabindex="0"
                aria-labelledby="cycle-title-evaluate"
                aria-describedby="cycle-desc-evaluate"
            >
                <span class="cycle-icon" aria-hidden="true">🧪</span>
                <span id="cycle-title-evaluate" class="cycle-title">Evaluate</span>
                <div class="cycle-tip" role="tooltip">
                    Inspect precision and recall, review borderline decisions, and tune thresholds to reflect your risk posture.
                </div>
            </div>

            <div class="cycle-arrow cycle-arrow--evaluate-use" aria-hidden="true">➝</div>

            <div
                class="cycle-node cycle-node--use"
                tabindex="0"
                aria-labelledby="cycle-title-use"
                aria-describedby="cycle-desc-use"
            >
                <span class="cycle-icon" aria-hidden="true">📬</span>
                <span id="cycle-title-use" class="cycle-title">Use</span>
                <div class="cycle-tip" role="tooltip">
                    Deploy the model to live traffic, monitor its calls in context, and capture feedback to enrich the next training loop.
                </div>
            </div>

            <div class="cycle-arrow cycle-arrow--use-prepare" aria-hidden="true">➝</div>
            <div class="cycle-loop" aria-hidden="true">↺</div>
        </div>

        <div class="lifecycle-legend" role="list">
            <div class="lifecycle-legend__item" id="cycle-desc-prepare" role="listitem">
                <div class="lifecycle-legend__header">
                    <span class="lifecycle-legend__icon" aria-hidden="true">📊</span>
                    <span class="lifecycle-legend__title">Prepare Data</span>
                </div>
                <p class="lifecycle-legend__body">
                    Assemble balanced, well-labeled examples and strip sensitive signals so the dataset reflects the behaviour you intend to teach.
                </p>
            </div>
            <div class="lifecycle-legend__item" id="cycle-desc-train" role="listitem">
                <div class="lifecycle-legend__header">
                    <span class="lifecycle-legend__icon" aria-hidden="true">🧠</span>
                    <span class="lifecycle-legend__title">Train</span>
                </div>
                <p class="lifecycle-legend__body">
                    Run the learning routine against the prepared corpus, reserve a holdout split, and iterate on settings until the model captures durable patterns.
                </p>
            </div>
            <div class="lifecycle-legend__item" id="cycle-desc-evaluate" role="listitem">
                <div class="lifecycle-legend__header">
                    <span class="lifecycle-legend__icon" aria-hidden="true">🧪</span>
                    <span class="lifecycle-legend__title">Evaluate</span>
                </div>
                <p class="lifecycle-legend__body">
                    Review metrics and edge cases, pressure-test failure scenarios, and calibrate decision thresholds against your compliance requirements.
                </p>
            </div>
            <div class="lifecycle-legend__item" id="cycle-desc-use" role="listitem">
                <div class="lifecycle-legend__header">
                    <span class="lifecycle-legend__icon" aria-hidden="true">📬</span>
                    <span class="lifecycle-legend__title">Use</span>
                </div>
                <p class="lifecycle-legend__body">
                    Put the system in front of incoming messages, supervise the outcomes, and log feedback to kick off the next improvement sprint.
                </p>
            </div>
        </div>
    </section>
    """
)


@dataclass(frozen=True)
class StageMeta:
    key: str
    title: str
    icon: str
    summary: str
    description: str


STAGES: List[StageMeta] = [
    StageMeta("intro", "Welcome", "🚀", "Kickoff", "Meet the mission and trigger the guided build."),
    StageMeta(
        "overview",
        "Start your machine",
        "🧭",
        "See the journey",
        "Tour the steps, set Nerd Mode, and align on what you'll do.",
    ),
    StageMeta("data", "Prepare Data", "📊", "Curate examples", "Inspect labeled emails and get the dataset ready for learning."),
    StageMeta("train", "Train", "🧠", "Teach the model", "Configure the split and teach the model on your dataset."),
    StageMeta("evaluate", "Evaluate", "🧪", "Check results", "Check metrics, inspect the confusion matrix, and stress-test."),
    StageMeta("classify", "Use", "📬", "Route emails", "Route new messages, correct predictions, and adapt."),
    StageMeta("model_card", "Model Card", "📄", "Document system", "Summarize performance, intended use, and governance."),
]

STAGE_INDEX = {stage.key: idx for idx, stage in enumerate(STAGES)}
STAGE_BY_KEY = {stage.key: stage for stage in STAGES}

CLASSES = ["spam", "safe"]
AUTONOMY_LEVELS = [
    "Moderate autonomy (recommendation)",
    "High autonomy (auto-route)",
]

SUSPICIOUS_TLD_SUFFIXES = (
    ".ru",
    ".top",
    ".xyz",
    ".click",
    ".pw",
    ".info",
    ".icu",
    ".win",
    ".gq",
    ".tk",
    ".cn",
)

DATASET_SUSPICIOUS_TLDS = [
    ".ru",
    ".xyz",
    ".top",
    ".biz",
    ".win",
    ".loan",
    ".live",
    ".icu",
    ".cn",
    ".gq",
]

DATASET_LEGIT_DOMAINS = [
    "intranet.corp",
    "sharepoint.corp",
    "confluence.corp",
    "workday.corp",
    "hr.corp",
    "vpn.corp",
    "it.corp",
]

BRANDS = ["DocuSign", "Office 365", "OneDrive", "Zoom", "Adobe", "Okta", "Teams"]

COURIERS = ["DHL", "UPS", "FedEx", "PostNL", "PostNord"]

URGENCY = ["URGENT", "ACTION REQUIRED", "FINAL NOTICE", "IMMEDIATE", "24H", "TODAY"]
