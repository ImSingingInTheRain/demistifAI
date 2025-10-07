from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="demistifAI", page_icon="📧", layout="wide")

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
    display: none;
}

.section-surface-block {
    position: relative;
    margin-bottom: 1.45rem;
    border-radius: var(--surface-radius);
    border: 1px solid var(--surface-border);
    background: var(--surface-gradient);
    box-shadow: var(--surface-shadow);
    padding: 1.85rem 2.1rem;
    overflow: hidden;
    color: #0f172a;
}

.section-surface-block::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(150deg, rgba(59, 130, 246, 0.15), transparent 65%);
    opacity: 0;
    transition: opacity 0.3s ease;
    pointer-events: none;
}

.section-surface-block:hover::before {
    opacity: 1;
}

.section-surface-block > [data-testid="stElementContainer"] {
    margin-bottom: 0.9rem;
}

.section-surface-block > [data-testid="stElementContainer"]:first-child,
.section-surface-block > [data-testid="stElementContainer"]:last-child {
    margin-bottom: 0;
}

.section-surface-block h2,
.section-surface-block h3,
.section-surface-block h4 {
    margin-top: 0;
    color: inherit;
}

.section-surface-block p,
.section-surface-block ul {
    font-size: 0.98rem;
    line-height: 1.65;
    color: inherit;
}

.section-surface-block ul {
    padding-left: 1.25rem;
}

.section-surface-block .section-caption {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: rgba(15, 23, 42, 0.65);
}

.section-surface-block .section-caption span {
    display: inline-flex;
    width: 40px;
    height: 2px;
    background: rgba(15, 23, 42, 0.12);
}

.section-surface-block.section-surface--hero {
    padding: 2.6rem 2.8rem;
    background: linear-gradient(160deg, #1d4ed8, #312e81);
    color: #f8fafc;
    border: 1px solid rgba(255, 255, 255, 0.28);
    box-shadow: 0 28px 70px rgba(30, 64, 175, 0.35);
}

.section-surface-block.section-surface--hero::before {
    display: none;
}

.section-surface-block.section-surface--hero .section-caption {
    color: rgba(241, 245, 249, 0.85);
}

.section-surface-block.section-surface--hero .section-caption span {
    background: rgba(241, 245, 249, 0.35);
}

.section-surface-block .surface-columns {
    display: grid;
    gap: 1.4rem;
}

.section-surface-block .surface-columns.two {
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}

.section-surface-block .surface-columns.three {
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
    color: #e2e8f0;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {
    color: #e2e8f0;
}

[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.5);
    border-radius: 999px;
}

[data-testid="stAlert"] {
    border-radius: 18px !important;
    padding: 1rem 1.25rem !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
    box-shadow: 0 18px 36px rgba(15, 23, 42, 0.14) !important;
}

.stButton>button,
.stDownloadButton>button {
    border-radius: 999px;
    font-weight: 600;
    letter-spacing: 0.01em;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    padding: 0.35rem;
    background: rgba(148, 163, 184, 0.22);
    border-radius: 999px;
    margin-bottom: 1.25rem;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 999px;
    padding: 0.55rem 1.2rem;
    color: #1e293b;
    font-weight: 600;
    border: 1px solid transparent;
    transition: background 0.2s ease, color 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.82);
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.16);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(140deg, rgba(59, 130, 246, 0.22), rgba(37, 99, 235, 0.4));
    color: #0f172a;
    box-shadow: 0 18px 40px rgba(37, 99, 235, 0.28);
    border-color: rgba(37, 99, 235, 0.35);
    transform: translateY(-1px);
}

.stTabs [data-baseweb="tab"]:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.35);
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 0.35rem;
}

.mailbox-summary {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.mailbox-summary-card {
    border-radius: 18px;
    padding: 1rem 1.2rem;
    background: linear-gradient(160deg, rgba(255, 255, 255, 0.96), rgba(226, 232, 240, 0.75));
    border: 1px solid rgba(148, 163, 184, 0.35);
    box-shadow: 0 18px 38px rgba(15, 23, 42, 0.14);
}

.mailbox-summary-card .summary-title {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(15, 23, 42, 0.58);
    margin-bottom: 0.35rem;
}

.mailbox-summary-card .summary-value {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.05;
}

.mailbox-summary-card .summary-hint {
    margin-top: 0.6rem;
    font-size: 0.9rem;
    color: #475569;
}

.mailbox-summary-card .summary-rows {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    margin-top: 0.35rem;
}

.mailbox-summary-card .summary-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.92rem;
    color: #1e293b;
}

.mailbox-summary-card .summary-row span:last-child {
    font-weight: 600;
}

.hero-info-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.9rem;
}

.hero-info-card {
    position: relative;
    border-radius: 20px;
    padding: 1.15rem 1.35rem 1.25rem;
    background: linear-gradient(160deg, rgba(191, 219, 254, 0.85), rgba(147, 197, 253, 0.65));
    border: 1px solid rgba(59, 130, 246, 0.28);
    box-shadow: 0 24px 46px rgba(30, 64, 175, 0.18);
    color: #0f172a;
}

.hero-info-card::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    background: linear-gradient(140deg, rgba(59, 130, 246, 0.12), rgba(29, 78, 216, 0.12));
    opacity: 0;
    transition: opacity 0.3s ease;
    pointer-events: none;
}

.hero-info-card:hover::after {
    opacity: 1;
}

.hero-info-card > * {
    position: relative;
    z-index: 1;
}

.hero-info-card h3 {
    margin: 0 0 0.55rem;
    font-size: 1.05rem;
    color: #1d4ed8;
}

.hero-info-card p {
    margin: 0;
    font-size: 0.95rem;
    line-height: 1.6;
    color: rgba(15, 23, 42, 0.82);
}

.hero-info-steps {
    margin: 0.55rem 0 0;
    padding: 0;
    list-style: none;
    display: grid;
    gap: 0.4rem;
}

.hero-info-steps li {
    display: grid;
    grid-template-columns: 32px minmax(0, 1fr);
    gap: 0.5rem;
    align-items: start;
    font-size: 0.93rem;
    color: rgba(15, 23, 42, 0.82);
}

.hero-info-steps .step-index {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 999px;
    background: rgba(37, 99, 235, 0.18);
    color: #1d4ed8;
    font-weight: 600;
    font-size: 0.85rem;
    box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.28);
}

.lifecycle-flow {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    padding: 0.85rem 1rem;
    margin-top: 0.85rem;
    border-radius: 999px;
    background: linear-gradient(120deg, rgba(191, 219, 254, 0.65), rgba(147, 197, 253, 0.35));
    box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.22);
}

.lifecycle-step {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.85);
    box-shadow: 0 12px 28px rgba(30, 64, 175, 0.18);
    color: #1d4ed8;
    font-weight: 600;
    font-size: 0.95rem;
}

.lifecycle-step .lifecycle-icon {
    font-size: 1.1rem;
}

.lifecycle-step .lifecycle-label {
    color: #0f172a;
    font-weight: 600;
}

.lifecycle-arrow {
    font-size: 1.15rem;
    color: rgba(15, 23, 42, 0.7);
}

.lifecycle-loop {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: rgba(30, 64, 175, 0.12);
    color: #1d4ed8;
    font-size: 1.1rem;
    box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.22);
}

@media (max-width: 1024px) {
    .hero-info-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 768px) {
    .hero-info-grid {
        grid-template-columns: 1fr;
    }
}

.mailbox-summary-card .summary-list {
    margin: 0.45rem 0 0;
    padding-left: 1.1rem;
    color: #1e293b;
    font-size: 0.92rem;
}

.mailbox-summary-card .summary-list li {
    margin-bottom: 0.25rem;
}

.mailbox-summary-card--empty {
    text-align: center;
    color: #475569;
    font-style: italic;
}

.mailbox-summary-card--empty strong {
    display: block;
    margin-bottom: 0.35rem;
}

@media (max-width: 1100px) {
    .mailbox-summary {
        flex-direction: row;
        flex-wrap: wrap;
    }
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

st.markdown(APP_THEME_CSS, unsafe_allow_html=True)

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

.stage-card.complete {
    border-color: rgba(52, 199, 123, 0.55);
    background: linear-gradient(155deg, rgba(52, 199, 123, 0.18), rgba(240, 253, 244, 0.95));
}

.stage-progress-grid .stage-card {
    min-height: auto;
    padding: 0.45rem 0.75rem;
    border-radius: 999px;
    box-shadow: none;
    background: rgba(255, 255, 255, 0.94);
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    text-align: left;
    cursor: pointer;
}

.stage-progress-grid .stage-card.complete {
    background: linear-gradient(145deg, rgba(52, 199, 123, 0.16), rgba(240, 253, 244, 0.92));
    border-color: rgba(52, 199, 123, 0.45);
}

.stage-progress-grid .stage-card.active {
    background: linear-gradient(145deg, rgba(74, 108, 247, 0.16), rgba(241, 245, 255, 0.95));
    border-color: rgba(74, 108, 247, 0.45);
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.15);
}

.stage-progress-grid .stage-icon {
    font-size: 1.15rem;
    margin-bottom: 0;
}

.stage-progress-grid .stage-summary {
    display: none;
}

.stage-progress-grid .stage-title {
    font-size: 0.9rem;
    font-weight: 600;
}

.stage-nav-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 1.75rem;
}

.stage-nav-buttons .stButton>button {
    flex: 1;
    min-width: 200px;
    border-radius: 999px;
    font-weight: 600;
    padding: 0.65rem 1.1rem;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
}

.stage-nav-buttons .stButton>button:hover {
    box-shadow: 0 16px 32px rgba(15, 23, 42, 0.12);
}

.stage-nav-buttons .stButton>button:focus {
    outline: none;
    box-shadow: 0 0 0 3px rgba(74, 108, 247, 0.35);
}

.ai-quote-box {
    margin: 1.35rem 0;
    padding: 1.4rem 1.6rem;
    border-radius: 20px;
    border: 1px solid rgba(37, 99, 235, 0.28);
    background: linear-gradient(140deg, rgba(219, 234, 254, 0.82), rgba(191, 219, 254, 0.45));
    color: #0f172a;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 1rem;
    align-items: flex-start;
    box-shadow: 0 18px 42px rgba(30, 64, 175, 0.16);
    position: relative;
    overflow: hidden;
}

.ai-quote-box::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    background: linear-gradient(160deg, rgba(37, 99, 235, 0.18), transparent 62%);
    pointer-events: none;
}

.ai-quote-box__icon {
    font-size: 1.8rem;
    line-height: 1;
    color: #1d4ed8;
    position: relative;
    z-index: 1;
    margin-top: 0.2rem;
}

.ai-quote-box__content {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
}

.ai-quote-box__source {
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(30, 64, 175, 0.75);
}

.ai-quote-box p {
    margin: 0;
    font-size: 1rem;
    line-height: 1.6;
    color: #0f172a;
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

LIFECYCLE_CYCLE_CSS = """
<style>
.lifecycle-cycle {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 0.9rem;
    margin: 1.2rem 0 1.6rem;
}

.lifecycle-cycle .cycle-node {
    width: 118px;
    height: 118px;
    border-radius: 50%;
    background: linear-gradient(160deg, rgba(59, 130, 246, 0.15), rgba(59, 130, 246, 0.05));
    border: 2px solid rgba(37, 99, 235, 0.35);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 0.65rem;
    color: #1e3a8a;
    font-weight: 600;
    box-shadow: inset 0 0 12px rgba(30, 64, 175, 0.18), 0 18px 34px rgba(15, 23, 42, 0.12);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.lifecycle-cycle .cycle-node .cycle-icon {
    font-size: 1.8rem;
    margin-bottom: 0.35rem;
}

.lifecycle-cycle .cycle-node-active {
    background: linear-gradient(160deg, rgba(52, 211, 153, 0.25), rgba(52, 211, 153, 0.1));
    border-color: rgba(5, 150, 105, 0.55);
    color: #065f46;
    transform: scale(1.04);
    box-shadow: inset 0 0 16px rgba(16, 185, 129, 0.25), 0 20px 40px rgba(13, 148, 136, 0.18);
}

.lifecycle-cycle .cycle-arrow {
    font-size: 1.75rem;
    color: rgba(30, 64, 175, 0.65);
}

.lifecycle-cycle .cycle-loop {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    border: 2px dashed rgba(30, 64, 175, 0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.65rem;
    color: rgba(30, 64, 175, 0.7);
    box-shadow: inset 0 0 12px rgba(37, 99, 235, 0.15);
}
</style>
"""

st.markdown(STAGE_TEMPLATE_CSS, unsafe_allow_html=True)


@contextmanager
def section_surface(class_name: str = ""):
    block = st.container()
    marker_id = f"section-surface-{uuid4().hex}"
    classes = "section-surface" + (f" {class_name}" if class_name else "")
    marker = block.empty()
    marker.markdown(f"<div id='{marker_id}' class='{classes}'></div>", unsafe_allow_html=True)

    applied_classes = class_name.split() if class_name else []
    class_script = "".join(f"block.classList.add('{cls}');" for cls in applied_classes)
    components.html(
        f"""
<script>
(function() {{
  const attach = () => {{
    const rootDoc = window.parent?.document;
    if (!rootDoc) {{
      return;
    }}
    const marker = rootDoc.getElementById('{marker_id}');
    if (!marker) {{
      return;
    }}
    const elementContainer = marker.closest('[data-testid="stElementContainer"]');
    const wrapper = elementContainer?.parentElement?.nextElementSibling;
    const block = wrapper?.querySelector('[data-testid="stVerticalBlock"]') ?? marker.closest('[data-testid="stVerticalBlock"]');
    if (!block) {{
      setTimeout(attach, 50);
      return;
    }}
    block.classList.add('section-surface-block');
    {class_script}
    if (elementContainer) {{
      elementContainer.remove();
    }} else {{
      marker.remove();
    }}
  }};
  attach();
}})();
</script>
""",
        height=0,
    )

    with block:
        yield


def render_nerd_mode_toggle(
    *,
    key: str,
    title: str = "Nerd Mode",
    description: Optional[str] = None,
    icon: str = "🧠",
    default: Optional[bool] = None,
) -> bool:
    """Render a styled Nerd Mode toggle and return its value."""

    existing = st.session_state.get(key, None)
    initial_value = existing if existing is not None else default

    container = st.container()
    with container:
        st.markdown("<div class='nerd-toggle-card'>", unsafe_allow_html=True)
        info_col, toggle_col = st.columns([4, 1])
        with info_col:
            st.markdown(
                f"<div class='nerd-toggle-title'>{html.escape(icon)} {html.escape(title)}</div>",
                unsafe_allow_html=True,
            )
            if description:
                st.markdown(
                    f"<div class='nerd-toggle-description'>{html.escape(description)}</div>",
                    unsafe_allow_html=True,
                )
        with toggle_col:
            value = st.toggle(
                title,
                key=key,
                value=bool(initial_value) if initial_value is not None else False,
                label_visibility="collapsed",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    return value


def _stage_card_html(
    stage: StageMeta,
    status_class: str,
    *,
    show_description: bool,
    clickable: bool,
) -> str:
    body_text = stage.description if show_description else stage.summary
    classes = "stage-card" + (f" {status_class}" if status_class else "")
    tag = "a" if clickable else "div"
    href_attr = ""
    if clickable:
        params = {k: list(st.query_params.get_all(k)) for k in st.query_params}
        params["stage"] = [stage.key]
        href_attr = f" href=\"?{urlencode(params, doseq=True)}\" target=\"_self\""
    aria_attr = " aria-current=\"step\"" if status_class == "active" else ""
    return (
        f"<{tag} class=\"{classes}\"{href_attr}{aria_attr}>"
        "<div class=\"stage-icon\">{icon}</div>"
        "<div class=\"stage-title\">{title}</div>"
        "<div class=\"stage-summary\">{summary}</div>"
        f"</{tag}>"
    ).format(
        icon=html.escape(stage.icon),
        title=html.escape(stage.title),
        summary=html.escape(body_text),
    )


def render_stage_cards(
    active_stage: str,
    *,
    variant: str = "grid",
    stages: Optional[List[StageMeta]] = None,
):
    """Render the reusable stage template as a grid or compact progress bar."""

    wrapper_class = "stage-grid" if variant == "grid" else "stage-progress-grid"
    show_description = variant == "grid"
    active_index = STAGE_INDEX.get(active_stage, 0)
    stage_list = stages if stages is not None else STAGES
    clickable = variant != "grid"
    cards: List[str] = []
    for stage in stage_list:
        idx = STAGE_INDEX[stage.key]
        if idx < active_index:
            status_class = "complete"
        elif idx == active_index:
            status_class = "active"
        else:
            status_class = ""
        cards.append(
            _stage_card_html(
                stage,
                status_class=status_class,
                show_description=show_description,
                clickable=clickable,
            )
        )

    st.markdown(
        f"<div class=\"{wrapper_class}\">{''.join(cards)}</div>",
        unsafe_allow_html=True,
    )


def set_active_stage(stage_key: str):
    if stage_key not in STAGE_BY_KEY:
        return
    st.session_state["active_stage"] = stage_key
    if st.query_params.get_all("stage") != [stage_key]:
        st.query_params["stage"] = stage_key


def render_stage_navigation_controls(active_stage: str):
    idx = STAGE_INDEX.get(active_stage, 0)
    prev_stage = STAGES[idx - 1] if idx > 0 else None
    next_stage = STAGES[idx + 1] if idx < len(STAGES) - 1 else None

    st.markdown("<div class='stage-nav-buttons'>", unsafe_allow_html=True)
    col_prev, col_status, col_next = st.columns([1.2, 0.9, 1.2])

    with col_prev:
        if prev_stage:
            st.button(
                f"⬅️ Back to {prev_stage.title}",
                key=f"nav_back_{active_stage}",
                on_click=set_active_stage,
                args=(prev_stage.key,),
                width="stretch",
            )
        else:
            st.write(" ")

    with col_status:
        if active_stage != "intro":
            stage = STAGE_BY_KEY[active_stage]
            st.markdown(
                """
                <div style="text-align:center; padding:0.65rem 0.75rem; background: rgba(15, 23, 42, 0.03); border-radius: 14px; font-weight: 600; color: #475569;">
                    Currently on <span style="color:#1d4ed8;">{icon} {title}</span>
                </div>
                """.format(icon=html.escape(stage.icon), title=html.escape(stage.title)),
                unsafe_allow_html=True,
            )
        else:
            st.write(" ")

    with col_next:
        if next_stage and active_stage != "intro":
            st.button(
                f"Next • {next_stage.title} ➡️",
                key=f"nav_next_{active_stage}",
                on_click=set_active_stage,
                args=(next_stage.key,),
                width="stretch",
                type="primary",
            )
        else:
            st.write(" ")

    st.markdown("</div>", unsafe_allow_html=True)


def render_email_inbox_table(
    df: pd.DataFrame,
    *,
    title: str,
    subtitle: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> None:
    st.markdown(EMAIL_INBOX_TABLE_CSS, unsafe_allow_html=True)

    if df is None or df.empty:
        header_html = (
            f"<div class='email-inbox-header'><span>{html.escape(title)}</span><span>0 messages</span></div>"
        )
        body_html = html.escape(subtitle) if subtitle else "Inbox is empty."
        st.markdown(
            f"<div class='email-inbox-wrapper'>{header_html}<div class='email-inbox-empty'>{body_html}</div></div>",
            unsafe_allow_html=True,
        )
        return

    display_df = df.copy()
    if columns:
        existing_cols = [col for col in columns if col in display_df.columns]
        if existing_cols:
            display_df = display_df[existing_cols]

    if "title" in display_df.columns:
        display_df.rename(columns={"title": "Subject"}, inplace=True)

    if "body" in df.columns:
        preview_series = df["body"].fillna("").apply(
            lambda text: text if len(text) <= 90 else text[:87].rstrip() + "…"
        )
        insert_position = len(display_df.columns)
        if "Subject" in display_df.columns:
            insert_position = list(display_df.columns).index("Subject") + 1
        display_df.insert(insert_position, "Preview", preview_series)
        if "body" in display_df.columns:
            display_df.drop(columns=["body"], inplace=True)

    header_right = f"{len(df)} message{'s' if len(df) != 1 else ''}"
    header_html = f"<div class='email-inbox-header'><span>{html.escape(title)}</span><span>{html.escape(header_right)}</span></div>"
    subtitle_html = (
        f"<div style='padding:0.45rem 1.1rem; color:#475569;'>{html.escape(subtitle)}</div>"
        if subtitle
        else ""
    )
    table_html = display_df.to_html(classes="email-inbox-table", index=False, escape=False)
    st.markdown(
        f"<div class='email-inbox-wrapper'>{header_html}{subtitle_html}{table_html}</div>",
        unsafe_allow_html=True,
    )


def render_mailbox_summary(df: pd.DataFrame, mailbox_title: str) -> None:
    st.markdown(f"#### Quick insights — {mailbox_title}")

    if df is None or df.empty:
        st.markdown(
            f"""
            <div class="mailbox-summary">
                <div class="mailbox-summary-card mailbox-summary-card--empty">
                    <strong>No messages yet in {html.escape(mailbox_title)}</strong>
                    As soon as emails appear in this mailbox, a snapshot of activity will show up here.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    total = len(df)
    avg_spam = float(df["p_spam"].mean()) if "p_spam" in df.columns else 0.0

    prediction_rows = ""
    if "pred" in df.columns and not df["pred"].isna().all():
        counts = df["pred"].value_counts().sort_values(ascending=False)
        prediction_rows = "".join(
            f"<div class='summary-row'><span>{html.escape(str(label).title())}</span><span>{count} ({count / total:.0%})</span></div>"
            for label, count in counts.items()
        )
    if not prediction_rows:
        prediction_rows = "<div class='summary-row'>No predictions recorded yet.</div>"

    subjects: List[str] = []
    if "title" in df.columns:
        subjects = []
        for subject in df["title"].fillna(""):
            subject_text = str(subject).strip()
            if subject_text:
                subjects.append(html.escape(subject_text))
            if len(subjects) == 3:
                break
    subject_items = "".join(f"<li>{item}</li>" for item in subjects)
    if not subject_items:
        subject_items = "<li>No subject lines available yet.</li>"

    threshold = float(ss.get("threshold", 0.5))

    summary_html = f"""
    <div class="mailbox-summary">
        <div class="mailbox-summary-card">
            <div class="summary-title">Messages</div>
            <div class="summary-value">{total}</div>
            <div class="summary-hint">Avg. spam score: {avg_spam:.0%}</div>
        </div>
        <div class="mailbox-summary-card">
            <div class="summary-title">Prediction mix</div>
            <div class="summary-rows">{prediction_rows}</div>
        </div>
        <div class="mailbox-summary-card">
            <div class="summary-title">Latest subjects</div>
            <ul class="summary-list">{subject_items}</ul>
            <div class="summary-hint">Routing threshold: {threshold:.2f}</div>
        </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)


def render_mailbox_panel(
    records: Optional[List[Dict[str, Any]]],
    *,
    mailbox_title: str,
    filled_subtitle: str,
    empty_subtitle: str,
) -> None:
    df_raw = pd.DataFrame(records) if records else pd.DataFrame()
    table_col, summary_col = st.columns((2.8, 1.2))

    with table_col:
        if df_raw.empty:
            render_email_inbox_table(pd.DataFrame(), title=mailbox_title, subtitle=empty_subtitle)
        else:
            display_df = df_raw.rename(columns={"pred": "Prediction", "p_spam": "P(spam)"})
            render_email_inbox_table(
                display_df,
                title=mailbox_title,
                subtitle=filled_subtitle,
            )

    with summary_col:
        render_mailbox_summary(df_raw, mailbox_title)


def render_lifecycle_cycle(active_stage: str) -> None:
    st.markdown(LIFECYCLE_CYCLE_CSS, unsafe_allow_html=True)

    ordered_keys = ["data", "train", "evaluate", "classify"]
    parts: List[str] = []
    for idx, key in enumerate(ordered_keys):
        stage = STAGE_BY_KEY[key]
        active_cls = " cycle-node-active" if key == active_stage else ""
        parts.append(
            """
            <div class="cycle-node{active_cls}">
                <span class="cycle-icon">{icon}</span>
                <span class="cycle-label">{title}</span>
            </div>
            """.format(
                active_cls=active_cls,
                icon=html.escape(stage.icon),
                title=html.escape(stage.title),
            )
        )
        if idx < len(ordered_keys) - 1:
            parts.append("<div class='cycle-arrow'>➝</div>")

    parts.append("<div class='cycle-loop' title='Back to the start'>↺</div>")
    st.markdown(f"<div class='lifecycle-cycle'>{''.join(parts)}</div>", unsafe_allow_html=True)

CLASSES = ["spam", "safe"]
AUTONOMY_LEVELS = [
    "Moderate autonomy (recommendation)",
    "High autonomy (auto-route)",
]

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

STARTER_LABELED: List[Dict] = [
    # ----------------------- SPAM (100) -----------------------
    {"title": "URGENT: Verify your payroll information today", "body": "Your salary deposit is on hold. Confirm your bank details via this external link to avoid delay.", "label": "spam"},
    {"title": "WIN a FREE iPhone — final round eligibility", "body": "Congratulations! Complete the short survey to claim your iPhone. Offer expires in 2 hours.", "label": "spam"},
    {"title": "Password will expire — action required", "body": "Reset your password here: http://accounts-security.example-reset.com. Failure to act may lock your account.", "label": "spam"},
    {"title": "Delivery notice: package waiting for customs clearance", "body": "Pay a small fee to release your parcel. Use our quick checkout link to avoid return.", "label": "spam"},
    {"title": "Your account was suspended", "body": "Unusual login detected. Open the attached HTML and confirm your credentials to restore access.", "label": "spam"},
    {"title": "Invoice discrepancy for March", "body": "We overcharged your account. Provide your card number for an immediate refund.", "label": "spam"},
    {"title": "Corporate survey: guaranteed €100 voucher", "body": "Finish this 1-minute survey and receive a voucher instantly. No employee ID needed.", "label": "spam"},
    {"title": "Security alert from IT desk", "body": "We’ve updated our security policy. Download the attached ZIP to review the changes.", "label": "spam"},
    {"title": "Limited-time premium subscription at 90% off", "body": "Upgrade now to unlock executive insights. Click the promotional link and pay today.", "label": "spam"},
    {"title": "Payment overdue: settle immediately", "body": "Your service will be interrupted. Wire funds to the account in the attachment to avoid penalties.", "label": "spam"},
    {"title": "HR update: bonus eligibility check", "body": "Confirm your identity by entering your national ID on our verification page to receive your bonus.", "label": "spam"},
    {"title": "Conference invite: free registration + gift", "body": "Register via the link to receive a €50 gift card. Limited seats; confirm with your credit card.", "label": "spam"},
    {"title": "DocuSign: You received a secure document", "body": "Open this third-party portal to review and log in with your email password for access.", "label": "spam"},
    {"title": "Crypto opportunity: double your balance", "body": "Transfer funds to our wallet and we’ll return 2× within 24 hours. Trusted by leaders.", "label": "spam"},
    {"title": "Password reset required immediately", "body": "We noticed unusual activity. Reset at http://security-reset.example-login.net to avoid permanent lock.", "label": "spam"},
    {"title": "Delivery fee required for redelivery", "body": "Your parcel is pending. Pay a €2.99 fee to schedule a new delivery slot.", "label": "spam"},
    {"title": "Payroll correction: refund available", "body": "Send your IBAN and CVV to process your refund now.", "label": "spam"},
    {"title": "COVID relief grant for employees", "body": "Claim your €500 benefit by verifying your bank details today.", "label": "spam"},
    {"title": "Urgent compliance training overdue", "body": "Access the training via our partner portal. Use your email password to sign in.", "label": "spam"},
    {"title": "Security notice: leaked credentials", "body": "Your email was found in a breach. Download the spreadsheet and confirm your password inside.", "label": "spam"},
    {"title": "Special discount on executive coaching", "body": "90% off today only. Pay via the link to secure your slot.", "label": "spam"},
    {"title": "Invoice correction required", "body": "We can refund you now if you send your card number and CVV for verification.", "label": "spam"},
    {"title": "IT ticket auto-closure warning", "body": "To keep your ticket open, log into the external support site and confirm your identity.", "label": "spam"},
    {"title": "One-time password: verify to unlock account", "body": "Enter the OTP on our portal along with your email password to restore access.", "label": "spam"},
    {"title": "Conference pass sponsored — confirm card", "body": "You’ve been awarded a free pass. Confirm your credit card to activate sponsorship.", "label": "spam"},
    {"title": "VPN certificate expired", "body": "Download the new certificate from our public link and install today.", "label": "spam"},
    {"title": "Tax rebate waiting", "body": "We owe you €248. Submit your bank details on our secure (external) claim page.", "label": "spam"},
    {"title": "CEO request: urgent payment", "body": "Transfer €4,900 immediately to the vendor in the attached invoice. Do not call.", "label": "spam"},
    {"title": "Doc review access restricted", "body": "Log in with Office365 password here to unlock the secure document.", "label": "spam"},
    {"title": "Account verification — final warning", "body": "Your mailbox will be closed. Verify now at http://mailbox-verify.example.org.", "label": "spam"},
    {"title": "Prize draw: employee appreciation", "body": "All staff eligible. Enter with your bank card details to receive your prize.", "label": "spam"},
    {"title": "Password reset confirmation (external)", "body": "Confirm reset by clicking this link and entering your credentials to finalize.", "label": "spam"},
    {"title": "Secure file delivered", "body": "Open the HTML attachment, enable macros, and sign in to download the file.", "label": "spam"},
    {"title": "Upgrade your mailbox storage", "body": "Pay €1 via micro-transaction to extend your mailbox by 50GB.", "label": "spam"},
    {"title": "Two-factor disabled", "body": "We turned off MFA on your account. Click the link to re-enable (login required).", "label": "spam"},
    {"title": "SaaS subscription renewal failed", "body": "Provide your card details to avoid losing access to premium features.", "label": "spam"},
    {"title": "Payroll bonus — confirm identity", "body": "Submit your personal ID and mobile TAN on the page to receive your bonus.", "label": "spam"},
    {"title": "Password policy update", "body": "Download the attached PDF from an unknown sender to review mandatory changes.", "label": "spam"},
    {"title": "Company survey — guaranteed gift card", "body": "Complete the survey; a €50 card will be emailed to you instantly after verification.", "label": "spam"},
    {"title": "License key expired", "body": "Activate your software by installing the attached executable and signing in.", "label": "spam"},
    {"title": "Unpaid toll invoice", "body": "Settle your unpaid toll by providing card details at the link below.", "label": "spam"},
    {"title": "Security incident report needed", "body": "Fill in this external Google Form with your employee credentials.", "label": "spam"},
    {"title": "Bank verification needed", "body": "We detected a failed withdrawal. Confirm your bank access to continue.", "label": "spam"},
    {"title": "Password rotation overdue", "body": "Rotate your password here: http://corp-passwords-reset.me to keep access.", "label": "spam"},
    {"title": "Delivery confirmation required", "body": "Click to pay a small customs fee and confirm your address to receive your parcel.", "label": "spam"},
    {"title": "Executive webinar: instant access", "body": "Reserve now; pay via partner portal and log in with your email credentials.", "label": "spam"},
    {"title": "Mail storage full", "body": "To avoid message loss, sign into the storage recovery portal with your password.", "label": "spam"},
    {"title": "Prize payout — action needed", "body": "We are ready to wire your winnings. Send IBAN and CVV to complete transfer.", "label": "spam"},
    {"title": "Compliance escalation", "body": "You are out of compliance. Download the attached document and log in to resolve.", "label": "spam"},
    {"title": "Tax invoice attached — payment portal", "body": "Open the link to pay immediately; failure to do so results in penalties.", "label": "spam"},

    # ----------------------- SAFE (100) -----------------------
    {"title": "Password reset confirmation for corporate portal", "body": "You requested a password change via the internal IT portal. If this wasn’t you, contact IT on Teams.", "label": "safe"},
    {"title": "DHL tracking: package out for delivery", "body": "Your parcel is scheduled today. Track via the official DHL site with your tracking ID (no payment required).", "label": "safe"},
    {"title": "March invoice attached — Accounts Payable", "body": "Hi, please find the March invoice attached in PDF. PO referenced below; no payment info requested.", "label": "safe"},
    {"title": "Minutes from the compliance workshop", "body": "Attached are the minutes and action items agreed during today’s workshop. Feedback welcome by Friday.", "label": "safe"},
    {"title": "Security advisory — internal policy update", "body": "Please review the new password guidelines on the intranet. No external links or attachments.", "label": "safe"},
    {"title": "Reminder: Q4 budget review on Tuesday", "body": "Please upload your cost worksheets to the internal SharePoint before 16:00 CEST.", "label": "safe"},
    {"title": "Travel itinerary update", "body": "Your train platform changed. Updated PDF itinerary is attached; no action required.", "label": "safe"},
    {"title": "Team meeting moved to 14:00", "body": "Join via the regular Teams link. Agenda: quarterly KPIs, risk register, and roadmap.", "label": "safe"},
    {"title": "Draft for review — policy document", "body": "Could you review the attached draft and add comments in Word track changes by EOD Thursday?", "label": "safe"},
    {"title": "Onboarding checklist for new starters", "body": "HR checklist attached; forms to be submitted via Workday. Reach out if anything is unclear.", "label": "safe"},
    {"title": "Canteen menu and wellness events", "body": "This week’s healthy menu and yoga session schedule are included. No RSVP needed.", "label": "safe"},
    {"title": "Customer feedback summary — Q3", "body": "Please see the attached slide deck with survey trends and next steps for CX improvements.", "label": "safe"},
    {"title": "Security alert — new device sign-in (confirmed)", "body": "You signed in from a new laptop. If recognized, no action needed. Audit log available on the intranet.", "label": "safe"},
    {"title": "Coffee catch-up?", "body": "Are you free on Thursday afternoon for a quick chat about the training roadmap?", "label": "safe"},
    {"title": "Password rotation reminder", "body": "This is an automated reminder to rotate your password on the internal portal this month.", "label": "safe"},
    {"title": "Delivery rescheduled by courier", "body": "Your parcel will arrive tomorrow morning. No fees due; track on the official courier portal.", "label": "safe"},
    {"title": "Payroll update posted", "body": "Payslips are available on the HR portal. Do not email personal details; use the secure site.", "label": "safe"},
    {"title": "COVID-19 office guidance", "body": "Updated office access rules are published on the intranet. Masks optional in open areas.", "label": "safe"},
    {"title": "Compliance training enrolled", "body": "You have been enrolled in the mandatory training via the LMS. Deadline next Friday.", "label": "safe"},
    {"title": "Security bulletin — phishing simulation next week", "body": "IT will run a phishing simulation. Do not click unknown links; report suspicious emails via the button.", "label": "safe"},
    {"title": "Corporate survey — help improve the office", "body": "Share your thoughts in a 3-minute internal survey on the intranet (no incentives offered).", "label": "safe"},
    {"title": "Quarterly planning session", "body": "Please add your slides to the shared folder before the meeting. No external links.", "label": "safe"},
    {"title": "Policy update: remote work guidelines", "body": "The updated policy is available on the intranet. Please acknowledge by Friday.", "label": "safe"},
    {"title": "Security alert — new device sign-in", "body": "Was this you? If recognized, ignore. Otherwise, reset password from the internal portal.", "label": "safe"},
    {"title": "AP reminder — PO mismatch on ticket #4923", "body": "Please correct the PO reference in the invoice metadata on SharePoint; no payment info needed.", "label": "safe"},
    {"title": "Diversity & inclusion town hall", "body": "Join the all-hands event next Wednesday. Questions welcome; recording will be posted.", "label": "safe"},
    {"title": "Mentorship program enrollment", "body": "Sign up via the HR portal. Matching will occur next month; no external forms.", "label": "safe"},
    {"title": "Travel approval granted", "body": "Your travel request has been approved. Book via the internal tool; corporate rates apply.", "label": "safe"},
    {"title": "Laptop replacement schedule", "body": "IT will swap your device on Friday. Back up local files; data will sync via OneDrive.", "label": "safe"},
    {"title": "Facilities maintenance notice", "body": "Air-conditioning maintenance on Level 3, 18:00–20:00. Access may be restricted.", "label": "safe"},
    {"title": "Team offsite: dietary requirements", "body": "Please submit your preferences by Wednesday; vegetarian and vegan options available.", "label": "safe"},
    {"title": "All-hands recording posted", "body": "The recording and slides are available on the intranet page for two weeks.", "label": "safe"},
    {"title": "Workday: benefits enrollment opens", "body": "Benefits enrollment opens Monday. Make selections in Workday by the end of the month.", "label": "safe"},
    {"title": "Internal tool outage resolved", "body": "The analytics portal is back online. Root cause will be shared in the incident report.", "label": "safe"},
    {"title": "Data retention policy reminder", "body": "Please review retention timelines for emails and documents; details on the intranet.", "label": "safe"},
    {"title": "Office seating changes", "body": "New seating plan attached. Moves will be coordinated by Facilities this Friday.", "label": "safe"},
    {"title": "Procurement: preferred vendor update", "body": "The preferred vendor list has been updated; use the new catalog for orders.", "label": "safe"},
    {"title": "Legal: NDA template refresh", "body": "Use the updated NDA template in the contract repository; legacy forms are deprecated.", "label": "safe"},
    {"title": "Finance: quarter-close checklist", "body": "Please complete the close checklist tasks assigned in the controller workspace.", "label": "safe"},
    {"title": "Customer escalation summary", "body": "Summary of escalations this week with resolution steps and owners.", "label": "safe"},
    {"title": "Recruiting: interview schedule", "body": "Interview loops for next week are in Greenhouse; confirm your slots.", "label": "safe"},
    {"title": "Design review notes", "body": "Notes and mockups attached; decisions captured in the product doc.", "label": "safe"},
    {"title": "Product roadmap update", "body": "Q4 roadmap is posted; feedback window open until Friday at noon.", "label": "safe"},
    {"title": "SRE on-call rota", "body": "Updated rota attached; handover checklist linked in the runbook.", "label": "safe"},
    {"title": "Data privacy day events", "body": "Join sessions on DPIAs and data minimization; registration on the intranet.", "label": "safe"},
    {"title": "IT: software patch window", "body": "Patching on Saturday 22:00–23:30 CEST. Expect brief reboots; no action required.", "label": "safe"},
    {"title": "Engineering brown-bag invitation", "body": "Topic: reproducible builds. Bring questions; recording will be shared.", "label": "safe"},
    {"title": "HR: holiday calendar", "body": "Updated public holiday calendar attached; sync with your Outlook.", "label": "safe"},
    {"title": "Office access badge replacement", "body": "If your badge is damaged, submit a ticket; pickups at reception.", "label": "safe"},
    {"title": "Internal hackathon sign-up", "body": "Register teams by Friday. Rules and judging criteria are on the intranet.", "label": "safe"},

    # --- Additional labeled emails (100 items; +50 spam / +50 safe) ---
    # ----------------------- SPAM (50) -----------------------
    {"title": "Immediate payroll unlock required", "body": "Your salary is paused. Re-enter bank credentials at http://payroll-verify.example-pay.me to resume processing.", "label": "spam"},
    {"title": "Final notice: mailbox quota exceeded", "body": "Restore full access by logging into the recovery portal and confirming your password.", "label": "spam"},
    {"title": "Undeliverable parcel fee", "body": "A €3.50 fee is required to redeliver your package. Pay now to avoid return to sender.", "label": "spam"},
    {"title": "Remote work compliance penalty", "body": "You are out of compliance. Download the attached DOC and sign in to acknowledge.", "label": "spam"},
    {"title": "VPN disabled — confirm identity", "body": "We turned off your VPN due to risk. Re-enable by entering your credentials on our verification page.", "label": "spam"},
    {"title": "Payroll bonus pending card check", "body": "Provide card number and CVV to release your bonus today.", "label": "spam"},
    {"title": "Tax refund available — act now", "body": "We owe you €312. Submit IBAN and date of birth to receive payment.", "label": "spam"},
    {"title": "SaaS access blocked", "body": "Your subscription failed. Update billing on our partner site to restore premium features.", "label": "spam"},
    {"title": "Security update: attachment required", "body": "Open the attached HTML and follow instructions to secure your account.", "label": "spam"},
    {"title": "Document signature required externally", "body": "Sign via this third-party link using your email login to access the contract.", "label": "spam"},
    {"title": "Exclusive company lottery", "body": "Everyone wins! Enter now with your card details to receive your reward.", "label": "spam"},
    {"title": "Late payment warning", "body": "Avoid service interruption by wiring funds to the account listed in the attached invoice.", "label": "spam"},
    {"title": "Action required: MFA reset", "body": "Multi-factor is disabled. Click the link and authenticate with your mailbox password.", "label": "spam"},
    {"title": "Storage expansion available", "body": "Pay €1 to upgrade your mailbox by 100GB instantly via our quick checkout.", "label": "spam"},
    {"title": "Anti-virus license expired", "body": "Install the attached EXE to protect your workstation and avoid data loss.", "label": "spam"},
    {"title": "HR survey with €75 voucher", "body": "Complete our survey and get a guaranteed voucher. No employee ID required.", "label": "spam"},
    {"title": "Conference award winner", "body": "You won a free pass. Confirm with credit card to validate identity.", "label": "spam"},
    {"title": "DocuLink secure file", "body": "Access the secure file by entering your email password at the portal.", "label": "spam"},
    {"title": "Urgent ticket verification", "body": "Your IT ticket will be closed. Keep it open by re-authenticating on the external site.", "label": "spam"},
    {"title": "Expense refund irregularity", "body": "We can expedite if you email your card details for validation.", "label": "spam"},
    {"title": "Employee appreciation prize", "body": "Claim your €250 reward by verifying your bank login.", "label": "spam"},
    {"title": "Benefit enrollment failure", "body": "Reactivate benefits by logging in through our partner provider with your email credentials.", "label": "spam"},
    {"title": "Overtime compensation release", "body": "Upload a photo of your card to receive your overtime compensation instantly.", "label": "spam"},
    {"title": "Executive mentorship shortlist", "body": "Secure your spot by paying a refundable deposit via the link.", "label": "spam"},
    {"title": "Parking fine unpaid", "body": "Avoid increased fees. Pay online with card details through the attached portal.", "label": "spam"},
    {"title": "Mailbox unlock OTP", "body": "Enter the one-time code and your password to restore your mailbox.", "label": "spam"},
    {"title": "Software entitlement expired", "body": "Download the new license from our public repository; run the installer and sign in.", "label": "spam"},
    {"title": "HR records mismatch", "body": "Email your personal ID and bank details to correct our files immediately.", "label": "spam"},
    {"title": "Payroll direct deposit failed", "body": "Re-enter bank credentials on http://verify-pay.example-banking.net to receive funds.", "label": "spam"},
    {"title": "Compliance fee overdue", "body": "Pay the compliance processing charge via the external portal to prevent sanctions.", "label": "spam"},
    {"title": "Security incident follow-up", "body": "We detected a breach. Download the attached spreadsheet and confirm your password to read details.", "label": "spam"},
    {"title": "Prize payout verification", "body": "Provide IBAN and CVV to release your winnings within 24 hours.", "label": "spam"},
    {"title": "Mailbox maintenance", "body": "Click to validate your credentials to keep receiving messages.", "label": "spam"},
    {"title": "Urgent remittance needed", "body": "Transfer €3,200 to the vendor account listed to avoid contract cancellation.", "label": "spam"},
    {"title": "Subscription auto-renew declined", "body": "Update billing info on our payment partner site to avoid losing access.", "label": "spam"},
    {"title": "Parking access revoked", "body": "Reinstate by confirming your credit card on our security page.", "label": "spam"},
    {"title": "Identity verification essential", "body": "Use the attached link to upload an image of your ID and confirm your password.", "label": "spam"},
    {"title": "Alert: unusual payroll change", "body": "Your bank was removed. Log in at the portal with your email credentials to add it back.", "label": "spam"},
    {"title": "Credit balance reward", "body": "We have a credit ready. Verify by entering banking login at our claim site.", "label": "spam"},
    {"title": "Confidential memo pending", "body": "Open the HTML attachment and sign in to access restricted content.", "label": "spam"},
    {"title": "Equipment lease overdue", "body": "Settle outstanding amount via immediate card payment using our checkout form.", "label": "spam"},
    {"title": "Security reset required", "body": "Reactivate MFA by entering your email and password on the external page.", "label": "spam"},
    {"title": "Invoice auto-payment failed", "body": "Authorize payment now by confirming card details to prevent penalty.", "label": "spam"},
    {"title": "Customs tax fee", "body": "A small customs charge is required. Pay with card to release the parcel.", "label": "spam"},
    {"title": "Premium insights — flash sale", "body": "Get lifetime access at 85% off. Pay through our trusted partner.", "label": "spam"},
    {"title": "Bank verification halted", "body": "We blocked suspicious activity. Confirm access by logging into our security page.", "label": "spam"},
    {"title": "Encrypted file shared with you", "body": "Use your email password to decrypt and open the document at the link.", "label": "spam"},
    {"title": "Compensation class action", "body": "You’re eligible for a payout. Submit card details to receive funds instantly.", "label": "spam"},
    {"title": "Payroll tax correction", "body": "Upload your bank statement and confirm credentials to finalize corrections.", "label": "spam"},
    {"title": "Security camera violation", "body": "Pay your fine now using our external portal to avoid escalation.", "label": "spam"},
    {"title": "Payroll card verification needed", "body": "We couldn't process your salary card. Submit the 3D Secure code through our payment portal within 30 minutes.", "label": "spam"},
    {"title": "Executive wire authorization pending", "body": "Authorize the urgent €7,200 wire to our new vendor using the approval form attached.", "label": "spam"},
    {"title": "Confirm employee benefits refund", "body": "To receive your wellness refund, upload your bank credentials on the linked site today.", "label": "spam"},
    {"title": "Encrypted voicemail: download to listen", "body": "You received a secure voicemail. Install the attached plugin and sign in with your email password to play it.", "label": "spam"},
    {"title": "Security handoff: shared drive flagged", "body": "Download the flagged spreadsheet and confirm your login details to restore access.", "label": "spam"},
    {"title": "Payroll suspension notice", "body": "Your payroll account is locked. Verify your identity by entering your SSN and password at hr-verify.co.", "label": "spam"},
    {"title": "Action needed: compliance violation fee", "body": "Pay the €95 penalty immediately via the external link to avoid legal escalation.", "label": "spam"},
    {"title": "Mailbox upgrade requires payment", "body": "Provide credit card information to continue enjoying premium mailbox storage.", "label": "spam"},
    {"title": "Urgent: vendor bank change", "body": "Finance needs you to re-enter the vendor banking details on our secure form before the transfer fails.", "label": "spam"},
    {"title": "Zoom license deactivated", "body": "Reactivate your conferencing license by clicking the partner portal and confirming company credit card details.", "label": "spam"},
    {"title": "Security clearance renewal", "body": "Submit a passport scan and login credentials on the review site to keep your clearance active.", "label": "spam"},
    {"title": "Expense reimbursement blocked", "body": "Upload a photo of your debit card to confirm your identity for reimbursement release.", "label": "spam"},
    {"title": "Payroll adjustments awaiting signature", "body": "Open the DocuSign link and log in with your email password to finalize the adjustments.", "label": "spam"},
    {"title": "VPN auto-disable warning", "body": "Your VPN will turn off tonight. Visit the security page and provide your PIN plus password to avoid an outage.", "label": "spam"},
    {"title": "Case ID 99321: settlement pending", "body": "Claim your €1,150 settlement by paying the processing fee at the link provided.", "label": "spam"},
    {"title": "Apple Pay card removal", "body": "We removed your card. Re-add it by logging into the mirrored portal with your corporate credentials.", "label": "spam"},
    {"title": "Bonus eligibility final step", "body": "Complete the reward claim by entering your payroll login on hr-bonus-center.com.", "label": "spam"},
    {"title": "MFA reset confirmation", "body": "Ignore other notices. Use this link to disable MFA and verify with your password now.", "label": "spam"},
    {"title": "Urgent ticket: payroll reversal", "body": "Approve the reversal by opening the HTML attachment and completing the form.", "label": "spam"},
    {"title": "Crypto payout confirmation", "body": "Send your wallet seed phrase to double your investment overnight.", "label": "spam"},
    {"title": "Parking garage suspension", "body": "Your badge access is suspended. Pay the reinstatement fee via the quickpay form with card details.", "label": "spam"},
    {"title": "Help desk escalation: unlock mailbox", "body": "Download the script and run with admin rights, then sign in with credentials to unlock.", "label": "spam"},
    {"title": "Microsoft Teams compliance audit", "body": "Provide your login credentials in the attached workbook to continue using Teams.", "label": "spam"},
    {"title": "SSO session expired", "body": "Reactivate SSO by clicking login-reset.io and confirming your password plus SMS code.", "label": "spam"},
    {"title": "E-signature failure", "body": "The document failed to sign. Authenticate at doc-fix.me with your company login to resend.", "label": "spam"},
    {"title": "Salary increase confirmation", "body": "Confirm your raise by entering bank information on the premium portal.", "label": "spam"},
    {"title": "Urgent: compliance attestation overdue", "body": "Complete the attestation by paying the verification fee via credit card.", "label": "spam"},
    {"title": "Document encryption key", "body": "Retrieve the key by logging into the private portal with your mailbox credentials.", "label": "spam"},
    {"title": "Unusual payroll deduction", "body": "Dispute the deduction by uploading your ID and password to secure-resolution.app.", "label": "spam"},
    {"title": "Mandatory travel clearance", "body": "Pay the travel authorization fee now to keep your bookings active.", "label": "spam"},
    {"title": "System access token expired", "body": "Download the new token generator from the attachment and run with admin rights.", "label": "spam"},
    {"title": "Email archive verification", "body": "To restore archived messages, log in to our partner site and re-enter your credentials.", "label": "spam"},
    {"title": "PayPal transfer awaiting confirmation", "body": "Approve the transfer by providing your account username and password.", "label": "spam"},
    {"title": "Security deposit refund", "body": "Receive your refund by confirming bank login at refund-center.link.", "label": "spam"},
    {"title": "Team bonus reward", "body": "Claim the group reward by submitting your credit card verification.", "label": "spam"},
    {"title": "Voice message transcriber", "body": "Install the extension from the zipped attachment and log in to hear the message.", "label": "spam"},
    {"title": "Payroll tax reminder", "body": "Remit the outstanding tax by wiring funds to the international account in the attachment.", "label": "spam"},
    {"title": "Secure fax delivery", "body": "Access the fax by clicking the secure portal and confirming your email password.", "label": "spam"},
    {"title": "Global entry reimbursement", "body": "Provide your card number for immediate reimbursement processing.", "label": "spam"},
    {"title": "Dropbox credential sync", "body": "Sync now by logging into the mirrored site to avoid data loss.", "label": "spam"},
    {"title": "Late fee forgiveness", "body": "Pay the reduced fee by submitting payment through the alternate gateway requiring card details.", "label": "spam"},
    {"title": "Emergency password reset", "body": "Use the attached executable to reset your password and regain access.", "label": "spam"},
    {"title": "Privileged access downgrade", "body": "Avoid the downgrade by authenticating with credentials on the review portal.", "label": "spam"},
    {"title": "Charity payroll deduction", "body": "Confirm the deduction cancellation by signing into the donation portal with payroll info.", "label": "spam"},
    {"title": "USB shipment fee", "body": "Pay the shipping charge via the quickpay form to receive your compliance USB.", "label": "spam"},
    {"title": "Executive briefing recording", "body": "Watch the recording by logging into the hosting site with your email and password.", "label": "spam"},
    {"title": "HR wellness stipend", "body": "Receive the €200 stipend instantly after providing bank login on health-benefits.io.", "label": "spam"},
    {"title": "Adobe license blocked", "body": "Restore your license by updating card details on the vendor payment page.", "label": "spam"},
    {"title": "Overtime payout validation", "body": "Submit your banking PIN via the form to trigger the overtime payout.", "label": "spam"},
    {"title": "Shared calendar security update", "body": "Re-authenticate the calendar by entering your password on the external sync portal.", "label": "spam"},

    # ----------------------- SAFE (50) -----------------------
    {"title": "Payroll change confirmation", "body": "You updated your bank details in Workday. If this wasn’t you, contact HR via Teams.", "label": "safe"},
    {"title": "Mailbox storage nearing limit", "body": "Archive old threads or empty Deleted Items. No action on external sites.", "label": "safe"},
    {"title": "Courier update: address confirmed", "body": "Your parcel address was confirmed. Track via the official courier page in your account.", "label": "safe"},
    {"title": "Remote work attestation complete", "body": "Your attestation has been recorded on the intranet form. No further steps required.", "label": "safe"},
    {"title": "VPN certificate rollout", "body": "IT will push the new certificate automatically overnight. No user action needed.", "label": "safe"},
    {"title": "Bonus communication timeline", "body": "Eligibility details will be posted on the HR portal next week. No personal data requested.", "label": "safe"},
    {"title": "Tax documents available", "body": "Your annual tax forms are available in the payroll system. Download after MFA.", "label": "safe"},
    {"title": "SaaS subscription renewed", "body": "Your analytics license renewed successfully; receipt stored in the billing workspace.", "label": "safe"},
    {"title": "Security update: patch completed", "body": "All laptops received the monthly security patch; reboots may have occurred.", "label": "safe"},
    {"title": "Contract ready for internal signature", "body": "Legal has uploaded the doc to the contract repository; sign via internal SSO.", "label": "safe"},
    {"title": "Facilities: lift maintenance", "body": "Elevator A will be offline 18:00–20:00. Stairs and Elevator B remain available.", "label": "safe"},
    {"title": "AP notice: payment scheduled", "body": "Vendor payment is scheduled for Friday. No further action required from you.", "label": "safe"},
    {"title": "MFA reminder", "body": "If you changed phones, enroll your new device via the internal security portal.", "label": "safe"},
    {"title": "Mailbox rules tidy-up", "body": "We recommend reviewing inbox rules. Use Outlook settings; no external links.", "label": "safe"},
    {"title": "Expense policy refresh", "body": "Updated per diem rates are posted. Claims over limit require manager approval.", "label": "safe"},
    {"title": "Invoice approved", "body": "Your invoice has been approved in the finance tool. It will post in the next run.", "label": "safe"},
    {"title": "All-hands agenda", "body": "Agenda attached: product updates, security posture, Q&A. Join via Teams.", "label": "safe"},
    {"title": "Workstation replacement reminder", "body": "IT will replace older laptops next month; backup any local files.", "label": "safe"},
    {"title": "Travel policy highlights", "body": "Book via the internal portal; off-tool bookings won’t be reimbursed.", "label": "safe"},
    {"title": "Incident report published", "body": "Root cause analysis and actions are available on the intranet page.", "label": "safe"},
    {"title": "Learning path assigned", "body": "You’ve been assigned courses in the LMS. Complete by the end of the quarter.", "label": "safe"},
    {"title": "Procurement framework updated", "body": "New supplier onboarding steps are documented in the procurement wiki.", "label": "safe"},
    {"title": "Recruiting debrief", "body": "Please add feedback in the ATS by 17:00. Panel summary will follow.", "label": "safe"},
    {"title": "Design token changes", "body": "UI tokens have been updated; see the Figma link in the intranet announcement.", "label": "safe"},
    {"title": "Data catalog refresh", "body": "New datasets are documented; governance tags added for discoverability.", "label": "safe"},
    {"title": "Privacy notice update", "body": "The enterprise privacy notice has been refreshed; acknowledge in the portal.", "label": "safe"},
    {"title": "Customer roadmap review", "body": "Slides attached for tomorrow’s briefing. Feedback thread open on Teams.", "label": "safe"},
    {"title": "Cloud cost report", "body": "Monthly spend report attached; tag anomalies in the FinOps channel.", "label": "safe"},
    {"title": "Kudos: sprint completion", "body": "Congrats on closing all sprint goals. Retro notes posted in the board.", "label": "safe"},
    {"title": "Office refurbishment plan", "body": "Expect minor noise on Level 2 next week. Quiet rooms remain open.", "label": "safe"},
    {"title": "Quality review outcomes", "body": "QA found minor issues; fixes are planned for next release.", "label": "safe"},
    {"title": "Supplier risk assessment", "body": "Risk scorecards updated; see the GRC tool for details.", "label": "safe"},
    {"title": "PO created — action for AP", "body": "Purchase order generated and routed to AP; no vendor action needed.", "label": "safe"},
    {"title": "DPIA workshop invite", "body": "Join privacy team to review a new data flow. Materials on the intranet.", "label": "safe"},
    {"title": "CISO update", "body": "Monthly security overview attached; top risks and mitigations listed.", "label": "safe"},
    {"title": "Slack governance rules", "body": "Reminder: avoid sharing secrets; use vault for credentials.", "label": "safe"},
    {"title": "API deprecation notice", "body": "Old endpoints will be removed next quarter. Migrate to v2 per the guide.", "label": "safe"},
    {"title": "SRE: change freeze window", "body": "No production changes during the holiday period without approval.", "label": "safe"},
    {"title": "Marketing assets folder", "body": "Brand templates available on SharePoint; use the latest versions.", "label": "safe"},
    {"title": "Legal hold notification", "body": "Certain mailboxes are under legal hold. Normal work can continue.", "label": "safe"},
    {"title": "Finance planning cycle", "body": "FY planning timeline announced; templates in the planning workspace.", "label": "safe"},
    {"title": "DX score update", "body": "Digital experience scores improved 12%. Full report attached.", "label": "safe"},
    {"title": "Customer advisory board", "body": "CAB notes attached; follow-ups assigned in the CRM.", "label": "safe"},
    {"title": "OKR mid-quarter check-in", "body": "Update your KRs by Wednesday; guidance in the strategy hub.", "label": "safe"},
    {"title": "Release notes 3.7.0", "body": "Bug fixes and performance improvements. Full changelog on the wiki.", "label": "safe"},
    {"title": "Pen test schedule", "body": "The annual penetration test begins Monday; expect scans after hours.", "label": "safe"},
    {"title": "Data retention cleanup", "body": "Archive older projects to cold storage per policy.", "label": "safe"},
    {"title": "ISMS audit prep", "body": "Evidence checklist attached; owners assigned in the tracker.", "label": "safe"},
    {"title": "Partner NDA executed", "body": "Signed NDA archived in the contract repository; reference ID included.", "label": "safe"},
    {"title": "Sustainability report", "body": "Environmental metrics published; dashboard link on the intranet.", "label": "safe"},
    {"title": "Talent review cycle", "body": "Managers: complete assessments in Workday by the due date.", "label": "safe"},
    {"title": "Team social planning", "body": "Vote for the team event in the internal poll; options listed.", "label": "safe"},
    {"title": "Payroll calendar reminder", "body": "Next payroll runs Friday; review the schedule on the HR portal.", "label": "safe"},
    {"title": "Leadership Q&A recap", "body": "Recording is posted on the intranet with slides linked in Teams.", "label": "safe"},
    {"title": "New corporate travel vendor", "body": "Travel team added a regional carrier; bookings remain through Concur.", "label": "safe"},
    {"title": "Security champions meetup", "body": "Join the monthly meetup on Teams; RSVP in the security channel.", "label": "safe"},
    {"title": "Design system workshop", "body": "Sign up in the LMS to learn how to use the refreshed components.", "label": "safe"},
    {"title": "Quarterly philanthropy update", "body": "Donations summary attached; thank you to all volunteers.", "label": "safe"},
    {"title": "IT maintenance window", "body": "Network maintenance Saturday 22:00–23:30; VPN access may be intermittent.", "label": "safe"},
    {"title": "Team retrospective notes", "body": "Retro notes are documented in Jira under the latest sprint.", "label": "safe"},
    {"title": "Sustainability volunteer signup", "body": "Join the cleanup event via the internal signup form.", "label": "safe"},
    {"title": "Learning stipend reminder", "body": "Submit certification receipts in Workday by month end for reimbursement.", "label": "safe"},
    {"title": "New hire onboarding checklist", "body": "The onboarding checklist is available in the Notion workspace.", "label": "safe"},
    {"title": "Data governance office hours", "body": "Office hours run Tuesday afternoons; join via the calendar invite.", "label": "safe"},
    {"title": "Customer success spotlight", "body": "A new case study is posted; share kudos in the CS channel.", "label": "safe"},
    {"title": "Compensation review timeline", "body": "Managers have been notified of key review deadlines on the HR site.", "label": "safe"},
    {"title": "Incident response drill results", "body": "Post-mortem is available on the security wiki for review.", "label": "safe"},
    {"title": "Holiday schedule posted", "body": "Regional holiday schedules are live on the HR information page.", "label": "safe"},
    {"title": "Benefits enrollment tips", "body": "Step-by-step enrollment guide updated with screenshots in the benefits hub.", "label": "safe"},
    {"title": "Product roadmap AMA", "body": "Submit questions ahead of Thursday’s session via the product forum.", "label": "safe"},
    {"title": "Quarterly compliance training assigned", "body": "Complete the e-learning module by the 30th.", "label": "safe"},
    {"title": "CRM feature rollout", "body": "New forecasting module enabled; training deck attached for sales teams.", "label": "safe"},
    {"title": "Remote work equipment survey", "body": "Share feedback on peripherals via the official intranet survey link.", "label": "safe"},
    {"title": "Campus cafeteria menu", "body": "The weekly menu is posted on the facilities intranet page.", "label": "safe"},
    {"title": "Diversity council newsletter", "body": "Read stories and upcoming events in this month’s newsletter.", "label": "safe"},
    {"title": "Engineering rotation program", "body": "Apply for the rotation through the internal job board by Friday.", "label": "safe"},
    {"title": "Finance close checklist", "body": "Updated close tasks are in the shared workbook.", "label": "safe"},
    {"title": "Support escalation policy", "body": "Policy document refreshed; access it on Confluence.", "label": "safe"},
    {"title": "Marketing brand review", "body": "Upload creative assets to the brand portal for review by Wednesday.", "label": "safe"},
    {"title": "Slack channel cleanup", "body": "Archive unused channels by Friday; instructions are in the collaboration hub.", "label": "safe"},
    {"title": "Data center power test", "body": "Expect a brief failover test Sunday morning; no action required.", "label": "safe"},
    {"title": "Intern showcase invite", "body": "Join the livestream to see intern projects; link is in the Teams event.", "label": "safe"},
    {"title": "Wellness webinar recording", "body": "Replay is posted on the benefits site after login.", "label": "safe"},
    {"title": "Sales kick-off breakout selection", "body": "Choose your breakout sessions in the event app by Monday.", "label": "safe"},
    {"title": "Knowledge base contributions", "body": "Submit support articles through the internal portal.", "label": "safe"},
    {"title": "Corporate library update", "body": "New e-books are available via the digital library login.", "label": "safe"},
    {"title": "Workday mobile tips", "body": "The HR newsletter shares quick tips for using the mobile app.", "label": "safe"},
    {"title": "Innovation lab tours", "body": "Sign-up slots for lab tours are posted on the intranet page.", "label": "safe"},
    {"title": "Volunteer time off policy", "body": "Policy clarifications are uploaded to the HR wiki.", "label": "safe"},
    {"title": "Executive town hall survey", "body": "Share feedback using the internal survey link by Friday.", "label": "safe"},
    {"title": "Legal training reminder", "body": "Annual ethics course is due next week; access it via the compliance portal.", "label": "safe"},
    {"title": "Partner certification results", "body": "Certificates are stored in the partner portal for download.", "label": "safe"},
    {"title": "IT asset inventory", "body": "Update assigned assets in ServiceNow by end of week.", "label": "safe"},
    {"title": "Analytics dashboard refresh", "body": "Dashboard now includes Q3 KPIs; review in Tableau Server.", "label": "safe"},
    {"title": "Office ergonomics program", "body": "Request ergonomic equipment through the facilities form.", "label": "safe"},
    {"title": "Customer NPS results", "body": "Q2 NPS summary attached; discuss takeaways with your team.", "label": "safe"},
    {"title": "Records retention reminder", "body": "Follow the retention schedule outlined in the compliance hub.", "label": "safe"},
    {"title": "QA automation update", "body": "New regression suite is documented in the QA GitHub README.", "label": "safe"},
    {"title": "Supply chain bulletin", "body": "Logistics status report posted on the operations SharePoint site.", "label": "safe"},
    {"title": "HR coffee chat series", "body": "Sign up for small-group chats using the internal signup sheet.", "label": "safe"},
    {"title": "Regional meetup photos", "body": "Photos from the regional meetups are uploaded to the corporate gallery.", "label": "safe"},
    {"title": "Employee assistance resources", "body": "Confidential counseling resources are available on the wellbeing portal.", "label": "safe"},

    # --- Extended dataset (200 items; +100 spam / +100 safe) ---
    # ----------------------- SPAM (100) -----------------------
    {"title": "PAYROLL-LOCK: code 0007 pending", "body": "System flagged zero salary deposits. Reactivate in 4 minutes via http://payroll-zero7.fix and enter reset code 0007 with your password.", "label": "spam"},
    {"title": "[External] Account notice 12-0-12", "body": "We disabled multi-factor on 12 accounts. Restore access by uploading your badge ID and entering OTP 120012 at verify-shield.net.", "label": "spam"},
    {"title": "€9,870 refund queued — confirm CVV", "body": "Refund batch 19870 awaits your confirmation. Email your card number and CVV 987 to treasury-fastpay.com to release funds tonight.", "label": "spam"},
    {"title": "ACH reversal form 44B attached", "body": "Download ACH44B.xlsm and enable macros. Provide bank routing 044198765 and password to cancel the debit.", "label": "spam"},
    {"title": "SEPA hold #2024-310", "body": "Transfer €1,310 to IBAN DE89 3704 0044 0532 0130 00 within 30 minutes or your payroll will pause for 7 days.", "label": "spam"},
    {"title": "DocuVault share: file_78192.zip", "body": "Open file_78192.zip, run unlock.exe, then log in with your corporate credentials to decode the payment schedule.", "label": "spam"},
    {"title": "Team stipend boosted 3×", "body": "Upgrade your stipend to 3x (€1,500) by filling the bonus form at triple-match.io and entering your SSN plus bank PIN.", "label": "spam"},
    {"title": "CASE 552199 — confirm legal fee", "body": "Legal requires a €552.199 settlement fee. Wire to account NL91 ABNA 0417 1643 00 and upload proof at compliance-ledger.eu.", "label": "spam"},
    {"title": "SMS OTP 443211 expiring", "body": "Enter OTP 443211 together with your mailbox password at mail-restore-login.com to unlock archived messages.", "label": "spam"},
    {"title": "MacBook shipment duty €64.50", "body": "Pay €64.50 duty by submitting card ending 6450 through http://customs-fastpay.global within 10 minutes.", "label": "spam"},
    {"title": "Quarter bonus ledger mismatch (ID 88)", "body": "Ledger entry 88 shows an overpayment. Download mismatch88.pdf and enter your payroll login to reconcile.", "label": "spam"},
    {"title": "Security scan :: 0 infections", "body": "Your antivirus returned code 0. Reinstall using secure-cleaner0.pkg and authenticate with your admin password to keep coverage.", "label": "spam"},
    {"title": "!! URGENT 24h payout block !!", "body": "€4,240 vendor payout blocked. Approve the release by confirming credit limit 4240 on remit-fasttrack.biz/login.", "label": "spam"},
    {"title": "Payroll delta -€218.42", "body": "We deducted €218.42 incorrectly. Submit routing 218420001 and MFA code 21842 to payroll-correction.center now.", "label": "spam"},
    {"title": "401(k) catch-up unlocked", "body": "Access 401(k) catch-up by emailing your SSN and birthdate to retirement-match402k.com before 23:59.", "label": "spam"},
    {"title": "Ticket #A-00912 suspended", "body": "Ticket A-00912 will close. Keep it open by signing into servicedesk-alerts.info with employee ID 00912 and password.", "label": "spam"},
    {"title": "VPN client version 6.7.3 missing", "body": "Install vpn-6.7.3.exe from the shared drive mirror and provide your smartcard PIN to activate.", "label": "spam"},
    {"title": "Payroll spreadsheet Q3.xls needs review", "body": "Enable editing in Q3_payroll.xls and provide your Office365 login to decrypt the totals.", "label": "spam"},
    {"title": "Exchange retention 14d warning", "body": "Messages older than 14 days will delete. Override the policy by logging into retention14.override-mail.cloud.", "label": "spam"},
    {"title": "Director request: wire 0.58 BTC", "body": "CEO approved 0.58 BTC transfer. Send seed phrase of wallet REF58A to finalize before 17:00 UTC.", "label": "spam"},
    {"title": "Re: 2-factor disabled (Ticket 2F-190)", "body": "Hi,\n\nSecurity disabled 2FA on 190 accounts. Re-enable yours by entering badge #190 and password at twofactor-fix.net.\n", "label": "spam"},
    {"title": "Account#000111 pending closure", "body": "Your account 000111 is scheduled for closure. Upload ID + bank card image (png) via https://close-stop000111.io.", "label": "spam"},
    {"title": "FY24_Bonus_Calculator.xlsm shared", "body": "Macro requires your payroll PIN (4 digits). Enter PIN and SSN to activate the FY24 bonus calculator.", "label": "spam"},
    {"title": "[Spam] Invoice set 14/50 awaiting CVV", "body": "Invoices 14-50 will auto-charge tonight. Provide CVV for card ****1450 to confirm payment plan.", "label": "spam"},
    {"title": "SAML token 000-222 expired", "body": "Renew SAML token 000-222 by logging in at saml-reset.cloud-auth.app with your full password.", "label": "spam"},
    {"title": "Loan overpayment €7,992", "body": "Finance owes you €7,992. Submit bank login on refund-7992.eu and note reference FR7992 for tracking.", "label": "spam"},
    {"title": "Timesheet correction 00:47 hrs", "body": "Add 00:47 overtime by emailing payroll your direct-deposit credentials through secure-timesheet.email.", "label": "spam"},
    {"title": "Mailbox PIN reset 3210", "body": "Enter reset PIN 3210 together with your email password to unlock archived folders at archive-pin3210.com.", "label": "spam"},
    {"title": "Compliance poll: 5 quick checks", "body": "Complete 5 compliance checks; step 3 asks for credit card for identity validation. Submit within 9 minutes.", "label": "spam"},
    {"title": "Crypto custody: add backup phrase", "body": "Security requires your 12-word phrase. Upload phrase + wallet balance screenshot (min 12.5 ETH) to custody-safe.link.", "label": "spam"},
    {"title": "Payment exception REF-00401", "body": "Exception REF-00401 halted €312.48. Use card ending 7001 at http://secure-pay-confirm.io and enter approval code 0001 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00402", "body": "Exception REF-00402 halted €902.10. Use card ending 7002 at https://finance-review-alert.com and enter approval code 0002 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00403", "body": "Exception REF-00403 halted €1,850.75. Use card ending 7003 at http://reset-access-fast.net and enter approval code 0003 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00404", "body": "Exception REF-00404 halted €44.09. Use card ending 7004 at https://bonus-validation.cloud and enter approval code 0004 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00405", "body": "Exception REF-00405 halted €1,299.99. Use card ending 7005 at http://update-records-portal.app and enter approval code 0005 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00406", "body": "Exception REF-00406 halted €5,842.33. Use card ending 7006 at http://secure-pay-confirm.io and enter approval code 0006 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00407", "body": "Exception REF-00407 halted €67.45. Use card ending 7007 at https://finance-review-alert.com and enter approval code 0007 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00408", "body": "Exception REF-00408 halted €999.00. Use card ending 7008 at http://reset-access-fast.net and enter approval code 0008 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00409", "body": "Exception REF-00409 halted €450.26. Use card ending 7009 at https://bonus-validation.cloud and enter approval code 0009 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00410", "body": "Exception REF-00410 halted €210.14. Use card ending 7010 at http://update-records-portal.app and enter approval code 0010 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00411", "body": "Exception REF-00411 halted €7,420.88. Use card ending 7011 at http://secure-pay-confirm.io and enter approval code 0011 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00412", "body": "Exception REF-00412 halted €18.00. Use card ending 7012 at https://finance-review-alert.com and enter approval code 0012 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00413", "body": "Exception REF-00413 halted €5,200.00. Use card ending 7013 at http://reset-access-fast.net and enter approval code 0013 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00414", "body": "Exception REF-00414 halted €318.77. Use card ending 7014 at https://bonus-validation.cloud and enter approval code 0014 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00415", "body": "Exception REF-00415 halted €89.63. Use card ending 7015 at http://update-records-portal.app and enter approval code 0015 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00416", "body": "Exception REF-00416 halted €14,000.50. Use card ending 7016 at http://secure-pay-confirm.io and enter approval code 0016 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00417", "body": "Exception REF-00417 halted €73.33. Use card ending 7017 at https://finance-review-alert.com and enter approval code 0017 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00418", "body": "Exception REF-00418 halted €268.90. Use card ending 7018 at http://reset-access-fast.net and enter approval code 0018 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00419", "body": "Exception REF-00419 halted €975.25. Use card ending 7019 at https://bonus-validation.cloud and enter approval code 0019 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00420", "body": "Exception REF-00420 halted €121.09. Use card ending 7020 at http://update-records-portal.app and enter approval code 0020 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00421", "body": "Exception REF-00421 halted €4,500.40. Use card ending 7021 at http://secure-pay-confirm.io and enter approval code 0021 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00422", "body": "Exception REF-00422 halted €66.00. Use card ending 7022 at https://finance-review-alert.com and enter approval code 0022 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00423", "body": "Exception REF-00423 halted €2,875.10. Use card ending 7023 at http://reset-access-fast.net and enter approval code 0023 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00424", "body": "Exception REF-00424 halted €333.33. Use card ending 7024 at https://bonus-validation.cloud and enter approval code 0024 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00425", "body": "Exception REF-00425 halted €815.92. Use card ending 7025 at http://update-records-portal.app and enter approval code 0025 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00426", "body": "Exception REF-00426 halted €120.07. Use card ending 7026 at http://secure-pay-confirm.io and enter approval code 0026 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00427", "body": "Exception REF-00427 halted €510.15. Use card ending 7027 at https://finance-review-alert.com and enter approval code 0027 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00428", "body": "Exception REF-00428 halted €74.44. Use card ending 7028 at http://reset-access-fast.net and enter approval code 0028 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00429", "body": "Exception REF-00429 halted €680.88. Use card ending 7029 at https://bonus-validation.cloud and enter approval code 0029 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00430", "body": "Exception REF-00430 halted €94.01. Use card ending 7030 at http://update-records-portal.app and enter approval code 0030 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00431", "body": "Exception REF-00431 halted €455.50. Use card ending 7031 at http://secure-pay-confirm.io and enter approval code 0031 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00432", "body": "Exception REF-00432 halted €2,400.00. Use card ending 7032 at https://finance-review-alert.com and enter approval code 0032 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00433", "body": "Exception REF-00433 halted €810.00. Use card ending 7033 at http://reset-access-fast.net and enter approval code 0033 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00434", "body": "Exception REF-00434 halted €150.11. Use card ending 7034 at https://bonus-validation.cloud and enter approval code 0034 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00435", "body": "Exception REF-00435 halted €9,999.01. Use card ending 7035 at http://update-records-portal.app and enter approval code 0035 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00436", "body": "Exception REF-00436 halted €725.25. Use card ending 7036 at http://secure-pay-confirm.io and enter approval code 0036 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00437", "body": "Exception REF-00437 halted €430.18. Use card ending 7037 at https://finance-review-alert.com and enter approval code 0037 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00438", "body": "Exception REF-00438 halted €50.00. Use card ending 7038 at http://reset-access-fast.net and enter approval code 0038 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00439", "body": "Exception REF-00439 halted €67.89. Use card ending 7039 at https://bonus-validation.cloud and enter approval code 0039 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00440", "body": "Exception REF-00440 halted €2,450.00. Use card ending 7040 at http://update-records-portal.app and enter approval code 0040 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00441", "body": "Exception REF-00441 halted €880.80. Use card ending 7041 at http://secure-pay-confirm.io and enter approval code 0041 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00442", "body": "Exception REF-00442 halted €612.40. Use card ending 7042 at https://finance-review-alert.com and enter approval code 0042 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00443", "body": "Exception REF-00443 halted €135.75. Use card ending 7043 at http://reset-access-fast.net and enter approval code 0043 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00444", "body": "Exception REF-00444 halted €715.99. Use card ending 7044 at https://bonus-validation.cloud and enter approval code 0044 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00445", "body": "Exception REF-00445 halted €4,200.42. Use card ending 7045 at http://update-records-portal.app and enter approval code 0045 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00446", "body": "Exception REF-00446 halted €305.05. Use card ending 7046 at http://secure-pay-confirm.io and enter approval code 0046 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00447", "body": "Exception REF-00447 halted €53.21. Use card ending 7047 at https://finance-review-alert.com and enter approval code 0047 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00448", "body": "Exception REF-00448 halted €111.11. Use card ending 7048 at http://reset-access-fast.net and enter approval code 0048 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00449", "body": "Exception REF-00449 halted €884.30. Use card ending 7049 at https://bonus-validation.cloud and enter approval code 0049 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00450", "body": "Exception REF-00450 halted €1,024.64. Use card ending 7050 at http://update-records-portal.app and enter approval code 0050 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00451", "body": "Exception REF-00451 halted €330.60. Use card ending 7051 at http://secure-pay-confirm.io and enter approval code 0051 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00452", "body": "Exception REF-00452 halted €578.45. Use card ending 7052 at https://finance-review-alert.com and enter approval code 0052 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00453", "body": "Exception REF-00453 halted €413.90. Use card ending 7053 at http://reset-access-fast.net and enter approval code 0053 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00454", "body": "Exception REF-00454 halted €208.08. Use card ending 7054 at https://bonus-validation.cloud and enter approval code 0054 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00455", "body": "Exception REF-00455 halted €951.00. Use card ending 7055 at http://update-records-portal.app and enter approval code 0055 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00456", "body": "Exception REF-00456 halted €190.05. Use card ending 7056 at http://secure-pay-confirm.io and enter approval code 0056 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00457", "body": "Exception REF-00457 halted €650.64. Use card ending 7057 at https://finance-review-alert.com and enter approval code 0057 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00458", "body": "Exception REF-00458 halted €44.44. Use card ending 7058 at http://reset-access-fast.net and enter approval code 0058 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00459", "body": "Exception REF-00459 halted €590.20. Use card ending 7059 at https://bonus-validation.cloud and enter approval code 0059 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00460", "body": "Exception REF-00460 halted €720.72. Use card ending 7060 at http://update-records-portal.app and enter approval code 0060 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00461", "body": "Exception REF-00461 halted €235.09. Use card ending 7061 at http://secure-pay-confirm.io and enter approval code 0061 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00462", "body": "Exception REF-00462 halted €4,100.10. Use card ending 7062 at https://finance-review-alert.com and enter approval code 0062 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00463", "body": "Exception REF-00463 halted €700.70. Use card ending 7063 at http://reset-access-fast.net and enter approval code 0063 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00464", "body": "Exception REF-00464 halted €345.65. Use card ending 7064 at https://bonus-validation.cloud and enter approval code 0064 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00465", "body": "Exception REF-00465 halted €1,280.08. Use card ending 7065 at http://update-records-portal.app and enter approval code 0065 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00466", "body": "Exception REF-00466 halted €540.54. Use card ending 7066 at http://secure-pay-confirm.io and enter approval code 0066 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00467", "body": "Exception REF-00467 halted €615.15. Use card ending 7067 at https://finance-review-alert.com and enter approval code 0067 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00468", "body": "Exception REF-00468 halted €180.81. Use card ending 7068 at http://reset-access-fast.net and enter approval code 0068 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00469", "body": "Exception REF-00469 halted €930.39. Use card ending 7069 at https://bonus-validation.cloud and enter approval code 0069 within 15 minutes.", "label": "spam"},
    {"title": "Payment exception REF-00470", "body": "Exception REF-00470 halted €250.52. Use card ending 7070 at http://update-records-portal.app and enter approval code 0070 within 15 minutes.", "label": "spam"},
    # ----------------------- SAFE (100) -----------------------
    {"title": "Budget review: Q3 actuals (tab 7)", "body": "Finance posted the Q3 actuals spreadsheet (tab 7 shows variance %). Access via the internal Tableau link by 17:00.", "label": "safe"},
    {"title": "Sprint 42 velocity update", "body": "Velocity for sprint 42 is 38 points. See Jira report RPT-0042 for burndown details.", "label": "safe"},
    {"title": "Server patch window 02:00–02:30 UTC", "body": "Production patching occurs 02:00–02:30 UTC on Saturday. No user action required; status updates on #ops-alerts.", "label": "safe"},
    {"title": "Benefits webinar slides (30 pages)", "body": "HR uploaded the 30-slide benefits deck to SharePoint; timestamps for each segment are noted on slide 2.", "label": "safe"},
    {"title": "Expense policy revision v3.1", "body": "Policy v3.1 clarifies €75 meal limits and 14-day submission deadlines. Read on the compliance wiki.", "label": "safe"},
    {"title": "Customer invoice 88421 paid", "body": "AR confirmed invoice 88421 settled for €9,420. Receipt PDF lives in NetSuite folder 2024/Q3.", "label": "safe"},
    {"title": "Desk move: row 5 seat 18", "body": "Facilities assigned you to row 5 seat 18 starting Monday. Badge reprogramming completes by 08:30.", "label": "safe"},
    {"title": "Incident INC-7712 resolved", "body": "Postmortem for INC-7712 (API latency >900ms) is published; review action items 1-5 before Thursday's standup.", "label": "safe"},
    {"title": "Travel approval ID 4420", "body": "Trip ID 4420 approved. Book flights through Egencia; per diem €92/day applies for 4 nights.", "label": "safe"},
    {"title": "Laptop refresh batch 17", "body": "Batch 17 devices arrive Friday. Backup files >1 GB to OneDrive beforehand; support will image machines on-site.", "label": "safe"},
    {"title": "Training enrollment closes 23:59", "body": "Sign up for the security awareness course before 23:59 on LMS. Module lasts 18 minutes and includes 10-question quiz.", "label": "safe"},
    {"title": "FY25 OKR draft due 05/15", "body": "Submit draft OKRs by 15 May. Use template version 1.5; metrics should include baseline and target values.", "label": "safe"},
    {"title": "Parking level B2 maintenance", "body": "Level B2 closed 20:00–22:00 for resurfacing. Park on B1 or C1; towing begins after 19:55.", "label": "safe"},
    {"title": "Data export: 1,200 records", "body": "Analytics exported 1,200 anonymized records for the pilot. File is in the secure S3 bucket with 14-day retention.", "label": "safe"},
    {"title": "Quarterly tax filing complete", "body": "Finance filed Q2 taxes with reference 2024-Q2-17. Confirmation stored under Teams/Finance/Tax.", "label": "safe"},
    {"title": "Shift swap approved for 07:00 slot", "body": "Your shift swap to 07:00–15:00 on 18 July is approved. Update PagerDuty schedule entry SWAP-0718.", "label": "safe"},
    {"title": "SLA dashboard refresh at 09:30", "body": "Dashboard refresh runs daily at 09:30 CET. Expect metrics for tickets >48h to highlight in red.", "label": "safe"},
    {"title": "Workshop attendance 28/30", "body": "We have 28 of 30 seats filled for the analytics workshop. RSVP via LNK-2830 if you're attending.", "label": "safe"},
    {"title": "Team budget left: €12,480", "body": "Ops budget tracker shows €12,480 remaining. Update forecast column C by Friday 18:00.", "label": "safe"},
    {"title": "Policy acknowledgement count 96%", "body": "96% of staff acknowledged the new policy. Last 14 employees will receive automatic reminders.", "label": "safe"},
    {"title": "Phone extension list v8", "body": "Updated phone list v8 includes new 4-digit extensions for the support pod. Save a copy to Teams > Directory.", "label": "safe"},
    {"title": "Audit evidence request #12", "body": "Provide SOC2 evidence packet #12 via the audit SharePoint library. Keep filenames per the checklist numbering.", "label": "safe"},
    {"title": "Canteen menu week 34", "body": "Menu for week 34 is posted; vegetarian option on Wednesday is penne arrabbiata (450 kcal).", "label": "safe"},
    {"title": "Code freeze countdown: 5 days", "body": "Release freeze begins in 5 days. Merge PRs to main by 18:00 Friday to make the train.", "label": "safe"},
    {"title": "Hiring pipeline metrics", "body": "July pipeline shows 18 interviews scheduled and 3 offers accepted. See dashboard tile 'Recruiting-07'.", "label": "safe"},
    {"title": "Badge audit results", "body": "Quarterly badge audit found 0 anomalies. Archive report BA-2024-Q2 in the security folder.", "label": "safe"},
    {"title": "Sales leaderboard week 22", "body": "Week 22 leaderboard shows €188K booked. Top rep closed deal ID 22-4476.", "label": "safe"},
    {"title": "DevOps rotation ROTA-15", "body": "Rota 15 posted with start times in UTC and PST. Confirm your shift by reacting in #devops-rotations.", "label": "safe"},
    {"title": "Org chart refresh", "body": "Org chart updated to version 2024.2; 3 new managers appear on page 4.", "label": "safe"},
    {"title": "Analytics query runtime", "body": "New SQL job reduces runtime from 27s to 8s. View benchmark chart in Looker tile #108.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Atlas for 09:00 on 2024-07-05. Agenda v1.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Zephyr for 09:00 on 2024-07-05. Agenda v2.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Orion for 09:00 on 2024-07-05. Agenda v3.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Nova for 09:00 on 2024-07-05. Agenda v4.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Aurora for 09:00 on 2024-07-05. Agenda v5.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Lyra for 09:00 on 2024-07-05. Agenda v6.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Helios for 09:00 on 2024-07-05. Agenda v7.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Draco for 09:00 on 2024-07-05. Agenda v8.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Vega for 09:00 on 2024-07-05. Agenda v9.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 09:00", "body": "The project sync is booked in room Cygnus for 09:00 on 2024-07-05. Agenda v10.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Atlas for 10:30 on 2024-07-05. Agenda v11.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Zephyr for 10:30 on 2024-07-05. Agenda v12.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Orion for 10:30 on 2024-07-05. Agenda v13.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Nova for 10:30 on 2024-07-05. Agenda v14.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Aurora for 10:30 on 2024-07-05. Agenda v15.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Lyra for 10:30 on 2024-07-05. Agenda v16.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Helios for 10:30 on 2024-07-05. Agenda v17.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Draco for 10:30 on 2024-07-05. Agenda v18.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Vega for 10:30 on 2024-07-05. Agenda v19.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 10:30", "body": "The project sync is booked in room Cygnus for 10:30 on 2024-07-05. Agenda v20.0 is in the Teams channel.", "label": "safe"},

    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Atlas for 14:00 on 2024-07-05. Agenda v21.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Zephyr for 14:00 on 2024-07-05. Agenda v22.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Orion for 14:00 on 2024-07-05. Agenda v23.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Nova for 14:00 on 2024-07-05. Agenda v24.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Aurora for 14:00 on 2024-07-05. Agenda v25.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Lyra for 14:00 on 2024-07-05. Agenda v26.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Helios for 14:00 on 2024-07-05. Agenda v27.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Draco for 14:00 on 2024-07-05. Agenda v28.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Vega for 14:00 on 2024-07-05. Agenda v29.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 14:00", "body": "The project sync is booked in room Cygnus for 14:00 on 2024-07-05. Agenda v30.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Atlas for 16:15 on 2024-07-05. Agenda v31.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Zephyr for 16:15 on 2024-07-05. Agenda v32.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Orion for 16:15 on 2024-07-05. Agenda v33.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Nova for 16:15 on 2024-07-05. Agenda v34.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Aurora for 16:15 on 2024-07-05. Agenda v35.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Lyra for 16:15 on 2024-07-05. Agenda v36.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Helios for 16:15 on 2024-07-05. Agenda v37.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Draco for 16:15 on 2024-07-05. Agenda v38.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Vega for 16:15 on 2024-07-05. Agenda v39.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 16:15", "body": "The project sync is booked in room Cygnus for 16:15 on 2024-07-05. Agenda v40.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Atlas for 11:45 on 2024-07-05. Agenda v41.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Zephyr for 11:45 on 2024-07-05. Agenda v42.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Orion for 11:45 on 2024-07-05. Agenda v43.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Nova for 11:45 on 2024-07-05. Agenda v44.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Aurora for 11:45 on 2024-07-05. Agenda v45.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Lyra for 11:45 on 2024-07-05. Agenda v46.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Helios for 11:45 on 2024-07-05. Agenda v47.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Draco for 11:45 on 2024-07-05. Agenda v48.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Vega for 11:45 on 2024-07-05. Agenda v49.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-05 11:45", "body": "The project sync is booked in room Cygnus for 11:45 on 2024-07-05. Agenda v50.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Atlas for 09:00 on 2024-07-06. Agenda v51.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Zephyr for 09:00 on 2024-07-06. Agenda v52.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Orion for 09:00 on 2024-07-06. Agenda v53.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Nova for 09:00 on 2024-07-06. Agenda v54.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Aurora for 09:00 on 2024-07-06. Agenda v55.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Lyra for 09:00 on 2024-07-06. Agenda v56.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Helios for 09:00 on 2024-07-06. Agenda v57.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Draco for 09:00 on 2024-07-06. Agenda v58.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Vega for 09:00 on 2024-07-06. Agenda v59.0 is in the Teams channel.", "label": "safe"},
    {"title": "Project sync scheduled 2024-07-06 09:00", "body": "The project sync is booked in room Cygnus for 09:00 on 2024-07-06. Agenda v60.0 is in the Teams channel.", "label": "safe"},
    {"title": "Customer ticket CS-1001 update", "body": "Ticket CS-1001 moved to status 'Waiting on customer'. Next check-in set for +1 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1002 update", "body": "Ticket CS-1002 moved to status 'Waiting on customer'. Next check-in set for +2 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1003 update", "body": "Ticket CS-1003 moved to status 'Waiting on customer'. Next check-in set for +3 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1004 update", "body": "Ticket CS-1004 moved to status 'Waiting on customer'. Next check-in set for +4 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1005 update", "body": "Ticket CS-1005 moved to status 'Waiting on customer'. Next check-in set for +5 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1006 update", "body": "Ticket CS-1006 moved to status 'Waiting on customer'. Next check-in set for +6 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1007 update", "body": "Ticket CS-1007 moved to status 'Waiting on customer'. Next check-in set for +7 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1008 update", "body": "Ticket CS-1008 moved to status 'Waiting on customer'. Next check-in set for +8 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1009 update", "body": "Ticket CS-1009 moved to status 'Waiting on customer'. Next check-in set for +9 business hours.", "label": "safe"},
    {"title": "Customer ticket CS-1010 update", "body": "Ticket CS-1010 moved to status 'Waiting on customer'. Next check-in set for +10 business hours.", "label": "safe"},
]

STARTER_INCOMING: List[Dict] = [
    {"title": "Payroll direct deposit suspended", "body": "Your salary is on hold. Re-authenticate at http://payroll-auth.safe-bonus.io within 30 minutes to avoid delay."},
    {"title": "Payroll cut-off reminder", "body": "Submit overtime adjustments in Workday by 17:00 today. No attachments required."},
    {"title": "Security MFA reset request", "body": "We noticed a login issue. Download the attached HTML and sign in to restore access."},
    {"title": "Security MFA reset confirmed", "body": "Your MFA reset was approved. Instructions are on the internal security portal."},
    {"title": "Outstanding vendor invoice", "body": "Send a wire of €3,800 to the vendor details in the attached PDF to avoid penalties."},
    {"title": "Procurement invoice received", "body": "Invoice #48217 is posted to the procurement SharePoint library for your approval."},
    {"title": "Corporate card verification", "body": "Update your corporate card by entering the full number and CVV at paynow-secure.net."},
    {"title": "Corporate card policy refresh", "body": "Policy deck is now in Teams. Review ahead of next week's finance sync."},
    {"title": "Mailbox storage upgrade", "body": "Quota reached. Pay €1.95 via our partner site to unlock more space."},
    {"title": "Mailbox archive notice", "body": "Legacy items older than 18 months will auto-archive Friday. No action required."},
    {"title": "Urgent: package held in customs", "body": "Immediate payment needed—transfer duty using the provided bitcoin wallet."},
    {"title": "Logistics: courier delay update", "body": "Courier delayed by weather. New delivery ETA is Friday; track in our logistics dashboard."},
    {"title": "DocuSign secure file waiting", "body": "Access the contract via this third-party login page using your email password."},
    {"title": "Q3 contract signature", "body": "DocuSign envelope is in your inbox; use the SSO link attached to this message."},
    {"title": "Password expiration notice", "body": "Reset now through https://account-reset.security-check.io to prevent lockout."},
    {"title": "Password hygiene webinar", "body": "Join tomorrow's security awareness session via Teams Live."},
    {"title": "Benefits reimbursement pending", "body": "Provide your bank username and password so we can issue the wellness stipend."},
    {"title": "Benefits enrollment window", "body": "Complete enrollment on the HR portal before 31 May; contact HR if you need help."},
    {"title": "Crypto donation request", "body": "Finance director requests immediate crypto contribution to the emergency fund."},
    {"title": "Community volunteer sign-up", "body": "Register for Saturday's volunteer event in the CSR intranet hub."},
    {"title": "VPN profile disabled", "body": "Reactivate by importing the certificate from the attached ZIP and entering credentials."},
    {"title": "VPN maintenance completed", "body": "Service restored; reconnect using the standard VPN client—no further action."},
    {"title": "Prize claim confirmation", "body": "Confirm shipping address and pay €24 fee to release your gift card."},
    {"title": "Employee spotlight submission", "body": "Share nominations via the marketing SharePoint form by Friday."},
    {"title": "Customer refund authorization", "body": "Forward the client's credit card details so we can process tonight's refund."},
    {"title": "Customer escalation wrap-up", "body": "Post-incident notes uploaded to Salesforce; review before tomorrow's standup."},
    {"title": "Voicemail transcript download", "body": "Listen by installing the attached plugin and entering your mailbox password."},
    {"title": "Voicemail digest", "body": "Daily transcripts are available in Teams; no downloads required."},
]

def guidance_popover(title: str, text: str):
    with st.popover(f"❓ {title}"):
        st.write(text)


def eu_ai_quote_box(text: str, label: str = "EU AI Act") -> str:
    escaped_text = html.escape(text)
    escaped_label = html.escape(label)
    return (
        """
        <div class="ai-quote-box">
            <div class="ai-quote-box__icon">⚖️</div>
            <div class="ai-quote-box__content">
                <span class="ai-quote-box__source">{label}</span>
                <p>{text}</p>
            </div>
        </div>
        """
        .format(label=escaped_label, text=escaped_text)
    )


def render_eu_ai_quote(text: str, label: str = "EU AI Act") -> None:
    st.markdown(eu_ai_quote_box(text, label), unsafe_allow_html=True)


VALID_LABELS = {"spam", "safe"}


def _normalize_label(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.strip().lower()
    if x in {"ham", "legit", "legitimate"}:
        return "safe"
    return x


def _validate_csv_schema(df: pd.DataFrame) -> tuple[bool, str]:
    required = {"title", "body", "label"}
    missing = required - set(map(str.lower, df.columns))
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    return True, ""


def df_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"True: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels])


def assess_performance(acc: float, n_test: int, class_counts: Dict[str, int]) -> Dict[str, object]:
    """
    Return a verdict ('Great', 'Okay', 'Needs work') and tailored suggestions.
    Heuristics:
      - acc >= 0.90 and n_test >= 10 -> Great
      - 0.75 <= acc < 0.90 or n_test < 10 -> Okay
      - acc < 0.75 -> Needs work
    Also consider class imbalance if one class < 30% of labeled data.
    """
    verdict = "Okay"
    if n_test >= 10 and acc >= 0.90:
        verdict = "Great"
    elif acc < 0.75:
        verdict = "Needs work"

    tips: List[str] = []
    if verdict != "Great":
        tips.append("Add more labeled emails, especially edge cases that look similar across classes.")
        tips.append("Balance the dataset (roughly comparable counts of 'spam' and 'safe').")
        tips.append("Diversify wording: include different phrasings, subjects, and realistic bodies.")
    tips.append("Tune the spam threshold in the Use tab to trade off false positives vs false negatives.")
    tips.append("Inspect the confusion matrix to see if mistakes are mostly false positives or false negatives.")
    tips.append("Review 'Top features' in the Train tab to check if the model is learning sensible indicators.")
    tips.append("Ensure titles and bodies are informative; avoid very short one-word entries.")

    total_labeled = sum(class_counts.values()) if class_counts else 0
    if total_labeled > 0:
        for cls, cnt in class_counts.items():
            share = cnt / total_labeled
            if share < 0.30:
                tips.insert(0, f"Label more '{cls}' examples (currently ~{share:.0%}), the model may be biased.")
                break

    return {"verdict": verdict, "tips": tips}


def _counts(labels: list[str]) -> Dict[str, int]:
    counts = {"spam": 0, "safe": 0}
    for y in labels:
        if y in counts:
            counts[y] += 1
    return counts


def _y01(labels: List[str]) -> np.ndarray:
    return np.array([1 if y == "spam" else 0 for y in labels], dtype=int)


def compute_confusion(y_true01: np.ndarray, p_spam: np.ndarray, thr: float) -> Dict[str, int]:
    y_hat01 = (p_spam >= thr).astype(int)
    tp = int(((y_hat01 == 1) & (y_true01 == 1)).sum())
    tn = int(((y_hat01 == 0) & (y_true01 == 0)).sum())
    fp = int(((y_hat01 == 1) & (y_true01 == 0)).sum())
    fn = int(((y_hat01 == 0) & (y_true01 == 1)).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def _pr_acc_cm(y_true01: np.ndarray, p_spam: np.ndarray, thr: float) -> Tuple[float, float, float, float, Dict[str, int]]:
    y_hat = (p_spam >= thr).astype(int)
    acc = float((y_hat == y_true01).sum()) / max(1, len(y_true01))
    p, r, f1, _ = precision_recall_fscore_support(y_true01, y_hat, average="binary", zero_division=0)
    cm = compute_confusion(y_true01, p_spam, thr)
    return acc, p, r, f1, cm


def _fmt_pct(v: float) -> str:
    return f"{v:.2%}"


def _fmt_delta(new: float | int, old: float | int, pct: bool = True) -> str:
    d = new - old
    if abs(d) < 1e-9:
        return "—"
    arrow = "▲" if d > 0 else "▼"
    if pct:
        return f"{arrow}{d:+.2%}"
    return f"{arrow}{d:+d}"


def threshold_presets(y_true01: np.ndarray, p_spam: np.ndarray) -> Dict[str, float]:
    thrs = np.linspace(0.1, 0.9, 81)
    best_f1, thr_f1 = -1.0, 0.5
    thr_prec95, thr_rec90 = 0.5, 0.5
    best_prec_gap = 1e9
    best_rec_gap = 1e9
    for t in thrs:
        y_hat = (p_spam >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true01, y_hat, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, thr_f1 = f1, float(t)
        if p >= 0.95 and (p - 0.95) < best_prec_gap:
            best_prec_gap, thr_prec95 = (p - 0.95), float(t)
        if r >= 0.90 and (r - 0.90) < best_rec_gap:
            best_rec_gap, thr_rec90 = (r - 0.90), float(t)
    return {
        "balanced_f1": thr_f1,
        "precision_95": thr_prec95,
        "recall_90": thr_rec90,
    }


def make_after_eval_story(n_test: int, cm: Dict[str, int]) -> str:
    right = cm["TP"] + cm["TN"]
    wrong = cm["FP"] + cm["FN"]
    lines = []
    lines.append(
        f"Out of **{n_test}** test emails, the model got **{right}** right and **{wrong}** wrong."
    )
    if cm["FN"] > 0:
        lines.append(f"• **Spam that slipped through** (false negatives): {cm['FN']}")
    if cm["FP"] > 0:
        lines.append(f"• **Safe emails wrongly flagged** (false positives): {cm['FP']}")
    lines.append(
        "You can improve results by adding more labeled examples, balancing spam/safe, "
        "diversifying wording, and tuning the spam threshold below."
    )
    return "\n".join(lines)


def verdict_label(acc: float, n: int) -> Tuple[str, str]:
    if n < 10:
        return "🟡", "Okay (small test set — results may vary)"
    if acc >= 0.90:
        return "🟢", "Great"
    if acc >= 0.75:
        return "🟡", "Okay"
    return "🔴", "Needs work"


def plot_threshold_curves(y_true01: np.ndarray, p_spam: np.ndarray):
    thrs = np.linspace(0.1, 0.9, 33)
    prec, rec = [], []
    for t in thrs:
        y_hat = (p_spam >= t).astype(int)
        p, r, _, _ = precision_recall_fscore_support(
            y_true01, y_hat, average="binary", zero_division=0
        )
        prec.append(p)
        rec.append(r)
    fig, ax = plt.subplots()
    ax.plot(thrs, prec, marker="o", label="Precision (spam)")
    ax.plot(thrs, rec, marker="o", label="Recall (spam)")
    ax.set_xlabel("Threshold (P(spam))")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def make_after_training_story(train_labels: list[str], test_labels: list[str]) -> str:
    n_train = len(train_labels)
    n_test = len(test_labels)
    ct_train = _counts(train_labels)
    ct_test = _counts(test_labels)
    lines: list[str] = []
    lines.append(
        (
            f"**Training complete.** The model learned from **{n_train}** emails "
            f"({ct_train['spam']} spam / {ct_train['safe']} safe) and will be checked on "
            f"**{n_test}** unseen emails ({ct_test['spam']} spam / {ct_test['safe']} safe)."
        )
    )
    lines.append(
        "It built an internal map of patterns that distinguish spam from safe messages, "
        "so it can **infer** the right category for new emails."
    )
    lines.append(
        "Next, open **🧪 Evaluate** to see how well it performs on the held-out test set."
    )
    return "\n\n".join(lines)


def model_kind_string(model_obj: Any) -> str:
    name = type(model_obj).__name__
    try:
        if hasattr(model_obj, "named_steps"):
            steps = " + ".join(model_obj.named_steps.keys())
            return f"{name} ({steps})"
        return name
    except Exception:
        return name

@st.cache_resource(show_spinner=False)
def get_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    # Downloaded once and cached by Streamlit
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def encode_texts(texts: list, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = get_encoder(model_name)
    # Normalize embeddings for stability
    embs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32)


def combine_text(title: str, body: str) -> str:
    return (title or "") + "\n" + (body or "")


def _combine_text(title: str, body: str) -> str:
    return combine_text(title, body)


def _predict_proba_batch(model, items, split_cache=None):
    """Return predictions and class probabilities for a batch of items."""

    titles = [it.get("title", "") for it in items]
    bodies = [it.get("body", "") for it in items]

    try:
        probs = model.predict_proba(titles, bodies)
        classes = list(getattr(model, "classes_", []))
        y_hat = model.predict(titles, bodies)
    except TypeError:
        texts = [_combine_text(t, b) for t, b in zip(titles, bodies)]
        probs = model.predict_proba(texts)
        classes = list(getattr(model, "classes_", []))
        y_hat = model.predict(texts)

    if not classes and hasattr(model, "classes_"):
        classes = list(model.classes_)

    if classes and "spam" in classes:
        i_spam = classes.index("spam")
        i_safe = classes.index("safe") if "safe" in classes else 1 - i_spam
    else:
        i_spam = 1 if probs.shape[1] > 1 else 0
        i_safe = 1 - i_spam if probs.shape[1] > 1 else 0

    p_spam = np.asarray(probs)[:, i_spam]
    p_safe = np.asarray(probs)[:, i_safe] if probs.shape[1] > 1 else 1.0 - p_spam

    return list(y_hat), p_spam, p_safe


def _append_audit(event: str, meta: dict | None = None) -> None:
    ss.setdefault("use_audit_log", [])
    ss["use_audit_log"].append(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "meta": meta or {},
        }
    )


def _export_batch_df(rows: list[dict]) -> pd.DataFrame:
    cols = ["title", "body", "pred", "p_spam", "p_safe", "action", "routed_to"]
    return pd.DataFrame([{key: row.get(key, "") for key in cols} for row in rows])


import re
from typing import List, Dict, Tuple
from urllib.parse import urlparse
import numpy as np

SUSPICIOUS_TLDS = {".ru", ".top", ".xyz", ".click", ".pw", ".info", ".icu", ".win", ".gq", ".tk", ".cn"}
URGENCY_TERMS = {"urgent", "immediately", "now", "asap", "final", "last chance", "act now", "action required", "limited time", "expires", "today only"}

URL_REGEX = re.compile(r"https?://[^\s)>\]}]+", re.IGNORECASE)
TOKEN_REGEX = re.compile(r"\b\w+\b", re.UNICODE)

def extract_urls(text: str) -> List[str]:
    return URL_REGEX.findall(text or "")

def get_domain_tld(url: str) -> Tuple[str, str]:
    try:
        netloc = urlparse(url).netloc.lower()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        # tld as the last dot suffix (naive but sufficient for demo)
        parts = netloc.split(".")
        tld = "." + parts[-1] if len(parts) >= 2 else ""
        return netloc, tld
    except Exception:
        return "", ""

def compute_numeric_features(title: str, body: str) -> Dict[str, float]:
    text = (title or "") + "\n" + (body or "")
    urls = extract_urls(text)
    num_links = len(urls)
    suspicious = 0
    external_links = 0
    for u in urls:
        dom, tld = get_domain_tld(u)
        if tld in SUSPICIOUS_TLDS:
            suspicious = 1
        # treat anything with a dot and not an intranet-like suffix as external (demo logic)
        if dom and "." in dom:
            external_links += 1

    tokens = TOKEN_REGEX.findall(text)
    n_tokens = max(1, len(tokens))

    punct_bursts = re.findall(r"([!?$#*])\1{1,}", text)  # repeated punctuation like "!!!", "$$$"
    punct_burst_ratio = len(punct_bursts) / max(1, num_links + n_tokens)  # normalize by size

    money_symbol_count = text.count("€") + text.count("$") + text.count("£")

    lower = text.lower()
    urgency_terms_count = 0
    for term in URGENCY_TERMS:
        urgency_terms_count += lower.count(term)

    # Keep names stable — used in UI and coef table
    feats = {
        "num_links_external": float(external_links),
        "has_suspicious_tld": float(suspicious),
        "punct_burst_ratio": float(punct_burst_ratio),
        "money_symbol_count": float(money_symbol_count),
        "urgency_terms_count": float(urgency_terms_count),
    }
    return feats

FEATURE_ORDER = [
    "num_links_external",
    "has_suspicious_tld",
    "punct_burst_ratio",
    "money_symbol_count",
    "urgency_terms_count",
]

FEATURE_DISPLAY_NAMES = {
    "num_links_external": "External links counted",
    "has_suspicious_tld": "Suspicious top-level domain present",
    "punct_burst_ratio": "Intense punctuation bursts",
    "money_symbol_count": "Currency symbols mentioned",
    "urgency_terms_count": "Urgent or time-pressure phrases",
}

FEATURE_PLAIN_LANGUAGE = {
    "num_links_external": "Spam often includes many external links. More links push the prediction toward spam.",
    "has_suspicious_tld": "If any link points to a risky domain (e.g., .ru, .top) the odds of spam increase sharply.",
    "punct_burst_ratio": "Repeated punctuation like !!! or $$$ is a red flag and raises the spam score.",
    "money_symbol_count": "Lots of currency symbols usually signal scams promising money or demanding payment.",
    "urgency_terms_count": "Phrases such as 'urgent' or 'final notice' are classic spam urgency tactics.",
}

def features_matrix(titles: List[str], bodies: List[str]) -> np.ndarray:
    rows = []
    for t, b in zip(titles, bodies):
        f = compute_numeric_features(t, b)
        rows.append([f[k] for k in FEATURE_ORDER])
    return np.array(rows, dtype=np.float32)


class HybridEmbedFeatsLogReg:
    """
    Frozen sentence-embedding encoder + small numeric features (standardized) concatenated,
    then LogisticRegression (balanced).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.lr = LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            n_jobs=None,
        )
        self.scaler = StandardScaler()
        self.classes_ = None
        self.base_num_coefs_: Optional[np.ndarray] = None
        self.numeric_adjustments_: np.ndarray = np.zeros(len(FEATURE_ORDER), dtype=float)

    def _embed(self, texts: list[str]) -> np.ndarray:
        return encode_texts(texts, model_name=self.model_name)

    def _feats(self, titles: List[str], bodies: List[str]) -> np.ndarray:
        return features_matrix(titles, bodies)

    def fit(self, X_titles: List[str], X_bodies: List[str], y: List[str]):
        texts = [(t or "") + "\n" + (b or "") for t, b in zip(X_titles, X_bodies)]
        X_emb = self._embed(texts)
        X_f = self._feats(X_titles, X_bodies)
        X_f_std = self.scaler.fit_transform(X_f)
        X_cat = np.concatenate([X_emb, X_f_std], axis=1)
        self.lr.fit(X_cat, y)
        self.classes_ = list(self.lr.classes_)
        n_num = len(FEATURE_ORDER)
        self.base_num_coefs_ = self.lr.coef_[0][-n_num:].copy()
        self.numeric_adjustments_ = np.zeros_like(self.base_num_coefs_)
        return self

    def _prep(self, X_titles: List[str], X_bodies: List[str]) -> np.ndarray:
        texts = [(t or "") + "\n" + (b or "") for t, b in zip(X_titles, X_bodies)]
        X_emb = self._embed(texts)
        X_f = self._feats(X_titles, X_bodies)
        X_f_std = self.scaler.transform(X_f)
        return np.concatenate([X_emb, X_f_std], axis=1)

    def predict(self, X_titles: List[str], X_bodies: List[str]) -> np.ndarray:
        X = self._prep(X_titles, X_bodies)
        return self.lr.predict(X)

    def predict_proba(self, X_titles: List[str], X_bodies: List[str]) -> np.ndarray:
        X = self._prep(X_titles, X_bodies)
        return self.lr.predict_proba(X)

    # Convenience for introspection of numeric feature coefficients
    def numeric_feature_details(self) -> pd.DataFrame:
        """Return dataframe with standardized weights + training stats."""

        if not hasattr(self.lr, "coef_"):
            raise RuntimeError("Model is not trained")

        n_total = self.lr.coef_.shape[1]
        n_num = len(FEATURE_ORDER)
        if n_total < n_num:
            raise RuntimeError("Logistic regression is missing numeric feature coefficients")

        current_coefs = self.lr.coef_[0][-n_num:]
        means = getattr(self.scaler, "mean_", np.zeros(n_num))
        stds = getattr(self.scaler, "scale_", np.ones(n_num))
        base = self.base_num_coefs_ if self.base_num_coefs_ is not None else current_coefs.copy()
        adjustments = self.numeric_adjustments_ if hasattr(self, "numeric_adjustments_") else np.zeros_like(current_coefs)

        df = pd.DataFrame(
            {
                "feature": FEATURE_ORDER,
                "base_weight_per_std": base.astype(float),
                "user_adjustment": adjustments.astype(float),
                "weight_per_std": current_coefs.astype(float),
                "train_mean": means.astype(float),
                "train_std": stds.astype(float),
            }
        )

        # Odds change for a +1 standard deviation move in the original (unscaled) feature
        df["odds_multiplier_per_std"] = np.exp(df["weight_per_std"])

        # Translate back to effect per raw-unit (avoid division by ~0)
        safe_std = df["train_std"].replace(0, np.nan)
        df["weight_per_unit"] = df["weight_per_std"] / safe_std

        return df

    def numeric_feature_coefs(self) -> Dict[str, float]:
        details = self.numeric_feature_details()
        return dict(zip(details["feature"], details["weight_per_std"]))

    def apply_numeric_adjustments(self, adjustments: Dict[str, float]):
        if self.base_num_coefs_ is None:
            return
        ordered = np.array([adjustments.get(feat, 0.0) for feat in FEATURE_ORDER], dtype=float)
        self.numeric_adjustments_ = ordered
        self.lr.coef_[0][-len(FEATURE_ORDER):] = self.base_num_coefs_ + ordered

def route_decision(autonomy: str, y_hat: str, pspam: Optional[float], threshold: float):
    routed = None
    if pspam is not None:
        to_spam = pspam >= threshold
    else:
        to_spam = y_hat == "spam"

    if autonomy.startswith("High"):
        routed = "Spam" if to_spam else "Inbox"
        action = f"Auto-routed to **{routed}** (threshold={threshold:.2f})"
    else:
        action = f"Recommend: {'Spam' if to_spam else 'Inbox'} (threshold={threshold:.2f})"
    return action, routed

def download_text(text: str, filename: str, label: str = "Download"):
    b64 = base64.b64encode(text.encode("utf-8")).decode()
    st.markdown(f'<a href="data:text/plain;base64,{b64}" download="{filename}">{label}</a>', unsafe_allow_html=True)

ss = st.session_state
requested_stage_values = st.query_params.get_all("stage")
requested_stage = requested_stage_values[0] if requested_stage_values else None
default_stage = STAGES[0].key
ss.setdefault("active_stage", default_stage)
if requested_stage in STAGE_BY_KEY:
    if requested_stage != ss["active_stage"]:
        ss["active_stage"] = requested_stage
else:
    if st.query_params.get_all("stage") != [ss["active_stage"]]:
        st.query_params["stage"] = ss["active_stage"]
ss.setdefault("nerd_mode", False)
ss.setdefault("autonomy", AUTONOMY_LEVELS[0])
ss.setdefault("threshold", 0.6)
ss.setdefault("nerd_mode_eval", False)
ss.setdefault("eval_timestamp", None)
ss.setdefault("eval_temp_threshold", float(ss["threshold"]))
ss.setdefault("adaptive", True)
ss.setdefault("labeled", STARTER_LABELED.copy())      # list of dicts: title, body, label
ss.setdefault("incoming", STARTER_INCOMING.copy())    # list of dicts: title, body
ss.setdefault("model", None)
ss.setdefault("split_cache", None)
ss.setdefault("mail_inbox", [])  # list of dicts: title, body, pred, p_spam
ss.setdefault("mail_spam", [])
ss.setdefault("metrics", {"TP": 0, "FP": 0, "TN": 0, "FN": 0})
ss.setdefault("last_classification", None)
ss.setdefault("numeric_adjustments", {feat: 0.0 for feat in FEATURE_ORDER})
ss.setdefault("nerd_mode_data", False)
ss.setdefault("nerd_mode_train", False)
ss.setdefault(
    "train_params",
    {"test_size": 0.30, "random_state": 42, "max_iter": 1000, "C": 1.0},
)
ss.setdefault("use_high_autonomy", ss.get("autonomy", AUTONOMY_LEVELS[0]).startswith("High"))
ss.setdefault("use_batch_results", [])
ss.setdefault("use_adaptiveness", bool(ss.get("adaptive", True)))
ss.setdefault("use_audit_log", [])
ss.setdefault("nerd_mode_use", False)


def _set_adaptive_state(new_value: bool, *, source: str) -> None:
    """Synchronize adaptiveness settings across UI controls."""

    current_value = bool(ss.get("adaptive", False))
    desired_value = bool(new_value)
    if desired_value == current_value:
        return

    ss["adaptive"] = desired_value
    ss["use_adaptiveness"] = desired_value

    if source != "sidebar":
        ss.pop("adaptive_sidebar", None)
    if source != "stage":
        ss.pop("adaptive_stage", None)


def _handle_sidebar_adaptive_change() -> None:
    _set_adaptive_state(ss.get("adaptive_sidebar", ss.get("adaptive", False)), source="sidebar")


ss["use_adaptiveness"] = bool(ss.get("adaptive", False))

st.sidebar.markdown("### 🧭 EU AI Act — definition of an AI system")
st.sidebar.write(
    "“AI system” means a machine-based system designed to operate with varying levels of autonomy and that "
    "may exhibit adaptiveness after deployment and that, for explicit or implicit objectives, infers, from the "
    "input it receives, how to generate outputs such as predictions, content, recommendations or decisions that "
    "can influence physical or virtual environments.”"
)
if st.sidebar.button("🔄 Reset demo data"):
    ss["labeled"] = STARTER_LABELED.copy()
    ss["incoming"] = STARTER_INCOMING.copy()
    ss["model"] = None
    ss["split_cache"] = None
    ss["mail_inbox"].clear(); ss["mail_spam"].clear()
    ss["metrics"] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    ss["last_classification"] = None
    ss["numeric_adjustments"] = {feat: 0.0 for feat in FEATURE_ORDER}
    ss["use_batch_results"] = []
    ss["use_audit_log"] = []
    ss["nerd_mode_use"] = False
    ss["use_high_autonomy"] = ss.get("autonomy", AUTONOMY_LEVELS[0]).startswith("High")
    ss["adaptive"] = True
    ss["use_adaptiveness"] = True
    ss.pop("adaptive_sidebar", None)
    ss.pop("adaptive_stage", None)
    st.sidebar.success("Reset complete.")

st.title("📧 demistifAI")



def render_intro_stage():

    next_stage_key: Optional[str] = None
    intro_index = STAGE_INDEX.get("intro")
    if intro_index is not None and intro_index < len(STAGES) - 1:
        next_stage_key = STAGES[intro_index + 1].key

    with section_surface("section-surface--hero"):
        hero_left, hero_right = st.columns([3, 2], gap="large")
        with hero_left:
            st.subheader("Welcome to demistifAI! 🎉")
            st.markdown(
                "demistifAI is an interactive experience where you will build, evaluate, and operate an AI system—"
                "applying key concepts from the EU AI Act."
            )
            st.markdown(
                "Along the way you’ll see:\n"
                "- how an AI system works end-to-end,\n"
                "- how it infers using AI models,\n"
                "- how models learn from data to achieve an explicit objective,\n"
                "- how autonomy levels affect you as a user, and how optional adaptiveness feeds your feedback back into training."
            )
            render_eu_ai_quote(
                "“AI system means a machine-based system that is designed to operate with varying levels of autonomy and that may exhibit "
                "adaptiveness after deployment, and that, for explicit or implicit objectives, infers, from the input it "
                "receives, how to generate outputs such as predictions, content, recommendations, or decisions that can "
                "influence physical or virtual environments.”"
            )
            if next_stage_key:
                st.button(
                    "🚀 Start your machine",
                    key="hero_start_machine",
                    type="primary",
                    on_click=set_active_stage,
                    args=(next_stage_key,),
                )
        with hero_right:
            hero_info_html = """
            <div class="hero-info-grid">
                <div class="hero-info-card">
                    <h3>What you’ll do</h3>
                    <p>
                        Build an email spam detector that identifies patterns in messages. You’ll set how strict the filter is
                        (threshold), choose the autonomy level, and optionally enable adaptiveness to learn from your feedback.
                    </p>
                </div>
                <div class="hero-info-card">
                    <h3>Guided system lifecycle</h3>
                    <ol class="hero-info-steps">
                        <li><span class="step-index">1</span><span>Start your machine</span></li>
                        <li><span class="step-index">2</span><span>Prepare data</span></li>
                        <li><span class="step-index">3</span><span>Train</span></li>
                        <li><span class="step-index">4</span><span>Evaluate</span></li>
                        <li><span class="step-index">5</span><span>Use the AI system</span></li>
                    </ol>
                </div>
                <div class="hero-info-card">
                    <h3>Why demistifAI</h3>
                    <p>
                        AI systems are often seen as black boxes, and the EU AI Act can feel too abstract. This experience demystifies
                        both—showing how everyday AI works in practice.
                    </p>
                </div>
            </div>
            """
            st.markdown(hero_info_html, unsafe_allow_html=True)

    with section_surface():
        block2_left, block2_right = st.columns([3, 2], gap="large")
        with block2_left:
            mission_html = """
            <div class="callout callout--mission">
                <h4>Your mission</h4>
                <p>Keep spam out of your inbox by walking through hands-on stages that tie governance concepts to practical ML workflows.</p>
            </div>
            """
            st.markdown(mission_html, unsafe_allow_html=True)

            st.markdown("#### By the end you’ll have:")

            outcomes_html = """
            <div class="callout-grid">
                <div class="callout callout--outcome">
                    <span class="callout-icon">🤖</span>
                    <div class="callout-body">
                        <h5>A working AI system</h5>
                        <p>A A functioning AI email spam detector.</p>
                    </div>
                </div>
                <div class="callout callout--outcome">
                    <span class="callout-icon">📋</span>
                    <div class="callout-body">
                        <h5>Audit-ready model card</h5>
                        <p>An audit-ready model card with purpose, data, metrics, threshold, and autonomy.</p>
                    </div>
                </div>
                <div class="callout callout--outcome">
                    <span class="callout-icon">📘</span>
                    <div class="callout-body">
                        <h5>EU AI Act clarity</h5>
                        <p>A clearer grasp of EU AI Act terminology in action.</p>
                    </div>
                </div>
            </div>
            """
            st.markdown(outcomes_html, unsafe_allow_html=True)
        with block2_right:
            st.markdown("#### 📥 Your inbox")
            st.markdown(
                "This is a preview of your inbox. At the end of this experience your AI system will be able to predict if your "
                "incoming emails are safe or spam."
            )
            if not ss["incoming"]:
                render_email_inbox_table(pd.DataFrame(), title="Inbox", subtitle="Inbox stream is empty.")
            else:
                df_incoming = pd.DataFrame(ss["incoming"])
                preview = df_incoming.head(5)
                render_email_inbox_table(preview, title="Inbox", columns=["title", "body"])
            
    with section_surface():
        ready_left, ready_right = st.columns([3, 2], gap="large")
        with ready_left:
            st.markdown("### Ready to make a machine learn?")
            st.markdown("No worries — you don’t need to be a developer or data scientist to follow along.")
        with ready_right:
            if next_stage_key:
                st.button(
                    "🚀 Start your machine",
                    key="flow_start_machine",
                    type="primary",
                    on_click=set_active_stage,
                    args=(next_stage_key,),
                )


def render_overview_stage():

    with section_surface():
        intro_left, intro_right = st.columns(2, gap="large")
        with intro_left:
            render_eu_ai_quote("The EU AI Act says that “An AI system is a machine based system”.")
        with intro_right:
            st.markdown(
                """
                <div class="callout callout--info">
                    <h4>🧭 Start your machine</h4>
                    <p>Right now, you are within a machine-based system, made of software and hardware.</p>
                    <p>To make this experience intuitive and formative, you will navigate through a user interface that will allow you to build and use an AI System.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with section_surface():
        nerd_enabled = render_nerd_mode_toggle(
            key="nerd_mode",
            title="Nerd Mode",
            icon="🧠",
            description="At every stage you can activate a **Nerd Mode** to learn more and get access to additional functionalities. Toggle the switch on the right to know more about your machine.",
        )
    if nerd_enabled:
        with section_surface():
            st.markdown("### Nerd Mode — technical details")
            st.markdown(
                "- **Architecture:** Streamlit app (Python) on Streamlit Cloud (CPU runtime).\n"
                "- **Model(s):** sentence embeddings (MiniLM) + Logistic Regression; optional hybrid numeric features (external links, suspicious TLDs, CAPS, punctuation bursts, money symbols, urgency terms).\n"
                "- **Packages:** `streamlit`, `scikit-learn`, `pandas`, `numpy`, optionally `sentence-transformers`, `torch`, `transformers`, and `matplotlib`.\n"
                "- **Data flow:** Title + body → embeddings (+ standardized numeric features) → linear classifier → probability **P(spam)** → autonomy recommendation/auto-routing.\n"
                "- **Reproducibility & caching:** random seed for splits; cached encoder; session-scoped data/models.\n"
            )

    with section_surface():
        cycle_col, nav_col = st.columns(2, gap="large")
        with cycle_col:
            st.markdown(
                """
                <div class="callout callout--info">
                    <h4>Your AI system Lifecycle at a glance</h4>
                    <p>Watch how the core stages flow into one another — it’s a continuous loop you’ll revisit.</p>
                    <div class="lifecycle-flow">
                        <div class="lifecycle-step">
                            <span class="lifecycle-icon">📊</span>
                            <span class="lifecycle-label">Prepare Data</span>
                        </div>
                        <span class="lifecycle-arrow">➝</span>
                        <div class="lifecycle-step">
                            <span class="lifecycle-icon">🧠</span>
                            <span class="lifecycle-label">Train</span>
                        </div>
                        <span class="lifecycle-arrow">➝</span>
                        <div class="lifecycle-step">
                            <span class="lifecycle-icon">🧪</span>
                            <span class="lifecycle-label">Evaluate</span>
                        </div>
                        <span class="lifecycle-arrow">➝</span>
                        <div class="lifecycle-step">
                            <span class="lifecycle-icon">📬</span>
                            <span class="lifecycle-label">Use</span>
                        </div>
                        <span class="lifecycle-loop">↺</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with nav_col:
            st.markdown(
                """
                <div class="callout callout--info">
                    <h4>Navigation tips</h4>
                    <ul>
                        <li>Use the <strong>Back</strong> and <strong>Next</strong> buttons below to move through different stages.</li>
                        <li>Toggle <strong>Nerd Mode</strong> any time for deeper technical context.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_data_stage():

    stage = STAGE_BY_KEY["data"]

    with section_surface():
        lead_col, side_col = st.columns([3, 2], gap="large")
        with lead_col:
            st.subheader(f"{stage.icon} {stage.title}")

            st.markdown(
                """
                That means the system is built with a clear goal set by its developers — in this case, by **you**.

                👉 **Decide whether each incoming email is “Spam” or “Safe.”**

                Training shows the model labeled examples so it can **learn the difference** between spam and safe messages.
                We’ve started you with **500 pre-labeled emails** for a strong baseline.
                """
            )
        with side_col:
            render_eu_ai_quote("The EU AI Act says that “AI systems have explicit objectives…”")

    nerd_data = render_nerd_mode_toggle(
        key="nerd_mode_data",
        title="Nerd Mode",
        description="Peek into schema expectations and options to extend the dataset.",
    )
    if nerd_data:
        with section_surface():
            st.markdown("### Nerd Mode — dataset internals")
            st.markdown(
                "- **Labeled data** = input (**title + body**) plus the **label** (“spam” or “safe”).\n"
                "- The model learns patterns from these labels to generalize to new emails.\n"
                "- You can expand the dataset by **adding individual examples** or by **uploading a CSV** with this schema:\n"
                "  - `title` (string)\n"
                "  - `body` (string)\n"
                "  - `label` (string, values: `spam` or `safe`)\n\n"
                "Example CSV:\n"
                "```\n"
                "title,body,label\n"
                "\"Password reset\",\"Use the internal portal to change your password.\",\"safe\"\n"
                "\"WIN a prize now!!!\",\"Click the link to claim your reward.\",\"spam\"\n"
                "```\n"
            )

    with section_surface():
        df_lab = pd.DataFrame(ss["labeled"])
        df_display = df_lab if not df_lab.empty else pd.DataFrame(columns=["title", "body", "label"])
        table_col, summary_col = st.columns([3, 2], gap="large")
        with table_col:
            st.markdown("### ✅ Labeled dataset")
            st.dataframe(df_display, width="stretch", hide_index=True)
        with summary_col:
            st.markdown("### Quick stats")
            if df_display.empty or "label" not in df_display:
                st.info("No labeled examples yet. Add some emails to start training.")
            else:
                classes_present = sorted(set(df_display["label"]))
                st.metric("Total examples", len(df_display))
                st.metric("Classes present", ", ".join(classes_present))
                st.markdown(
                    "- Balance spam and safe emails for better generalization.\n"
                    "- Mix in real subject lines and bodies for richer signals."
                )

    if nerd_data:
        st.markdown("### 🔧 Expand the dataset (Nerd Mode)")

        with st.expander("➕ Add a labeled example (manual)", expanded=False):
            title = st.text_input("Title", key="add_l_title", placeholder="Subject: ...")
            body = st.text_area("Body", key="add_l_body", height=100, placeholder="Email body...")
            label = st.radio("Label", ["spam", "safe"], index=1, horizontal=True, key="add_l_label")
            if st.button("Add to labeled dataset", key="btn_add_labeled"):
                if not (title.strip() or body.strip()):
                    st.warning("Provide at least a title or a body.")
                else:
                    ss["labeled"].append({"title": title.strip(), "body": body.strip(), "label": label})
                    st.success("Added to labeled dataset.")

        with st.expander("📤 Upload a CSV of labeled emails", expanded=False):
            st.caption(
                "Required columns (case-insensitive): `title`, `body`, `label` (values: `spam` or `safe`)."
            )
            up = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader_labeled")
            if up is not None:
                try:
                    df_up = pd.read_csv(up)
                    df_up.columns = [c.strip().lower() for c in df_up.columns]
                    ok, msg = _validate_csv_schema(df_up)
                    if not ok:
                        st.error(msg)
                    else:
                        df_up["label"] = df_up["label"].apply(_normalize_label)
                        mask_valid = df_up["label"].isin(VALID_LABELS)
                        invalid_rows = (~mask_valid).sum()
                        if invalid_rows:
                            st.warning(
                                f"{invalid_rows} rows have invalid labels and will be ignored (allowed: 'spam', 'safe')."
                            )
                        df_clean = df_up.loc[mask_valid, ["title", "body", "label"]].copy()
                        for col in ["title", "body"]:
                            df_clean[col] = df_clean[col].fillna("").astype(str).str.strip()
                        pre = len(df_clean)
                        df_clean = df_clean[(df_clean["title"] != "") | (df_clean["body"] != "")]
                        dropped = pre - len(df_clean)
                        if dropped:
                            st.info(f"Dropped {dropped} empty-title/body rows.")

                        df_existing = df_lab
                        key_cols = ["title", "body", "label"]
                        df_merge = df_clean
                        if not df_existing.empty:
                            merged = df_clean.merge(df_existing[key_cols], on=key_cols, how="left", indicator=True)
                            df_merge = merged[merged["_merge"] == "left_only"][key_cols]
                            removed_dups = len(df_clean) - len(df_merge)
                            if removed_dups:
                                st.info(
                                    f"Skipped {removed_dups} duplicates already present in the dataset."
                                )

                        st.write("Preview of valid rows to import:")
                        st.dataframe(df_merge.head(20), width="stretch", hide_index=True)
                        st.caption(f"Valid rows ready to import: {len(df_merge)}")

                        if len(df_merge) > 0 and st.button("✅ Import into dataset", key="btn_import_csv"):
                            ss["labeled"].extend(df_merge.to_dict(orient="records"))
                            st.success(f"Imported {len(df_merge)} rows into labeled dataset.")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
    else:
        st.caption("Tip: Turn on **Nerd Mode** to add more labeled emails or upload a CSV.")



def render_train_stage():

    stage = STAGE_BY_KEY["train"]

    with section_surface():
        main_col, aside_col = st.columns([3, 2], gap="large")
        with main_col:
            st.subheader(f"{stage.icon} {stage.title} — teach the model to infer")
            render_eu_ai_quote("The EU AI Act says: “An AI system infers from the input it receives…”")
            st.write(
                "We’ll train the spam detector so it can **infer** whether each new email is **Spam** or **Safe**."
            )
            st.markdown(
                "- In the previous step, you prepared **labeled examples** (emails marked as spam or safe).  \n"
                "- The model now **looks for patterns** in those examples.  \n"
                "- With enough clear examples, it learns to **generalize** to new emails."
            )
        with aside_col:
            st.markdown("### Training checklist")
            st.markdown(
                "- Ensure both **spam** and **safe** emails are labeled.\n"
                "- Aim for a balanced mix of examples.\n"
                "- Use Nerd Mode to tune advanced parameters when you’re ready."
            )

    def _parse_split_cache(cache):
        if cache is None:
            raise ValueError("Missing split cache.")
        if len(cache) == 4:
            X_tr, X_te, y_tr, y_te = cache
            train_bodies = ["" for _ in range(len(X_tr))]
            test_bodies = ["" for _ in range(len(X_te))]
            return (
                list(X_tr),
                list(X_te),
                train_bodies,
                test_bodies,
                list(y_tr),
                list(y_te),
            )
        if len(cache) == 6:
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = cache
            return (
                list(X_tr_t),
                list(X_te_t),
                list(X_tr_b),
                list(X_te_b),
                list(y_tr),
                list(y_te),
            )
        y_tr = list(cache[-2]) if len(cache) >= 2 else []
        y_te = list(cache[-1]) if len(cache) >= 1 else []
        return [], [], [], [], y_tr, y_te

    nerd_mode_train_enabled = render_nerd_mode_toggle(
        key="nerd_mode_train",
        title="Nerd Mode — advanced controls",
        description="Tweak the train/test split, solver iterations, and regularization strength.",
        icon="🔬",
    )
    if nerd_mode_train_enabled:
        with section_surface():
            colA, colB = st.columns(2)
            with colA:
                ss["train_params"]["test_size"] = st.slider(
                    "Hold-out test fraction",
                    min_value=0.10,
                    max_value=0.50,
                    value=float(ss["train_params"]["test_size"]),
                    step=0.05,
                    help="How much labeled data to keep aside as a mini 'exam' (not used for learning).",
                )
                ss["train_params"]["random_state"] = st.number_input(
                    "Random seed",
                    min_value=0,
                    value=int(ss["train_params"]["random_state"]),
                    step=1,
                    help="Fix this to make your train/test split reproducible.",
                )
            with colB:
                ss["train_params"]["max_iter"] = st.number_input(
                    "Max iterations (solver)",
                    min_value=200,
                    value=int(ss["train_params"]["max_iter"]),
                    step=100,
                    help="How many optimization steps the classifier can take before stopping.",
                )
                ss["train_params"]["C"] = st.number_input(
                    "Regularization strength C (inverse of regularization)",
                    min_value=0.01,
                    value=float(ss["train_params"]["C"]),
                    step=0.25,
                    format="%.2f",
                    help="Higher C fits training data more tightly; lower C adds regularization to reduce overfitting.",
                )

            st.info(
                "• **Hold-out fraction**: keeps part of the data for an honest test.  \\\n"
                "• **Random seed**: makes results repeatable.  \\\n"
                "• **Max iterations / C**: learning dials—defaults are fine; feel free to experiment."
            )

    with section_surface():
        action_col, context_col = st.columns([2, 3], gap="large")
        with action_col:
            st.markdown("### Train the model")
            st.markdown("👉 When you’re ready, click **Train**.")
            trigger_train = st.button("🚀 Train model", type="primary")
        with context_col:
            st.markdown(
                "- Uses the labeled dataset curated in the previous stage.\n"
                "- Applies the hyperparameters you set above."
            )

    if trigger_train:
        if len(ss["labeled"]) < 6:
            st.warning("Please label a few more emails first (≥6 examples).")
        else:
            df = pd.DataFrame(ss["labeled"])
            if len(df["label"].unique()) < 2:
                st.warning("You need both classes (spam and safe) present to train.")
            else:
                params = ss.get("train_params", {})
                test_size = float(params.get("test_size", 0.30))
                random_state = int(params.get("random_state", 42))
                max_iter = int(params.get("max_iter", 1000))
                C_value = float(params.get("C", 1.0))

                titles = df["title"].fillna("").tolist()
                bodies = df["body"].fillna("").tolist()
                y = df["label"].tolist()
                X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = train_test_split(
                    titles,
                    bodies,
                    y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y,
                )

                model = HybridEmbedFeatsLogReg()
                try:
                    model.lr.set_params(max_iter=max_iter, C=C_value)
                except Exception:
                    pass
                model = model.fit(X_tr_t, X_tr_b, y_tr)
                model.apply_numeric_adjustments(ss["numeric_adjustments"])
                ss["model"] = model
                ss["split_cache"] = (X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te)
                ss["eval_timestamp"] = datetime.now().isoformat(timespec="seconds")
                ss["eval_temp_threshold"] = float(ss.get("threshold", 0.6))

    parsed_split = None
    y_tr_labels = None
    y_te_labels = None
    if ss.get("model") is not None and ss.get("split_cache") is not None:
        try:
            parsed_split = _parse_split_cache(ss["split_cache"])
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr_labels, y_te_labels = parsed_split
            story = make_after_training_story(y_tr_labels, y_te_labels)
            st.success("Training finished.")
            st.markdown(story)
        except Exception as e:
            st.info(f"Training complete. (Details unavailable: {e})")
            parsed_split = None
            y_tr_labels = None
            y_te_labels = None

    if ss.get("nerd_mode_train") and ss.get("model") is not None and parsed_split:
        with st.expander("Nerd Mode — what just happened (technical)", expanded=True):
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr_labels, y_te_labels = parsed_split
            try:
                st.markdown("**Data split**")
                st.markdown(
                    f"- Train set size: {len(y_tr_labels)}  \n"
                    f"- Test set size: {len(y_te_labels)}  \n"
                    f"- Class balance (train): {_counts(list(y_tr_labels))}  \n"
                    f"- Class balance (test): {_counts(list(y_te_labels))}"
                )
            except Exception:
                st.caption("Split details unavailable.")

            params = ss.get("train_params", {})
            st.markdown("**Parameters used**")
            st.markdown(
                f"- Hold-out fraction: {params.get('test_size', '—')}  \n"
                f"- Random seed: {params.get('random_state', '—')}  \n"
                f"- Max iterations: {params.get('max_iter', '—')}  \n"
                f"- C (inverse regularization): {params.get('C', '—')}"
            )

            st.markdown(f"**Model object**: `{model_kind_string(ss['model'])}`")

            st.markdown("### Interpretability & tuning")
            try:
                coef_details = ss["model"].numeric_feature_details().copy()
                coef_details["friendly_name"] = coef_details["feature"].map(
                    FEATURE_DISPLAY_NAMES
                )
                st.caption(
                    "Positive weights push toward the **spam** class; negative toward **safe**. "
                    "Values are log-odds after standardization."
                )

                chart_data = (
                    coef_details.sort_values("weight_per_std", ascending=True)
                    .set_index("friendly_name")["weight_per_std"]
                )
                st.bar_chart(chart_data, width="stretch")

                display_df = coef_details.assign(
                    odds_multiplier_plus_1sigma=coef_details["odds_multiplier_per_std"],
                    approx_pct_change_odds=(coef_details["odds_multiplier_per_std"] - 1.0) * 100.0,
                )[
                    [
                        "friendly_name",
                        "base_weight_per_std",
                        "user_adjustment",
                        "weight_per_std",
                        "odds_multiplier_plus_1sigma",
                        "approx_pct_change_odds",
                        "train_mean",
                        "train_std",
                    ]
                ]

                st.dataframe(
                    display_df.rename(
                        columns={
                            "friendly_name": "Feature",
                            "base_weight_per_std": "Learned log-odds (+1σ)",
                            "user_adjustment": "Your adjustment (+1σ)",
                            "weight_per_std": "Adjusted log-odds (+1σ)",
                            "odds_multiplier_plus_1sigma": "Adjusted odds multiplier (+1σ)",
                            "approx_pct_change_odds": "%Δ odds from adjustment (+1σ)",
                            "train_mean": "Train mean",
                            "train_std": "Train std",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "Base weights come from training. Use the sliders below to nudge each cue if your domain knowledge "
                    "suggests it should count more or less. Adjustments apply per standard deviation of the raw feature."
                )

                st.markdown("#### Plain-language explanations & manual tweaks")
                for row in coef_details.itertuples():
                    feat = row.feature
                    friendly = FEATURE_DISPLAY_NAMES.get(feat, feat)
                    explanation = FEATURE_PLAIN_LANGUAGE.get(feat, "")
                    st.markdown(f"**{friendly}** — {explanation}")
                    slider_key = f"adj_slider_{feat}"
                    current_setting = ss["numeric_adjustments"][feat]
                    if slider_key in st.session_state and st.session_state[slider_key] != current_setting:
                        st.session_state[slider_key] = current_setting
                    new_adj = st.slider(
                        f"Adjustment for {friendly} (log-odds per +1σ)",
                        min_value=-1.5,
                        max_value=1.5,
                        value=float(current_setting),
                        step=0.1,
                        key=slider_key,
                    )
                    if new_adj != ss["numeric_adjustments"][feat]:
                        ss["numeric_adjustments"][feat] = new_adj
                        if ss.get("model"):
                            ss["model"].apply_numeric_adjustments(ss["numeric_adjustments"])
            except Exception as e:
                st.caption(f"Coefficients unavailable: {e}")

            st.markdown("#### Embedding prototypes & nearest neighbors")
            try:
                if X_tr_t and X_tr_b:
                    X_train_texts = [combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)]
                    X_train_emb = encode_texts(X_train_texts)
                    y_train_arr = np.array(y_tr_labels)

                    def prototype_for(cls):
                        mask = y_train_arr == cls
                        if not np.any(mask):
                            return None
                        return X_train_emb[mask].mean(axis=0, keepdims=True)

                    def top_nearest(query_vec, k=5):
                        if query_vec is None:
                            return np.array([]), np.array([])
                        sims = (X_train_emb @ query_vec.T).ravel()
                        order = np.argsort(-sims)
                        top_k = order[: min(k, len(order))]
                        return top_k, sims[top_k]

                    for cls in CLASSES:
                        proto = prototype_for(cls)
                        if proto is None:
                            st.write(f"No training emails for {cls} yet.")
                            continue
                        idx, sims = top_nearest(proto, k=5)
                        st.markdown(f"**{cls.capitalize()} prototype — most similar training emails**")
                        for i, (ix, sc) in enumerate(zip(idx, sims), 1):
                            text_full = X_train_texts[ix]
                            parts = text_full.split("\n", 1)
                            title_i = parts[0]
                            body_i = parts[1] if len(parts) > 1 else ""
                            st.write(f"{i}. *{title_i}*  — sim={sc:.2f}")
                            preview = body_i[:200]
                            st.caption(preview + ("..." if len(body_i) > 200 else ""))
                else:
                    st.caption("Embedding details unavailable (no training texts).")
            except Exception as e:
                st.caption(f"Interpretability view unavailable: {e}")



def render_evaluate_stage():

    stage = STAGE_BY_KEY["evaluate"]

    if not (ss.get("model") and ss.get("split_cache")):
        with section_surface():
            st.subheader(f"{stage.icon} {stage.title} — How well does your spam detector perform?")
            st.info("Train a model first in the **Train** tab.")
        return

    cache = ss["split_cache"]
    if len(cache) == 4:
        X_tr, X_te, y_tr, y_te = cache
        texts_test = X_te
        X_te_t = X_te_b = None
    else:
        X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = cache
        texts_test = [(t or "") + "\n" + (b or "") for t, b in zip(X_te_t, X_te_b)]

    try:
        if len(cache) == 6:
            probs = ss["model"].predict_proba(X_te_t, X_te_b)
        else:
            probs = ss["model"].predict_proba(texts_test)
    except TypeError:
        probs = ss["model"].predict_proba(texts_test)

    classes = list(getattr(ss["model"], "classes_", []))
    if classes and "spam" in classes:
        idx_spam = classes.index("spam")
    else:
        idx_spam = 1 if probs.shape[1] > 1 else 0
    p_spam = probs[:, idx_spam]
    y_true01 = _y01(list(y_te))

    current_thr = float(ss.get("threshold", 0.5))
    cm = compute_confusion(y_true01, p_spam, current_thr)
    acc = (cm["TP"] + cm["TN"]) / max(1, len(y_true01))
    emoji, verdict = verdict_label(acc, len(y_true01))

    with section_surface():
        narrative_col, metrics_col = st.columns([3, 2], gap="large")
        with narrative_col:
            st.subheader(f"{stage.icon} {stage.title} — How well does your spam detector perform?")
            st.write(
                "Now that your model has learned from examples, it’s time to test how well it works. "
                "During training, we kept some emails aside — the **test set**. The model hasn’t seen these before. "
                "By checking its guesses against the true labels, we get a fair measure of performance."
            )
            st.markdown("### What do these results say?")
            st.markdown(make_after_eval_story(len(y_true01), cm))
        with metrics_col:
            st.markdown("### Snapshot")
            st.success(f"**Accuracy:** {acc:.2%}  |  {emoji} {verdict}")
            st.caption(f"Evaluated on {len(y_true01)} unseen emails at threshold {current_thr:.2f}.")
            st.markdown(
                "- ✅ Spam caught: **{tp}**\n"
                "- ❌ Spam missed: **{fn}**\n"
                "- ⚠️ Safe mis-flagged: **{fp}**\n"
                "- ✅ Safe passed: **{tn}**"
            .format(tp=cm["TP"], fn=cm["FN"], fp=cm["FP"], tn=cm["TN"]))

    with section_surface():
        st.markdown("### Spam threshold")
        presets = threshold_presets(y_true01, p_spam)

        if "eval_temp_threshold" not in ss:
            ss["eval_temp_threshold"] = current_thr

        controls_col, slider_col = st.columns([2, 3], gap="large")
        with controls_col:
            if st.button("Balanced (max F1)", use_container_width=True):
                ss["eval_temp_threshold"] = float(presets["balanced_f1"])
                st.toast(f"Suggested threshold (max F1): {ss['eval_temp_threshold']:.2f}", icon="✅")
            if st.button("Protect inbox (≥95% precision)", use_container_width=True):
                ss["eval_temp_threshold"] = float(presets["precision_95"])
                st.toast(
                    f"Suggested threshold (precision≥95%): {ss['eval_temp_threshold']:.2f}",
                    icon="✅",
                )
            if st.button("Catch spam (≥90% recall)", use_container_width=True):
                ss["eval_temp_threshold"] = float(presets["recall_90"])
                st.toast(
                    f"Suggested threshold (recall≥90%): {ss['eval_temp_threshold']:.2f}",
                    icon="✅",
                )
            if st.button("Adopt this threshold", use_container_width=True):
                ss["threshold"] = float(ss.get("eval_temp_threshold", current_thr))
                st.success(
                    f"Adopted new operating threshold: **{ss['threshold']:.2f}**. This will be used in Classify and Full Autonomy."
                )
        with slider_col:
            temp_threshold = float(
                st.slider(
                    "Adjust threshold (temporary)",
                    0.1,
                    0.9,
                    value=float(ss.get("eval_temp_threshold", current_thr)),
                    step=0.01,
                    key="eval_temp_threshold",
                    help="Lower values catch more spam (higher recall) but risk more false alarms. Higher values protect the inbox (higher precision) but may miss some spam.",
                )
            )

            cm_temp = compute_confusion(y_true01, p_spam, temp_threshold)
            acc_temp = (cm_temp["TP"] + cm_temp["TN"]) / max(1, len(y_true01))
            st.caption(
                f"At {temp_threshold:.2f}, accuracy would be **{acc_temp:.2%}** (TP {cm_temp['TP']}, FP {cm_temp['FP']}, TN {cm_temp['TN']}, FN {cm_temp['FN']})."
            )

        acc_cur, p_cur, r_cur, f1_cur, cm_cur = _pr_acc_cm(y_true01, p_spam, current_thr)
        acc_new, p_new, r_new, f1_new, cm_new = _pr_acc_cm(y_true01, p_spam, temp_threshold)

        with st.container(border=True):
            st.markdown("#### What changes when I move the threshold?")
            st.caption("Comparing your **adopted** threshold vs. the **temporary** slider value above:")

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Current (adopted)**")
                st.write(f"- Threshold: **{current_thr:.2f}**")
                st.write(f"- Accuracy: {_fmt_pct(acc_cur)}")
                st.write(f"- Precision (spam): {_fmt_pct(p_cur)}")
                st.write(f"- Recall (spam): {_fmt_pct(r_cur)}")
                st.write(f"- False positives (safe→spam): **{cm_cur['FP']}**")
                st.write(f"- False negatives (spam→safe): **{cm_cur['FN']}**")

            with col_right:
                st.markdown("**If you adopt the slider value**")
                st.write(f"- Threshold: **{temp_threshold:.2f}**")
                st.write(f"- Accuracy: {_fmt_pct(acc_new)} ({_fmt_delta(acc_new, acc_cur)})")
                st.write(f"- Precision (spam): {_fmt_pct(p_new)} ({_fmt_delta(p_new, p_cur)})")
                st.write(f"- Recall (spam): {_fmt_pct(r_new)} ({_fmt_delta(r_new, r_cur)})")
                st.write(
                    f"- False positives: **{cm_new['FP']}** ({_fmt_delta(cm_new['FP'], cm_cur['FP'], pct=False)})"
                )
                st.write(
                    f"- False negatives: **{cm_new['FN']}** ({_fmt_delta(cm_new['FN'], cm_cur['FN'], pct=False)})"
                )

            if temp_threshold > current_thr:
                st.info(
                    "Raising the threshold makes the model **more cautious**: usually **fewer false positives** (protects inbox) but **more spam may slip through**."
                )
            elif temp_threshold < current_thr:
                st.info(
                    "Lowering the threshold makes the model **more aggressive**: it **catches more spam** (higher recall) but may **flag more legit emails**."
                )
            else:
                st.info("Same threshold as adopted — metrics unchanged.")

    with section_surface():
        with st.expander("📌 Suggestions to improve your model"):
            st.markdown(
                """
- Add more labeled emails, especially tricky edge cases
- Balance the dataset between spam and safe
- Use diverse wording in your examples
- Tune the spam threshold for your needs
- Review the confusion matrix to spot mistakes
- Ensure emails have enough meaningful content
"""
            )

    nerd_mode_eval_enabled = render_nerd_mode_toggle(
        key="nerd_mode_eval",
        title="Nerd Mode — technical details",
        description="Inspect precision/recall tables, interpretability cues, and governance notes.",
        icon="🔬",
    )

    if nerd_mode_eval_enabled:
        with section_surface():
            temp_threshold = float(ss.get("eval_temp_threshold", current_thr))
            y_hat_temp = (p_spam >= temp_threshold).astype(int)
            prec_spam, rec_spam, f1_spam, sup_spam = precision_recall_fscore_support(
                y_true01, y_hat_temp, average="binary", zero_division=0
            )
            y_true_safe = 1 - y_true01
            y_hat_safe = 1 - y_hat_temp
            prec_safe, rec_safe, f1_safe, sup_safe = precision_recall_fscore_support(
                y_true_safe, y_hat_safe, average="binary", zero_division=0
            )

            st.markdown("### Detailed metrics (at current threshold)")

            def _as_int(value, fallback):
                if value is None:
                    return int(fallback)
                try:
                    return int(value)
                except TypeError:
                    return int(fallback)

            spam_support = _as_int(sup_spam, np.sum(y_true01))
            safe_support = _as_int(sup_safe, np.sum(1 - y_true01))

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "class": "spam",
                            "precision": prec_spam,
                            "recall": rec_spam,
                            "f1": f1_spam,
                            "support": spam_support,
                        },
                        {
                            "class": "safe",
                            "precision": prec_safe,
                            "recall": rec_safe,
                            "f1": f1_safe,
                            "support": safe_support,
                        },
                    ]
                ).round(3),
                width="stretch",
                hide_index=True,
            )

            st.markdown("### Precision & Recall vs Threshold (validation)")
            fig = plot_threshold_curves(y_true01, p_spam)
            st.pyplot(fig)

            st.markdown("### Interpretability")
            try:
                if hasattr(ss["model"], "named_steps"):
                    clf = ss["model"].named_steps.get("clf")
                    vec = ss["model"].named_steps.get("tfidf")
                    if hasattr(clf, "coef_") and vec is not None:
                        vocab = np.array(vec.get_feature_names_out())
                        coefs = clf.coef_[0]
                        top_spam = vocab[np.argsort(coefs)[-10:]][::-1]
                        top_safe = vocab[np.argsort(coefs)[:10]]
                        col_i1, col_i2 = st.columns(2)
                        with col_i1:
                            st.write("Top signals → **Spam**")
                            st.write(", ".join(top_spam))
                        with col_i2:
                            st.write("Top signals → **Safe**")
                            st.write(", ".join(top_safe))
                    else:
                        st.caption("Coefficients unavailable for this classifier.")
                elif hasattr(ss["model"], "numeric_feature_coefs"):
                    coef_map = ss["model"].numeric_feature_coefs()
                    st.caption("Numeric feature weights (positive → Spam, negative → Safe):")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "feature": k,
                                    "weight_toward_spam": v,
                                }
                                for k, v in coef_map.items()
                            ]
                        ).sort_values("weight_toward_spam", ascending=False),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.caption("Interpretability: no compatible inspector for this model.")
            except Exception as e:
                st.caption(f"Interpretability view unavailable: {e}")

            st.markdown("### Governance & reproducibility")
            try:
                if len(cache) == 4:
                    n_tr, n_te = len(y_tr), len(y_te)
                else:
                    n_tr, n_te = len(y_tr), len(y_te)
                split = ss.get("train_params", {}).get("test_size", "—")
                seed = ss.get("train_params", {}).get("random_state", "—")
                ts = ss.get("eval_timestamp", "—")
                st.write(f"- Train set: {n_tr}  |  Test set: {n_te}  |  Hold-out fraction: {split}")
                st.write(f"- Random seed: {seed}")
                st.write(f"- Training time: {ts}")
                st.write(f"- Adopted threshold: {ss.get('threshold', 0.5):.2f}")
            except Exception:
                st.caption("Governance info unavailable.")


def render_classify_stage():

    stage = STAGE_BY_KEY["classify"]

    with section_surface():
        overview_col, guidance_col = st.columns([3, 2], gap="large")
        with overview_col:
            st.subheader(f"{stage.icon} {stage.title} — Run the spam detector")
            render_eu_ai_quote(
                "The EU AI Act says “an AI system infers, from the input it receives, how to generate outputs such as content, predictions, recommendations or decisions.”"
            )
            st.write(
                "In this step, the system takes each email (title + body) as **input** and produces an **output**: "
                "a **prediction** (*Spam* or *Safe*) with a confidence score. By default, it also gives a **recommendation** "
                "about where to place the email (Spam or Inbox)."
            )
        with guidance_col:
            st.markdown("### Operating tips")
            st.markdown(
                "- Monitor predictions before enabling full autonomy.\n"
                "- Keep an eye on confidence values to decide when to intervene."
            )

    with section_surface():
        st.markdown("### Autonomy")
        default_high_autonomy = ss.get("autonomy", AUTONOMY_LEVELS[0]).startswith("High")
        auto_col, explain_col = st.columns([2, 3], gap="large")
        with auto_col:
            use_high_autonomy = st.toggle(
                "High autonomy (auto-move emails)", value=default_high_autonomy, key="use_high_autonomy"
            )
        with explain_col:
            if use_high_autonomy:
                ss["autonomy"] = AUTONOMY_LEVELS[1]
                st.success("High autonomy ON — the system will **move** emails to Spam or Inbox automatically.")
            else:
                ss["autonomy"] = AUTONOMY_LEVELS[0]
                st.warning("High autonomy OFF — review recommendations before moving emails.")
        if not ss.get("model"):
            st.warning("Train a model first in the **Train** tab.")
            st.stop()

    st.markdown("### Incoming preview")
    if not ss.get("incoming"):
        st.caption("No incoming emails. Add or import more in **📊 Prepare Data**, or paste a custom email below.")
        with st.expander("Add a custom email to process"):
            title_val = st.text_input("Title", key="use_custom_title", placeholder="Subject…")
            body_val = st.text_area("Body", key="use_custom_body", height=100, placeholder="Email body…")
            if st.button("Add to incoming", key="btn_add_to_incoming"):
                if title_val.strip() or body_val.strip():
                    ss["incoming"].append({"title": title_val.strip(), "body": body_val.strip()})
                    st.success("Added to incoming.")
                    _append_audit("incoming_added", {"title": title_val[:64]})
                else:
                    st.warning("Please provide at least a title or a body.")
    else:
        preview_n = min(10, len(ss["incoming"]))
        preview_df = pd.DataFrame(ss["incoming"][:preview_n])
        if not preview_df.empty:
            subtitle = f"Showing the first {preview_n} incoming emails (unlabeled)."
            render_email_inbox_table(preview_df, title="Incoming emails", subtitle=subtitle, columns=["title", "body"])
        else:
            render_email_inbox_table(pd.DataFrame(), title="Incoming emails", subtitle="No incoming emails available.")

        if st.button(f"Process {preview_n} email(s)", type="primary", key="btn_process_batch"):
            batch = ss["incoming"][:preview_n]
            y_hat, p_spam, p_safe = _predict_proba_batch(ss["model"], batch)
            thr = float(ss.get("threshold", 0.5))

            batch_rows: list[dict] = []
            moved_spam = moved_inbox = 0
            for idx, item in enumerate(batch):
                pred = y_hat[idx]
                prob_spam = float(p_spam[idx])
                prob_safe = float(p_safe[idx]) if hasattr(p_safe, "__len__") else float(1.0 - prob_spam)
                action = "Recommend: Spam" if prob_spam >= thr else "Recommend: Inbox"
                routed_to = None
                if ss["use_high_autonomy"]:
                    routed_to = "Spam" if prob_spam >= thr else "Inbox"
                    mailbox_record = {
                        "title": item.get("title", ""),
                        "body": item.get("body", ""),
                        "pred": pred,
                        "p_spam": round(prob_spam, 3),
                    }
                    if routed_to == "Spam":
                        ss["mail_spam"].append(mailbox_record)
                        moved_spam += 1
                    else:
                        ss["mail_inbox"].append(mailbox_record)
                        moved_inbox += 1
                    action = f"Moved: {routed_to}"
                row = {
                    "title": item.get("title", ""),
                    "body": item.get("body", ""),
                    "pred": pred,
                    "p_spam": round(prob_spam, 3),
                    "p_safe": round(prob_safe, 3),
                    "action": action,
                    "routed_to": routed_to,
                }
                batch_rows.append(row)

            ss["use_batch_results"] = batch_rows
            ss["incoming"] = ss["incoming"][preview_n:]
            if ss["use_high_autonomy"]:
                st.success(
                    f"Processed {preview_n} emails — decisions applied (Inbox: {moved_inbox}, Spam: {moved_spam})."
                )
                _append_audit(
                    "batch_processed_auto", {"n": preview_n, "inbox": moved_inbox, "spam": moved_spam}
                )
            else:
                st.info(f"Processed {preview_n} emails — recommendations ready.")
                _append_audit("batch_processed_reco", {"n": preview_n})

    if ss.get("use_batch_results"):
        with section_surface():
            st.markdown("### Results")
            df_res = pd.DataFrame(ss["use_batch_results"])
            show_cols = ["title", "pred", "p_spam", "action", "routed_to"]
            existing_cols = [col for col in show_cols if col in df_res.columns]
            display_df = df_res[existing_cols].rename(
                columns={"pred": "Prediction", "p_spam": "P(spam)", "action": "Action", "routed_to": "Routed"}
            )
            render_email_inbox_table(display_df, title="Batch results", subtitle="Predictions and actions just taken.")
            st.caption(
                "Each row shows the predicted label, confidence (P(spam)), and the recommendation or action taken."
            )

        nerd_mode_enabled = render_nerd_mode_toggle(
            key="nerd_mode_use",
            title="Nerd Mode — details for this batch",
            description="Inspect raw probabilities, distributions, and the session audit trail.",
            icon="🔬",
        )
        if nerd_mode_enabled:
            df_res = pd.DataFrame(ss["use_batch_results"])
            with section_surface():
                st.markdown("### Nerd Mode — batch diagnostics")
                col_nm1, col_nm2 = st.columns([2, 1])
                with col_nm1:
                    st.markdown("**Raw probabilities (per email)**")
                    detail_cols = ["title", "p_spam", "p_safe", "pred", "action", "routed_to"]
                    det_existing = [col for col in detail_cols if col in df_res.columns]
                    st.dataframe(df_res[det_existing], width="stretch", hide_index=True)
                with col_nm2:
                    st.markdown("**Batch metrics**")
                    n_items = len(df_res)
                    mean_conf = float(df_res["p_spam"].mean()) if "p_spam" in df_res else 0.0
                    n_spam = int((df_res["pred"] == "spam").sum()) if "pred" in df_res else 0
                    n_safe = n_items - n_spam
                    st.write(f"- Items: {n_items}")
                    st.write(f"- Predicted Spam: {n_spam} | Safe: {n_safe}")
                    st.write(f"- Mean P(spam): {mean_conf:.2f}")

                    if "p_spam" in df_res:
                        fig, ax = plt.subplots()
                        ax.hist(df_res["p_spam"], bins=10)
                        ax.set_xlabel("P(spam)")
                        ax.set_ylabel("Count")
                        ax.set_title("Spam score distribution")
                        st.pyplot(fig)

                st.markdown("**Per-email cues (if available)**")
                st.caption(
                    "If your model exposes feature weights or signals, show a brief ‘why’ per email here."
                )
                st.info(
                    "Tip: reuse your Train/Evaluate interpretability hooks to display top words or numeric feature weights for the selected email."
                )

            with section_surface():
                st.markdown("### Audit trail (this session)")
                if ss.get("use_audit_log"):
                    st.dataframe(pd.DataFrame(ss["use_audit_log"]), width="stretch", hide_index=True)
                else:
                    st.caption("No events recorded yet.")

            exp_df = _export_batch_df(ss["use_batch_results"])
            csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
            json_bytes = json.dumps(ss["use_batch_results"], ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                "⬇️ Download results (CSV)", data=csv_bytes, file_name="batch_results.csv", mime="text/csv"
            )
            st.download_button(
                "⬇️ Download results (JSON)", data=json_bytes, file_name="batch_results.json", mime="application/json"
            )

    st.markdown("### Adaptiveness — learn from your corrections")
    render_eu_ai_quote(
        "The EU AI Act says “AI systems may exhibit adaptiveness.” Enable adaptiveness to confirm or correct results; the model can retrain on your feedback."
    )
    def _handle_stage_adaptive_change() -> None:
        _set_adaptive_state(ss.get("adaptive_stage", ss.get("adaptive", False)), source="stage")

    st.toggle(
        "Enable adaptiveness (learn from feedback)",
        value=bool(ss.get("adaptive", False)),
        key="adaptive_stage",
        on_change=_handle_stage_adaptive_change,
    )
    use_adaptiveness = bool(ss.get("adaptive", False))

    if use_adaptiveness and ss.get("use_batch_results"):
        st.markdown("#### Review and give feedback")
        for i, row in enumerate(ss["use_batch_results"]):
            with st.container(border=True):
                st.markdown(f"**Title:** {row.get('title', '')}")
                pspam_value = row.get("p_spam")
                if isinstance(pspam_value, (int, float)):
                    pspam_text = f"{pspam_value:.2f}"
                else:
                    pspam_text = pspam_value
                action_display = row.get("action", "")
                pred_display = row.get("pred", "")
                st.markdown(
                    f"**Predicted:** {pred_display}  •  **P(spam):** {pspam_text}  •  **Action:** {action_display}"
                )
                col_a, col_b, col_c = st.columns(3)
                if col_a.button("Confirm", key=f"use_confirm_{i}"):
                    _append_audit("confirm_label", {"i": i, "pred": pred_display})
                    st.toast("Thanks — recorded your confirmation.", icon="✅")
                if col_b.button("Correct → Spam", key=f"use_correct_spam_{i}"):
                    ss["labeled"].append(
                        {"title": row.get("title", ""), "body": row.get("body", ""), "label": "spam"}
                    )
                    _append_audit("correct_label", {"i": i, "new": "spam"})
                    st.toast("Recorded correction → Spam.", icon="✍️")
                if col_c.button("Correct → Safe", key=f"use_correct_safe_{i}"):
                    ss["labeled"].append(
                        {"title": row.get("title", ""), "body": row.get("body", ""), "label": "safe"}
                    )
                    _append_audit("correct_label", {"i": i, "new": "safe"})
                    st.toast("Recorded correction → Safe.", icon="✍️")

        if st.button("🔁 Retrain now with feedback", key="btn_retrain_feedback"):
            df_all = pd.DataFrame(ss["labeled"])
            if not df_all.empty and len(df_all["label"].unique()) >= 2:
                params = ss.get("train_params", {})
                test_size = float(params.get("test_size", 0.30))
                random_state = int(params.get("random_state", 42))
                max_iter = int(params.get("max_iter", 1000))
                C_value = float(params.get("C", 1.0))

                titles = df_all["title"].fillna("").tolist()
                bodies = df_all["body"].fillna("").tolist()
                labels = df_all["label"].tolist()
                X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te = train_test_split(
                    titles,
                    bodies,
                    labels,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=labels,
                )

                model = HybridEmbedFeatsLogReg()
                try:
                    model.lr.set_params(max_iter=max_iter, C=C_value)
                except Exception:
                    pass
                model = model.fit(X_tr_t, X_tr_b, y_tr)
                model.apply_numeric_adjustments(ss["numeric_adjustments"])
                ss["model"] = model
                ss["split_cache"] = (X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te)
                ss["eval_timestamp"] = datetime.now().isoformat(timespec="seconds")
                ss["eval_temp_threshold"] = float(ss.get("threshold", 0.6))
                st.success("Adaptive learning: model retrained with your feedback.")
                _append_audit("retrain_feedback", {"n_labeled": len(df_all)})
            else:
                st.warning("Need both classes (spam & safe) in labeled data to retrain.")

    st.markdown("### 📥 Mailboxes")
    inbox_tab, spam_tab = st.tabs(
        [
            f"Inbox (safe) — {len(ss['mail_inbox'])}",
            f"Spam — {len(ss['mail_spam'])}",
        ]
    )
    with inbox_tab:
        render_mailbox_panel(
            ss.get("mail_inbox"),
            mailbox_title="Inbox (safe)",
            filled_subtitle="Messages the system kept in your inbox.",
            empty_subtitle="Inbox is empty so far.",
        )
    with spam_tab:
        render_mailbox_panel(
            ss.get("mail_spam"),
            mailbox_title="Spam",
            filled_subtitle="What the system routed away from the inbox.",
            empty_subtitle="No emails have been routed to spam yet.",
        )

    st.caption(
        f"Threshold used for routing: **{float(ss.get('threshold', 0.5)):.2f}**. "
        "Adjust it in **🧪 Evaluate** to change how cautious/aggressive the system is."
    )

def render_model_card_stage():


    with section_surface():
        st.subheader("Model Card — transparency")
        guidance_popover("Transparency", """
Model cards summarize intended purpose, data, metrics, autonomy & adaptiveness settings.
They help teams reason about risks and the appropriate oversight controls.
""")
        algo = "Sentence embeddings (MiniLM) + standardized numeric cues + Logistic Regression"
        n_samples = len(ss["labeled"])
        labels_present = sorted({row["label"] for row in ss["labeled"]}) if ss["labeled"] else []
        metrics_text = ""
        holdout_n = 0
        if ss.get("model") and ss.get("split_cache"):
            _, X_te_t, _, X_te_b, _, y_te = ss["split_cache"]
            y_pred = ss["model"].predict(X_te_t, X_te_b)
            holdout_n = len(y_te)
            metrics_text = f"Accuracy on hold‑out: {accuracy_score(y_te, y_pred):.2%} (n={holdout_n})"
        card_md = f"""
# Model Card — demistifAI (Spam Detector)
**Intended purpose**: Educational demo to illustrate the AI Act definition of an **AI system** via a spam classifier.

**Algorithm**: {algo}
**Features**: Sentence embeddings (MiniLM) concatenated with small, interpretable numeric features:
- num_links_external, has_suspicious_tld, punct_burst_ratio, money_symbol_count, urgency_terms_count.
These are standardized and combined with the embedding before a linear classifier.

**Classes**: spam, safe
**Dataset size**: {n_samples} labeled examples
**Classes present**: {', '.join(labels_present) if labels_present else '[not trained]'}

**Key metrics**: {metrics_text or 'Train a model to populate metrics.'}

**Autonomy**: {ss['autonomy']} (threshold={ss['threshold']:.2f})
**Adaptiveness**: {'Enabled' if ss['adaptive'] else 'Disabled'} (learn from user corrections).

**Data**: user-augmented seed set (title + body); session-only.
**Known limitations**: tiny datasets; vocabulary sensitivity; no MIME/URL/metadata features.

**AI Act mapping**
- **Machine-based system**: Streamlit app (software) running on cloud runtime (hardware).
- **Inference**: model learns patterns from labeled examples.
- **Output generation**: predictions + confidence; used to recommend/route emails.
    - **Varying autonomy**: user selects autonomy level; at high autonomy, the system acts.
- **Adaptiveness**: optional feedback loop that updates the model.
"""
        content_col, highlight_col = st.columns([3, 2], gap="large")
        with content_col:
            st.markdown(card_md)
            download_text(card_md, "model_card.md", "Download model_card.md")
        with highlight_col:
            st.markdown(
                """
                <div class="info-metric-grid">
                    <div class="info-metric-card">
                        <div class="label">Labeled dataset</div>
                        <div class="value">{samples}</div>
                    </div>
                    <div class="info-metric-card">
                        <div class="label">Hold-out size</div>
                        <div class="value">{holdout}</div>
                    </div>
                    <div class="info-metric-card">
                        <div class="label">Autonomy</div>
                        <div class="value">{autonomy}</div>
                    </div>
                    <div class="info-metric-card">
                        <div class="label">Adaptiveness</div>
                        <div class="value">{adaptive}</div>
                    </div>
                </div>
                """.format(
                    samples=n_samples,
                    holdout=holdout_n or "—",
                    autonomy=html.escape(ss.get("autonomy", AUTONOMY_LEVELS[0])),
                    adaptive="On" if ss.get("adaptive") else "Off",
                ),
                unsafe_allow_html=True,
            )


STAGE_RENDERERS = {
    'intro': render_intro_stage,
    'overview': render_overview_stage,
    'data': render_data_stage,
    'train': render_train_stage,
    'evaluate': render_evaluate_stage,
    'classify': render_classify_stage,
    'model_card': render_model_card_stage,
}


active_stage = ss['active_stage']
render_stage_cards(active_stage, variant='progress')
renderer = STAGE_RENDERERS.get(active_stage, render_intro_stage)
renderer()
render_stage_navigation_controls(active_stage)

st.markdown("---")
st.caption("© demistifAI — Built for interactive learning and governance discussions.")
