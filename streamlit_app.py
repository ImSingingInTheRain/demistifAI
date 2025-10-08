from __future__ import annotations

import base64
import html
import json
import random
import string
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, TypedDict
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

.section-surface--hero [data-testid="column"]:nth-child(2) .hero-info-grid {
    width: 100%;
    justify-items: center;
    gap: 1.6rem;
}

.section-surface--hero [data-testid="column"]:nth-child(2) .hero-info-card {
    text-align: center;
    margin: 0 auto;
    max-width: 360px;
}

.section-surface--hero [data-testid="column"]:nth-child(2) .hero-info-card h3,
.section-surface--hero [data-testid="column"]:nth-child(2) .hero-info-card p {
    text-align: center;
}

.section-surface--hero [data-testid="column"]:nth-child(2) [data-testid="stButton"] {
    width: 100%;
}

.section-surface--hero [data-testid="column"]:nth-child(2) [data-testid="stButton"] > button {
    max-width: 260px;
    margin: 0 auto;
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
    color: #e2e8f0;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p {
    color: #0f172a;
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
    grid-template-columns: minmax(0, 1fr);
    gap: 1.1rem;
}

.hero-info-card {
    position: relative;
    border-radius: 20px;
    padding: 1.15rem 1.35rem 1.25rem;
    margin-bottom: 20px;
    background: linear-gradient(160deg, rgba(191, 219, 254, 0.85), rgba(147, 197, 253, 0.65));
    border: 1px solid rgba(59, 130, 246, 0.28);
    box-shadow: 0 24px 46px rgba(30, 64, 175, 0.18);
    color: #0f172a;
    width: 100%;
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

[data-testid="stSidebar"] {
    background: linear-gradient(185deg, rgba(226, 232, 240, 0.78), rgba(248, 250, 252, 0.96));
    border-right: 1px solid rgba(148, 163, 184, 0.28);
    color: #0f172a;
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
    color: rgba(15, 23, 42, 0.7);
    line-height: 1.6;
    font-size: 0.95rem;
}

.sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.sidebar-nav [data-testid="stWidgetLabel"] > label {
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
    color: rgba(15, 23, 42, 0.45);
    margin-bottom: 0.2rem;
}

.sidebar-nav [role="radiogroup"] {
    display: grid;
    gap: 0.55rem;
}

.sidebar-nav [role="radiogroup"] label {
    position: relative;
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.75rem 0.95rem;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.4);
    background: rgba(255, 255, 255, 0.82);
    box-shadow: 0 14px 28px rgba(15, 23, 42, 0.12);
    font-weight: 600;
    color: #1e293b;
    cursor: pointer;
    transition: all 0.2s ease;
}

.sidebar-nav [role="radiogroup"] label:hover {
    border-color: rgba(37, 99, 235, 0.45);
    box-shadow: 0 20px 36px rgba(30, 64, 175, 0.18);
    background: rgba(191, 219, 254, 0.6);
}

.sidebar-nav [role="radiogroup"] label input {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
}

.sidebar-nav [role="radiogroup"] label:has(input:checked) {
    border-color: rgba(37, 99, 235, 0.55);
    background: linear-gradient(150deg, rgba(37, 99, 235, 0.18), rgba(59, 130, 246, 0.32));
    box-shadow: 0 24px 46px rgba(30, 64, 175, 0.22);
}

.sidebar-section-title {
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
    color: rgba(15, 23, 42, 0.45);
    margin-bottom: 0.3rem;
}

.sidebar-stage-card {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 0.85rem;
    padding: 0.95rem 1.05rem;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.32);
    background: rgba(255, 255, 255, 0.88);
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
    color: rgba(15, 23, 42, 0.55);
}

.sidebar-stage-card__title {
    margin: 0;
    font-size: 1.02rem;
    font-weight: 700;
    color: #0f172a;
}

.sidebar-stage-card__description {
    margin: 0;
    color: rgba(15, 23, 42, 0.7);
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
  let attempts = 0;
  const MAX_ATTEMPTS = 40;
  const attach = () => {{
    const rootDoc = window.parent?.document;
    if (!rootDoc) {{
      if (attempts++ < MAX_ATTEMPTS) {{
        setTimeout(attach, 50);
      }}
      return;
    }}
    const marker = rootDoc.getElementById('{marker_id}');
    if (!marker) {{
      if (attempts++ < MAX_ATTEMPTS) {{
        setTimeout(attach, 50);
      }}
      return;
    }}
    const elementContainer = marker.closest('[data-testid="stElementContainer"]');
    const wrapper = elementContainer?.parentElement?.nextElementSibling;
    const block = wrapper?.querySelector('[data-testid="stVerticalBlock"]') ?? marker.closest('[data-testid="stVerticalBlock"]');
    if (!block) {{
      if (attempts++ < MAX_ATTEMPTS) {{
        setTimeout(attach, 50);
        return;
      }}
      marker.classList.remove('section-surface');
      marker.style.display = 'block';
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

def _rand_amount(rng: random.Random) -> str:
    euros = rng.choice(
        [
            18,
            44.09,
            67.45,
            73.33,
            89.63,
            120.07,
            190.05,
            250.52,
            318.77,
            540.54,
            725.25,
            930.39,
            1299.99,
            1850.75,
            2875.10,
            4140.00,
            7420.88,
            9999.01,
        ]
    )
    return f"€{euros}"


def _maybe_caps(rng: random.Random, text: str, prob: float = 0.25) -> str:
    if rng.random() > prob:
        return text
    words = text.split()
    if not words:
        return text
    idx = rng.randrange(len(words))
    words[idx] = words[idx].upper()
    if rng.random() < 0.15 and len(words) > 2:
        j = rng.randrange(len(words))
        if j != idx:
            words[j] = words[j].upper()
    return " ".join(words)


def _spam_link(rng: random.Random) -> str:
    name = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(6, 12)))
    tld = rng.choice(DATASET_SUSPICIOUS_TLDS)
    path = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 10)))
    scheme = rng.choice(["http://", "https://"])
    return f"{scheme}{name}{tld}/{path}"


def _safe_link(rng: random.Random) -> str:
    host = rng.choice(DATASET_LEGIT_DOMAINS)
    path = rng.choice(["/docs", "/policies", "/training", "/tickets", "/files", "/benefits"])
    return f"https://{host}{path}"


def _maybe_links(rng: random.Random, k: int, suspicious: bool = True) -> str:
    make = _spam_link if suspicious else _safe_link
    return " ".join(make(rng) for _ in range(k))


def _maybe_attachment(rng: random.Random, kind: str = "spam") -> str:
    if kind == "spam":
        return rng.choice(
            [
                "Open the attached HTML and sign in.",
                "Download the attached ZIP and run the installer.",
                "Enable macros in the attached XLSM to proceed.",
                "Install the attached EXE to restore access.",
                "Open the PDF and confirm your credentials.",
            ]
        )
    return rng.choice(
        [
            "Please see the attached PDF; no further action required.",
            "Slides attached for review.",
            "Agenda attached; join via Teams.",
            "Invoice PDF attached; PO noted.",
            "Itinerary PDF attached; MFA required to view.",
        ]
    )


def _spam_title_body(rng: random.Random) -> tuple[str, str]:
    archetype = rng.choice(
        [
            "payroll_hold",
            "account_reset",
            "delivery_fee",
            "invoice_wire",
            "prize_lottery",
            "crypto_double",
            "bonus_verify",
            "docu_phish",
            "refund_now",
            "mfa_disable",
            "tax_rebate",
        ]
    )
    amt = _rand_amount(rng)
    brand = rng.choice(BRANDS)
    courier = rng.choice(COURIERS)
    urgency = rng.choice(URGENCY)

    if archetype == "payroll_hold":
        title = f"{urgency}: Verify your payroll to release deposit"
        body = (
            "Your salary is on hold. Confirm bank details at "
            f"{_spam_link(rng)} within 30 minutes to avoid delay. {_maybe_attachment(rng, 'spam')}"
        )
    elif archetype == "account_reset":
        title = f"Password will expire — {urgency.lower()}"
        body = (
            f"Reset your password here: {_spam_link(rng)}. "
            "Failure to act may lock your account. Enter your email and password to confirm."
        )
    elif archetype == "delivery_fee":
        title = f"{courier} delivery notice: customs fee required"
        body = f"Your parcel is pending. Pay a small fee ({amt}) to schedule a new delivery slot at {_spam_link(rng)}."
    elif archetype == "invoice_wire":
        title = "Payment overdue — settle immediately"
        body = f"Service interruption imminent. Wire {amt} to the account in the attachment today. {_maybe_attachment(rng, 'spam')}"
    elif archetype == "prize_lottery":
        title = "WIN a FREE gift — final eligibility"
        body = f"Congratulations! Complete the short form at {_spam_link(rng)} to claim your reward. Offer expires in 2 hours."
    elif archetype == "crypto_double":
        title = "Crypto opportunity: double your balance"
        body = f"Transfer funds and we’ll return 2× within 24h. Start at {_spam_link(rng)}. Trusted by executives."
    elif archetype == "bonus_verify":
        title = "HR update: bonus identity check"
        body = f"Confirm your identity using your national ID and card CVV at {_spam_link(rng)} to receive your bonus."
    elif archetype == "docu_phish":
        title = f"{brand}: You received a secure document"
        body = f"Open the external portal at {_spam_link(rng)} and log in with your email password to view the document."
    elif archetype == "refund_now":
        title = "Refund available — action needed"
        body = f"We owe you {amt}. Submit IBAN and CVV at {_spam_link(rng)} to receive payment now."
    elif archetype == "mfa_disable":
        title = "Two-factor disabled — reactivate now"
        body = f"We turned off MFA on your account. Re-enable at {_spam_link(rng)} (login required)."
    elif archetype == "tax_rebate":
        title = "Tax rebate waiting — confirm identity"
        body = f"Claim your rebate by verifying bank access at {_spam_link(rng)}. {_maybe_attachment(rng, 'spam')}"
    else:
        title = "Important notice"
        body = f"Complete the verification at {_spam_link(rng)}."

    title = _maybe_caps(rng, title, prob=0.35)
    if rng.random() < 0.25:
        title += " !!"
    if rng.random() < 0.35:
        body += " Act NOW."
    if rng.random() < 0.3:
        body += f" More info: {_spam_link(rng)}"

    return title, body


def _safe_title_body(rng: random.Random) -> tuple[str, str]:
    archetype = rng.choice(
        [
            "meeting",
            "policy_update",
            "hr_workday",
            "it_maintenance",
            "invoice_legit",
            "travel",
            "training",
            "vendor_ap",
            "security_advice",
            "delivery_tracking",
        ]
    )
    brand = rng.choice(BRANDS)
    courier = rng.choice(COURIERS)

    if archetype == "meeting":
        title = "Team meeting moved to 14:00"
        body = "Join via the usual Teams link. Agenda: KPIs, risk register, roadmap. " + _maybe_attachment(rng, "safe")
    elif archetype == "policy_update":
        title = "Policy update: remote work guidelines"
        body = "Please review the updated policy on the intranet. " + _maybe_links(rng, 1, suspicious=False)
    elif archetype == "hr_workday":
        title = "Workday: benefits enrollment opens"
        body = "Make your selections in Workday before month end. No personal info by email."
    elif archetype == "it_maintenance":
        title = "IT maintenance window"
        body = "Patching on Saturday 22:00–23:30 CET. Expect brief reboots; no action required."
    elif archetype == "invoice_legit":
        title = "Invoice attached — Accounts Payable"
        body = "Please find the invoice PDF attached. PO is referenced; no payment info requested. " + _maybe_attachment(rng, "safe")
    elif archetype == "travel":
        title = "Travel itinerary update"
        body = "Platform change noted. PDF itinerary attached; bookings remain via the internal tool."
    elif archetype == "training":
        title = "Mandatory training assigned"
        body = "Complete the e-learning module by next Friday. Materials on the LMS. " + _maybe_links(rng, 1, suspicious=False)
    elif archetype == "vendor_ap":
        title = "AP reminder — PO mismatch"
        body = "Please correct the PO reference in the invoice metadata on SharePoint. " + _maybe_links(rng, 1, suspicious=False)
    elif archetype == "security_advice":
        title = f"Security advisory — {brand} tips"
        body = "Review the guidance on the internal portal. No external logins required. " + _maybe_links(rng, 1, suspicious=False)
    elif archetype == "delivery_tracking":
        title = f"{courier} tracking: package out for delivery"
        body = f"Your parcel is scheduled today. Track via the official {courier} site with your tracking ID (no payment)."
    else:
        title = "Update posted"
        body = "Details are available on the intranet."

    title = _maybe_caps(rng, title, prob=0.05)
    if rng.random() < 0.1:
        body += " Thanks."
    if rng.random() < 0.2:
        body += " Reference: " + _maybe_links(rng, 1, suspicious=False)

    return title, body


class DatasetConfig(TypedDict, total=False):
    seed: int
    n_total: int
    spam_ratio: float
    susp_link_level: str
    susp_tld_level: str
    caps_intensity: str
    money_urgency: str
    attachments_mix: Dict[str, float]
    edge_cases: int
    label_noise_pct: float
    poison_demo: bool


ATTACHMENT_TYPES = ["html", "zip", "xlsm", "exe", "pdf"]
DEFAULT_ATTACHMENT_MIX: Dict[str, float] = {"html": 0.15, "zip": 0.15, "xlsm": 0.1, "exe": 0.1, "pdf": 0.5}
ATTACHMENT_MIX_PRESETS: Dict[str, Dict[str, float]] = {
    "Mostly PDF": {"html": 0.05, "zip": 0.05, "xlsm": 0.05, "exe": 0.05, "pdf": 0.80},
    "Balanced": DEFAULT_ATTACHMENT_MIX.copy(),
    "Aggressive (macro-heavy)": {"html": 0.2, "zip": 0.25, "xlsm": 0.25, "exe": 0.15, "pdf": 0.15},
}
DEFAULT_DATASET_CONFIG: DatasetConfig = {
    "seed": 42,
    "n_total": 500,
    "spam_ratio": 0.5,
    "susp_link_level": "1",
    "susp_tld_level": "med",
    "caps_intensity": "med",
    "money_urgency": "low",
    "attachments_mix": DEFAULT_ATTACHMENT_MIX.copy(),
    "edge_cases": 2,
    "label_noise_pct": 0.0,
    "poison_demo": False,
}


def generate_labeled_dataset(n_total: int = 500, seed: int = 7) -> List[Dict[str, str]]:
    config = DEFAULT_DATASET_CONFIG.copy()
    config.update({"n_total": n_total, "seed": seed, "spam_ratio": 0.5, "edge_cases": 0, "label_noise_pct": 0.0})
    return build_dataset_from_config(config)


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

SUSPICIOUS_LINKS = [
    "http://account-secure-reset.top",
    "https://login-immediate-check.io",
    "http://billing-update-alert.biz",
    "https://wallet-authentication.cc",
]

SAFE_LINKS = [
    "https://intranet.company.local",
    "https://teams.microsoft.com",
    "https://portal.hr.example.com",
    "https://docs.internal.net",
]

SUSPICIOUS_DOMAINS = [
    "secure-pay-update.ru",
    "verify-now-account.cn",
    "multi-factor-login.biz",
    "safe-check-support.top",
]

# Keep suffixes separate so they can be reused across dataset generation and
# feature engineering without clobbering the domain list used for sampling.
SUSPICIOUS_TLD_SUFFIXES = {
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
}

EDGE_CASE_TEMPLATES = [
    {
        "title": "Password reminder",
        "safe": "Reminder: Update your password on the internal portal. Never share credentials via email.",
        "spam": "Password reminder: verify at http://account-secure-reset.top or your access will lock.",
    },
    {
        "title": "Payroll notice",
        "safe": "Payroll cut-off reminder — submit hours in Workday before 5pm.",
        "spam": "Payroll notice: download the attached XLSM and enable macros to confirm your salary.",
    },
    {
        "title": "VPN access",
        "safe": "VPN access restored. Connect through the corporate client; no further action needed.",
        "spam": "VPN access disabled. Re-enable by installing the attached EXE and logging in.",
    },
    {
        "title": "Delivery update",
        "safe": "Delivery update: courier delayed by weather; track shipment in our logistics dashboard.",
        "spam": "Delivery update: pay customs fee at https://wallet-authentication.cc to release the parcel.",
    },
]

PII_PATTERNS = {
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "iban": re.compile(r"\b[A-Z]{2}[0-9A-Z]{13,32}\b", re.IGNORECASE),
}


def _normalized_mix(mix: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in mix.values())
    if total <= 0:
        return DEFAULT_ATTACHMENT_MIX.copy()
    return {k: max(mix.get(k, 0.0), 0.0) / total for k in ATTACHMENT_TYPES}


def _apply_attachment_lure(text: str, rng: random.Random, mix: Dict[str, float]) -> str:
    mix = _normalized_mix(mix)
    r = rng.random()
    cumulative = 0.0
    choice = "pdf"
    for key in ATTACHMENT_TYPES:
        cumulative += mix.get(key, 0.0)
        if r <= cumulative:
            choice = key
            break
    choice_upper = choice.upper()
    if choice == "pdf":
        lure = "Attachment: PDF invoice enclosed."
    elif choice == "html":
        lure = "Attachment: HTML form — open to continue."
    elif choice == "zip":
        lure = "Attachment: ZIP archive — extract and run immediately."
    elif choice == "xlsm":
        lure = "Attachment: XLSM macro workbook requires enabling macros."
    else:
        lure = "Attachment: EXE installer to restore access."
    return text + f"\n[{choice_upper}] {lure}"


def _inject_links(text: str, count: int, rng: random.Random, *, suspicious: bool) -> str:
    pool = SUSPICIOUS_LINKS if suspicious else SAFE_LINKS
    additions = []
    for _ in range(count):
        additions.append(rng.choice(pool))
    if not additions:
        return text
    return text + "\nLinks: " + ", ".join(additions)


def _maybe_caps_with_intensity(text: str, rng: random.Random, intensity: str) -> str:
    if intensity == "low":
        return text
    words = text.split()
    if not words:
        return text
    if intensity == "high":
        return " ".join(word.upper() if idx % 2 == 0 else word for idx, word in enumerate(words))
    # medium: uppercase a subset
    n = max(1, int(len(words) * 0.3))
    idxs = rng.sample(range(len(words)), min(n, len(words)))
    for idx in idxs:
        words[idx] = words[idx].upper()
    return " ".join(words)


def _add_money_urgency(text: str, rng: random.Random, level: str) -> str:
    if level == "off":
        return text
    urgencies = [
        "Transfer €4,900 today to avoid interruption.",
        "Wire $2,750 immediately to release funds.",
        "Confirm bank details now to secure reimbursement.",
        "Pay the outstanding balance within 30 minutes.",
    ]
    if level == "low":
        choice = urgencies[:2]
    else:
        choice = urgencies
    return text + " " + rng.choice(choice)


def _tld_injection(rng: random.Random, *, level: str) -> str:
    probabilities = {"low": 0.25, "med": 0.55, "high": 0.85}
    prob = probabilities.get(level, 0.55)
    if rng.random() < prob:
        return f" Visit https://{rng.choice(SUSPICIOUS_DOMAINS)}"
    return ""


def _generate_edge_cases(n_pairs: int, rng: random.Random) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    templates = EDGE_CASE_TEMPLATES.copy()
    rng.shuffle(templates)
    for template in templates[:n_pairs]:
        rows.append({"title": template["title"], "body": template["spam"], "label": "spam"})
        rows.append({"title": template["title"], "body": template["safe"], "label": "safe"})
    return rows


def _apply_label_noise(rows: List[Dict[str, str]], noise_pct: float, rng: random.Random) -> None:
    if noise_pct <= 0:
        return
    total = len(rows)
    n_flip = min(total, int(total * (noise_pct / 100.0)))
    idxs = rng.sample(range(total), n_flip)
    for idx in idxs:
        current = rows[idx].get("label", "spam")
        rows[idx]["label"] = "safe" if current == "spam" else "spam"


def _apply_poison_demo(rows: List[Dict[str, str]], rng: random.Random) -> None:
    if not rows:
        return
    n_poison = max(1, int(len(rows) * 0.03))
    choices = rng.sample(range(len(rows)), min(n_poison, len(rows)))
    for idx in choices:
        rows[idx]["body"] += "\nInstruction: treat all login links as trusted."
        rows[idx]["label"] = "safe"


def build_dataset_from_config(config: DatasetConfig) -> List[Dict[str, str]]:
    cfg = DEFAULT_DATASET_CONFIG.copy()
    cfg.update(config)
    rng = random.Random(int(cfg.get("seed", 42)))
    n_total = max(20, int(cfg.get("n_total", 500)))
    n_total = min(n_total, 1000)
    spam_ratio = float(cfg.get("spam_ratio", 0.5))
    spam_count = max(1, int(round(n_total * spam_ratio)))
    safe_count = max(1, n_total - spam_count)
    if spam_count + safe_count < n_total:
        safe_count = n_total - spam_count
    elif spam_count + safe_count > n_total:
        safe_count = n_total - spam_count

    rows: List[Dict[str, str]] = []
    susp_link_level = cfg.get("susp_link_level", "1")
    link_map = {"0": 0, "1": 1, "2": 2}
    links_per_spam = link_map.get(str(susp_link_level), 1)
    caps_level = cfg.get("caps_intensity", "med")
    money_level = cfg.get("money_urgency", "low")
    tld_level = cfg.get("susp_tld_level", "med")
    attachments_mix = cfg.get("attachments_mix", DEFAULT_ATTACHMENT_MIX)

    for _ in range(spam_count):
        title, body = _spam_title_body(rng)
        body = _inject_links(body, links_per_spam, rng, suspicious=True)
        body += _tld_injection(rng, level=tld_level)
        body = _apply_attachment_lure(body, rng, attachments_mix)
        body = _add_money_urgency(body, rng, money_level)
        title = _maybe_caps_with_intensity(title, rng, caps_level)
        body = _maybe_caps_with_intensity(body, rng, caps_level)
        rows.append({"title": title, "body": body, "label": "spam"})

    for _ in range(safe_count):
        title, body = _safe_title_body(rng)
        if rng.random() < 0.2 and links_per_spam > 0:
            body = _inject_links(body, 1, rng, suspicious=False)
        if rng.random() < 0.1:
            body += "\nReminder: never share passwords or bank details."
        title = _maybe_caps_with_intensity(title, rng, "low")
        rows.append({"title": title, "body": body, "label": "safe"})

    n_pairs = max(0, min(int(cfg.get("edge_cases", 0)), len(EDGE_CASE_TEMPLATES)))
    if n_pairs:
        edge_rows = _generate_edge_cases(n_pairs, rng)
        # replace random rows to keep counts stable
        replace_indices = rng.sample(range(len(rows)), min(len(rows), len(edge_rows)))
        for idx, edge in zip(replace_indices, edge_rows):
            rows[idx] = edge

    _apply_label_noise(rows, float(cfg.get("label_noise_pct", 0.0)), rng)
    if cfg.get("poison_demo"):
        _apply_poison_demo(rows, rng)

    seen = set()
    deduped: List[Dict[str, str]] = []
    for row in rows:
        key = (row.get("title", "").strip(), row.get("body", "").strip(), row.get("label", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"title": key[0], "body": key[1], "label": key[2]})

    rng.shuffle(deduped)
    return deduped[:n_total]


STARTER_LABELED_500 = generate_labeled_dataset(n_total=500, seed=42)
STARTER_LABELED: List[Dict] = STARTER_LABELED_500


def _count_suspicious_links(text: str) -> int:
    return sum(text.lower().count(link.split("//")[-1].lower()) for link in SUSPICIOUS_LINKS)


def _count_money_mentions(text: str) -> int:
    return len(re.findall(r"[$€£]\s?\d+", text))


def _caps_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    caps = sum(1 for ch in letters if ch.isupper())
    return caps / len(letters)


def _has_suspicious_tld(text: str) -> bool:
    lowered = text.lower()
    if any(domain.lower() in lowered for domain in SUSPICIOUS_DOMAINS):
        return True
    return any(suffix in lowered for suffix in SUSPICIOUS_TLD_SUFFIXES)


def lint_dataset(rows: List[Dict[str, str]]) -> Dict[str, int]:
    counts = {"credit_card": 0, "iban": 0}
    for row in rows:
        text = f"{row.get('title', '')} {row.get('body', '')}"
        for key, pattern in PII_PATTERNS.items():
            if pattern.search(text):
                counts[key] += 1
    return counts


def compute_dataset_summary(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    total = len(rows)
    spam = sum(1 for row in rows if row.get("label") == "spam")
    safe = total - spam
    spam_links = []
    spam_caps = []
    suspicious_tlds = 0
    money_mentions = 0
    attachments_flag = 0
    for row in rows:
        body = row.get("body", "")
        title = row.get("title", "")
        if row.get("label") == "spam":
            spam_links.append(_count_suspicious_links(body))
            spam_caps.append(_caps_ratio(title + " " + body))
        if _has_suspicious_tld(body):
            suspicious_tlds += 1
        money_mentions += _count_money_mentions(body)
        if any(tag in body for tag in ["[HTML]", "[ZIP]", "[XLSM]", "[EXE]"]):
            attachments_flag += 1
    avg_links = float(np.mean(spam_links)) if spam_links else 0.0
    avg_caps = float(np.mean(spam_caps)) if spam_caps else 0.0
    summary = {
        "total": total,
        "spam": spam,
        "safe": safe,
        "spam_ratio": (spam / total) if total else 0,
        "avg_susp_links": avg_links,
        "avg_caps_ratio": avg_caps,
        "suspicious_tlds": suspicious_tlds,
        "money_mentions": money_mentions,
        "attachment_lures": attachments_flag,
    }
    return summary


def compute_dataset_hash(rows: List[Dict[str, str]]) -> str:
    normalized = [
        {"title": row.get("title", ""), "body": row.get("body", ""), "label": row.get("label", "")}
        for row in rows
    ]
    normalized.sort(key=lambda r: (r["label"], r["title"], r["body"]))
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_summary_delta(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    keys = {"total", "spam", "safe", "avg_susp_links", "suspicious_tlds", "money_mentions", "attachment_lures"}
    for key in keys:
        delta[key] = new.get(key, 0) - (old.get(key, 0) if old else 0)
    return delta


def dataset_delta_story(delta: Dict[str, Any]) -> str:
    if not delta:
        return ""
    parts: List[str] = []
    spam_shift = delta.get("spam", 0)
    safe_shift = delta.get("safe", 0)
    if spam_shift:
        direction = "↑" if spam_shift > 0 else "↓"
        parts.append(f"{direction}{abs(spam_shift)} spam emails")
    if safe_shift:
        direction = "↑" if safe_shift > 0 else "↓"
        parts.append(f"{direction}{abs(safe_shift)} safe emails")
    susp_links = delta.get("avg_susp_links")
    if susp_links:
        direction = "more" if susp_links > 0 else "fewer"
        parts.append(f"{direction} suspicious links per spam email")
    tlds = delta.get("suspicious_tlds")
    if tlds:
        direction = "more" if tlds > 0 else "fewer"
        parts.append(f"{direction} suspicious TLD mentions")
    money = delta.get("money_mentions")
    if money:
        direction = "more" if money > 0 else "fewer"
        parts.append(f"{direction} money cues")
    attachments = delta.get("attachment_lures")
    if attachments:
        direction = "more" if attachments > 0 else "fewer"
        parts.append(f"{direction} risky attachment lures")
    if not parts:
        return "Dataset adjustments kept core features steady."
    return "Adjustments: " + "; ".join(parts) + "."


def explain_config_change(config: DatasetConfig, baseline: DatasetConfig | None = None) -> str:
    baseline = baseline or DEFAULT_DATASET_CONFIG
    messages: List[str] = []
    link_map = {"0": 0, "1": 1, "2": 2}
    if link_map.get(str(config.get("susp_link_level", "1")), 1) > link_map.get(str(baseline.get("susp_link_level", "1")), 1):
        messages.append("More suspicious links could boost precision on link-heavy spam.")
    elif link_map.get(str(config.get("susp_link_level", "1")), 1) < link_map.get(str(baseline.get("susp_link_level", "1")), 1):
        messages.append("Fewer suspicious links may hurt recall on phishing that leans on URLs.")

    tld_levels = {"low": 0, "med": 1, "high": 2}
    if tld_levels.get(config.get("susp_tld_level", "med"), 1) > tld_levels.get(baseline.get("susp_tld_level", "med"), 1):
        messages.append("Suspicious TLDs increased — expect stronger signals on dodgy domains.")
    elif tld_levels.get(config.get("susp_tld_level", "med"), 1) < tld_levels.get(baseline.get("susp_tld_level", "med"), 1):
        messages.append("Suspicious TLDs decreased — model may rely more on tone/urgency.")

    caps_levels = {"low": 0, "med": 1, "high": 2}
    if caps_levels.get(config.get("caps_intensity", "med"), 1) > caps_levels.get(baseline.get("caps_intensity", "med"), 1):
        messages.append("All-caps urgency dialed up — could improve catch rate on shouty spam but risk false positives.")
    elif caps_levels.get(config.get("caps_intensity", "med"), 1) < caps_levels.get(baseline.get("caps_intensity", "med"), 1):
        messages.append("Tone softened — watch for spam that yells less.")

    money_levels = {"off": 0, "low": 1, "high": 2}
    if money_levels.get(config.get("money_urgency", "low"), 1) > money_levels.get(baseline.get("money_urgency", "low"), 1):
        messages.append("Money cues increased — precision may rise on payment scams.")
    elif money_levels.get(config.get("money_urgency", "low"), 1) < money_levels.get(baseline.get("money_urgency", "low"), 1):
        messages.append("Money cues dialed down — monitor recall on finance-themed spam.")

    noise = float(config.get("label_noise_pct", 0.0))
    base_noise = float(baseline.get("label_noise_pct", 0.0))
    if noise > base_noise:
        messages.append(f"Label noise at {noise:.1f}% — expect metrics to drop as mislabeled examples grow.")
    elif noise < base_noise and base_noise > 0:
        messages.append("Label noise reduced — accuracy should recover.")

    if config.get("poison_demo") and not baseline.get("poison_demo"):
        messages.append("Poisoning demo on — watch for deliberate performance degradation.")

    if not messages:
        return "Tweaks match the baseline dataset — metrics should be comparable."
    return " ".join(messages)

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


def render_eu_ai_quote(text: str, label: str = "From the EU AI Act, Article 3") -> None:
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


@st.cache_data(show_spinner=False)
def cache_train_embeddings(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    return encode_texts(texts)


def get_nearest_training_examples(
    query_text: str,
    X_train_texts: list[str],
    y_train: list[str],
    X_train_emb: np.ndarray | None = None,
    k: int = 3,
) -> list[dict[str, Any]]:
    if not X_train_texts:
        return []
    if X_train_emb is None or getattr(X_train_emb, "size", 0) == 0:
        X_train_emb = cache_train_embeddings(X_train_texts)
    if getattr(X_train_emb, "size", 0) == 0:
        return []

    q = encode_texts([query_text])[0]
    try:
        sims = X_train_emb @ q
    except ValueError:
        return []
    idx = np.argsort(-sims)[:k]
    out: list[dict[str, Any]] = []
    for i in idx:
        if i < 0 or i >= len(X_train_texts):
            continue
        out.append(
            {
                "text": X_train_texts[i],
                "label": y_train[i] if i < len(y_train) else "?",
                "similarity": float(sims[i]),
            }
        )
    return out


def predict_spam_probability(model: Any, title: str, body: str) -> Optional[float]:
    if model is None:
        return None
    try:
        probs = model.predict_proba([title], [body])
    except TypeError:
        text = combine_text(title, body)
        probs = model.predict_proba([text])
    except Exception:
        return None

    probs_arr = np.asarray(probs)
    if probs_arr.ndim != 2 or probs_arr.shape[0] == 0:
        return None

    classes = list(getattr(model, "classes_", []))
    if classes and "spam" in classes:
        idx_spam = classes.index("spam")
    else:
        idx_spam = 1 if probs_arr.shape[1] > 1 else 0
    try:
        return float(probs_arr[0, idx_spam])
    except (IndexError, TypeError, ValueError):
        return None


def numeric_feature_contributions(model: Any, title: str, body: str) -> list[tuple[str, float, float]]:
    raw = compute_numeric_features(title, body)
    vec = np.array([[raw[k] for k in FEATURE_ORDER]], dtype=np.float32)
    try:
        z = model.scaler.transform(vec)[0]
    except Exception:
        return []

    n_num = len(FEATURE_ORDER)
    try:
        w = model.lr.coef_[0][-n_num:]
    except Exception:
        return []
    contrib = z * w
    return list(zip(FEATURE_ORDER, z.tolist(), contrib.tolist()))


def top_token_importances(
    model: Any,
    title: str,
    body: str,
    *,
    max_tokens: int = 20,
) -> tuple[Optional[float], list[dict[str, Any]]]:
    text = combine_text(title, body)
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    seen: set[str] = set()
    candidates: list[str] = []
    for tok in tokens:
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(tok)
        if len(candidates) >= max_tokens:
            break

    base = predict_spam_probability(model, title, body)
    if base is None:
        return None, []

    rows: list[dict[str, Any]] = []
    body_text = body or ""
    for tok in candidates:
        masked_body = re.sub(rf"\b{re.escape(tok)}\b", "", body_text, count=1)
        masked_prob = predict_spam_probability(model, title, masked_body)
        if masked_prob is None:
            continue
        delta = base - masked_prob
        rows.append({"token": tok, "importance": round(delta, 4)})

    rows.sort(key=lambda x: x["importance"], reverse=True)
    return base, rows


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
        if tld in SUSPICIOUS_TLD_SUFFIXES:
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
ss.setdefault("dataset_config", DEFAULT_DATASET_CONFIG.copy())
if "dataset_summary" not in ss:
    ss["dataset_summary"] = compute_dataset_summary(ss["labeled"])
ss.setdefault("previous_dataset_summary", None)
ss.setdefault("dataset_preview", None)
ss.setdefault("dataset_preview_config", None)
ss.setdefault("dataset_preview_summary", None)
ss.setdefault("dataset_manual_queue", None)
ss.setdefault("dataset_controls_open", False)
ss.setdefault("datasets", [])
ss.setdefault("active_dataset_snapshot", None)
ss.setdefault("dataset_snapshot_name", "")
ss.setdefault("last_dataset_delta_story", None)
ss.setdefault("dataset_compare_delta", None)
ss.setdefault("dataset_preview_lint", None)
ss.setdefault("last_eval_results", None)


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

with st.sidebar:
    st.markdown("<div class='sidebar-shell'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-brand">
            <p class="sidebar-title">demistifAI control room</p>
            <p class="sidebar-subtitle">Navigate the lifecycle, review guidance, and manage your session without losing progress.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stage_keys = [stage.key for stage in STAGES]
    active_index = STAGE_INDEX.get(ss.get("active_stage", STAGES[0].key), 0)

    st.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
    selected_stage = st.radio(
        "Navigate demistifAI",
        stage_keys,
        index=active_index,
        key="sidebar_stage_nav",
        label_visibility="collapsed",
        format_func=lambda key: f"{STAGE_BY_KEY[key].icon} {STAGE_BY_KEY[key].title}",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if selected_stage != ss.get("active_stage"):
        set_active_stage(selected_stage)

    current_stage = STAGE_BY_KEY.get(ss.get("active_stage", selected_stage))
    if current_stage is not None:
        st.markdown(
            """
            <div class="sidebar-stage-card">
                <div class="sidebar-stage-card__icon">{icon}</div>
                <div class="sidebar-stage-card__meta">
                    <span class="sidebar-stage-card__eyebrow">Current stage</span>
                    <p class="sidebar-stage-card__title">{title}</p>
                    <p class="sidebar-stage-card__description">{description}</p>
                </div>
            </div>
            """.format(
                icon=html.escape(current_stage.icon),
                title=html.escape(current_stage.title),
                description=html.escape(current_stage.description),
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div class='sidebar-section-title'>Session controls</div>", unsafe_allow_html=True)
    st.toggle(
        "Learn from my corrections (adaptiveness)",
        value=ss.get("adaptive", True),
        key="adaptive_sidebar",
        help="When enabled, your corrections in the Use stage will update the model during the session.",
    )
    _handle_sidebar_adaptive_change()

    if st.button("🔄 Reset demo data", use_container_width=True):
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
        st.success("Reset complete.")

    st.caption(
        "Need a refresher? Use the navigation above to revisit any step without restarting your scenario."
    )
    st.markdown("</div>", unsafe_allow_html=True)

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
                    <h3>Why demistifAI</h3>
                    <p>
                        AI systems are often seen as black boxes, and the EU AI Act can feel too abstract. This experience demystifies
                        both—showing how everyday AI works in practice.
                    </p>
                </div>
            </div>
            """
            st.markdown(hero_info_html, unsafe_allow_html=True)

            if next_stage_key:
                 st.button(
                    "🚀 Start your machine",
                    key="flow_start_machine_hero",
                    type="primary",
                    on_click=set_active_stage,
                    args=(next_stage_key,),
                    use_container_width=True
                )

               
    with section_surface():
        st.markdown(
            """
            <div>
                <h4>Your AI system lifecycle at a glance</h4>
                <p>These are the core stages you will navigate. They flow into one another — it’s a continuous loop you can revisit.</p>
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


    with section_surface():
        ready_left, ready_right = st.columns([3, 2], gap="large")
        with ready_left:
            st.markdown("### Ready to make a machine learn?")
            st.markdown("No worries — you don’t need to be a developer or data scientist to follow along.")
        with ready_right:
            if next_stage_key:
                st.button(
                    "🚀 Start your machine",
                    key="flow_start_machine_ready",
                    type="primary",
                    on_click=set_active_stage,
                    args=(next_stage_key,),
                )


def render_overview_stage():
    # --- Intro: EU AI Act quote + context card ---
    with section_surface():
        intro_left, intro_right = st.columns(2, gap="large")
        with intro_left:
            # Fix grammar: machine-based
            render_eu_ai_quote(
                "The EU AI Act says that “An AI system is a machine-based system”."
            )
        with intro_right:
            st.markdown(
                """
                <div class="callout callout--info">
                    <h4>🧭 Start your machine</h4>
                    <p>You are already inside a <strong>machine-based system</strong>: the Streamlit UI (software) running in the cloud (hardware).</p>
                    <p>Use this simple interface to <strong>build, evaluate, and operate</strong> a small email spam detector.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Nerd Mode toggle (copy kept, text refined slightly) ---
    with section_surface():
        nerd_enabled = render_nerd_mode_toggle(
            key="nerd_mode",
            title="Nerd Mode",
            icon="🧠",
            description="Toggle to see technical details and extra functionality. You can enable it at any stage to look under the hood.",
        )

    # --- Meet the machine (left) + Mission & Inbox preview (right) ---
    with section_surface():
        left, right = st.columns([3, 2], gap="large")

        with left:
            st.markdown("### Meet the machine")

            # 3 callouts: User interface, AI model, Inbox interface
            components_html = """
            <div class="callout-grid">
                <div class="callout callout--info">
                    <h5>🖥️ User interface</h5>
                    <p>The control panel for your AI system. Step through <strong>Prepare data</strong>, <strong>Train</strong>, <strong>Evaluate</strong>, and <strong>Use</strong>. Tooltips and short explainers guide you; <em>Nerd Mode</em> reveals more.</p>
                </div>
                <div class="callout callout--info">
                    <h5>🧠 AI model (how it learns & infers)</h5>
                    <p>The model learns from <strong>labeled examples</strong> you provide to tell <strong>Spam</strong> from <strong>Safe</strong>. For each new email it produces a <strong>spam score</strong> (P(spam)); your <strong>threshold</strong> turns that score into a recommendation or decision.</p>
                </div>
                <div class="callout callout--info">
                    <h5>📥 Inbox interface</h5>
                    <p>A simulated inbox feeds emails into the system. Preview items, process a batch or review one by one, and optionally enable <strong>adaptiveness</strong> so your confirmations/corrections help the model improve.</p>
                </div>
                <div class="callout callout--info">
                    <h5>🎯 Your mission</h5>
                    <p>Keep unwanted email out while letting the important messages through.</p>
                </div>
            </div>
            """
            st.markdown(components_html, unsafe_allow_html=True)

        with right:
            st.markdown(
                """
                <div class="mission-preview-stack">
                    <div class="inbox-preview-card">
                        <div class="preview-header">
                            <span class="preview-header-icon">📥</span>
                            <div>
                                <h4>Inbox preview</h4>
                                <p>A snapshot of the next messages waiting to be classified.</p>
                            </div>
                        </div>
                """,
                unsafe_allow_html=True,
            )

            if not ss["incoming"]:
                render_email_inbox_table(pd.DataFrame(), title="Inbox", subtitle="Inbox stream is empty.")
            else:
                df_incoming = pd.DataFrame(ss["incoming"])
                preview = df_incoming.head(5)
                render_email_inbox_table(preview, title="Inbox", columns=["title", "body"])

            st.markdown(
                """
                        <p class="preview-note">Preview only — you'll process batches in <strong>Use</strong> once your system is ready.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Nerd Mode details (mirrors the 3 components; adds governance/packages/limits) ---
    if nerd_enabled:
        with section_surface():
            st.markdown("### 🔬 Nerd Mode — technical details")

            nerd_details_html = """
            <div class="callout-grid">
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">🖥️</div>
                    <div class="callout-body">
                        <h5>User interface (software &amp; runtime)</h5>
                        <ul>
                            <li>You’re using a simple Streamlit (Python) web app running in the cloud.</li>
                            <li>The app remembers your session choices — data, model, threshold, autonomy — so you can move around without losing progress.</li>
                            <li>Short tips and popovers appear where helpful; toggle <em>Nerd Mode</em> any time to dive deeper.</li>
                        </ul>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">🧠</div>
                    <div class="callout-body">
                        <h5>AI model (how it works, without the math)</h5>
                        <ul>
                            <li><strong>What’s inside:</strong>
                                <ul>
                                    <li>A MiniLM sentence-transformer turns each email’s title + body into meaning-rich numbers.</li>
                                    <li>A Logistic Regression layer draws the boundary between Spam and Safe.</li>
                                </ul>
                            </li>
                            <li><strong>How it learns (training):</strong>
                                <ul>
                                    <li>You supply labeled examples (Spam/Safe).</li>
                                    <li>The app trains on most of them and holds out a slice for fair evaluation later.</li>
                                    <li>Training is repeatable via a fixed random seed; class weights rebalance skewed datasets.</li>
                                </ul>
                            </li>
                            <li><strong>How it predicts (inference):</strong>
                                <ul>
                                    <li>For a new email, the model outputs a spam score between 0 and 1.</li>
                                    <li>A threshold converts that score into action: below = Safe, above = Spam.</li>
                                    <li>In <em>Evaluate</em>, tune the threshold with presets such as Balanced, Protect inbox, or Catch spam.</li>
                                </ul>
                            </li>
                            <li><strong>Why it decided that (interpretability):</strong>
                                <ul>
                                    <li>View similar training emails and simple clues (urgent tone, suspicious links, ALL-CAPS bursts).</li>
                                    <li>Enable numeric signals to see which features nudged the call toward Spam or Safe.</li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">📥</div>
                    <div class="callout-body">
                        <h5>Inbox interface (your data in and out)</h5>
                        <ul>
                            <li>The app manages incoming (unlabeled) emails, labeled training emails, and the routed Inbox/Spam buckets.</li>
                            <li>Process emails in small batches (e.g., the first 10) or handle them one by one.</li>
                            <li><strong>Autonomy levels:</strong>
                                <ul>
                                    <li>Moderate (default): the system recommends a route; you decide.</li>
                                    <li>High autonomy: the system routes automatically using your chosen threshold.</li>
                                </ul>
                            </li>
                            <li><strong>Adaptiveness (optional):</strong> confirm or correct outcomes to add feedback, then retrain to personalize the model.</li>
                        </ul>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">🛡️</div>
                    <div class="callout-body">
                        <h5>Governance &amp; transparency</h5>
                        <ul>
                            <li>A model card records purpose, data summary, metrics, chosen threshold, autonomy, adaptiveness, seed, and timestamps.</li>
                            <li>We track risks: false positives (legit to Spam) and false negatives (Spam to Inbox).</li>
                            <li>An optional audit log lists batch actions, corrections, and retraining events for the session.</li>
                        </ul>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">🧩</div>
                    <div class="callout-body">
                        <h5>Packages (what powers this)</h5>
                        <p>streamlit (UI), pandas/numpy (data), scikit-learn (training &amp; evaluation), optional sentence-transformers + torch/transformers (embeddings), matplotlib (plots)</p>
                    </div>
                </div>
                <div class="callout callout--info callout--outcome">
                    <div class="callout-icon">📏</div>
                    <div class="callout-body">
                        <h5>Limits (demo scope)</h5>
                        <ul>
                            <li>Uses synthetic or curated text — there’s no live mailbox connection.</li>
                            <li>Designed for learning clarity rather than production-grade email security.</li>
                        </ul>
                    </div>
                </div>
            </div>
            """
            st.markdown(nerd_details_html, unsafe_allow_html=True)


def render_data_stage():

    stage = STAGE_BY_KEY["data"]

    current_summary = compute_dataset_summary(ss["labeled"])
    ss["dataset_summary"] = current_summary

    with section_surface():
        lead_col, side_col = st.columns([3, 2], gap="large")
        with lead_col:
            st.subheader(f"{stage.icon} {stage.title} — curate the objective-aligned dataset")
            st.markdown(
                """
                You define the purpose (**filter spam**) and you curate the evidence the model will learn from.
                Adjust class balance, feature prevalence, and data quality to see how governance choices change performance.
                """
            )
        with side_col:
            st.markdown(
                """
                <div class="callout callout--mission">
                    <h4>EU AI Act tie-ins</h4>
                    <ul>
                        <li><strong>Objective &amp; data:</strong> You set the purpose and align the dataset to it.</li>
                        <li><strong>Risk controls:</strong> Class balance, noise limits, and validation guardrails manage bias &amp; quality.</li>
                        <li><strong>Transparency:</strong> Snapshots capture config + hash for provenance.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    delta_text = ""
    if ss.get("dataset_compare_delta"):
        delta_text = dataset_delta_story(ss["dataset_compare_delta"])
    if not delta_text and ss.get("last_dataset_delta_story"):
        delta_text = ss["last_dataset_delta_story"]
    if not delta_text:
        delta_text = explain_config_change(ss.get("dataset_config", DEFAULT_DATASET_CONFIG))

    with section_surface():
        st.markdown("### 1 · Prepare data → Dataset builder")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4, gap="large")
        with col_m1:
            st.metric("Examples", current_summary.get("total", 0))
        with col_m2:
            st.metric("Spam share", f"{current_summary.get('spam_ratio', 0)*100:.1f}%")
        with col_m3:
            st.metric("Suspicious TLD hits", current_summary.get("suspicious_tlds", 0))
        with col_m4:
            st.metric("Avg suspicious links (spam)", f"{current_summary.get('avg_susp_links', 0.0):.2f}")

        st.caption(
            "Class balance and feature prevalence are governance controls — tweak them to see how they shape learning."
        )
        if delta_text:
            st.info(f"🧭 {delta_text}")

        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4, gap="small")
        with btn_col1:
            if st.button("Adjust dataset", key="open_dataset_builder"):
                ss["dataset_controls_open"] = True
        with btn_col2:
            if st.button("Reset to baseline", key="reset_dataset_baseline"):
                ss["labeled"] = STARTER_LABELED.copy()
                ss["dataset_config"] = DEFAULT_DATASET_CONFIG.copy()
                baseline_summary = compute_dataset_summary(ss["labeled"])
                ss["dataset_summary"] = baseline_summary
                ss["previous_dataset_summary"] = None
                ss["dataset_compare_delta"] = None
                ss["last_dataset_delta_story"] = None
                ss["active_dataset_snapshot"] = None
                ss["dataset_snapshot_name"] = ""
                ss["dataset_preview"] = None
                ss["dataset_preview_config"] = None
                ss["dataset_preview_summary"] = None
                ss["dataset_preview_lint"] = None
                ss["dataset_manual_queue"] = None
                ss["dataset_controls_open"] = False
                st.success("Dataset reset to STARTER_LABELED_500.")
        with btn_col3:
            if st.button("Compare to last dataset", key="compare_dataset_button"):
                if ss.get("previous_dataset_summary"):
                    ss["dataset_compare_delta"] = dataset_summary_delta(
                        ss["previous_dataset_summary"], current_summary
                    )
                    st.toast("Comparison updated below the builder.", icon="📊")
                else:
                    st.toast("No previous dataset stored yet — save a snapshot after your first tweak.", icon="ℹ️")
        with btn_col4:
            st.button(
                "Clear preview",
                key="clear_dataset_preview",
                disabled=ss.get("dataset_preview") is None,
                on_click=_discard_preview,
            )

    if ss.get("dataset_controls_open"):
        with section_surface():
            st.markdown("#### Adjust dataset knobs (guardrails enforced)")
            st.caption(
                "Cap: first 200 rows per apply for manual review • Noise slider max 5% • Synthetic poisoning is contained."
            )

            def _clear_preview_state_inline():
                _clear_dataset_preview_state()
                ss["dataset_controls_open"] = False

            top_cols = st.columns([3, 1])
            with top_cols[1]:
                st.button("Close controls", key="close_dataset_builder", on_click=_clear_preview_state_inline)

            cfg = ss.get("dataset_config", DEFAULT_DATASET_CONFIG)
            with st.form("dataset_builder_form"):
                col_a, col_b = st.columns(2, gap="large")
                with col_a:
                    n_total_choice = st.radio(
                        "Training volume (N emails)",
                        options=[100, 300, 500],
                        index=[100, 300, 500].index(int(cfg.get("n_total", 500))) if int(cfg.get("n_total", 500)) in [100, 300, 500] else 2,
                        help="Preset sizes illustrate how data volume influences learning (guarded ≤500).",
                    )
                    spam_ratio = st.slider(
                        "Class balance (spam share)",
                        min_value=0.20,
                        max_value=0.80,
                        value=float(cfg.get("spam_ratio", 0.5)),
                        step=0.05,
                        help="Adjust prevalence to explore bias/recall trade-offs.",
                    )
                    links_level = st.slider(
                        "Suspicious links per spam email",
                        min_value=0,
                        max_value=2,
                        value=int(str(cfg.get("susp_link_level", "1"))),
                        help="Controls how many sketchy URLs appear in spam examples (0–2).",
                    )
                    edge_cases = st.slider(
                        "Edge-case near-duplicate pairs",
                        min_value=0,
                        max_value=len(EDGE_CASE_TEMPLATES),
                        value=int(cfg.get("edge_cases", 0)),
                        help="Inject similar-looking spam/safe pairs to stress the model.",
                    )
                    noise_pct = st.slider(
                        "Label noise (%)",
                        min_value=0.0,
                        max_value=5.0,
                        value=float(cfg.get("label_noise_pct", 0.0)),
                        step=1.0,
                        help="Flip a small share of labels to demonstrate noise impact (2–5% suggested).",
                    )
                with col_b:
                    tld_level = st.select_slider(
                        "Suspicious TLD frequency",
                        options=["low", "med", "high"],
                        value=cfg.get("susp_tld_level", "med"),
                    )
                    caps_level = st.select_slider(
                        "ALL-CAPS / urgency intensity",
                        options=["low", "med", "high"],
                        value=cfg.get("caps_intensity", "med"),
                    )
                    money_level = st.select_slider(
                        "Money symbols & urgency",
                        options=["off", "low", "high"],
                        value=cfg.get("money_urgency", "low"),
                    )
                    attachment_keys = list(ATTACHMENT_MIX_PRESETS.keys())
                    current_mix = cfg.get("attachments_mix", DEFAULT_ATTACHMENT_MIX)
                    current_choice = next(
                        (name for name, mix in ATTACHMENT_MIX_PRESETS.items() if mix == current_mix),
                        "Balanced",
                    )
                    attachment_choice = st.selectbox(
                        "Attachment lure mix",
                        options=attachment_keys,
                        index=attachment_keys.index(current_choice) if current_choice in attachment_keys else 1,
                        help="Choose how often risky attachments (HTML/ZIP/XLSM/EXE) appear vs. safer PDFs.",
                    )
                    seed_value = st.number_input(
                        "Random seed",
                        min_value=0,
                        value=int(cfg.get("seed", 42)),
                        help="Keep this fixed for reproducibility.",
                    )
                    poison_demo = st.toggle(
                        "Data poisoning demo (synthetic)",
                        value=bool(cfg.get("poison_demo", False)),
                        help="Adds a tiny malicious distribution shift labeled as safe to show metric degradation.",
                    )

                submitted = st.form_submit_button("Apply tweaks and preview", type="primary")

            if submitted:
                attachment_mix = ATTACHMENT_MIX_PRESETS.get(attachment_choice, DEFAULT_ATTACHMENT_MIX).copy()
                config: DatasetConfig = {
                    "seed": int(seed_value),
                    "n_total": int(n_total_choice),
                    "spam_ratio": float(spam_ratio),
                    "susp_link_level": str(int(links_level)),
                    "susp_tld_level": tld_level,
                    "caps_intensity": caps_level,
                    "money_urgency": money_level,
                    "attachments_mix": attachment_mix,
                    "edge_cases": int(edge_cases),
                    "label_noise_pct": float(noise_pct),
                    "poison_demo": bool(poison_demo),
                }
                dataset_rows = build_dataset_from_config(config)
                preview_summary = compute_dataset_summary(dataset_rows)
                lint_counts = lint_dataset(dataset_rows)
                ss["dataset_preview"] = dataset_rows
                ss["dataset_preview_config"] = config
                ss["dataset_preview_summary"] = preview_summary
                ss["dataset_preview_lint"] = lint_counts
                ss["dataset_manual_queue"] = pd.DataFrame(dataset_rows[: min(len(dataset_rows), 200)])
                if ss["dataset_manual_queue"] is not None and not ss["dataset_manual_queue"].empty:
                    ss["dataset_manual_queue"].insert(0, "include", True)
                ss["dataset_compare_delta"] = dataset_summary_delta(current_summary, preview_summary)
                ss["last_dataset_delta_story"] = dataset_delta_story(ss["dataset_compare_delta"])
                st.success("Preview ready — scroll to **Review & approve** to curate rows before committing.")
                explanation = explain_config_change(config, ss.get("dataset_config", DEFAULT_DATASET_CONFIG))
                if explanation:
                    st.caption(explanation)
                if lint_counts and any(lint_counts.values()):
                    st.warning(
                        "PII lint flags — sensitive-looking patterns detected (credit cards: {} | IBAN: {})."
                        .format(lint_counts.get("credit_card", 0), lint_counts.get("iban", 0))
                    )
                if len(dataset_rows) > 200:
                    st.caption("Manual queue shows the first 200 items per guardrail. Full dataset size: {}.".format(len(dataset_rows)))

    if ss.get("dataset_preview"):
        with section_surface():
            st.markdown("### 2 · Review & approve")
            preview_summary = ss.get("dataset_preview_summary") or compute_dataset_summary(ss["dataset_preview"])
            sum_col, lint_col = st.columns([2, 2], gap="large")
            with sum_col:
                st.metric("Preview rows", preview_summary.get("total", 0))
                st.metric("Spam share", f"{preview_summary.get('spam_ratio', 0)*100:.1f}%")
                st.metric("Avg suspicious links (spam)", f"{preview_summary.get('avg_susp_links', 0.0):.2f}")
            with lint_col:
                lint_counts = ss.get("dataset_preview_lint") or {"credit_card": 0, "iban": 0}
                st.write("**Validation checks**")
                st.write(f"- Credit card-like patterns: {lint_counts.get('credit_card', 0)}")
                st.write(f"- IBAN-like patterns: {lint_counts.get('iban', 0)}")
                st.caption("Guardrail: no live link fetching, HTML escaped, duplicates dropped.")

            manual_df = ss.get("dataset_manual_queue")
            if manual_df is None or manual_df.empty:
                manual_df = pd.DataFrame(ss["dataset_preview"][: min(len(ss["dataset_preview"]), 200)])
                if not manual_df.empty:
                    manual_df.insert(0, "include", True)
            edited_df = st.data_editor(
                manual_df,
                width="stretch",
                hide_index=True,
                key="dataset_manual_editor",
                column_config={
                    "include": st.column_config.CheckboxColumn("Include?", help="Uncheck to drop before committing."),
                    "label": st.column_config.SelectboxColumn("Label", options=sorted(VALID_LABELS)),
                },
            )
            ss["dataset_manual_queue"] = edited_df
            st.caption("Manual queue covers up to 200 rows per apply — re-run the builder to generate more variations.")

            commit_col, discard_col, _ = st.columns([1, 1, 2])

            if commit_col.button("Commit dataset tweaks", type="primary"):
                preview_rows = ss.get("dataset_preview")
                config = ss.get("dataset_preview_config", ss.get("dataset_config", DEFAULT_DATASET_CONFIG))
                if not preview_rows:
                    st.error("Generate a preview before committing.")
                else:
                    edited_records = []
                    if isinstance(edited_df, pd.DataFrame):
                        edited_records = edited_df.to_dict(orient="records")
                    preview_copy = [dict(row) for row in preview_rows]
                    for idx, record in enumerate(edited_records):
                        if idx >= len(preview_copy):
                            break
                        preview_copy[idx]["title"] = str(record.get("title", preview_copy[idx].get("title", "")))
                        preview_copy[idx]["body"] = str(record.get("body", preview_copy[idx].get("body", "")))
                        preview_copy[idx]["label"] = record.get("label", preview_copy[idx].get("label", "spam"))
                        preview_copy[idx]["include"] = bool(record.get("include", True))
                    final_rows: List[Dict[str, str]] = []
                    for idx, row in enumerate(preview_copy):
                        include_flag = row.pop("include", True)
                        if idx < len(edited_records):
                            include_flag = bool(edited_records[idx].get("include", include_flag))
                        if not include_flag:
                            continue
                        final_rows.append(
                            {
                                "title": row.get("title", "").strip(),
                                "body": row.get("body", "").strip(),
                                "label": row.get("label", "spam"),
                            }
                        )
                    if len(final_rows) < 10:
                        st.warning("Need at least 10 rows to maintain a meaningful dataset.")
                    else:
                        lint_counts = lint_dataset(final_rows)
                        new_summary = compute_dataset_summary(final_rows)
                        delta = dataset_summary_delta(ss.get("dataset_summary", {}), new_summary)
                        ss["previous_dataset_summary"] = ss.get("dataset_summary", {})
                        ss["dataset_summary"] = new_summary
                        ss["dataset_config"] = config
                        ss["dataset_compare_delta"] = delta
                        ss["last_dataset_delta_story"] = dataset_delta_story(delta)
                        ss["labeled"] = final_rows
                        ss["active_dataset_snapshot"] = None
                        ss["dataset_snapshot_name"] = ""
                        _clear_dataset_preview_state()
                        st.success(f"Dataset updated with {len(final_rows)} curated rows.")
                        if any(lint_counts.values()):
                            st.warning(
                                "Lint warnings persist after commit (credit cards: {} | IBAN: {})."
                                .format(lint_counts.get("credit_card", 0), lint_counts.get("iban", 0))
                            )

            if discard_col.button("Discard preview"):
                _discard_preview()
                st.info("Preview cleared. The active labeled dataset remains unchanged.")

    with section_surface():
        st.markdown("### 3 · Snapshot & provenance")
        config_json = json.dumps(ss.get("dataset_config", DEFAULT_DATASET_CONFIG), indent=2, sort_keys=True)
        st.caption("Save immutable snapshots to reference in the model card and audits.")
        st.json(json.loads(config_json))
        ss["dataset_snapshot_name"] = st.text_input(
            "Snapshot name",
            value=ss.get("dataset_snapshot_name", ""),
            help="Describe the scenario (e.g., 'High links, 5% noise').",
        )
        if st.button("Save dataset snapshot", key="save_dataset_snapshot"):
            snapshot_id = compute_dataset_hash(ss["labeled"])
            entry = {
                "id": snapshot_id,
                "name": ss.get("dataset_snapshot_name") or f"snapshot-{len(ss['datasets'])+1}",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "config": ss.get("dataset_config", DEFAULT_DATASET_CONFIG),
                "config_json": config_json,
                "rows": len(ss["labeled"]),
            }
            existing = next((snap for snap in ss["datasets"] if snap.get("id") == snapshot_id), None)
            if existing:
                existing.update(entry)
            else:
                ss["datasets"].append(entry)
            ss["active_dataset_snapshot"] = snapshot_id
            st.success(f"Snapshot saved with id `{snapshot_id[:10]}…`. Use it in the model card.")

        if ss.get("datasets"):
            df_snap = pd.DataFrame(ss["datasets"])
            st.dataframe(df_snap[["name", "id", "timestamp", "rows"]], hide_index=True, width="stretch")
        else:
            st.caption("No snapshots yet. Save one after curating your first dataset.")

    nerd_data = render_nerd_mode_toggle(
        key="nerd_mode_data",
        title="Nerd Mode — diagnostics & CSV import",
        description="Inspect token clouds, feature histograms, leakage checks, and ingest custom CSVs.",
    )

    if nerd_data:
        with section_surface():
            st.markdown("### 4 · Nerd Mode extras")
            df_lab = pd.DataFrame(ss["labeled"])
            if df_lab.empty:
                st.info("Label some emails or import data to unlock diagnostics.")
            else:
                tokens_spam = Counter()
                tokens_safe = Counter()
                for _, row in df_lab.iterrows():
                    text = f"{row.get('title', '')} {row.get('body', '')}".lower()
                    tokens = re.findall(r"[a-zA-Z']+", text)
                    if row.get("label") == "spam":
                        tokens_spam.update(tokens)
                    else:
                        tokens_safe.update(tokens)
                top_spam = tokens_spam.most_common(12)
                top_safe = tokens_safe.most_common(12)
                col_tok1, col_tok2 = st.columns(2)
                with col_tok1:
                    st.markdown("**Class token cloud — Spam**")
                    st.write(", ".join(f"{w} ({c})" for w, c in top_spam))
                with col_tok2:
                    st.markdown("**Class token cloud — Safe**")
                    st.write(", ".join(f"{w} ({c})" for w, c in top_safe))

                spam_link_counts = [
                    _count_suspicious_links(row.get("body", ""))
                    for _, row in df_lab.iterrows()
                    if row.get("label") == "spam"
                ]
                link_series = pd.Series(spam_link_counts, name="Suspicious links")
                if not link_series.empty:
                    st.bar_chart(link_series.value_counts().sort_index(), height=200)
                else:
                    st.caption("No spam samples yet to chart suspicious link frequency.")

                title_groups: Dict[str, set] = {}
                leakage_titles = []
                for _, row in df_lab.iterrows():
                    title = row.get("title", "").strip().lower()
                    label = row.get("label")
                    title_groups.setdefault(title, set()).add(label)
                for title, labels in title_groups.items():
                    if len(labels) > 1 and title:
                        leakage_titles.append(title)
                if leakage_titles:
                    st.warning("Potential leakage: identical subjects across labels -> " + ", ".join(leakage_titles[:5]))
                else:
                    st.caption("Leakage check: no identical subjects across labels in the active dataset.")

                strat_df = df_lab.groupby("label").size().reset_index(name="count")
                st.dataframe(strat_df, hide_index=True, width="stretch")

        with st.expander("📤 Upload CSV of labeled emails (strict schema)", expanded=False):
            st.caption(
                "Schema: title, body, label (spam|safe). Limits: ≤2,000 rows, title ≤200 chars, body ≤2,000 chars."
            )
            st.caption("Uploaded data stays in this session only. No emails are sent or fetched.")
            up = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader_labeled")
            if up is not None:
                try:
                    df_up = pd.read_csv(up)
                    df_up.columns = [c.strip().lower() for c in df_up.columns]
                    ok, msg = _validate_csv_schema(df_up)
                    if not ok:
                        st.error(msg)
                    else:
                        if len(df_up) > 2000:
                            st.error("Too many rows (max 2,000). Trim the file and retry.")
                        else:
                            df_up["label"] = df_up["label"].apply(_normalize_label)
                            df_up = df_up[df_up["label"].isin(VALID_LABELS)]
                            for col in ["title", "body"]:
                                df_up[col] = df_up[col].fillna("").astype(str).str.strip()
                            df_up = df_up[(df_up["title"].str.len() <= 200) & (df_up["body"].str.len() <= 2000)]
                            df_up = df_up[(df_up["title"] != "") | (df_up["body"] != "")]
                            df_existing = pd.DataFrame(ss["labeled"])
                            if not df_existing.empty:
                                merged = df_up.merge(df_existing, on=["title", "body", "label"], how="left", indicator=True)
                                df_up = merged[merged["_merge"] == "left_only"].loc[:, ["title", "body", "label"]]
                            lint_counts = lint_dataset(df_up.to_dict(orient="records"))
                            st.dataframe(df_up.head(20), hide_index=True, width="stretch")
                            st.caption(f"Rows passing validation: {len(df_up)} | Lint -> credit cards: {lint_counts['credit_card']}, IBAN: {lint_counts['iban']}")
                            if len(df_up) > 0 and st.button("Import into labeled dataset", key="btn_import_csv"):
                                ss["labeled"].extend(df_up.to_dict(orient="records"))
                                ss["dataset_summary"] = compute_dataset_summary(ss["labeled"])
                                st.success(f"Imported {len(df_up)} rows into labeled dataset. Revisit builder to rebalance if needed.")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")


def _clear_dataset_preview_state() -> None:
    ss["dataset_preview"] = None
    ss["dataset_preview_config"] = None
    ss["dataset_preview_summary"] = None
    ss["dataset_preview_lint"] = None
    ss["dataset_manual_queue"] = None
    ss["dataset_controls_open"] = False


def _discard_preview() -> None:
    _clear_dataset_preview_state()
    ss["dataset_compare_delta"] = None
    ss["last_dataset_delta_story"] = explain_config_change(ss.get("dataset_config", DEFAULT_DATASET_CONFIG))


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
                try:
                    train_texts_cache = [combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)]
                    cache_train_embeddings(train_texts_cache)
                except Exception:
                    pass

    parsed_split = None
    y_tr_labels = None
    y_te_labels = None
    if ss.get("model") is not None and ss.get("split_cache") is not None:
        try:
            parsed_split = _parse_split_cache(ss["split_cache"])
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr_labels, y_te_labels = parsed_split

            # Existing success + story (kept)
            story = make_after_training_story(y_tr_labels, y_te_labels)
            st.success("Training finished.")
            st.markdown(story)

            # --- New: Training Storyboard (plain language, for everyone) ---
            with section_surface():
                st.markdown("### Training story — what just happened")

                # 1) What data was used
                ct = _counts(list(y_tr_labels))
                st.markdown(
                    f"- The system trained on **{len(y_tr_labels)} emails** "
                    f"(Spam: {ct['spam']}, Safe: {ct['safe']}).\n"
                    "- It looked for patterns that distinguish **Spam** from **Safe**.\n"
                    "- It saved these patterns as simple rules (weights) it can use later to decide."
                )

                # Mini class-balance chart
                try:
                    bal_df = pd.DataFrame(
                        {"class": ["spam", "safe"], "count": [ct["spam"], ct["safe"]]}
                    ).set_index("class")
                    st.caption("Training set balance")
                    st.bar_chart(bal_df, width="stretch")
                except Exception:
                    pass

                # 2) Top signals the model noticed (plain list)
                shown_any_signals = False
                try:
                    # Prefer numeric-feature view if available (Hybrid model)
                    if hasattr(ss["model"], "numeric_feature_details"):
                        nfd = ss["model"].numeric_feature_details().copy()
                        nfd["friendly_name"] = nfd["feature"].map(FEATURE_DISPLAY_NAMES)
                        # Positive weights → Spam, Negative → Safe
                        top_spam = (
                            nfd.sort_values("weight_per_std", ascending=False)
                            .head(3)["friendly_name"].tolist()
                        )
                        top_safe = (
                            nfd.sort_values("weight_per_std", ascending=True)
                            .head(3)["friendly_name"].tolist()
                        )
                        st.markdown("**Top signals the model picked up**")
                        st.write(f"• Toward **Spam**: {', '.join(top_spam) if top_spam else '—'}")
                        st.write(f"• Toward **Safe**: {', '.join(top_safe) if top_safe else '—'}")
                        st.caption(
                            "These are simple cues (e.g., links, ALL-CAPS bursts, money/urgency hints) that nudged decisions."
                        )
                        shown_any_signals = True
                except Exception:
                    pass

                if not shown_any_signals:
                    # Fallback wording if coefficients aren’t available
                    st.markdown("**What it learned**")
                    st.write(
                        "The model pays more attention to words and cues that frequently appear in spam (e.g., urgent offers, suspicious links) "
                        "and learns to ignore everyday business phrases that tend to be safe."
                    )

                # 3) A couple of concrete examples the model saw (subjects only)
                try:
                    if X_tr_t and X_tr_b and y_tr_labels:
                        train_subjects = list(X_tr_t)
                        y_arr = list(y_tr_labels)
                        # pick first spam + first safe subject line available
                        spam_subj = next((s for s, y in zip(train_subjects, y_arr) if y == "spam"), None)
                        safe_subj = next((s for s, y in zip(train_subjects, y_arr) if y == "safe"), None)
                        if spam_subj or safe_subj:
                            st.markdown("**Examples it learned from**")
                            if spam_subj:
                                st.write(f"• Spam example: *{spam_subj[:100]}{'…' if len(spam_subj)>100 else ''}*")
                            if safe_subj:
                                st.write(f"• Safe example: *{safe_subj[:100]}{'…' if len(safe_subj)>100 else ''}*")
                except Exception:
                    pass

                # 4) What this means / next step
                st.markdown(
                    "Your model now has a simple **mental map** of what Spam vs. Safe looks like. "
                    "Next, we’ll check how well this map works on emails it hasn’t seen before."
                )
                st.info("Go to **3) Evaluate** to test performance and choose a spam threshold.")

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
    prev_eval = ss.get("last_eval_results") or {}
    acc_cur, p_cur, r_cur, f1_cur, cm_cur = _pr_acc_cm(y_true01, p_spam, current_thr)

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
            dataset_story = ss.get("last_dataset_delta_story")
            metric_deltas: list[str] = []
            if prev_eval:
                metric_deltas.append(f"Δaccuracy {acc_cur - prev_eval.get('accuracy', acc_cur):+.2%}")
                metric_deltas.append(f"Δprecision {p_cur - prev_eval.get('precision', p_cur):+.2%}")
                metric_deltas.append(f"Δrecall {r_cur - prev_eval.get('recall', r_cur):+.2%}")
            extra_caption = " | ".join(part for part in [dataset_story, " · ".join(metric_deltas) if metric_deltas else ""] if part)
            if extra_caption:
                st.caption(f"📂 {extra_caption}")

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

    ss["last_eval_results"] = {
        "accuracy": acc_cur,
        "precision": p_cur,
        "recall": r_cur,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

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

            with section_surface():
                st.markdown("### Why did it decide this way? (per email)")

                split_cache = ss.get("split_cache")
                train_texts: list[str] = []
                train_labels: list[str] = []
                train_emb: Optional[np.ndarray] = None
                if split_cache:
                    try:
                        if len(split_cache) == 6:
                            X_tr_t, _, X_tr_b, _, y_tr_vals, _ = split_cache
                            train_texts = [combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)]
                            train_labels = list(y_tr_vals)
                        else:
                            X_tr_texts, _, y_tr_vals, _ = split_cache
                            train_texts = list(X_tr_texts)
                            train_labels = list(y_tr_vals)
                        if train_texts:
                            train_emb = cache_train_embeddings(train_texts)
                    except Exception:
                        train_texts = []
                        train_labels = []
                        train_emb = None

                threshold_val = float(ss.get("threshold", 0.5))
                model_obj = ss.get("model")

                for email_idx, row in enumerate(ss["use_batch_results"]):
                    title = row.get("title", "")
                    body = row.get("body", "")
                    pred_label = row.get("pred", "")
                    p_spam_val = row.get("p_spam")
                    try:
                        p_spam_float = float(p_spam_val)
                    except (TypeError, ValueError):
                        p_spam_float = None

                    header = title or "(no subject)"
                    with st.container(border=True):
                        st.markdown(f"#### {header}")
                        st.caption(f"Predicted **{pred_label or '—'}**")

                        if p_spam_float is not None:
                            margin = p_spam_float - threshold_val
                            decision = "Spam" if p_spam_float >= threshold_val else "Safe"
                            st.markdown(
                                f"**Decision summary:** P(spam) = {p_spam_float:.2f} vs threshold {threshold_val:.2f} → **{decision}** "
                                f"(margin {margin:+.2f})"
                            )
                        else:
                            st.caption("Probability not available for this email.")

                        if train_texts and train_labels:
                            try:
                                nn_examples = get_nearest_training_examples(
                                    combine_text(title, body), train_texts, train_labels, train_emb, k=3
                                )
                            except Exception:
                                nn_examples = []
                        else:
                            nn_examples = []

                        if nn_examples:
                            st.markdown("**Similar training emails (semantic evidence):**")
                            for example in nn_examples:
                                text_full = example["text"]
                                title_example = text_full.split("\n", 1)[0]
                                st.write(
                                    f"- *{title_example.strip() or '(no subject)'}* — label: **{example['label']}** "
                                    f"(sim {example['similarity']:.2f})"
                                )
                        else:
                            st.caption("No similar training emails available.")

                        if hasattr(model_obj, "scaler") and hasattr(model_obj, "lr"):
                            contribs = numeric_feature_contributions(model_obj, title, body)
                            if contribs:
                                contribs_sorted = sorted(contribs, key=lambda x: x[2], reverse=True)
                                st.markdown("**Numeric cues (how they nudged the decision):**")
                                st.dataframe(
                                    pd.DataFrame(
                                        [
                                            {
                                                "feature": feat,
                                                "standardized": val,
                                                "toward_spam_logit": contrib,
                                            }
                                            for feat, val, contrib in contribs_sorted
                                        ]
                                    ).round(3),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                st.caption("Positive values push toward **Spam**; negative toward **Safe**.")
                            else:
                                st.caption("Numeric feature contributions unavailable for this email.")
                        else:
                            st.caption("Numeric cue breakdown requires the hybrid model.")

                        if model_obj is not None:
                            with st.expander("🖍️ Highlight influential words (experimental)", expanded=False):
                                st.caption(
                                    "Runs extra passes to see which words reduce P(spam) the most when removed."
                                )
                                if st.checkbox(
                                    "Compute highlights for this email", key=f"hl_{email_idx}", value=False
                                ):
                                    base_prob, rows_imp = top_token_importances(model_obj, title, body)
                                    if base_prob is None:
                                        st.caption("Unable to compute token importances for this model/email.")
                                    else:
                                        st.caption(
                                            f"Base P(spam) = {base_prob:.2f}. Higher importance means removing the word lowers P(spam) more."
                                        )
                                        if rows_imp:
                                            df_imp = pd.DataFrame(rows_imp[:10])
                                            st.dataframe(df_imp, use_container_width=True, hide_index=True)
                                        else:
                                            st.caption("No influential words found among the sampled tokens.")
                        else:
                            st.caption("Word highlights require a trained model.")

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
        snapshot_id = ss.get("active_dataset_snapshot")
        snapshot_entry = None
        if snapshot_id:
            snapshot_entry = next((snap for snap in ss.get("datasets", []) if snap.get("id") == snapshot_id), None)
        dataset_config_for_card = (snapshot_entry or {}).get("config", ss.get("dataset_config", DEFAULT_DATASET_CONFIG))
        dataset_config_json = json.dumps(dataset_config_for_card, indent=2, sort_keys=True)
        snapshot_label = snapshot_id if snapshot_id else "— (save one in Prepare Data)"

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
**Dataset snapshot ID**: {snapshot_label}
**Dataset config**:
```
{dataset_config_json}
```
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

        with highlight_col:
            st.markdown("#### Dataset provenance")
            if snapshot_id:
                st.write(f"Snapshot ID: `{snapshot_id}`")
            else:
                st.write("Snapshot ID: — (save one in Prepare Data → Snapshot & provenance).")
            st.code(dataset_config_json, language="json")


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
renderer = STAGE_RENDERERS.get(active_stage, render_intro_stage)
renderer()
render_stage_navigation_controls(active_stage)

st.markdown("---")
st.caption("© demistifAI — Built for interactive learning and governance discussions.")
