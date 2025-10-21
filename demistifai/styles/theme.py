"""Streamlit theme stylesheet shared across the application."""

__all__ = ["APP_THEME_CSS"]

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
    padding-top: 0;
    padding-bottom: 0.25rem;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1200px;
    padding-top: 1.5rem !important;
    padding-right: clamp(0.75rem, 3.5vw, 1.5rem);
    padding-bottom: 3rem !important;
    padding-left: clamp(0.75rem, 3.5vw, 1.5rem);
}

@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] {
        max-width: 100%;
    }
}

@media (max-width: 720px) {
    [data-testid="stMainBlockContainer"] {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    .section-surface > div[data-testid="stVerticalBlock"] {
        margin-bottom: clamp(1rem, 3vw, 1.5rem);
        border-radius: 18px;
        padding: clamp(1rem, 4.8vw, 1.7rem);
    }
    .section-surface.section-surface--hero > div[data-testid="stVerticalBlock"] {
        border-radius: 26px;
        padding: clamp(1.8rem, 6vw, 2.6rem);
    }
}

.section-surface {
    display: block;
    width: 100%;
}

.section-surface > div[data-testid="stVerticalBlock"] {
    position: relative;
    z-index: 0;
    margin-bottom: clamp(1.2rem, 1.2vw + 0.8rem, 1.8rem);
    border-radius: var(--surface-radius);
    border: 1px solid var(--surface-border);
    background: var(--surface-gradient);
    box-shadow: var(--surface-shadow);
    padding: clamp(1.15rem, 1.4vw + 0.85rem, 2.1rem);
    overflow: hidden;
    color: #0f172a;
}

.section-surface > div[data-testid="stVerticalBlock"]::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(150deg, rgba(59, 130, 246, 0.15), transparent 65%);
    opacity: 0;
    transition: opacity 0.3s ease;
    pointer-events: none;
    z-index: 0;
}

.section-surface > div[data-testid="stVerticalBlock"]:hover::before {
    opacity: 1;
}

.section-surface > div[data-testid="stVerticalBlock"] > * {
    position: relative;
    z-index: 1;
}

.section-surface > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] {
    margin-bottom: 0.9rem;
}

.section-surface > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:first-child,
.section-surface > div[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child {
    margin-bottom: 0;
}

.section-surface > div[data-testid="stVerticalBlock"] h2,
.section-surface > div[data-testid="stVerticalBlock"] h3,
.section-surface > div[data-testid="stVerticalBlock"] h4 {
    margin-top: 0;
    color: inherit;
}

.section-surface > div[data-testid="stVerticalBlock"] p,
.section-surface > div[data-testid="stVerticalBlock"] ul {
    font-size: 0.98rem;
    line-height: 1.65;
    color: inherit;
}

.section-surface.section-surface--hero > div[data-testid="stVerticalBlock"] {
    position: relative;
    padding: clamp(2.4rem, 5vw, 3.6rem);
    background: linear-gradient(160deg, #1d4ed8, #312e81);
    color: #f8fafc;
    border: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: 32px;
    box-shadow: 0 28px 70px rgba(30, 64, 175, 0.35);
    overflow: hidden;
}

.section-surface.section-surface--hero > div[data-testid="stVerticalBlock"]::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at top right, rgba(96, 165, 250, 0.35), transparent 42%),
        radial-gradient(circle at bottom left, rgba(99, 102, 241, 0.28), transparent 46%);
    pointer-events: none;
}

.section-surface.section-surface--hero > div[data-testid="stVerticalBlock"]::before {
    display: none;
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

.cta-sticky [data-testid="stButton"] button,
.cta-sticky [data-testid="stFormSubmitButton"] button {
    width: 100%;
}

.section-surface.dataset-builder-surface > div[data-testid="stVerticalBlock"] {
    background: radial-gradient(circle at top left, rgba(15, 118, 110, 0.32), transparent 52%),
        linear-gradient(160deg, rgba(13, 17, 23, 0.94), rgba(2, 6, 23, 0.92));
    border: 1px solid rgba(94, 234, 212, 0.24);
    box-shadow: 0 28px 62px rgba(8, 47, 73, 0.45);
    color: #e2e8f0;
}

.section-surface.dataset-builder-surface > div[data-testid="stVerticalBlock"]::before {
    display: none;
}

.section-surface.dataset-builder-surface > div[data-testid="stVerticalBlock"] > * {
    color: inherit;
}

.dataset-builder {
    display: grid;
    gap: 0.75rem;
    font-family: "Fira Code", "JetBrains Mono", "SFMono-Regular", ui-monospace, monospace;
    position: relative;
}

.dataset-builder__intro {
    display: flex;
    justify-content: space-between;
    align-items: stretch;
    gap: 1.5rem;
    background: linear-gradient(135deg, rgba(13, 148, 136, 0.4), rgba(56, 189, 248, 0.22));
    border: 1px solid rgba(94, 234, 212, 0.32);
    border-radius: 20px;
    padding: 1.35rem 1.5rem;
    box-shadow: inset 0 0 0 1px rgba(8, 47, 73, 0.4), 0 24px 48px rgba(8, 47, 73, 0.36);
}

.dataset-builder__intro-copy {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    flex: 1 1 55%;
    color: #e2e8f0;
}

.dataset-builder__eyebrow {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.34em;
    font-weight: 700;
    color: rgba(165, 243, 252, 0.78);
}

.dataset-builder__title {
    margin: 0;
    font-size: 1.55rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: #f8fafc;
}

.dataset-builder__lead {
    margin: 0;
    color: rgba(226, 232, 240, 0.92);
    font-size: 0.98rem;
    line-height: 1.6;
    max-width: 40ch;
}

.dataset-builder__intro-aside {
    flex: 1 1 45%;
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.dataset-builder__metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
}

.dataset-builder__metric {
    background: rgba(2, 6, 23, 0.7);
    border: 1px solid rgba(94, 234, 212, 0.24);
    border-radius: 16px;
    padding: 0.85rem 1rem;
    box-shadow: inset 0 0 0 1px rgba(8, 47, 73, 0.46);
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}

.dataset-builder__metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.26em;
    color: rgba(148, 163, 184, 0.78);
    font-weight: 700;
}

.dataset-builder__metric-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 0.04em;
}

.dataset-builder__metric-subtext {
    font-size: 0.8rem;
    color: rgba(203, 213, 225, 0.85);
    letter-spacing: 0.04em;
}

.dataset-builder__status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.75rem 1.15rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    border: 1px solid rgba(94, 234, 212, 0.32);
    background: rgba(15, 23, 42, 0.68);
    color: rgba(226, 232, 240, 0.9);
    flex-wrap: wrap;
}

.dataset-builder__status-pill--active {
    background: linear-gradient(90deg, rgba(13, 148, 136, 0.45), rgba(56, 189, 248, 0.28));
    color: #0f172a;
    border-color: rgba(94, 234, 212, 0.6);
}

.dataset-builder__status-pill--inactive {
    border-color: rgba(148, 163, 184, 0.38);
    color: rgba(203, 213, 225, 0.88);
}

.dataset-builder__status-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: rgba(148, 163, 184, 0.75);
    box-shadow: 0 0 0 4px rgba(148, 163, 184, 0.18);
}

.dataset-builder__status-pill--active .dataset-builder__status-dot {
    background: #22d3ee;
    box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.3);
}

.dataset-builder__status-caption {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.22em;
    opacity: 0.8;
}

.dataset-builder__terminal {
    background: rgba(2, 6, 23, 0.78);
    border: 1px solid rgba(94, 234, 212, 0.24);
    border-radius: 18px;
    padding: 1.1rem 1.25rem;
    box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.52);
    display: grid;
    gap: 0.5rem;
}

.dataset-builder__terminal .dataset-builder__comment {
    padding-left: 1.35rem;
    color: rgba(148, 163, 184, 0.92);
}

.dataset-builder__nerd-callout {
    background: rgba(15, 23, 42, 0.62);
    border: 1px solid rgba(56, 189, 248, 0.32);
    border-radius: 14px;
    padding: 0.65rem 0.85rem;
    margin-bottom: 0.2rem;
}

.dataset-builder__nerd-callout .dataset-builder__comment {
    padding-left: 0.95rem;
}

.dataset-builder__command-line {
    display: flex;
    align-items: baseline;
    gap: 0.65rem;
    font-size: 0.98rem;
    line-height: 1.6;
    color: #e2e8f0;
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(94, 234, 212, 0.2);
    border-radius: 12px;
    padding: 0.6rem 0.85rem;
    box-shadow: inset 0 0 0 1px rgba(2, 6, 23, 0.58);
}

.dataset-builder__prompt {
    color: #22d3ee;
    font-weight: 700;
    letter-spacing: 0.04em;
}

.dataset-builder__command {
    color: #facc15;
    letter-spacing: 0.02em;
}

.dataset-builder__value {
    color: #38bdf8;
    font-weight: 600;
}

.dataset-builder__meta {
    margin-left: auto;
    font-size: 0.78rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(125, 211, 252, 0.82);
}

.dataset-builder__comment {
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.5;
    padding-left: 1.85rem;
}

.dataset-builder__divider {
    position: relative;
    text-align: center;
    margin: 0.6rem 0 0.2rem;
    font-size: 0.78rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: rgba(125, 211, 252, 0.82);
}

.dataset-builder__divider::before {
    content: "";
    display: block;
    height: 1px;
    background: rgba(148, 163, 184, 0.24);
    opacity: 0.7;
    margin-bottom: 1.1rem;
}

.dataset-builder__divider span {
    background: rgba(13, 17, 23, 0.95);
    padding: 0 0.65rem;
    position: relative;
    top: -1.65rem;
}

.dataset-builder__form-shell {
    position: relative;
    background: rgba(2, 6, 23, 0.78);
    border: 1px solid rgba(94, 234, 212, 0.2);
    border-radius: 18px;
    padding: 1.1rem 1.25rem 5rem;
    display: grid;
    gap: 1.05rem;
    box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.52);
}

.dataset-builder__form-shell > div[data-testid="stVerticalBlock"],
.dataset-builder__form-shell div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
    margin-bottom: 0.4rem;
}

.dataset-builder__form-shell [data-testid="column"] > div[data-testid="stVerticalBlock"] {
    margin-bottom: 0;
}

.dataset-builder-surface [data-testid="stRadio"] label,
.dataset-builder-surface [data-testid="stSlider"] label,
.dataset-builder-surface [data-testid="stSelectbox"] label,
.dataset-builder-surface [data-testid="stNumberInput"] label,
.dataset-builder-surface [data-testid="stSelectSlider"] label,
.dataset-builder-surface [data-testid="stToggle"] label {
    color: #cbd5f5;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dataset-builder-surface [data-testid="stRadio"] div[role="radiogroup"] {
    gap: 0.45rem;
}

.dataset-builder-surface [data-testid="stRadio"] label[data-baseweb="radio"] {
    background: rgba(15, 23, 42, 0.68);
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    padding: 0.35rem 0.75rem;
    color: #e2e8f0;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.dataset-builder-surface [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
    border-color: rgba(56, 189, 248, 0.6);
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.18);
}

.dataset-builder-surface [data-testid="stRadio"] input:checked + div {
    background: linear-gradient(90deg, rgba(14, 165, 233, 0.15), rgba(8, 145, 178, 0.08));
}

.dataset-builder-surface [data-testid="stSlider"] label p,
.dataset-builder-surface [data-testid="stSelectbox"] label p,
.dataset-builder-surface [data-testid="stNumberInput"] label p,
.dataset-builder-surface [data-testid="stSelectSlider"] label p,
.dataset-builder-surface [data-testid="stToggle"] label p {
    color: inherit;
}

.dataset-builder-surface [data-testid="stSlider"] [data-baseweb="slider"] {
    margin-top: 0.4rem;
}

.dataset-builder-surface [data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(1) {
    background: rgba(15, 23, 42, 0.65);
    height: 6px;
    border-radius: 999px;
}

.dataset-builder-surface [data-testid="stSlider"] [data-baseweb="slider"] > div:nth-child(2) {
    background: linear-gradient(90deg, #0ea5e9, #22d3ee);
}

.dataset-builder-surface [data-testid="stSlider"] [role="slider"] {
    width: 18px;
    height: 18px;
    background: #22d3ee;
    border: 2px solid #0f172a;
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.24);
}

.dataset-builder-surface [data-testid="stSlider"] [role="slider"]:hover {
    box-shadow: 0 0 0 4px rgba(34, 211, 238, 0.28);
}

.dataset-builder-surface [data-baseweb="select"],
.dataset-builder-surface [data-baseweb="input"] {
    background: rgba(15, 23, 42, 0.72);
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    color: #e2e8f0;
}

.dataset-builder-surface [data-baseweb="select"]:hover,
.dataset-builder-surface [data-baseweb="input"]:hover {
    border-color: rgba(56, 189, 248, 0.55);
}

.dataset-builder-surface [data-baseweb="select"] input,
.dataset-builder-surface [data-baseweb="input"] input {
    color: #e2e8f0;
}

.dataset-builder-surface [data-baseweb="select"] svg,
.dataset-builder-surface [data-baseweb="input"] svg {
    color: rgba(125, 211, 252, 0.86);
}

.dataset-builder-surface [data-testid="stToggle"] [role="switch"] {
    background: rgba(30, 64, 175, 0.35);
    border: 1px solid rgba(148, 163, 184, 0.42);
}

.dataset-builder-surface [data-testid="stToggle"] [role="switch"][aria-checked="true"] {
    background: linear-gradient(90deg, #06b6d4, #38bdf8);
    border-color: rgba(56, 189, 248, 0.82);
}

.dataset-builder-surface [data-testid="stToggle"] [role="switch"]::before {
    background: #0b1120;
}

.dataset-builder-surface .cta-sticky {
    background: linear-gradient(180deg, rgba(13, 17, 23, 0.98), rgba(2, 6, 23, 0.96));
    border-top: 1px solid rgba(56, 189, 248, 0.32);
    box-shadow: 0 -18px 36px rgba(8, 47, 73, 0.45);
    padding-bottom: 0.85rem;
}

.dataset-builder-surface .cta-sticky [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #0ea5e9, #22d3ee);
    border: 1px solid rgba(14, 165, 233, 0.45);
    color: #0f172a;
    font-family: "Fira Code", "JetBrains Mono", ui-monospace, monospace;
    font-weight: 700;
}

.dataset-builder-surface .cta-sticky > div[data-testid="column"]:nth-child(2) [data-testid="stFormSubmitButton"] button {
    background: transparent;
    color: #e2e8f0;
    border: 1px solid rgba(148, 163, 184, 0.45);
}

.dataset-builder-surface .cta-sticky > div[data-testid="column"]:nth-child(2) [data-testid="stFormSubmitButton"] button:hover {
    border-color: rgba(56, 189, 248, 0.6);
    color: #bae6fd;
}

@media (max-width: 1024px) {
    .dataset-builder__intro {
        flex-direction: column;
        gap: 1.25rem;
    }

    .dataset-builder__intro-copy {
        flex: 1 1 auto;
    }

    .dataset-builder__intro-aside {
        width: 100%;
    }
}

@media (max-width: 860px) {
    .dataset-builder__metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 720px) {
    .dataset-builder__metrics {
        grid-template-columns: minmax(0, 1fr);
    }

    .dataset-builder__metric {
        padding: 0.8rem 0.95rem;
    }

    .dataset-builder__status-pill {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.4rem;
    }

    .dataset-builder__terminal {
        padding: 0.95rem 1.05rem;
    }

    .dataset-builder__form-shell {
        padding: 1rem 1rem 6rem;
    }

    .dataset-builder__form-shell div[data-testid="stHorizontalBlock"] {
        flex-direction: column;
        align-items: stretch;
        gap: 0.75rem;
    }

    .dataset-builder__form-shell div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        width: 100% !important;
    }

    .dataset-builder__comment {
        padding-left: 1.45rem;
    }

    .dataset-builder-surface .cta-sticky {
        position: sticky;
        bottom: 0;
        left: 0;
        right: 0;
        margin: 1.25rem -1rem -1.05rem;
        padding: 1.1rem 1rem 1.3rem;
        border-radius: 20px;
        box-shadow: 0 -12px 28px rgba(8, 47, 73, 0.45);
    }

    .dataset-builder-surface .cta-sticky [data-testid="stFormSubmitButton"] button {
        min-height: 52px;
        font-size: 0.95rem;
    }
}

@media (max-width: 520px) {
    .dataset-builder__intro {
        padding: 1.05rem 0.9rem;
    }

    .dataset-builder__title {
        font-size: 1.38rem;
    }

    .dataset-builder__lead {
        font-size: 0.9rem;
    }

    .dataset-builder__terminal .dataset-builder__comment {
        font-size: 0.84rem;
    }

    .dataset-builder__command-line {
        font-size: 0.9rem;
        flex-wrap: wrap;
    }

    .dataset-builder-surface .cta-sticky {
        margin: 1.25rem -0.75rem -1rem;
        padding: 1.05rem 0.75rem 1.25rem;
    }
}

.dataset-health-card {
    border-radius: 1.1rem;
    border: 1px solid rgba(15, 23, 42, 0.08);
    padding: 1.35rem;
    background: rgba(255, 255, 255, 0.88);
    box-shadow: 0 14px 36px rgba(15, 23, 42, 0.1);
    backdrop-filter: blur(6px);
}

.dataset-health-panel {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.dataset-health-panel__status {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.dataset-health-panel__status-copy {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
}

.dataset-health-panel__status-copy h5 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
}

.dataset-health-panel__status-copy small {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(15, 23, 42, 0.55);
    font-weight: 700;
}

.dataset-health-status {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.4rem 1rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
    line-height: 1;
}

.dataset-health-status__dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.85);
}

.dataset-health-status--good {
    background: rgba(34, 197, 94, 0.18);
    color: #15803d;
}

.dataset-health-status--good .dataset-health-status__dot {
    background: #22c55e;
}

.dataset-health-status--warn {
    background: rgba(234, 179, 8, 0.2);
    color: #b45309;
}

.dataset-health-status--warn .dataset-health-status__dot {
    background: #fbbf24;
}

.dataset-health-status--risk {
    background: rgba(248, 113, 113, 0.2);
    color: #b91c1c;
}

.dataset-health-status--risk .dataset-health-status__dot {
    background: #f87171;
}

.dataset-health-status--neutral {
    background: rgba(148, 163, 184, 0.22);
    color: #1f2937;
}

.dataset-health-status--neutral .dataset-health-status__dot {
    background: rgba(148, 163, 184, 0.9);
}

.dataset-health-panel__row {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.dataset-health-panel__row--bar {
    margin-top: 0.35rem;
}

.dataset-health-panel__row--meta {
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.dataset-health-panel__meta-primary {
    display: inline-flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: rgba(15, 23, 42, 0.75);
}

.dataset-health-panel__lint-placeholder {
    font-size: 0.8rem;
    color: rgba(15, 23, 42, 0.55);
}

.dataset-health-panel__bar {
    flex: 1;
    display: flex;
    height: 10px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(15, 23, 42, 0.08);
}

.dataset-health-panel__bar span {
    display: block;
    height: 100%;
}

.dataset-health-panel__bar-spam {
    background: linear-gradient(90deg, #fb7185 0%, #f43f5e 100%);
}

.dataset-health-panel__bar-safe {
    background: linear-gradient(90deg, #38bdf8 0%, #0ea5e9 100%);
}

.dataset-delta-panel {
    position: sticky;
    top: 5.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 1.25rem;
    border-radius: 1rem;
    border: 1px solid rgba(15, 23, 42, 0.08);
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    backdrop-filter: blur(6px);
    margin-top: 1.5rem;
}

.dataset-delta-panel h5 {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 700;
}

.dataset-delta-panel__items {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.dataset-delta-panel__item {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.95rem;
}

.dataset-delta-panel__item span:first-child {
    color: #0f172a;
    font-weight: 600;
}

.delta-arrow {
    font-weight: 700;
}

.delta-arrow--up {
    color: #16a34a;
}

.delta-arrow--down {
    color: #dc2626;
}

.dataset-delta-panel__hint {
    font-size: 0.9rem;
    color: rgba(15, 23, 42, 0.75);
    border-top: 1px solid rgba(15, 23, 42, 0.08);
    padding-top: 0.75rem;
}

.dataset-delta-panel__story {
    font-size: 0.85rem;
    color: rgba(15, 23, 42, 0.7);
}

.section-surface--hero .demai-hero {
    margin: 0 0 10px 0;
    padding: 0;
}

.section-surface--hero [data-testid="column"]:nth-child(2) [data-testid="stButton"] {
    width: 100%;
}

.section-surface--hero [data-testid="column"]:nth-child(2) [data-testid="stButton"] button {
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

.stButton button,
[data-testid="stFormSubmitButton"] button {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.55rem;
    border-radius: 14px;
    padding: 0.65rem 1.4rem;
    font-weight: 700;
    font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border: 1px solid rgba(94, 234, 212, 0.45);
    background: linear-gradient(135deg, rgba(13, 17, 23, 0.96), rgba(8, 47, 73, 0.88));
    color: rgba(226, 232, 240, 0.95);
    text-shadow: 0 0 12px rgba(94, 234, 212, 0.32);
    box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.82), 0 22px 48px rgba(8, 47, 73, 0.55);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, filter 0.18s ease;
    overflow: hidden;
}

.stButton button::after,
[data-testid="stFormSubmitButton"] button::after {
    content: '';
    position: absolute;
    inset: 1px;
    border-radius: 12px;
    background: linear-gradient(140deg, rgba(94, 234, 212, 0.18), transparent 60%);
    pointer-events: none;
}

.stButton button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px);
    border-color: rgba(94, 234, 212, 0.65);
    box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.72), 0 28px 58px rgba(8, 47, 73, 0.62);
    filter: brightness(1.05);
}

.stButton button:focus-visible,
[data-testid="stFormSubmitButton"] button:focus-visible {
    outline: 2px solid rgba(56, 189, 248, 0.7);
    outline-offset: 3px;
}

.stButton button:disabled,
[data-testid="stFormSubmitButton"] button:disabled {
    opacity: 0.55;
    border-color: rgba(148, 163, 184, 0.35);
    box-shadow: inset 0 0 0 1px rgba(30, 41, 59, 0.7), 0 12px 28px rgba(8, 47, 73, 0.35);
    filter: grayscale(10%);
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

    .section-surface > div[data-testid="stVerticalBlock"] {
        padding: 1.6rem 1.8rem;
    }
}

@media (max-width: 768px) {
    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.25rem;
    }

    .section-surface.section-surface--hero > div[data-testid="stVerticalBlock"] {
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

    .section-surface > div[data-testid="stVerticalBlock"] {
        padding: 1.4rem 1.35rem;
    }

    .stButton button,
    [data-testid="stFormSubmitButton"] button {
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

/* Stage command grid + navigation controls */
div[data-testid="stHorizontalBlock"]:has(.stage-top-grid__nav-card) {
    position: relative;
    margin-bottom: clamp(1.6rem, 2vw, 2.4rem);
    padding: clamp(1.8rem, 3vw, 2.6rem);
    border-radius: 28px;
    border: 1px solid rgba(56, 189, 248, 0.32);
    background: linear-gradient(155deg, rgba(15, 23, 42, 0.96), rgba(8, 47, 73, 0.9));
    box-shadow: 0 32px 64px rgba(8, 47, 73, 0.45);
    overflow: hidden;
    z-index: 0;
    gap: clamp(1.4rem, 3vw, 2.4rem);
    align-items: stretch;
}

@media (max-width: 980px) {
    div[data-testid="stHorizontalBlock"]:has(.stage-top-grid__nav-card) {
        padding: clamp(1.4rem, 4vw, 2.2rem);
        border-radius: 24px;
        gap: clamp(1.1rem, 3.5vw, 2rem);
    }
    div[data-testid="stHorizontalBlock"]:has(.stage-top-grid__nav-card) > div[data-testid="column"] {
        flex: 1 1 100% !important;
        min-width: 0 !important;
    }
}

@media (max-width: 720px) {
    div[data-testid="stHorizontalBlock"]:has(.stage-top-grid__nav-card) {
        padding: clamp(1rem, 5vw, 1.6rem);
        border-radius: 20px;
        flex-direction: column;
    }
    div[data-testid="stHorizontalBlock"]:has(.stage-top-grid__nav-card) > div[data-testid="column"] {
        width: 100% !important;
    }
}

div[data-testid="stHorizontalBlock"]:has(.stage-top-grid__nav-card)::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at top right, rgba(59, 130, 246, 0.32), transparent 55%),
        radial-gradient(circle at bottom left, rgba(20, 184, 166, 0.28), transparent 60%);
    opacity: 0.65;
    pointer-events: none;
}

div[data-testid="stHorizontalBlock"]:has(.stage-top-grid__nav-card) > div[data-testid="column"] {
    position: relative;
    z-index: 1;
}

div[data-testid="stHorizontalBlock"]:has(.stage-top-grid__nav-card) > div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
    background: transparent;
    border: none;
    padding: 0;
    box-shadow: none;
}

.stage-top-grid__placeholder {
    background: linear-gradient(160deg, rgba(15, 23, 42, 0.9), rgba(8, 47, 73, 0.85));
    border: 1px dashed rgba(56, 189, 248, 0.45);
    border-radius: 18px;
    color: rgba(226, 232, 240, 0.88);
    font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    font-size: 0.92rem;
    letter-spacing: 0.01em;
    line-height: 1.6;
    padding: 1.25rem 1.4rem 1.25rem 2.25rem;
    position: relative;
    text-align: left;
}

.stage-top-grid__placeholder::before {
    content: '▌';
    color: rgba(56, 189, 248, 0.9);
    position: absolute;
    left: 1.2rem;
    top: 1.2rem;
    font-size: 0.95rem;
}

.stage-top-grid__placeholder strong {
    color: #38bdf8;
    display: block;
    font-size: 1rem;
    margin-bottom: 0.35rem;
}

.stage-top-grid__placeholder--compact {
    font-size: 0.88rem;
    padding: 1rem 1.15rem 1rem 2rem;
}

.stage-top-grid__nav-card,
:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_next_"],
:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_prev_"] {
    position: relative;
    padding: 1.4rem 1.5rem 1.55rem;
    font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}

.stage-top-grid__nav-card {
    border-radius: 20px;
    border: 1px solid rgba(148, 163, 184, 0.28);
    background: linear-gradient(155deg, rgba(15, 23, 42, 0.94), rgba(30, 64, 175, 0.68));
    box-shadow: 0 22px 44px rgba(15, 23, 42, 0.34);
    backdrop-filter: blur(6px);
}

:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_next_"],
:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_prev_"] {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

@media (max-width: 720px) {
    .stage-top-grid__nav-card,
    :is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_next_"],
    :is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_prev_"] {
        padding: clamp(1.1rem, 5vw, 1.5rem);
    }
    :is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_next_"],
    :is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_prev_"] {
        gap: 0.75rem;
    }
}

.stage-top-grid__nav-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at top right, rgba(59, 130, 246, 0.32), transparent 55%),
        radial-gradient(circle at bottom left, rgba(20, 184, 166, 0.28), transparent 62%);
    opacity: 0.58;
    mix-blend-mode: screen;
    pointer-events: none;
    border-radius: inherit;
}

.stage-top-grid__nav-card > *,
:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_next_"] > *,
:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_prev_"] > * {
    position: relative;
    z-index: 1;
}

:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_next_"]::before,
:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_prev_"]::before {
    display: none;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: rgba(94, 234, 212, 0.78);
    font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    font-weight: 600;
    position: relative;
    z-index: 2;
    margin-bottom: 0.6rem;
}

:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_next_"]::before {
    content: 'next.stage';
}

:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_prev_"]::before {
    content: 'previous.stage';
}

:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_next_"]::after,
:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_prev_"]::after {
    content: '';
    position: absolute;
    inset: 0;
    opacity: 0.6;
    pointer-events: none;
    border-radius: inherit;
}

:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_next_"] div[data-testid="stButton"],
:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_prev_"] div[data-testid="stButton"] {
    margin: 0;
}

:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_next_"] div[data-testid="stButton"] button,
:is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_prev_"] div[data-testid="stButton"] button {
    width: 100%;
    justify-content: space-between;
    gap: 0.75rem;
}



.stage-top-grid__nav-card-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.stage-top-grid__nav-prompt {
    color: rgba(94, 234, 212, 0.82);
    font-weight: 700;
}

.stage-top-grid__nav-title {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 1.22rem;
    font-weight: 700;
    margin-top: 0.9rem;
    color: #f8fafc;
}

.stage-top-grid__nav-description {
    margin-top: 0.75rem;
    color: rgba(203, 213, 225, 0.86);
    font-size: 0.93rem;
    line-height: 1.65;
}

.stage-top-grid__gap {
    height: 0.85rem;
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) {
    position: relative;
    background: linear-gradient(182deg, rgba(8, 16, 29, 0.96), rgba(12, 25, 46, 0.92));
    border-radius: 20px;
    border: 1px solid rgba(56, 189, 248, 0.42);
    box-shadow: 0 32px 60px rgba(8, 47, 73, 0.5), inset 0 0 0 1px rgba(15, 23, 42, 0.82);
    padding: 1.45rem 1.65rem 1.55rem;
    color: rgba(226, 232, 240, 0.94);
    font-family: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    overflow: hidden;
    gap: 1.1rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title)::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at top right, rgba(59, 130, 246, 0.35), transparent 58%),
        radial-gradient(circle at bottom left, rgba(20, 184, 166, 0.32), transparent 64%);
    opacity: 0.75;
    pointer-events: none;
    border-radius: inherit;
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title)::after {
    content: '';
    position: absolute;
    inset: 2px;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.14);
    pointer-events: none;
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) > [data-testid="column"] {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    flex: 1 1 260px;
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) > [data-testid="column"]:first-of-type {
    align-items: flex-start;
    min-width: 0;
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) > [data-testid="column"]:last-of-type {
    flex: 0 0 auto;
    align-items: center;
    justify-content: center;
}

.nerd-toggle__title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.12rem;
    font-weight: 700;
    color: #f8fafc;
}

.nerd-toggle__title-text {
    display: inline-block;
    letter-spacing: 0.01em;
}

.nerd-toggle__icon {
    font-size: 1.35rem;
    line-height: 1;
}

.nerd-toggle__description {
    color: rgba(203, 213, 225, 0.88);
    font-size: 0.9rem;
    line-height: 1.65;
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) [data-testid="stToggle"] {
    display: flex;
    justify-content: flex-end;
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) label[data-testid="stToggle"] {
    justify-content: flex-end;
    align-items: center;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(13, 23, 38, 0.68);
    border: 1px solid rgba(56, 189, 248, 0.36);
    box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.88), 0 14px 32px rgba(7, 89, 133, 0.48);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) label[data-testid="stToggle"]:hover {
    transform: translateY(-1px);
    box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.85), 0 18px 36px rgba(7, 89, 133, 0.52);
    border-color: rgba(94, 234, 212, 0.45);
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) label[data-testid="stToggle"]:focus-within {
    outline: 2px solid rgba(94, 234, 212, 0.62);
    outline-offset: 2px;
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) label[data-testid="stToggle"] > div[role="switch"] {
    position: relative;
    width: 3.4rem;
    height: 1.92rem;
    border-radius: 999px;
    background: linear-gradient(160deg, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 0.9));
    box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.28), inset 0 -6px 12px rgba(15, 23, 42, 0.92);
    transition: background 0.25s ease, box-shadow 0.25s ease;
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) label[data-testid="stToggle"] > div[role="switch"]::before {
    content: 'OFF';
    position: absolute;
    left: 0.6rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    font-weight: 600;
    color: rgba(148, 163, 184, 0.78);
    transition: opacity 0.2s ease;
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) label[data-testid="stToggle"] > div[role="switch"] > div {
    width: 1.26rem;
    height: 1.26rem;
    margin: 0.32rem;
    border-radius: 999px;
    background: linear-gradient(145deg, rgba(226, 232, 240, 0.95), rgba(148, 163, 184, 0.9));
    box-shadow: 0 3px 8px rgba(15, 23, 42, 0.55);
    transition: transform 0.25s ease, background 0.25s ease, box-shadow 0.25s ease;
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) label[data-testid="stToggle"] > div[role="switch"][aria-checked="true"] {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.9), rgba(59, 130, 246, 0.82));
    box-shadow: inset 0 0 0 1px rgba(226, 232, 240, 0.42), inset 0 -6px 16px rgba(37, 99, 235, 0.55);
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) label[data-testid="stToggle"] > div[role="switch"][aria-checked="true"]::before {
    content: 'ON';
    left: auto;
    right: 0.65rem;
    color: rgba(15, 23, 42, 0.82);
}

[data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) label[data-testid="stToggle"] > div[role="switch"][aria-checked="true"] > div {
    background: linear-gradient(145deg, rgba(248, 250, 252, 0.98), rgba(191, 219, 254, 0.95));
    box-shadow: 0 4px 10px rgba(30, 64, 175, 0.55);
}

@media (max-width: 900px) {
    .stage-top-grid__nav-card,
    :is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_next_"],
    :is([data-testid="stElementContainer"], [data-testid="element-container"])[class*="st-key-stage_grid_prev_"] {
        padding: 1.1rem 1.2rem 1.2rem;
    }
    [data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) {
        padding: 1.2rem 1.3rem 1.3rem;
        gap: 1rem;
    }
    [data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) > [data-testid="column"] {
        flex: 1 1 100%;
        align-items: flex-start;
    }
    [data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) > [data-testid="column"]:last-of-type {
        width: 100%;
        align-items: stretch;
    }
    [data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) [data-testid="stToggle"] {
        justify-content: flex-start;
        width: 100%;
    }
    [data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) label[data-testid="stToggle"] {
        width: 100%;
        justify-content: space-between;
        padding: 0.4rem 0.8rem;
    }
}

@media (max-width: 600px) {
    [data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) {
        padding: 1.05rem 1.1rem 1.15rem;
        border-radius: 18px;
    }
    [data-testid="stHorizontalBlock"]:has(.nerd-toggle__title)::after {
        inset: 1.5px;
        border-radius: 16px;
    }
    .nerd-toggle__title {
        font-size: 1.05rem;
    }
    .nerd-toggle__description {
        font-size: 0.86rem;
    }
    [data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) label[data-testid="stToggle"] > div[role="switch"] {
        width: 3.2rem;
        height: 1.8rem;
    }
    [data-testid="stHorizontalBlock"]:has(.nerd-toggle__title) label[data-testid="stToggle"] > div[role="switch"] > div {
        margin: 0.28rem;
        width: 1.18rem;
        height: 1.18rem;
    }
}

</style>
"""
