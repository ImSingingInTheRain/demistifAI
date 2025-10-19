# demistifai/ui/components/train_intro.py
from __future__ import annotations

import html
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingNotesColumn:
    """Encapsulate the helper column that narrates the training animation."""

    html: str
    css: str


_TRAINING_NOTES_CSS = textwrap.dedent(
    """
    .mac-window .train-animation__notes {
      display: grid;
      gap: 0.75rem;
    }
    .mac-window .train-animation__notes h4 {
      margin: 0;
      font-size: 1.0rem;
      font-weight: 700;
      color: #0f172a;
    }
    .mac-window .train-animation__notes ul {
      margin: 0;
      padding-left: 1.15rem;
      display: grid;
      gap: 0.45rem;
      font-size: 0.95rem;
      line-height: 1.45;
      color: rgba(15, 23, 42, 0.78);
    }
    .mac-window .train-animation__notes li strong {
      color: #1d4ed8;
    }
    """
).strip()


_TRAIN_STAGE_CSS = textwrap.dedent(
    """
    .train-intro-card {
      background: linear-gradient(135deg, rgba(79, 70, 229, 0.16), rgba(14, 165, 233, 0.16));
      border-radius: 1.25rem;
      padding: 1.28rem 1.6rem;
      border: 1px solid rgba(15, 23, 42, 0.08);
      box-shadow: 0 18px 38px rgba(15, 23, 42, 0.1);
      margin-bottom: 1.8rem;
    }
    .train-intro-card__header {
      display: flex;
      gap: 1rem;
      align-items: flex-start;
      margin-bottom: 0.85rem;
    }
    .train-intro-card__icon {
      font-size: 1.75rem;
      line-height: 1;
      background: rgba(15, 23, 42, 0.08);
      border-radius: 1rem;
      padding: 0.55rem 0.95rem;
      box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.05);
    }
    .train-intro-card__eyebrow {
      font-size: 0.75rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      font-weight: 700;
      color: rgba(15, 23, 42, 0.55);
      display: inline-block;
    }
    .train-intro-card__title {
      margin: 0;
      font-size: 1.4rem;
      font-weight: 700;
      color: #0f172a;
    }
    .train-intro-card__body {
      margin: 0 0 1.15rem 0;
      color: rgba(15, 23, 42, 0.82);
      font-size: 0.95rem;
      line-height: 1.6;
    }
    .train-intro-card__steps {
      display: grid;
      gap: 0.75rem;
    }
    .train-intro-card__step {
      display: flex;
      gap: 0.75rem;
      align-items: flex-start;
    }
    .train-intro-card__step-index {
      width: 2rem;
      height: 2rem;
      border-radius: 999px;
      background: rgba(79, 70, 229, 0.18);
      color: rgba(30, 64, 175, 0.9);
      font-weight: 700;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 0.85rem;
    }
    .train-intro-card__step-body {
      font-size: 0.9rem;
      color: rgba(15, 23, 42, 0.8);
      line-height: 1.55;
    }
    .train-launchpad-card {
      border-radius: 1.1rem;
      border: 1px solid rgba(15, 23, 42, 0.08);
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 16px 36px rgba(15, 23, 42, 0.12);
      padding: 0.96rem 1.35rem;
      margin-bottom: 1rem;
    }
    .train-launchpad-card__title {
      margin: 0 0 0.65rem 0;
      font-size: 1.05rem;
      font-weight: 700;
      color: #0f172a;
    }
    .train-launchpad-card__list {
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 0.55rem;
    }
    .train-launchpad-card__list li {
      display: grid;
      grid-template-columns: 1.8rem 1fr;
      gap: 0.55rem;
      font-size: 0.9rem;
      color: rgba(15, 23, 42, 0.78);
      line-height: 1.4;
    }
    .train-launchpad-card__bullet {
      width: 1.8rem;
      height: 1.8rem;
      border-radius: 0.75rem;
      background: rgba(79, 70, 229, 0.12);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1rem;
    }
    .train-launchpad-card--secondary {
      background: rgba(30, 64, 175, 0.06);
      border: 1px dashed rgba(30, 64, 175, 0.35);
    }
    .train-how-card {
      border-radius: 1.1rem;
      border: 1px solid rgba(15, 23, 42, 0.08);
      padding: 0.96rem 1.35rem;
      background: rgba(255, 255, 255, 0.94);
      box-shadow: 0 14px 32px rgba(15, 23, 42, 0.1);
    }
    .train-how-card__header {
      display: flex;
      gap: 0.75rem;
      align-items: center;
      margin-bottom: 0.75rem;
    }
    .train-how-card__icon {
      font-size: 1.35rem;
    }
    .train-how-card__title {
      margin: 0;
      font-size: 1.05rem;
      font-weight: 700;
      color: #0f172a;
    }
    .train-how-card__body {
      margin: 0 0 0.85rem 0;
      font-size: 0.9rem;
      color: rgba(15, 23, 42, 0.78);
      line-height: 1.55;
    }
    .train-how-card__steps {
      margin: 0 0 0.9rem 1.15rem;
      padding: 0;
      font-size: 0.9rem;
      color: rgba(15, 23, 42, 0.82);
    }
    .train-how-card__grid {
      display: grid;
      gap: 0.9rem;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      margin-bottom: 0.6rem;
    }
    .train-how-card__panel {
      position: relative;
      border-radius: 1rem;
      border: 1px solid rgba(79, 70, 229, 0.16);
      background: rgba(79, 70, 229, 0.06);
      padding: 0.85rem 1rem;
      display: grid;
      gap: 0.35rem;
    }
    .train-how-card__panel-icon {
      font-size: 1.1rem;
    }
    .train-how-card__panel-title {
      margin: 0;
      font-size: 0.9rem;
      font-weight: 700;
      color: rgba(30, 64, 175, 0.95);
    }
    .train-how-card__panel-body {
      margin: 0;
      font-size: 0.85rem;
      color: rgba(15, 23, 42, 0.75);
      line-height: 1.5;
    }
    .train-how-card__divider {
      height: 1px;
      background: linear-gradient(90deg, rgba(15, 23, 42, 0), rgba(15, 23, 42, 0.25), rgba(15, 23, 42, 0));
      margin: 0.75rem 0 0.9rem 0;
    }
    .train-how-card__body--muted {
      color: rgba(15, 23, 42, 0.6);
      font-size: 0.88rem;
      margin-bottom: 0.65rem;
    }
    .train-how-card__step-grid {
      display: grid;
      gap: 0.75rem;
    }
    .train-how-card__step-box {
      border-radius: 1rem;
      border: 1px solid rgba(15, 23, 42, 0.08);
      background: linear-gradient(135deg, rgba(14, 116, 144, 0.08), rgba(59, 130, 246, 0.08));
      padding: 0.95rem 1.1rem;
      display: grid;
      gap: 0.5rem;
    }
    .train-how-card__step-label {
      display: flex;
      align-items: center;
      gap: 0.55rem;
    }
    .train-how-card__step-number {
      width: 2rem;
      height: 2rem;
      border-radius: 999px;
      background: rgba(14, 116, 144, 0.18);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 0.95rem;
      color: rgba(12, 74, 110, 0.95);
    }
    .train-how-card__step-icon {
      font-size: 1.25rem;
    }
    .train-how-card__step-title {
      font-weight: 700;
      font-size: 0.95rem;
      color: rgba(15, 23, 42, 0.9);
    }
    .train-how-card__step-list {
      margin: 0;
      padding-left: 1.2rem;
      font-size: 0.88rem;
      color: rgba(15, 23, 42, 0.78);
      line-height: 1.55;
    }
    .train-how-card__step-sublist {
      margin: 0.45rem 0 0 1.1rem;
      padding-left: 1rem;
      font-size: 0.85rem;
      color: rgba(15, 23, 42, 0.75);
      line-height: 1.5;
    }
    .train-how-card__step-example {
      margin: 0;
      font-size: 0.82rem;
      color: rgba(15, 23, 42, 0.68);
      background: rgba(255, 255, 255, 0.6);
      border-radius: 0.75rem;
      padding: 0.55rem 0.75rem;
      border: 1px dashed rgba(12, 74, 110, 0.25);
    }
    .train-launchpad-card__grid {
      display: grid;
      gap: 0.65rem;
    }
    .train-launchpad-card__item {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 0.6rem;
      padding: 0.6rem 0.75rem;
      border-radius: 0.85rem;
      background: rgba(15, 23, 42, 0.04);
      align-items: flex-start;
    }
    .train-launchpad-card__badge {
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }
    .train-launchpad-card__badge-num {
      width: 1.8rem;
      height: 1.8rem;
      border-radius: 999px;
      background: rgba(79, 70, 229, 0.16);
      color: rgba(30, 64, 175, 0.92);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 0.85rem;
    }
    .train-launchpad-card__badge-icon {
      font-size: 1.05rem;
    }
    .train-launchpad-card__item-title {
      margin: 0;
      font-size: 0.9rem;
      font-weight: 600;
      color: #0f172a;
    }
    .train-launchpad-card__status {
      display: inline-block;
      background: rgba(14, 165, 233, 0.12);
      color: #0369a1;
      border-radius: 999px;
      padding: 0.1rem 0.6rem;
      font-size: 0.75rem;
      font-weight: 600;
      margin-left: 0.4rem;
    }
    .train-launchpad-card__item-body {
      margin: 0.2rem 0 0 0;
      font-size: 0.83rem;
      color: rgba(15, 23, 42, 0.75);
      line-height: 1.45;
    }
    .train-token-chip {
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
      border-radius: 999px;
      padding: 0.4rem 0.85rem;
      background: rgba(59, 130, 246, 0.12);
      color: rgba(30, 64, 175, 0.9);
      font-size: 0.8rem;
      font-weight: 600;
    }
    .numeric-clue-card-grid {
      display: grid;
      gap: 0.85rem;
    }
    .numeric-clue-card {
      border-radius: 1rem;
      border: 1px solid rgba(15, 23, 42, 0.08);
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(14, 165, 233, 0.08));
      box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
      padding: 0.85rem 1rem;
    }
    .numeric-clue-card__header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 0.75rem;
    }
    .numeric-clue-card__subject {
      font-size: 0.95rem;
      font-weight: 700;
      color: #0f172a;
      flex: 1 1 auto;
    }
    .numeric-clue-card__tags {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      justify-content: flex-end;
    }
    .numeric-clue-tag {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 0.2rem 0.6rem;
      font-size: 0.72rem;
      font-weight: 600;
      background: rgba(15, 23, 42, 0.06);
      color: rgba(15, 23, 42, 0.7);
    }
    .numeric-clue-tag--truth {
      background: rgba(15, 23, 42, 0.08);
      color: rgba(15, 23, 42, 0.65);
    }
    .numeric-clue-tag--spam {
      background: rgba(239, 68, 68, 0.18);
      color: #b91c1c;
    }
    .numeric-clue-tag--safe {
      background: rgba(59, 130, 246, 0.18);
      color: #1d4ed8;
    }
    .numeric-clue-tag--unknown {
      background: rgba(148, 163, 184, 0.22);
      color: rgba(30, 41, 59, 0.72);
    }
    .numeric-clue-chip {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 0.28rem 0.65rem;
      font-size: 0.76rem;
      font-weight: 600;
      background: rgba(15, 23, 42, 0.08);
      color: rgba(15, 23, 42, 0.75);
    }
    .numeric-clue-chip--spam {
      background: rgba(239, 68, 68, 0.18);
      color: #b91c1c;
    }
    .numeric-clue-chip--safe {
      background: rgba(34, 197, 94, 0.18);
      color: #15803d;
    }
    .numeric-clue-card__chips {
      margin-top: 0.65rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
    }
    .numeric-clue-card__reason {
      margin-top: 0.55rem;
      font-size: 0.82rem;
      color: rgba(15, 23, 42, 0.78);
    }
    .numeric-clue-card__meta {
      margin-top: 0.45rem;
      font-size: 0.72rem;
      color: rgba(15, 23, 42, 0.62);
    }
    .numeric-clue-preview {
      border-radius: 1rem;
      border: 1px dashed rgba(37, 99, 235, 0.35);
      background: rgba(191, 219, 254, 0.28);
      padding: 1rem 1.1rem;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.4);
    }
    .numeric-clue-preview__header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 0.5rem;
      font-size: 0.85rem;
      font-weight: 600;
      color: rgba(30, 64, 175, 0.85);
      margin-bottom: 0.6rem;
    }
    .numeric-clue-preview__band {
      border-radius: 0.85rem;
      background: linear-gradient(90deg, rgba(59, 130, 246, 0.14), rgba(14, 165, 233, 0.16));
      border: 1px dashed rgba(37, 99, 235, 0.45);
      padding: 0.9rem 0.9rem 0.95rem;
    }
    .numeric-clue-preview__ticks {
      display: flex;
      justify-content: space-between;
      font-size: 0.75rem;
      color: rgba(30, 58, 138, 0.78);
      margin-bottom: 0.7rem;
    }
    .numeric-clue-preview__chips {
      display: flex;
      flex-wrap: wrap;
      gap: 0.45rem;
    }
    .numeric-clue-preview__chip {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 0.3rem 0.7rem;
      font-size: 0.78rem;
      font-weight: 600;
      background: rgba(255, 255, 255, 0.78);
      color: rgba(30, 41, 59, 0.8);
      box-shadow: 0 6px 16px rgba(37, 99, 235, 0.18);
    }
    .numeric-clue-preview__note {
      margin: 0.8rem 0 0 0;
      font-size: 0.78rem;
      color: rgba(30, 41, 59, 0.72);
      line-height: 1.4;
    }
    .train-inline-note {
      margin-top: 0.35rem;
      font-size: 0.8rem;
      color: rgba(30, 64, 175, 0.82);
      background: rgba(191, 219, 254, 0.35);
      border-radius: 0.6rem;
      padding: 0.45rem 0.75rem;
      display: inline-block;
    }
    .train-action-card {
      border-radius: 1.1rem;
      border: 1px solid rgba(15, 23, 42, 0.08);
      padding: 1rem 1.35rem;
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(14, 165, 233, 0.12));
      box-shadow: 0 16px 34px rgba(15, 23, 42, 0.1);
      margin-bottom: 0.75rem;
    }
    .train-action-card__eyebrow {
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-weight: 700;
      color: rgba(30, 64, 175, 0.75);
    }
    .train-action-card__title {
      margin: 0.35rem 0 0.6rem 0;
      font-size: 1.25rem;
      font-weight: 700;
      color: #0f172a;
    }
    .train-action-card__body {
      margin: 0;
      font-size: 0.9rem;
      color: rgba(15, 23, 42, 0.78);
      line-height: 1.5;
    }
    .train-context-card {
      border-radius: 1rem;
      border: 1px solid rgba(15, 23, 42, 0.08);
      padding: 0.8rem 1.1rem;
      background: rgba(255, 255, 255, 0.94);
      box-shadow: 0 10px 26px rgba(15, 23, 42, 0.08);
      margin-bottom: 0.9rem;
    }
    .train-context-card h5 {
      margin: 0 0 0.5rem 0;
      font-size: 1rem;
      font-weight: 700;
      color: #0f172a;
    }
    .train-context-card ul {
      margin: 0;
      padding-left: 1.1rem;
      font-size: 0.88rem;
      color: rgba(15, 23, 42, 0.78);
      line-height: 1.5;
    }
    .train-context-card--tip {
      background: rgba(236, 233, 254, 0.6);
      border: 1px dashed rgba(79, 70, 229, 0.35);
      color: rgba(55, 48, 163, 0.95);
    }
    .train-band-card {
      border-radius: 1rem;
      border: 1px solid rgba(79, 70, 229, 0.25);
      padding: 0.95rem 1.1rem 1.05rem 1.1rem;
      background: rgba(79, 70, 229, 0.08);
      box-shadow: inset 0 0 0 1px rgba(79, 70, 229, 0.08);
    }
    .train-band-card__title {
      margin: 0 0 0.55rem 0;
      font-size: 0.95rem;
      font-weight: 700;
      color: rgba(49, 46, 129, 0.9);
    }
    .train-band-card__bar {
      position: relative;
      width: 100%;
      height: 16px;
      border-radius: 999px;
      background: rgba(30, 64, 175, 0.12);
    }
    .train-band-card__band {
      position: absolute;
      top: 0;
      bottom: 0;
      border-radius: 999px;
      background: rgba(59, 130, 246, 0.35);
    }
    .train-band-card__threshold {
      position: absolute;
      top: -4px;
      bottom: -4px;
      width: 2px;
      border-radius: 2px;
      background: rgba(30, 64, 175, 0.95);
    }
    .train-band-card__scale {
      display: flex;
      justify-content: space-between;
      font-size: 0.68rem;
      color: rgba(30, 64, 175, 0.8);
      margin-top: 0.35rem;
    }
    .train-band-card__caption {
      margin-top: 0.45rem;
      font-size: 0.75rem;
      color: rgba(49, 46, 129, 0.85);
    }
    .train-band-card__hint {
      margin-top: 0.35rem;
      font-size: 0.78rem;
      color: rgba(49, 46, 129, 0.75);
    }
    .train-nerd-intro {
      border-radius: 1rem;
      border: 1px solid rgba(15, 23, 42, 0.08);
      padding: 1rem 1.1rem;
      background: rgba(14, 165, 233, 0.08);
      box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
      margin-bottom: 1rem;
    }
    .train-nerd-intro h4 {
      margin: 0 0 0.35rem 0;
      font-size: 1.05rem;
      font-weight: 700;
      color: rgba(7, 89, 133, 0.9);
    }
    .train-nerd-intro p {
      margin: 0;
      font-size: 0.88rem;
      color: rgba(7, 89, 133, 0.78);
      line-height: 1.5;
    }
    .train-nerd-hint {
      margin-top: 0.85rem;
      font-size: 0.8rem;
      color: rgba(15, 23, 42, 0.7);
      background: rgba(191, 219, 254, 0.45);
      border-radius: 0.75rem;
      padding: 0.55rem 0.75rem;
    }
    """
).strip()


def build_training_notes_column() -> TrainingNotesColumn:
    """Return the HTML and scoped CSS for the training animation notes column."""

    notes_html = textwrap.dedent(
        """
        <div class="train-animation__notes">
          <h4>Training loop snapshot</h4>
          <ul>
            <li><strong>MiniLM embeddings</strong> map emails into a 384D space.</li>
            <li><strong>Logistic regression</strong> learns a separating boundary.</li>
            <li><strong>Epoch playback</strong> shows how spam/work clusters settle.</li>
            <li>Toggle Nerd Mode to inspect features and weights in detail.</li>
          </ul>
        </div>
        """
    ).strip()
    return TrainingNotesColumn(html=notes_html, css=_TRAINING_NOTES_CSS)


def training_stage_stylesheet() -> str:
    """Return the shared stylesheet for the Train stage UI helpers."""

    return f"<style>\n{_TRAIN_STAGE_CSS}\n</style>"


def build_train_intro_card(*, stage_number: int, icon: str, title: str) -> str:
    """Return the intro card markup for the Train stage."""

    icon_html = html.escape(icon)
    title_html = html.escape(title)

    return textwrap.dedent(
        f"""
        <div class="train-intro-card">
          <div class="train-intro-card__header">
            <span class="train-intro-card__icon">{icon_html}</span>
            <div>
              <span class="train-intro-card__eyebrow">Stage {stage_number}</span>
              <h4 class="train-intro-card__title">{title_html}</h4>
            </div>
          </div>
          <div class="train-how-card__header">
            <div>
              <h5 class="train-how-card__title">Now your AI system will learn how to achieve an objective</h5>
              <p class="train-how-card__body">Think of this as teaching your AI assistant how to tell the difference between spam and safe emails.</p>
            </div>
          </div>
          <div class="train-how-card__grid">
            <div class="train-how-card__panel">
              <div class="train-how-card__panel-icon">🫶</div>
              <h6 class="train-how-card__panel-title">Your part</h6>
              <p class="train-how-card__panel-body">Provide examples with clear labels (“This one is spam, that one is safe”).</p>
            </div>
            <div class="train-how-card__panel">
              <div class="train-how-card__panel-icon">🤖</div>
              <h6 class="train-how-card__panel-title">The system’s part</h6>
              <p class="train-how-card__panel-body">Spot patterns that generalize to emails it hasn’t seen yet.</p>
            </div>
          </div>
        </div>
        """
    ).strip()


def build_launchpad_card(title: str = "🧭 Training Launchpad — readiness & controls") -> str:
    """Return the launchpad shell card markup."""

    title_html = html.escape(title)
    return textwrap.dedent(
        f"""
        <div class="train-launchpad-card">
          <div class="train-launchpad-card__title">{title_html}</div>
        </div>
        """
    ).strip()


def build_launchpad_status_item(*, title: str, status: str, body: str) -> str:
    """Return a launchpad readiness item with a status badge."""

    title_html = html.escape(title)
    status_html = html.escape(status)
    body_html = html.escape(body)
    return textwrap.dedent(
        f"""
        <div class="train-launchpad-card__item">
          <p class="train-launchpad-card__item-title">{title_html}<span class="train-launchpad-card__status">{status_html}</span></p>
          <p class="train-launchpad-card__item-body">{body_html}</p>
        </div>
        """
    ).strip()


def build_token_chip(text: str) -> str:
    """Return the numeric token budget chip markup."""

    return f"<div class='train-token-chip'>{html.escape(text)}</div>"


def build_inline_note(text: str) -> str:
    """Return the inline informational note markup."""

    return f"<div class='train-inline-note'>{html.escape(text)}</div>"


def build_nerd_intro_card() -> str:
    """Return the Nerd Mode intro card markup."""

    return textwrap.dedent(
        """
        <div class="train-nerd-intro">
          <h4>🛡️ Advanced split & guardrail controls</h4>
          <p>Fine-tune how much data we hold out, solver behaviour, and the numeric assist rules that complement the text model.</p>
        </div>
        """
    ).strip()
