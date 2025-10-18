"""Mission briefing component helpers for the overview stage."""

from __future__ import annotations

import html
import re
from textwrap import dedent
from typing import Iterable, Mapping


def mission_brief_styles() -> str:
    """Return the scoped CSS for the overview mission briefing."""

    return dedent(
        """
        <style>
          .mission-brief {
              background: rgba(255, 255, 255, 0.96);
              border-radius: 1.4rem;
              border: 1px solid rgba(15, 23, 42, 0.08);
              box-shadow: 0 28px 46px rgba(15, 23, 42, 0.12);
              padding: 1.75rem 1.9rem;
              display: grid;
              gap: 1.4rem;
          }
          .mission-brief__header {
              display: flex;
              gap: 1rem;
              align-items: center;
          }
          .mission-brief__icon {
              font-size: 2.1rem;
          }
          .mission-brief__eyebrow {
              font-size: 0.78rem;
              letter-spacing: 0.16em;
              text-transform: uppercase;
              color: rgba(37, 99, 235, 0.7);
              font-weight: 700;
          }
          .mission-brief__title {
              margin: 0;
              font-size: 1.45rem;
              font-weight: 700;
              color: #0f172a;
          }
          .mission-brief__bridge {
              margin: 0;
              font-size: 0.98rem;
              color: rgba(15, 23, 42, 0.75);
          }
          .mission-brief__grid {
              display: grid;
              grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
              gap: 1.8rem;
          }
          .mission-brief__objective {
              display: grid;
              gap: 0.95rem;
          }
          .mission-brief__list {
              margin: 0;
              padding-left: 1.2rem;
              display: grid;
              gap: 0.55rem;
              font-size: 0.92rem;
              color: rgba(15, 23, 42, 0.72);
          }
          .mission-brief__preview {
              display: grid;
              gap: 0.9rem;
          }
          .mission-brief__preview-card {
              background: rgba(37, 99, 235, 0.08);
              border-radius: 1.1rem;
              padding: 1.1rem 1.3rem;
              box-shadow: 0 16px 34px rgba(15, 23, 42, 0.12);
              display: grid;
              gap: 0.65rem;
          }
          .mission-brief__preview-card--mailbox {
              background: linear-gradient(135deg, rgba(37, 99, 235, 0.16), rgba(59, 130, 246, 0.14));
          }
          .mission-brief__preview-content {
              display: flex;
              flex-direction: column;
              gap: 0.85rem;
              width: 100%;
          }
          .mission-brief__preview-eyebrow {
              font-size: 0.72rem;
              letter-spacing: 0.14em;
              text-transform: uppercase;
              color: rgba(37, 99, 235, 0.8);
              font-weight: 700;
          }
          .mission-brief__preview-title {
              margin: 0;
              font-size: 1.1rem;
              color: #1d4ed8;
              font-weight: 600;
          }
          .mission-brief__preview-intro {
              margin: 0;
              font-size: 0.92rem;
              color: rgba(15, 23, 42, 0.7);
          }
          .mission-brief__inbox-list {
              list-style: none;
              margin: 0;
              padding: 0;
              display: flex;
              flex-direction: column;
              gap: 0.65rem;
          }
          .mission-brief__inbox-item {
              display: flex;
              flex-direction: column;
              gap: 0.25rem;
          }
          .mission-brief__subject {
              font-weight: 600;
              font-size: 0.96rem;
              color: #0f172a;
          }
          .mission-brief__snippet {
              font-size: 0.9rem;
              color: rgba(15, 23, 42, 0.68);
              line-height: 1.4;
          }
          .mission-brief__empty {
              font-size: 0.9rem;
              color: rgba(15, 23, 42, 0.65);
          }
          .mission-brief__preview-note {
              margin: 0;
              font-size: 0.85rem;
              color: rgba(15, 23, 42, 0.6);
          }
          .mission-brief__highlights {
              display: flex;
              flex-wrap: wrap;
              gap: 0.7rem;
          }
          .mission-highlight {
              display: inline-flex;
              align-items: center;
              gap: 0.55rem;
              background: rgba(255, 255, 255, 0.85);
              border-radius: 999px;
              padding: 0.55rem 0.95rem;
              font-size: 0.9rem;
              box-shadow: 0 10px 22px rgba(15, 23, 42, 0.12);
              border: 1px solid rgba(37, 99, 235, 0.18);
          }
          .mission-highlight__icon {
              font-size: 1.1rem;
          }
          .mailbox-preview {
              background: rgba(255, 255, 255, 0.9);
              border-radius: 1.35rem;
              border: 1px solid rgba(15, 23, 42, 0.08);
              box-shadow: 0 20px 44px rgba(15, 23, 42, 0.1);
              overflow: hidden;
          }
          .mailbox-preview__header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              padding: 1.1rem 1.4rem;
              background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(59, 130, 246, 0.1));
              border-bottom: 1px solid rgba(37, 99, 235, 0.16);
          }
          .mailbox-preview__header h4 {
              margin: 0;
              font-size: 1.05rem;
              font-weight: 600;
              color: #1e3a8a;
          }
          .mailbox-preview__header span {
              font-size: 0.85rem;
              color: rgba(15, 23, 42, 0.65);
          }
          .mail-rows {
              display: flex;
              flex-direction: column;
          }
          .mail-row {
              display: grid;
              grid-template-columns: auto 1fr auto;
              align-items: start;
              gap: 1rem;
              padding: 1rem 1.4rem;
              border-bottom: 1px solid rgba(15, 23, 42, 0.06);
              background: rgba(248, 250, 252, 0.7);
          }
          .mail-row:nth-child(even) {
              background: rgba(255, 255, 255, 0.92);
          }
          .mail-row__status {
              width: 12px;
              height: 12px;
              border-radius: 999px;
              background: linear-gradient(135deg, #1d4ed8, #2563eb);
              margin-top: 0.35rem;
              box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.18);
          }
          .mail-row__details {
              display: flex;
              flex-direction: column;
              gap: 0.35rem;
          }
          .mail-row__subject {
              margin: 0;
              font-size: 0.98rem;
              font-weight: 600;
              color: #0f172a;
          }
          .mail-row__snippet {
              margin: 0;
              font-size: 0.9rem;
              color: rgba(15, 23, 42, 0.68);
              line-height: 1.45;
          }
          .mail-row__meta {
              display: flex;
              flex-direction: column;
              align-items: flex-end;
              gap: 0.4rem;
              font-size: 0.8rem;
              color: rgba(15, 23, 42, 0.6);
          }
          .mail-row__tag {
              display: inline-flex;
              align-items: center;
              justify-content: center;
              padding: 0.25rem 0.6rem;
              border-radius: 999px;
              background: rgba(37, 99, 235, 0.14);
              color: #1d4ed8;
              font-weight: 600;
              font-size: 0.75rem;
              letter-spacing: 0.08em;
              text-transform: uppercase;
          }
          .mail-empty {
              padding: 1.35rem 1.4rem;
              font-size: 0.92rem;
              color: rgba(15, 23, 42, 0.65);
          }
          @media (max-width: 960px) {
              .mission-brief__grid {
                  grid-template-columns: 1fr;
                  gap: 1.4rem;
              }
              .mission-brief {
                  padding: 1.35rem 1.4rem;
              }
              .mission-brief__preview {
                  margin-top: 0.4rem;
              }
              .mail-row {
                  grid-template-columns: auto 1fr;
              }
              .mail-row__meta {
                  align-items: flex-start;
              }
          }
        </style>
        """
    ).strip()


def mission_brief_markup(*, mailbox_html: str) -> str:
    """Return the overview mission briefing HTML shell."""

    return dedent(
        f"""
        <div class="mission-brief">
            <div class="mission-brief__header">
                <span class="mission-brief__icon">🎯</span>
                <div>
                    <span class="mission-brief__eyebrow">Mission briefing</span>
                    <h3 class="mission-brief__title">Your mission</h3>
                </div>
            </div>
            <p class="mission-brief__bridge">You’re stepping into the control room of an email triage machine. The inbox snapshot on the right matches the live preview you’ll work from in a moment.</p>
            <div class="mission-brief__grid">
                <div class="mission-brief__objective">
                    <p>Keep unwanted email out while letting the important messages through. You’ll steer the controls, set the operating thresholds, and verify the system’s choices.</p>
                    <ul class="mission-brief__list">
                        <li>Scan the inbox feed and spot risky patterns early.</li>
                        <li>Decide how strict the spam filter should be and when autonomy applies.</li>
                        <li>Confirm or correct decisions so the system learns your judgement.</li>
                    </ul>
                </div>
                <div class="mission-brief__preview">
                    <div class="mission-brief__preview-card mission-brief__preview-card--mailbox">
                        <span class="mission-brief__preview-eyebrow">Inbox stream</span>
                        {mailbox_html}
                    </div>
                </div>
            </div>
        </div>
        """
    ).strip()


def mailbox_preview_markup(
    records: Iterable[Mapping[str, object]],
    *,
    snippet_limit: int = 110,
) -> str:
    """Return HTML for the mission briefing mailbox preview."""

    record_list = list(records)
    inbox_rows_html: list[str] = []

    for record in record_list:
        subject = html.escape(str(record.get("title", "Untitled email")))
        snippet = html.escape(_format_snippet(record.get("body"), limit=snippet_limit))
        inbox_rows_html.append(
            dedent(
                """
                <div class="mail-row">
                    <div class="mail-row__status"></div>
                    <div class="mail-row__details">
                        <p class="mail-row__subject">{subject}</p>
                        <p class="mail-row__snippet">{snippet}</p>
                    </div>
                    <div class="mail-row__meta">
                        <span class="mail-row__tag">Queued</span>
                    </div>
                </div>
                """
            ).format(subject=subject, snippet=snippet).strip()
        )

    if not inbox_rows_html:
        inbox_rows_html.append(
            """
            <div class="mail-empty">
                Inbox feed warming up — generate incoming emails in the Use stage to populate this preview.
            </div>
            """.strip()
        )

    return dedent(
        """
        <div class="mailbox-preview">
            <div class="mailbox-preview__header">
                <h4>Live inbox preview</h4>
                <span>First {count} messages waiting for triage</span>
            </div>
            <div class="mail-rows">{rows}</div>
        </div>
        """
    ).format(count=len(record_list) or 0, rows="".join(inbox_rows_html)).strip()


def _format_snippet(text: object | None, *, limit: int) -> str:
    snippet = (str(text or "")).strip()
    snippet = re.sub(r"\s+", " ", snippet)
    if len(snippet) > limit:
        snippet = snippet[: limit - 1].rstrip() + "…"
    return snippet
