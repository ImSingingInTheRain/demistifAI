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
          .mac-window .mission-brief {
              display: grid;
              gap: 1.4rem;
          }
          .mac-window .mission-brief__header {
              display: flex;
              gap: 1rem;
              align-items: center;
          }
          .mac-window .mission-brief__icon {
              font-size: 2.1rem;
          }
          .mac-window .mission-brief__eyebrow {
              font-size: 0.78rem;
              letter-spacing: 0.16em;
              text-transform: uppercase;
              color: rgba(37, 99, 235, 0.7);
              font-weight: 700;
          }
          .mac-window .mission-brief__title {
              margin: 0;
              font-size: 1.45rem;
              font-weight: 700;
              color: #0f172a;
          }
          .mac-window .mission-brief__bridge,
          .mac-window .mission-brief__summary {
              margin: 0;
              font-size: 0.98rem;
              color: rgba(15, 23, 42, 0.75);
              line-height: 1.55;
          }
          .mac-window .mission-brief__summary {
              color: rgba(15, 23, 42, 0.72);
          }
          .mac-window .mission-brief__objective {
              display: grid;
              gap: 0.95rem;
          }
          .mac-window .mission-brief__list {
              margin: 0;
              padding-left: 1.2rem;
              display: grid;
              gap: 0.55rem;
              font-size: 0.92rem;
              color: rgba(15, 23, 42, 0.72);
              line-height: 1.5;
          }
          .mac-window .mission-brief__list li::marker {
              color: rgba(37, 99, 235, 0.7);
          }
          .mac-window .mailbox-preview {
              background: rgba(255, 255, 255, 0.85);
              border-radius: 1.1rem;
              border: 1px solid rgba(37, 99, 235, 0.16);
              overflow: hidden;
              display: grid;
              gap: 0;
          }
          .mac-window .mailbox-preview__header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              padding: 1.05rem 1.3rem;
              background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(59, 130, 246, 0.08));
              border-bottom: 1px solid rgba(37, 99, 235, 0.18);
          }
          .mac-window .mailbox-preview__header h4 {
              margin: 0;
              font-size: 1.05rem;
              font-weight: 600;
              color: #1e3a8a;
          }
          .mac-window .mailbox-preview__header span {
              font-size: 0.85rem;
              color: rgba(15, 23, 42, 0.65);
          }
          .mac-window .mail-rows {
              display: flex;
              flex-direction: column;
          }
          .mac-window .mail-row {
              display: grid;
              grid-template-columns: auto 1fr auto;
              align-items: start;
              gap: 1rem;
              padding: 0.95rem 1.3rem;
              border-bottom: 1px solid rgba(15, 23, 42, 0.06);
              background: rgba(248, 250, 252, 0.78);
          }
          .mac-window .mail-row:nth-child(even) {
              background: rgba(255, 255, 255, 0.92);
          }
          .mac-window .mail-row__status {
              width: 12px;
              height: 12px;
              border-radius: 999px;
              background: linear-gradient(135deg, #1d4ed8, #2563eb);
              margin-top: 0.35rem;
              box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
          }
          .mac-window .mail-row__details {
              display: flex;
              flex-direction: column;
              gap: 0.35rem;
          }
          .mac-window .mail-row__subject {
              margin: 0;
              font-size: 0.98rem;
              font-weight: 600;
              color: #0f172a;
          }
          .mac-window .mail-row__snippet {
              margin: 0;
              font-size: 0.9rem;
              color: rgba(15, 23, 42, 0.68);
              line-height: 1.45;
          }
          .mac-window .mail-row__meta {
              display: flex;
              flex-direction: column;
              align-items: flex-end;
              gap: 0.4rem;
              font-size: 0.8rem;
              color: rgba(15, 23, 42, 0.6);
          }
          .mac-window .mail-row__tag {
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
          .mac-window .mail-empty {
              padding: 1.2rem 1.3rem;
              font-size: 0.92rem;
              color: rgba(15, 23, 42, 0.65);
          }
          @media (max-width: 960px) {
              .mac-window .mission-brief {
                  gap: 1.2rem;
              }
              .mac-window .mail-row {
                  grid-template-columns: auto 1fr;
              }
              .mac-window .mail-row__meta {
                  align-items: flex-start;
              }
          }
          @media (max-width: 768px) {
              .mac-window .mail-row {
                  grid-template-columns: 1fr;
                  padding: 0.78rem 1rem;
                  gap: 0.6rem;
              }
              .mac-window .mail-row__status {
                  display: flex;
                  align-items: center;
                  justify-content: flex-start;
                  grid-row: 1;
                  grid-column: 1;
                  margin-top: 0.1rem;
              }
              .mac-window .mail-row__details {
                  grid-row: 1;
                  grid-column: 1;
                  padding-left: 1.6rem;
                  gap: 0.3rem;
              }
              .mac-window .mail-row__subject {
                  line-height: 1.3;
              }
              .mac-window .mail-row__snippet {
                  font-size: 0.88rem;
              }
              .mac-window .mail-row__meta {
                  grid-column: 1 / -1;
                  margin-top: 0.35rem;
                  align-items: flex-start;
                  gap: 0.3rem;
              }
              .mac-window .mail-row__tag {
                  padding: 0.2rem 0.5rem;
                  font-size: 0.72rem;
              }
          }
        </style>
        """
    ).strip()


def mission_overview_column_markup() -> str:
    """Return HTML for the mission overview column."""

    return dedent(
        """
        <div class="mission-brief">
            <div class="mission-brief__header">
                <span class="mission-brief__icon">🎯</span>
                <div>
                    <span class="mission-brief__eyebrow">Mission briefing</span>
                    <h3 class="mission-brief__title">Your mission</h3>
                </div>
            </div>
            <p class="mission-brief__bridge">You’re stepping into the control room of an email triage machine. The inbox snapshot on the right mirrors the live preview you’ll work from in a moment.</p>
            <div class="mission-brief__objective">
                <p class="mission-brief__summary">Keep unwanted email out while letting the important messages through. You’ll steer the controls, set the operating thresholds, and verify the system’s choices.</p>
                <ul class="mission-brief__list">
                    <li>Scan the inbox feed and spot risky patterns early.</li>
                    <li>Decide how strict the spam filter should be and when autonomy applies.</li>
                    <li>Confirm or correct decisions so the system learns your judgement.</li>
                </ul>
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
