"""Mission briefing component helpers for the overview stage."""

from __future__ import annotations

import html
import re
from textwrap import dedent
from typing import Iterable, Mapping

from demistifai.ui.components.shared.macos_iframe_window import MacWindowPane


_MISSION_BRIEF_CSS = dedent(
    """
    <style>
      .mission-brief {
          display: grid;
          gap: 1.4rem;
          font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          color: #0f172a;
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
      .mission-brief__bridge,
      .mission-brief__summary {
          margin: 0;
          font-size: 0.98rem;
          color: rgba(15, 23, 42, 0.75);
          line-height: 1.55;
      }
      .mission-brief__summary {
          color: rgba(15, 23, 42, 0.72);
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
          line-height: 1.5;
      }
      .mission-brief__list li::marker {
          color: rgba(37, 99, 235, 0.7);
      }
      @media (max-width: 960px) {
          .mission-brief {
              gap: 1.2rem;
          }
      }
    </style>
    """
).strip()


_MAILBOX_PREVIEW_CSS = dedent(
    """
    <style>
      .mailbox-preview {
          background: rgba(255, 255, 255, 0.85);
          border-radius: 1.1rem;
          border: 1px solid rgba(37, 99, 235, 0.16);
          overflow: hidden;
          display: grid;
          gap: 0;
          font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          color: #0f172a;
      }
      .mailbox-preview__header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1.05rem 1.3rem;
          background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(59, 130, 246, 0.08));
          border-bottom: 1px solid rgba(37, 99, 235, 0.18);
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
          padding: 0.95rem 1.3rem;
          border-bottom: 1px solid rgba(15, 23, 42, 0.06);
          background: rgba(248, 250, 252, 0.78);
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
          box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
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
          padding: 1.2rem 1.3rem;
          font-size: 0.92rem;
          color: rgba(15, 23, 42, 0.65);
      }
      @media (max-width: 960px) {
          .mail-row {
              grid-template-columns: auto 1fr;
          }
          .mail-row__meta {
              align-items: flex-start;
          }
      }
      @media (max-width: 768px) {
          .mail-row {
              grid-template-columns: 1fr;
              padding: 0.78rem 1rem;
              gap: 0.6rem;
          }
          .mail-row__status {
              display: flex;
              align-items: center;
              justify-content: flex-start;
              grid-row: 1;
              grid-column: 1;
              margin-top: 0.1rem;
          }
          .mail-row__details {
              grid-row: 1;
              grid-column: 1;
              padding-left: 1.6rem;
              gap: 0.3rem;
          }
          .mail-row__subject {
              line-height: 1.3;
          }
      }
      @media (max-width: 640px) {
          .mail-row__details {
              font-size: 0.88rem;
          }
          .mail-row__meta {
              grid-column: 1 / -1;
              margin-top: 0.35rem;
              align-items: flex-start;
              gap: 0.3rem;
          }
          .mail-row__tag {
              padding: 0.2rem 0.5rem;
              font-size: 0.72rem;
          }
      }
    </style>
    """
).strip()


def mission_brief_styles() -> str:
    """Return scoped CSS for the mission briefing pane."""

    return _MISSION_BRIEF_CSS


def mailbox_preview_styles() -> str:
    """Return scoped CSS for the mailbox preview pane."""

    return _MAILBOX_PREVIEW_CSS

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


def mission_brief_pane() -> MacWindowPane:
    """Return the mission briefing pane for iframe embedding."""

    return MacWindowPane(
        html=mission_overview_column_markup(),
        css=mission_brief_styles(),
        min_height=420,
        pane_id="overview-mission-brief",
    )


def mailbox_preview_pane(
    records: Iterable[Mapping[str, object]], *, snippet_limit: int = 110
) -> MacWindowPane:
    """Return the mailbox preview pane for iframe embedding."""

    return MacWindowPane(
        html=mailbox_preview_markup(records, snippet_limit=snippet_limit),
        css=mailbox_preview_styles(),
        min_height=420,
        pane_id="overview-mission-mailbox",
    )


def _format_snippet(text: object | None, *, limit: int) -> str:
    snippet = (str(text or "")).strip()
    snippet = re.sub(r"\s+", " ", snippet)
    if len(snippet) > limit:
        snippet = snippet[: limit - 1].rstrip() + "…"
    return snippet
