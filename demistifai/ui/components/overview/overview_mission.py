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
          background: linear-gradient(160deg, rgba(37, 99, 235, 0.06), rgba(37, 99, 235, 0));
          border: 1px solid rgba(37, 99, 235, 0.16);
          border-radius: 1.2rem;
          padding: 1.5rem 1.75rem;
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
              gap: 1.25rem;
              padding: 1.35rem 1.55rem;
          }
      }
      @media (max-width: 768px) {
          .mission-brief {
              padding: 1.2rem 1.35rem;
          }
          .mission-brief__title {
              font-size: 1.32rem;
          }
          .mission-brief__bridge,
          .mission-brief__summary {
              font-size: 0.94rem;
          }
      }
      @media (max-width: 600px) {
          .mission-brief {
              padding: 1.1rem 1.25rem;
              border-radius: 1rem;
          }
          .mission-brief__header {
              align-items: flex-start;
              gap: 0.8rem;
          }
          .mission-brief__title {
              font-size: 1.26rem;
          }
          .mission-brief__bridge,
          .mission-brief__summary {
              line-height: 1.55;
          }
      }
      @media (max-width: 480px) {
          .mission-brief {
              gap: 1.05rem;
              padding: 1rem 1.15rem 1.15rem;
              box-shadow: 0 16px 40px -28px rgba(15, 23, 42, 0.45);
          }
          .mission-brief__header {
              flex-direction: column;
              align-items: flex-start;
              gap: 0.45rem;
          }
          .mission-brief__icon {
              font-size: 1.8rem;
          }
          .mission-brief__eyebrow {
              font-size: 0.72rem;
              letter-spacing: 0.14em;
          }
          .mission-brief__title {
              font-size: 1.18rem;
          }
          .mission-brief__bridge,
          .mission-brief__summary {
              font-size: 0.9rem;
          }
      }
    </style>
    """
).strip()


_MAILBOX_PREVIEW_CSS = dedent(
    """
    <style>
      .mailbox-preview {
          background: rgba(255, 255, 255, 0.9);
          border-radius: 1.1rem;
          border: 1px solid rgba(37, 99, 235, 0.16);
          overflow: hidden;
          display: grid;
          gap: 0;
          font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          color: #0f172a;
          box-shadow: 0 20px 40px -32px rgba(15, 23, 42, 0.45);
      }
      .mailbox-preview__header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 1.05rem 1.3rem;
          background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(59, 130, 246, 0.08));
          border-bottom: 1px solid rgba(37, 99, 235, 0.18);
          gap: 0.75rem;
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
          display: flex;
          flex-direction: column;
          gap: 0.6rem;
          padding: 0.9rem 1.3rem 1rem;
          border-bottom: 1px solid rgba(15, 23, 42, 0.06);
          background: rgba(248, 250, 252, 0.78);
      }
      .mail-row:nth-child(even) {
          background: rgba(255, 255, 255, 0.92);
      }
      .mail-row__header {
          display: flex;
          align-items: flex-start;
          gap: 0.85rem;
      }
      .mail-row__status {
          width: 12px;
          height: 12px;
          border-radius: 999px;
          background: linear-gradient(135deg, #1d4ed8, #2563eb);
          margin-top: 0.25rem;
          box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
          flex-shrink: 0;
      }
      .mail-row__details {
          display: flex;
          flex-direction: column;
          gap: 0.45rem;
          min-width: 0;
      }
      .mail-row__subject-line {
          display: flex;
          align-items: center;
          gap: 0.6rem;
          flex-wrap: wrap;
      }
      .mail-row__subject {
          margin: 0;
          font-size: 0.98rem;
          font-weight: 600;
          color: #0f172a;
          flex: 1 1 auto;
          line-height: 1.35;
          word-break: break-word;
      }
      .mail-row__tag {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          padding: 0.25rem 0.65rem;
          border-radius: 999px;
          background: rgba(37, 99, 235, 0.16);
          color: #1d4ed8;
          font-weight: 600;
          font-size: 0.74rem;
          letter-spacing: 0.08em;
          text-transform: uppercase;
      }
      .mail-row__snippet {
          margin: 0;
          font-size: 0.9rem;
          color: rgba(15, 23, 42, 0.7);
          line-height: 1.5;
          word-break: break-word;
      }
      .mail-empty {
          padding: 1.2rem 1.3rem;
          font-size: 0.92rem;
          color: rgba(15, 23, 42, 0.65);
      }
      @media (max-width: 1024px) {
          .mailbox-preview {
              border-radius: 1rem;
          }
      }
      @media (max-width: 960px) {
          .mailbox-preview__header {
              flex-direction: column;
              align-items: flex-start;
              padding: 1rem 1.15rem;
              gap: 0.35rem;
          }
          .mailbox-preview__header h4 {
              font-size: 1.02rem;
          }
      }
      @media (max-width: 820px) {
          .mailbox-preview {
              box-shadow: 0 16px 40px -30px rgba(15, 23, 42, 0.5);
          }
          .mail-row {
              padding: 0.85rem 1rem 0.95rem;
              gap: 0.55rem;
          }
      }
      @media (max-width: 720px) {
          .mail-row__header {
              gap: 0.65rem;
          }
          .mail-row__subject-line {
              gap: 0.5rem;
          }
      }
      @media (max-width: 640px) {
          .mailbox-preview {
              border-radius: 0.95rem;
          }
          .mail-row {
              padding: 0.8rem 0.9rem 0.9rem;
          }
          .mail-row__subject {
              font-size: 0.95rem;
          }
          .mail-row__snippet {
              font-size: 0.88rem;
          }
      }
      @media (max-width: 520px) {
          .mailbox-preview__header {
              background: linear-gradient(135deg, rgba(37, 99, 235, 0.18), rgba(59, 130, 246, 0.12));
              padding: 0.9rem 1.05rem;
              border-bottom-width: 0;
              border-bottom-left-radius: 0;
              border-bottom-right-radius: 0;
          }
          .mailbox-preview__header h4 {
              font-size: 0.98rem;
          }
          .mailbox-preview__header span {
              font-size: 0.8rem;
          }
          .mail-row__subject-line {
              flex-direction: column;
              align-items: flex-start;
          }
          .mail-row__tag {
              margin-top: 0.1rem;
          }
      }
      @media (max-width: 460px) {
          .mailbox-preview {
              border-radius: 0.9rem;
          }
          .mail-row {
              padding: 0.78rem 0.85rem 0.85rem;
          }
          .mail-row__subject {
              font-size: 0.92rem;
          }
          .mail-row__snippet {
              font-size: 0.84rem;
          }
          .mail-row__tag {
              font-size: 0.7rem;
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
                    <div class="mail-row__header">
                        <span class="mail-row__status" aria-hidden="true"></span>
                        <div class="mail-row__details">
                            <div class="mail-row__subject-line">
                                <p class="mail-row__subject">{subject}</p>
                                <span class="mail-row__tag">Queued</span>
                            </div>
                            <p class="mail-row__snippet">{snippet}</p>
                        </div>
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
