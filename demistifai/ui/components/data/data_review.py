"""HTML helpers for the data review surface."""

from __future__ import annotations

import html
from typing import Iterable, Mapping, Sequence, Tuple
from textwrap import dedent


def data_review_styles() -> str:
    """Return the scoped CSS for the data review components."""

    share_classes = []
    for pct in range(0, 101):
        share_classes.append(
            f".dataset-balance-bar__segment--spam-share-{pct} {{ flex: {pct}; }}"
        )
        share_classes.append(
            f".dataset-balance-bar__segment--safe-share-{pct} {{ flex: {pct}; }}"
        )

    base_css = dedent(
        """
        <style>
            .dataset-balance-bar {
                border-radius: 6px;
                overflow: hidden;
                border: 1px solid #DDD;
                font-size: 0.75rem;
                margin: 0.5rem 0 0.75rem;
            }

            .dataset-balance-bar__segments {
                display: flex;
                height: 28px;
            }

            .dataset-balance-bar__segment {
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                padding: 4px 8px;
                font-weight: 600;
                min-width: 0;
            }

            .dataset-balance-bar__segment--spam {
                background-color: #ff4b4b;
            }

            .dataset-balance-bar__segment--safe {
                background-color: #1c83e1;
            }

            .sample-card {
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                padding: 0.75rem;
                margin-bottom: 0.5rem;
            }

            .sample-card__label {
                display: flex;
                align-items: center;
                gap: 0.35rem;
                font-weight: 600;
            }

            .sample-card__label-icon {
                font-size: 0.95rem;
            }

            .sample-card__title {
                font-weight: 600;
                margin: 0.25rem 0 0.35rem;
            }

            .sample-card__excerpt {
                font-size: 0.85rem;
                color: #374151;
                line-height: 1.35;
            }

            .edge-case-card {
                border: 1px solid #E5E7EB;
                border-radius: 10px;
                padding: 0.75rem;
                margin-bottom: 0.5rem;
            }

            .edge-case-card__title {
                font-weight: 600;
                margin-bottom: 0.4rem;
            }

            .edge-case-card__body {
                display: flex;
                gap: 0.5rem;
            }

            .edge-case-card__panel {
                flex: 1;
                border-radius: 8px;
                padding: 0.5rem;
                font-size: 0.8rem;
            }

            .edge-case-card__panel--spam {
                background: #FEE2E2;
            }

            .edge-case-card__panel--safe {
                background: #DBEAFE;
            }

            .edge-case-card__panel--spam .edge-case-card__label {
                color: #B91C1C;
            }

            .edge-case-card__panel--spam .edge-case-card__excerpt {
                color: #7F1D1D;
            }

            .edge-case-card__panel--safe .edge-case-card__label {
                color: #1D4ED8;
            }

            .edge-case-card__panel--safe .edge-case-card__excerpt {
                color: #1E3A8A;
            }

            .edge-case-card__label {
                display: flex;
                align-items: center;
                gap: 0.35rem;
                font-weight: 600;
                margin-bottom: 0.25rem;
            }

            .edge-case-card__excerpt {
                line-height: 1.35;
            }

            @media (max-width: 640px) {
                .edge-case-card__body {
                    flex-direction: column;
                }
            }

        """
    )

    share_css = "\n".join(share_classes)
    return f"{base_css}\n{share_css}\n</style>"


def dataset_snapshot_styles() -> str:
    """Return the scoped CSS for dataset snapshot surfaces."""

    return dedent(
        """
        <style>
            .dataset-snapshot-card {
                border: 1px solid #E5E7EB;
                border-radius: 12px;
                padding: 1rem 1.25rem;
                margin-bottom: 0.75rem;
            }

            .dataset-snapshot-card__sections {
                display: flex;
                gap: 1.5rem;
                flex-wrap: wrap;
            }

            .dataset-snapshot-card__section {
                flex: 1 1 240px;
                min-width: 220px;
            }

            .dataset-snapshot-card__section-title {
                font-weight: 600;
                font-size: 0.95rem;
                margin-bottom: 0.6rem;
            }

            .dataset-snapshot-card__rows {
                display: flex;
                flex-direction: column;
                gap: 0.4rem;
                font-size: 0.9rem;
            }

            .dataset-snapshot-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .dataset-snapshot-row__label {
                color: #111827;
            }

            .dataset-snapshot-row__value {
                font-weight: 600;
                color: #111827;
            }

            .dataset-snapshot-name {
                display: flex;
                align-items: center;
                gap: 0.35rem;
            }

            .dataset-snapshot-name__text {
                font-weight: 600;
            }

            .dataset-snapshot-badge {
                display: inline-flex;
                align-items: center;
                border-radius: 999px;
                padding: 2px 8px;
                font-size: 0.7rem;
                font-weight: 600;
                line-height: 1;
            }

            .dataset-snapshot-badge--active {
                background: #DCFCE7;
                color: #166534;
            }
        </style>
        """
    )


def _dataset_snapshot_rows_html(rows: Sequence[Tuple[object, object]]) -> str:
    """Return HTML for snapshot rows."""

    fragments = []
    for label, value in rows:
        label_text = html.escape(str(label) if label is not None else "")
        value_text = html.escape(str(value) if value is not None else "")
        fragments.append(
            dedent(
                f"""
                <div class="dataset-snapshot-row">
                    <span class="dataset-snapshot-row__label">{label_text}</span>
                    <span class="dataset-snapshot-row__value">{value_text}</span>
                </div>
                """
            ).strip()
        )

    return "\n".join(fragments)


def dataset_snapshot_card_html(
    summary_rows: Sequence[Tuple[object, object]],
    fingerprint_rows: Sequence[Tuple[object, object]],
) -> str:
    """Build the dataset snapshot summary card markup."""

    summary_html = _dataset_snapshot_rows_html(summary_rows)
    fingerprint_html = _dataset_snapshot_rows_html(fingerprint_rows)

    return dedent(
        f"""
        <div class="dataset-snapshot-card">
            <div class="dataset-snapshot-card__sections">
                <div class="dataset-snapshot-card__section">
                    <div class="dataset-snapshot-card__section-title">Summary</div>
                    <div class="dataset-snapshot-card__rows">
                        {summary_html}
                    </div>
                </div>
                <div class="dataset-snapshot-card__section">
                    <div class="dataset-snapshot-card__section-title">Fingerprint</div>
                    <div class="dataset-snapshot-card__rows">
                        {fingerprint_html}
                    </div>
                </div>
            </div>
        </div>
        """
    ).strip()


def dataset_snapshot_active_badge(is_active: bool, label: str = "Active") -> str:
    """Return the active badge markup when the snapshot is active."""

    if not is_active:
        return ""

    return (
        f'<span class="dataset-snapshot-badge dataset-snapshot-badge--active">{html.escape(label)}</span>'
    )


def dataset_balance_bar_html(spam_ratio: float) -> str:
    """Return the HTML for the dataset balance bar."""

    if spam_ratio is None:
        spam_ratio = 0.0
    spam_ratio = max(0.0, min(1.0, float(spam_ratio)))
    spam_pct = int(round(spam_ratio * 100))
    safe_pct = max(0, 100 - spam_pct)

    return dedent(
        f"""
        <div class="dataset-balance-bar">
            <div class="dataset-balance-bar__segments">
                <div class="dataset-balance-bar__segment dataset-balance-bar__segment--spam dataset-balance-bar__segment--spam-share-{spam_pct}">
                    Spam {spam_pct}%
                </div>
                <div class="dataset-balance-bar__segment dataset-balance-bar__segment--safe dataset-balance-bar__segment--safe-share-{safe_pct}">
                    Safe {safe_pct}%
                </div>
            </div>
        </div>
        """
    )


def stratified_sample_cards_html(cards: Sequence[Mapping[str, object]]) -> str:
    """Return the HTML for the stratified sample cards."""

    card_fragments = []
    for card in cards:
        label_value = html.escape(str(card.get("label", "")).strip().lower())
        normalized_label = (card.get("label", "") or "").strip().lower()
        label_text = html.escape(normalized_label.title() or "Unlabeled")
        label_icon = {"spam": "🚩", "safe": "📥"}.get(normalized_label, "✉️")
        title = html.escape(str(card.get("title", "")))
        body = str(card.get("body", "") or "")
        excerpt = body[:160] + ("…" if len(body) > 160 else "")
        excerpt = html.escape(excerpt.replace("\n", " "))

        card_fragments.append(
            dedent(
                f"""
                <div class="sample-card" data-label="{label_value}">
                    <div class="sample-card__label"><span class="sample-card__label-icon">{label_icon}</span><span>{label_text}</span></div>
                    <div class="sample-card__title">{title}</div>
                    <div class="sample-card__excerpt">{excerpt}</div>
                </div>
                """
            ).strip()
        )

    return "\n".join(card_fragments)


def edge_case_pairs_html(
    pairs: Iterable[Tuple[Mapping[str, object], Mapping[str, object]]]
) -> str:
    """Return the HTML for edge-case comparison cards."""

    fragments = []
    for spam_row, safe_row in pairs:
        spam_title = html.escape(str(spam_row.get("title", "Untitled")))
        spam_body = str(spam_row.get("body", "") or "")
        safe_body = str(safe_row.get("body", "") or "")

        spam_excerpt = spam_body[:120] + ("…" if len(spam_body) > 120 else "")
        safe_excerpt = safe_body[:120] + ("…" if len(safe_body) > 120 else "")

        spam_excerpt = html.escape(spam_excerpt.replace("\n", " "))
        safe_excerpt = html.escape(safe_excerpt.replace("\n", " "))

        fragments.append(
            dedent(
                f"""
                <div class="edge-case-card">
                    <div class="edge-case-card__title">{spam_title}</div>
                    <div class="edge-case-card__body">
                        <div class="edge-case-card__panel edge-case-card__panel--spam">
                            <div class="edge-case-card__label"><span class="sample-card__label-icon">🚩</span><span>Spam</span></div>
                            <div class="edge-case-card__excerpt">{spam_excerpt}</div>
                        </div>
                        <div class="edge-case-card__panel edge-case-card__panel--safe">
                            <div class="edge-case-card__label"><span class="sample-card__label-icon">📥</span><span>Safe</span></div>
                            <div class="edge-case-card__excerpt">{safe_excerpt}</div>
                        </div>
                    </div>
                </div>
                """
            ).strip()
        )

    return "\n".join(fragments)

