"""UI helpers for the Prepare stage."""

from __future__ import annotations

import html
from typing import Any, Dict, Optional


def cli_command_line(command: str, *, meta: Optional[str] = None, unsafe: bool = False) -> str:
    """Return HTML representing a faux CLI command line."""

    command_html = command if unsafe else html.escape(command)
    meta_html = (
        f"<span class='dataset-builder__meta'>{html.escape(meta)}</span>" if meta else ""
    )
    return (
        "<div class='dataset-builder__command-line'>"
        "<span class='dataset-builder__prompt'>$</span>"
        f"<span class='dataset-builder__command'>{command_html}</span>"
        f"{meta_html}"
        "</div>"
    )


def cli_comment(text: str) -> str:
    """Return HTML representing a CLI-style comment."""

    return f"<div class='dataset-builder__comment'># {html.escape(text)}</div>"


def build_compare_panel_html(
    base_summary: Optional[Dict[str, Any]],
    target_summary: Optional[Dict[str, Any]],
    delta_summary: Optional[Dict[str, Any]],
    delta_story_text: str,
) -> str:
    """Render the HTML block that summarises dataset deltas."""

    panel_items: list[tuple[str, str, str, str]] = []
    spam_share_delta_pp: Optional[float] = None

    def _add_panel_item(
        label: str,
        value: Any,
        *,
        unit: str = "",
        decimals: Optional[int] = None,
    ) -> None:
        if value is None:
            return
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return
        if abs(numeric_value) < 1e-6:
            return
        arrow = "▲" if numeric_value > 0 else "▼"
        arrow_class = "delta-arrow--up" if numeric_value > 0 else "delta-arrow--down"
        abs_value = abs(numeric_value)
        if decimals is not None:
            value_str = f"{abs_value:.{decimals}f}".rstrip("0").rstrip(".")
        else:
            if abs(abs_value - round(abs_value)) < 1e-6:
                value_str = f"{int(round(abs_value))}"
            else:
                value_str = f"{abs_value:.2f}".rstrip("0").rstrip(".")
        if unit:
            value_str = f"{value_str}{unit}"
        panel_items.append((label, arrow, arrow_class, value_str))

    if base_summary and target_summary:
        try:
            base_ratio = float(base_summary.get("spam_ratio") or 0.0)
            target_ratio = float(target_summary.get("spam_ratio") or 0.0)
            spam_share_delta_pp = (target_ratio - base_ratio) * 100.0
        except (TypeError, ValueError):
            spam_share_delta_pp = None
        if spam_share_delta_pp is not None and abs(spam_share_delta_pp) >= 0.1:
            _add_panel_item("Spam share", spam_share_delta_pp, unit="pp", decimals=1)

    if delta_summary:
        _add_panel_item("Examples", delta_summary.get("total"))
        _add_panel_item(
            "Avg suspicious links",
            delta_summary.get("avg_susp_links"),
            decimals=2,
        )
        _add_panel_item("Suspicious TLD hits", delta_summary.get("suspicious_tlds"))
        _add_panel_item("Money cues", delta_summary.get("money_mentions"))
        _add_panel_item("Attachment lures", delta_summary.get("attachment_lures"))

    effect_hint = ""
    if spam_share_delta_pp is not None and abs(spam_share_delta_pp) >= 0.1:
        if spam_share_delta_pp > 0:
            effect_hint = (
                "Higher spam share → recall ↑, precision may ↓; adjust threshold later in Evaluate."
            )
        else:
            effect_hint = (
                "Lower spam share → precision ↑, recall may ↓; consider rebalancing spam examples before training."
            )
    elif delta_summary:
        link_delta = float(delta_summary.get("avg_susp_links") or 0.0)
        tld_delta = float(delta_summary.get("suspicious_tlds") or 0.0)
        money_delta = float(delta_summary.get("money_mentions") or 0.0)
        attachment_delta = float(delta_summary.get("attachment_lures") or 0.0)
        if link_delta > 0:
            effect_hint = "More suspicious links → phishing recall should improve via URL cues."
        elif link_delta < 0:
            effect_hint = "Fewer suspicious links → URL-heavy spam might slip by; monitor precision/recall."
        elif tld_delta > 0:
            effect_hint = "Suspicious TLD hits increased — domain heuristics strengthen spam recall."
        elif tld_delta < 0:
            effect_hint = "Suspicious TLD hits dropped — rely more on text patterns and validate in Evaluate."
        elif money_delta > 0:
            effect_hint = "Money cues rose — expect better coverage on payment scams."
        elif money_delta < 0:
            effect_hint = "Money cues fell — finance-themed recall could dip."
        elif attachment_delta > 0:
            effect_hint = "Attachment lures increased — the model leans on risky file signals."
        elif attachment_delta < 0:
            effect_hint = "Attachment lures decreased — detection may hinge on text clues."

    if not effect_hint:
        if panel_items:
            effect_hint = "Changes logged — move to Evaluate to measure the impact."
        else:
            effect_hint = "No changes yet—adjust and preview."

    panel_html = ["<div class='dataset-delta-panel'>", "<h5>Compare datasets</h5>"]
    if panel_items:
        panel_html.append("<div class='dataset-delta-panel__items'>")
        for label, arrow, arrow_class, value_str in panel_items:
            panel_html.append(
                "<div class='dataset-delta-panel__item'><span>{label}</span>"
                "<span class='delta-arrow {cls}'>{arrow}{value}</span></div>".format(
                    label=html.escape(label),
                    cls=arrow_class,
                    arrow=html.escape(arrow),
                    value=html.escape(value_str),
                )
            )
        panel_html.append("</div>")
    else:
        panel_html.append(
            "<p class='dataset-delta-panel__story'>After you generate a dataset, you can tweak the configuration and preview here how these impact your data.</p>"
        )
        effect_hint = ""

    if effect_hint:
        panel_html.append(
            "<div class='dataset-delta-panel__hint'>{}</div>".format(
                html.escape(effect_hint)
            )
        )
    if delta_story_text:
        panel_html.append(
            "<div class='dataset-delta-panel__story'>{}</div>".format(
                html.escape(delta_story_text)
            )
        )
    panel_html.append("</div>")
    return "".join(panel_html)


__all__ = [
    "build_compare_panel_html",
    "cli_command_line",
    "cli_comment",
]

