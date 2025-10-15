from __future__ import annotations
from typing import Any, Dict, Optional

def _evaluate_dataset_health(summary: Dict[str, Any] | None, lint_counts: Optional[Dict[str, int]]) -> Dict[str, Any]:
    summary = summary or {}
    if lint_counts is None:
        lint_counts = {}
        lint_label = "Unknown"
    else:
        lint_flags = sum(int(v or 0) for v in lint_counts.values())
        lint_label = "Clean" if lint_flags == 0 else f"{lint_flags} flag{'s' if lint_flags != 1 else ''}"
    lint_flags = sum(int(v or 0) for v in lint_counts.values())
    spam_ratio = summary.get("spam_ratio")
    total_rows = summary.get("total")

    badge_text = None
    health_emoji = None
    spam_pct = None
    failures = 0
    missing_required = spam_ratio is None or total_rows is None

    if not missing_required:
        try:
            spam_pct = max(0.0, min(100.0, float(spam_ratio) * 100.0))
        except (TypeError, ValueError):
            missing_required = True
        if not isinstance(total_rows, (int, float)):
            missing_required = True
        else:
            if total_rows < 300:
                failures += 1
        if spam_pct is not None and not (40.0 <= spam_pct <= 60.0):
            failures += 1
        if lint_counts and lint_flags > 0:
            failures += 1

    if not missing_required:
        if failures == 0:
            badge_text = "🟢 Good"; health_emoji = "🟢"
        elif failures <= 2:
            badge_text = "🟡 Needs work"; health_emoji = "🟡"
        else:
            badge_text = "🔴 Risky"; health_emoji = "🔴"

    return {
        "badge_text": badge_text,
        "health_emoji": health_emoji,
        "spam_pct": spam_pct,
        "total_rows": total_rows,
        "lint_label": lint_label,
        "lint_flags": lint_flags,
    }
