"""Minimal terminal animation: per-char typing, colored tokens, progressive post-install."""

from __future__ import annotations

from typing import List

from .shared_renderer import make_terminal_renderer

_SUFFIX = "ai_use"

# Preserve the historical constant name referenced throughout the module. A
# previous refactor renamed the suffix constant but missed the call sites,
# leaving `_TERMINAL_SUFFIX` undefined at import time. Keep both names in sync
# so downstream imports remain stable.
_TERMINAL_SUFFIX = _SUFFIX

_DEFAULT_DEMAI_LINES: List[str] = [
    "> Use: run the spam detector on incoming emails\n",
    "$ quote EU_AI_ACT.AI_system_definition --fragment 'infers how to generate content, predictions, recommendations or decisions'\n",
    "‘An AI system infers how to generate content, predictions, recommendations or decisions’\n",
    "$ predict start --stream\n",
    "In this step, the system takes each email (title + body) as input and produces an output.\n",
    "> Output format\n",
    "Prediction: Spam | Safe  •  Confidence: 0.00–1.00  •  Recommendation: place in Spam | Inbox\n",
    "$ predict example --show\n",
    "Title: “Limited time offer!!!”\n",
    "Body:  “Click to claim your prize — ends today.”\n",
    "→ Prediction: Spam  •  Confidence: 0.97  •  Recommendation: Spam ✓\n",
    "$ predict example --show\n",
    "Title: “Meeting notes: Q4 planning”\n",
    "Body:  “Attached are action items from today’s meeting.”\n",
    "→ Prediction: Safe  •  Confidence: 0.91  •  Recommendation: Inbox ✓\n",
    "$ predict batch --size 100\n",
    "Progress 0%   [████████████████████] 100%\n",
    "$ threshold get\n",
    "Current decision threshold: 0.50 • Change affects Spam/Inbox recommendations ✓\n",
    "HINT: Toggle Nerd Mode to adjust thresholding, view per-email confidence, and export predictions.\n",
    "$ continue\n",
]

# Optional alias if you keep *_LINES naming parallelism
_USE_LINES = _DEFAULT_DEMAI_LINES

render_ai_act_terminal = make_terminal_renderer(
    suffix=_TERMINAL_SUFFIX,
    default_lines=_DEFAULT_DEMAI_LINES,
    default_key="ai_act_terminal",
)
