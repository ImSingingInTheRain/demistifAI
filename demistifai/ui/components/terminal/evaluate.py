"""Minimal terminal animation: per-char typing, colored tokens, progressive post-install."""

from __future__ import annotations

from typing import List

from .shared_renderer import make_terminal_renderer

_SUFFIX = "ai_evaluate"

# Preserve the historical constant name referenced throughout the module. A
# previous refactor renamed the suffix constant but missed the call sites,
# leaving `_TERMINAL_SUFFIX` undefined at import time. Keep both names in sync
# so downstream imports remain stable.
_TERMINAL_SUFFIX = _SUFFIX

_DEFAULT_DEMAI_LINES: List[str] = [
    "> Evaluate: how well does your spam detector perform?\n",
    "$ quote EU_AI_ACT.AI_system_definition --fragment 'infers how to generate outputs'\n",
    "‘An AI system infers how to generate output […]’\n",
    "$ evaluate start --on test_set\n",
    "Now that your model has learned from examples, it’s time to test how well it works.\n",
    "During training, we kept some emails aside — the test set. The model hasn’t seen these before.\n",
    "> Method\n",
    "Compare the model’s predictions with the true labels to get a fair measure of performance.\n",
    "$ evaluate metrics --set test --threshold 0.5\n",
    "Accuracy: 0.86 • Precision: 0.88 • Recall: 0.84 • F1: 0.86 ✓\n",
    "$ evaluate confusion_matrix\n",
    "TP=430  FP=60   FN=82   TN=428 ✓\n",
    "$ evaluate curves --roc --pr\n",
    "ROC AUC: 0.93 • PR AUC: 0.92 ✓\n",
    "$ evaluate threshold --sweep 0.20..0.90 step 0.10\n",
    "Best F1 at 0.55 • Trade-off: fewer false positives at 0.60 ✓\n",
    "Progress 0%   [████████████████████] 100%\n",
    "HINT: Toggle Nerd Mode to inspect misclassified emails, calibration, per-class metrics, and threshold effects.\n",
    "$ continue\n",
]

# Optional alias if you keep a naming convention for *_LINES variables
_EVALUATE_LINES = _DEFAULT_DEMAI_LINES

render_ai_act_terminal = make_terminal_renderer(
    suffix=_TERMINAL_SUFFIX,
    default_lines=_DEFAULT_DEMAI_LINES,
    default_key="ai_act_terminal",
)
