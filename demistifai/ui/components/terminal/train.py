"""Minimal terminal animation: per-char typing, colored tokens, progressive post-install."""

from __future__ import annotations

from typing import List

from .shared_renderer import make_terminal_renderer

_SUFFIX = "ai_train"

# Preserve the historical constant name referenced throughout the module. A
# previous refactor renamed the suffix constant but missed the call sites,
# leaving `_TERMINAL_SUFFIX` undefined at import time. Keep both names in sync
# so downstream imports remain stable.
_TERMINAL_SUFFIX = _SUFFIX

_DEFAULT_DEMAI_LINES: List[str] = [
    "> Train: teach the model to achieve an objective\n",
    "$ quote EU_AI_ACT.AI_system_definition --fragment 'infers for explicit or implicit objectives'\n",
    "‘An AI system infers for explicit or implicit objectives […]’\n",
    "$ objective set 'classify spam vs safe emails'\n",
    "Now your AI system will learn how to achieve an objective.\n",
    "> Your part\n",
    "Provide examples with clear labels — “spam” and “safe”.\n",
    "$ dataset.preview --show 3 spam, 3 safe\n",
    "OK ✓\n",
    "> The system’s part\n",
    "Spot patterns that generalize to emails it hasn’t seen yet.\n",
    "$ train start --epochs 3 --batch 64 --features subject,body,sender,links\n",
    "Epoch 1/3  loss=0.68  accuracy=0.74 ✓\n",
    "Epoch 2/3  loss=0.52  accuracy=0.83 ✓\n",
    "Epoch 3/3  loss=0.41  accuracy=0.88 ✓\n",
    "Progress 0%   [████████████████████] 100%\n",
    "$ validate --holdout 20%\n",
    "Holdout accuracy: 0.86 • Precision: 0.88 • Recall: 0.84 ✓\n",
    "HINT: Toggle Nerd Mode to tune hyperparameters, inspect features, and view the confusion matrix.\n",
    "$ continue\n",
]

# Optional alias if you keep a naming convention (not required by template)
_TRAIN_LINES = _DEFAULT_DEMAI_LINES

render_ai_act_terminal = make_terminal_renderer(
    suffix=_TERMINAL_SUFFIX,
    default_lines=_DEFAULT_DEMAI_LINES,
    default_key="ai_act_terminal",
)
