"""Minimal terminal animation: per-char typing, colored tokens, progressive post-install."""

from __future__ import annotations

from typing import Iterable, List, Optional

from .shared_renderer import make_terminal_renderer

_SUFFIX = "ai_prepare"

# Preserve the historical constant name referenced throughout the module. A
# previous refactor renamed the suffix constant but missed the call sites,
# leaving `_TERMINAL_SUFFIX` undefined at import time. Keep both names in sync
# so downstream imports remain stable.
_TERMINAL_SUFFIX = _SUFFIX

_DEFAULT_DEMAI_LINES: List[str] = [
    "> Prepare data: teach the model what spam looks like\n",
    "$ quote EU_AI_ACT.AI_system_definition --fragment 'infers from the input it receives'\n",
    "‘An AI system infers from the input it receives […]’\n",
    "$ goal spam_filter --learn 'spam vs safe emails'\n",
    "Your AI system must learn to distinguish a safe email from spam.\n",
    "> Step 1 — Use the Dataset Builder to generate a synthetic dataset.\n",
    "$ dataset.builder --fields subject, body, sender, links --size 2000 --balanced\n",
    "Generating synthetic emails… ✓\n",
    "Progress 0%   [████████████████████] 100%\n",
    "$ dataset.health\n",
    "Health score: 82/100 • Coverage good • Leakage none • Duplicates low ✓\n",
    "Recommendations: add borderline safe examples; diversify link patterns; rebalance 'promotions' subtype.\n",
    "HINT: Toggle Nerd Mode for advanced configuration and diagnostic controls.\n",
    "$ continue\n",
]

# Backwards-compatibility alias for callers that might expect a *_LINES variable
_PREPARE_LINES = _DEFAULT_DEMAI_LINES

render_ai_act_terminal = make_terminal_renderer(
    suffix=_TERMINAL_SUFFIX,
    default_lines=_DEFAULT_DEMAI_LINES,
    default_key="ai_act_terminal",
)


def render_prepare_terminal(
    demai_lines: Optional[Iterable[str]] = None,
    speed_type_ms: int = 20,
    pause_between_ops_ms: int = 360,
    key: str = "ai_prepare_terminal",
    show_caret: bool = True,
) -> None:
    """Backward-compatible wrapper for the Prepare/Data stage terminal animation."""

    render_ai_act_terminal(
        demai_lines=demai_lines,
        speed_type_ms=speed_type_ms,
        pause_between_ops_ms=pause_between_ops_ms,
        key=key,
        show_caret=show_caret,
    )
