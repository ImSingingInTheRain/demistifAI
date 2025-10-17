"""Primitive UI widgets for demistifAI."""

from .mailbox import render_email_inbox_table, render_mailbox_panel
from .quotes import eu_ai_quote_box, render_eu_ai_quote
from .sections import render_nerd_mode_toggle, section_surface
from .text import shorten_text
from .typing_quote import (
    get_eu_ai_act_typing_inline_bootstrap,
    get_eu_ai_act_typing_inline_markup,
)

__all__ = [
    "eu_ai_quote_box",
    "get_eu_ai_act_typing_inline_bootstrap",
    "get_eu_ai_act_typing_inline_markup",
    "render_email_inbox_table",
    "render_eu_ai_quote",
    "render_mailbox_panel",
    "render_nerd_mode_toggle",
    "section_surface",
    "shorten_text",
]
