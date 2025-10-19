"""Primitive UI widgets for demistifAI."""

from .mailbox import render_email_inbox_table, render_mailbox_panel
from .popovers import guidance_popover
from .quotes import eu_ai_quote_box, render_eu_ai_quote
from .sections import render_nerd_mode_toggle, section_surface
from .text import shorten_text

__all__ = [
    "eu_ai_quote_box",
    "guidance_popover",
    "render_email_inbox_table",
    "render_eu_ai_quote",
    "render_mailbox_panel",
    "render_nerd_mode_toggle",
    "section_surface",
    "shorten_text",
]
