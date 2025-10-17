"""Legacy helpers retained for backward compatibility with stage modules."""

from demistifai import app_core


def init_app():
    """Initialize shared state via the central app core."""

    return app_core.initialize_app()


section_surface = app_core.section_surface
render_nerd_mode_toggle = app_core.render_nerd_mode_toggle
render_eu_ai_quote = app_core.render_eu_ai_quote
