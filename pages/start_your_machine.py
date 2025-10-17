"""Start Your Machine page wrapper."""

from demistifai import app_core


def start_your_machine() -> None:
    """Render the overview stage."""

    app_core.initialize_app()
    app_core.set_active_stage("overview")
    app_core.render_stage("overview")
