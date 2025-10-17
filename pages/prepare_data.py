"""Prepare Data page wrapper."""

from demistifai import app_core


def prepare_data() -> None:
    """Render the prepare data stage."""

    app_core.initialize_app()
    app_core.set_active_stage("data")
    app_core.render_stage("data")
