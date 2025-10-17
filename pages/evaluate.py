"""Evaluate stage page wrapper."""

from demistifai import app_core


def evaluate() -> None:
    """Render the evaluate stage."""

    app_core.initialize_app()
    app_core.set_active_stage("evaluate")
    app_core.render_stage("evaluate")
