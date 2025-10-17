"""Use stage page wrapper."""

from demistifai import app_core


def use() -> None:
    """Render the use/classify stage."""

    app_core.initialize_app()
    app_core.set_active_stage("classify")
    app_core.render_stage("classify")
