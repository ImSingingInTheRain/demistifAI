"""Train stage page wrapper."""

from demistifai import app_core


def train() -> None:
    """Render the train stage."""

    app_core.initialize_app()
    app_core.set_active_stage("train")
    app_core.render_stage("train")
