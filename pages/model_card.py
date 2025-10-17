"""Model Card stage page wrapper."""

from demistifai import app_core


def model_card() -> None:
    """Render the model card stage."""

    app_core.initialize_app()
    app_core.set_active_stage("model_card")
    app_core.render_stage("model_card")
