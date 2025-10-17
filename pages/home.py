"""Home page for the demistifAI Streamlit app."""

from demistifai import app_core


def home() -> None:
    """Render the welcome stage via the shared app core."""

    app_core.initialize_app()
    app_core.set_active_stage("intro")
    app_core.render_stage("intro")
