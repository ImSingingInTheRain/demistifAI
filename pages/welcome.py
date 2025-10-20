"""Welcome stage rendering utilities for the demistifAI app."""

from __future__ import annotations

from textwrap import dedent
from typing import Callable, ContextManager, Optional

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from demistifai.core.navigation import activate_stage
from demistifai.core.utils import streamlit_rerun
from demistifai.constants import STAGE_INDEX, STAGES, StageMeta
from demistifai.ui.components import render_stage_top_grid
from demistifai.ui.components.intro import (
    INTRO_HERO_MAP_PANE_ID,
    INTRO_HERO_SIDECAR_PANE_ID,
    intro_hero_panes,
    render_lifecycle_ring_component,
)
from demistifai.ui.components.shared.macos_iframe_window import (
    MacWindowConfig,
    render_macos_iframe_window,
)
from demistifai.ui.components.terminal.article3 import (
    _WELCOME_LINES,
    render_ai_act_terminal as render_welcome_ai_act_terminal,
)
from demistifai.ui.primitives import render_eu_ai_quote

EU_AI_ACT_DEF = (
    "An AI system infers how to generate outputs that can influence physical or virtual environments."
)

SectionSurface = Callable[[Optional[str]], ContextManager[None]]


def render_intro_stage(*, section_surface: SectionSurface) -> None:
    """Render the welcome/intro stage surface."""

    next_stage_key: Optional[str] = None
    next_stage_meta: Optional[StageMeta] = None
    intro_index = STAGE_INDEX.get("intro")
    if intro_index is not None and intro_index < len(STAGES) - 1:
        next_stage_meta = STAGES[intro_index + 1]
        next_stage_key = next_stage_meta.key

    def _render_intro_terminal(slot: DeltaGenerator) -> None:
        with slot:
            render_welcome_ai_act_terminal(
                demai_lines=_WELCOME_LINES,
                speed_type_ms=20,
                pause_between_ops_ms=360,
            )

    render_stage_top_grid("intro", left_renderer=_render_intro_terminal)

    with section_surface("section-surface--hero"):
        hero_panes = intro_hero_panes()
        render_macos_iframe_window(
            st,
            MacWindowConfig(
                panes=hero_panes,
                rows=1,
                columns=2,
                column_ratios=(0.4, 0.6),
            ),
        )

        render_lifecycle_ring_component(height=0)

        if next_stage_meta is not None and next_stage_key is not None:
            button_key = f"intro_stage_start_{next_stage_key}"
            with st.container():
                cta_clicked = st.button(
                    f"{next_stage_meta.icon} {next_stage_meta.title} ➡️",
                    key=button_key,
                    type="primary",
                    use_container_width=True,
                    help="Jump to the next stage",
                )
                if cta_clicked and activate_stage(next_stage_key):
                    streamlit_rerun()

            st.markdown(
                dedent(
                    f"""
                    <script>
                        (function() {{
                            const doc = window.parent && window.parent.document ? window.parent.document : document;
                            const buttonClass = 'st-key-{button_key}';
                            const sidecarPaneId = '{INTRO_HERO_SIDECAR_PANE_ID}';

                            function getPaneDocument(paneId) {{
                                const heroSurface = doc.querySelector('.section-surface--hero');
                                if (!heroSurface) {{
                                    return null;
                                }}

                                const heroIframe = heroSurface.querySelector('[data-testid="stHtml"] iframe');
                                if (!heroIframe) {{
                                    return null;
                                }}

                                let heroDoc;
                                try {{
                                    heroDoc = heroIframe.contentDocument || (heroIframe.contentWindow && heroIframe.contentWindow.document) || null;
                                }} catch (error) {{
                                    heroDoc = null;
                                }}

                                if (!heroDoc) {{
                                    return null;
                                }}

                                const selector = `iframe[data-pane-id="${{paneId}}"]`;
                                const paneIframe = heroDoc.querySelector(selector);
                                if (!paneIframe) {{
                                    return null;
                                }}

                                try {{
                                    return paneIframe.contentDocument || (paneIframe.contentWindow && paneIframe.contentWindow.document) || null;
                                }} catch (error) {{
                                    return null;
                                }}
                            }}

                            function mountIntroStartButton() {{
                                const paneDoc = getPaneDocument(sidecarPaneId);
                                const wrapper = doc.querySelector('.' + buttonClass);
                                if (!paneDoc || !wrapper) {{
                                    return false;
                                }}

                                const target = paneDoc.querySelector('.intro-start-button-source');
                                if (!target) {{
                                    return false;
                                }}
                                if (target.contains(wrapper)) {{
                                    return true;
                                }}

                                const originBlock = wrapper.closest('[data-testid="stVerticalBlock"]');
                                target.innerHTML = '';
                                target.classList.add('intro-start-button-source--mounted');
                                try {{
                                    const adopted = paneDoc.adoptNode(wrapper);
                                    target.appendChild(adopted);
                                }} catch (error) {{
                                    const clone = paneDoc.importNode(wrapper, true);
                                    target.appendChild(clone);
                                    wrapper.remove();
                                }}
                                if (originBlock) {{
                                    originBlock.style.display = 'none';
                                }}
                                return true;
                            }}

                            if (!mountIntroStartButton()) {{
                                const retry = setInterval(() => {{
                                    if (mountIntroStartButton()) {{
                                        clearInterval(retry);
                                    }}
                                }}, 200);
                            }}
                        }})();
                    </script>
                    """
                ),
                unsafe_allow_html=True,
            )

