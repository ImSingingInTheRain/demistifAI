"""Welcome stage rendering utilities for the demistifAI app."""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Callable, ContextManager, Optional

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit.components.v1 import html as components_html

from demistifai.core.navigation import activate_stage
from demistifai.core.utils import streamlit_rerun
from demistifai.constants import STAGE_INDEX, STAGES, StageMeta
from demistifai.ui.components import render_stage_top_grid
from demistifai.ui.components.intro import (
    intro_lifecycle_ring_markup,
    render_intro_hero,
)
from demistifai.ui.theme.macos_window import (
    inject_macos_window_theme,
    macos_window_markup,
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
        hero_css, left_col_html, right_col_html = render_intro_hero()
        hero_title = "Start your demAI journey"
        hero_subtitle = "Explore the AI system lifecycle with EU AI Act guidance."
        hero_columns = (left_col_html, right_col_html)

        inject_macos_window_theme(st)
        window_markup = macos_window_markup(
            hero_title,
            subtitle=hero_subtitle,
            columns=len(hero_columns),
            ratios=(0.33, 0.67),
            id_suffix="intro-lifecycle",
            column_blocks=hero_columns,
            max_width=1200,
        )
        combined_markup = f"{hero_css}\n{window_markup}" if hero_css else window_markup
        st.markdown(combined_markup, unsafe_allow_html=True)

        lifecycle_markup = intro_lifecycle_ring_markup()
        components_html(
            dedent(
                f"""
                <script>
                    (function() {{
                        const doc = window.parent && window.parent.document ? window.parent.document : document;
                        const markup = {json.dumps(lifecycle_markup)};

                        function executeScripts(root) {{
                            root.querySelectorAll('script').forEach((script) => {{
                                const replacement = doc.createElement('script');
                                Array.from(script.attributes).forEach((attr) => {{
                                    replacement.setAttribute(attr.name, attr.value);
                                }});
                                replacement.textContent = script.textContent;
                                script.replaceWith(replacement);
                            }});
                        }}

                        function mountLifecycle() {{
                            const slot = doc.querySelector('[data-intro-lifecycle-slot]');
                            if (!slot) {{
                                return false;
                            }}

                            const template = doc.createElement('template');
                            template.innerHTML = markup;
                            const content = template.content.cloneNode(true);

                            slot.replaceChildren(content);
                            executeScripts(slot);
                            slot.setAttribute('data-intro-lifecycle-mounted', '1');
                            return true;
                        }}

                        if (!mountLifecycle()) {{
                            const maxWaitMs = 4000;
                            const start = Date.now();

                            const interval = setInterval(() => {{
                                const mounted = mountLifecycle();
                                const expired = Date.now() - start > maxWaitMs;
                                if (mounted || expired) {{
                                    clearInterval(interval);
                                }}
                            }}, 100);
                        }}
                    }})();
                </script>
                """
            ),
            height=0,
        )

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
                            function mountIntroStartButton() {{
                                const target = doc.querySelector('.intro-lifecycle-sidecar .intro-start-button-source');
                                const wrapper = doc.querySelector('.' + buttonClass);
                                if (!target || !wrapper) {{
                                    return false;
                                }}
                                if (target.contains(wrapper)) {{
                                    return true;
                                }}
                                const originBlock = wrapper.closest('[data-testid="stVerticalBlock"]');
                                target.innerHTML = '';
                                target.classList.add('intro-start-button-source--mounted');
                                target.appendChild(wrapper);
                                if (originBlock) {{
                                    originBlock.style.display = 'none';
                                }}
                                return true;
                            }}
                            if (!mountIntroStartButton()) {{
                                const retry = setInterval(function() {{
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

