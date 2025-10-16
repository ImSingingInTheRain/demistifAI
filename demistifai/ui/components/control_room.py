"""Reusable Stage Control Room surface for Streamlit stages."""

from __future__ import annotations

from collections.abc import Callable
import html
import re
from textwrap import dedent

import streamlit as st
from streamlit.delta_generator import DeltaGenerator


StageRowRenderer = Callable[[DeltaGenerator], None]


def _normalise_theme(theme: str | None) -> str:
    if not theme:
        return "glass"
    theme = theme.lower().strip()
    return theme if theme in {"glass", "terminal", "mac"} else "glass"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "-", value).strip("-")
    return slug.lower() or "stage"


def stage_control_room(
    *,
    title: str,
    subtitle: str = "",
    nerd_state_key: str = "nerd_mode",
    theme: str = "glass",
    rows: list[StageRowRenderer] | None = None,
    prev_label: str = "\u2190 Previous",
    next_label: str = "Next \u2192",
    show_nerd_toggle: bool = True,
) -> tuple[bool, bool, bool]:
    """Render a stacked Stage Control Room surface.

    Parameters mirror the UX contract described in the design brief. Each
    callable in ``rows`` receives a fresh :class:`~streamlit.delta_generator.DeltaGenerator`
    container so that callers can compose arbitrary Streamlit widgets without
    leaking layout concerns back into this helper. Set ``show_nerd_toggle`` to
    ``False`` when the Nerd Mode control is rendered elsewhere but its state
    should still be read and returned.

    Returns
    -------
    tuple[bool, bool, bool]
        ``(clicked_prev, clicked_next, nerd_on)`` representing the CTA states
        and the persisted Nerd Mode toggle.
    """

    if rows is None:
        rows = []

    theme_name = _normalise_theme(theme)

    css = dedent(
        """
        <style>
          .hub {
            --ink:#0f172a; --muted:rgba(15,23,42,.78);
            --ring1: rgba(99,102,241,.08); --ring2: rgba(14,165,233,.06);
            --card:#fff; --stroke:rgba(15,23,42,.10);
            border-radius:16px; padding: clamp(12px, 2.2vw, 20px);
            margin: .25rem 0 1rem 0;
          }
          .hub--glass {
            background: radial-gradient(130% 120% at 50% 0%, var(--ring1), var(--ring2));
            box-shadow: inset 0 0 0 1px rgba(15,23,42,.06);
          }
          .hub--terminal {
            background:#0d1117; color:#e5e7eb;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.06);
          }
          .hub--terminal .hub__subtitle { color: rgba(226,232,240,.85); }
          .hub--terminal .hub__frame     { background: rgba(255,255,255,.03); }
          .hub--terminal .hub__btn       { background:#111827; color:#e5e7eb; }

          .hub--mac .hub__chrome {
            display:flex; gap:8px; align-items:center; margin-bottom:.35rem;
          }
          .hub--mac .hub__dot { width:12px; height:12px; border-radius:999px; display:inline-block; }
          .hub--mac .hub__dot--r{ background:#ff5f56; }
          .hub--mac .hub__dot--y{ background:#ffbd2e; }
          .hub--mac .hub__dot--g{ background:#27c93f; }

          .hub__frame {
            border-radius:14px;
            background: linear-gradient(180deg, rgba(255,255,255,.92), rgba(248,250,252,.82));
            box-shadow: inset 0 0 0 1px var(--stroke);
            padding: clamp(12px, 1.6vw, 16px);
          }

          .hub__header {
            display:grid; gap:.3rem; margin-bottom:.6rem;
          }
          .hub__title {
            font-weight: 800; color: var(--ink);
            font-size: clamp(1.15rem, 1.1vw + 1rem, 1.6rem);
          }
          .hub__subtitle {
            color: var(--muted);
            font-size: clamp(.95rem, .25vw + .9rem, 1rem);
          }

          .hub__toolbar {
            display:flex; gap:.6rem; align-items:center; justify-content: space-between;
            margin: .2rem 0 .6rem;
          }
          .hub__right { display:flex; gap:.6rem; align-items:center; }

          .hub__nerd {
            display:inline-flex; align-items:center; gap:.5rem;
            padding:.35rem .6rem; border-radius:10px;
            background: rgba(226,232,240,.65); color:#0f172a; font-weight:600;
            font-size:.9rem;
          }
          .hub--terminal .hub__nerd { background:#111827; color:#e5e7eb; }

          .hub__rows { display:grid; gap: clamp(10px, 1.2vw, 14px); }
          .hub__row {
            border-radius:12px; padding: clamp(10px, 1.4vw, 14px);
            background:#fff; box-shadow: 0 10px 22px rgba(15,23,42,.08), inset 0 0 0 1px rgba(15,23,42,.06);
          }
          .hub--terminal .hub__row { background:#0f172a; box-shadow: inset 0 0 0 1px rgba(255,255,255,.06); }

          .hub__footer {
            display:flex; gap:.6rem; justify-content: space-between; margin-top:.8rem;
          }
          .hub__btn {
            border: none; border-radius:12px; padding:.55rem .9rem;
            font-weight:700; font-size:.95rem; cursor:pointer;
            background:#0f172a; color:#fff;
            box-shadow: 0 10px 22px rgba(15,23,42,.12);
          }
          .hub__btn--ghost {
            background: #fff; color:#0f172a;
            box-shadow: inset 0 0 0 1px rgba(15,23,42,.12);
          }
          .hub--terminal .hub__btn--ghost { background:#111827; color:#e5e7eb; }
        </style>
        """
    )

    skin_class = {
        "glass": "hub hub--glass",
        "terminal": "hub hub--terminal",
        "mac": "hub hub--glass hub--mac",
    }[theme_name]

    st.markdown(css, unsafe_allow_html=True)

    title_slug = _slugify(title)

    with st.container():
        st.markdown(f'<div class="{skin_class}">', unsafe_allow_html=True)

        if theme_name == "mac":
            st.markdown(
                """
                <div class="hub__chrome">
                  <span class="hub__dot hub__dot--r"></span>
                  <span class="hub__dot hub__dot--y"></span>
                  <span class="hub__dot hub__dot--g"></span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="hub__frame">', unsafe_allow_html=True)

        escaped_title = html.escape(title)
        subtitle_html = (
            f'<div class="hub__subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
        )
        st.markdown(
            f'<div class="hub__header"><div class="hub__title">{escaped_title}</div>{subtitle_html}</div>',
            unsafe_allow_html=True,
        )

        nerd_on = bool(st.session_state.get(nerd_state_key, False))
        if show_nerd_toggle:
            st.markdown('<div class="hub__toolbar">', unsafe_allow_html=True)
            left_col, right_col = st.columns([1, 1], gap="small")
            with left_col:
                nerd_on = st.toggle(
                    "Nerd Mode",
                    key=nerd_state_key,
                    value=nerd_on,
                )
            with right_col:
                right_col.empty()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="hub__rows">', unsafe_allow_html=True)
        for render_row in rows:
            st.markdown('<div class="hub__row">', unsafe_allow_html=True)
            row_slot = st.container()
            try:
                render_row(row_slot)
            except Exception as exc:  # pragma: no cover - render safeguard
                row_slot.warning(f"Row failed to render: {exc}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="hub__footer">', unsafe_allow_html=True)
        cta_left, cta_right = st.columns(2, gap="small")
        with cta_left:
            clicked_prev = st.button(
                prev_label,
                key=f"{title_slug}__prev",
                use_container_width=True,
            )
        with cta_right:
            clicked_next = st.button(
                next_label,
                key=f"{title_slug}__next",
                type="primary",
                use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    return clicked_prev, clicked_next, st.session_state.get(nerd_state_key, False)


__all__ = ["stage_control_room", "StageRowRenderer"]

