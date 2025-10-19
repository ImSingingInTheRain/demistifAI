from __future__ import annotations

import html
import re
import uuid
from collections.abc import Iterable
from typing import Literal
from textwrap import dedent, indent

def _strip_style_wrappers(css_chunk: str) -> str:
    """Return CSS stripped of optional <style> wrappers."""

    cleaned = (css_chunk or "").strip()
    if not cleaned:
        return ""

    if "<style" in cleaned.lower():
        cleaned = re.sub(r"^<style[^>]*>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned = re.sub(r"</style>$", "", cleaned, flags=re.IGNORECASE).strip()

    return cleaned


def _normalise_scoped_css(scoped_css: str | Iterable[str] | None) -> str:
    """Normalise scoped CSS blocks into an indented string for the template."""

    if scoped_css is None:
        return ""

    if isinstance(scoped_css, str):
        css_chunks = [scoped_css]
    else:
        css_chunks = [chunk for chunk in scoped_css if chunk]

    cleaned_chunks = [_strip_style_wrappers(chunk) for chunk in css_chunks]
    cleaned_chunks = [chunk for chunk in cleaned_chunks if chunk]

    if not cleaned_chunks:
        return ""

    combined = "\n\n".join(cleaned_chunks)
    return "\n" + indent(combined, "          ")


ColumnVariant = Literal["standard", "flush"]


def mac_window_html(
    title: str = "demAI",
    subtitle: str | None = None,
    columns: int = 2,
    ratios: tuple[float, ...] | None = None,
    col_html: list[str] | None = None,
    dense: bool = False,
    theme: str = "light",
    id_suffix: str | None = None,
    scoped_css: str | Iterable[str] | None = None,
    max_width: float | int | str = 1020,
    column_variant: ColumnVariant = "standard",
) -> str:
    """Return a scoped HTML/CSS macOS-style window.

    Args:
        title: Window title shown in the chrome.
        subtitle: Optional subtitle rendered under the title.
        columns: Number of content columns (1–3).
        ratios: Relative column width ratios.
        col_html: Pre-rendered HTML for each column.
        dense: When ``True`` the body/columns use reduced padding.
        theme: ``"light"`` or ``"dark"`` chrome palette.
        id_suffix: Optional suffix used to scope the CSS classes.
        scoped_css: Extra CSS scoped to the rendered window.
        max_width: Maximum width constraint for the window. Numeric
            values are interpreted as pixel widths, preserving the
            previous 1020px default when omitted.
        column_variant: Surface treatment for column containers. Use
            ``"standard"`` for the padded gradient blocks (default) or
            ``"flush"`` to remove padding, background, and shadows so the
            injected HTML can control its own layout edge-to-edge.
    """
    if columns not in (1, 2, 3):
        raise ValueError("columns must be 1, 2 or 3")
    if ratios is None:
        ratios = tuple(1 for _ in range(columns))
    if len(ratios) != columns or any(r <= 0 for r in ratios):
        raise ValueError("ratios must match columns and be positive")
    if col_html is None:
        col_html = [None] * columns
    if len(col_html) != columns:
        raise ValueError("col_html length must equal columns")

    suf = id_suffix or uuid.uuid4().hex[:8]
    total = float(sum(ratios))
    cols_css = " ".join(f"{(r / total) * 100:.5f}%" for r in ratios)

    if isinstance(max_width, (int, float)):
        if max_width <= 0:
            raise ValueError("max_width must be positive")
        max_width_css = f"{format(max_width, 'g')}px"
    else:
        max_width_css = str(max_width).strip()
        if not max_width_css:
            raise ValueError("max_width must not be empty")

    chrome_bg = "#f5f7fb" if theme == "light" else "#111827"
    chrome_border = "rgba(15,23,42,.12)" if theme == "light" else "rgba(255,255,255,.12)"
    body_bg = "#ffffff" if theme == "light" else "#0b1220"
    title_color = "#0f172a" if theme == "light" else "#e5e7eb"
    sub_color = "rgba(15,23,42,.70)" if theme == "light" else "rgba(229,231,235,.70)"
    ph_text = "rgba(15,23,42,.55)" if theme == "light" else "rgba(229,231,235,.55)"

    body_padding = "clamp(1rem, 1.2vw + 0.95rem, 1.45rem)"
    col_padding = "clamp(0.95rem, 1.4vw + 0.7rem, 1.35rem)"
    col_mobile_padding = "clamp(.85rem, 5vw, 1.2rem)"
    col_background = "linear-gradient(160deg, rgba(248,250,252,.97), rgba(226,232,240,.7))"
    col_box_shadow = "inset 0 0 0 1px rgba(148,163,184,.25)"
    if dense:
        body_padding = ".45rem .6rem"
        col_padding = ".65rem .75rem"
        col_mobile_padding = "clamp(.75rem, 4.5vw, 1rem)"

    if column_variant not in ("standard", "flush"):
        raise ValueError("column_variant must be 'standard' or 'flush'")

    if column_variant == "flush":
        col_padding = "0"
        col_mobile_padding = "0"
        col_background = "none"
        col_box_shadow = "none"

    placeholders = [
        dedent(
            f"""
            <div class=\"mw-{suf}__placeholder\">
              <div class=\"mw-{suf}__ph-title\">Placeholder</div>
              <div class=\"mw-{suf}__ph-lines\">
                <span></span><span></span><span class=\"short\"></span>
              </div>
            </div>
            """
        ).strip()
        for _ in range(columns)
    ]

    cols = []
    for i in range(columns):
        inner = col_html[i] if col_html[i] else placeholders[i]
        cols.append(f'<div class="mw-{suf}__col">{inner}</div>')

    subtitle_html = (
        f'<div class="mw-{suf}__subtitle">{html.escape(subtitle)}</div>'
        if subtitle
        else ""
    )

    extra_scoped_css = _normalise_scoped_css(scoped_css)

    return dedent(
        f"""
        <style>
          /* === mac window (scoped) ======================================== */
          .mw-{suf} {{
            --chrome-bg: {chrome_bg};
            --chrome-border: {chrome_border};
            --body-bg: {body_bg};
            --title: {title_color};
            --subtitle: {sub_color};
            --ph-text: {ph_text};
            --radius: 14px;
            --shadow: 0 16px 40px rgba(15,23,42,.12);
            border-radius: var(--radius);
            overflow: hidden;
            box-shadow: var(--shadow);
            background: var(--body-bg);
            width: min(100%, {max_width_css});
            margin: clamp(0.75rem, 3vw, 1.75rem) auto;
          }}
          .mw-{suf}__chrome {{
            display: grid;
            grid-template-columns: auto 1fr;
            align-items: center;
            gap: .75rem;
            padding: .65rem 1rem;
            background: var(--chrome-bg);
            border-bottom: 1px solid var(--chrome-border);
          }}
          .mw-{suf}__lights {{
            display: inline-flex; gap: .45rem; padding-left: .15rem;
          }}
          .mw-{suf}__light {{
            width: 12px; height: 12px; border-radius: 999px;
            box-shadow: inset 0 0 0 1px rgba(0,0,0,.08);
          }}
          .mw-{suf}__light--red    {{ background:#ff5f56; }}
          .mw-{suf}__light--yellow {{ background:#ffbd2e; }}
          .mw-{suf}__light--green  {{ background:#27c93f; }}
          .mw-{suf}__titles {{
            display: grid; gap: .15rem;
          }}
          .mw-{suf}__title {{
            font-weight: 800; color: var(--title); line-height: 1.2;
          }}
          .mw-{suf}__subtitle {{
            font-size: .92rem; color: var(--subtitle);
          }}

          .mw-{suf}__body {{
            padding: {body_padding};
          }}
          .mw-{suf}__grid {{
            display: grid;
            grid-template-columns: {cols_css};
            gap: 1.15rem;
            align-items: stretch;
          }}
          .mw-{suf}__col {{
            border-radius: 12px;
            background: {col_background};
            box-shadow: {col_box_shadow};
            padding: {col_padding};
            min-height: 180px;
            display: grid;
            align-content: start;
            gap: .65rem;
          }}

          /* Placeholder skeleton */
          .mw-{suf}__placeholder {{
            color: var(--ph-text);
          }}
          .mw-{suf}__ph-title {{
            font-weight: 700; margin-bottom: .15rem;
          }}
          .mw-{suf}__ph-lines {{
            display: grid; gap: .45rem;
          }}
          .mw-{suf}__ph-lines span {{
            display:block; height: 9px; border-radius: 6px;
            background: linear-gradient(90deg, rgba(148,163,184,.28), rgba(148,163,184,.16), rgba(148,163,184,.28));
            background-size: 180% 100%;
            animation: mw-{suf}-shimmer 1.6s infinite linear;
          }}
          .mw-{suf}__ph-lines span.short {{ width: 60%; }}
          @keyframes mw-{suf}-shimmer {{
            0% {{ background-position: 180% 0; }}
            100% {{ background-position: -20% 0; }}
          }}

          /* Responsive */
          @media (max-width: 920px){{
            .mw-{suf}__grid {{ grid-template-columns: 1fr; }}
          }}

          @media (max-width: 720px){{
            .mw-{suf} {{
              --radius: 12px;
              box-shadow: 0 14px 32px rgba(15,23,42,.12);
            }}
            .mw-{suf}__chrome {{
              padding: .55rem .85rem;
              grid-template-columns: 1fr;
              row-gap: .5rem;
            }}
            .mw-{suf}__lights {{
              order: 2;
              justify-self: start;
            }}
            .mw-{suf}__body {{
              padding: clamp(.85rem, 5vw, 1.15rem);
            }}
            .mw-{suf}__grid {{
              gap: clamp(.75rem, 4.5vw, 1rem);
            }}
            .mw-{suf}__col {{
              min-height: 0;
              padding: {col_mobile_padding};
            }}
          }}

          @media (max-width: 600px){{
            .mw-{suf} {{
              --radius: 0px;
              --shadow: none;
              width: 100%;
              margin: 0;
              border-radius: 0;
              box-shadow: none;
              border: none;
            }}
            .mw-{suf}__chrome {{
              display: none;
            }}
            .mw-{suf}__body {{
              padding: 0 1rem 1.25rem;
            }}
            .mw-{suf}__grid {{
              grid-template-columns: 1fr;
              gap: clamp(.75rem, 4.5vw, 1rem);
            }}
            .mw-{suf}__col {{
              padding: 0;
              border-radius: 0;
              background: none;
              box-shadow: none;
              border: 0;
              min-height: 0;
            }}
          }}{extra_scoped_css}
        </style>

        <section class="mw-{suf}" role="group" aria-label="{html.escape(title)} window">
          <header class="mw-{suf}__chrome" aria-hidden="false">
            <div class="mw-{suf}__lights" aria-hidden="true">
              <span class="mw-{suf}__light mw-{suf}__light--red"></span>
              <span class="mw-{suf}__light mw-{suf}__light--yellow"></span>
              <span class="mw-{suf}__light mw-{suf}__light--green"></span>
            </div>
            <div class="mw-{suf}__titles">
              <div class="mw-{suf}__title">{html.escape(title)}</div>
              {subtitle_html}
            </div>
          </header>

          <div class="mw-{suf}__body">
            <div class="mw-{suf}__grid">
              {''.join(cols)}
            </div>
          </div>
        </section>
        <script>
          (function() {{
            const root = document.querySelector('.mw-{suf}');
            if (!root) return;

            const postSize = () => {{
              const height = Math.ceil(root.getBoundingClientRect().height + 24);
              if (window.parent && window.parent !== window) {{
                window.parent.postMessage({{"type": "streamlit:resize", "height": height}}, "*");
              }}
            }};

            postSize();

            if (typeof ResizeObserver === 'function') {{
              const resizeObserver = new ResizeObserver(() => {{
                window.requestAnimationFrame(postSize);
              }});
              resizeObserver.observe(root);
            }} else {{
              window.addEventListener('transitionend', postSize);
              window.addEventListener('animationend', postSize);
            }}

            if (document.readyState === 'complete') {{
              postSize();
            }} else {{
              window.addEventListener('load', postSize, {{ once: true }});
              window.addEventListener('DOMContentLoaded', postSize, {{ once: true }});
            }}

            window.addEventListener('resize', postSize);
          }})();
        </script>
        """
    )



def render_mac_window(
    st,
    *,
    fallback_height: int | None = None,
    scoped_css: str | Iterable[str] | None = None,
    max_width: float | int | str = 1020,
    **kwargs,
):
    """Render the macOS-style window in Streamlit.

    Args:
        st: Streamlit module proxy used for rendering.
        fallback_height: Optional iframe height when Streamlit cannot
            auto-size the HTML container.
        scoped_css: Additional CSS scoped to the rendered window.
        max_width: Maximum width passed through to :func:`mac_window_html`.
        **kwargs: Remaining keyword arguments forwarded to
            :func:`mac_window_html`.
    """

    html_str = mac_window_html(scoped_css=scoped_css, max_width=max_width, **kwargs)

    # ``components.html`` ensures rich HTML (including scripts like the training
    # animation) render without sanitisation. Always prefer it so interactive
    # payloads behave consistently across Streamlit versions.
    if fallback_height is None:
        # Keep a minimal height for scenarios where client-side resizing is
        # unavailable (e.g., JS disabled). The resize script injected by
        # :func:`mac_window_html` will immediately grow the iframe when enabled.
        fallback_height = 360

    st.components.v1.html(
        html_str,
        height=fallback_height,
        scrolling=False,
    )
