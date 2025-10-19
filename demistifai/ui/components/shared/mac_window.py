from __future__ import annotations

import html
import re
import uuid
from collections.abc import Iterable
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
) -> str:
    """Return a scoped HTML/CSS macOS-style window."""
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

    chrome_bg = "#f5f7fb" if theme == "light" else "#111827"
    chrome_border = "rgba(15,23,42,.12)" if theme == "light" else "rgba(255,255,255,.12)"
    body_bg = "#ffffff" if theme == "light" else "#0b1220"
    title_color = "#0f172a" if theme == "light" else "#e5e7eb"
    sub_color = "rgba(15,23,42,.70)" if theme == "light" else "rgba(229,231,235,.70)"
    ph_text = "rgba(15,23,42,.55)" if theme == "light" else "rgba(229,231,235,.55)"

    body_padding = "clamp(1rem, 1.2vw + 0.95rem, 1.45rem)"
    col_padding = "clamp(0.95rem, 1.4vw + 0.7rem, 1.35rem)"
    if dense:
        body_padding = ".45rem .6rem"
        col_padding = ".65rem .75rem"

    col_padding_reset_css = ""
    if id_suffix == "overview-mac-placeholder":
        col_padding_reset_css = (
            "\n"
            + indent(
                dedent(
                    f"""
                    .mw-{suf}__col {{
                      padding-top: 0px;
                      padding-bottom: 0px;
                      padding-left: 0px;
                      padding-right: 0px;
                    }}
                    """
                ).strip("\n"),
                "          ",
            )
        )

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
            width: min(100%, 1020px);
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
            background: linear-gradient(160deg, rgba(248,250,252,.97), rgba(226,232,240,.7));
            box-shadow: inset 0 0 0 1px rgba(148,163,184,.25);
            padding: {col_padding};
            min-height: 180px;
            display: grid;
            align-content: start;
            gap: .65rem;
          }}{col_padding_reset_css}

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
              padding: clamp(.85rem, 5vw, 1.2rem);
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
        """
    )



def render_mac_window(
    st,
    *,
    fallback_height: int | None = None,
    scoped_css: str | Iterable[str] | None = None,
    **kwargs,
):
    """Render the macOS-style window in Streamlit."""

    html_str = mac_window_html(scoped_css=scoped_css, **kwargs)

    # ``components.html`` ensures rich HTML (including scripts like the training
    # animation) render without sanitisation. Always prefer it so interactive
    # payloads behave consistently across Streamlit versions.
    if fallback_height is None:
        # ``components.html`` requires an explicit height; without it, the iframe
        # collapses to 0px which hides rich content like the training animation.
        # Pick a generous default so the mac window has room to render while
        # still allowing call sites to opt in to a custom height when needed.
        fallback_height = 720

    st.components.v1.html(
        html_str,
        height=fallback_height,
        scrolling=True,
    )
