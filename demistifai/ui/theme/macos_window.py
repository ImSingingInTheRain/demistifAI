"""macOS-style window theme helpers."""

from __future__ import annotations

import html
import re
from textwrap import dedent
from typing import Iterable, Literal, Sequence

ColumnVariant = Literal["standard", "flush"]
ThemeVariant = Literal["light", "dark"]

_MAC_WINDOW_BASE_STYLE = dedent(
    """
    <style>
      .mac-window {
        --mw-max-width: 1020px;
        --mw-chrome-bg: #f5f7fb;
        --mw-chrome-border: rgba(15,23,42,.12);
        --mw-body-bg: #ffffff;
        --mw-title: #0f172a;
        --mw-subtitle: rgba(15,23,42,.70);
        --mw-placeholder: rgba(15,23,42,.55);
        --mw-body-padding: clamp(1rem, 1.2vw + 0.95rem, 1.45rem);
        --mw-col-padding: clamp(0.95rem, 1.4vw + 0.7rem, 1.35rem);
        --mw-col-mobile-padding: clamp(.85rem, 5vw, 1.2rem);
        --mw-col-background: linear-gradient(160deg, rgba(248,250,252,.97), rgba(226,232,240,.7));
        --mw-col-shadow: inset 0 0 0 1px rgba(148,163,184,.25);
        --mw-grid-gap: 1.15rem;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 16px 40px rgba(15,23,42,.12);
        background: var(--mw-body-bg);
        width: min(100%, var(--mw-max-width));
        margin: clamp(0.75rem, 3vw, 1.75rem) auto;
        isolation: isolate;
      }

      .mac-window--dark {
        --mw-chrome-bg: #111827;
        --mw-chrome-border: rgba(255,255,255,.12);
        --mw-body-bg: #0b1220;
        --mw-title: #e5e7eb;
        --mw-subtitle: rgba(229,231,235,.70);
        --mw-placeholder: rgba(229,231,235,.55);
      }

      .mac-window--dense {
        --mw-body-padding: .45rem .6rem;
        --mw-col-padding: .65rem .75rem;
        --mw-col-mobile-padding: clamp(.75rem, 4.5vw, 1rem);
      }

      .mac-window--flush {
        --mw-col-padding: 0;
        --mw-col-mobile-padding: 0;
        --mw-col-background: none;
        --mw-col-shadow: none;
      }

      .mac-window__chrome {
        display: grid;
        grid-template-columns: auto 1fr;
        align-items: center;
        gap: .75rem;
        padding: .65rem 1rem;
        background: var(--mw-chrome-bg);
        border-bottom: 1px solid var(--mw-chrome-border);
      }

      .mac-window__lights {
        display: inline-flex;
        gap: .45rem;
        padding-left: .15rem;
      }

      .mac-window__light {
        width: 12px;
        height: 12px;
        border-radius: 999px;
        box-shadow: inset 0 0 0 1px rgba(0,0,0,.08);
      }

      .mac-window__light--red { background:#ff5f56; }
      .mac-window__light--yellow { background:#ffbd2e; }
      .mac-window__light--green { background:#27c93f; }

      .mac-window__titles {
        display: grid;
        gap: .15rem;
      }

      .mac-window__title {
        font-weight: 800;
        color: var(--mw-title);
        line-height: 1.2;
      }

      .mac-window__subtitle {
        font-size: .92rem;
        color: var(--mw-subtitle);
      }

      .mac-window__mobile-header {
        display: none;
        margin-bottom: clamp(.75rem, 4.5vw, 1rem);
        gap: .35rem;
      }

      .mac-window__body {
        padding: var(--mw-body-padding);
      }

      .mac-window__grid {
        display: grid;
        grid-template-columns: var(--mw-grid-template, 1fr);
        gap: var(--mw-grid-gap);
        align-items: stretch;
      }

      .mac-window__col {
        border-radius: 12px;
        background: var(--mw-col-background);
        box-shadow: var(--mw-col-shadow);
        padding: var(--mw-col-padding);
        min-height: 180px;
        display: grid;
        align-content: start;
        gap: .65rem;
      }

      .mac-window__placeholder {
        color: var(--mw-placeholder);
      }

      .mac-window__placeholder-title {
        font-weight: 700;
        margin-bottom: .15rem;
      }

      .mac-window__placeholder-lines {
        display: grid;
        gap: .45rem;
      }

      .mac-window__placeholder-lines span {
        display:block;
        height: 9px;
        border-radius: 6px;
        background: linear-gradient(90deg, rgba(148,163,184,.28), rgba(148,163,184,.16), rgba(148,163,184,.28));
        background-size: 180% 100%;
        animation: mac-window-shimmer 1.6s infinite linear;
      }

      .mac-window__placeholder-lines span.short {
        width: 60%;
      }

      @keyframes mac-window-shimmer {
        0% { background-position: 180% 0; }
        100% { background-position: -20% 0; }
      }

      @media (max-width: 920px) {
        .mac-window__grid {
          grid-template-columns: 1fr;
        }
      }

      @media (max-width: 720px) {
        .mac-window {
          --mw-max-width: 100%;
          border-radius: 12px;
          box-shadow: 0 14px 32px rgba(15,23,42,.12);
        }

        .mac-window__chrome {
          padding: .55rem .85rem;
          grid-template-columns: 1fr;
          row-gap: .5rem;
        }

        .mac-window__lights {
          order: 2;
          justify-self: start;
        }

        .mac-window__body {
          padding: clamp(.85rem, 5vw, 1.15rem);
        }

        .mac-window__grid {
          gap: clamp(.75rem, 4.5vw, 1rem);
        }

        .mac-window__col {
          min-height: 0;
          padding: var(--mw-col-mobile-padding);
        }
      }

      @media (max-width: 600px) {
        .mac-window {
          border-radius: 0;
          box-shadow: none;
          margin: 0;
          border: none;
        }

        .mac-window__chrome {
          display: none;
        }

        .mac-window__body {
          padding: clamp(.9rem, 5.5vw, 1.35rem) 1rem 1.25rem;
        }

        .mac-window__grid {
          grid-template-columns: 1fr;
          gap: clamp(.75rem, 4.5vw, 1rem);
        }

        .mac-window__col {
          padding: 0;
          border-radius: 0;
          background: none;
          box-shadow: none;
          border: 0;
          min-height: 0;
        }

        .mac-window__mobile-header {
          display: grid;
        }
      }
    </style>
    """
).strip()

_BASE_STYLE_EMITTED = False


def _strip_style_wrappers(css_chunk: str) -> str:
    """Return CSS stripped of optional ``<style>`` wrappers."""

    cleaned = (css_chunk or "").strip()
    if not cleaned:
        return ""

    if "<style" in cleaned.lower():
        cleaned = re.sub(r"^<style[^>]*>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        cleaned = re.sub(r"</style>$", "", cleaned, flags=re.IGNORECASE).strip()

    return cleaned


def _normalise_scoped_css(scoped_css: str | Iterable[str] | None) -> str:
    """Normalise scoped CSS blocks into a ``<style>`` element."""

    if scoped_css is None:
        return ""

    if isinstance(scoped_css, str):
        css_chunks: Iterable[str] = [scoped_css]
    else:
        css_chunks = (chunk for chunk in scoped_css if chunk)

    cleaned_chunks = [_strip_style_wrappers(chunk) for chunk in css_chunks]
    cleaned_chunks = [chunk for chunk in cleaned_chunks if chunk]

    if not cleaned_chunks:
        return ""

    combined = "\n\n".join(cleaned_chunks)
    return f"<style>\n{combined}\n</style>"


def _format_max_width(max_width: float | int | str) -> str:
    """Normalise ``max_width`` values into CSS tokens."""

    if isinstance(max_width, (int, float)):
        if max_width <= 0:
            raise ValueError("max_width must be positive")
        return f"{format(max_width, 'g')}px"

    max_width_css = str(max_width).strip()
    if not max_width_css:
        raise ValueError("max_width must not be empty")
    return max_width_css


def _format_grid_template(columns: int, ratios: Sequence[float] | None) -> str:
    """Return the CSS ``grid-template-columns`` declaration for the window."""

    if columns not in (1, 2, 3):
        raise ValueError("columns must be 1, 2 or 3")

    if ratios is None:
        ratios = tuple(1 for _ in range(columns))

    if len(ratios) != columns or any(r <= 0 for r in ratios):
        raise ValueError("ratios must match columns and be positive")

    total = float(sum(ratios))
    return " ".join(f"{(r / total) * 100:.5f}%" for r in ratios)


def _suffix_class(prefix: str | None, element: str) -> str:
    """Return a suffix-scoped class for the provided element."""

    if not prefix:
        return ""
    return f" {prefix}{element}"


def macos_window_styles() -> str:
    """Return the shared macOS window stylesheet."""

    return _MAC_WINDOW_BASE_STYLE


def macos_window_markup(
    title: str,
    *,
    subtitle: str | None = None,
    columns: int = 1,
    ratios: Sequence[float] | None = None,
    id_suffix: str | None = None,
    column_blocks: Sequence[str | None] | None = None,
    dense: bool = False,
    theme: ThemeVariant = "light",
    column_variant: ColumnVariant = "standard",
    max_width: float | int | str = 1020,
) -> str:
    """Return HTML markup for a macOS-style window."""

    if column_variant not in ("standard", "flush"):
        raise ValueError("column_variant must be 'standard' or 'flush'")

    max_width_css = _format_max_width(max_width)
    grid_template = _format_grid_template(columns, ratios)

    root_classes = ["mac-window"]
    if theme == "dark":
        root_classes.append("mac-window--dark")
    if dense:
        root_classes.append("mac-window--dense")
    if column_variant == "flush":
        root_classes.append("mac-window--flush")

    suffix_prefix = f"mw-{id_suffix}" if id_suffix else None
    if suffix_prefix:
        root_classes.append(suffix_prefix)

    escaped_title = html.escape(title)
    subtitle_html = html.escape(subtitle) if subtitle else ""

    subtitle_block = (
        f'<div class="mac-window__subtitle{_suffix_class(suffix_prefix, "__subtitle")}">{subtitle_html}</div>'
        if subtitle_html
        else ""
    )

    mobile_subtitle_block = (
        f'<div class="mac-window__subtitle{_suffix_class(suffix_prefix, "__subtitle")}">{subtitle_html}</div>'
        if subtitle_html
        else ""
    )

    column_contents = list(column_blocks or [])
    if not column_contents:
        column_contents = ["" for _ in range(columns)]
    elif len(column_contents) < columns:
        column_contents.extend(["" for _ in range(columns - len(column_contents))])
    elif len(column_contents) > columns:
        raise ValueError("column_blocks length cannot exceed the number of columns")

    column_class = f"mac-window__col{_suffix_class(suffix_prefix, '__col')}"
    columns_markup = "\n".join(
        f'<div class="{column_class}">{content or ""}</div>' for content in column_contents
    )

    return (
        dedent(
            f"""
            <section class="{' '.join(root_classes)}" style="--mw-max-width: {max_width_css}; --mw-grid-template: {grid_template};" role="group" aria-label="{escaped_title} window">
              <header class="mac-window__chrome{_suffix_class(suffix_prefix, '__chrome')}" aria-hidden="false">
                <div class="mac-window__lights" aria-hidden="true">
                  <span class="mac-window__light mac-window__light--red"></span>
                  <span class="mac-window__light mac-window__light--yellow"></span>
                  <span class="mac-window__light mac-window__light--green"></span>
                </div>
                <div class="mac-window__titles{_suffix_class(suffix_prefix, '__titles')}">
                  <div class="mac-window__title{_suffix_class(suffix_prefix, '__title')}">{escaped_title}</div>
                  {subtitle_block}
                </div>
              </header>
              <div class="mac-window__body{_suffix_class(suffix_prefix, '__body')}">
                <div class="mac-window__mobile-header{_suffix_class(suffix_prefix, '__mobile-header')}">
                  <div class="mac-window__title{_suffix_class(suffix_prefix, '__title')}">{escaped_title}</div>
                  {mobile_subtitle_block}
                </div>
                <div class="mac-window__grid{_suffix_class(suffix_prefix, '__grid')}">
                  {columns_markup}
                </div>
              </div>
            </section>
            """
        )
        .strip()
    )


def build_macos_window(
    *,
    title: str,
    subtitle: str | None = None,
    column_blocks: Sequence[str | None],
    ratios: Sequence[float] | None = None,
    id_suffix: str | None = None,
    dense: bool = False,
    theme: ThemeVariant = "light",
    column_variant: ColumnVariant = "standard",
    max_width: float | int | str = 1020,
    columns: int | None = None,
    scoped_css: str | Iterable[str] | None = None,
) -> str:
    """Return combined CSS (if any) and markup for a macOS window."""

    resolved_columns = columns or len(column_blocks)
    if resolved_columns <= 0:
        raise ValueError("columns must be positive")

    markup = macos_window_markup(
        title,
        subtitle=subtitle,
        columns=resolved_columns,
        ratios=ratios,
        id_suffix=id_suffix,
        column_blocks=column_blocks,
        dense=dense,
        theme=theme,
        column_variant=column_variant,
        max_width=max_width,
    )

    scoped_style = _normalise_scoped_css(scoped_css)
    if scoped_style:
        return f"{scoped_style}\n{markup}"
    return markup


def inject_macos_window_theme(st) -> None:
    """Ensure the shared mac-window stylesheet is registered once."""

    global _BASE_STYLE_EMITTED
    if _BASE_STYLE_EMITTED:
        return

    st.markdown(macos_window_styles(), unsafe_allow_html=True)
    _BASE_STYLE_EMITTED = True
