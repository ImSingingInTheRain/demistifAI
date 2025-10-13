from __future__ import annotations

import html
import uuid
from textwrap import dedent, indent


def mac_window_html(
    title: str = "demAI",
    subtitle: str | None = None,
    columns: int = 2,
    ratios: tuple[float, ...] | None = None,
    col_html: list[str] | None = None,
    dense: bool = False,
    theme: str = "light",
    id_suffix: str | None = None,
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

    pad = ".9rem 1.05rem" if dense else "0rem 0rem"

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
            padding: {pad};
          }}
          .mw-{suf}__grid {{
            display: grid;
            grid-template-columns: {cols_css};
            gap: 1.0rem;
            align-items: stretch;
          }}
          .mw-{suf}__col {{
            border-radius: 12px;
            background: linear-gradient(155deg, rgba(248,250,252,.95), rgba(226,232,240,.55));
            box-shadow: inset 0 0 0 1px rgba(148,163,184,.25);
            padding: 0rem 0rem;
            min-height: 200px;
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



def render_mac_window(st, **kwargs):
    """Render the macOS-style window in Streamlit."""

    html_str = mac_window_html(**kwargs)

    # Streamlit 1.50 introduced ``st.html`` which properly renders rich HTML and
    # scripts without escaping them back to text. Use it when available so the
    # lifecycle map (and other interactive payloads) execute as intended.
    render_html = getattr(st, "html", None)
    if callable(render_html):
        render_html(html_str)
        return

    # Fall back to the legacy markdown pathway for older Streamlit versions.
    st.markdown(html_str, unsafe_allow_html=True)
