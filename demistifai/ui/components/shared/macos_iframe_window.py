"""Mac-style iframe window renderer with responsive auto-sizing panes."""

from __future__ import annotations

from dataclasses import dataclass
import html
from textwrap import dedent
from typing import List, Literal, Sequence


@dataclass
class MacWindowPane:
    """Definition for a pane rendered inside the macOS-style iframe grid."""

    html: str
    css: str | None = None
    min_height: int | None = None
    max_height: int | None = None
    min_width: int | None = None
    max_width: int | None = None
    pane_id: str | None = None


ThemeVariant = Literal["light", "dark"]


@dataclass
class MacWindowConfig:
    """Configuration for laying out macOS-style iframe panes."""

    panes: Sequence[MacWindowPane]
    rows: int = 1
    columns: int = 1
    column_ratios: Sequence[float] | None = None
    row_ratios: Sequence[float] | None = None
    theme: ThemeVariant = "light"
    mobile_breakpoint: int = 768

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("rows and columns must be positive integers")
        if self.column_ratios is not None and len(self.column_ratios) != self.columns:
            raise ValueError("column_ratios length must match the number of columns")
        if self.row_ratios is not None and len(self.row_ratios) != self.rows:
            raise ValueError("row_ratios length must match the number of rows")
        expected_cells = self.rows * self.columns
        if len(self.panes) != expected_cells:
            raise ValueError(
                "Number of panes must equal rows * columns for macOS window layout"
            )


def build_srcdoc(pane: MacWindowPane, *, window_id: str) -> str:
    """Assemble srcdoc HTML for an iframe pane."""

    if pane.pane_id is None:
        raise ValueError("Pane must have pane_id assigned before building srcdoc")

    base_style = dedent(
        """
        <style>
            html, body {
                margin: 0;
                padding: 0;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                line-height: 1.6;
                color: #0f172a;
                background: transparent;
            }
            body {
                display: block;
            }
            *, *::before, *::after {
                box-sizing: border-box;
            }
            a {
                color: #2563eb;
            }
        </style>
        """
    )
    if pane.css:
        pane_css = pane.css if "<style" in pane.css else f"<style>{pane.css}</style>"
    else:
        pane_css = ""
    monitor_script = dedent(
        f"""
        <script>
            (function() {{
                const paneId = {pane.pane_id!r};
                const windowId = {window_id!r};
                const postHeight = () => {{
                    try {{
                        const height = document.documentElement.scrollHeight || document.body.scrollHeight || 0;
                        parent.postMessage({{ type: "macosPaneHeight", paneId, height, windowId }}, "*");
                    }} catch (error) {{
                        // Suppress cross-origin issues silently.
                    }}
                }};

                const supportsResizeObserver = typeof ResizeObserver !== "undefined";
                if (supportsResizeObserver) {{
                    const observer = new ResizeObserver(() => postHeight());
                    observer.observe(document.documentElement);
                }} else {{
                    setInterval(postHeight, 500);
                }}

                window.addEventListener("load", postHeight);
                if (document.readyState !== "loading") {{
                    postHeight();
                }}

                window.addEventListener("message", (event) => {{
                    const data = event.data;
                    if (!data || data.type !== "macosPanePing") {{
                        return;
                    }}
                    if (data.paneId && data.paneId !== paneId) {{
                        return;
                    }}
                    if (data.windowId && data.windowId !== windowId) {{
                        return;
                    }}
                    postHeight();
                }});

                postHeight();
            }})();
        </script>
        """
    )
    return base_style + pane_css + pane.html + monitor_script


def _resolve_ratio_styles(ratios: Sequence[float] | None, count: int) -> str:
    if not ratios:
        return "repeat(%d, minmax(0, 1fr))" % count
    normalized: List[str] = []
    for value in ratios:
        normalized.append(f"{max(value, 0.01)}fr")
    return " ".join(normalized)


def _build_iframe_styles(pane: MacWindowPane) -> str:
    styles: List[str] = ["width: 100%", "border: 0", "background: transparent", "overflow: hidden"]
    if pane.min_height is not None:
        styles.append(f"min-height: {pane.min_height}px")
    if pane.max_height is not None:
        styles.append(f"max-height: {pane.max_height}px")
    if pane.min_width is not None:
        styles.append(f"min-width: {pane.min_width}px")
    if pane.max_width is not None:
        styles.append(f"max-width: {pane.max_width}px")
    return "; ".join(styles)


def render_macos_iframe_window(st, config: MacWindowConfig) -> None:
    """Render a macOS-inspired iframe window with responsive auto-sized panes."""

    import uuid

    window_id = f"mac-ifw-{uuid.uuid4().hex}"
    theme_class = "miw--dark" if config.theme == "dark" else "miw--light"
    columns_class = f"miw__grid--cols-{config.columns}"
    rows_class = f"miw__grid--rows-{config.rows}"

    panes: List[MacWindowPane] = list(config.panes)
    for index, pane in enumerate(panes):
        pane_id = pane.pane_id or f"{window_id}-pane-{index}"
        pane.pane_id = pane_id

    grid_column_style = _resolve_ratio_styles(config.column_ratios, config.columns)
    grid_row_style = _resolve_ratio_styles(config.row_ratios, config.rows)

    iframe_cells: List[str] = []
    for pane in panes:
        srcdoc = build_srcdoc(pane, window_id=window_id)
        escaped_srcdoc = html.escape(srcdoc, quote=True)
        iframe_style = _build_iframe_styles(pane)
        iframe_cells.append(
            dedent(
                f"""
                <div class="miw__cell" data-pane-id="{pane.pane_id}">
                    <iframe
                        title="macOS window pane"
                        data-pane-id="{pane.pane_id}"
                        srcdoc="{escaped_srcdoc}"
                        sandbox="allow-scripts allow-same-origin"
                        loading="lazy"
                        style="{iframe_style}"
                    ></iframe>
                </div>
                """
            ).strip()
        )

    panes_html = "\n".join(iframe_cells)
    css = dedent(
        f"""
        <style>
            [data-miw="{window_id}"] .miw {{
                --miw-radius: 18px;
                --miw-chrome-bg: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(226,232,240,0.9));
                --miw-chrome-border: rgba(15,23,42,0.15);
                --miw-body-bg: rgba(246,248,252,0.94);
                --miw-body-shadow: 0 28px 60px rgba(15,23,42,0.22);
                --miw-body-padding: clamp(1rem, 2vw, 1.75rem);
                --miw-grid-gap: clamp(1rem, 2.5vw, 1.8rem);
                --miw-light-shadow: inset 0 0 0 1px rgba(15,23,42,0.08);
                --miw-title-color: #0f172a;
                --miw-subtitle-color: rgba(15,23,42,0.65);
                --miw-border-color: rgba(15,23,42,0.08);
                --miw-cell-bg: rgba(255,255,255,0.82);
                --miw-cell-border: 1px solid rgba(148,163,184,0.35);
                --miw-cell-radius: 16px;
                --miw-cell-shadow: inset 0 0 0 1px rgba(148,163,184,0.15);
                position: relative;
                border-radius: var(--miw-radius);
                overflow: hidden;
                background: var(--miw-body-bg);
                box-shadow: var(--miw-body-shadow);
                border: 1px solid var(--miw-border-color);
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                color: var(--miw-title-color);
                isolation: isolate;
            }}

            [data-miw="{window_id}"] .miw--dark {{
                --miw-chrome-bg: linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
                --miw-chrome-border: rgba(255,255,255,0.12);
                --miw-body-bg: rgba(15,23,42,0.92);
                --miw-body-shadow: 0 28px 60px rgba(2,6,23,0.6);
                --miw-title-color: rgba(226,232,240,0.95);
                --miw-subtitle-color: rgba(226,232,240,0.65);
                --miw-border-color: rgba(148,163,184,0.25);
                --miw-cell-bg: rgba(30,41,59,0.85);
                --miw-cell-shadow: inset 0 0 0 1px rgba(148,163,184,0.25);
            }}

            [data-miw="{window_id}"] .miw__chrome {{
                display: grid;
                grid-template-columns: auto 1fr;
                gap: 0.75rem;
                align-items: center;
                padding: 0.85rem 1.2rem;
                background: var(--miw-chrome-bg);
                border-bottom: 1px solid var(--miw-chrome-border);
            }}

            [data-miw="{window_id}"] .miw__lights {{
                display: inline-flex;
                gap: 0.45rem;
            }}

            [data-miw="{window_id}"] .miw__light {{
                width: 12px;
                height: 12px;
                border-radius: 999px;
                box-shadow: var(--miw-light-shadow);
            }}

            [data-miw="{window_id}"] .miw__light--close {{ background: #ff5f56; }}
            [data-miw="{window_id}"] .miw__light--min {{ background: #ffbd2e; }}
            [data-miw="{window_id}"] .miw__light--max {{ background: #27c93f; }}

            [data-miw="{window_id}"] .miw__titles {{
                display: grid;
                gap: 0.2rem;
            }}

            [data-miw="{window_id}"] .miw__title {{
                font-weight: 700;
                font-size: 1.02rem;
                color: var(--miw-title-color);
            }}

            [data-miw="{window_id}"] .miw__subtitle {{
                font-size: 0.85rem;
                color: var(--miw-subtitle-color);
            }}

            [data-miw="{window_id}"] .miw__body {{
                padding: var(--miw-body-padding);
            }}

            [data-miw="{window_id}"] .miw__grid {{
                display: grid;
                gap: var(--miw-grid-gap);
                align-items: stretch;
                grid-template-columns: {grid_column_style};
                grid-template-rows: {grid_row_style};
            }}

            [data-miw="{window_id}"] .miw__grid--cols-1 {{
                grid-template-columns: minmax(0, 1fr);
            }}

            [data-miw="{window_id}"] .miw__grid--cols-2 {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}

            [data-miw="{window_id}"] .miw__grid--cols-3 {{
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }}

            [data-miw="{window_id}"] .miw__grid--rows-1 {{
                grid-template-rows: repeat(1, minmax(0, 1fr));
            }}

            [data-miw="{window_id}"] .miw__grid--rows-2 {{
                grid-template-rows: repeat(2, minmax(0, 1fr));
            }}

            [data-miw="{window_id}"] .miw__grid--rows-3 {{
                grid-template-rows: repeat(3, minmax(0, 1fr));
            }}

            [data-miw="{window_id}"] .miw__cell {{
                display: flex;
                align-items: stretch;
                border-radius: var(--miw-cell-radius);
                background: var(--miw-cell-bg);
                box-shadow: var(--miw-cell-shadow);
                border: var(--miw-cell-border);
                overflow: hidden;
            }}

            [data-miw="{window_id}"] .miw__cell iframe {{
                display: block;
            }}

            @media (max-width: 1100px) {{
                [data-miw="{window_id}"] .miw__grid--cols-3 {{
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }}
            }}

            @media (max-width: 900px) {{
                [data-miw="{window_id}"] .miw__grid--cols-2,
                [data-miw="{window_id}"] .miw__grid--cols-3 {{
                    grid-template-columns: minmax(0, 1fr);
                }}
            }}

            @media (max-width: {config.mobile_breakpoint}px) {{
                [data-miw="{window_id}"] .miw {{
                    border-radius: 16px;
                    box-shadow: 0 20px 44px rgba(15,23,42,0.25);
                }}
                [data-miw="{window_id}"] .miw__chrome {{
                    grid-template-columns: minmax(0, 1fr);
                    row-gap: 0.5rem;
                }}
                [data-miw="{window_id}"] .miw__grid {{
                    grid-template-columns: minmax(0, 1fr) !important;
                    grid-auto-rows: auto;
                }}
                [data-miw="{window_id}"] .miw__cell {{
                    min-height: 0;
                }}
            }}
        </style>
        """
    )

    html_markup = dedent(
        f"""
        <div data-miw="{window_id}" class="miw-container">
            <div class="miw {theme_class}">
                <div class="miw__chrome">
                    <div class="miw__lights">
                        <span class="miw__light miw__light--close" aria-hidden="true"></span>
                        <span class="miw__light miw__light--min" aria-hidden="true"></span>
                        <span class="miw__light miw__light--max" aria-hidden="true"></span>
                    </div>
                    <div class="miw__titles">
                        <span class="miw__title">demistifAI</span>
                        <span class="miw__subtitle">Live window preview</span>
                    </div>
                </div>
                <div class="miw__body">
                    <div class="miw__grid {columns_class} {rows_class}">
                        {panes_html}
                    </div>
                </div>
            </div>
        </div>
        """
    )

    parent_script = dedent(
        f"""
        <script>
            (function() {{
                const windowId = {window_id!r};
                const root = document.querySelector('[data-miw="' + windowId + '"]');
                if (!root) {{
                    return;
                }}
                const iframes = Array.from(root.querySelectorAll('iframe[data-pane-id]'));
                const frameMap = new Map();
                iframes.forEach((frame) => {{
                    frameMap.set(frame.dataset.paneId, frame);
                }});

                const applyHeight = (paneId, height) => {{
                    const frame = frameMap.get(paneId);
                    if (!frame) {{
                        return;
                    }}
                    const rounded = Math.max(40, Math.ceil(Number(height)));
                    frame.style.height = rounded + 'px';
                }};

                window.addEventListener('message', (event) => {{
                    const data = event.data;
                    if (!data || data.type !== 'macosPaneHeight') {{
                        return;
                    }}
                    if (data.windowId && data.windowId !== windowId) {{
                        return;
                    }}
                    applyHeight(data.paneId, data.height);
                }});

                const requestHeight = (frame) => {{
                    if (!frame.contentWindow) {{
                        return;
                    }}
                    frame.contentWindow.postMessage(
                        {{ type: 'macosPanePing', paneId: frame.dataset.paneId, windowId }},
                        '*'
                    );
                }};

                iframes.forEach((frame) => {{
                    frame.addEventListener('load', () => requestHeight(frame), {{ once: true }});
                    setTimeout(() => requestHeight(frame), 120);
                    setInterval(() => requestHeight(frame), 1500);
                }});
            }})();
        </script>
        """
    )

    output = f"{css}{html_markup}{parent_script}"
    st.markdown(output, unsafe_allow_html=True)

    return None
