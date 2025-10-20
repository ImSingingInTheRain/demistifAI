"""Mac-style iframe window renderer with responsive auto-sizing panes."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import re
from textwrap import dedent
from typing import Callable, Dict, List, Literal, Sequence


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


_UNESCAPED_AMPERSAND_PATTERN = re.compile(r"&(?![#\w]+;)")


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


def _escape_srcdoc_attribute(value: str) -> str:
    """Escape a srcdoc payload for safe embedding in an iframe attribute."""

    sanitized = _UNESCAPED_AMPERSAND_PATTERN.sub("&amp;", value)
    return sanitized.replace('"', "&quot;")


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


def _estimate_component_height(config: MacWindowConfig, panes: Sequence[MacWindowPane]) -> int:
    """Best-effort initial height for embedding the macOS window."""

    if not panes:
        return 360

    chrome_height = 88  # Title bar and light strip area.
    body_padding = 72  # Top + bottom padding defined via CSS variables.
    row_gap = 28  # Approximate grid gap between stacked rows.
    default_pane_height = 360

    row_heights: List[int] = [0 for _ in range(config.rows)]
    for index, pane in enumerate(panes):
        row_index = min(index // config.columns, config.rows - 1)
        candidate = pane.min_height or pane.max_height or default_pane_height
        row_heights[row_index] = max(row_heights[row_index], int(candidate))

    total_height = chrome_height + body_padding + sum(row_heights)
    if config.rows > 1:
        total_height += row_gap * (config.rows - 1)

    return max(total_height, 320)


def _resolve_html_renderer(st) -> Callable[..., None] | None:
    """Return a callable that can render raw HTML within ``st`` containers."""

    html_renderer = getattr(st, "html", None)
    if callable(html_renderer):
        return html_renderer

    components = getattr(st, "components", None)
    if components is not None:
        v1 = getattr(components, "v1", None)
        if v1 is not None:
            html_renderer = getattr(v1, "html", None)
            if callable(html_renderer):
                return html_renderer

    return None


def _render_with_compatible_signature(
    renderer: Callable[..., None],
    markup: str,
    *,
    fallback_height: int,
) -> None:
    """Invoke ``renderer`` with kwargs supported by its signature."""

    try:
        signature = inspect.signature(renderer)
    except (TypeError, ValueError):
        signature = None

    kwargs: Dict[str, object] = {}
    if signature is not None:
        parameters = signature.parameters
        accepts_var_kw = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if accepts_var_kw or "height" in parameters:
            kwargs["height"] = fallback_height
        if accepts_var_kw or "scrolling" in parameters:
            kwargs["scrolling"] = False
    else:
        kwargs = {"height": fallback_height, "scrolling": False}

    try:
        renderer(markup, **kwargs)
    except TypeError:
        renderer(markup)


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
    default_fallback_height = 360
    for pane in panes:
        srcdoc = build_srcdoc(pane, window_id=window_id)
        escaped_srcdoc = _escape_srcdoc_attribute(srcdoc)
        iframe_style = _build_iframe_styles(pane)
        fallback_height = (
            pane.min_height
            or pane.max_height
            or default_fallback_height
        )
        data_attributes = [
            f'data-pane-id="{pane.pane_id}"',
            f'data-fallback-height="{int(fallback_height)}"',
        ]
        if pane.min_height is not None:
            data_attributes.append(f'data-min-height="{int(pane.min_height)}"')
        if pane.max_height is not None:
            data_attributes.append(f'data-max-height="{int(pane.max_height)}"')
        iframe_data_attrs = " ".join(data_attributes)
        iframe_cells.append(
            dedent(
                f"""
                <div class="miw__cell" data-pane-id="{pane.pane_id}">
                    <iframe
                        title="macOS window pane"
                        {iframe_data_attrs}
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

                const isEmbedded = window.parent && window.parent !== window;
                let pendingHeightFrame = null;

                const computeContainerHeight = () => {{
                    const doc = document.documentElement;
                    const body = document.body;
                    const primary = root.querySelector('.miw');
                    const baseHeight = primary ? primary.getBoundingClientRect().height : 0;
                    const offsetTop = primary ? primary.offsetTop || 0 : 0;
                    const docHeights = [
                        doc ? doc.scrollHeight : 0,
                        doc ? doc.offsetHeight : 0,
                        body ? body.scrollHeight : 0,
                        body ? body.offsetHeight : 0,
                        baseHeight + offsetTop,
                    ];
                    const measured = Math.max(...docHeights);
                    return Math.ceil(measured);
                }};

                const scheduleComponentHeightUpdate = () => {{
                    if (!isEmbedded) {{
                        return;
                    }}
                    if (pendingHeightFrame !== null) {{
                        cancelAnimationFrame(pendingHeightFrame);
                    }}
                    pendingHeightFrame = requestAnimationFrame(() => {{
                        pendingHeightFrame = null;
                        const height = computeContainerHeight();
                        try {{
                            window.parent.postMessage(
                                {{ isStreamlitMessage: true, type: 'streamlit:setFrameHeight', height }},
                                '*'
                            );
                        }} catch (error) {{
                            // Ignore cross-origin or missing parent exceptions.
                        }}
                    }});
                }};

                const DEFAULT_PANE_HEIGHT = {default_fallback_height};

                const resolveHeight = (frame, height) => {{
                    const numericHeight = Number(height);
                    if (Number.isFinite(numericHeight) && numericHeight > 0) {{
                        return numericHeight;
                    }}

                    const fallback = Number(frame.dataset.fallbackHeight);
                    if (Number.isFinite(fallback) && fallback > 0) {{
                        return fallback;
                    }}

                    return DEFAULT_PANE_HEIGHT;
                }};

                const clampHeight = (frame, value) => {{
                    let result = value;
                    const minHeight = Number(frame.dataset.minHeight);
                    if (Number.isFinite(minHeight) && minHeight > 0) {{
                        result = Math.max(result, minHeight);
                    }}
                    const maxHeight = Number(frame.dataset.maxHeight);
                    if (Number.isFinite(maxHeight) && maxHeight > 0) {{
                        result = Math.min(result, maxHeight);
                    }}
                    return result;
                }};

                const applyHeight = (paneId, height) => {{
                    const frame = frameMap.get(paneId);
                    if (!frame) {{
                        return;
                    }}
                    const resolved = resolveHeight(frame, height);
                    const clamped = clampHeight(frame, resolved);
                    const rounded = Math.max(40, Math.ceil(clamped));
                    frame.style.height = rounded + 'px';
                    scheduleComponentHeightUpdate();
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

                scheduleComponentHeightUpdate();
                window.addEventListener('load', scheduleComponentHeightUpdate, {{ once: true }});
                window.addEventListener('resize', scheduleComponentHeightUpdate);
            }})();
        </script>
        """
    )

    output = f"{css}{html_markup}{parent_script}"

    html_renderer = _resolve_html_renderer(st)
    if html_renderer is not None:
        fallback_height = _estimate_component_height(config, panes)
        _render_with_compatible_signature(
            html_renderer, output, fallback_height=fallback_height
        )
    else:
        st.markdown(output, unsafe_allow_html=True)

    return None
