# demistifai/ui/components/train_animation.py
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
try:  # Optional dependency — guard so the UI degrades gracefully when absent.
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover
    go = None

import streamlit as st

# --------- Tunables ----------
MAP_X = (-3.0, 3.0)
MAP_Y = (-3.0, 3.0)
N_POINTS = 500                 # 300–800 feels good on Streamlit Cloud
EPOCHS = 10                    # frames 0..10
EASING = 0.25                  # 0..1, higher = faster convergence
RANDOM_SEED = 7

# Cluster definitions (center x,y and radius-ish spread)
# Coordinates loosely aligned to your static diagram
CLUSTERS = [
    # name,                cx,   cy,   rx,  ry,  color, dot_class
    ("Promotions",        -2.0, -1.0, 0.8, 0.6, "#FAD4D4", "spam"),
    ("Security alerts",   -0.8, -0.5, 1.0, 0.6, "#F9D7DF", "spam"),
    ("Courier tracking",   0.4,  0.4, 0.9, 0.6, "#F9EEBD", "spam"),
    ("Project updates",    1.8,  1.0, 0.9, 0.6, "#D7E8F7", "work"),
    ("Meeting emails",     2.4,  2.2, 0.8, 0.6, "#C9E3F1", "work"),
]

# --------------------- Brand Token Bridge (light mode only) ---------------------
def _brand_tokens() -> Dict[str, Any]:
    """
    Pulls brand colors from demistifai.constants if available, otherwise uses
    polished, light-mode defaults. You can override by defining in constants.py:

        BRAND_COLORS = {
            "primary": "#3C7BE5",      # used for 'work'
            "danger":  "#E55B3C",      # used for 'spam'
            "paper":   "#FAFCFF",
            "plot_bg": "rgba(240,245,255,0.4)",
            "grid":    "rgba(148,163,184,0.28)",
            "text":    "#1F2937",
            "text_muted": "rgba(31,41,55,0.75)",
            "axis_title": "rgba(30,41,59,0.9)",
            "area_label_bg": "rgba(241,245,249,0.92)",
            "area_label_fg": "rgba(15,23,42,0.9)",
        }
    """
    defaults = dict(
        paper="#FAFCFF",
        plot_bg="rgba(240,245,255,0.4)",
        grid="rgba(148,163,184,0.28)",
        text="#1F2937",
        text_muted="rgba(30,41,59,0.75)",
        axis_title="rgba(30,41,59,0.9)",
        note_txt="rgba(40,40,40,0.8)",
        border="rgba(15,23,42,0.10)",
        spam="#E55B3C",           # warm accent
        work="#3C7BE5",           # cool accent
        area_stroke="rgba(60,60,60,0.28)",
        area_label_bg="rgba(241,245,249,0.92)",
        area_label_fg="rgba(15,23,42,0.9)",
    )
    try:
        # You can expose either BRAND_COLORS or COLORS in constants.py.
        from demistifai.constants import BRAND_COLORS as BC  # type: ignore
        brand = dict(defaults)
        brand.update({
            "paper": BC.get("paper", brand["paper"]),
            "plot_bg": BC.get("plot_bg", brand["plot_bg"]),
            "grid": BC.get("grid", brand["grid"]),
            "text": BC.get("text", brand["text"]),
            "text_muted": BC.get("text_muted", brand["text_muted"]),
            "axis_title": BC.get("axis_title", brand["axis_title"]),
            "spam": BC.get("danger", brand["spam"]),
            "work": BC.get("primary", brand["work"]),
            "area_label_bg": BC.get("area_label_bg", brand["area_label_bg"]),
            "area_label_fg": BC.get("area_label_fg", brand["area_label_fg"]),
        })
        return brand
    except Exception:
        # Fallback: try a generic COLORS dict if present
        try:
            from demistifai.constants import COLORS as C  # type: ignore
            brand = dict(defaults)
            brand.update({
                "paper": C.get("paper", brand["paper"]),
                "plot_bg": C.get("plot_bg", brand["plot_bg"]),
                "grid": C.get("grid", brand["grid"]),
                "text": C.get("text", brand["text"]),
                "text_muted": C.get("text_muted", brand["text_muted"]),
                "axis_title": C.get("axis_title", brand["axis_title"]),
                "spam": C.get("danger", brand["spam"]),
                "work": C.get("primary", brand["work"]),
                "area_label_bg": C.get("area_label_bg", brand["area_label_bg"]),
                "area_label_fg": C.get("area_label_fg", brand["area_label_fg"]),
            })
            return brand
        except Exception:
            return defaults

# ---------------------------- Wrapper styles (HTML) ----------------------------
BASE_STYLES = textwrap.dedent(
    """
    <style>
      .train-animation__body {
        position: relative;
        display: grid;
        gap: 0.85rem;
        padding: 1.15rem 1.25rem 1.35rem;
        border-radius: 1rem;
        background: linear-gradient(160deg, rgba(148,163,184,0.08), rgba(226,232,240,0.22));
        box-shadow: 0 10px 24px -18px rgba(15,23,42,.38);
      }
      .train-animation__eyebrow {
        font-size: 0.74rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 600;
        color: rgba(15,23,42,0.58);
      }
      .train-animation__title {
        margin: 0;
        font-size: 1.08rem;
        font-weight: 700;
        color: #0f172a;
      }
      .train-animation__caption {
        margin: 0;
        font-size: 0.94rem;
        line-height: 1.5;
        color: rgba(15,23,42,0.74);
      }
      .train-animation__legend {
        display: inline-flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        padding: 0.6rem 0.75rem;
        border-radius: 0.75rem;
        background: rgba(15,23,42,0.05);
      }
      .train-animation__legend-item {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.82rem;
        color: rgba(15,23,42,0.75);
        font-weight: 600;
      }
      .train-animation__legend-swatch {
        width: 0.85rem;
        height: 0.85rem;
        border-radius: 999px;
        box-shadow: inset 0 0 0 2px rgba(255,255,255,0.85);
      }
      .train-animation__legend-swatch--spam { background: #e55b3c; }
      .train-animation__legend-swatch--work { background: #3c7be5; }
      .train-animation__plotly {
        width: 100%;
        border-radius: 0.85rem;
        overflow: hidden;
        background: radial-gradient(circle at 35% -10%, rgba(148,163,184,.14), rgba(15,23,42,.03));
      }
      .train-animation__plotly > div { width: 100% !important; }
      .train-animation__footnote {
        margin: 0;
        font-size: 0.78rem;
        color: rgba(15,23,42,0.55);
      }
      .train-animation__fallback {
        padding: 0.9rem 1rem;
        border-radius: 0.85rem;
        border: 1px dashed rgba(15,23,42,0.25);
        background: rgba(59,130,246,0.08);
        color: rgba(15,23,42,0.78);
        font-size: 0.92rem;
        line-height: 1.5;
      }
      .train-animation__fallback strong { color: #1d4ed8; }
      .train-animation__fallback code {
        background: rgba(15,23,42,0.08);
        padding: 0.1rem 0.35rem;
        border-radius: 0.35rem;
        font-size: 0.85rem;
      }
      @media (max-width: 780px) {
        .train-animation__body { padding: 1rem; gap: 0.75rem; }
        .train-animation__legend { width: 100%; justify-content: space-between; }
      }
      @media (max-width: 520px) {
        .train-animation__body { padding: 0.85rem 0.9rem 1.05rem; }
        .train-animation__title { font-size: 1rem; }
        .train-animation__caption { font-size: 0.88rem; }
        .train-animation__legend-item { font-size: 0.78rem; }
        .train-animation__plotly { border-radius: 0.75rem; }
      }
    </style>
    """
).strip()

# ------------------------------ Small utilities -------------------------------
@dataclass(frozen=True)
class TrainingAnimationColumn:
    """Encapsulate the column markup for the training animation."""
    html: str
    fallback_height: int = 720

def _sample_targets(rng: np.random.Generator, n: int):
    """Assign each point a class and a target cluster center."""
    n_spam = int(n * 0.55)
    n_work = n - n_spam
    spam_clusters = [c for c in CLUSTERS if c[6] == "spam"]
    work_clusters = [c for c in CLUSTERS if c[6] == "work"]

    def pick_targets(k, pool):
        idx = rng.integers(0, len(pool), size=k)
        targets = [pool[i] for i in idx]
        tx = np.array([t[1] for t in targets]) + rng.normal(0, 0.25, size=k)
        ty = np.array([t[2] for t in targets]) + rng.normal(0, 0.25, size=k)
        cls = [t[6] for t in targets]
        return tx, ty, cls

    sx, sy, scls = pick_targets(n_spam, spam_clusters)
    wx, wy, wcls = pick_targets(n_work, work_clusters)
    target_x = np.concatenate([sx, wx])
    target_y = np.concatenate([sy, wy])
    cls = np.array(scls + wcls)
    return target_x, target_y, cls

def _interpolate(x0, y0, x1, y1, t):
    """Ease points toward their targets with a smooth-step feeling."""
    a = t
    eased = 1 - pow(1 - a, 3)  # cubic ease-out
    xi = x0 + (x1 - x0) * eased
    yi = y0 + (y1 - y0) * eased
    return xi, yi

# ---------------------------------- Figure ------------------------------------
def build_training_animation_figure(*, seed: Optional[int] = None) -> "go.Figure":
    """Return the Plotly figure used in the training animation (light mode)."""
    if go is None:  # pragma: no cover
        raise RuntimeError("Plotly is required to build the training animation.")

    T = _brand_tokens()
    CLASS_TO_COLOR = {"spam": T["spam"], "work": T["work"]}
    rng = np.random.default_rng(RANDOM_SEED if seed is None else seed)

    # Random start positions
    x0 = rng.uniform(MAP_X[0], MAP_X[1], size=N_POINTS)
    y0 = rng.uniform(MAP_Y[0], MAP_Y[1], size=N_POINTS)

    # Target positions + classes
    tx, ty, classes = _sample_targets(rng, N_POINTS)
    colors = np.vectorize(CLASS_TO_COLOR.get)(classes)

    # Precompute frames (epoch 0..EPOCHS)
    frames = []
    for epoch in range(EPOCHS + 1):
        t = (epoch / EPOCHS) ** (1.0 - EASING)
        xi, yi = _interpolate(x0, y0, tx, ty, t)
        frames.append(go.Frame(
            name=f"Epoch {epoch}",
            data=[go.Scatter(
                x=xi, y=yi,
                mode="markers",
                marker=dict(
                    size=7.5,
                    opacity=0.9,
                    color=colors,
                    line=dict(color="rgba(0,0,0,0.12)", width=0.7),
                ),
                hoverinfo="skip",
                showlegend=False,
            )]
        ))

    base_scatter = frames[0].data[0]

    # Cluster "blobs" + pill labels
    shapes, annotations = [], []
    for (name, cx, cy, rx, ry, fill, _cls) in CLUSTERS:
        shapes.append(dict(
            type="circle",
            x0=cx - rx, x1=cx + rx, y0=cy - ry, y1=cy + ry,
            line=dict(color=T["area_stroke"], width=2, dash="dot"),
            fillcolor=fill, opacity=0.22, layer="below",
        ))
        annotations.append(dict(
            x=cx, y=cy + ry - 0.01, text=name,
            xanchor="center", yanchor="bottom", showarrow=False,
            font=dict(size=12.5, color=T["area_label_fg"]),
            bgcolor=T["area_label_bg"], borderpad=4, bordercolor="rgba(0,0,0,0)",
        ))

    # Decision boundary
    shapes.append(dict(
        type="line", x0=0, x1=0, y0=MAP_Y[0], y1=MAP_Y[1],
        line=dict(color=T["grid"], width=2, dash="dash")
    ))

    fig = go.Figure(
        data=[base_scatter],
        layout=go.Layout(
            margin=dict(l=26, r=16, b=46, t=42),
            paper_bgcolor=T["paper"],
            plot_bgcolor=T["plot_bg"],
            xaxis=dict(range=[MAP_X[0], MAP_X[1]], title="Meaning dimension 1", zeroline=False),
            yaxis=dict(range=[MAP_Y[0], MAP_Y[1]], title="Meaning dimension 2", zeroline=False),
            shapes=shapes,
            annotations=annotations + [
                dict(x=0.02, y=0.98, xref="paper", yref="paper",
                     text="miniLM training: points move toward meaning areas",
                     xanchor="left", yanchor="top", showarrow=False,
                     font=dict(size=12, color=T["note_txt"]))
            ],
            updatemenus=[dict(
                type="buttons", direction="left",
                x=0.0, y=1.12, xanchor="left", yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"fromcurrent": True,
                                      "frame": {"duration": 320, "redraw": True},
                                      "transition": {"duration": 240}}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                active=0, len=0.94, x=0.03, pad=dict(t=10),
                currentvalue={"prefix": "Epoch: ", "font": {"size": 12, "color": T["text_muted"]}},
                steps=[dict(method="animate",
                            args=[[f"Epoch {k}"], {"mode": "immediate",
                                                   "frame": {"duration": 0, "redraw": True},
                                                   "transition": {"duration": 0}}],
                            label=str(k)) for k in range(EPOCHS + 1)]
            )],
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor=T["border"], borderwidth=1,
                font=dict(size=12, color=T["text_muted"]),
            ),
        ),
        frames=frames
    )

    # Axes styling
    fig.update_xaxes(
        showgrid=True, gridcolor=T["grid"], zeroline=False,
        tickfont=dict(size=11, color=T["text_muted"]),
        title=dict(font=dict(size=12, color=T["axis_title"])),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=T["grid"], zeroline=False,
        tickfont=dict(size=11, color=T["text_muted"]),
        title=dict(font=dict(size=12, color=T["axis_title"])),
    )

    # Legend proxies (kept minimal; you can remove if using HTML legend only)
    for cls_name, color in {"spam": T["spam"], "work": T["work"]}.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color,
                        line=dict(color="rgba(15,23,42,0.25)", width=1)),
            name="Spam-like" if cls_name == "spam" else "Work-like"
        ))

    return fig

# ---------------------------- HTML wrapper (optional) --------------------------
def build_training_animation_column(*, seed: Optional[int] = None) -> TrainingAnimationColumn:
    body_html = textwrap.dedent(
        """
        <div class="train-animation__body">
          <span class="train-animation__eyebrow">Conceptual view</span>
          <h4 class="train-animation__title">How miniLM learns a meaning space</h4>
          <p class="train-animation__caption">
            Watch the embedding points move toward <strong>spam-like</strong> and <strong>work-like</strong> regions as epochs progress.
          </p>
          <div class="train-animation__legend">
            <span class="train-animation__legend-item"><span class="train-animation__legend-swatch train-animation__legend-swatch--spam"></span>Spam-like emails</span>
            <span class="train-animation__legend-item"><span class="train-animation__legend-swatch train-animation__legend-swatch--work"></span>Work-like emails</span>
          </div>
        {content}
          <p class="train-animation__footnote">Epochs advance automatically — press play to see the clusters tighten.</p>
        </div>
        """
    ).strip()

    if go is None:
        fallback_html = body_html.format(
            content=textwrap.dedent(
                """
                <div class="train-animation__fallback">
                  <p><strong>Interactive animation unavailable.</strong> This view relies on the optional <code>plotly</code> dependency.</p>
                  <p>Install <code>plotly</code> in your environment and refresh the page to see emails cluster during training.</p>
                </div>
                """
            ).strip()
        )
        return TrainingAnimationColumn(html=f"{BASE_STYLES}\n{fallback_html}")

    fig = build_training_animation_figure(seed=seed)
    interactive_html = fig.to_html(
        include_plotlyjs="inline",
        full_html=False,
        config={"displayModeBar": False},
    )

    animated_html = body_html.format(
        content=textwrap.dedent(
            f"""
          <div class="train-animation__plotly">
            {interactive_html}
          </div>
            """
        ).strip()
    )

    return TrainingAnimationColumn(html=f"{BASE_STYLES}\n{animated_html}")

# -------------------------------- Public render --------------------------------
def render_training_animation():
    """Simple Streamlit render (uses st.plotly_chart)."""
    st.subheader("How miniLM learns a meaning space")
    if go is None:
        st.info("Install the optional 'plotly' dependency to view the interactive training animation.")
        return
    fig = build_training_animation_figure()
    st.plotly_chart(fig, use_container_width=True)