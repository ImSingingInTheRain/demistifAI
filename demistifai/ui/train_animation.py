# demistifai/ui/components/train_animation.py
from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Optional

import numpy as np
try:  # Optional dependency — guard so the UI degrades gracefully when absent.
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover - simple availability branch
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

CLASS_TO_COLOR = {"spam": "#E55B3C", "work": "#3C7BE5"}  # red / blue
CLASS_PROBS = {"spam": 0.55, "work": 0.45}               # tweak if needed


BASE_STYLES = textwrap.dedent(
    """
    <style>
      .train-animation__body {
        display: grid;
        gap: 0.75rem;
      }
      .train-animation__title {
        margin: 0;
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
      }
      .train-animation__caption {
        margin: 0;
        font-size: 0.92rem;
        line-height: 1.45;
        color: rgba(15, 23, 42, 0.74);
      }
      .train-animation__plotly {
        width: 100%;
      }
      .train-animation__plotly > div {
        width: 100% !important;
      }
      .train-animation__fallback {
        padding: 0.85rem 1rem;
        border-radius: 0.75rem;
        border: 1px dashed rgba(15, 23, 42, 0.25);
        background: rgba(59, 130, 246, 0.08);
        color: rgba(15, 23, 42, 0.78);
        font-size: 0.92rem;
        line-height: 1.5;
      }
      .train-animation__fallback strong {
        color: #1d4ed8;
      }
      .train-animation__fallback code {
        background: rgba(15, 23, 42, 0.08);
        padding: 0.1rem 0.35rem;
        border-radius: 0.35rem;
        font-size: 0.85rem;
      }
    </style>
    """
).strip()


@dataclass(frozen=True)
class TrainingAnimationColumn:
    """Encapsulate the mac window column markup for the training animation."""

    html: str
    fallback_height: int = 720

def _sample_targets(rng: np.random.Generator, n: int):
    """Assign each point a class and a target cluster center."""
    # Pre-split counts by class
    n_spam = int(n * CLASS_PROBS["spam"])
    n_work = n - n_spam

    spam_clusters = [c for c in CLUSTERS if c[6] == "spam"]
    work_clusters = [c for c in CLUSTERS if c[6] == "work"]

    # Weighted uniform across clusters of the same class
    def pick_targets(k, pool):
        idx = rng.integers(0, len(pool), size=k)
        targets = [pool[i] for i in idx]
        # Add small jitter so points fill the ellipse
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
    # cubic ease-out-ish
    a = t
    eased = 1 - pow(1 - a, 3)
    xi = x0 + (x1 - x0) * eased
    yi = y0 + (y1 - y0) * eased
    return xi, yi

def build_training_animation_figure(*, seed: Optional[int] = None) -> go.Figure:
    """Return the Plotly figure used in the training animation."""

    if go is None:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "Plotly is required to build the training animation. Install 'plotly' to enable "
            "the interactive visualization."
        )

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
        t = (epoch / EPOCHS) ** (1.0 - EASING)  # control convergence speed
        xi, yi = _interpolate(x0, y0, tx, ty, t)

        frames.append(
            go.Frame(
                name=f"Epoch {epoch}",
                data=[go.Scatter(
                    x=xi, y=yi,
                    mode="markers",
                    marker=dict(size=6, opacity=0.9, line=dict(width=0), color=colors),
                    hoverinfo="skip",
                    showlegend=False,
                )]
            )
        )

    # Base scatter (epoch 0)
    base_scatter = frames[0].data[0]

    # Shapes: rounded rectangles for meaning areas (fade in by setting initial low opacity)
    shapes = []
    annotations = []
    for (name, cx, cy, rx, ry, fill, _cls) in CLUSTERS:
        shapes.append(dict(
            type="rect",
            x0=cx - rx, x1=cx + rx, y0=cy - ry, y1=cy + ry,
            line=dict(color="rgba(60,60,60,0.35)", width=2),
            fillcolor=fill,
            opacity=0.15,  # subtle
            layer="below"
        ))
        annotations.append(dict(
            x=cx, y=cy + ry - 0.05, text=name,
            xanchor="center", yanchor="bottom",
            showarrow=False, font=dict(size=13, color="rgba(20,20,20,0.85)")
        ))

    # Decision boundary (vertical dashed line near x=0)
    shapes.append(dict(
        type="line", x0=0, x1=0, y0=MAP_Y[0], y1=MAP_Y[1],
        line=dict(color="rgba(70,70,70,0.4)", width=2, dash="dash")
    ))

    fig = go.Figure(
        data=[base_scatter],
        layout=go.Layout(
            xaxis=dict(range=[MAP_X[0], MAP_X[1]], title="Meaning dimension 1", zeroline=False),
            yaxis=dict(range=[MAP_Y[0], MAP_Y[1]], title="Meaning dimension 2", zeroline=False, scaleanchor=None),
            margin=dict(l=40, r=20, b=40, t=40),
            shapes=shapes,
            annotations=annotations + [
                dict(x=0.02, y=0.98, xref="paper", yref="paper",
                     text="miniLM training: points move toward meaning areas",
                     xanchor="left", yanchor="top", showarrow=False,
                     font=dict(size=12, color="rgba(40,40,40,0.8)"))
            ],
            updatemenus=[dict(
                type="buttons",
                direction="left",
                x=0.0, y=1.12, xanchor="left", yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"fromcurrent": True, "frame": {"duration": 350, "redraw": True},
                                      "transition": {"duration": 250}}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Epoch: "},
                steps=[dict(method="animate",
                            args=[[f"Epoch {k}"], {"mode": "immediate",
                                                   "frame": {"duration": 0, "redraw": True},
                                                   "transition": {"duration": 0}}],
                            label=str(k)) for k in range(EPOCHS + 1)]
            )],
        ),
        frames=frames
    )

    # Legend proxy (two invisible traces) – keeps the main scatter light
    for cls_name, color in CLASS_TO_COLOR.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color=color),
            name="Spam-like" if cls_name == "spam" else "Work-like"
        ))

    return fig


def build_training_animation_column(
    *, seed: Optional[int] = None
) -> TrainingAnimationColumn:
    body_html = textwrap.dedent(
        """
        <div class="train-animation__body">
          <h4 class="train-animation__title">How miniLM learns a meaning space</h4>
          <p class="train-animation__caption">
            Watch the embedding points move into spam-like and work-like regions as epochs progress.
          </p>
        {content}
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


def render_training_animation():
    st.subheader("How miniLM learns a meaning space")
    if go is None:
        st.info("Install the optional 'plotly' dependency to view the interactive training animation.")
        return

    fig = build_training_animation_figure()

    st.plotly_chart(fig, use_container_width=True)
