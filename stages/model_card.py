from __future__ import annotations

import html
import json
from typing import Callable, ContextManager, Optional

import streamlit as st
from sklearn.metrics import accuracy_score
from streamlit.delta_generator import DeltaGenerator

from demistifai.constants import AUTONOMY_LEVELS
from demistifai.core.downloads import download_text
from demistifai.core.nav import render_stage_top_grid
from demistifai.dataset import DEFAULT_DATASET_CONFIG


GuidancePopover = Callable[[str, str], None]
SectionSurface = Callable[[Optional[str]], ContextManager[DeltaGenerator]]


def render_model_card_stage(
    *,
    section_surface: SectionSurface,
    guidance_popover: GuidancePopover,
) -> None:
    """Render the model card transparency stage."""

    ss = st.session_state

    render_stage_top_grid("model_card")

    with section_surface():
        st.subheader("Model Card — transparency")
        guidance_popover(
            "Transparency",
            """
Model cards summarize intended purpose, data, metrics, autonomy & adaptiveness settings.
They help teams reason about risks and the appropriate oversight controls.
""",
        )
        algo = "Sentence embeddings (MiniLM) + standardized numeric cues + Logistic Regression"
        n_samples = len(ss["labeled"])
        labels_present = sorted({row["label"] for row in ss["labeled"]}) if ss["labeled"] else []
        metrics_text = ""
        holdout_n = 0
        if ss.get("model") and ss.get("split_cache"):
            _, X_te_t, _, X_te_b, _, y_te = ss["split_cache"]
            y_pred = ss["model"].predict(X_te_t, X_te_b)
            holdout_n = len(y_te)
            metrics_text = f"Accuracy on hold‑out: {accuracy_score(y_te, y_pred):.2%} (n={holdout_n})"
        snapshot_id = ss.get("active_dataset_snapshot")
        snapshot_entry = None
        if snapshot_id:
            snapshot_entry = next(
                (
                    snap
                    for snap in ss.get("datasets", [])
                    if snap.get("id") == snapshot_id
                ),
                None,
            )
        dataset_config_for_card = (snapshot_entry or {}).get(
            "config", ss.get("dataset_config", DEFAULT_DATASET_CONFIG)
        )
        dataset_config_json = json.dumps(dataset_config_for_card, indent=2, sort_keys=True)
        snapshot_label = snapshot_id if snapshot_id else "— (save one in Prepare Data)"

        card_md = f"""
# Model Card — demistifAI (Spam Detector)
**Intended purpose**: Educational demo to illustrate the AI Act definition of an **AI system** via a spam classifier.

**Algorithm**: {algo}
**Features**: Sentence embeddings (MiniLM) concatenated with small, interpretable numeric features:
- num_links_external, has_suspicious_tld, punct_burst_ratio, money_symbol_count, urgency_terms_count.
These are standardized and combined with the embedding before a linear classifier.

**Classes**: spam, safe
**Dataset size**: {n_samples} labeled examples
**Classes present**: {', '.join(labels_present) if labels_present else '[not trained]'}

**Key metrics**: {metrics_text or 'Train a model to populate metrics.'}

**Autonomy**: {ss['autonomy']} (threshold={ss['threshold']:.2f})
**Adaptiveness**: {'Enabled' if ss['adaptive'] else 'Disabled'} (learn from user corrections).

**Data**: user-augmented seed set (title + body); session-only.
**Dataset snapshot ID**: {snapshot_label}
**Dataset config**:
```
{dataset_config_json}
```
**Known limitations**: tiny datasets; vocabulary sensitivity; no MIME/URL/metadata features.

**AI Act mapping**
- **Machine-based system**: Streamlit app (software) running on cloud runtime (hardware).
- **Inference**: model learns patterns from labeled examples.
- **Output generation**: predictions + confidence; used to recommend/route emails.
    - **Varying autonomy**: user selects autonomy level; at high autonomy, the system acts.
- **Adaptiveness**: optional feedback loop that updates the model.
"""
        content_col, highlight_col = st.columns([3, 2], gap="large")
        with content_col:
            st.markdown(card_md)
            download_text(card_md, "model_card.md", "Download model_card.md")
        with highlight_col:
            st.markdown(
                """
                <div class="info-metric-grid">
                    <div class="info-metric-card">
                        <div class="label">Labeled dataset</div>
                        <div class="value">{samples}</div>
                    </div>
                    <div class="info-metric-card">
                        <div class="label">Hold-out size</div>
                        <div class="value">{holdout}</div>
                    </div>
                    <div class="info-metric-card">
                        <div class="label">Autonomy</div>
                        <div class="value">{autonomy}</div>
                    </div>
                    <div class="info-metric-card">
                        <div class="label">Adaptiveness</div>
                        <div class="value">{adaptive}</div>
                    </div>
                </div>
                """.format(
                    samples=n_samples,
                    holdout=holdout_n or "—",
                    autonomy=html.escape(ss.get("autonomy", AUTONOMY_LEVELS[0])),
                    adaptive="On" if ss.get("adaptive") else "Off",
                ),
                unsafe_allow_html=True,
            )

        with highlight_col:
            st.markdown("#### Dataset provenance")
            if snapshot_id:
                st.write(f"Snapshot ID: `{snapshot_id}`")
            else:
                st.write("Snapshot ID: — (save one in Prepare Data → Snapshot & provenance).")
            st.code(dataset_config_json, language="json")
