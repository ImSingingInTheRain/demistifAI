from __future__ import annotations

from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from sklearn import __version__ as sklearn_version
from sklearn.metrics import precision_recall_fscore_support

from demistifai.ui.components import (
    render_guardrail_audit,
    render_stage_top_grid,
    render_threshold_controls,
)
from demistifai.constants import STAGE_BY_KEY
from demistifai.core.state import ensure_state, hash_dict, validate_invariants
from demistifai.modeling import (
    _pr_acc_cm,
    _y01,
    compute_confusion,
    make_after_eval_story,
    plot_threshold_curves,
    threshold_presets,
    verdict_label,
)
from demistifai.ui.components.terminal.evaluate import render_evaluate_terminal


SectionSurface = Callable[..., object]
RenderNerdToggle = Callable[..., bool]
ShortenText = Callable[[str, int], str]


def render_evaluate_stage_page(
    *,
    section_surface: SectionSurface,
    render_nerd_mode_toggle: RenderNerdToggle,
    shorten_text: ShortenText,
) -> None:
    """Render the Evaluate stage UI."""

    s = ensure_state()
    validate_invariants(s)
    ss = st.session_state

    stage = STAGE_BY_KEY["evaluate"]

    def _render_evaluate_terminal(slot: DeltaGenerator) -> None:
        with slot:
            render_evaluate_terminal()

    render_stage_top_grid("evaluate", left_renderer=_render_evaluate_terminal)

    model_state = s.setdefault("model", {})
    data_state = s.setdefault("data", {})
    split_state = s.setdefault("split", {})

    if model_state.get("status") != "trained":
        with section_surface():
            st.subheader(f"{stage.icon} {stage.title} — How well does your spam detector perform?")
            st.info("Model not trained or stale — please (re)train.")
        return

    model_obj = model_state.get("clf")
    if model_obj is not None and ss.get("model") is None:
        ss["model"] = model_obj

    if not ss.get("split_cache"):
        x_train_payload = split_state.get("X_train") or {}
        x_test_payload = split_state.get("X_test") or {}
        y_train_payload = split_state.get("y_train") or []
        y_test_payload = split_state.get("y_test") or []
        if any(
            [
                x_train_payload,
                x_test_payload,
                y_train_payload,
                y_test_payload,
            ]
        ):
            ss["split_cache"] = (
                list(x_train_payload.get("titles") or []),
                list(x_test_payload.get("titles") or []),
                list(x_train_payload.get("bodies") or []),
                list(x_test_payload.get("bodies") or []),
                list(y_train_payload or []),
                list(y_test_payload or []),
            )

    if not (ss.get("model") and ss.get("split_cache")):
        with section_surface():
            st.subheader(f"{stage.icon} {stage.title} — How well does your spam detector perform?")
            st.info("Train a model first in the **Train** tab.")
        return

    cache = ss["split_cache"]
    if len(cache) == 4:
        X_tr, X_te, y_tr, y_te_raw = cache
        texts_test = X_te
        X_te_t = X_te_b = None
    else:
        X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te_raw = cache
        texts_test = [(t or "") + "\n" + (b or "") for t, b in zip(X_te_t, X_te_b)]

    y_test_labels = list(y_te_raw)

    metrics_state = model_state.get("metrics")
    if not isinstance(metrics_state, dict):
        metrics_state = {}
        model_state["metrics"] = metrics_state

    model_params = dict(model_state.get("params") or {})
    data_hash = data_state.get("hash", "") or ""
    lib_versions = {"numpy": np.__version__, "sklearn": sklearn_version}
    metrics_key = hash_dict(
        {
            "data_hash": data_hash,
            "model_params": model_params,
            "lib_versions": lib_versions,
        }
    )

    cached_p_spam = metrics_state.get("p_spam")
    cached_y_true = metrics_state.get("y_true")
    metrics_valid = (
        metrics_state.get("cache_key") == metrics_key
        and cached_p_spam is not None
        and cached_y_true is not None
    )

    if not metrics_valid:
        model_for_eval = ss["model"]
        try:
            if len(cache) == 6:
                probs = model_for_eval.predict_proba(X_te_t, X_te_b)
            else:
                probs = model_for_eval.predict_proba(texts_test)
        except TypeError:
            probs = model_for_eval.predict_proba(texts_test)

        classes = list(getattr(model_for_eval, "classes_", []))
        if classes and "spam" in classes:
            idx_spam = classes.index("spam")
        else:
            idx_spam = 1 if probs.shape[1] > 1 else 0
        p_spam_arr = probs[:, idx_spam]

        metrics_state = {
            "cache_key": metrics_key,
            "computed_at": datetime.now().isoformat(timespec="seconds"),
            "p_spam": p_spam_arr.tolist(),
            "y_true": list(y_test_labels),
            "classes": classes,
            "data_hash": data_hash,
            "model_params": model_params,
            "lib_versions": lib_versions,
        }
        model_state["metrics"] = metrics_state
    else:
        classes = list(metrics_state.get("classes", []))

    p_spam = np.asarray(metrics_state.get("p_spam", []), dtype=float)
    y_te = list(metrics_state.get("y_true", y_test_labels))
    y_true01 = _y01(list(y_te))

    current_thr = float(ss.get("threshold", 0.5))
    cm = compute_confusion(y_true01, p_spam, current_thr)
    acc = (cm["TP"] + cm["TN"]) / max(1, len(y_true01))
    emoji, verdict = verdict_label(acc, len(y_true01))
    prev_eval = ss.get("last_eval_results") or {}
    acc_cur, p_cur, r_cur, f1_cur, cm_cur = _pr_acc_cm(y_true01, p_spam, current_thr)

    with section_surface():
        narrative_col, metrics_col = st.columns([3, 2], gap="large")
        with narrative_col:
            st.subheader(f"{stage.icon} {stage.title} — How well does your spam detector perform?")
            st.write(
                "Now that your model has learned from examples, it’s time to test how well it works. "
                "During training, we kept some emails aside — the **test set**. The model hasn’t seen these before. "
                "By checking its guesses against the true labels, we get a fair measure of performance."
            )
            st.markdown("### What do these results say?")
            st.markdown(make_after_eval_story(len(y_true01), cm))
        with metrics_col:
            st.markdown("### Snapshot")
            st.success(f"**Accuracy:** {acc:.2%}  |  {emoji} {verdict}")
            st.caption(f"Evaluated on {len(y_true01)} unseen emails at threshold {current_thr:.2f}.")
            st.markdown(
                "- ✅ Spam caught: **{tp}**\n"
                "- ❌ Spam missed: **{fn}**\n"
                "- ⚠️ Safe mis-flagged: **{fp}**\n"
                "- ✅ Safe passed: **{tn}**"
                .format(tp=cm["TP"], fn=cm["FN"], fp=cm["FP"], tn=cm["TN"]))
            dataset_story = ss.get("last_dataset_delta_story")
            metric_deltas: list[str] = []
            if prev_eval:
                metric_deltas.append(f"Δaccuracy {acc_cur - prev_eval.get('accuracy', acc_cur):+.2%}")
                metric_deltas.append(f"Δprecision {p_cur - prev_eval.get('precision', p_cur):+.2%}")
                metric_deltas.append(f"Δrecall {r_cur - prev_eval.get('recall', r_cur):+.2%}")
            extra_caption = " | ".join(
                part
                for part in [dataset_story, " · ".join(metric_deltas) if metric_deltas else ""]
                if part
            )
            if extra_caption:
                st.caption(f"📂 {extra_caption}")

    with section_surface():
        threshold_result = render_threshold_controls(
            session_state=ss,
            y_true01=y_true01,
            p_spam=p_spam,
            current_threshold=current_thr,
            threshold_presets_fn=threshold_presets,
            pr_acc_cm_fn=_pr_acc_cm,
            current_metrics={
                "accuracy": acc_cur,
                "precision": p_cur,
                "recall": r_cur,
                "f1": f1_cur,
                "confusion": cm_cur,
            },
        )
        temp_threshold = threshold_result["temporary_threshold"]

    with section_surface():
        with st.expander("📌 Suggestions to improve your model"):
            st.markdown(
                """
  - Add more labeled emails, especially tricky edge cases
  - Balance the dataset between spam and safe
  - Use diverse wording in your examples
  - Tune the spam threshold for your needs
  - Review the confusion matrix to spot mistakes
  - Ensure emails have enough meaningful content
  """
            )

    with section_surface():
        st.markdown("### Guardrail audit (borderline emails)")
        render_guardrail_audit(
            p_spam=p_spam,
            y_true=y_te,
            current_threshold=current_thr,
            shorten_text=shorten_text,
            subjects=X_te_t if X_te_t is not None else None,
            bodies=X_te_b if X_te_b is not None else None,
            combined_texts=texts_test,
        )

    ss["last_eval_results"] = {
        "accuracy": acc_cur,
        "precision": p_cur,
        "recall": r_cur,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    nerd_mode_eval_enabled = render_nerd_mode_toggle(
        key="nerd_mode_eval",
        title="Nerd Mode — technical details",
        description="Inspect precision/recall tables, interpretability cues, and governance notes.",
        icon="🔬",
    )

    if nerd_mode_eval_enabled:
        with section_surface():
            temp_threshold = float(ss.get("eval_temp_threshold", current_thr))
            y_hat_temp = (p_spam >= temp_threshold).astype(int)
            prec_spam, rec_spam, f1_spam, sup_spam = precision_recall_fscore_support(
                y_true01, y_hat_temp, average="binary", zero_division=0
            )
            y_true_safe = 1 - y_true01
            y_hat_safe = 1 - y_hat_temp
            prec_safe, rec_safe, f1_safe, sup_safe = precision_recall_fscore_support(
                y_true_safe, y_hat_safe, average="binary", zero_division=0
            )

            st.markdown("### Detailed metrics (at current threshold)")

            def _as_int(value, fallback):
                if value is None:
                    return int(fallback)
                try:
                    return int(value)
                except TypeError:
                    return int(fallback)

            spam_support = _as_int(sup_spam, np.sum(y_true01))
            safe_support = _as_int(sup_safe, np.sum(1 - y_true01))

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "class": "spam",
                            "precision": prec_spam,
                            "recall": rec_spam,
                            "f1": f1_spam,
                            "support": spam_support,
                        },
                        {
                            "class": "safe",
                            "precision": prec_safe,
                            "recall": rec_safe,
                            "f1": f1_safe,
                            "support": safe_support,
                        },
                    ]
                ).round(3),
                width="stretch",
                hide_index=True,
            )

            st.markdown("### Precision & Recall vs Threshold (validation)")
            fig = plot_threshold_curves(y_true01, p_spam)
            st.pyplot(fig)

            st.markdown("### Interpretability")
            try:
                if hasattr(ss["model"], "named_steps"):
                    clf = ss["model"].named_steps.get("clf")
                    vec = ss["model"].named_steps.get("tfidf")
                    if hasattr(clf, "coef_") and vec is not None:
                        vocab = np.array(vec.get_feature_names_out())
                        coefs = clf.coef_[0]
                        top_spam = vocab[np.argsort(coefs)[-10:]][::-1]
                        top_safe = vocab[np.argsort(coefs)[:10]]
                        col_i1, col_i2 = st.columns(2)
                        with col_i1:
                            st.write("Top signals → **Spam**")
                            st.write(", ".join(top_spam))
                        with col_i2:
                            st.write("Top signals → **Safe**")
                            st.write(", ".join(top_safe))
                    else:
                        st.caption("Coefficients unavailable for this classifier.")
                elif hasattr(ss["model"], "numeric_feature_coefs"):
                    coef_map = ss["model"].numeric_feature_coefs()
                    st.caption("Numeric feature weights (positive → Spam, negative → Safe):")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "feature": k,
                                    "weight_toward_spam": v,
                                }
                                for k, v in coef_map.items()
                            ]
                        ).sort_values("weight_toward_spam", ascending=False),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.caption("Interpretability: no compatible inspector for this model.")
            except Exception as e:
                st.caption(f"Interpretability view unavailable: {e}")

            st.markdown("### Governance & reproducibility")
            try:
                if len(cache) == 4:
                    n_tr, n_te = len(y_tr), len(y_te)
                else:
                    n_tr, n_te = len(y_tr), len(y_te)
                split = ss.get("train_params", {}).get("test_size", "—")
                seed = ss.get("train_params", {}).get("random_state", "—")
                ts = ss.get("eval_timestamp", "—")
                st.write(f"- Train set: {n_tr}  |  Test set: {n_te}  |  Hold-out fraction: {split}")
                st.write(f"- Random seed: {seed}")
                st.write(f"- Training time: {ts}")
                st.write(f"- Adopted threshold: {ss.get('threshold', 0.5):.2f}")
            except Exception:
                st.caption("Governance info unavailable.")
