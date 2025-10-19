from __future__ import annotations

import html
from collections import Counter
from datetime import datetime
from typing import Callable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from sklearn import __version__ as sklearn_version
from sklearn.metrics import precision_recall_fscore_support

from demistifai.ui.components import render_guardrail_panel, render_stage_top_grid
from demistifai.constants import STAGE_BY_KEY
from demistifai.core.guardrails import (
    GUARDRAIL_LABEL_ICONS,
    _guardrail_badges_html,
    _guardrail_signals,
)
from demistifai.core.state import ensure_state, hash_dict, validate_invariants
from demistifai.modeling import (
    _fmt_delta,
    _fmt_pct,
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
        st.markdown("### Spam threshold")
        presets = threshold_presets(y_true01, p_spam)

        if "eval_temp_threshold" not in ss:
            ss["eval_temp_threshold"] = current_thr

        controls_col, slider_col = st.columns([2, 3], gap="large")
        with controls_col:
            if st.button("Balanced (max F1)", use_container_width=True):
                ss["eval_temp_threshold"] = float(presets["balanced_f1"])
                st.toast(f"Suggested threshold (max F1): {ss['eval_temp_threshold']:.2f}", icon="✅")
            if st.button("Protect inbox (≥95% precision)", use_container_width=True):
                ss["eval_temp_threshold"] = float(presets["precision_95"])
                st.toast(
                    f"Suggested threshold (precision≥95%): {ss['eval_temp_threshold']:.2f}",
                    icon="✅",
                )
            if st.button("Catch spam (≥90% recall)", use_container_width=True):
                ss["eval_temp_threshold"] = float(presets["recall_90"])
                st.toast(
                    f"Suggested threshold (recall≥90%): {ss['eval_temp_threshold']:.2f}",
                    icon="✅",
                )
            if st.button("Adopt this threshold", use_container_width=True):
                ss["threshold"] = float(ss.get("eval_temp_threshold", current_thr))
                st.success(
                    f"Adopted new operating threshold: **{ss['threshold']:.2f}**. This will be used in Classify and Full Autonomy."
                )
        with slider_col:
            temp_threshold = float(
                st.slider(
                    "Adjust threshold (temporary)",
                    0.1,
                    0.9,
                    value=float(ss.get("eval_temp_threshold", current_thr)),
                    step=0.01,
                    key="eval_temp_threshold",
                    help="Lower values catch more spam (higher recall) but risk more false alarms. Higher values protect the inbox (higher precision) but may miss some spam.",
                )
            )

            cm_temp = compute_confusion(y_true01, p_spam, temp_threshold)
            acc_temp = (cm_temp["TP"] + cm_temp["TN"]) / max(1, len(y_true01))
            st.caption(
                f"At {temp_threshold:.2f}, accuracy would be **{acc_temp:.2%}** (TP {cm_temp['TP']}, FP {cm_temp['FP']}, TN {cm_temp['TN']}, FN {cm_temp['FN']})."
            )

        acc_new, p_new, r_new, f1_new, cm_new = _pr_acc_cm(y_true01, p_spam, temp_threshold)

        with st.container(border=True):
            st.markdown("#### What changes when I move the threshold?")
            st.caption("Comparing your **adopted** threshold vs. the **temporary** slider value above:")

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**Current (adopted)**")
                st.write(f"- Threshold: **{current_thr:.2f}**")
                st.write(f"- Accuracy: {_fmt_pct(acc_cur)}")
                st.write(f"- Precision (spam): {_fmt_pct(p_cur)}")
                st.write(f"- Recall (spam): {_fmt_pct(r_cur)}")
                st.write(f"- False positives (safe→spam): **{cm_cur['FP']}**")
                st.write(f"- False negatives (spam→safe): **{cm_cur['FN']}**")

            with col_right:
                st.markdown("**If you adopt the slider value**")
                st.write(f"- Threshold: **{temp_threshold:.2f}**")
                st.write(f"- Accuracy: {_fmt_pct(acc_new)} ({_fmt_delta(acc_new, acc_cur)})")
                st.write(f"- Precision (spam): {_fmt_pct(p_new)} ({_fmt_delta(p_new, p_cur)})")
                st.write(
                    f"- False positives: **{cm_new['FP']}** ({_fmt_delta(cm_new['FP'], cm_cur['FP'], pct=False)})"
                )
                st.write(
                    f"- False negatives: **{cm_new['FN']}** ({_fmt_delta(cm_new['FN'], cm_cur['FN'], pct=False)})"
                )

            if temp_threshold > current_thr:
                st.info(
                    "Raising the threshold makes the model **more cautious**: usually **fewer false positives** (protects inbox) but **more spam may slip through**."
                )
            elif temp_threshold < current_thr:
                st.info(
                    "Lowering the threshold makes the model **more aggressive**: it **catches more spam** (higher recall) but may **flag more legit emails**."
                )
            else:
                st.info("Same threshold as adopted — metrics unchanged.")

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

        guardrail_cards: list[dict[str, str]] = []
        guardrail_counts: Counter[str] = Counter()

        if len(p_spam) == 0:
            st.caption("Evaluation set empty — guardrail audit unavailable.")
        else:
            margin_window = 0.15
            max_cards = 8
            sorted_indices = sorted(
                range(len(p_spam)),
                key=lambda i: abs(float(p_spam[i]) - current_thr),
            )

            for idx in sorted_indices:
                if len(guardrail_cards) >= max_cards:
                    break

                prob_val = float(p_spam[idx])
                margin = abs(prob_val - current_thr)
                if margin > margin_window:
                    continue

                subject_raw = ""
                body_raw = ""
                if X_te_t is not None and X_te_b is not None:
                    if idx < len(X_te_t):
                        subject_raw = X_te_t[idx]
                    if idx < len(X_te_b):
                        body_raw = X_te_b[idx]
                else:
                    text_val = texts_test[idx] if idx < len(texts_test) else ""
                    text_str = text_val if isinstance(text_val, str) else str(text_val or "")
                    parts = text_str.split("\n", 1)
                    subject_raw = parts[0] if parts else ""
                    body_raw = parts[1] if len(parts) > 1 else ""

                subject_raw = str(subject_raw or "").strip()
                body_raw = str(body_raw or "").strip()

                signals = _guardrail_signals(subject_raw, body_raw)
                active_keys = [key for key, flag in signals.items() if flag]
                if not active_keys:
                    continue

                guardrail_counts.update(active_keys)

                subject_display = html.escape(
                    shorten_text(subject_raw or "(no subject)", limit=100)
                )
                pred_label = "spam" if prob_val >= current_thr else "safe"
                icon = GUARDRAIL_LABEL_ICONS.get(pred_label, "✉️")

                try:
                    true_label_raw = y_te[idx]
                except Exception:
                    true_label_raw = ""
                true_label_clean = str(true_label_raw or "").strip().title()

                meta_left = html.escape(
                    f"P(spam) {prob_val:.2f} • Δ {prob_val - current_thr:+.2f}"
                )
                meta_right_parts = [f"{icon} {pred_label.title()}"]
                if true_label_clean:
                    meta_right_parts.append(f"True: {true_label_clean}")
                meta_right = html.escape(" • ".join(meta_right_parts))

                excerpt = (
                    shorten_text(body_raw or "", limit=220)
                    .replace("\n", " ")
                    .strip()
                )
                excerpt_html = html.escape(excerpt) if excerpt else "No body text."
                badges_html = _guardrail_badges_html(signals)
                body_html = "".join(
                    [
                        badges_html,
                        f"<div style=\"margin-top:0.4rem; color: rgba(55, 65, 81, 0.85);\">{excerpt_html}</div>",
                    ]
                )

                guardrail_cards.append(
                    {
                        "subject": subject_display,
                        "meta_left": meta_left,
                        "meta_right": meta_right,
                        "body": body_html,
                    }
                )

            if not guardrail_cards:
                st.caption(
                    "No borderline emails triggered guardrail signals in the test set."
                )
            else:
                chart_obj = None
                counts_sorted = guardrail_counts.most_common()
                if counts_sorted:
                    guardrail_df = pd.DataFrame(
                        counts_sorted, columns=["guardrail", "count"]
                    )
                    guardrail_chart = (
                        alt.Chart(guardrail_df)
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                        .encode(
                            x=alt.X("count:Q", title="Emails flagged"),
                            y=alt.Y(
                                "guardrail:N",
                                sort="-x",
                                title="Signal",
                            ),
                            color=alt.Color("guardrail:N", legend=None),
                            tooltip=[
                                alt.Tooltip("guardrail:N", title="Signal"),
                                alt.Tooltip("count:Q", title="Emails"),
                            ],
                        )
                        .properties(height=220)
                    )
                    chart_obj = st.altair_chart(
                        guardrail_chart, use_container_width=True
                    )

                render_guardrail_panel(chart=chart_obj, cards=guardrail_cards)

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
