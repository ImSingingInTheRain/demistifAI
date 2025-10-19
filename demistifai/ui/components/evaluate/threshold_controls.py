from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any, Dict, Tuple

import numpy as np
import streamlit as st

from demistifai.modeling import _fmt_delta, _fmt_pct, compute_confusion


ThresholdPresetsFn = Callable[[Sequence[int], Sequence[float]], Mapping[str, float]]
PrAccCmFn = Callable[[Sequence[int], Sequence[float], float], Tuple[float, float, float, float, Dict[str, int]]]


def render_threshold_controls(
    *,
    session_state: MutableMapping[str, Any],
    y_true01: Sequence[int] | np.ndarray,
    p_spam: Sequence[float] | np.ndarray,
    current_threshold: float,
    threshold_presets_fn: ThresholdPresetsFn,
    pr_acc_cm_fn: PrAccCmFn,
    current_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Render the spam threshold controls and return updated metrics.

    Parameters
    ----------
    session_state:
        Streamlit session state mapping for persisting slider interactions.
    y_true01:
        Ground-truth spam labels encoded as 0/1.
    p_spam:
        Model probabilities for the spam class.
    current_threshold:
        The currently adopted operating threshold.
    threshold_presets_fn:
        Callable that returns suggested thresholds for the provided data.
    pr_acc_cm_fn:
        Callable that computes accuracy/precision/recall/F1 and confusion matrix
        for a supplied threshold.
    current_metrics:
        Mapping containing metrics for the adopted threshold. Expected keys:
        ``accuracy``, ``precision``, ``recall``, ``f1``, and ``confusion``.

    Returns
    -------
    dict
        Dictionary containing the adopted threshold, temporary slider value,
        and metrics/confusion matrix at the slider value.
    """

    presets = threshold_presets_fn(y_true01, p_spam)

    if "eval_temp_threshold" not in session_state:
        session_state["eval_temp_threshold"] = float(current_threshold)

    st.markdown("### Spam threshold")

    controls_col, slider_col = st.columns([2, 3], gap="large")

    with controls_col:
        if st.button("Balanced (max F1)", use_container_width=True):
            session_state["eval_temp_threshold"] = float(presets["balanced_f1"])
            st.toast(
                f"Suggested threshold (max F1): {session_state['eval_temp_threshold']:.2f}",
                icon="✅",
            )

        if st.button("Protect inbox (≥95% precision)", use_container_width=True):
            session_state["eval_temp_threshold"] = float(presets["precision_95"])
            st.toast(
                f"Suggested threshold (precision≥95%): {session_state['eval_temp_threshold']:.2f}",
                icon="✅",
            )

        if st.button("Catch spam (≥90% recall)", use_container_width=True):
            session_state["eval_temp_threshold"] = float(presets["recall_90"])
            st.toast(
                f"Suggested threshold (recall≥90%): {session_state['eval_temp_threshold']:.2f}",
                icon="✅",
            )

        if st.button("Adopt this threshold", use_container_width=True):
            session_state["threshold"] = float(
                session_state.get("eval_temp_threshold", current_threshold)
            )
            st.success(
                "Adopted new operating threshold: **{threshold:.2f}**. This will be used in "
                "Classify and Full Autonomy.".format(threshold=session_state["threshold"])
            )

    with slider_col:
        temp_threshold = float(
            st.slider(
                "Adjust threshold (temporary)",
                0.1,
                0.9,
                value=float(session_state.get("eval_temp_threshold", current_threshold)),
                step=0.01,
                key="eval_temp_threshold",
                help=(
                    "Lower values catch more spam (higher recall) but risk more false alarms. "
                    "Higher values protect the inbox (higher precision) but may miss some spam."
                ),
            )
        )

        cm_temp = compute_confusion(y_true01, p_spam, temp_threshold)
        acc_temp = (cm_temp["TP"] + cm_temp["TN"]) / max(1, len(y_true01))
        st.caption(
            f"At {temp_threshold:.2f}, accuracy would be **{acc_temp:.2%}** "
            f"(TP {cm_temp['TP']}, FP {cm_temp['FP']}, TN {cm_temp['TN']}, FN {cm_temp['FN']})."
        )

    acc_cur = float(current_metrics.get("accuracy", 0.0))
    p_cur = float(current_metrics.get("precision", 0.0))
    r_cur = float(current_metrics.get("recall", 0.0))
    cm_cur = dict(current_metrics.get("confusion", {}))

    acc_new, p_new, r_new, f1_new, cm_new = pr_acc_cm_fn(y_true01, p_spam, temp_threshold)

    with st.container(border=True):
        st.markdown("#### What changes when I move the threshold?")
        st.caption(
            "Comparing your **adopted** threshold vs. the **temporary** slider value above:"
        )

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Current (adopted)**")
            st.write(f"- Threshold: **{current_threshold:.2f}**")
            st.write(f"- Accuracy: {_fmt_pct(acc_cur)}")
            st.write(f"- Precision (spam): {_fmt_pct(p_cur)}")
            st.write(f"- Recall (spam): {_fmt_pct(r_cur)}")
            st.write(f"- False positives (safe→spam): **{cm_cur.get('FP', 0)}**")
            st.write(f"- False negatives (spam→safe): **{cm_cur.get('FN', 0)}**")

        with col_right:
            st.markdown("**If you adopt the slider value**")
            st.write(f"- Threshold: **{temp_threshold:.2f}**")
            st.write(
                f"- Accuracy: {_fmt_pct(acc_new)} ({_fmt_delta(acc_new, acc_cur)})"
            )
            st.write(
                f"- Precision (spam): {_fmt_pct(p_new)} ({_fmt_delta(p_new, p_cur)})"
            )
            st.write(
                f"- Recall (spam): {_fmt_pct(r_new)} ({_fmt_delta(r_new, r_cur)})"
            )
            st.write(
                f"- False positives: **{cm_new['FP']}** ({_fmt_delta(cm_new['FP'], cm_cur.get('FP', 0), pct=False)})"
            )
            st.write(
                f"- False negatives: **{cm_new['FN']}** ({_fmt_delta(cm_new['FN'], cm_cur.get('FN', 0), pct=False)})"
            )

        if temp_threshold > current_threshold:
            st.info(
                "Raising the threshold makes the model **more cautious**: usually **fewer false "
                "positives** (protects inbox) but **more spam may slip through**."
            )
        elif temp_threshold < current_threshold:
            st.info(
                "Lowering the threshold makes the model **more aggressive**: it **catches more "
                "spam** (higher recall) but may **flag more legit emails**."
            )
        else:
            st.info("Same threshold as adopted — metrics unchanged.")

    return {
        "adopted_threshold": float(session_state.get("threshold", current_threshold)),
        "temporary_threshold": temp_threshold,
        "temporary_metrics": {
            "accuracy": acc_new,
            "precision": p_new,
            "recall": r_new,
            "f1": f1_new,
            "confusion": cm_new,
        },
    }
