from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, MutableMapping, Optional

import numpy as np
import pandas as pd
import streamlit as st

from demistifai.core.guardrails import extract_candidate_spans
from demistifai.modeling import (
    FEATURE_DISPLAY_NAMES,
    PlattProbabilityCalibrator,
    _counts,
    _y01,
    cache_train_embeddings,
    combine_text,
    encode_texts,
    make_after_training_story,
    model_kind_string,
)
from demistifai.ui.primitives import shorten_text

from .visualizations import build_calibration_chart

logger = logging.getLogger(__name__)


def render_training_results(
    ss: MutableMapping[str, Any],
    *,
    has_embed: bool,
    has_langdetect: bool,
    render_language_mix_chip_rows,
    summarize_language_mix,
    parse_split_cache,
) -> None:
    """Render post-training evaluation, nerd panels, and calibration controls."""

    parsed_split = None
    y_tr_labels = None
    y_te_labels = None
    train_texts_combined_cache: list[str] = []
    test_texts_combined_cache: list[str] = []
    lang_mix_train: Optional[Dict[str, Any]] = None
    lang_mix_test: Optional[Dict[str, Any]] = None
    lang_mix_error: Optional[str] = None
    has_model = ss.get("model") is not None
    has_split_cache = ss.get("split_cache") is not None
    if has_model and has_split_cache:
        try:
            parsed_split = parse_split_cache(ss["split_cache"])
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr_labels, y_te_labels = parsed_split

            if X_tr_t is not None and X_tr_b is not None:
                train_texts_combined_cache = [
                    combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)
                ]
            if X_te_t is not None and X_te_b is not None:
                test_texts_combined_cache = [
                    combine_text(t, b) for t, b in zip(X_te_t, X_te_b)
                ]

            if has_langdetect:
                try:
                    lang_mix_train = summarize_language_mix(train_texts_combined_cache)
                    lang_mix_test = summarize_language_mix(test_texts_combined_cache)
                except Exception as exc:
                    lang_mix_error = str(exc) or exc.__class__.__name__
                    lang_mix_train = None
                    lang_mix_test = None
            else:
                lang_mix_error = "language detector unavailable"

            story = make_after_training_story(y_tr_labels, y_te_labels)
            st.markdown(story)
        except Exception as exc:
            st.caption(f"Training storyboard unavailable ({exc}).")
            logger.exception("Training storyboard failed")

    if ss.get("nerd_mode_train") and ss.get("model") is not None and parsed_split:
        with st.expander("Nerd Mode — what just happened (technical)", expanded=True):
            X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr_labels, y_te_labels = parsed_split
            train_texts_combined: list[str] = list(train_texts_combined_cache)
            train_embeddings: Optional[np.ndarray] = None
            train_embeddings_error: Optional[str] = None
            if not train_texts_combined and X_tr_t and X_tr_b:
                train_texts_combined = [combine_text(t, b) for t, b in zip(X_tr_t, X_tr_b)]
            if train_texts_combined:
                if has_embed:
                    try:
                        train_embeddings = cache_train_embeddings(train_texts_combined)
                        if getattr(train_embeddings, "size", 0) == 0:
                            train_embeddings = None
                    except Exception as exc:
                        train_embeddings = None
                        train_embeddings_error = str(exc) or exc.__class__.__name__
                    if train_embeddings is None:
                        try:
                            train_embeddings = encode_texts(train_texts_combined)
                            if getattr(train_embeddings, "size", 0) == 0:
                                train_embeddings = None
                        except Exception as exc:
                            train_embeddings = None
                            train_embeddings_error = str(exc) or exc.__class__.__name__
                        else:
                            train_embeddings_error = None
                else:
                    train_embeddings_error = "text encoder unavailable"
            try:
                st.markdown("**Data split**")
                st.markdown(
                    f"- Train set size: {len(y_tr_labels)}  \n"
                    f"- Test set size: {len(y_te_labels)}  \n"
                    f"- Class balance (train): {_counts(list(y_tr_labels))}  \n"
                    f"- Class balance (test): {_counts(list(y_te_labels))}"
                )
            except Exception:
                st.caption("Split details unavailable.")

            if lang_mix_error:
                st.caption(f"Language mix unavailable ({lang_mix_error}).")
            elif has_langdetect:
                try:
                    render_language_mix_chip_rows(lang_mix_train, lang_mix_test)
                except Exception as exc:
                    msg = str(exc) or exc.__class__.__name__
                    st.caption(f"Language mix unavailable ({msg}).")

            centroid_distance: Optional[float] = None
            centroid_message: Optional[str] = None
            try:
                if not train_texts_combined:
                    centroid_message = "Centroid distance unavailable (no training texts)."
                elif not has_embed:
                    centroid_message = "Centroid distance unavailable (text encoder unavailable)."
                elif train_embeddings is None or getattr(train_embeddings, "size", 0) == 0:
                    detail = train_embeddings_error or "embeddings missing"
                    centroid_message = f"Centroid distance unavailable ({detail})."
                elif not y_tr_labels:
                    centroid_message = "Centroid distance unavailable (labels missing)."
                else:
                    y_train_arr = np.asarray(y_tr_labels)
                    if train_embeddings.shape[0] != y_train_arr.shape[0]:
                        centroid_message = "Centroid distance unavailable (embedding count mismatch)."
                    else:
                        spam_mask = y_train_arr == "spam"
                        safe_mask = y_train_arr == "safe"
                        if not np.any(spam_mask) or not np.any(safe_mask):
                            centroid_message = "Centroid distance requires at least one spam and one safe email."
                        else:
                            spam_centroid = train_embeddings[spam_mask].mean(axis=0)
                            safe_centroid = train_embeddings[safe_mask].mean(axis=0)
                            spam_norm = float(np.linalg.norm(spam_centroid))
                            safe_norm = float(np.linalg.norm(safe_centroid))
                            if spam_norm == 0.0 or safe_norm == 0.0:
                                centroid_message = "Centroid distance unavailable (zero-length centroid)."
                            else:
                                cosine_similarity = float(
                                    np.clip(
                                        np.dot(spam_centroid, safe_centroid)
                                        / (spam_norm * safe_norm),
                                        -1.0,
                                        1.0,
                                    )
                                )
                                centroid_distance = 1.0 - cosine_similarity
            except Exception:
                centroid_message = "Centroid distance unavailable."

            if centroid_distance is not None:
                st.metric("Centroid cosine distance", f"{centroid_distance:.2f}")
                meter_width = float(np.clip(centroid_distance, 0.0, 1.0)) * 100.0
                meter_html = f"""
                <div style="margin-top:-0.5rem; margin-bottom:0.75rem;">
                    <div style="background:rgba(49, 51, 63, 0.1); border-radius:999px; height:10px; width:100%;">
                        <div style="background:linear-gradient(90deg, #4ade80, #22c55e); border-radius:999px; height:100%; width:{meter_width:.0f}%;"></div>
                    </div>
                    <div style="font-size:0.7rem; color:rgba(49,51,63,0.6); margin-top:0.25rem;">0 = identical • 1 = orthogonal</div>
                </div>
                """
                st.markdown(meter_html, unsafe_allow_html=True)
                st.caption(
                    "Bigger distance means spam and safe live farther apart in meaning space—good separation."
                )
            elif centroid_message:
                st.caption(centroid_message)

            st.markdown("#### Decision margin spread (text head)")
            margins: Optional[np.ndarray] = None
            margin_error = False
            model_obj = ss.get("model")
            if not train_texts_combined:
                st.caption("Margin distribution unavailable (no training texts).")
            else:
                logits: Optional[np.ndarray] = None
                try:
                    if hasattr(model_obj, "predict_logit"):
                        logits = np.asarray(model_obj.predict_logit(train_texts_combined), dtype=float)
                    if logits is None or logits.size == 0:
                        probs = model_obj.predict_proba(X_tr_t, X_tr_b)[:, getattr(model_obj, "_i_spam", 1)]
                        probs = np.clip(probs, 1e-6, 1 - 1e-6)
                        logits = np.log(probs / (1.0 - probs))
                    logits = np.asarray(logits, dtype=float).reshape(-1)
                    logits = logits[np.isfinite(logits)]
                    if logits.size > 0:
                        margins = np.abs(logits)
                except Exception as exc:
                    st.caption(f"Margin distribution unavailable: {exc}")
                    margins = None
                    margin_error = True

            if margins is not None and margins.size > 0:
                try:
                    bins = min(12, max(5, int(np.ceil(np.log2(margins.size + 1)))))
                    counts, edges = np.histogram(margins, bins=bins)
                    labels = [
                        f"{edges[i]:.2f}–{edges[i + 1]:.2f}" for i in range(len(edges) - 1)
                    ]
                    hist_df = pd.DataFrame({"margin": labels, "count": counts})
                    st.bar_chart(hist_df.set_index("margin"), width="stretch")
                    st.caption(
                        "Higher margins = clearer decisions; lots of small margins means many borderline emails."
                    )
                except Exception as exc:
                    st.caption(f"Could not render margin histogram: {exc}")
            elif train_texts_combined and not margin_error:
                st.caption("Margin distribution unavailable (no valid logit values).")

            params = ss.get("train_params", {})
            st.markdown("**Parameters used**")
            st.markdown(
                f"- Hold-out fraction: {params.get('test_size', '—')}  \n"
                f"- Random seed: {params.get('random_state', '—')}  \n"
                f"- Max iterations: {params.get('max_iter', '—')}  \n"
                f"- C (inverse regularization): {params.get('C', '—')}"
            )

            model_obj = ss.get("model")
            has_calibration = bool(model_obj)
            calibrate_default = bool(ss.get("calibrate_probabilities", False))
            calib_toggle = st.toggle(
                "Calibrate probabilities (test set)",
                value=calibrate_default,
                key="train_calibrate_toggle",
                help="Platt scaling if test size ≥ 30, else isotonic disabled.",
                disabled=not has_calibration,
            )
            calib_active = bool(calib_toggle and has_calibration)
            ss["calibrate_probabilities"] = bool(calib_active)

            calibration_details = None
            if not has_calibration:
                st.caption("Unavailable")
                if hasattr(model_obj, "set_calibration"):
                    try:
                        model_obj.set_calibration(None)
                    except Exception:
                        pass
                ss["calibration_result"] = None
            elif calib_active:
                test_size = len(y_te_labels) if y_te_labels is not None else 0
                if test_size < 30:
                    st.caption("Unavailable")
                    if hasattr(model_obj, "set_calibration"):
                        try:
                            model_obj.set_calibration(None)
                        except Exception:
                            pass
                    calibration_details = {"status": "too_small", "test_size": test_size}
                    ss["calibration_result"] = calibration_details
                elif model_obj is None:
                    st.caption("Unavailable")
                else:
                    try:
                        spam_index = getattr(model_obj, "_i_spam", 1)
                        if hasattr(model_obj, "predict_proba_base"):
                            base_matrix = model_obj.predict_proba_base(X_te_t, X_te_b)
                        else:
                            base_matrix = model_obj.predict_proba(X_te_t, X_te_b)
                        base_matrix = np.asarray(base_matrix, dtype=float)
                        if base_matrix.ndim != 2 or base_matrix.shape[0] == 0:
                            raise ValueError("Empty probability matrix from model.")
                        base_probs = base_matrix[:, spam_index]
                        base_probs = np.clip(base_probs, 1e-6, 1 - 1e-6)
                        y_true01 = np.asarray(_y01(list(y_te_labels)), dtype=float)
                        base_logits = np.log(base_probs / (1.0 - base_probs))
                        calibrator = PlattProbabilityCalibrator(
                            random_state=int(params.get("random_state", 42))
                        )
                        calibrator.fit(base_logits, list(y_te_labels))
                        if hasattr(model_obj, "set_calibration"):
                            model_obj.set_calibration(calibrator)
                        calibrated_matrix = model_obj.predict_proba(X_te_t, X_te_b)
                        calibrated_matrix = np.asarray(calibrated_matrix, dtype=float)
                        calibrated_probs = calibrated_matrix[:, spam_index]
                        calibrated_probs = np.clip(calibrated_probs, 1e-6, 1 - 1e-6)
                        brier_before = float(np.mean((base_probs - y_true01) ** 2))
                        brier_after = float(np.mean((calibrated_probs - y_true01) ** 2))
                        bins = np.linspace(0.0, 1.0, 11)
                        reliability_rows: list[Dict[str, object]] = []
                        stages = [
                            ("Before calibration", base_probs),
                            ("After calibration", calibrated_probs),
                        ]
                        for stage_label, probs in stages:
                            bin_ids = np.digitize(probs, bins, right=True) - 1
                            bin_ids = np.clip(bin_ids, 0, len(bins) - 2)
                            for b in range(len(bins) - 1):
                                mask = bin_ids == b
                                if not np.any(mask):
                                    continue
                                reliability_rows.append(
                                    {
                                        "stage": stage_label,
                                        "bin": b,
                                        "expected": float(np.mean(probs[mask])),
                                        "observed": float(np.mean(y_true01[mask])),
                                        "count": int(mask.sum()),
                                    }
                                )
                        reliability_df = pd.DataFrame(reliability_rows)
                        reliability_df["bin_label"] = reliability_df["bin"].map(
                            lambda b: f"{bins[b]:.1f}–{bins[b + 1]:.1f}"
                        )
                        calibration_details = {
                            "status": "ok",
                            "brier_before": brier_before,
                            "brier_after": brier_after,
                            "test_size": test_size,
                            "reliability": reliability_df,
                        }
                        ss["calibration_result"] = calibration_details
                    except Exception:
                        st.caption("Unavailable")
                        if hasattr(model_obj, "set_calibration"):
                            try:
                                model_obj.set_calibration(None)
                            except Exception:
                                pass
                        calibration_details = {"status": "error"}
                        ss["calibration_result"] = calibration_details
            else:
                if hasattr(model_obj, "set_calibration"):
                    try:
                        model_obj.set_calibration(None)
                    except Exception:
                        pass
                ss["calibration_result"] = None

            if calibration_details and calibration_details.get("status") == "ok":
                brier_before = calibration_details["brier_before"]
                brier_after = calibration_details["brier_after"]
                delta = brier_after - brier_before
                col_b1, col_b2 = st.columns(2)
                col_b1.metric("Brier score (uncalibrated)", f"{brier_before:.4f}")
                col_b2.metric(
                    "Brier score (calibrated)",
                    f"{brier_after:.4f}",
                    delta=f"{delta:+.4f}",
                    delta_color="inverse",
                )
                st.caption(
                    f"Calibrated on {calibration_details['test_size']} hold-out examples using Platt scaling."
                )
                reliability_df = calibration_details.get("reliability")
                chart = build_calibration_chart(reliability_df)
                if chart is not None:
                    st.altair_chart(chart, use_container_width=True)
                    st.caption(
                        "We align predicted probabilities to reality. If the curve hugs the diagonal, scores are trustworthy."
                    )

            st.markdown(f"**Model object**: `{model_kind_string(ss['model'])}`")

            st.markdown("### Interpretability & tuning")
            try:
                coef_details = ss["model"].numeric_feature_details().copy()
                coef_details["friendly_name"] = coef_details["feature"].map(
                    FEATURE_DISPLAY_NAMES
                )
                st.caption(
                    "Positive weights push toward the **spam** class; negative toward **safe**. "
                    "Values are log-odds after standardization."
                )

                chart_data = (
                    coef_details.sort_values("weight_per_std", ascending=True)
                    .set_index("friendly_name")["weight_per_std"]
                )
                st.bar_chart(chart_data, width="stretch")
                st.caption(
                    "Bars to the right push toward 'Spam'; left bars push toward 'Safe'. Longer bar = stronger nudge."
                )

                display_df = coef_details.assign(
                    odds_multiplier_plus_1sigma=coef_details["odds_multiplier_per_std"],
                    approx_pct_change_odds=(coef_details["odds_multiplier_per_std"] - 1.0) * 100.0,
                )[
                    [
                        "friendly_name",
                        "base_weight_per_std",
                        "user_adjustment",
                        "weight_per_std",
                        "odds_multiplier_plus_1sigma",
                        "approx_pct_change_odds",
                        "train_mean",
                        "train_std",
                    ]
                ]

                st.dataframe(
                    display_df.rename(
                        columns={
                            "friendly_name": "Feature",
                            "base_weight_per_std": "Learned log-odds (+1σ)",
                            "user_adjustment": "Your adjustment (+1σ)",
                            "weight_per_std": "Adjusted log-odds (+1σ)",
                            "odds_multiplier_plus_1sigma": "Adjusted odds multiplier (+1σ)",
                            "approx_pct_change_odds": "%Δ odds from adjustment (+1σ)",
                            "train_mean": "Train mean",
                            "train_std": "Train std",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

                st.caption(
                    "Base weights come from training. Use the sliders below to nudge each cue if your domain knowledge "
                    "suggests it should count more or less. Adjustments apply per standard deviation of the raw feature."
                )

                st.markdown("#### What influenced the score (span knockout demo)")
                if not train_texts_combined:
                    st.caption("Span influence demo unavailable (no training emails).")
                elif model_obj is None or not hasattr(model_obj, "predict_logit"):
                    st.caption("Span influence demo unavailable (text-head logits missing).")
                else:
                    options = list(range(len(train_texts_combined)))

                    def _format_train_option(i: int) -> str:
                        label = (
                            y_tr_labels[i]
                            if y_tr_labels and 0 <= i < len(y_tr_labels)
                            else "?"
                        )
                        subject = X_tr_t[i] if X_tr_t and 0 <= i < len(X_tr_t) else ""
                        if not isinstance(subject, str) or not subject.strip():
                            subject = train_texts_combined[i][:80]
                        subject_short = shorten_text(str(subject).strip(), limit=80)
                        return f"{i + 1}. [{label.upper()}] {subject_short}" if label else subject_short

                    selected_idx = st.selectbox(
                        "Pick a training email",
                        options,
                        format_func=_format_train_option,
                        key="nerd_span_email_index",
                    )

                    selected_text = train_texts_combined[selected_idx]
                    candidate_spans = extract_candidate_spans(selected_text)

                    if not candidate_spans:
                        st.info("No influential spans detected for this email.")
                    else:
                        cache_bucket = ss.setdefault("nerd_span_cache", {})
                        text_hash = hashlib.sha1(selected_text.encode("utf-8")).hexdigest()
                        cache_key = f"{id(model_obj)}:{text_hash}"
                        cached = cache_bucket.get(cache_key)

                        if not cached:
                            try:
                                base_logit = float(model_obj.predict_logit([selected_text])[0])
                            except Exception as exc:
                                base_logit = None
                                cached = {"error": str(exc)}
                            else:
                                influence_rows: list[dict[str, float | str]] = []
                                for span_text, (start, end) in candidate_spans:
                                    modified = selected_text[:start] + selected_text[end:]
                                    try:
                                        new_logit = float(
                                            model_obj.predict_logit([modified])[0]
                                        )
                                    except Exception:
                                        continue
                                    delta = base_logit - new_logit
                                    influence_rows.append(
                                        {
                                            "span": span_text,
                                            "delta": delta,
                                        }
                                    )

                                influence_rows.sort(
                                    key=lambda row: row.get("delta", 0.0), reverse=True
                                )
                                cached = {
                                    "base_logit": base_logit,
                                    "rows": influence_rows,
                                }
                            cache_bucket[cache_key] = cached

                        if cached.get("error"):
                            st.caption(
                                "Could not compute span influence: "
                                f"{cached['error']}"
                            )
                        else:
                            base_logit = cached.get("base_logit")
                            rows = cached.get("rows", [])
                            positive_rows = [
                                row
                                for row in rows
                                if isinstance(row.get("delta"), (int, float))
                                and float(row["delta"]) > 0.0
                            ]

                            if base_logit is not None:
                                st.caption(
                                    f"Base text-head logit: {float(base_logit):+.3f}"
                                )

                            if not positive_rows:
                                st.info(
                                    "Removing detected spans did not lower the score."
                                )
                            else:
                                top_rows = positive_rows[:8]
                                display_rows = []
                                for row in top_rows:
                                    span_text = str(row.get("span", "")).replace("\n", " ")
                                    span_text = " ".join(span_text.split())
                                    if len(span_text) > 120:
                                        span_text = span_text[:117] + "…"
                                    display_rows.append(
                                        {
                                            "Span": span_text,
                                            "Δ logit (drop)": round(float(row["delta"]), 4),
                                        }
                                    )

                                df_spans = pd.DataFrame(display_rows)
                                st.dataframe(df_spans, hide_index=True, width="stretch")
                                st.caption(
                                    "We remove phrases and see how the score drops; bigger drops = more influence."
                                )

            except Exception as exc:
                st.caption(f"Coefficients unavailable: {exc}")
