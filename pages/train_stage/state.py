from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from demistifai.dataset import _estimate_token_stats, compute_dataset_hash
from demistifai.modeling import HybridEmbedFeatsLogReg, cache_train_embeddings, combine_text

MEANING_MAP_STATE_KEYS: Tuple[str, ...] = (
    "meaning_map_show_examples",
    "meaning_map_show_centers",
    "meaning_map_highlight_borderline",
    "meaning_map_show_pair_trigger",
)


@dataclass
class TokenBudgetSummary:
    """Snapshot of average token usage across the labeled dataset."""

    text: str
    show_truncation_tip: bool
    average_tokens: Optional[float]


@dataclass
class TrainRunParams:
    """Typed view over the training hyper-parameters stored in session state."""

    test_size: float
    random_state: int
    max_iter: int
    C: float


@dataclass
class TrainSplit:
    """Container for the train/test split used during training."""

    train_titles: List[str]
    test_titles: List[str]
    train_bodies: List[str]
    test_bodies: List[str]
    train_labels: List[str]
    test_labels: List[str]

    def as_tuple(self) -> Tuple[List[str], ...]:
        return (
            self.train_titles,
            self.test_titles,
            self.train_bodies,
            self.test_bodies,
            self.train_labels,
            self.test_labels,
        )


@dataclass
class TrainingOutcome:
    """Encapsulate the result of a training run for downstream state updates."""

    model: HybridEmbedFeatsLogReg
    split: TrainSplit
    guard_params: Dict[str, Any]
    cached_embeddings: bool
    auto_select_error: Optional[BaseException] = None


def ensure_train_stage_state(ss: MutableMapping[str, Any], *, threshold: float) -> Dict[str, Any]:
    """Ensure Train stage specific state exists and return guard parameters."""

    ss.setdefault("token_budget_cache", {})
    ss.setdefault("train_in_progress", False)
    ss.setdefault("train_refresh_expected", False)
    ss.setdefault("train_refresh_attempts", 0)
    ss.setdefault("train_params", {"test_size": 0.30, "random_state": 42, "max_iter": 1000, "C": 1.0})

    guard_params = dict(ss.get("guard_params") or {})
    guard_params.setdefault("assist_center_mode", "manual")
    guard_params.setdefault("assist_center", float(threshold))
    guard_params.setdefault("uncertainty_band", 0.08)
    guard_params.setdefault("numeric_scale", 0.5)
    guard_params.setdefault("numeric_logit_cap", 1.0)
    guard_params.setdefault("combine_strategy", "blend")
    guard_params.setdefault("shift_suspicious_tld", -0.04)
    guard_params.setdefault("shift_many_links", -0.03)
    guard_params.setdefault("shift_calm_text", 0.02)
    ss["guard_params"] = guard_params
    return guard_params


def compute_token_budget_summary(
    labeled_rows: Sequence[MutableMapping[str, Any]] | Sequence[Dict[str, Any]],
    cache: MutableMapping[str, Any],
    *,
    max_tokens: int = 384,
) -> Tuple[TokenBudgetSummary, MutableMapping[str, Any]]:
    """Compute cached token statistics for the labeled dataset."""

    try:
        if not labeled_rows:
            return TokenBudgetSummary("Token budget: —", False, None), cache

        dataset_hash = compute_dataset_hash(labeled_rows)
        stats = cache.get(dataset_hash)
        if stats is None:
            titles = [str(row.get("title", "")) for row in labeled_rows]
            bodies = [str(row.get("body", "")) for row in labeled_rows]
            stats = _estimate_token_stats(titles, bodies, max_tokens=max_tokens)
            cache[dataset_hash] = stats

        if not stats or not stats.get("n"):
            return TokenBudgetSummary("Token budget: —", False, None), cache

        avg_tokens_value = float(stats.get("avg_tokens", 0.0))
        pct_trunc = float(stats.get("p_truncated", 0.0)) * 100.0
        if np.isfinite(avg_tokens_value) and avg_tokens_value > 0.0:
            avg_tokens = avg_tokens_value
            text = f"Token budget: avg ~{avg_tokens_value:.0f} • truncated: {pct_trunc:.1f}%"
        else:
            avg_tokens = None
            text = "Token budget: —"
        show_tip = float(stats.get("p_truncated", 0.0)) > 0.05
        return TokenBudgetSummary(text, show_tip, avg_tokens), cache
    except Exception:
        return TokenBudgetSummary("Token budget: —", False, None), cache


def parse_train_params(raw: MutableMapping[str, Any] | Dict[str, Any]) -> TrainRunParams:
    """Normalise the training parameters stored in session state."""

    return TrainRunParams(
        test_size=float(raw.get("test_size", 0.30)),
        random_state=int(raw.get("random_state", 42)),
        max_iter=int(raw.get("max_iter", 1000)),
        C=float(raw.get("C", 1.0)),
    )


def create_train_split(
    titles: Sequence[str],
    bodies: Sequence[str],
    labels: Sequence[str],
    *,
    test_size: float,
    random_state: int,
) -> TrainSplit:
    """Create a stratified train/test split for the labeled dataset."""

    (X_tr_t, X_te_t, X_tr_b, X_te_b, y_tr, y_te) = train_test_split(
        list(titles),
        list(bodies),
        list(labels),
        test_size=test_size,
        random_state=random_state,
        stratify=list(labels),
    )
    return TrainSplit(
        train_titles=list(X_tr_t),
        test_titles=list(X_te_t),
        train_bodies=list(X_tr_b),
        test_bodies=list(X_te_b),
        train_labels=list(y_tr),
        test_labels=list(y_te),
    )


def train_model_on_split(
    split: TrainSplit,
    *,
    params: TrainRunParams,
    guard_params: Dict[str, Any],
) -> HybridEmbedFeatsLogReg:
    """Fit the HybridEmbedFeatsLogReg model using the provided split and guard settings."""

    model = HybridEmbedFeatsLogReg(
        max_iter=int(params.max_iter),
        C=float(params.C),
        random_state=int(params.random_state),
        numeric_assist_center=float(guard_params.get("assist_center", 0.6)),
        uncertainty_band=float(guard_params.get("uncertainty_band", 0.08)),
        numeric_scale=float(guard_params.get("numeric_scale", 0.5)),
        numeric_logit_cap=float(guard_params.get("numeric_logit_cap", 1.0)),
        combine_strategy=str(guard_params.get("combine_strategy", "blend")),
        shift_suspicious_tld=float(guard_params.get("shift_suspicious_tld", -0.04)),
        shift_many_links=float(guard_params.get("shift_many_links", -0.03)),
        shift_calm_text=float(guard_params.get("shift_calm_text", 0.02)),
    )
    return model.fit(split.train_titles, split.train_bodies, split.train_labels)


def execute_training_pipeline(
    titles: Sequence[str],
    bodies: Sequence[str],
    labels: Sequence[str],
    *,
    params: TrainRunParams,
    guard_params: Dict[str, Any],
    numeric_adjustments: Optional[MutableMapping[str, Any]],
    has_embed: bool,
    fallback_center: float,
) -> TrainingOutcome:
    """Run the end-to-end training flow and return the outcome for state updates."""

    split = create_train_split(
        titles,
        bodies,
        labels,
        test_size=params.test_size,
        random_state=params.random_state,
    )

    model = train_model_on_split(split, params=params, guard_params=guard_params)

    adjustments = dict(numeric_adjustments or {})
    if adjustments:
        model.apply_numeric_adjustments(adjustments)

    auto_select_error: Optional[BaseException] = None
    updated_guard_params = dict(guard_params)
    try:
        updated_guard_params = auto_select_assist_center(
            model,
            split,
            updated_guard_params,
            fallback_center=fallback_center,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        auto_select_error = exc

    cached_embeddings = False
    if has_embed:
        try:
            train_texts_cache = [
                combine_text(t, b)
                for t, b in zip(split.train_titles, split.train_bodies)
            ]
            if train_texts_cache:
                cache_train_embeddings(train_texts_cache)
                cached_embeddings = True
        except Exception:  # pragma: no cover - cache warm-up best effort
            cached_embeddings = False

    return TrainingOutcome(
        model=model,
        split=split,
        guard_params=updated_guard_params,
        cached_embeddings=cached_embeddings,
        auto_select_error=auto_select_error,
    )


def auto_select_assist_center(
    model: HybridEmbedFeatsLogReg,
    split: TrainSplit,
    guard_params: Dict[str, Any],
    *,
    fallback_center: float,
) -> Dict[str, Any]:
    """Update the guard rail center based on hold-out performance when in auto mode."""

    gp = dict(guard_params)
    if gp.get("assist_center_mode") != "auto":
        return gp
    if not split.test_titles or not split.test_bodies:
        return gp

    probs_te_raw = model.predict_proba(split.test_titles, split.test_bodies)
    probs_te = np.asarray(probs_te_raw, dtype=float)
    if probs_te.ndim != 2 or probs_te.shape[1] < 2:
        raise ValueError("predict_proba must return a 2D array with at least two columns")
    spam_idx = int(getattr(model, "_i_spam", 1))
    if spam_idx < 0 or spam_idx >= probs_te.shape[1]:
        raise IndexError("Spam index is out of bounds for predict_proba output")
    p_spam_te = np.clip(probs_te[:, spam_idx], 1e-6, 1 - 1e-6)
    y_true = np.asarray(split.test_labels) == "spam"
    best_tau, best_f1 = fallback_center, -1.0
    if y_true.size:
        for tau in np.linspace(0.30, 0.90, 61):
            y_pred = p_spam_te >= tau
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1, best_tau = float(f1), float(tau)
    gp["assist_center"] = float(best_tau)
    return gp


def persist_training_outcome(
    ss: MutableMapping[str, Any],
    *,
    model: HybridEmbedFeatsLogReg,
    split: TrainSplit,
    guard_params: Dict[str, Any],
) -> None:
    """Store the model, split cache, and guard parameters back into session state."""

    ss["model"] = model
    ss["split_cache"] = split.as_tuple()
    ss["guard_params"] = dict(guard_params)


def register_training_refresh(ss: MutableMapping[str, Any], *, threshold: float) -> None:
    """Mark that training finished so downstream charts refresh."""

    ss["eval_timestamp"] = datetime.now().isoformat(timespec="seconds")
    ss["eval_temp_threshold"] = float(threshold)
    ss["train_story_run_id"] = uuid4().hex
    ss["train_flash_finished"] = True
    ss["train_refresh_expected"] = True
    ss["train_refresh_attempts"] = 1


def reset_meaning_map_flags(ss: MutableMapping[str, Any]) -> None:
    """Clear cached meaning map toggles so they recompute after training."""

    for key in MEANING_MAP_STATE_KEYS:
        ss.pop(key, None)
