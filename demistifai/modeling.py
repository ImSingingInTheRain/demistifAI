"""Modeling utilities for demistifAI."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import HashingVectorizer
try:  # sentence-transformers is optional at runtime
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency safeguard
    SentenceTransformer = None  # type: ignore
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .constants import SUSPICIOUS_TLD_SUFFIXES

__all__ = [
    "df_confusion",
    "assess_performance",
    "_counts",
    "_y01",
    "compute_confusion",
    "_pr_acc_cm",
    "_fmt_pct",
    "_fmt_delta",
    "threshold_presets",
    "make_after_eval_story",
    "verdict_label",
    "plot_threshold_curves",
    "make_after_training_story",
    "model_kind_string",
    "get_encoder",
    "encode_texts",
    "cache_train_embeddings",
    "get_nearest_training_examples",
    "predict_spam_probability",
    "numeric_feature_contributions",
    "top_token_importances",
    "combine_text",
    "_combine_text",
    "_predict_proba_batch",
    "extract_urls",
    "get_domain_tld",
    "compute_numeric_features",
    "FEATURE_ORDER",
    "FEATURE_DISPLAY_NAMES",
    "FEATURE_PLAIN_LANGUAGE",
    "features_matrix",
    "HybridEmbedFeatsLogReg",
    "PlattProbabilityCalibrator",
    "embedding_backend_info",
]


def _safe_logit(p: np.ndarray) -> np.ndarray:
    """Compute logit scores while avoiding infinities at 0 or 1."""

    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


class PlattProbabilityCalibrator:
    """One-dimensional logistic regression for probability calibration."""

    def __init__(self, max_iter: int = 1000, random_state: int | None = None):
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.lr = LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
        self._fitted = False

    def fit(self, logits: np.ndarray, labels: List[str] | np.ndarray) -> "PlattProbabilityCalibrator":
        logits = np.asarray(logits, dtype=float).reshape(-1, 1)
        if logits.size == 0:
            raise ValueError("Cannot fit calibrator without logits.")
        if isinstance(labels, np.ndarray):
            y_bin = np.array([1 if y == "spam" else 0 for y in labels.tolist()], dtype=int)
        else:
            y_bin = _y01(list(labels))
        self.lr.fit(logits, y_bin)
        self._fitted = True
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Calibrator not fitted.")
        logits = np.asarray(logits, dtype=float).reshape(-1, 1)
        if logits.size == 0:
            return np.empty((0,), dtype=float)
        return self.lr.predict_proba(logits)[:, 1]

    def __call__(self, probs: np.ndarray) -> np.ndarray:
        logits = _safe_logit(np.asarray(probs, dtype=float))
        calibrated = self.transform_logits(logits)
        return np.clip(calibrated, 0.0, 1.0)

def df_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"True: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels])


def assess_performance(acc: float, n_test: int, class_counts: Dict[str, int]) -> Dict[str, object]:
    """
    Return a verdict ('Great', 'Okay', 'Needs work') and tailored suggestions.
    Heuristics:
      - acc >= 0.90 and n_test >= 10 -> Great
      - 0.75 <= acc < 0.90 or n_test < 10 -> Okay
      - acc < 0.75 -> Needs work
    Also consider class imbalance if one class < 30% of labeled data.
    """
    verdict = "Okay"
    if n_test >= 10 and acc >= 0.90:
        verdict = "Great"
    elif acc < 0.75:
        verdict = "Needs work"

    tips: List[str] = []
    if verdict != "Great":
        tips.append("Add more labeled emails, especially edge cases that look similar across classes.")
        tips.append("Balance the dataset (roughly comparable counts of 'spam' and 'safe').")
        tips.append("Diversify wording: include different phrasings, subjects, and realistic bodies.")
    tips.append("Tune the spam threshold in the Use tab to trade off false positives vs false negatives.")
    tips.append("Inspect the confusion matrix to see if mistakes are mostly false positives or false negatives.")
    tips.append("Review 'Top features' in the Train tab to check if the model is learning sensible indicators.")
    tips.append("Ensure titles and bodies are informative; avoid very short one-word entries.")

    total_labeled = sum(class_counts.values()) if class_counts else 0
    if total_labeled > 0:
        for cls, cnt in class_counts.items():
            share = cnt / total_labeled
            if share < 0.30:
                tips.insert(0, f"Label more '{cls}' examples (currently ~{share:.0%}), the model may be biased.")
                break

    return {"verdict": verdict, "tips": tips}


def _counts(labels: list[str]) -> Dict[str, int]:
    counts = {"spam": 0, "safe": 0}
    for y in labels:
        if y in counts:
            counts[y] += 1
    return counts


def _y01(labels: List[str]) -> np.ndarray:
    return np.array([1 if y == "spam" else 0 for y in labels], dtype=int)


def compute_confusion(y_true01: np.ndarray, p_spam: np.ndarray, thr: float) -> Dict[str, int]:
    y_hat01 = (p_spam >= thr).astype(int)
    tp = int(((y_hat01 == 1) & (y_true01 == 1)).sum())
    tn = int(((y_hat01 == 0) & (y_true01 == 0)).sum())
    fp = int(((y_hat01 == 1) & (y_true01 == 0)).sum())
    fn = int(((y_hat01 == 0) & (y_true01 == 1)).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def _pr_acc_cm(y_true01: np.ndarray, p_spam: np.ndarray, thr: float) -> Tuple[float, float, float, float, Dict[str, int]]:
    y_hat = (p_spam >= thr).astype(int)
    acc = float((y_hat == y_true01).sum()) / max(1, len(y_true01))
    p, r, f1, _ = precision_recall_fscore_support(y_true01, y_hat, average="binary", zero_division=0)
    cm = compute_confusion(y_true01, p_spam, thr)
    return acc, p, r, f1, cm


def _fmt_pct(v: float) -> str:
    return f"{v:.2%}"


def _fmt_delta(new: float | int, old: float | int, pct: bool = True) -> str:
    d = new - old
    if abs(d) < 1e-9:
        return "—"
    arrow = "▲" if d > 0 else "▼"
    if pct:
        return f"{arrow}{d:+.2%}"
    return f"{arrow}{d:+d}"


def threshold_presets(y_true01: np.ndarray, p_spam: np.ndarray) -> Dict[str, float]:
    thrs = np.linspace(0.1, 0.9, 81)
    best_f1, thr_f1 = -1.0, 0.5
    thr_prec95, thr_rec90 = 0.5, 0.5
    best_prec_gap = 1e9
    best_rec_gap = 1e9
    for t in thrs:
        y_hat = (p_spam >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true01, y_hat, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, thr_f1 = f1, float(t)
        if p >= 0.95 and (p - 0.95) < best_prec_gap:
            best_prec_gap, thr_prec95 = (p - 0.95), float(t)
        if r >= 0.90 and (r - 0.90) < best_rec_gap:
            best_rec_gap, thr_rec90 = (r - 0.90), float(t)
    return {
        "balanced_f1": thr_f1,
        "precision_95": thr_prec95,
        "recall_90": thr_rec90,
    }


def make_after_eval_story(n_test: int, cm: Dict[str, int]) -> str:
    right = cm["TP"] + cm["TN"]
    wrong = cm["FP"] + cm["FN"]
    lines = []
    lines.append(
        f"Out of **{n_test}** test emails, the model got **{right}** right and **{wrong}** wrong."
    )
    if cm["FN"] > 0:
        lines.append(f"• **Spam that slipped through** (false negatives): {cm['FN']}")
    if cm["FP"] > 0:
        lines.append(f"• **Safe emails wrongly flagged** (false positives): {cm['FP']}")
    lines.append(
        "You can improve results by adding more labeled examples, balancing spam/safe, "
        "diversifying wording, and tuning the spam threshold below."
    )
    return "\n".join(lines)


def verdict_label(acc: float, n: int) -> Tuple[str, str]:
    if n < 10:
        return "🟡", "Okay (small test set — results may vary)"
    if acc >= 0.90:
        return "🟢", "Great"
    if acc >= 0.75:
        return "🟡", "Okay"
    return "🔴", "Needs work"


def plot_threshold_curves(y_true01: np.ndarray, p_spam: np.ndarray):
    thrs = np.linspace(0.1, 0.9, 33)
    prec, rec = [], []
    for t in thrs:
        y_hat = (p_spam >= t).astype(int)
        p, r, _, _ = precision_recall_fscore_support(
            y_true01, y_hat, average="binary", zero_division=0
        )
        prec.append(p)
        rec.append(r)
    fig, ax = plt.subplots()
    ax.plot(thrs, prec, marker="o", label="Precision (spam)")
    ax.plot(thrs, rec, marker="o", label="Recall (spam)")
    ax.set_xlabel("Threshold (P(spam))")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def make_after_training_story(train_labels: list[str], test_labels: list[str]) -> str:
    n_train = len(train_labels)
    n_test = len(test_labels)
    ct_train = _counts(train_labels)
    ct_test = _counts(test_labels)
    lines: list[str] = []
    lines.append(
        (
            f"**Training complete.** The model learned from **{n_train}** emails "
            f"({ct_train['spam']} spam / {ct_train['safe']} safe) and will be checked on "
            f"**{n_test}** unseen emails ({ct_test['spam']} spam / {ct_test['safe']} safe)."
        )
    )
    lines.append(
        "It built an internal map of patterns that distinguish spam from safe messages, "
        "so it can **infer** the right category for new emails."
    )
    lines.append(
        "Next, open **🧪 Evaluate** to see how well it performs on the held-out test set."
    )
    return "\n\n".join(lines)


def model_kind_string(model_obj: Any) -> str:
    name = type(model_obj).__name__
    try:
        if hasattr(model_obj, "named_steps"):
            steps = " + ".join(model_obj.named_steps.keys())
            return f"{name} ({steps})"
        return name
    except Exception:
        return name

_HASHING_ENCODER = HashingVectorizer(
    n_features=512,
    alternate_sign=False,
    norm="l2",
)


_EMBEDDING_BACKEND_STATE: dict[str, str | None] = {"kind": "uninitialized", "error": None}
_SENTENCE_ENCODER_ERROR: str | None = None
_SENTENCE_ENCODER_ENABLED = True


def _record_embedding_backend(kind: str, error: str | None = None) -> None:
    """Persist the active embedding backend for UI diagnostics."""

    _EMBEDDING_BACKEND_STATE["kind"] = kind
    _EMBEDDING_BACKEND_STATE["error"] = error
    try:
        st.session_state["_embedding_backend"] = _EMBEDDING_BACKEND_STATE.copy()
    except Exception:  # pragma: no cover - session state unavailable in tests
        pass


def embedding_backend_info() -> dict[str, str | None]:
    """Return the most recent embedding backend metadata."""

    return _EMBEDDING_BACKEND_STATE.copy()


def _encode_with_hashing(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    matrix = _HASHING_ENCODER.transform(texts)
    arr = matrix.toarray().astype(np.float32, copy=False)
    _record_embedding_backend("hashing-vectorizer", _SENTENCE_ENCODER_ERROR)
    return arr


@st.cache_resource(show_spinner=False)
def get_encoder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SentenceTransformer:
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError("sentence_transformers_unavailable")
    # Downloaded once and cached by Streamlit
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def encode_texts(
    texts: list,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    global _SENTENCE_ENCODER_ENABLED, _SENTENCE_ENCODER_ERROR

    if _SENTENCE_ENCODER_ENABLED and _SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            model = get_encoder(model_name)
            embs = model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            arr = np.asarray(embs, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - external dependency failure
            _SENTENCE_ENCODER_ENABLED = False
            _SENTENCE_ENCODER_ERROR = str(exc) or exc.__class__.__name__
        else:
            _SENTENCE_ENCODER_ERROR = None
            _record_embedding_backend("sentence-transformer", None)
            return arr

    return _encode_with_hashing([str(t) for t in texts])


@st.cache_data(show_spinner=False)
def cache_train_embeddings(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    return encode_texts(texts)


def get_nearest_training_examples(
    query_text: str,
    X_train_texts: list[str],
    y_train: list[str],
    X_train_emb: np.ndarray | None = None,
    k: int = 3,
) -> list[dict[str, Any]]:
    if not X_train_texts:
        return []
    if X_train_emb is None or getattr(X_train_emb, "size", 0) == 0:
        X_train_emb = cache_train_embeddings(X_train_texts)
    if getattr(X_train_emb, "size", 0) == 0:
        return []

    q = encode_texts([query_text])[0]
    try:
        sims = X_train_emb @ q
    except ValueError:
        return []
    idx = np.argsort(-sims)[:k]
    out: list[dict[str, Any]] = []
    for i in idx:
        if i < 0 or i >= len(X_train_texts):
            continue
        out.append(
            {
                "text": X_train_texts[i],
                "label": y_train[i] if i < len(y_train) else "?",
                "similarity": float(sims[i]),
            }
        )
    return out


def predict_spam_probability(model: Any, title: str, body: str) -> Optional[float]:
    if model is None:
        return None
    try:
        probs = model.predict_proba([title], [body])
    except TypeError:
        text = combine_text(title, body)
        probs = model.predict_proba([text])
    except Exception:
        return None

    probs_arr = np.asarray(probs)
    if probs_arr.ndim != 2 or probs_arr.shape[0] == 0:
        return None

    classes = list(getattr(model, "classes_", []))
    if classes and "spam" in classes:
        idx_spam = classes.index("spam")
    else:
        idx_spam = 1 if probs_arr.shape[1] > 1 else 0
    try:
        return float(probs_arr[0, idx_spam])
    except (IndexError, TypeError, ValueError):
        return None


def numeric_feature_contributions(model: Any, title: str, body: str) -> list[tuple[str, float, float]]:
    raw = compute_numeric_features(title, body)
    vec = np.array([[raw[k] for k in FEATURE_ORDER]], dtype=np.float32)
    try:
        z = model.scaler.transform(vec)[0]
    except Exception:
        return []

    n_num = len(FEATURE_ORDER)
    weights = None
    if hasattr(model, "lr_num") and getattr(model, "lr_num", None) is not None:
        try:
            coef = model.lr_num.coef_[0][:n_num]
            adj_dict = getattr(model, "_user_adj", {})
            adj = np.array([adj_dict.get(feat, 0.0) for feat in FEATURE_ORDER], dtype=np.float32)
            weights = coef + adj
        except Exception:
            weights = None
    elif hasattr(model, "lr"):
        try:
            weights = model.lr.coef_[0][-n_num:]
        except Exception:
            weights = None

    if weights is None:
        return []
    contrib = z * weights
    return list(zip(FEATURE_ORDER, z.tolist(), contrib.tolist()))


def top_token_importances(
    model: Any,
    title: str,
    body: str,
    *,
    max_tokens: int = 20,
) -> tuple[Optional[float], list[dict[str, Any]]]:
    text = combine_text(title, body)
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text)
    seen: set[str] = set()
    candidates: list[str] = []
    for tok in tokens:
        key = tok.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(tok)
        if len(candidates) >= max_tokens:
            break

    base = predict_spam_probability(model, title, body)
    if base is None:
        return None, []

    rows: list[dict[str, Any]] = []
    body_text = body or ""
    for tok in candidates:
        masked_body = re.sub(rf"\b{re.escape(tok)}\b", "", body_text, count=1)
        masked_prob = predict_spam_probability(model, title, masked_body)
        if masked_prob is None:
            continue
        delta = base - masked_prob
        rows.append({"token": tok, "importance": round(delta, 4)})

    rows.sort(key=lambda x: x["importance"], reverse=True)
    return base, rows


def combine_text(title: str, body: str) -> str:
    return (title or "") + "\n" + (body or "")


def _combine_text(title: str, body: str) -> str:
    return combine_text(title, body)


def _predict_proba_batch(model, items, split_cache=None):
    """Return predictions and class probabilities for a batch of items."""

    titles = [it.get("title", "") for it in items]
    bodies = [it.get("body", "") for it in items]

    try:
        probs = model.predict_proba(titles, bodies)
        classes = list(getattr(model, "classes_", []))
        y_hat = model.predict(titles, bodies)
    except TypeError:
        texts = [_combine_text(t, b) for t, b in zip(titles, bodies)]
        probs = model.predict_proba(texts)
        classes = list(getattr(model, "classes_", []))
        y_hat = model.predict(texts)

    if not classes and hasattr(model, "classes_"):
        classes = list(model.classes_)

    if classes and "spam" in classes:
        i_spam = classes.index("spam")
        i_safe = classes.index("safe") if "safe" in classes else 1 - i_spam
    else:
        i_spam = 1 if probs.shape[1] > 1 else 0
        i_safe = 1 - i_spam if probs.shape[1] > 1 else 0

    p_spam = np.asarray(probs)[:, i_spam]
    p_safe = np.asarray(probs)[:, i_safe] if probs.shape[1] > 1 else 1.0 - p_spam

    return list(y_hat), p_spam, p_safe



URGENCY_TERMS = {"urgent", "immediately", "now", "asap", "final", "last chance", "act now", "action required", "limited time", "expires", "today only"}

URL_REGEX = re.compile(r"https?://[^\s)>\]}]+", re.IGNORECASE)
TOKEN_REGEX = re.compile(r"\b\w+\b", re.UNICODE)

def extract_urls(text: str) -> List[str]:
    return URL_REGEX.findall(text or "")

def get_domain_tld(url: str) -> Tuple[str, str]:
    try:
        netloc = urlparse(url).netloc.lower()
        if ":" in netloc:
            netloc = netloc.split(":")[0]
        # tld as the last dot suffix (naive but sufficient for demo)
        parts = netloc.split(".")
        tld = "." + parts[-1] if len(parts) >= 2 else ""
        return netloc, tld
    except Exception:
        return "", ""

def compute_numeric_features(title: str, body: str) -> Dict[str, float]:
    text = (title or "") + "\n" + (body or "")
    urls = extract_urls(text)
    num_links = len(urls)
    suspicious = 0
    external_links = 0
    for u in urls:
        dom, tld = get_domain_tld(u)
        if tld in SUSPICIOUS_TLD_SUFFIXES:
            suspicious = 1
        # treat anything with a dot and not an intranet-like suffix as external (demo logic)
        if dom and "." in dom:
            external_links += 1

    tokens = TOKEN_REGEX.findall(text)
    n_tokens = max(1, len(tokens))

    punct_bursts = re.findall(r"([!?$#*])\1{1,}", text)  # repeated punctuation like "!!!", "$$$"
    punct_burst_ratio = len(punct_bursts) / max(1, num_links + n_tokens)  # normalize by size

    money_symbol_count = text.count("€") + text.count("$") + text.count("£")

    lower = text.lower()
    urgency_terms_count = 0
    for term in URGENCY_TERMS:
        urgency_terms_count += lower.count(term)

    # Keep names stable — used in UI and coef table
    feats = {
        "num_links_external": float(external_links),
        "has_suspicious_tld": float(suspicious),
        "punct_burst_ratio": float(punct_burst_ratio),
        "money_symbol_count": float(money_symbol_count),
        "urgency_terms_count": float(urgency_terms_count),
    }
    return feats

FEATURE_ORDER = [
    "num_links_external",
    "has_suspicious_tld",
    "punct_burst_ratio",
    "money_symbol_count",
    "urgency_terms_count",
]

FEATURE_DISPLAY_NAMES = {
    "num_links_external": "External links counted",
    "has_suspicious_tld": "Suspicious top-level domain present",
    "punct_burst_ratio": "Intense punctuation bursts",
    "money_symbol_count": "Currency symbols mentioned",
    "urgency_terms_count": "Urgent or time-pressure phrases",
}

FEATURE_PLAIN_LANGUAGE = {
    "num_links_external": "Spam often includes many external links. More links push the prediction toward spam.",
    "has_suspicious_tld": "If any link points to a risky domain (e.g., .ru, .top) the odds of spam increase sharply.",
    "punct_burst_ratio": "Repeated punctuation like !!! or $$$ is a red flag and raises the spam score.",
    "money_symbol_count": "Lots of currency symbols usually signal scams promising money or demanding payment.",
    "urgency_terms_count": "Phrases such as 'urgent' or 'final notice' are classic spam urgency tactics.",
}

def features_matrix(titles: List[str], bodies: List[str]) -> np.ndarray:
    rows = []
    for t, b in zip(titles, bodies):
        f = compute_numeric_features(t, b)
        rows.append([f[k] for k in FEATURE_ORDER])
    return np.array(rows, dtype=np.float32)


class HybridEmbedFeatsLogReg:
    """Guarded hybrid model combining text embeddings with numeric cues."""

    def __init__(
        self,
        max_iter: int = 1000,
        C: float = 1.0,
        random_state: int = 42,
        # guarded-combination params
        numeric_assist_center: float = 0.5,
        uncertainty_band: float = 0.08,
        numeric_scale: float = 0.5,
        numeric_logit_cap: float = 1.0,
        combine_strategy: str = "blend",
        # threshold-shift micro-rules
        shift_suspicious_tld: float = -0.04,
        shift_many_links: float = -0.03,
        shift_calm_text: float = +0.02,
    ):
        self.max_iter = max_iter
        self.C = C
        self.random_state = random_state

        self.lr_text: CalibratedClassifierCV | None = None
        self.lr_text_base: LogisticRegression | None = None
        self.lr_num: LogisticRegression | None = None

        self.scaler = StandardScaler()
        self.classes_ = np.array(["safe", "spam"])

        self.numeric_assist_center = float(numeric_assist_center)
        self.uncertainty_band = float(uncertainty_band)
        self.numeric_scale = float(numeric_scale)
        self.numeric_logit_cap = float(numeric_logit_cap)
        self.combine_strategy = str(combine_strategy)

        self.shift_suspicious_tld = float(shift_suspicious_tld)
        self.shift_many_links = float(shift_many_links)
        self.shift_calm_text = float(shift_calm_text)

        self.notes: list[str] = []
        self._user_adj = {k: 0.0 for k in FEATURE_ORDER}
        self.last_thr_eff: tuple[np.ndarray, np.ndarray] | None = None
        self.calibrator: Callable[[np.ndarray], np.ndarray] | None = None

    # --- compatibility helpers ---
    def _encode_concat(self, titles, bodies):
        texts = [combine_text(t, b) for t, b in zip(titles, bodies)]
        emb = encode_texts(texts)
        return emb

    def _numeric_raw(self, titles, bodies):
        rows = []
        for t, b in zip(titles, bodies):
            feats = compute_numeric_features(t, b)
            rows.append([feats[k] for k in FEATURE_ORDER])
        return np.array(rows, dtype=np.float32)

    def _numeric_std_block(self, titles, bodies):
        X_num = self._numeric_raw(titles, bodies)
        X_std = self.scaler.transform(X_num)
        return X_std

    def numeric_feature_details(self):
        coef = np.zeros(len(FEATURE_ORDER))
        if self.lr_num is not None:
            coef = self.lr_num.coef_[0]
        train_mean = getattr(self, "_train_num_mean", np.zeros_like(coef))
        train_std = getattr(self, "_train_num_std", np.ones_like(coef))
        base = coef
        adj = np.array([self._user_adj[k] for k in FEATURE_ORDER], dtype=np.float32)
        final = base + adj
        odds_mult = np.exp(final)
        import pandas as _pd

        return _pd.DataFrame(
            {
                "feature": FEATURE_ORDER,
                "base_weight_per_std": base,
                "user_adjustment": adj,
                "weight_per_std": final,
                "odds_multiplier_per_std": odds_mult,
                "train_mean": train_mean,
                "train_std": train_std,
            }
        )

    def apply_numeric_adjustments(self, adj_dict: dict):
        for k, v in adj_dict.items():
            if k in self._user_adj:
                self._user_adj[k] = float(v)

    def fit(self, titles, bodies, y):
        y_bin = np.array([1 if _y == "spam" else 0 for _y in y], dtype=np.int32)

        X_emb = self._encode_concat(titles, bodies)

        X_num_raw = self._numeric_raw(titles, bodies)
        self.scaler.fit(X_num_raw)
        self._train_num_mean = self.scaler.mean_.copy()
        self._train_num_std = np.sqrt(self.scaler.var_.copy())

        X_num_std = self.scaler.transform(X_num_raw)

        idx = np.arange(len(y_bin))
        (
            X_emb_tr,
            X_emb_val,
            y_tr,
            y_val,
            idx_tr,
            idx_val,
        ) = train_test_split(
            X_emb,
            y_bin,
            idx,
            test_size=0.2,
            stratify=y_bin,
            random_state=self.random_state,
        )
        self._cv_idx_tr = np.asarray(idx_tr, dtype=int)
        self._cv_idx_val = np.asarray(idx_val, dtype=int)

        base_text = LogisticRegression(
            max_iter=self.max_iter, C=self.C, random_state=self.random_state
        )
        base_text.fit(X_emb_tr, y_tr)
        self.lr_text_base = base_text
        frozen_base = FrozenEstimator(base_text)
        self.lr_text = CalibratedClassifierCV(
            frozen_base,
            method="isotonic",
            ensemble=False,
        )
        self.lr_text.fit(X_emb_val, y_val)

        self.lr_num = LogisticRegression(
            max_iter=1000,
            C=0.8,
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.lr_num.fit(X_num_std, y_bin)

        expected_pos = {"external_link_count", "suspicious_tld", "all_caps_ratio"}
        coef = self.lr_num.coef_[0]
        for i, feat in enumerate(FEATURE_ORDER):
            if feat in expected_pos and coef[i] < 0:
                coef[i] *= 0.5
                self.notes.append(f"Auto-damped numeric weight sign for {feat}")
        self.lr_num.coef_[0] = coef

        return self

    @property
    def _i_spam(self):
        return 1

    def _p_text(self, titles, bodies):
        if self.lr_text is None:
            raise RuntimeError("Text model not trained")
        X_emb = self._encode_concat(titles, bodies)
        return self.lr_text.predict_proba(X_emb)[:, self._i_spam]

    def _p_num(self, titles, bodies):
        if self.lr_num is None:
            raise RuntimeError("Numeric model not trained")
        X_std = self._numeric_std_block(titles, bodies)
        adj = np.array([self._user_adj[k] for k in FEATURE_ORDER], dtype=np.float32)
        logit = (X_std @ (self.lr_num.coef_[0] + adj)) + self.lr_num.intercept_[0]
        logit = np.clip(logit, -abs(self.numeric_logit_cap), +abs(self.numeric_logit_cap))
        p = 1.0 / (1.0 + np.exp(-logit))
        return p, X_std

    def _predict_proba_base(
        self, titles, bodies, threshold: float | None = None
    ):
        self.last_thr_eff = None
        if threshold is None:
            threshold = float(getattr(self, "numeric_assist_center", 0.5))
        p_txt = self._p_text(titles, bodies)
        low, high = threshold - self.uncertainty_band, threshold + self.uncertainty_band
        consult = (p_txt >= low) & (p_txt <= high)

        if not np.any(consult):
            return np.vstack([1 - p_txt, p_txt]).T

        p_out = p_txt.copy()
        idx = np.where(consult)[0]

        if self.combine_strategy == "blend":
            p_num, _ = self._p_num([titles[i] for i in idx], [bodies[i] for i in idx])
            p_blend = p_txt[idx] * (1 - self.numeric_scale) + p_num * self.numeric_scale
            p_out[idx] = p_blend
        else:
            p_num, X_std = self._p_num([titles[i] for i in idx], [bodies[i] for i in idx])

            feat_index = {f: i for i, f in enumerate(FEATURE_ORDER)}

            def has_feat(std_row, name, raw_threshold_std=0.0):
                j = feat_index.get(name, None)
                if j is None:
                    return False
                return std_row[j] > raw_threshold_std

            shifts = []
            for r in range(X_std.shape[0]):
                row = X_std[r]
                shift = 0.0
                if has_feat(row, "suspicious_tld", raw_threshold_std=0.5):
                    shift += self.shift_suspicious_tld
                if has_feat(row, "external_link_count", raw_threshold_std=0.5):
                    shift += self.shift_many_links
                if not has_feat(row, "all_caps_ratio", raw_threshold_std=0.0):
                    shift += self.shift_calm_text
                shifts.append(shift)

            thr_eff = np.clip(threshold + np.array(shifts), 0.05, 0.95)
            self.last_thr_eff = (idx, thr_eff)

        return np.vstack([1 - p_out, p_out]).T

    def predict_proba_base(
        self, titles, bodies, threshold: float | None = None
    ):
        return self._predict_proba_base(titles, bodies, threshold=threshold)

    def predict_proba(
        self, titles, bodies, threshold: float | None = None
    ):
        if threshold is None:
            threshold = float(getattr(self, "numeric_assist_center", 0.5))
        probs = self._predict_proba_base(titles, bodies, threshold=threshold)
        if self.calibrator is not None:
            try:
                spam_probs = probs[:, self._i_spam]
                calibrated = self.calibrator(spam_probs)
                calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
                probs[:, self._i_spam] = calibrated
                probs[:, 1 - self._i_spam] = 1.0 - calibrated
            except Exception:
                # If calibration fails, fall back to base probabilities.
                pass
        return probs

    def set_calibration(self, calibrator: Callable[[np.ndarray], np.ndarray] | None) -> None:
        self.calibrator = calibrator

    def predict(self, titles, bodies, threshold: float | None = None):
        if threshold is None:
            threshold = float(getattr(self, "numeric_assist_center", 0.5))
        probs = self.predict_proba(titles, bodies, threshold=threshold)[:, self._i_spam]
        thr = np.full_like(probs, threshold, dtype=float)
        if self.combine_strategy == "threshold_shift" and self.last_thr_eff is not None:
            idxs, thr_eff = self.last_thr_eff
            thr[idxs] = thr_eff
        labels = np.where(probs >= thr, "spam", "safe")
        return labels

    def predict_logit(self, texts: Sequence[str] | str) -> np.ndarray:
        """Return raw logit scores from the text-only head before calibration."""

        if self.lr_text_base is None and self.lr_text is None:
            raise RuntimeError("Text model not trained")

        if isinstance(texts, str):
            texts = [texts]

        embeddings = encode_texts(list(texts))

        if self.lr_text_base is not None:
            decision = self.lr_text_base.decision_function(embeddings)
            decision = np.asarray(decision, dtype=float)
            if decision.ndim == 2 and decision.shape[1] > 1:
                decision = decision[:, self._i_spam]
            return decision.reshape(-1)

        if self.lr_text is None:
            raise RuntimeError("Text model not trained")

        probs = self.lr_text.predict_proba(embeddings)[:, self._i_spam]
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        logits = np.log(probs / (1.0 - probs))
        return logits.reshape(-1)

    def audit_numeric_interplay(self, titles, bodies) -> dict:
        """Quantify how often numeric cues assisted the text head during CV."""

        threshold = float(getattr(self, "numeric_assist_center", 0.5))

        if titles is None or bodies is None:
            return {"pct_consulted": 0.0, "avg_threshold_shift": 0.0, "avg_prob_blend_weight": 0.0}

        n_total = len(titles)
        if n_total == 0:
            return {"pct_consulted": 0.0, "avg_threshold_shift": 0.0, "avg_prob_blend_weight": 0.0}

        idx_val = getattr(self, "_cv_idx_val", None)
        if idx_val is not None and len(idx_val) > 0:
            safe_idx = [i for i in idx_val if 0 <= int(i) < n_total]
        else:
            safe_idx = list(range(n_total))

        if not safe_idx:
            return {"pct_consulted": 0.0, "avg_threshold_shift": 0.0, "avg_prob_blend_weight": 0.0}

        sub_titles = [titles[int(i)] for i in safe_idx]
        sub_bodies = [bodies[int(i)] for i in safe_idx]

        p_txt = self._p_text(sub_titles, sub_bodies)
        low, high = threshold - self.uncertainty_band, threshold + self.uncertainty_band
        consult_mask = (p_txt >= low) & (p_txt <= high)

        consulted_count = int(np.sum(consult_mask))
        pct_consulted = 100.0 * consulted_count / max(1, len(sub_titles))

        result = {
            "pct_consulted": float(pct_consulted),
            "avg_threshold_shift": 0.0,
            "avg_prob_blend_weight": 0.0,
        }

        if consulted_count == 0:
            return result

        idx_consult = np.where(consult_mask)[0]
        consult_titles = [sub_titles[i] for i in idx_consult]
        consult_bodies = [sub_bodies[i] for i in idx_consult]

        if self.combine_strategy == "blend":
            p_num, _ = self._p_num(consult_titles, consult_bodies)
            p_blend = p_txt[idx_consult] * (1 - self.numeric_scale) + p_num * self.numeric_scale
            avg_delta = float(np.mean(p_blend - p_txt[idx_consult]))
            result["avg_prob_blend_weight"] = avg_delta
        else:
            p_num, X_std = self._p_num(consult_titles, consult_bodies)
            _ = p_num  # unused but keeps signature consistent

            feat_index = {f: i for i, f in enumerate(FEATURE_ORDER)}

            def has_feat(std_row, name, raw_threshold_std=0.0):
                j = feat_index.get(name, None)
                if j is None:
                    return False
                return std_row[j] > raw_threshold_std

            shifts = []
            for r in range(X_std.shape[0]):
                row = X_std[r]
                shift = 0.0
                if has_feat(row, "suspicious_tld", raw_threshold_std=0.5):
                    shift += self.shift_suspicious_tld
                if has_feat(row, "external_link_count", raw_threshold_std=0.5):
                    shift += self.shift_many_links
                if not has_feat(row, "all_caps_ratio", raw_threshold_std=0.0):
                    shift += self.shift_calm_text
                shifts.append(shift)

            avg_shift = float(np.mean(shifts)) if shifts else 0.0
            result["avg_threshold_shift"] = avg_shift

        return result

