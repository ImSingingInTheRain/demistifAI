"""Modeling utilities for demistifAI."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
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
]

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

@st.cache_resource(show_spinner=False)
def get_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    # Downloaded once and cached by Streamlit
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def encode_texts(texts: list, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    model = get_encoder(model_name)
    # Normalize embeddings for stability
    embs = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32)


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
    try:
        w = model.lr.coef_[0][-n_num:]
    except Exception:
        return []
    contrib = z * w
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
    """
    Frozen sentence-embedding encoder + small numeric features (standardized) concatenated,
    then LogisticRegression (balanced).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.lr = LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            n_jobs=None,
        )
        self.scaler = StandardScaler()
        self.classes_ = None
        self.base_num_coefs_: Optional[np.ndarray] = None
        self.numeric_adjustments_: np.ndarray = np.zeros(len(FEATURE_ORDER), dtype=float)

    def _embed(self, texts: list[str]) -> np.ndarray:
        return encode_texts(texts, model_name=self.model_name)

    def _feats(self, titles: List[str], bodies: List[str]) -> np.ndarray:
        return features_matrix(titles, bodies)

    def fit(self, X_titles: List[str], X_bodies: List[str], y: List[str]):
        texts = [(t or "") + "\n" + (b or "") for t, b in zip(X_titles, X_bodies)]
        X_emb = self._embed(texts)
        X_f = self._feats(X_titles, X_bodies)
        X_f_std = self.scaler.fit_transform(X_f)
        X_cat = np.concatenate([X_emb, X_f_std], axis=1)
        self.lr.fit(X_cat, y)
        self.classes_ = list(self.lr.classes_)
        n_num = len(FEATURE_ORDER)
        self.base_num_coefs_ = self.lr.coef_[0][-n_num:].copy()
        self.numeric_adjustments_ = np.zeros_like(self.base_num_coefs_)
        return self

    def _prep(self, X_titles: List[str], X_bodies: List[str]) -> np.ndarray:
        texts = [(t or "") + "\n" + (b or "") for t, b in zip(X_titles, X_bodies)]
        X_emb = self._embed(texts)
        X_f = self._feats(X_titles, X_bodies)
        X_f_std = self.scaler.transform(X_f)
        return np.concatenate([X_emb, X_f_std], axis=1)

    def predict(self, X_titles: List[str], X_bodies: List[str]) -> np.ndarray:
        X = self._prep(X_titles, X_bodies)
        return self.lr.predict(X)

    def predict_proba(self, X_titles: List[str], X_bodies: List[str]) -> np.ndarray:
        X = self._prep(X_titles, X_bodies)
        return self.lr.predict_proba(X)

    # Convenience for introspection of numeric feature coefficients
    def numeric_feature_details(self) -> pd.DataFrame:
        """Return dataframe with standardized weights + training stats."""

        if not hasattr(self.lr, "coef_"):
            raise RuntimeError("Model is not trained")

        n_total = self.lr.coef_.shape[1]
        n_num = len(FEATURE_ORDER)
        if n_total < n_num:
            raise RuntimeError("Logistic regression is missing numeric feature coefficients")

        current_coefs = self.lr.coef_[0][-n_num:]
        means = getattr(self.scaler, "mean_", np.zeros(n_num))
        stds = getattr(self.scaler, "scale_", np.ones(n_num))
        base = self.base_num_coefs_ if self.base_num_coefs_ is not None else current_coefs.copy()
        adjustments = self.numeric_adjustments_ if hasattr(self, "numeric_adjustments_") else np.zeros_like(current_coefs)

        df = pd.DataFrame(
            {
                "feature": FEATURE_ORDER,
                "base_weight_per_std": base.astype(float),
                "user_adjustment": adjustments.astype(float),
                "weight_per_std": current_coefs.astype(float),
                "train_mean": means.astype(float),
                "train_std": stds.astype(float),
            }
        )

        # Odds change for a +1 standard deviation move in the original (unscaled) feature
        df["odds_multiplier_per_std"] = np.exp(df["weight_per_std"])

        # Translate back to effect per raw-unit (avoid division by ~0)
        safe_std = df["train_std"].replace(0, np.nan)
        df["weight_per_unit"] = df["weight_per_std"] / safe_std

        return df

    def numeric_feature_coefs(self) -> Dict[str, float]:
        details = self.numeric_feature_details()
        return dict(zip(details["feature"], details["weight_per_std"]))

    def apply_numeric_adjustments(self, adjustments: Dict[str, float]):
        if self.base_num_coefs_ is None:
            return
        ordered = np.array([adjustments.get(feat, 0.0) for feat in FEATURE_ORDER], dtype=float)
        self.numeric_adjustments_ = ordered
        self.lr.coef_[0][-len(FEATURE_ORDER):] = self.base_num_coefs_ + ordered

