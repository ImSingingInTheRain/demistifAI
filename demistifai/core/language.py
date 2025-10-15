from __future__ import annotations
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Tuple, List
import streamlit as st

# Optional dependency handling
try:
    from langdetect import DetectorFactory as _LangDetectFactory
    from langdetect import LangDetectException as _LangDetectException
    from langdetect import detect as _langdetect_detect
    _LangDetectFactory.seed = 0
    HAS_LANGDETECT = True
except Exception:  # pragma: no cover
    _langdetect_detect = None
    _LangDetectException = Exception  # type: ignore
    HAS_LANGDETECT = False

def _detect_language_code(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if HAS_LANGDETECT and _langdetect_detect is not None:
        try:
            return str(_langdetect_detect(text)).lower()
        except _LangDetectException:
            return ""
        except Exception:
            return ""
    return "en"

def summarize_language_mix(texts: Iterable[str], top_k: int = 3) -> Dict[str, Any]:
    if not HAS_LANGDETECT:
        return {"available": False, "top": [], "total": 0, "other": 0.0, "counts": Counter()}
    counts: Counter[str] = Counter()
    detected_total = 0
    for raw_text in texts:
        code = _detect_language_code(raw_text)
        if not code:
            continue
        counts[code] += 1
        detected_total += 1
    if detected_total == 0:
        return {"available": True, "top": [], "total": 0, "other": 0.0, "counts": counts}
    top_items = counts.most_common(max(1, int(top_k)))
    top = [(lang, count / detected_total) for lang, count in top_items]
    covered = sum(share for _, share in top)
    other_share = max(0.0, 1.0 - covered)
    return {"available": True, "top": top, "total": detected_total, "other": other_share, "counts": counts}

def format_language_mix_summary(label: str, mix: Optional[Dict[str, Any]]) -> str:
    if not mix or not mix.get("available"):
        return "Lang mix: unknown"
    total = int(mix.get("total", 0))
    top = list(mix.get("top", []))
    if total <= 0 or not top:
        return f"Lang mix ({label}): —."
    parts = [f"{lang} {share * 100:.0f}%" for lang, share in top]
    other_share = float(mix.get("other", 0.0))
    if other_share > 0.005:
        parts.append(f"other {other_share * 100:.0f}%")
    return f"Lang mix ({label}): {', '.join(parts)}."

def render_language_mix_chip_rows(train_mix: Optional[Dict[str, Any]], test_mix: Optional[Dict[str, Any]]) -> None:
    if not train_mix and not test_mix:
        st.caption("Lang mix: unknown")
        return
    train_available = bool(train_mix and train_mix.get("available"))
    test_available = bool(test_mix and test_mix.get("available"))
    if not train_available and not test_available:
        st.caption("Lang mix: unknown")
        return
    col_train, col_test = st.columns(2)
    _render_language_mix_column(col_train, "Train", train_mix)
    _render_language_mix_column(col_test, "Test", test_mix)

def _render_language_mix_column(container, title: str, mix: Optional[Dict[str, Any]]) -> None:
    container.markdown(f"**{title} language mix**")
    if not mix or not mix.get("available"):
        container.caption("Unknown (language detector unavailable).")
        return
    total = int(mix.get("total", 0))
    top = list(mix.get("top", []))
    if total <= 0 or not top:
        container.caption("No detected language (blank emails).")
        return
    chip_parts = ['<div style="display:flex;flex-wrap:wrap;gap:0.35rem;margin-top:0.35rem;">']
    for lang, share in top:
        chip_parts.append(
            "<span style=\"background:rgba(49,51,63,0.08);color:rgba(15,23,42,0.85);"
            "border-radius:999px;padding:0.2rem 0.6rem;font-size:0.75rem;font-weight:600;\">"
            f"{lang.upper()} {share * 100:.0f}%</span>"
        )
    other_share = float(mix.get("other", 0.0))
    if other_share > 0.005:
        chip_parts.append(
            "<span style=\"background:rgba(49,51,63,0.08);color:rgba(15,23,42,0.65);"
            "border-radius:999px;padding:0.2rem 0.6rem;font-size:0.75rem;font-weight:600;\">"
            f"OTHER {other_share * 100:.0f}%</span>"
        )
    chip_parts.append("</div>")
    container.markdown("".join(chip_parts), unsafe_allow_html=True)
    container.caption(f"n={total}")
