from __future__ import annotations
from typing import Dict, Tuple
import pandas as pd

VALID_LABELS = {"spam", "safe"}

def _normalize_label(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.strip().lower()
    if x in {"ham", "legit", "legitimate"}:
        return "safe"
    return x

def _validate_csv_schema(df: pd.DataFrame) -> tuple[bool, str]:
    required = {"title", "body", "label"}
    missing = required - set(map(str.lower, df.columns))
    if missing:
        return False, f"Missing required columns: {', '.join(sorted(missing))}"
    return True, ""
