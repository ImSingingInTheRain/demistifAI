from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd

def _export_batch_df(rows: Optional[List[Dict[str, Any]]]) -> pd.DataFrame:
    base_cols = ["title", "body", "pred", "p_spam", "p_safe", "action", "routed_to"]
    if not rows:
        return pd.DataFrame(columns=base_cols)
    df_rows = pd.DataFrame(rows)
    for col in base_cols:
        if col not in df_rows.columns:
            df_rows[col] = None
    return df_rows[base_cols]
