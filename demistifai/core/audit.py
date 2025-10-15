from __future__ import annotations
from typing import Any, Dict, Optional
from datetime import datetime
import streamlit as st

def _append_audit(event: str, details: Optional[Dict[str, Any]] = None) -> None:
    entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), "event": event}
    if details:
        entry["details"] = details
    log = st.session_state.setdefault("use_audit_log", [])
    log.append(entry)
