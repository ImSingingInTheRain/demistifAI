from __future__ import annotations
from typing import Dict, List, Tuple

from demistifai.constants import (
    STAGE_BY_KEY as _STAGE_BY_KEY,
    STAGES as _STAGES,
    URGENCY as _URGENCY,
)
from demistifai.modeling import URGENCY_TERMS as _URGENCY_TERMS

TOKEN_POLICY: Dict[str, str] = {
    "email": "{{EMAIL}}",
    "phone": "{{PHONE}}",
    "iban": "{{IBAN}}",
    "card16": "{{CARD_16}}",
    "otp6": "{{OTP_6}}",
    "url": "{{URL_SUSPICIOUS}}",
}

PII_DISPLAY_LABELS: List[Tuple[str, str]] = [
    ("iban", "IBAN"),
    ("credit_card", "Card"),
    ("email", "Emails"),
    ("phone", "Phones"),
    ("otp6", "OTPs"),
    ("url", "Suspicious URLs"),
]

PII_CHIP_CONFIG: List[Tuple[str, str, str]] = [
    ("credit_card", "💳", "Credit card"),
    ("iban", "🏦", "IBAN"),
    ("email", "📧", "Emails"),
    ("phone", "☎️", "Phones"),
    ("otp6", "🔐", "OTPs"),
    ("url", "🌐", "Suspicious URLs"),
]

STAGES = _STAGES
STAGE_BY_KEY = _STAGE_BY_KEY
URGENCY = _URGENCY
URGENCY_TERMS = _URGENCY_TERMS

__all__ = [
    "TOKEN_POLICY",
    "PII_DISPLAY_LABELS",
    "PII_CHIP_CONFIG",
    "STAGES",
    "STAGE_BY_KEY",
    "URGENCY",
    "URGENCY_TERMS",
]
