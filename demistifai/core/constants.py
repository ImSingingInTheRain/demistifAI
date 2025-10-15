from __future__ import annotations
from typing import Dict, List, Tuple

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
