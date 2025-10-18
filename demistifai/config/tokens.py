"""Personal data masking and display policies."""

TOKEN_POLICY = {
    "email": "{{EMAIL}}",
    "phone": "{{PHONE}}",
    "iban": "{{IBAN}}",
    "card16": "{{CARD_16}}",
    "otp6": "{{OTP_6}}",
    "url": "{{URL_SUSPICIOUS}}",
}

PII_DISPLAY_LABELS = [
    ("iban", "IBAN"),
    ("credit_card", "Card"),
    ("email", "Emails"),
    ("phone", "Phones"),
    ("otp6", "OTPs"),
    ("url", "Suspicious URLs"),
]

PII_CHIP_CONFIG = [
    ("credit_card", "💳", "Credit card"),
    ("iban", "🏦", "IBAN"),
    ("email", "📧", "Emails"),
    ("phone", "☎️", "Phones"),
    ("otp6", "🔐", "OTPs"),
    ("url", "🌐", "Suspicious URLs"),
]

__all__ = ["TOKEN_POLICY", "PII_DISPLAY_LABELS", "PII_CHIP_CONFIG"]
