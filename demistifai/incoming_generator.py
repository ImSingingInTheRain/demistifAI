"""Synthetic incoming email generator for unlabeled inbox streams."""

from __future__ import annotations

import random
import string
from typing import Dict, List

from .constants import BRANDS, COURIERS, DATASET_LEGIT_DOMAINS, DATASET_SUSPICIOUS_TLDS, URGENCY

SUSPICIOUS_TLDS = DATASET_SUSPICIOUS_TLDS
LEGIT_DOMAINS = DATASET_LEGIT_DOMAINS


def _spam_link(rng: random.Random) -> str:
    name = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(6, 12)))
    tld = rng.choice(SUSPICIOUS_TLDS)
    path = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 10)))
    scheme = rng.choice(["http://", "https://"])
    return f"{scheme}{name}{tld}/{path}"


def _safe_link(rng: random.Random) -> str:
    host = rng.choice(LEGIT_DOMAINS)
    path = rng.choice(["/docs", "/policies", "/training", "/tickets", "/files", "/benefits"])
    return f"https://{host}{path}"


def _maybe_caps(rng: random.Random, text: str, prob: float = 0.25) -> str:
    if rng.random() > prob:
        return text
    words = text.split()
    if not words:
        return text
    idx = rng.randrange(len(words))
    words[idx] = words[idx].upper()
    if rng.random() < 0.15 and len(words) > 2:
        j = rng.randrange(len(words))
        if j != idx:
            words[j] = words[j].upper()
    return " ".join(words)


def _spam_title_body(rng: random.Random) -> Dict[str, str]:
    archetype = rng.choice(
        [
            "payroll_hold",
            "account_reset",
            "delivery_fee",
            "invoice_wire",
            "prize_lottery",
            "crypto_double",
            "bonus_verify",
            "docu_phish",
            "refund_now",
            "mfa_disable",
            "tax_rebate",
        ]
    )
    brand = rng.choice(BRANDS)
    courier = rng.choice(COURIERS)
    urgency = rng.choice(URGENCY)

    if archetype == "payroll_hold":
        title = f"{urgency}: Verify your payroll to release deposit"
        body = (
            "Your salary is on hold. Confirm bank details at "
            f"{_spam_link(rng)} within 30 minutes to avoid delay."
        )
    elif archetype == "account_reset":
        title = f"Password will expire — {urgency.lower()}"
        body = (
            f"Reset your password here: {_spam_link(rng)}. "
            "Failure to act may lock your account."
        )
    elif archetype == "delivery_fee":
        title = f"{courier} delivery notice: customs fee required"
        body = f"Your parcel is pending. Pay a small fee at {_spam_link(rng)} to schedule a new slot."
    elif archetype == "invoice_wire":
        title = "Payment overdue — settle immediately"
        body = "Service interruption imminent. Wire funds to the account in the attachment today."
    elif archetype == "prize_lottery":
        title = "WIN a FREE gift — final eligibility"
        body = f"Congratulations! Complete the short form at {_spam_link(rng)} to claim your reward."
    elif archetype == "crypto_double":
        title = "Crypto opportunity: double your balance"
        body = f"Transfer funds and we’ll return 2× within 24h. Start at {_spam_link(rng)}."
    elif archetype == "bonus_verify":
        title = "HR update: bonus identity check"
        body = f"Confirm your identity using your national ID and card CVV at {_spam_link(rng)}."
    elif archetype == "docu_phish":
        title = f"{brand}: You received a secure document"
        body = f"Open the external portal at {_spam_link(rng)} and log in to view the document."
    elif archetype == "refund_now":
        title = "Refund available — action needed"
        body = f"We owe you money. Submit IBAN and CVV at {_spam_link(rng)} to receive payment."
    elif archetype == "mfa_disable":
        title = "Two-factor disabled — reactivate now"
        body = f"We turned off MFA. Re-enable at {_spam_link(rng)} (login required)."
    else:  # tax_rebate
        title = "Tax rebate waiting — confirm identity"
        body = f"Claim your rebate by verifying bank access at {_spam_link(rng)}."

    title = _maybe_caps(rng, title, prob=0.35)
    if rng.random() < 0.25:
        title += " !!"
    if rng.random() < 0.30:
        body += f" More info: {_spam_link(rng)}"
    return {"title": title, "body": body}


def _safe_title_body(rng: random.Random) -> Dict[str, str]:
    archetype = rng.choice(
        [
            "meeting",
            "policy_update",
            "hr_workday",
            "it_maintenance",
            "invoice_legit",
            "travel",
            "training",
            "vendor_ap",
            "security_advice",
            "delivery_tracking",
        ]
    )
    brand = rng.choice(BRANDS)
    courier = rng.choice(COURIERS)

    if archetype == "meeting":
        title = "Team meeting moved to 14:00"
        body = "Join via the usual Teams link. Agenda: KPIs, risks, roadmap."
    elif archetype == "policy_update":
        title = "Policy update: remote work guidelines"
        body = "Please review the updated policy on the intranet. " + _safe_link(rng)
    elif archetype == "hr_workday":
        title = "Workday: benefits enrollment opens"
        body = "Make your selections in Workday before month end. No personal info by email."
    elif archetype == "it_maintenance":
        title = "IT maintenance window"
        body = "Patching on Saturday 22:00–23:30 CET. Expect brief reboots; no action required."
    elif archetype == "invoice_legit":
        title = "Invoice attached — Accounts Payable"
        body = "Please find the invoice PDF attached. PO referenced; no payment info requested."
    elif archetype == "travel":
        title = "Travel itinerary update"
        body = "Platform change noted. PDF itinerary attached; bookings remain via the internal tool."
    elif archetype == "training":
        title = "Mandatory training assigned"
        body = "Complete the e-learning module by next Friday. Materials on the LMS. " + _safe_link(rng)
    elif archetype == "vendor_ap":
        title = "AP reminder — PO mismatch"
        body = "Please correct the PO reference in the invoice metadata on SharePoint. " + _safe_link(rng)
    elif archetype == "security_advice":
        title = f"Security advisory — {brand} tips"
        body = "Review the guidance on the internal portal. No external logins required. " + _safe_link(rng)
    else:  # delivery_tracking
        title = f"{courier} tracking: package out for delivery"
        body = f"Your parcel is scheduled today. Track on the official {courier} site (no payment)."

    title = _maybe_caps(rng, title, prob=0.05)
    return {"title": title, "body": body}


def generate_incoming_batch(n: int = 30, seed: int = 123, spam_ratio: float = 0.3) -> List[Dict[str, str]]:
    """Generate an unlabeled batch of incoming emails for the inbox stream."""

    if not 0.0 <= spam_ratio <= 1.0:
        raise ValueError("spam_ratio must be between 0 and 1")

    rng = random.Random(seed)
    n_spam = int(round(n * spam_ratio))
    n_safe = n - n_spam
    items: List[Dict[str, str]] = []

    for _ in range(n_spam):
        items.append(_spam_title_body(rng))
    for _ in range(n_safe):
        items.append(_safe_title_body(rng))

    rng.shuffle(items)
    return items


__all__ = ["generate_incoming_batch"]
