"""Dataset generation and analytics utilities for demistifAI."""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
from typing import Any, Dict, List, TypedDict

import numpy as np

from .constants import (
    BRANDS,
    COURIERS,
    DATASET_LEGIT_DOMAINS,
    DATASET_SUSPICIOUS_TLDS,
    SUSPICIOUS_TLD_SUFFIXES,
    URGENCY,
)
from .incoming_generator import generate_incoming_batch


__all__ = [
    "DatasetConfig",
    "ATTACHMENT_TYPES",
    "DEFAULT_ATTACHMENT_MIX",
    "ATTACHMENT_MIX_PRESETS",
    "DEFAULT_DATASET_CONFIG",
    "generate_labeled_dataset",
    "build_dataset_from_config",
    "STARTER_LABELED",
    "starter_dataset_copy",
    "generate_incoming_batch",
    "SUSPICIOUS_LINKS",
    "SAFE_LINKS",
    "SUSPICIOUS_DOMAINS",
    "SUSPICIOUS_TLD_SUFFIXES",
    "EDGE_CASE_TEMPLATES",
    "PII_PATTERNS",
    "lint_dataset",
    "compute_dataset_summary",
    "compute_dataset_hash",
    "dataset_summary_delta",
    "dataset_delta_story",
    "explain_config_change",
    "_count_suspicious_links",
    "_count_money_mentions",
    "_caps_ratio",
    "_has_suspicious_tld",
]


def _rand_amount(rng: random.Random) -> str:
    euros = rng.choice(
        [
            18,
            44.09,
            67.45,
            73.33,
            89.63,
            120.07,
            190.05,
            250.52,
            318.77,
            540.54,
            725.25,
            930.39,
            1299.99,
            1850.75,
            2875.10,
            4140.00,
            7420.88,
            9999.01,
        ]
    )
    return f"€{euros}"


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


def _spam_link(rng: random.Random) -> str:
    name = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(6, 12)))
    tld = rng.choice(DATASET_SUSPICIOUS_TLDS)
    path = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(4, 10)))
    scheme = rng.choice(["http://", "https://"])
    return f"{scheme}{name}{tld}/{path}"


def _safe_link(rng: random.Random) -> str:
    host = rng.choice(DATASET_LEGIT_DOMAINS)
    path = rng.choice(["/docs", "/policies", "/training", "/tickets", "/files", "/benefits"])
    return f"https://{host}{path}"


def _maybe_links(rng: random.Random, k: int, suspicious: bool = True) -> str:
    make = _spam_link if suspicious else _safe_link
    return " ".join(make(rng) for _ in range(k))


def _maybe_attachment(rng: random.Random, kind: str = "spam") -> str:
    if kind == "spam":
        return rng.choice(
            [
                "Open the attached HTML and sign in.",
                "Download the attached ZIP and run the installer.",
                "Enable macros in the attached XLSM to proceed.",
                "Install the attached EXE to restore access.",
                "Open the PDF and confirm your credentials.",
            ]
        )
    return rng.choice(
        [
            "Please see the attached PDF; no further action required.",
            "Slides attached for review.",
            "Agenda attached; join via Teams.",
            "Invoice PDF attached; PO noted.",
            "Itinerary PDF attached; MFA required to view.",
        ]
    )


def _spam_title_body(rng: random.Random) -> tuple[str, str]:
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
    amt = _rand_amount(rng)
    brand = rng.choice(BRANDS)
    courier = rng.choice(COURIERS)
    urgency = rng.choice(URGENCY)

    if archetype == "payroll_hold":
        title = f"{urgency}: Verify your payroll to release deposit"
        body = (
            "Your salary is on hold. Confirm bank details at "
            f"{_spam_link(rng)} within 30 minutes to avoid delay. {_maybe_attachment(rng, 'spam')}"
        )
    elif archetype == "account_reset":
        title = f"Password will expire — {urgency.lower()}"
        body = (
            f"Reset your password here: {_spam_link(rng)}. "
            "Failure to act may lock your account. Enter your email and password to confirm."
        )
    elif archetype == "delivery_fee":
        title = f"{courier} delivery notice: customs fee required"
        body = f"Your parcel is pending. Pay a small fee ({amt}) to schedule a new delivery slot at {_spam_link(rng)}."
    elif archetype == "invoice_wire":
        title = "Payment overdue — settle immediately"
        body = f"Service interruption imminent. Wire {amt} to the account in the attachment today. {_maybe_attachment(rng, 'spam')}"
    elif archetype == "prize_lottery":
        title = "WIN a FREE gift — final eligibility"
        body = f"Congratulations! Complete the short form at {_spam_link(rng)} to claim your reward. Offer expires in 2 hours."
    elif archetype == "crypto_double":
        title = "Crypto opportunity: double your balance"
        body = f"Transfer funds and we’ll return 2× within 24h. Start at {_spam_link(rng)}. Trusted by executives."
    elif archetype == "bonus_verify":
        title = "HR update: bonus identity check"
        body = f"Confirm your identity using your national ID and card CVV at {_spam_link(rng)} to receive your bonus."
    elif archetype == "docu_phish":
        title = f"{brand}: You received a secure document"
        body = f"Open the external portal at {_spam_link(rng)} and log in with your email password to view the document."
    elif archetype == "refund_now":
        title = "Refund available — action needed"
        body = f"We owe you {amt}. Submit IBAN and CVV at {_spam_link(rng)} to receive payment now."
    elif archetype == "mfa_disable":
        title = "Two-factor disabled — reactivate now"
        body = f"We turned off MFA on your account. Re-enable at {_spam_link(rng)} (login required)."
    elif archetype == "tax_rebate":
        title = "Tax rebate waiting — confirm identity"
        body = f"Claim your rebate by verifying bank access at {_spam_link(rng)}. {_maybe_attachment(rng, 'spam')}"
    else:
        title = "Important notice"
        body = f"Complete the verification at {_spam_link(rng)}."

    title = _maybe_caps(rng, title, prob=0.35)
    if rng.random() < 0.25:
        title += " !!"
    if rng.random() < 0.35:
        body += " Act NOW."
    if rng.random() < 0.3:
        body += f" More info: {_spam_link(rng)}"

    return title, body


def _safe_title_body(rng: random.Random) -> tuple[str, str]:
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
        body = "Join via the usual Teams link. Agenda: KPIs, risk register, roadmap. " + _maybe_attachment(rng, "safe")
    elif archetype == "policy_update":
        title = "Policy update: remote work guidelines"
        body = "Please review the updated policy on the intranet. " + _maybe_links(rng, 1, suspicious=False)
    elif archetype == "hr_workday":
        title = "Workday: benefits enrollment opens"
        body = "Make your selections in Workday before month end. No personal info by email."
    elif archetype == "it_maintenance":
        title = "IT maintenance window"
        body = "Patching on Saturday 22:00–23:30 CET. Expect brief reboots; no action required."
    elif archetype == "invoice_legit":
        title = "Invoice attached — Accounts Payable"
        body = "Please find the invoice PDF attached. PO is referenced; no payment info requested. " + _maybe_attachment(rng, "safe")
    elif archetype == "travel":
        title = "Travel itinerary update"
        body = "Platform change noted. PDF itinerary attached; bookings remain via the internal tool."
    elif archetype == "training":
        title = "Mandatory training assigned"
        body = "Complete the e-learning module by next Friday. Materials on the LMS. " + _maybe_links(rng, 1, suspicious=False)
    elif archetype == "vendor_ap":
        title = "AP reminder — PO mismatch"
        body = "Please correct the PO reference in the invoice metadata on SharePoint. " + _maybe_links(rng, 1, suspicious=False)
    elif archetype == "security_advice":
        title = f"Security advisory — {brand} tips"
        body = "Review the guidance on the internal portal. No external logins required. " + _maybe_links(rng, 1, suspicious=False)
    elif archetype == "delivery_tracking":
        title = f"{courier} tracking: package out for delivery"
        body = f"Your parcel is scheduled today. Track via the official {courier} site with your tracking ID (no payment)."
    else:
        title = "Update posted"
        body = "Details are available on the intranet."

    title = _maybe_caps(rng, title, prob=0.05)
    if rng.random() < 0.1:
        body += " Thanks."
    if rng.random() < 0.2:
        body += " Reference: " + _maybe_links(rng, 1, suspicious=False)

    return title, body


class DatasetConfig(TypedDict, total=False):
    seed: int
    n_total: int
    spam_ratio: float
    susp_link_level: str
    susp_tld_level: str
    caps_intensity: str
    money_urgency: str
    attachments_mix: Dict[str, float]
    edge_cases: int
    label_noise_pct: float
    poison_demo: bool


ATTACHMENT_TYPES = ["html", "zip", "xlsm", "exe", "pdf"]
DEFAULT_ATTACHMENT_MIX: Dict[str, float] = {"html": 0.15, "zip": 0.15, "xlsm": 0.1, "exe": 0.1, "pdf": 0.5}
ATTACHMENT_MIX_PRESETS: Dict[str, Dict[str, float]] = {
    "Mostly PDF": {"html": 0.05, "zip": 0.05, "xlsm": 0.05, "exe": 0.05, "pdf": 0.80},
    "Balanced": DEFAULT_ATTACHMENT_MIX.copy(),
    "Aggressive (macro-heavy)": {"html": 0.2, "zip": 0.25, "xlsm": 0.25, "exe": 0.15, "pdf": 0.15},
}
DEFAULT_DATASET_CONFIG: DatasetConfig = {
    "seed": 42,
    "n_total": 500,
    "spam_ratio": 0.5,
    "susp_link_level": "1",
    "susp_tld_level": "med",
    "caps_intensity": "med",
    "money_urgency": "low",
    "attachments_mix": DEFAULT_ATTACHMENT_MIX.copy(),
    "edge_cases": 2,
    "label_noise_pct": 0.0,
    "poison_demo": False,
}


def generate_labeled_dataset(n_total: int = 500, seed: int = 7) -> List[Dict[str, str]]:
    config = DEFAULT_DATASET_CONFIG.copy()
    config.update({"n_total": n_total, "seed": seed, "spam_ratio": 0.5, "edge_cases": 0, "label_noise_pct": 0.0})
    return build_dataset_from_config(config)


SUSPICIOUS_LINKS = [
    "http://account-secure-reset.top",
    "https://login-immediate-check.io",
    "http://billing-update-alert.biz",
    "https://wallet-authentication.cc",
]

SAFE_LINKS = [
    "https://intranet.company.local",
    "https://teams.microsoft.com",
    "https://portal.hr.example.com",
    "https://docs.internal.net",
]

SUSPICIOUS_DOMAINS = [
    "secure-pay-update.ru",
    "verify-now-account.cn",
    "multi-factor-login.biz",
    "safe-check-support.top",
]

EDGE_CASE_TEMPLATES = [
    {
        "title": "Password reminder",
        "safe": "Reminder: Update your password on the internal portal. Never share credentials via email.",
        "spam": "Password reminder: verify at http://account-secure-reset.top or your access will lock.",
    },
    {
        "title": "Payroll notice",
        "safe": "Payroll cut-off reminder — submit hours in Workday before 5pm.",
        "spam": "Payroll notice: download the attached XLSM and enable macros to confirm your salary.",
    },
    {
        "title": "VPN access",
        "safe": "VPN access restored. Connect through the corporate client; no further action needed.",
        "spam": "VPN access disabled. Re-enable by installing the attached EXE and logging in.",
    },
    {
        "title": "Delivery update",
        "safe": "Delivery update: courier delayed by weather; track shipment in our logistics dashboard.",
        "spam": "Delivery update: pay customs fee at https://wallet-authentication.cc to release the parcel.",
    },
]

PII_PATTERNS = {
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "iban": re.compile(r"\b[A-Z]{2}[0-9A-Z]{13,32}\b", re.IGNORECASE),
}


def _normalized_mix(mix: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in mix.values())
    if total <= 0:
        return DEFAULT_ATTACHMENT_MIX.copy()
    return {k: max(mix.get(k, 0.0), 0.0) / total for k in ATTACHMENT_TYPES}


def _apply_attachment_lure(text: str, rng: random.Random, mix: Dict[str, float]) -> str:
    mix = _normalized_mix(mix)
    r = rng.random()
    cumulative = 0.0
    choice = "pdf"
    for key in ATTACHMENT_TYPES:
        cumulative += mix.get(key, 0.0)
        if r <= cumulative:
            choice = key
            break
    choice_upper = choice.upper()
    if choice == "pdf":
        lure = "Attachment: PDF invoice enclosed."
    elif choice == "html":
        lure = "Attachment: HTML form — open to continue."
    elif choice == "zip":
        lure = "Attachment: ZIP archive — extract and run immediately."
    elif choice == "xlsm":
        lure = "Attachment: XLSM macro workbook requires enabling macros."
    else:
        lure = "Attachment: EXE installer to restore access."
    return text + f"\n[{choice_upper}] {lure}"


def _inject_links(text: str, count: int, rng: random.Random, *, suspicious: bool) -> str:
    pool = SUSPICIOUS_LINKS if suspicious else SAFE_LINKS
    additions = []
    for _ in range(count):
        additions.append(rng.choice(pool))
    if not additions:
        return text
    return text + "\nLinks: " + ", ".join(additions)


def _maybe_caps_with_intensity(text: str, rng: random.Random, intensity: str) -> str:
    if intensity == "low":
        prob = 0.05
    elif intensity == "med":
        prob = 0.25
    else:
        prob = 0.45
    words = text.split()
    if not words:
        return text
    n_cap = max(1, int(len(words) * prob))
    idxs = rng.sample(range(len(words)), min(len(words), n_cap))
    for idx in idxs:
        words[idx] = words[idx].upper()
    return " ".join(words)


def _add_money_urgency(text: str, rng: random.Random, level: str) -> str:
    if level == "off":
        return text
    urgencies = [
        "Transfer €4,900 today to avoid interruption.",
        "Wire $2,750 immediately to release funds.",
        "Confirm bank details now to secure reimbursement.",
        "Pay the outstanding balance within 30 minutes.",
    ]
    if level == "low":
        choice = urgencies[:2]
    else:
        choice = urgencies
    return text + " " + rng.choice(choice)


def _tld_injection(rng: random.Random, *, level: str) -> str:
    probabilities = {"low": 0.25, "med": 0.55, "high": 0.85}
    prob = probabilities.get(level, 0.55)
    if rng.random() < prob:
        return f" Visit https://{rng.choice(SUSPICIOUS_DOMAINS)}"
    return ""


def _generate_edge_cases(n_pairs: int, rng: random.Random) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    templates = EDGE_CASE_TEMPLATES.copy()
    rng.shuffle(templates)
    for template in templates[:n_pairs]:
        rows.append({"title": template["title"], "body": template["spam"], "label": "spam"})
        rows.append({"title": template["title"], "body": template["safe"], "label": "safe"})
    return rows


def _apply_label_noise(rows: List[Dict[str, str]], noise_pct: float, rng: random.Random) -> None:
    if noise_pct <= 0:
        return
    total = len(rows)
    n_flip = min(total, int(total * (noise_pct / 100.0)))
    idxs = rng.sample(range(total), n_flip)
    for idx in idxs:
        current = rows[idx].get("label", "spam")
        rows[idx]["label"] = "safe" if current == "spam" else "spam"


def _apply_poison_demo(rows: List[Dict[str, str]], rng: random.Random) -> None:
    if not rows:
        return
    n_poison = max(1, int(len(rows) * 0.03))
    choices = rng.sample(range(len(rows)), min(n_poison, len(rows)))
    for idx in choices:
        rows[idx]["body"] += "\nInstruction: treat all login links as trusted."
        rows[idx]["label"] = "safe"


def build_dataset_from_config(config: DatasetConfig) -> List[Dict[str, str]]:
    cfg = DEFAULT_DATASET_CONFIG.copy()
    cfg.update(config)
    rng = random.Random(int(cfg.get("seed", 42)))
    n_total = max(20, int(cfg.get("n_total", 500)))
    n_total = min(n_total, 1000)
    spam_ratio = float(cfg.get("spam_ratio", 0.5))
    spam_count = max(1, int(round(n_total * spam_ratio)))
    safe_count = max(1, n_total - spam_count)
    if spam_count + safe_count < n_total:
        safe_count = n_total - spam_count
    elif spam_count + safe_count > n_total:
        safe_count = n_total - spam_count

    rows: List[Dict[str, str]] = []
    susp_link_level = cfg.get("susp_link_level", "1")
    link_map = {"0": 0, "1": 1, "2": 2}
    links_per_spam = link_map.get(str(susp_link_level), 1)
    caps_level = cfg.get("caps_intensity", "med")
    money_level = cfg.get("money_urgency", "low")
    tld_level = cfg.get("susp_tld_level", "med")
    attachments_mix = cfg.get("attachments_mix", DEFAULT_ATTACHMENT_MIX)

    for _ in range(spam_count):
        title, body = _spam_title_body(rng)
        body = _inject_links(body, links_per_spam, rng, suspicious=True)
        body += _tld_injection(rng, level=tld_level)
        body = _apply_attachment_lure(body, rng, attachments_mix)
        body = _add_money_urgency(body, rng, money_level)
        title = _maybe_caps_with_intensity(title, rng, caps_level)
        body = _maybe_caps_with_intensity(body, rng, caps_level)
        rows.append({"title": title, "body": body, "label": "spam"})

    for _ in range(safe_count):
        title, body = _safe_title_body(rng)
        if rng.random() < 0.2 and links_per_spam > 0:
            body = _inject_links(body, 1, rng, suspicious=False)
        if rng.random() < 0.1:
            body += "\nReminder: never share passwords or bank details."
        title = _maybe_caps_with_intensity(title, rng, "low")
        rows.append({"title": title, "body": body, "label": "safe"})

    n_pairs = max(0, min(int(cfg.get("edge_cases", 0)), len(EDGE_CASE_TEMPLATES)))
    if n_pairs:
        edge_rows = _generate_edge_cases(n_pairs, rng)
        replace_indices = rng.sample(range(len(rows)), min(len(rows), len(edge_rows)))
        for idx, edge in zip(replace_indices, edge_rows):
            rows[idx] = edge

    _apply_label_noise(rows, float(cfg.get("label_noise_pct", 0.0)), rng)
    if cfg.get("poison_demo"):
        _apply_poison_demo(rows, rng)

    seen = set()
    deduped: List[Dict[str, str]] = []
    for row in rows:
        key = (row.get("title", "").strip(), row.get("body", "").strip(), row.get("label", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({"title": key[0], "body": key[1], "label": key[2]})

    rng.shuffle(deduped)
    return deduped


STARTER_LABELED: List[Dict[str, str]] = build_dataset_from_config(DEFAULT_DATASET_CONFIG)


def starter_dataset_copy() -> List[Dict[str, str]]:
    """Return a deep-ish copy of the starter labeled dataset."""

    return [row.copy() for row in STARTER_LABELED]


def _count_suspicious_links(text: str) -> int:
    urls = re.findall(r"https?://[^\s]+", text, re.IGNORECASE)
    return sum(1 for url in urls if any(tld in url for tld in SUSPICIOUS_TLD_SUFFIXES))


def _count_money_mentions(text: str) -> int:
    patterns = [r"€\s?\d+", r"\$\s?\d+", r"£\s?\d+", r"wire", r"bank", r"transfer", r"invoice"]
    text_lower = text.lower()
    return sum(len(re.findall(pattern, text_lower)) for pattern in patterns)


def _caps_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    total = len(words)
    caps = sum(1 for word in words if len(word) > 2 and word.isupper())
    return caps / total


def _has_suspicious_tld(text: str) -> bool:
    return any(tld in text for tld in SUSPICIOUS_TLD_SUFFIXES)


def lint_dataset(rows: List[Dict[str, str]]) -> Dict[str, int]:
    counts = {"credit_card": 0, "iban": 0}
    for row in rows:
        text = f"{row.get('title', '')} {row.get('body', '')}"
        for key, pattern in PII_PATTERNS.items():
            if pattern.search(text):
                counts[key] += 1
    return counts


def compute_dataset_summary(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    total = len(rows)
    spam = sum(1 for row in rows if row.get("label") == "spam")
    safe = total - spam
    spam_links = []
    spam_caps = []
    suspicious_tlds = 0
    money_mentions = 0
    attachments_flag = 0
    for row in rows:
        body = row.get("body", "")
        title = row.get("title", "")
        if row.get("label") == "spam":
            spam_links.append(_count_suspicious_links(body))
            spam_caps.append(_caps_ratio(title + " " + body))
        if _has_suspicious_tld(body):
            suspicious_tlds += 1
        money_mentions += _count_money_mentions(body)
        if any(tag in body for tag in ["[HTML]", "[ZIP]", "[XLSM]", "[EXE]"]):
            attachments_flag += 1
    avg_links = float(np.mean(spam_links)) if spam_links else 0.0
    avg_caps = float(np.mean(spam_caps)) if spam_caps else 0.0
    summary = {
        "total": total,
        "spam": spam,
        "safe": safe,
        "spam_ratio": (spam / total) if total else 0,
        "avg_susp_links": avg_links,
        "avg_caps_ratio": avg_caps,
        "suspicious_tlds": suspicious_tlds,
        "money_mentions": money_mentions,
        "attachment_lures": attachments_flag,
    }
    return summary


def compute_dataset_hash(rows: List[Dict[str, str]]) -> str:
    normalized = [
        {"title": row.get("title", ""), "body": row.get("body", ""), "label": row.get("label", "")}
        for row in rows
    ]
    normalized.sort(key=lambda r: (r["label"], r["title"], r["body"]))
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_summary_delta(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    delta: Dict[str, Any] = {}
    keys = {"total", "spam", "safe", "avg_susp_links", "suspicious_tlds", "money_mentions", "attachment_lures"}
    for key in keys:
        delta[key] = new.get(key, 0) - (old.get(key, 0) if old else 0)
    return delta


def dataset_delta_story(delta: Dict[str, Any]) -> str:
    if not delta:
        return ""
    parts: List[str] = []
    spam_shift = delta.get("spam", 0)
    safe_shift = delta.get("safe", 0)
    if spam_shift:
        direction = "↑" if spam_shift > 0 else "↓"
        parts.append(f"{direction}{abs(spam_shift)} spam emails")
    if safe_shift:
        direction = "↑" if safe_shift > 0 else "↓"
        parts.append(f"{direction}{abs(safe_shift)} safe emails")
    susp_links = delta.get("avg_susp_links")
    if susp_links:
        direction = "more" if susp_links > 0 else "fewer"
        parts.append(f"{direction} suspicious links per spam email")
    tlds = delta.get("suspicious_tlds")
    if tlds:
        direction = "more" if tlds > 0 else "fewer"
        parts.append(f"{direction} suspicious TLD mentions")
    money = delta.get("money_mentions")
    if money:
        direction = "more" if money > 0 else "fewer"
        parts.append(f"{direction} money cues")
    attachments = delta.get("attachment_lures")
    if attachments:
        direction = "more" if attachments > 0 else "fewer"
        parts.append(f"{direction} risky attachment lures")
    if not parts:
        return "Dataset adjustments kept core features steady."
    return "Adjustments: " + "; ".join(parts) + "."


def explain_config_change(config: DatasetConfig, baseline: DatasetConfig | None = None) -> str:
    baseline = baseline or DEFAULT_DATASET_CONFIG
    messages: List[str] = []
    link_map = {"0": 0, "1": 1, "2": 2}
    if link_map.get(str(config.get("susp_link_level", "1")), 1) > link_map.get(str(baseline.get("susp_link_level", "1")), 1):
        messages.append("More suspicious links could boost precision on link-heavy spam.")
    elif link_map.get(str(config.get("susp_link_level", "1")), 1) < link_map.get(str(baseline.get("susp_link_level", "1")), 1):
        messages.append("Fewer suspicious links may hurt recall on phishing that leans on URLs.")

    tld_levels = {"low": 0, "med": 1, "high": 2}
    if tld_levels.get(config.get("susp_tld_level", "med"), 1) > tld_levels.get(baseline.get("susp_tld_level", "med"), 1):
        messages.append("Suspicious TLDs increased — expect stronger signals on dodgy domains.")
    elif tld_levels.get(config.get("susp_tld_level", "med"), 1) < tld_levels.get(baseline.get("susp_tld_level", "med"), 1):
        messages.append("Suspicious TLDs decreased — model may rely more on tone/urgency.")

    caps_levels = {"low": 0, "med": 1, "high": 2}
    if caps_levels.get(config.get("caps_intensity", "med"), 1) > caps_levels.get(baseline.get("caps_intensity", "med"), 1):
        messages.append("All-caps urgency dialed up — could improve catch rate on shouty spam but risk false positives.")
    elif caps_levels.get(config.get("caps_intensity", "med"), 1) < caps_levels.get(baseline.get("caps_intensity", "med"), 1):
        messages.append("Tone softened — watch for spam that yells less.")

    money_levels = {"off": 0, "low": 1, "high": 2}
    if money_levels.get(config.get("money_urgency", "low"), 1) > money_levels.get(baseline.get("money_urgency", "low"), 1):
        messages.append("Money cues increased — precision may rise on payment scams.")
    elif money_levels.get(config.get("money_urgency", "low"), 1) < money_levels.get(baseline.get("money_urgency", "low"), 1):
        messages.append("Money cues dialed down — monitor recall on finance-themed spam.")

    noise = float(config.get("label_noise_pct", 0.0))
    base_noise = float(baseline.get("label_noise_pct", 0.0))
    if noise > base_noise:
        messages.append(f"Label noise at {noise:.1f}% — expect metrics to drop as mislabeled examples grow.")
    elif noise < base_noise and base_noise > 0:
        messages.append("Label noise reduced — accuracy should recover.")

    if config.get("poison_demo") and not baseline.get("poison_demo"):
        messages.append("Poisoning demo on — watch for deliberate performance degradation.")

    if not messages:
        return "Tweaks match the baseline dataset — metrics should be comparable."
    return " ".join(messages)
