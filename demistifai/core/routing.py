from __future__ import annotations
from typing import Optional, Tuple

def route_decision(autonomy: str, y_hat: str, pspam: Optional[float], threshold: float):
    if pspam is not None:
        to_spam = pspam >= threshold
    else:
        to_spam = y_hat == "spam"
    if autonomy.startswith("High"):
        routed = "Spam" if to_spam else "Inbox"
        action = f"Auto-routed to **{routed}** (threshold={threshold:.2f})"
    else:
        routed = None
        action = f"Recommend: {'Spam' if to_spam else 'Inbox'} (threshold={threshold:.2f})"
    return action, routed
