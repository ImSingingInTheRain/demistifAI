from __future__ import annotations

from typing import Dict, List

import numpy as np


def _sample_indices_by_label(labels: List[str], limit: int) -> List[int]:
    n = len(labels)
    if n <= limit:
        return list(range(n))

    rng = np.random.default_rng(42)
    per_label: Dict[str, List[int]] = {}
    for idx, label in enumerate(labels):
        per_label.setdefault(label, []).append(idx)

    sampled: List[int] = []
    total = float(n)
    for label, idxs in per_label.items():
        if not idxs:
            continue
        target = max(1, round(limit * (len(idxs) / total)))
        target = min(target, len(idxs))
        picked = rng.choice(idxs, size=target, replace=False)
        sampled.extend(int(i) for i in picked)

    if len(sampled) > limit:
        sampled = rng.choice(sampled, size=limit, replace=False).tolist()

    return sorted(sampled)
