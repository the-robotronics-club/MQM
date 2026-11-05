"""Session-level bootstrap confidence intervals."""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np


def session_bootstrap(
    rows: list[dict],
    metric: Callable[[list[dict]], float],
    *,
    iterations: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    sessions: dict[str, list[dict]] = {}
    for row in rows:
        sessions.setdefault(str(row["session_id"]), []).append(row)
    if not sessions:
        raise ValueError("at least one session is required")
    identifiers = sorted(sessions)
    generator = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sample = [generator.choice(identifiers) for _ in identifiers]
        estimates.append(metric([row for session in sample for row in sessions[session]]))
    return {
        "mean": float(np.mean(estimates)),
        "lower_2_5": float(np.percentile(estimates, 2.5)),
        "upper_97_5": float(np.percentile(estimates, 97.5)),
        "iterations": float(iterations),
        "seed": float(seed),
    }
