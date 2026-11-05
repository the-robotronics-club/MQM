"""Intervals calibrated from held-out validation residuals."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def residual_quantile(errors: Sequence[float], quantile: float = 0.95) -> float:
    if not errors or not 0 < quantile < 1:
        raise ValueError("validation errors and a quantile in (0, 1) are required")
    return float(np.quantile(np.abs(np.asarray(errors, dtype=float)), quantile))


def interval(queue_count: int, delta: float | None) -> tuple[int, int]:
    if delta is None:
        return (queue_count, queue_count)
    if delta < 0:
        raise ValueError("interval delta cannot be negative")
    return (max(0, math.floor(queue_count - delta)), math.ceil(queue_count + delta))
