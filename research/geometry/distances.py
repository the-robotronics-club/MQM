"""Small distance primitives kept separate for experiment reuse."""

from __future__ import annotations

import math


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)
