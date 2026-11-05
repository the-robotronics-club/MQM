"""Density measures in projected ground-plane coordinates."""

from __future__ import annotations

import math

Point = tuple[float, float]


def local_density(points: list[Point], radius: float) -> list[int]:
    if radius <= 0:
        raise ValueError("density radius must be positive")
    return [
        sum(
            index != other and math.dist(point, candidate) <= radius
            for other, candidate in enumerate(points)
        )
        for index, point in enumerate(points)
    ]


def k_nearest_mean_distances(points: list[Point], k: int = 3) -> list[float]:
    if k <= 0:
        raise ValueError("k must be positive")
    result = []
    for index, point in enumerate(points):
        distances = sorted(
            math.dist(point, candidate) for other, candidate in enumerate(points) if index != other
        )
        result.append(sum(distances[:k]) / min(k, len(distances)) if distances else 0.0)
    return result
