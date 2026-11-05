"""Projection onto an ordered, curved ground-plane queue centreline."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

Point = tuple[float, float]


@dataclass(frozen=True)
class PathProjection:
    point: Point
    distance: float
    arc_length: float
    normalized_progress: float
    segment_index: int


class QueuePath:
    def __init__(self, points: Iterable[Point]) -> None:
        self.points = tuple((float(x), float(y)) for x, y in points)
        if len(self.points) < 2:
            raise ValueError("queue path needs at least two points")
        self.lengths = tuple(
            math.dist(a, b) for a, b in zip(self.points[:-1], self.points[1:], strict=True)
        )
        if any(length <= 0 for length in self.lengths):
            raise ValueError("queue path cannot contain repeated consecutive points")
        self.total_length = sum(self.lengths)

    def project(self, point: Point) -> PathProjection:
        best: tuple[float, Point, float, int] | None = None
        preceding = 0.0
        for index, (start, end, length) in enumerate(
            zip(self.points[:-1], self.points[1:], self.lengths, strict=True)
        ):
            dx, dy = end[0] - start[0], end[1] - start[1]
            fraction = max(
                0.0,
                min(
                    1.0,
                    ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / (length * length),
                ),
            )
            projected = (start[0] + fraction * dx, start[1] + fraction * dy)
            candidate = (
                math.dist(point, projected),
                projected,
                preceding + fraction * length,
                index,
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
            preceding += length
        assert best is not None
        return PathProjection(best[1], best[0], best[2], best[2] / self.total_length, best[3])

    def distance_to_polyline(self, point: Point) -> float:
        return self.project(point).distance

    def arc_length_position(self, point: Point) -> float:
        return self.project(point).arc_length

    def normalized_queue_progress(self, point: Point) -> float:
        return self.project(point).normalized_progress


def rank_along_path(points: list[Point], path: QueuePath) -> tuple[list[int], list[float]]:
    if not points:
        return [], []
    progress = [path.arc_length_position(point) for point in points]
    order = sorted(range(len(points)), key=lambda index: (progress[index], index))
    ranks = [0] * len(points)
    normalized = [0.0] * len(points)
    denominator = max(1, len(points) - 1)
    for rank, index in enumerate(order):
        ranks[index] = rank
        normalized[index] = rank / denominator
    return ranks, normalized
