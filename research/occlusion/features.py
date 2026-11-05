"""Geometry-only occlusion evidence for visible detections."""

from __future__ import annotations

from collections.abc import Sequence


def intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def occlusion_scores(
    boxes: list[Sequence[float]], frame_size: tuple[int, int], boundary_margin: float = 0.02
) -> list[float]:
    """Combine overlap fraction and boundary truncation; never infers hidden people."""
    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    scores = []
    for index, box in enumerate(boxes):
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        overlap = max(
            (
                intersection_area(box, other) / area
                for other_index, other in enumerate(boxes)
                if index != other_index and area
            ),
            default=0.0,
        )
        truncated = float(
            box[0] <= width * boundary_margin
            or box[1] <= height * boundary_margin
            or box[2] >= width * (1 - boundary_margin)
            or box[3] >= height * (1 - boundary_margin)
        )
        scores.append(min(1.0, 0.8 * overlap + 0.2 * truncated))
    return scores
