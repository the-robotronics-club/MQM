"""Bounding-box reference point functions used by ablations."""

from __future__ import annotations

from collections.abc import Sequence

Point = tuple[float, float]


def bottom_center(bbox: Sequence[float]) -> Point:
    x1, _, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, float(y2))


def centroid(bbox: Sequence[float]) -> Point:
    x1, y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def detection_point(bbox: Sequence[float], mode: str = "bottom_center") -> Point:
    if mode == "bottom_center":
        return bottom_center(bbox)
    if mode == "centroid":
        return centroid(bbox)
    raise ValueError(f"unknown reference point: {mode}")
