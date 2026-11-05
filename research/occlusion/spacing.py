"""Transparent spacing-anomaly missing-count estimator."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


def fit_expected_spacing(low_occlusion_gaps: list[float]) -> float:
    values = [value for value in low_occlusion_gaps if value > 0]
    if not values:
        raise ValueError("at least one positive low-occlusion gap is required")
    return float(statistics.median(values))


@dataclass(frozen=True)
class SpacingCorrection:
    hidden_estimate: int
    gaps: tuple[float, ...]
    expected_spacing: float


def estimate_missing_from_spacing(
    path_positions: list[float],
    expected_spacing: float,
    *,
    minimum_gap: float | None = None,
    max_correction: int = 5,
    max_path_gap: float | None = None,
) -> SpacingCorrection:
    if expected_spacing <= 0 or max_correction < 0:
        raise ValueError("expected spacing must be positive and correction non-negative")
    ordered = sorted(path_positions)
    gaps = tuple(b - a for a, b in zip(ordered[:-1], ordered[1:], strict=True))
    threshold = minimum_gap if minimum_gap is not None else 1.5 * expected_spacing
    hidden = 0
    for gap in gaps:
        if gap < threshold or (max_path_gap is not None and gap > max_path_gap):
            continue
        hidden += max(0, round(gap / expected_spacing) - 1)
    return SpacingCorrection(min(max_correction, hidden), gaps, expected_spacing)
