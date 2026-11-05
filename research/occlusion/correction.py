"""Combine visible membership with bounded correction and uncertainty."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from research.features.schemas import PersonFeatures
from research.membership.inference import MembershipPrediction

from .residual import predict_hidden
from .spacing import estimate_missing_from_spacing
from .uncertainty import interval


@dataclass(frozen=True)
class QueueEstimate:
    queue_count: int
    visible_count: int
    hidden_estimate: int
    lower_bound: int | None
    upper_bound: int | None
    method: str
    model_version: str
    interval_calibrated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def correct_queue_count(
    features: list[PersonFeatures],
    predictions: list[MembershipPrediction],
    *,
    method: str = "none",
    expected_spacing: float | None = None,
    residual_model_path: str | Path | None = None,
    max_correction: int = 5,
    uncertainty_delta: float | None = None,
    spacing_min_mean_density: float = 0.5,
    spacing_gap_multiplier: float = 1.5,
    spacing_max_gap_multiplier: float = 6.0,
    correction_model_version: str | None = None,
) -> QueueEstimate:
    visible_indices = [index for index, item in enumerate(predictions) if item.is_queue]
    visible = len(visible_indices)
    hidden = 0
    model_version = "none"
    applied_method = method
    queue_density = (
        sum(features[index].local_density for index in visible_indices) / visible
        if visible
        else 0.0
    )
    if (
        method == "spacing"
        and expected_spacing is not None
        and queue_density >= spacing_min_mean_density
    ):
        # Missing-person gaps must be measured in calibrated ground/path units.
        # Normalized progress would make expected spacing depend on path length.
        positions = [features[index].path_arc_length for index in visible_indices]
        hidden = estimate_missing_from_spacing(
            positions,
            expected_spacing,
            minimum_gap=expected_spacing * spacing_gap_multiplier,
            max_correction=max_correction,
            max_path_gap=expected_spacing * spacing_max_gap_multiplier,
        ).hidden_estimate
        model_version = correction_model_version or f"spacing-{expected_spacing:g}"
    elif method == "residual" and residual_model_path and Path(residual_model_path).is_file():
        densities = [item.local_density for item in features]
        occlusions = [item.occlusion_score for item in features]
        residual_features = {
            "visible_count": float(visible),
            "mean_local_density": sum(densities) / len(densities) if densities else 0.0,
            "max_local_density": float(max(densities, default=0)),
            "mean_occlusion": sum(occlusions) / len(occlusions) if occlusions else 0.0,
            "number_of_detections": float(len(features)),
        }
        hidden = predict_hidden(residual_features, residual_model_path, max_correction)
        model_version = Path(residual_model_path).stem
    elif method == "spacing":
        reason = "no_expected_spacing" if expected_spacing is None else "insufficient_density"
        applied_method = f"none_fallback_{reason}"
        model_version = f"spacing-{reason}"
    elif method == "residual":
        applied_method = "none_fallback_no_residual_model"
        model_version = "no-residual-artifact"
    elif method not in {"none", "spacing", "residual"}:
        raise ValueError(f"unknown correction method: {method}")
    hidden = min(max_correction, max(0, hidden))
    final = visible + hidden
    correction_applied = method == "none" or applied_method in {"spacing", "residual"}
    interval_calibrated = uncertainty_delta is not None and correction_applied
    low, high = interval(final, uncertainty_delta) if interval_calibrated else (None, None)
    return QueueEstimate(
        final,
        visible,
        hidden,
        low,
        high,
        f"P2_{applied_method.upper()}" if method != "none" else "P1_VISIBLE_ONLY",
        model_version,
        interval_calibrated,
    )
