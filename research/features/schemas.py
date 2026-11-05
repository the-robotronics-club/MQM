"""Typed, versioned feature records traceable to the paper formulation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from research import FEATURE_VERSION

PRIMARY_FEATURE_NAMES = (
    "x_ground",
    "y_ground",
    "distance_to_counter",
    "distance_to_path",
    "local_density",
    "rank_normalized",
    "detector_confidence",
    "occlusion_score",
)

SUPPORTED_FEATURE_NAMES = PRIMARY_FEATURE_NAMES + (
    "x_image",
    "y_image",
    "bbox_width",
    "bbox_height",
    "nearest_neighbor_distance",
    "path_progress",
    "path_arc_length",
)


@dataclass(frozen=True)
class PersonFeatures:
    detection_id: str
    bbox: tuple[float, float, float, float]
    image_point: tuple[float, float]
    ground_point: tuple[float, float]
    distance_to_counter: float
    distance_to_path: float
    local_density: int
    rank: int
    rank_normalized: float
    detector_confidence: float
    occlusion_score: float
    path_progress: float
    bbox_width: float
    bbox_height: float
    nearest_neighbor_distance: float
    perspective_band: str
    feature_version: str = FEATURE_VERSION
    # Absolute distance from the counter along the calibrated queue path.
    # This retains the calibration's units and is used by spacing correction.
    path_arc_length: float = 0.0

    def vector(self, names: tuple[str, ...] = PRIMARY_FEATURE_NAMES) -> list[float]:
        values: dict[str, float] = {
            "x_ground": self.ground_point[0],
            "y_ground": self.ground_point[1],
            "x_image": self.image_point[0],
            "y_image": self.image_point[1],
            "distance_to_counter": self.distance_to_counter,
            "distance_to_path": self.distance_to_path,
            "local_density": float(self.local_density),
            "rank_normalized": self.rank_normalized,
            "detector_confidence": self.detector_confidence,
            "occlusion_score": self.occlusion_score,
            "bbox_width": self.bbox_width,
            "bbox_height": self.bbox_height,
            "nearest_neighbor_distance": self.nearest_neighbor_distance,
            "path_progress": self.path_progress,
            "path_arc_length": self.path_arc_length,
        }
        try:
            return [values[name] for name in names]
        except KeyError as exc:
            raise ValueError(f"unknown feature name: {exc.args[0]}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
