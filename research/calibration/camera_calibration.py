"""Typed per-camera research configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from research.geometry.queue_path import QueuePath

from .homography import CalibrationError, Homography, Point, validate_calibration

DEFAULT_CAMERA_DIR = Path(__file__).resolve().parents[1] / "configs" / "cameras"


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    frame_size: tuple[int, int]
    homography: Homography
    counter_point: Point
    queue_path: tuple[Point, ...]
    density_radius: float
    distance_to_path_threshold: float
    max_counter_distance: float
    calibration_version: str
    coordinate_units: str
    min_counter_distance: float = 0.0


def _point(value: Any, name: str) -> Point:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise CalibrationError(f"{name} must be a two-coordinate point")
    point = (float(value[0]), float(value[1]))
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise CalibrationError(f"{name} must contain finite coordinates")
    return point


def load_calibration(camera_id: str, directory: Path = DEFAULT_CAMERA_DIR) -> CameraCalibration:
    path = directory / f"{camera_id}.yaml"
    if not path.is_file():
        raise CalibrationError(f"validated research calibration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("camera_id") != camera_id:
        raise CalibrationError(f"camera_id mismatch in {path}")
    homography = validate_calibration(raw)
    frame = raw["frame_size"]
    path_points = tuple(_point(item, "queue_path point") for item in raw["queue_path"])
    if len(path_points) < 2:
        raise CalibrationError("queue_path needs at least two points from counter to queue end")
    try:
        QueuePath(path_points)
    except ValueError as exc:
        raise CalibrationError(f"invalid queue_path: {exc}") from exc
    rules = raw.get("geometric", {})
    density_radius = float(raw.get("density_radius", 1.0))
    threshold = float(
        rules.get(
            "distance_to_path_threshold",
            rules.get("max_path_distance", rules.get("max_distance_to_path", 1.0)),
        )
    )
    max_counter = float(rules.get("max_counter_distance", float("inf")))
    min_counter = float(rules.get("min_counter_distance", 0.0))
    if density_radius <= 0 or threshold <= 0 or min_counter < 0 or max_counter <= min_counter:
        raise CalibrationError("density and geometric distance values must be positive")
    return CameraCalibration(
        camera_id=camera_id,
        frame_size=(int(frame["width"]), int(frame["height"])),
        homography=homography,
        counter_point=_point(raw["counter"]["position"], "counter.position"),
        queue_path=path_points,
        density_radius=density_radius,
        distance_to_path_threshold=threshold,
        max_counter_distance=max_counter,
        calibration_version=str(raw.get("calibration_version", "unversioned")),
        coordinate_units=str(raw.get("coordinate_units", "arbitrary_units")),
        min_counter_distance=min_counter,
    )
