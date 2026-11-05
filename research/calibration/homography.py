"""Validated image-to-ground homography computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

Point = tuple[float, float]


class CalibrationError(ValueError):
    pass


def _points(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1:] != (2,) or not np.isfinite(array).all():
        raise CalibrationError(f"{name} must be a finite Nx2 coordinate array")
    return array


@dataclass(frozen=True)
class Homography:
    matrix: np.ndarray
    reprojection_rmse: float

    def project_point(self, point: Point) -> Point:
        vector = self.matrix @ np.array([point[0], point[1], 1.0])
        if abs(vector[2]) < 1e-12:
            raise CalibrationError("point projects to infinity")
        return (float(vector[0] / vector[2]), float(vector[1] / vector[2]))

    def project_detection(self, detection: Any, reference: str = "bottom_center") -> Point:
        from research.geometry.projection import detection_point

        return self.project_point(detection_point(detection.bbox, reference))

    def project_detections(
        self, detections: list[Any], reference: str = "bottom_center"
    ) -> list[Point]:
        return [self.project_detection(item, reference) for item in detections]


def compute_homography(
    image_points: Any,
    ground_points: Any,
    *,
    max_reprojection_rmse: float | None = None,
) -> Homography:
    image = _points(image_points, "image_points")
    ground = _points(ground_points, "ground_points")
    if image.shape != ground.shape or len(image) < 4:
        raise CalibrationError(
            "image_points and ground_points need matching lengths of at least four"
        )
    if len(np.unique(image, axis=0)) < 4 or len(np.unique(ground, axis=0)) < 4:
        raise CalibrationError("calibration needs at least four distinct point pairs")
    matrix, _ = cv2.findHomography(image, ground, method=0)
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise CalibrationError("homography is singular")
    projected = cv2.perspectiveTransform(image.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum((projected - ground) ** 2, axis=1))))
    if max_reprojection_rmse is not None and rmse > max_reprojection_rmse:
        raise CalibrationError(
            f"reprojection RMSE {rmse:.4f} exceeds configured limit {max_reprojection_rmse:.4f}"
        )
    return Homography(matrix=matrix, reprojection_rmse=rmse)


def validate_calibration(
    config: dict[str, Any], frame_size: tuple[int, int] | None = None
) -> Homography:
    frame = config.get("frame_size")
    if not isinstance(frame, dict):
        raise CalibrationError("frame_size must contain width and height")
    try:
        width, height = int(frame["width"]), int(frame["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError("frame_size must contain positive integer width and height") from exc
    if width <= 0 or height <= 0:
        raise CalibrationError("frame dimensions must be positive")
    if frame_size is not None and frame_size != (width, height):
        raise CalibrationError(
            f"frame size {frame_size} does not match calibration {(width, height)}"
        )
    if config.get("calibration_status") not in {"validated", "metric", "arbitrary_units"}:
        raise CalibrationError("camera calibration is not marked validated")
    homography = config.get("homography")
    if not isinstance(homography, dict):
        raise CalibrationError("homography configuration is missing")
    image_points = _points(homography.get("image_points"), "image_points")
    if np.any(image_points[:, 0] < 0) or np.any(image_points[:, 0] > width):
        raise CalibrationError("image calibration x-coordinate lies outside the frame")
    if np.any(image_points[:, 1] < 0) or np.any(image_points[:, 1] > height):
        raise CalibrationError("image calibration y-coordinate lies outside the frame")
    result = compute_homography(
        image_points,
        homography.get("ground_points"),
        max_reprojection_rmse=config.get("max_reprojection_rmse"),
    )
    for name in ("counter", "queue_path"):
        if name not in config:
            raise CalibrationError(f"camera calibration is missing {name}")
    return result
