"""Deterministic projected feature extraction independent of learned models."""

from __future__ import annotations

import math

from research.calibration.camera_calibration import CameraCalibration
from research.detection import Detection
from research.geometry.density import k_nearest_mean_distances, local_density
from research.geometry.projection import detection_point
from research.geometry.queue_path import QueuePath, rank_along_path
from research.occlusion.features import occlusion_scores

from .schemas import PersonFeatures


def extract_features(
    detections: list[Detection],
    calibration: CameraCalibration,
    frame_size: tuple[int, int],
    *,
    reference: str = "bottom_center",
) -> list[PersonFeatures]:
    if frame_size != calibration.frame_size:
        raise ValueError(
            f"frame size {frame_size} does not match calibration {calibration.frame_size}"
        )
    image_points = [detection_point(item.bbox, reference) for item in detections]
    ground_points = [calibration.homography.project_point(point) for point in image_points]
    path = QueuePath(calibration.queue_path)
    ranks, normalized_ranks = rank_along_path(ground_points, path)
    densities = local_density(ground_points, calibration.density_radius)
    nearest = k_nearest_mean_distances(ground_points, 1)
    occlusion = occlusion_scores([item.bbox for item in detections], frame_size)
    result = []
    for index, (detection, image_point, ground_point) in enumerate(
        zip(detections, image_points, ground_points, strict=True)
    ):
        projection = path.project(ground_point)
        x1, y1, x2, y2 = detection.bbox
        vertical = image_point[1] / frame_size[1]
        band = "near" if vertical >= 2 / 3 else "middle" if vertical >= 1 / 3 else "far"
        result.append(
            PersonFeatures(
                detection_id=str(index),
                bbox=detection.bbox,
                image_point=image_point,
                ground_point=ground_point,
                distance_to_counter=math.dist(ground_point, calibration.counter_point),
                distance_to_path=projection.distance,
                local_density=densities[index],
                rank=ranks[index],
                rank_normalized=normalized_ranks[index],
                detector_confidence=detection.confidence,
                occlusion_score=occlusion[index],
                path_progress=projection.normalized_progress,
                bbox_width=x2 - x1,
                bbox_height=y2 - y1,
                nearest_neighbor_distance=nearest[index],
                perspective_band=band,
                path_arc_length=projection.arc_length,
            )
        )
    return result
