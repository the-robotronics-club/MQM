"""Decode one camera JPEG and run the real detection-to-metrics pipeline."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .metrics import crowd_level, metrics_from_assignments
from .zones import DEFAULT_CONFIG_PATH, ZoneAssignment, assign_zones, load_zones

if TYPE_CHECKING:
    from .classifiers import PersonAttributes
    from .posture import PosturePrediction

Detector = Callable[[np.ndarray], list[dict]]
AttributeClassifier = Callable[[np.ndarray, list[ZoneAssignment]], list["PersonAttributes"]]
PostureClassifier = Callable[[np.ndarray, list[dict]], list["PosturePrediction"]]


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a decodable JPEG image."""


def decode_jpeg(image_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """Return a BGR ndarray and its (width, height), rejecting non-JPEG data."""
    if not image_bytes:
        raise InvalidImageError("uploaded image is empty")
    if not image_bytes.startswith(b"\xff\xd8\xff"):
        raise InvalidImageError("expected JPEG bytes")
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError("uploaded file is not a valid JPEG")
    height, width = image.shape[:2]
    return image, (width, height)


def process_frame(
    image_bytes: bytes,
    camera_id: str,
    detector: Detector | None = None,
    attribute_classifier: AttributeClassifier | None = None,
    posture_classifier: PostureClassifier | None = None,
) -> dict:
    """Decode, detect, assign zones, and compute the complete metrics schema."""
    image, frame_size = decode_jpeg(image_bytes)
    mode = os.environ.get("FRAME_METRICS_MODE", "calibrated").strip().lower()
    if mode not in {"calibrated", "distance"}:
        raise ValueError("FRAME_METRICS_MODE must be calibrated or distance")
    if mode == "distance" and os.environ.get("QUEUE_ESTIMATOR", "production") != "production":
        raise ValueError("Distance snapshots require QUEUE_ESTIMATOR=production")
    config = load_zones(camera_id, DEFAULT_CONFIG_PATH)
    if detector is None:
        if os.environ.get("INFERENCE_BACKEND", "torch") == "onnx":
            from .onnx_runtime import get_onnx_models

            detector = get_onnx_models().detect_people
        else:
            from .model import detect_people

            detector = detect_people
    detections = detector(image)
    if mode == "distance":
        from .posture import classify_postures
        from .scene import summarize_scene

        postures = (posture_classifier or classify_postures)(image, detections)
        scene = summarize_scene(detections, postures)
        return {
            "camera_id": camera_id,
            "headcount": scene["headcount"],
            "queue_count": scene["queue_count"],
            "sitting": scene["sitting"],
            "standing_other_count": scene["standing_other_count"],
            "scene_method": "distance",
            # No capacity or occupancy claim until the actual camera is calibrated.
            "seats_total": None,
            "seats_occupied": None,
            "seat_occupancy_pct": None,
            "crowd_level": crowd_level(scene["headcount"], config.crowd_thresholds),
            "zone_counts": {},
            "detections_raw": len(detections),
            "detections_counted": len(detections),
        }
    assignments = assign_zones(detections, config, frame_size=frame_size)

    if attribute_classifier is None:
        from .classifiers import attribute_classifiers_enabled, classify_people

        if attribute_classifiers_enabled():
            attribute_classifier = classify_people

    attributes = attribute_classifier(image, assignments) if attribute_classifier else None
    metrics = metrics_from_assignments(
        assignments,
        config,
        detections_raw=len(detections),
        queue_flags=[attribute.is_queue for attribute in attributes] if attributes else None,
        seated_flags=[attribute.is_seated for attribute in attributes] if attributes else None,
    )
    strategy = os.environ.get("QUEUE_ESTIMATOR", "production").strip().lower()
    if strategy == "production":
        return metrics
    if strategy not in {"geometric", "membership", "occlusion"}:
        raise ValueError("QUEUE_ESTIMATOR must be production, geometric, membership, or occlusion")
    from research.estimator import estimate_queue

    research_detections = [
        detection
        for detection in detections
        if float(detection.get("confidence", 0.0)) >= config.min_confidence
    ]
    research_metrics = estimate_queue(image, research_detections, camera_id, frame_size, strategy)
    research_metrics.pop("research_diagnostics", None)
    metrics.update(research_metrics)
    return metrics
