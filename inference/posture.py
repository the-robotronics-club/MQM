"""Shared posture classification for photos, video, and evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PosturePrediction:
    label: str
    confidence: float


def classify_postures(
    image: np.ndarray, detections: list[dict[str, Any]]
) -> list[PosturePrediction]:
    """Classify each detected person crop as sitting or standing."""
    if not detections:
        return []
    if os.environ.get("INFERENCE_BACKEND", "torch") == "onnx":
        from inference.onnx_runtime import get_onnx_models

        return get_onnx_models().classify(image, detections)

    from inference.classifiers import get_seated_classifier
    from inference.zones import ZoneAssignment, reference_point

    assignments = [
        ZoneAssignment(
            bbox=detection["bbox"],
            confidence=float(detection.get("confidence", 0.0)),
            point=reference_point(detection["bbox"]),
        )
        for detection in detections
    ]
    labels, confidence = get_seated_classifier().predict(image, assignments)
    return [
        PosturePrediction(
            label="Sitting" if label == 0 else "Standing",
            confidence=score,
        )
        for label, score in zip(labels, confidence, strict=True)
    ]
