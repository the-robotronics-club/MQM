"""Shared distance-queue scene estimates for photos, video, and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from inference.distance_queue import assign_queue_by_distance
from inference.posture import PosturePrediction, classify_postures

# Notebook geometry with bounded rescue: use two original standing members,
# never let a rescued seated prediction extend the queue recursively.
DISTANCE_QUEUE_SETTINGS = {
    "dist_threshold": 1.6,
    "min_group_size": 3,
    "min_elongation": 2.0,
    "rescue_max_p_seated": 0.9,
    "rescue_head_tol": 0.2,
    "rescue_allow_chaining": False,
    "rescue_min_neighbors": 2,
}


def summarize_scene(
    detections: list[dict],
    postures: list[PosturePrediction],
    *,
    queue_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Group people and give each exactly one final category; never mutate inputs."""
    if len(detections) != len(postures):
        raise ValueError("Each person detection must have one posture prediction")
    people = []
    for detection, posture in zip(detections, postures, strict=True):
        seated = posture.label == "Sitting"
        people.append(
            {
                **detection,
                "bbox": list(detection["bbox"]),
                "seated": seated,
                "p_seated": round(posture.confidence if seated else 1 - posture.confidence, 3),
                "raw_posture": posture.label,
            }
        )
    settings = {**DISTANCE_QUEUE_SETTINGS, **(queue_settings or {})}
    groups, rejected = assign_queue_by_distance(people, **settings)
    for person in people:
        person["category"] = (
            "Queue" if person["in_queue"] else "Sitting" if person["seated"] else "Other standing"
        )
    queue = sum(p["category"] == "Queue" for p in people)
    sitting = sum(p["category"] == "Sitting" for p in people)
    other = sum(p["category"] == "Other standing" for p in people)
    return {
        "headcount": len(people),
        "queue_count": queue,
        "sitting": sitting,
        "standing": queue + other,
        "standing_other_count": other,
        "queue_groups": groups,
        "rescued_count": sum(p["rescued"] for p in people),
        "rejected_groups": rejected,
        "people": people,
        "method": "distance",
    }


def analyze_scene(image: np.ndarray, detections: list[dict]) -> dict[str, Any]:
    return summarize_scene(detections, classify_postures(image, detections))
