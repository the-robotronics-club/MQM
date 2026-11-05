"""Scientific baseline registry, including the actual production estimator."""

from __future__ import annotations

BASELINES = {
    "b0": "ground-truth boxes + polygon membership",
    "b1": "pretrained YOLO + polygon membership",
    "b2": "fine-tuned YOLO + polygon membership",
    "b3": "fine-tuned YOLO + bottom-centre polygon membership",
    "bcurrent": "production YOLO + polygon + MobileNetV2 queue classifier",
    "b4": "YOLO + homography + geometric membership",
    "p1": "B4 features + learned membership",
    "p2": "P1 + bounded missing-count correction",
    "p3": "P2 + temporal extension (requires suitable ordered clips)",
}


def require_supported(method: str, temporal_data_available: bool = False) -> None:
    if method not in BASELINES:
        raise ValueError(f"unknown baseline {method}; choose from {', '.join(BASELINES)}")
    if method == "p3" and not temporal_data_available:
        raise ValueError(
            "P3 requires valid ordered 1–5 FPS clips; sparse observations are insufficient"
        )
