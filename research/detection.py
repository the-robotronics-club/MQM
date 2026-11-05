"""Detector-neutral records and adapters for research experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"invalid detection bbox: {self.bbox}")
        if not 0 <= self.confidence <= 1:
            raise ValueError("detection confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PersonDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


class CallableDetectorAdapter:
    """Normalize the existing MQM detector or any matching callable."""

    def __init__(self, detector: Callable[[np.ndarray], list[dict[str, Any]]]) -> None:
        self.detector = detector

    def detect(self, image: np.ndarray) -> list[Detection]:
        return [
            Detection(
                bbox=tuple(float(value) for value in item["bbox"]),  # type: ignore[arg-type]
                confidence=float(item["confidence"]),
                class_id=int(item.get("class_id", 0)),
                metadata=dict(item.get("metadata", {})),
            )
            for item in self.detector(image)
        ]


def production_detector() -> CallableDetectorAdapter:
    from inference.model import detect_people

    return CallableDetectorAdapter(detect_people)
