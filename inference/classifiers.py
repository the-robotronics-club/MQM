"""Classify YOLO person crops as queued and seated/standing.

The two reviewed MobileNetV2 state dictionaries refine spatial-zone estimates:
the queue classifier is used only for people already in a queue zone, and the
seated classifier only for people already in a seating zone.  Zones remain the
camera-specific source of location; classifiers add posture/context evidence.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import mobilenet_v2

from .model import resolve_weights
from .zones import ZoneAssignment

QUEUE_WEIGHTS = "queue_classifier.pt"
SEATED_WEIGHTS = "seated_classifier.pt"
INPUT_SIZE = 128
_IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


@dataclass(frozen=True)
class PersonAttributes:
    """Predictions for one person crop, in the matching assignment order."""

    is_queue: bool
    is_seated: bool
    queue_confidence: float
    seated_confidence: float


def attribute_classifiers_enabled() -> bool:
    """Return whether reviewed attribute classifiers should refine live metrics."""
    value = os.environ.get("ATTRIBUTE_CLASSIFIERS_ENABLED", "false").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"", "0", "false", "no", "off"}:
        return False
    raise ValueError("ATTRIBUTE_CLASSIFIERS_ENABLED must be true or false")


def _load_mobilenet_v2(weights: str | Path) -> torch.nn.Module:
    """Load a reviewed, tensor-only two-class MobileNetV2 checkpoint on CPU."""
    resolved = Path(resolve_weights(weights))
    if not resolved.is_file():
        raise FileNotFoundError(f"attribute classifier weights not found: {resolved}")

    # `weights_only=True` refuses executable pickle globals. The supplied
    # checkpoints are state dictionaries, so no arbitrary checkpoint code runs.
    state = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{resolved.name} must contain a MobileNetV2 state dictionary")
    model = mobilenet_v2(weights=None, num_classes=2)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _crop_tensor(image: np.ndarray, bbox: list[float]) -> torch.Tensor:
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    height, width = image.shape[:2]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid person crop bounds: {bbox}")

    # Match the retraining notebook's PIL Resize (bilinear), RGB, normalization.
    crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    rgb = np.array(
        Image.fromarray(crop).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    )
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    return (tensor - _IMAGENET_MEAN) / _IMAGENET_STD


class PersonAttributeClassifiers:
    """Shared CPU classifiers, loaded once and serialized for safe reuse."""

    def __init__(
        self,
        queue_weights: str | Path = QUEUE_WEIGHTS,
        seated_weights: str | Path = SEATED_WEIGHTS,
    ) -> None:
        self.queue_model = _load_mobilenet_v2(queue_weights)
        self.seated_model = _load_mobilenet_v2(seated_weights)
        self._predict_lock = threading.Lock()

    @staticmethod
    def _predict(model: torch.nn.Module, crops: torch.Tensor) -> tuple[list[int], list[float]]:
        with torch.inference_mode():
            probabilities = torch.softmax(model(crops), dim=1)
        confidence, labels = probabilities.max(dim=1)
        return labels.tolist(), confidence.tolist()

    def predict(
        self,
        image: np.ndarray,
        assignments: list[ZoneAssignment],
    ) -> list[PersonAttributes]:
        if not assignments:
            return []
        crops = torch.stack([_crop_tensor(image, assignment.bbox) for assignment in assignments])
        with self._predict_lock:
            queue_labels, queue_confidence = self._predict(self.queue_model, crops)
            seated_labels, seated_confidence = self._predict(self.seated_model, crops)
        # Dataset folders sort alphabetically: not_queue=0/queue=1 and
        # seated=0/standing=1.
        return [
            PersonAttributes(
                is_queue=queue_label == 1,
                is_seated=seated_label == 0,
                queue_confidence=float(queue_score),
                seated_confidence=float(seated_score),
            )
            for queue_label, queue_score, seated_label, seated_score in zip(
                queue_labels,
                queue_confidence,
                seated_labels,
                seated_confidence,
                strict=True,
            )
        ]


_default_classifiers: PersonAttributeClassifiers | None = None
_default_classifiers_lock = threading.Lock()


class SeatedClassifier:
    """Distance grouping needs only the seated model, not the queue crop model."""

    def __init__(self, weights: str | Path = SEATED_WEIGHTS) -> None:
        self.model = _load_mobilenet_v2(weights)
        self._lock = threading.Lock()

    def predict(
        self, image: np.ndarray, assignments: list[ZoneAssignment]
    ) -> tuple[list[int], list[float]]:
        if not assignments:
            return [], []
        crops = torch.stack([_crop_tensor(image, assignment.bbox) for assignment in assignments])
        with self._lock:
            return PersonAttributeClassifiers._predict(self.model, crops)


_default_seated_classifier: SeatedClassifier | None = None
_default_seated_lock = threading.Lock()


def get_seated_classifier() -> SeatedClassifier:
    global _default_seated_classifier
    if _default_seated_classifier is not None:
        return _default_seated_classifier
    with _default_seated_lock:
        if _default_seated_classifier is None:
            _default_seated_classifier = SeatedClassifier(
                os.environ.get("SEATED_CLASSIFIER_WEIGHTS", SEATED_WEIGHTS)
            )
    return _default_seated_classifier


def get_attribute_classifiers() -> PersonAttributeClassifiers:
    global _default_classifiers
    if _default_classifiers is not None:
        return _default_classifiers
    with _default_classifiers_lock:
        if _default_classifiers is None:
            _default_classifiers = PersonAttributeClassifiers(
                queue_weights=os.environ.get("QUEUE_CLASSIFIER_WEIGHTS", QUEUE_WEIGHTS),
                seated_weights=os.environ.get("SEATED_CLASSIFIER_WEIGHTS", SEATED_WEIGHTS),
            )
    return _default_classifiers


def reset_attribute_classifiers() -> None:
    """Clear the singleton, primarily for isolated tests."""
    global _default_classifiers
    with _default_classifiers_lock:
        _default_classifiers = None


def classify_people(image: np.ndarray, assignments: list[ZoneAssignment]) -> list[PersonAttributes]:
    """Return queue/seated predictions in the same order as zone assignments."""
    return get_attribute_classifiers().predict(image, assignments)
