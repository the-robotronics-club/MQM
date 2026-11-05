"""
inference/model.py

Owner: Ishan

Loads YOLOv8n (pretrained on COCO) and runs person detection on a single
input image. Filters detections down to the "person" class (COCO class 0)
and returns a clean list of bounding boxes + confidence scores.

This module ONLY detects people and where they are in the frame. It does
NOT decide who is in the queue, who is seated, or who is staff — that's
zone logic, handled downstream by inference/zones.py and
inference/metrics.py (owned by Curio). This module has no knowledge of
zones, camera_id, or business logic — keep it that way so it stays easy
to swap models (e.g. YOLOv8n -> YOLOv8s) without touching anything else.

Output contract (do not change without telling Curio + the ingestion API):
    [
        {"bbox": [x1, y1, x2, y2], "confidence": 0.87},
        {"bbox": [x1, y1, x2, y2], "confidence": 0.63},
        ...
    ]
    bbox is [x1, y1, x2, y2] = top-left corner, bottom-right corner, in
    ORIGINAL-IMAGE pixel coordinates (not model-input/resized space —
    ultralytics rescales box.xyxy back to the source image size by
    default, regardless of the imgsz used for inference). zones.json
    polygons must be defined in this same original-image coordinate
    space, or every point-in-polygon check silently returns false and
    all metrics read zero with no error raised anywhere.

    Zone-layer logic (owned by Curio) derives whichever reference point
    it needs from each bbox to test against zone polygons — e.g.
    centroid ((x1+x2)/2, (y1+y2)/2) or bottom-center ((x1+x2)/2, y2).
    This is a zone-layer decision, not fixed by this module, and may
    change independent of anything here. bbox format itself (corners,
    original-image space) is the only part of the contract that's fixed.

    On no detections, returns [] — never None. An empty hall/frame is a
    normal case, not an error; treating it as anything but a plain empty
    list would break the ingestion endpoint on every poll where nobody's
    in frame.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from ultralytics import YOLO

from inference.preprocessing import crop_black_padding

# COCO class id for "person"
PERSON_CLASS_ID = 0

# --- Tunable inference parameters -------------------------------------
# These are intentionally NOT hardcoded inline — tune once real mess-hall
# camera footage is available. Crowded queue frames may need a lower
# confidence threshold to catch partially-occluded people, and a higher
# IoU threshold to retain distinct people whose boxes overlap.

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_CROWD_CONFIDENCE_THRESHOLD = 0.15
DEFAULT_CROWD_MIN_PEOPLE = 25
DEFAULT_IOU_THRESHOLD = 0.5  # NMS overlap threshold; lower suppresses
# more overlapping boxes, higher retains more crowd candidates.
DEFAULT_IMAGE_SIZE = 1280  # inference resolution; higher helps
# small/occluded people in crowds
DEFAULT_MODEL_WEIGHTS = "yolov8n.pt"

# Weights live in models/ at the repo root, kept out of version control by
# .gitignore (*.pt) because they run to hundreds of megabytes. A bare filename
# is resolved against that directory; an explicit path is used as given.
# Unresolved names fall through to ultralytics, which downloads on demand.
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "models"


def resolve_weights(weights: str | Path) -> str:
    """Resolve a weights filename against models/, falling back to the input."""
    path = Path(weights)
    if path.is_absolute() or path.parent != Path("."):
        return str(path)
    local = WEIGHTS_DIR / path.name
    return str(local) if local.exists() else str(path)


@dataclass
class Detection:
    bbox: list[float]  # [x1, y1, x2, y2]
    confidence: float

    def to_dict(self) -> dict:
        return {"bbox": self.bbox, "confidence": self.confidence}


class PersonDetector:
    """
    Thin wrapper around an ultralytics YOLO model, scoped to person
    detection only. Load once (model loading is slow), reuse across
    requests.
    """

    def __init__(
        self,
        weights_path: str = DEFAULT_MODEL_WEIGHTS,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        image_size: int = DEFAULT_IMAGE_SIZE,
        adaptive_confidence: bool = False,
        crowd_confidence_threshold: float = DEFAULT_CROWD_CONFIDENCE_THRESHOLD,
        crowd_min_people: int = DEFAULT_CROWD_MIN_PEOPLE,
        trim_black_padding: bool = False,
    ):
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("image_size must be greater than zero")
        if not 0 <= crowd_confidence_threshold <= 1:
            raise ValueError("crowd_confidence_threshold must be between 0 and 1")
        if crowd_min_people <= 0:
            raise ValueError("crowd_min_people must be greater than zero")
        self.model = YOLO(resolve_weights(weights_path))
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.adaptive_confidence = adaptive_confidence
        self.crowd_confidence_threshold = crowd_confidence_threshold
        self.crowd_min_people = crowd_min_people
        self.trim_black_padding = trim_black_padding
        self._predict_lock = threading.Lock()

    def detect_people(
        self,
        image: str | Path | np.ndarray,
        confidence_threshold: float | None = None,
        iou_threshold: float | None = None,
        image_size: int | None = None,
        augment: bool = False,
    ) -> list[dict]:
        """
        Run inference on a single image and return only person detections.

        Args:
            image: file path, or numpy array (e.g. decoded JPEG frame from
                the ingestion endpoint — BGR or RGB, ultralytics handles both).
            confidence_threshold / iou_threshold / image_size: override the
                instance defaults for this call (useful for A/B testing
                tuning once real footage comes in).
            augment: test-time augmentation, improves recall on hard/occluded
                cases at the cost of speed. Off by default; fine to enable
                given the ~10s capture cadence, not a real-time video feed.

        Returns:
            List of {"bbox": [x1, y1, x2, y2], "confidence": float},
            one entry per detected person, sorted by confidence descending.
        """
        conf = (
            confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        )
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        imgsz = image_size if image_size is not None else self.image_size
        # Explicit per-call cutoffs remain fixed for controlled experiments.
        adaptive = self.adaptive_confidence and confidence_threshold is None
        candidate_conf = min(conf, self.crowd_confidence_threshold) if adaptive else conf
        source = image
        offset_x = offset_y = 0
        if self.trim_black_padding and isinstance(image, np.ndarray):
            source, offset_x, offset_y = crop_black_padding(image)

        # Ultralytics mutates predictor state; serialize calls on the shared
        # singleton so concurrent camera uploads cannot corrupt one another.
        with self._predict_lock:
            results = self.model.predict(
                source=source,
                conf=candidate_conf,
                iou=iou,
                imgsz=imgsz,
                augment=augment,
                classes=[PERSON_CLASS_ID],  # filter to person at inference time
                verbose=False,
            )

        detections: list[Detection] = []
        for raw_result in results:
            # Ultralytics covers several task families with a Results | Tensor
            # return type; this person-detection model returns Results at runtime.
            result = cast(Any, raw_result)
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                if cls_id != PERSON_CLASS_ID:
                    continue  # belt-and-suspenders, classes= should already filter
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf.item())
                detections.append(
                    Detection(
                        bbox=[x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y],
                        confidence=confidence,
                    )
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        candidates = [d for d in detections if d.confidence >= candidate_conf]
        cutoff = candidate_conf if adaptive and len(candidates) >= self.crowd_min_people else conf
        return [d.to_dict() for d in candidates if d.confidence >= cutoff]


# --- Module-level convenience function ---------------------------------
# Lazily-loaded singleton so api/ingestion.py can just do:
#   from inference.model import detect_people
#   people = detect_people(image_bytes_or_array)
# without managing model lifecycle itself.

_default_detector: PersonDetector | None = None
_default_detector_lock = threading.Lock()


def get_detector() -> PersonDetector:
    global _default_detector
    if _default_detector is not None:
        return _default_detector
    with _default_detector_lock:
        if _default_detector is None:
            try:
                confidence = float(os.environ.get("MODEL_CONFIDENCE", DEFAULT_CONFIDENCE_THRESHOLD))
                iou = float(os.environ.get("MODEL_IOU", DEFAULT_IOU_THRESHOLD))
                image_size = int(os.environ.get("MODEL_IMAGE_SIZE", DEFAULT_IMAGE_SIZE))
                crowd_confidence = float(
                    os.environ.get("MODEL_CROWD_CONFIDENCE", DEFAULT_CROWD_CONFIDENCE_THRESHOLD)
                )
                crowd_min_people = int(
                    os.environ.get("MODEL_CROWD_MIN_PEOPLE", DEFAULT_CROWD_MIN_PEOPLE)
                )
            except ValueError as exc:
                raise ValueError(
                    "MODEL_CONFIDENCE, MODEL_IOU, MODEL_IMAGE_SIZE, MODEL_CROWD_CONFIDENCE, "
                    "and MODEL_CROWD_MIN_PEOPLE must be numeric"
                ) from exc
            adaptive_value = os.environ.get("MODEL_ADAPTIVE_CONFIDENCE", "true").strip().lower()
            if adaptive_value not in {"true", "false", "1", "0"}:
                raise ValueError("MODEL_ADAPTIVE_CONFIDENCE must be true or false")
            padding_value = os.environ.get("MODEL_TRIM_BLACK_PADDING", "false").strip().lower()
            if padding_value not in {"true", "false", "1", "0"}:
                raise ValueError("MODEL_TRIM_BLACK_PADDING must be true or false")
            _default_detector = PersonDetector(
                weights_path=os.environ.get("MODEL_WEIGHTS", DEFAULT_MODEL_WEIGHTS),
                confidence_threshold=confidence,
                iou_threshold=iou,
                image_size=image_size,
                adaptive_confidence=adaptive_value in {"true", "1"},
                crowd_confidence_threshold=crowd_confidence,
                crowd_min_people=crowd_min_people,
                trim_black_padding=padding_value in {"true", "1"},
            )
    return _default_detector


def detect_people(image: str | Path | np.ndarray, **kwargs) -> list[dict]:
    """Convenience wrapper around the default PersonDetector instance."""
    return get_detector().detect_people(image, **kwargs)


if __name__ == "__main__":
    # Quick manual smoke test:
    #   python inference/model.py path/to/test_image.jpg
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model.py <image_path>")
        sys.exit(1)

    dets = detect_people(sys.argv[1])
    print(f"Detected {len(dets)} person(s):")
    for d in dets:
        print(f"  bbox={d['bbox']}, confidence={d['confidence']:.2f}")
