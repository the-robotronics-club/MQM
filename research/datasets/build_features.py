"""Build labelled features by matching detector boxes to person annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from research import FEATURE_VERSION
from research.calibration.camera_calibration import DEFAULT_CAMERA_DIR, load_calibration
from research.detection import CallableDetectorAdapter, Detection
from research.features import extract_features
from research.features.schemas import SUPPORTED_FEATURE_NAMES

from .loader import load_manifest, load_split
from .validation import validate_manifest

Detector = Callable[[np.ndarray], list[dict[str, Any]]]


def intersection_over_union(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_annotations(
    detections: list[Detection], annotations: tuple[Any, ...], minimum_iou: float
) -> dict[int, int]:
    if not 0 < minimum_iou <= 1:
        raise ValueError("minimum_iou must be in (0, 1]")
    candidates = sorted(
        (
            (
                intersection_over_union(detection.bbox, annotation.bbox),
                detection_index,
                annotation_index,
            )
            for detection_index, detection in enumerate(detections)
            for annotation_index, annotation in enumerate(annotations)
        ),
        reverse=True,
    )
    matched_detections: set[int] = set()
    matched_annotations: set[int] = set()
    matches: dict[int, int] = {}
    for iou, detection_index, annotation_index in candidates:
        if iou < minimum_iou:
            break
        if detection_index in matched_detections or annotation_index in matched_annotations:
            continue
        matches[detection_index] = annotation_index
        matched_detections.add(detection_index)
        matched_annotations.add(annotation_index)
    return matches


def build_features(
    manifest_path: Path,
    split_path: Path,
    root: Path,
    output: Path,
    detector: Detector,
    *,
    calibration_dir: Path = DEFAULT_CAMERA_DIR,
    minimum_iou: float = 0.5,
    detector_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    split = load_split(split_path)
    report = validate_manifest(manifest, root, [split_path])
    if not report.valid:
        raise ValueError("dataset validation failed: " + "; ".join(report.errors))
    if split.get("dataset_version") != manifest.version:
        raise ValueError("split dataset version does not match manifest")
    selected = set(split["image_ids"])
    people: list[dict[str, Any]] = []
    unmatched_detections = 0
    unmatched_annotations = 0
    adapter = CallableDetectorAdapter(detector)

    for record in manifest.images:
        if record.image_id not in selected:
            continue
        image = cv2.imread(str(root / record.path))
        if image is None:
            raise ValueError(f"cannot decode image: {root / record.path}")
        height, width = image.shape[:2]
        calibration = load_calibration(record.camera_id, calibration_dir)
        detections = adapter.detect(image)
        features = extract_features(detections, calibration, (width, height))
        matches = match_annotations(detections, record.annotations, minimum_iou)
        unmatched_detections += len(detections) - len(matches)
        unmatched_annotations += len(record.annotations) - len(matches)
        for detection_index, annotation_index in sorted(matches.items()):
            feature = features[detection_index]
            annotation = record.annotations[annotation_index]
            row = {
                "image_id": record.image_id,
                "session_id": record.session_id,
                "detection_id": feature.detection_id,
                "role": annotation.role,
                "occlusion_label": annotation.occlusion,
                "truncation": annotation.truncation,
            }
            row.update(
                dict(
                    zip(
                        SUPPORTED_FEATURE_NAMES,
                        feature.vector(SUPPORTED_FEATURE_NAMES),
                        strict=True,
                    )
                )
            )
            people.append(row)

    payload = {
        "dataset_version": manifest.version,
        "split": str(split.get("name", split_path.stem)),
        "feature_version": FEATURE_VERSION,
        "matching_iou_threshold": minimum_iou,
        "detector": detector_metadata or {"name": "provided_callable"},
        "people": people,
        "matching_summary": {
            "matched_detections": len(people),
            "unmatched_detections": unmatched_detections,
            "unmatched_annotations": unmatched_annotations,
        },
    }
    if output.exists():
        raise FileExistsError(f"refusing to overwrite feature manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-weights", required=True)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CAMERA_DIR)
    parser.add_argument("--minimum-iou", type=float, default=0.5)
    args = parser.parse_args()

    from inference.model import PersonDetector, resolve_weights

    detector = PersonDetector(weights_path=args.detector_weights)
    resolved_weights = Path(resolve_weights(args.detector_weights))
    detector_metadata = {
        "weights": str(resolved_weights),
        "sha256": (
            hashlib.sha256(resolved_weights.read_bytes()).hexdigest()
            if resolved_weights.is_file()
            else None
        ),
    }
    payload = build_features(
        args.manifest,
        args.split_file,
        args.root,
        args.output,
        detector.detect_people,
        calibration_dir=args.calibration_dir,
        minimum_iou=args.minimum_iou,
        detector_metadata=detector_metadata,
    )
    print(json.dumps(payload["matching_summary"], indent=2))


if __name__ == "__main__":
    main()
