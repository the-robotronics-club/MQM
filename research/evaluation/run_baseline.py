"""Run one scientific baseline on a frozen manifest split and emit predictions.

This command performs inference only. Use ``research.evaluation.evaluate`` on
the resulting CSV so metrics and session bootstrap intervals remain centralized.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2

from inference.metrics import compute_metrics
from inference.zones import ReferencePoint, load_zones
from research.datasets.build_features import match_annotations
from research.datasets.loader import load_manifest, load_split
from research.detection import CallableDetectorAdapter

from .baselines import require_supported

FINE_TUNED_METHODS = {"b2", "b3", "b4", "p1", "p2"}


def _fixed_detector(detections: list[dict]) -> Callable[[Any], list[dict]]:
    def detect(_: Any) -> list[dict]:
        return detections

    return detect


def _annotation_detections(record: Any) -> list[dict]:
    return [
        {"bbox": list(annotation.bbox), "confidence": 1.0, "class_id": 0}
        for annotation in record.annotations
    ]


def _frame_occlusion(record: Any) -> str:
    levels = {annotation.occlusion for annotation in record.annotations}
    return "heavy" if "heavy" in levels else "partial" if "partial" in levels else "none"


def _frame_perspective(record: Any, height: int) -> str:
    bands = []
    for annotation in record.annotations:
        vertical = annotation.bbox[3] / height
        bands.append("near" if vertical >= 2 / 3 else "middle" if vertical >= 1 / 3 else "far")
    return bands[0] if bands and len(set(bands)) == 1 else "mixed" if bands else "unknown"


def _write_predictions(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite predictions: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("selected split contains no images")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    manifest_path: Path,
    split_path: Path,
    root: Path,
    method: str,
    output: Path,
    detector_weights: str | None = None,
    inference_resolution: int | None = None,
) -> None:
    require_supported(method)
    if method in FINE_TUNED_METHODS and not detector_weights:
        raise ValueError(f"{method.upper()} requires --detector-weights for a fine-tuned detector")
    manifest = load_manifest(manifest_path)
    split = load_split(split_path)
    if split.get("dataset_version") != manifest.version:
        raise ValueError("split dataset version does not match manifest")
    selected = set(split["image_ids"])
    split_name = str(split.get("name", split_path.stem))
    records = [record for record in manifest.images if record.image_id in selected]

    detector = None
    effective_inference_resolution = None
    if method != "b0":
        from inference.model import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL_WEIGHTS, PersonDetector

        effective_inference_resolution = inference_resolution or DEFAULT_IMAGE_SIZE
        detector = PersonDetector(
            weights_path=detector_weights or DEFAULT_MODEL_WEIGHTS,
            image_size=effective_inference_resolution,
        )

    rows: list[dict] = []
    for record in records:
        image_path = root / record.path
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot decode image: {image_path}")
        height, width = image.shape[:2]
        started = time.perf_counter()
        detections = (
            _annotation_detections(record) if method == "b0" else detector.detect_people(image)  # type: ignore[union-attr]
        )
        inference_method = method.upper()
        model_version = None
        membership_truth: list[bool] = []
        membership_prediction: list[bool] = []
        if method in {"b0", "b1", "b2", "b3"}:
            mode: ReferencePoint = "bottom_center" if method == "b3" else "centroid"
            prediction = compute_metrics(
                detections,
                load_zones(record.camera_id),
                frame_size=(width, height),
                mode=mode,
            )["queue_count"]
        elif method == "bcurrent":
            from inference.pipeline import process_frame

            previous = os.environ.get("QUEUE_ESTIMATOR")
            previous_classifiers = os.environ.get("ATTRIBUTE_CLASSIFIERS_ENABLED")
            os.environ["QUEUE_ESTIMATOR"] = "production"
            os.environ["ATTRIBUTE_CLASSIFIERS_ENABLED"] = "true"
            try:
                prediction = process_frame(
                    image_path.read_bytes(),
                    record.camera_id,
                    detector=_fixed_detector(detections),
                )["queue_count"]
            finally:
                if previous is None:
                    os.environ.pop("QUEUE_ESTIMATOR", None)
                else:
                    os.environ["QUEUE_ESTIMATOR"] = previous
                if previous_classifiers is None:
                    os.environ.pop("ATTRIBUTE_CLASSIFIERS_ENABLED", None)
                else:
                    os.environ["ATTRIBUTE_CLASSIFIERS_ENABLED"] = previous_classifiers
        else:
            from research.estimator import estimate_queue

            strategy = {"b4": "geometric", "p1": "membership", "p2": "occlusion"}[method]
            research_result = estimate_queue(
                image, detections, record.camera_id, (width, height), strategy
            )
            inference_method = str(research_result["queue_method"])
            model_version = str(research_result["queue_model_version"])
            if method in {"p1", "p2"} and "FALLBACK" in inference_method:
                raise RuntimeError(
                    f"{method.upper()} cannot be evaluated because inference used "
                    f"{inference_method}"
                )
            prediction = research_result["queue_count"]
            adapted = CallableDetectorAdapter(_fixed_detector(detections)).detect(image)
            matches = match_annotations(adapted, record.annotations, minimum_iou=0.5)
            diagnostics = research_result["research_diagnostics"]["people"]
            for detection_index, annotation_index in sorted(matches.items()):
                annotation = record.annotations[annotation_index]
                if annotation.role == "uncertain":
                    continue
                membership_truth.append(annotation.role == "queue_member")
                membership_prediction.append(
                    bool(diagnostics[detection_index]["membership"]["is_queue"])
                )
        latency_ms = (time.perf_counter() - started) * 1000
        rows.append(
            {
                "image_id": record.image_id,
                "session_id": record.session_id,
                "camera_id": record.camera_id,
                "timestamp": record.timestamp,
                "meal_period": record.meal_period,
                "crowd_band": record.crowd_band,
                "occlusion_band": _frame_occlusion(record),
                "perspective_band": _frame_perspective(record, height),
                "queue_ground_truth": record.queue_ground_truth,
                "queue_prediction": prediction,
                "latency_ms": round(latency_ms, 3),
                "method": method,
                "inference_method": inference_method,
                "model_version": model_version,
                "membership_truth": (json.dumps(membership_truth) if membership_truth else ""),
                "membership_prediction": (
                    json.dumps(membership_prediction) if membership_prediction else ""
                ),
                "dataset_version": manifest.version,
                "split": split_name,
                "detector_weights": (
                    (detector_weights or "yolov8n.pt") if method != "b0" else "ground_truth"
                ),
                "source_resolution": f"{width}x{height}",
                "inference_resolution": effective_inference_resolution,
            }
        )
    _write_predictions(output, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-weights")
    parser.add_argument("--inference-resolution", type=int)
    args = parser.parse_args()
    run(
        args.manifest,
        args.split_file,
        args.root,
        args.method.lower(),
        args.output,
        args.detector_weights,
        args.inference_resolution,
    )
    print(json.dumps({"predictions": str(args.output), "method": args.method.lower()}))


if __name__ == "__main__":
    main()
