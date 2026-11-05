"""Evaluate image-level count intervals with the production detector and posture model.

Run from the repo root: python -m tools.evaluation.real_images --help
Queue errors are measured only for images that include a queue_count annotation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from inference.model import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL_WEIGHTS,
    PersonDetector,
)
from inference.posture import classify_postures
from inference.scene import DISTANCE_QUEUE_SETTINGS, summarize_scene


def interval_error(prediction: int, label: dict) -> int:
    return max(label["min"] - prediction, 0, prediction - label["max"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/outputs/real-images"))
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD)
    parser.add_argument("--weights", default=DEFAULT_MODEL_WEIGHTS)
    parser.add_argument("--adaptive-confidence", action="store_true")
    parser.add_argument("--trim-black-padding", action="store_true")
    parser.add_argument("--queue-mode", choices=("distance", "posture"), default="distance")
    args = parser.parse_args()
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except ImportError:
        pass
    import torch

    torch.set_num_threads(4)
    detector = PersonDetector(
        weights_path=args.weights,
        confidence_threshold=args.confidence,
        image_size=args.image_size,
        iou_threshold=args.iou,
        adaptive_confidence=args.adaptive_confidence,
        trim_black_padding=args.trim_black_padding,
    )
    labels = json.loads(args.labels.read_text())["records"]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in labels:
        row = {
            "filename": label["filename"],
            "sheet_row": label["sheet_row"],
            "needs_review": label["needs_review"],
        }
        start = time.monotonic()
        try:
            if not label["filename"] or not label["sitting"] or not label["standing"]:
                raise ValueError("Missing filename or count annotation")
            path = args.images / label["filename"]
            if path.parent.resolve() != args.images.resolve():
                raise ValueError("Filename must be directly inside image directory")
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            rotation = label.get("rotation_degrees_ccw", 0)
            if rotation not in (0, 90, 180, 270):
                raise ValueError("rotation_degrees_ccw must be 0, 90, 180, or 270")
            if rotation:
                image = image.rotate(rotation, expand=True)
            row["rotation_degrees_ccw"] = rotation
            row["exclude_from_model_selection"] = label.get("exclude_from_model_selection", False)
            frame = np.asarray(image)[:, :, ::-1].copy()
            detections = detector.detect_people(frame)
            postures = classify_postures(frame, detections)
            counts = {
                "sitting": sum(p.label == "Sitting" for p in postures),
                "standing": sum(p.label == "Standing" for p in postures),
                "headcount": len(detections),
            }
            scene = summarize_scene(detections, postures) if args.queue_mode == "distance" else None
            if scene is not None:
                counts.update(
                    {
                        key: scene[key]
                        for key in ("sitting", "standing", "queue_count", "standing_other_count")
                    }
                )
            total = {key: label["sitting"][key] + label["standing"][key] for key in ("min", "max")}
            expected = {
                "sitting": label["sitting"],
                "standing": label["standing"],
                "headcount": total,
            }
            if scene is not None and label.get("queue_count") is not None:
                expected["queue_count"] = {"min": label["queue_count"], "max": label["queue_count"]}
            if scene is not None and label.get("roaming_count") is not None:
                expected["standing_other_count"] = {
                    "min": label["roaming_count"],
                    "max": label["roaming_count"],
                }
            row.update(
                status="ok",
                scene=scene,
                predicted=counts,
                expected=expected,
                absolute_interval_error={
                    key: interval_error(counts[key], interval) for key, interval in expected.items()
                },
                exact=all(label[key]["exact"] for key in ("sitting", "standing")),
                detections=detections,
                postures=[{"label": p.label, "confidence": p.confidence} for p in postures],
            )
            print(
                f"{label['sheet_row']}: people={counts['headcount']} sitting={counts['sitting']} "
                f"standing={counts['standing']} errors={row['absolute_interval_error']}",
                flush=True,
            )
        except Exception as exc:
            row.update(status="skipped", reason=f"{type(exc).__name__}: {exc}")
            print(f"{label['sheet_row']}: {row['reason']}", flush=True)
        row["seconds"] = round(time.monotonic() - start, 3)
        rows.append(row)
        (args.output / "predictions.json").write_text(json.dumps(rows, indent=2))
    summary = {
        "annotation": "Image-level counts; queue error only where queue labels exist",
        "queue_mode": args.queue_mode,
        "distance_settings": DISTANCE_QUEUE_SETTINGS if args.queue_mode == "distance" else None,
        "confidence": args.confidence,
        "image_size": args.image_size,
        "iou": args.iou,
        "adaptive_confidence": args.adaptive_confidence,
        "trim_black_padding": args.trim_black_padding,
        "crowd_confidence": detector.crowd_confidence_threshold,
        "crowd_min_people": detector.crowd_min_people,
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "detector": args.weights,
        "skipped": sum(r["status"] != "ok" for r in rows),
        "checkpoint_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("models").glob("*classifier.pt")
        },
    }
    for cohort in ("all", "all_usable", "exact_unflagged"):
        selected = [
            r
            for r in rows
            if r["status"] == "ok"
            and (
                cohort == "all"
                or (cohort == "all_usable" and not r["exclude_from_model_selection"])
                or (cohort == "exact_unflagged" and r["exact"] and not r["needs_review"])
            )
        ]
        summary[cohort] = {
            "images": len(selected),
            "mean_absolute_interval_error": {
                key: round(
                    sum(r["absolute_interval_error"][key] for r in selected) / len(selected), 3
                )
                if selected
                else None
                for key in ("headcount", "sitting", "standing")
            },
        }
    queue_labeled = [
        r for r in rows if r["status"] == "ok" and "queue_count" in r["absolute_interval_error"]
    ]
    summary["queue_labeled"] = {
        "images": len(queue_labeled),
        "mean_absolute_count_error": (
            sum(r["absolute_interval_error"]["queue_count"] for r in queue_labeled)
            / len(queue_labeled)
            if queue_labeled
            else None
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not summary["all"]["images"]:
        raise SystemExit("No images evaluated successfully")


if __name__ == "__main__":
    main()
