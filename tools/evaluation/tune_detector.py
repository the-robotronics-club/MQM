"""Compare detector candidates on grouped development and validation images.

Caches low-threshold detections and postures, then scores confidence cutoffs.
Validation sessions are fixed before candidate runs and never select the winner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from inference.model import PersonDetector
from inference.posture import classify_postures
from tools.evaluation.real_images import interval_error

CANDIDATES = [
    ("nano_baseline", "yolov8n.pt", 1280, 0.5),
    ("nano_overlap", "yolov8n.pt", 1280, 0.7),
    ("nano_large_input", "yolov8n.pt", 1600, 0.7),
    ("small_overlap", "yolov8s.pt", 1280, 0.7),
]
THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.35)


def split_for(filename: str) -> str:
    # Keep neighboring frames from these capture sessions together.
    if any(s in filename for s in ("20261002", "20260903", "REMAN DEY")):
        return "validation"
    return "development"


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"images": 0}
    return {
        "images": len(rows),
        "mae": {
            k: round(sum(r["error"][k] for r in rows) / len(rows), 3)
            for k in ("headcount", "sitting", "standing")
        },
        "exact_count_matches": {
            k: sum(r["error"][k] == 0 for r in rows) for k in ("headcount", "sitting", "standing")
        },
        "all_counts_match": sum(all(v == 0 for v in r["error"].values()) for r in rows),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/outputs/detector-tuning"))
    args = parser.parse_args()
    register_heif_opener()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    labels = [
        r
        for r in json.loads(args.labels.read_text())["records"]
        if r.get("filename") and r.get("sitting") and r.get("standing")
    ]
    classifiers_hash = hashlib.sha256(Path("models/seated_classifier.pt").read_bytes()).hexdigest()
    trials = []
    for name, weights, size, iou in CANDIDATES:
        detector = PersonDetector(
            weights_path=weights, confidence_threshold=0.05, image_size=size, iou_threshold=iou
        )
        weights_hash = hashlib.sha256((Path("models") / weights).read_bytes()).hexdigest()
        cached_rows = []
        for label in labels:
            path = args.images / label["filename"]
            image_bytes = path.read_bytes()
            identity = json.dumps(
                [
                    hashlib.sha256(image_bytes).hexdigest(),
                    classifiers_hash,
                    weights_hash,
                    size,
                    iou,
                    label.get("rotation_degrees_ccw", 0),
                ]
            )
            key = hashlib.sha256(identity.encode()).hexdigest()
            cache = args.output / "cache" / f"{key}.json"
            cache.parent.mkdir(exist_ok=True)
            if cache.exists():
                row = json.loads(cache.read_text())
            else:
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                rotation = label.get("rotation_degrees_ccw", 0)
                if rotation:
                    image = image.rotate(rotation, expand=True)
                frame = np.asarray(image)[:, :, ::-1].copy()
                start = time.monotonic()
                detections = detector.detect_people(frame)
                postures = classify_postures(frame, detections)
                row = {
                    "detections": detections,
                    "postures": [{"label": p.label, "confidence": p.confidence} for p in postures],
                    "seconds": time.monotonic() - start,
                }
                cache.write_text(json.dumps(row))
            cached_rows.append((label, row))
        for threshold in THRESHOLDS:
            scored = []
            for label, row in cached_rows:
                chosen = [
                    i for i, d in enumerate(row["detections"]) if d["confidence"] >= threshold
                ]
                counts = {
                    "headcount": len(chosen),
                    "sitting": sum(row["postures"][i]["label"] == "Sitting" for i in chosen),
                    "standing": sum(row["postures"][i]["label"] == "Standing" for i in chosen),
                }
                total = {k: label["sitting"][k] + label["standing"][k] for k in ("min", "max")}
                expected = {
                    "headcount": total,
                    "sitting": label["sitting"],
                    "standing": label["standing"],
                }
                scored.append(
                    {
                        "filename": label["filename"],
                        "split": split_for(label["filename"]),
                        "needs_review": label["needs_review"],
                        "predicted": counts,
                        "expected": expected,
                        "error": {k: interval_error(v, expected[k]) for k, v in counts.items()},
                        "seconds": row["seconds"],
                        "exclude": label.get("exclude_from_model_selection", False),
                    }
                )
            usable = [r for r in scored if not r["exclude"]]
            development = [
                r for r in usable if r["split"] == "development" and not r["needs_review"]
            ]
            validation = [r for r in usable if r["split"] == "validation"]
            trial = {
                "name": name,
                "weights": weights,
                "size": size,
                "iou": iou,
                "confidence": threshold,
                "development": summarize(development),
                "validation": summarize(validation),
                "all": summarize(usable),
            }
            trials.append(trial)
            (args.output / f"{name}-{threshold:.2f}.json").write_text(json.dumps(scored, indent=2))
            print(json.dumps(trial), flush=True)
        (args.output / "trials.json").write_text(json.dumps(trials, indent=2))
    winner = min(
        trials,
        key=lambda t: (
            t["development"]["mae"]["sitting"] + t["development"]["mae"]["standing"],
            t["development"]["mae"]["headcount"],
        ),
    )
    result = {
        "selection": "Min development sitting MAE + standing MAE; tie-break headcount MAE",
        "validation_groups": ["20261002", "20260903", "REMAN DEY"],
        "limitations": "Existing dataset previously inspected; not a new external test set. "
        "Near-duplicate sessions are grouped. Unknown training overlap.",
        "winner": winner,
        "trials": trials,
    }
    (args.output / "selection.json").write_text(json.dumps(result, indent=2))
    print("SELECTED", json.dumps(winner), flush=True)


if __name__ == "__main__":
    main()
