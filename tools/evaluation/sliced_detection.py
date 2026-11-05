"""One preregistered frozen detector comparison, without a threshold search.

Full frame versus full frame plus four overlapping tiles, both YOLO11m.
This is an experiment, not a runtime default or an unseen accuracy claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from torchvision.ops import nms

from inference.annotations import annotate_people
from inference.model import PersonDetector
from inference.posture import classify_postures
from inference.preprocessing import crop_black_padding
from inference.scene import DISTANCE_QUEUE_SETTINGS, summarize_scene
from tools.evaluation.real_images import interval_error
from tools.evaluation.tune_detector import split_for, summarize


def tile_windows(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Four 60%-sized crops; full-frame detections cover internal cut edges."""
    if min(width, height) < 32:
        return []
    cw, ch = int(width * 0.6), int(height * 0.6)
    return [(x, y, x + cw, y + ch) for y in (0, height - ch) for x in (0, width - cw)]


def restore_tile_boxes(
    detections: list[dict], window: tuple[int, int, int, int], size: tuple[int, int]
) -> list[dict]:
    x, y, right, bottom = window
    width, height = size
    cw, ch = right - x, bottom - y
    output = []
    for detection in detections:
        a, b, c, d = detection["bbox"]
        if (
            (x > 0 and a < 3)
            or (y > 0 and b < 3)
            or (right < width and c > cw - 3)
            or (bottom < height and d > ch - 3)
        ):
            continue
        output.append({**detection, "bbox": [a + x, b + y, c + x, d + y]})
    return output


def deduplicate(detections: list[dict]) -> list[dict]:
    if not detections:
        return []
    keep = nms(
        torch.tensor([d["bbox"] for d in detections], dtype=torch.float32),
        torch.tensor([d["confidence"] for d in detections], dtype=torch.float32),
        0.5,
    ).tolist()
    return [detections[i] for i in keep]


def crowd_filter(detections: list[dict]) -> list[dict]:
    """Apply production confidence guard AFTER deduplication, across the frame."""
    candidates = [d for d in detections if d["confidence"] >= 0.15]
    cutoff = 0.15 if len(candidates) >= 25 else 0.25
    return [d for d in candidates if d["confidence"] >= cutoff]


def checkpoint_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256((Path("models") / name).read_bytes()).hexdigest()
        for name in ("yolo11m.pt", "seated_classifier.pt", "queue_classifier.pt")
    }


def fixed_plan(labels_path: Path) -> dict:
    return {
        "experiment": "sliced-yolo11m-v1",
        "weights_sha256": checkpoint_hashes(),
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_size": 1280,
        "confidence": {"sparse": 0.25, "crowd": 0.15, "crowd_min_people": 25},
        "iou": 0.5,
        "padding": "production crop_black_padding",
        "tiles": {"fraction": 0.6, "internal_edge_margin_px": 3, "count": 4},
        "queue_settings": DISTANCE_QUEUE_SETTINGS,
        "variants": ["full", "sliced"],
        "training": False,
        "promotion": "None automatic; no parameter sweep; reused diagnostic images",
        "timing": "CPU, four threads; excludes image decoding; sliced includes full pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/outputs/sliced-detection"))
    args = parser.parse_args()
    register_heif_opener()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    plan = fixed_plan(args.labels)
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Experiment changed: choose a new output directory")
    plan_path.write_text(json.dumps(plan, indent=2))
    labels = [
        r
        for r in json.loads(args.labels.read_text())["records"]
        if r.get("filename") and r.get("sitting") and r.get("standing")
    ]
    detector = PersonDetector(weights_path="yolo11m.pt", confidence_threshold=0.15)
    rows: dict[str, list[dict]] = {"full": [], "sliced": []}
    cache_dir = args.output / "cache"
    cache_dir.mkdir(exist_ok=True)
    for index, label in enumerate(labels):
        path = args.images / label["filename"]
        if path.parent.resolve() != args.images.resolve():
            raise ValueError("Image must be directly inside image directory")
        rotation = label.get("rotation_degrees_ccw", 0)
        if rotation not in (0, 90, 180, 270):
            raise ValueError("Invalid orientation metadata")
        image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        cache_key = hashlib.sha256(json.dumps([plan, image_hash, rotation]).encode()).hexdigest()
        cache_path = cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            predictions = json.loads(cache_path.read_text())
        else:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            if rotation:
                image = image.rotate(rotation, expand=True)
            frame = np.asarray(image)[:, :, ::-1].copy()
            start = time.monotonic()
            view, ox, oy = crop_black_padding(frame)
            full = detector.detect_people(view)
            full_seconds = time.monotonic() - start
            height, width = view.shape[:2]
            combined = list(full)
            start = time.monotonic()
            for window in tile_windows(width, height):
                x, y, right, bottom = window
                local = detector.detect_people(view[y:bottom, x:right].copy())
                combined.extend(restore_tile_boxes(local, window, (width, height)))
            sliced = deduplicate(combined)
            sliced_seconds = full_seconds + time.monotonic() - start
            predictions = {}
            for name, boxes, seconds in (
                ("full", full, full_seconds),
                ("sliced", sliced, sliced_seconds),
            ):
                start = time.monotonic()
                boxes = [
                    {
                        **d,
                        "bbox": [
                            d["bbox"][0] + ox,
                            d["bbox"][1] + oy,
                            d["bbox"][2] + ox,
                            d["bbox"][3] + oy,
                        ],
                    }
                    for d in crowd_filter(boxes)
                ]
                postures = classify_postures(frame, boxes)
                scene = summarize_scene(boxes, postures)
                predictions[name] = {
                    "scene": scene,
                    "seconds": seconds + time.monotonic() - start,
                }
                # Annotation affects neither inference nor selection. Every image
                # receives an overlay; no special case for a benchmark filename.
                overlay = args.output / "overlays" / name / f"{path.stem}.png"
                overlay.parent.mkdir(parents=True, exist_ok=True)
                annotate_people(image, scene["people"]).save(overlay)
            cache_path.write_text(json.dumps(predictions))
        expected = {
            "headcount": {k: label["sitting"][k] + label["standing"][k] for k in ("min", "max")},
            "sitting": label["sitting"],
            "standing": label["standing"],
        }
        for name, prediction in predictions.items():
            scene = prediction["scene"]
            row = {
                "filename": label["filename"],
                "image_sha256": image_hash,
                "split": split_for(label["filename"]),
                "needs_review": label["needs_review"],
                "exact": all(label[k]["exact"] for k in ("sitting", "standing")),
                "expected": expected,
                "predicted": {k: scene[k] for k in expected},
                "error": {k: interval_error(scene[k], v) for k, v in expected.items()},
                **prediction,
            }
            if label.get("queue_count") is not None:
                row["queue_error"] = abs(scene["queue_count"] - label["queue_count"])
            rows[name].append(row)
            (args.output / f"{name}.json").write_text(json.dumps(rows[name], indent=2))
        print(
            f"{index + 1}/{len(labels)} {path.name}: "
            + " | ".join(
                f"{name} total={p['scene']['headcount']} Q={p['scene']['queue_count']} "
                f"S={p['scene']['sitting']} O={p['scene']['standing_other_count']}"
                for name, p in predictions.items()
            ),
            flush=True,
        )
    if checkpoint_hashes() != plan["weights_sha256"]:
        raise RuntimeError("Weights changed during experiment")
    summary = {"plan": plan, "weights_unchanged": True, "results": {}}
    for name, values in rows.items():
        cohorts = {"all": values, "exact": [r for r in values if r["exact"]]}
        cohorts.update(
            {
                split: [r for r in values if r["split"] == split and not r["needs_review"]]
                for split in ("development", "validation")
            }
        )
        queue = [r for r in values if "queue_error" in r]
        summary["results"][name] = {
            **{key: summarize(cohort) for key, cohort in cohorts.items()},
            "median_seconds": round(statistics.median(r["seconds"] for r in values), 3),
            "queue_labeled_images": len(queue),
            "queue_mae": statistics.mean(r["queue_error"] for r in queue) if queue else None,
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["results"], indent=2))


if __name__ == "__main__":
    main()
