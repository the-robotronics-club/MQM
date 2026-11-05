"""Exploratory count-accuracy experiments on supplied images, never filename corrections.

Cache detector/posture outputs; compare global count policies on grouped sessions.
These images have already been inspected, so even the session split is exploratory.
No candidate is automatically promoted into the production pipeline.
The all-data leaderboard is diagnostic only. Keep training data and the final
unseen test in separate capture sessions from these repeatedly inspected images.
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
from torchvision.ops import nms

from inference.model import PersonDetector
from inference.posture import classify_postures
from tools.evaluation.real_images import interval_error
from tools.evaluation.tune_detector import CANDIDATES, split_for, summarize


def trim_black_borders(frame: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Remove only outer near-black padding; retain original-coordinate offsets."""
    content = np.any(frame > 12, axis=2)
    ys, xs = np.where(content)
    if not len(xs):
        return frame, 0, 0
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    return frame[y0:y1, x0:x1].copy(), x0, y0


def experimental_detections(detector: PersonDetector, frame: np.ndarray, mode: str) -> list[dict]:
    view, ox, oy = trim_black_borders(frame)
    detections = detector.detect_people(view, augment=mode == "trim_tta")
    if mode == "trim_tiles":
        height, width = view.shape[:2]
        cw, ch = int(width * 0.6), int(height * 0.6)
        for x in (0, width - cw):
            for y in (0, height - ch):
                crop = view[y : y + ch, x : x + cw].copy()
                for detection in detector.detect_people(crop):
                    a, b, c, d = detection["bbox"]
                    # Reject fragments cut by an internal tile edge. The full
                    # image pass can still supply these people.
                    if (
                        (x > 0 and a < 3)
                        or (y > 0 and b < 3)
                        or (x + cw < width and c > cw - 3)
                        or (y + ch < height and d > ch - 3)
                    ):
                        continue
                    detections.append({**detection, "bbox": [a + x, b + y, c + x, d + y]})
        if detections:
            keep = nms(
                torch.tensor([d["bbox"] for d in detections], dtype=torch.float32),
                torch.tensor([d["confidence"] for d in detections]),
                0.5,
            ).tolist()
            detections = [detections[i] for i in keep]
    return [
        {**d, "bbox": [d["bbox"][0] + ox, d["bbox"][1] + oy, d["bbox"][2] + ox, d["bbox"][3] + oy]}
        for d in detections
    ]


def score(
    rows: list[tuple[dict, dict]], confidence: float, seated: float, adaptive: bool
) -> list[dict]:
    scored = []
    for label, row in rows:
        cutoff = confidence
        if adaptive and sum(d["confidence"] >= 0.15 for d in row["detections"]) >= 25:
            cutoff = 0.15
        chosen = [i for i, d in enumerate(row["detections"]) if d["confidence"] >= cutoff]
        sitting = sum(
            (
                row["postures"][i]["confidence"]
                if row["postures"][i]["label"] == "Sitting"
                else 1 - row["postures"][i]["confidence"]
            )
            >= seated
            for i in chosen
        )
        counts = {"headcount": len(chosen), "sitting": sitting, "standing": len(chosen) - sitting}
        expected = {
            "headcount": {k: label["sitting"][k] + label["standing"][k] for k in ("min", "max")},
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
            }
        )
    return scored


def suppress_rows(rows: list[tuple[dict, dict]], iou: float | None) -> list[tuple[dict, dict]]:
    if iou is None:
        return rows
    result = []
    for label, row in rows:
        detections = row["detections"]
        keep = (
            nms(
                torch.tensor([d["bbox"] for d in detections], dtype=torch.float32),
                torch.tensor([d["confidence"] for d in detections]),
                iou,
            ).tolist()
            if detections
            else []
        )
        result.append(
            (
                label,
                {
                    **row,
                    "detections": [detections[i] for i in keep],
                    "postures": [row["postures"][i] for i in keep],
                },
            )
        )
    return result


def grid_trials(name: str, rows: list[tuple[dict, dict]]) -> list[dict]:
    trials = []
    for post_nms in (None, 0.3, 0.4):
        reduced = suppress_rows(rows, post_nms)
        for confidence in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            for seated in (0.3, 0.4, 0.5, 0.6, 0.7):
                policies = (False, True) if confidence == 0.25 else (False,)
                for adaptive in policies:
                    scored = score(reduced, confidence, seated, adaptive)
                    trial = {
                        "candidate": name,
                        "confidence": confidence,
                        "seated": seated,
                        "adaptive": adaptive,
                        "post_nms": post_nms,
                    }
                    for cohort in ("all", "development", "validation"):
                        selected = (
                            scored
                            if cohort == "all"
                            else [
                                r for r in scored if r["split"] == cohort and not r["needs_review"]
                            ]
                        )
                        trial[cohort] = summarize(selected)
                    trial["exact_labels"] = summarize(
                        [
                            r
                            for r in scored
                            if all(
                                r["expected"][k].get("exact", False)
                                for k in ("sitting", "standing")
                            )
                        ]
                    )
                    trials.append(trial)
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/outputs/accuracy-trials"))
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--extra-weights", action="append", default=[])
    parser.add_argument("--only-extra", action="store_true")
    args = parser.parse_args()
    register_heif_opener()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    labels = [
        r
        for r in json.loads(args.labels.read_text())["records"]
        if r.get("filename") and r.get("sitting") and r.get("standing")
    ]
    classifier_hash = hashlib.sha256(Path("models/seated_classifier.pt").read_bytes()).hexdigest()
    trials = []
    candidate_rows = {}
    provenance = []
    candidates = [] if args.only_extra else [(n, w, s, i, "cached") for n, w, s, i in CANDIDATES]
    if not args.cached_only and not args.only_extra:
        candidates += [
            (m, "yolov8n.pt", 1280, 0.5, m) for m in ("trim_full", "trim_tta", "trim_tiles")
        ]
    candidates += [(Path(w).stem, w, 1280, 0.5, "trim_full") for w in args.extra_weights]
    if not candidates:
        parser.error("No candidates selected; supply --extra-weights with --only-extra")
    for name, weights, size, iou, mode in candidates:
        weights_hash = hashlib.sha256((Path("models") / weights).read_bytes()).hexdigest()
        provenance.append(
            {
                "name": name,
                "weights": weights,
                "weights_sha256": weights_hash,
                "image_size": size,
                "iou": iou,
                "mode": mode,
            }
        )
        detector = None
        rows = []
        for index, label in enumerate(labels):
            path = args.images / label["filename"]
            image_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            identity = [
                image_hash,
                classifier_hash,
                weights_hash,
                size,
                iou,
                label.get("rotation_degrees_ccw", 0),
            ]
            if mode == "cached":
                cache_root = Path("data/outputs/detector-tuning/cache")
            else:
                identity += [mode, "tiles60pct-nms0.5-internal-edge3-v1"]
                cache_root = args.output / "cache"
            cache_root.mkdir(parents=True, exist_ok=True)
            cache = cache_root / (
                hashlib.sha256(json.dumps(identity).encode()).hexdigest() + ".json"
            )
            if cache.exists():
                row = json.loads(cache.read_text())
            elif mode == "cached":
                raise FileNotFoundError(f"Missing prior detector cache: {cache}")
            else:
                if detector is None:
                    detector = PersonDetector(
                        weights_path=weights,
                        confidence_threshold=0.05,
                        image_size=size,
                        iou_threshold=iou,
                    )
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                if label.get("rotation_degrees_ccw"):
                    image = image.rotate(label["rotation_degrees_ccw"], expand=True)
                frame = np.asarray(image)[:, :, ::-1].copy()
                start = time.monotonic()
                detections = experimental_detections(detector, frame, mode)
                postures = classify_postures(frame, detections)
                row = {
                    "detections": detections,
                    "postures": [{"label": p.label, "confidence": p.confidence} for p in postures],
                    "seconds": time.monotonic() - start,
                }
                cache.write_text(json.dumps(row))
                print(
                    f"{name} {index + 1}/{len(labels)} {label['filename']}: "
                    f"{len(detections)} candidates",
                    flush=True,
                )
            rows.append((label, row))
        candidate_rows[name] = rows
        trials.extend(grid_trials(name, rows))
        (args.output / "trials.json").write_text(json.dumps(trials, indent=2))
        print("FINISHED", name, flush=True)

    # Choose on development sessions only, then report validation and all data.
    # The all-data leaderboard is descriptive, never a claim of held-out accuracy.
    def rank(t: dict) -> tuple:
        return (
            -t["development"]["all_counts_match"],
            t["development"]["mae"]["sitting"] + t["development"]["mae"]["standing"],
            t["development"]["mae"]["headcount"],
        )

    result = {
        "goal": "At least 30/60 images with all three counts inside supplied label intervals",
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "seated_classifier_sha256": classifier_hash,
        "candidates": provenance,
        "trial_count": len(trials),
        "training_performed": False,
        "production_promotion": "None: these repeatedly inspected images are diagnostic only.",
        "limitation": "Exploratory reused images; intervals are not exact labels. "
        "Queue not evaluated.",
        "selected_development": min(trials, key=rank),
        "best_all_exploratory": min(
            trials,
            key=lambda t: (
                -t["all"]["all_counts_match"],
                t["all"]["mae"]["sitting"] + t["all"]["mae"]["standing"],
            ),
        ),
        "best_headcount_exploratory": min(
            trials,
            key=lambda t: (
                -t["all"]["exact_count_matches"]["headcount"],
                t["all"]["mae"]["headcount"],
            ),
        ),
    }
    for key in ("selected_development", "best_all_exploratory", "best_headcount_exploratory"):
        policy = result[key]
        predictions = score(
            suppress_rows(candidate_rows[policy["candidate"]], policy["post_nms"]),
            policy["confidence"],
            policy["seated"],
            policy["adaptive"],
        )
        (args.output / f"{key}-predictions.json").write_text(json.dumps(predictions, indent=2))
    (args.output / "selection.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
