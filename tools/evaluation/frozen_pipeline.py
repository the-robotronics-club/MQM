"""Small, fixed frozen-weight experiments; no training or threshold search.

Tests flip-averaged posture, two-detector agreement, and bounded queue rescue.
The reused photo set remains diagnostic, even with capture-session grouping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from torchvision.ops import nms

from inference.classifiers import _crop_tensor, get_seated_classifier
from inference.posture import PosturePrediction
from inference.scene import DISTANCE_QUEUE_SETTINGS, summarize_scene
from tools.evaluation.tune_detector import split_for

POLICIES = {
    "control": {"rescue_allow_chaining": True, "rescue_min_neighbors": 1},
    "original_seeds": {"rescue_allow_chaining": False, "rescue_min_neighbors": 1},
    "two_original_seeds": {"rescue_allow_chaining": False, "rescue_min_neighbors": 2},
    "no_rescue": {"rescue_max_p_seated": None},
}


def overlap(a: list, b: list) -> float:
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
    return area / union if union > 0 else 0.0


def consensus_boxes(first: list[dict], second: list[dict]) -> list[dict]:
    """One-to-one cross-model matches; unmatched boxes require stronger evidence.

    Matched boxes overlap >=0.5; unmatched boxes need confidence >=0.35.
    Coordinates are confidence-weighted. Final NMS is fixed at 0.5.
    """
    used = set()
    merged = []
    for a in sorted(first, key=lambda d: d["confidence"], reverse=True):
        available = [j for j in range(len(second)) if j not in used]
        j = max(available, key=lambda j: overlap(a["bbox"], second[j]["bbox"]), default=None)
        if j is not None and overlap(a["bbox"], second[j]["bbox"]) >= 0.5:
            b = second[j]
            used.add(j)
            weight = a["confidence"] + b["confidence"]
            merged.append(
                {
                    "bbox": [
                        (x * a["confidence"] + y * b["confidence"]) / weight
                        for x, y in zip(a["bbox"], b["bbox"], strict=True)
                    ],
                    "confidence": weight / 2,
                }
            )
        elif a["confidence"] >= 0.35:
            merged.append({**a, "bbox": list(a["bbox"])})
    merged.extend(
        {**b, "bbox": list(b["bbox"])}
        for j, b in enumerate(second)
        if j not in used and b["confidence"] >= 0.35
    )
    if not merged:
        return []
    keep = nms(
        torch.tensor([d["bbox"] for d in merged], dtype=torch.float32),
        torch.tensor([d["confidence"] for d in merged]),
        0.5,
    ).tolist()
    return [merged[i] for i in keep]


def paired_postures(frame: np.ndarray, boxes: list[dict]) -> dict[str, list[dict]]:
    if not boxes:
        return {"original": [], "flip_mean": []}
    crops = torch.stack([_crop_tensor(frame, d["bbox"]) for d in boxes])
    classifier = get_seated_classifier()
    with classifier._lock, torch.inference_mode():
        p = torch.softmax(classifier.model(torch.cat([crops, crops.flip([3])])), dim=1)[:, 0]
    count = len(boxes)
    output = {}
    for name, probabilities in (
        ("original", p[:count]),
        ("flip_mean", (p[:count] + p[count:]) / 2),
    ):
        output[name] = [
            {"label": "Sitting" if v >= 0.5 else "Standing", "confidence": v if v >= 0.5 else 1 - v}
            for v in probabilities.tolist()
        ]
    return output


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"images": 0}
    keys = ("headcount", "sitting", "standing")
    return {
        "images": len(rows),
        "mae": {k: round(sum(r["error"][k] for r in rows) / len(rows), 3) for k in keys},
        "count_matches": {k: sum(r["error"][k] == 0 for r in rows) for k in keys},
        "all_counts_match": sum(all(r["error"][k] == 0 for k in keys) for r in rows),
        "rescued_count": sum(r["scene"]["rescued_count"] for r in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/outputs/frozen-pipeline"))
    args = parser.parse_args()
    register_heif_opener()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    labels = [
        r
        for r in json.loads(args.labels.read_text())["records"]
        if r.get("filename") and r.get("sitting") and r.get("standing")
    ]
    sources = {
        "nano": Path("data/outputs/final-distance-evaluation"),
        "yolo11m": Path("data/outputs/yolo11m-distance-evaluation"),
    }
    label_hash = hashlib.sha256(args.labels.read_bytes()).hexdigest()
    weight_hash = hashlib.sha256(Path("models/seated_classifier.pt").read_bytes()).hexdigest()
    saved = {}
    source_hashes = {}
    for name, path in sources.items():
        summary = json.loads((path / "summary.json").read_text())
        if summary["labels_sha256"] != label_hash:
            raise ValueError("Source inference labels differ from this benchmark")
        if summary["checkpoint_sha256"]["seated_classifier.pt"] != weight_hash:
            raise ValueError("Source classifier differs from the frozen checkpoint")
        raw = (path / "predictions.json").read_bytes()
        source_hashes[name] = hashlib.sha256(raw).hexdigest()
        saved[name] = {r["filename"]: r for r in json.loads(raw) if r.get("status") == "ok"}
    plan = {
        "training": False,
        "promotion": "None automatic; reused benchmark is diagnostic",
        "labels_sha256": label_hash,
        "seated_weights_sha256": weight_hash,
        "source_prediction_sha256": source_hashes,
        "queue_settings": DISTANCE_QUEUE_SETTINGS,
        "queue_policies": POLICIES,
        "posture_variants": ["original", "flip_mean"],
        "ensemble": {"match_iou": 0.5, "unmatched_confidence": 0.35, "nms_iou": 0.5},
        "selection": "Development sitting+standing MAE; validation reported after selection",
        "limitations": "Previously inspected sessions; not a pristine held-out test. "
        "50 exact labels, 10 count ranges; queue labels exist only for 419.",
    }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2))
    results = {}
    for name in ("nano", "yolo11m", "consensus"):
        for index, label in enumerate(labels):
            filename = label["filename"]
            detections = (
                consensus_boxes(
                    saved["nano"][filename]["detections"], saved["yolo11m"][filename]["detections"]
                )
                if name == "consensus"
                else saved[name][filename]["detections"]
            )
            path = args.images / filename
            identity = [
                hashlib.sha256(path.read_bytes()).hexdigest(),
                weight_hash,
                detections,
                label.get("rotation_degrees_ccw", 0),
                "paired-flip-v1",
            ]
            cache = (
                args.output
                / "cache"
                / (hashlib.sha256(json.dumps(identity).encode()).hexdigest() + ".json")
            )
            cache.parent.mkdir(exist_ok=True)
            if cache.exists():
                postures = json.loads(cache.read_text())
            else:
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                if label.get("rotation_degrees_ccw"):
                    image = image.rotate(label["rotation_degrees_ccw"], expand=True)
                frame = np.asarray(image)[:, :, ::-1].copy()
                postures = paired_postures(frame, detections)
                cache.write_text(json.dumps(postures))
            for posture_name, predictions in postures.items():
                for policy, settings in POLICIES.items():
                    scene = summarize_scene(
                        detections,
                        [PosturePrediction(**p) for p in predictions],
                        queue_settings=settings,
                    )
                    expected = {
                        "headcount": {
                            k: label["sitting"][k] + label["standing"][k] for k in ("min", "max")
                        },
                        "sitting": label["sitting"],
                        "standing": label["standing"],
                    }
                    error = {
                        k: max(v["min"] - scene[k], scene[k] - v["max"], 0)
                        for k, v in expected.items()
                    }
                    row = {
                        "filename": filename,
                        "split": split_for(filename),
                        "needs_review": label["needs_review"],
                        "expected": expected,
                        "predicted": {k: scene[k] for k in expected},
                        "error": error,
                        "scene": scene,
                    }
                    results.setdefault(f"{name}-{posture_name}-{policy}", []).append(row)
            print(f"{name} {index + 1}/{len(labels)} {filename}", flush=True)
        print("FINISHED", name, flush=True)
    trials = []
    for name, rows in results.items():
        trial = {"name": name, "all": metrics(rows)}
        for split in ("development", "validation"):
            trial[split] = metrics(
                [r for r in rows if r["split"] == split and not r["needs_review"]]
            )
        trial["exact_labels"] = metrics(
            [r for r in rows if all(r["expected"][k]["exact"] for k in ("sitting", "standing"))]
        )
        trials.append(trial)
        (args.output / f"{name}.json").write_text(json.dumps(rows, indent=2))
    development_leader = min(
        trials,
        key=lambda t: (
            t["development"]["mae"]["sitting"] + t["development"]["mae"]["standing"],
            -t["development"]["all_counts_match"],
        ),
    )
    summary = {
        "plan": plan,
        "trials": trials,
        "development_leader": development_leader,
        "max_all_count_matches_diagnostic": max(t["all"]["all_counts_match"] for t in trials),
        "goal_reached": any(t["all"]["all_counts_match"] >= 30 for t in trials),
        "weights_unchanged": hashlib.sha256(
            Path("models/seated_classifier.pt").read_bytes()
        ).hexdigest()
        == weight_hash,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "development_leader": development_leader,
                "max_all_count_matches_diagnostic": summary["max_all_count_matches_diagnostic"],
                "goal_reached": summary["goal_reached"],
                "weights_unchanged": summary["weights_unchanged"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
