"""Validate FP32 export counts against saved production predictions, without tuning."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from inference.onnx_runtime import OnnxSceneModels
from inference.scene import summarize_scene


def main() -> None:
    register_heif_opener()
    labels_path = Path("dataset_labels.json")
    labels = {r["filename"]: r for r in json.loads(labels_path.read_text())["records"]}
    source = Path("data/outputs/bounded-rescue-evaluation/predictions.json")
    baseline = [r for r in json.loads(source.read_text()) if r["status"] == "ok"]
    models = OnnxSceneModels()
    rows = []
    for row in baseline:
        filename = row["filename"]
        label = labels[filename]
        with Image.open(Path("drive-download-20261021T073157Z-1-001") / filename) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            if label.get("rotation_degrees_ccw"):
                image = image.rotate(label["rotation_degrees_ccw"], expand=True)
            frame = np.asarray(image)[:, :, ::-1].copy()
        start = time.monotonic()
        detections = models.detect_people(frame)
        scene = summarize_scene(detections, models.classify(frame, detections))
        keys = ("headcount", "queue_count", "sitting", "standing_other_count")
        counts = {k: scene[k] for k in keys}
        previous = {k: row["predicted"][k] for k in keys}
        result = {
            "filename": filename,
            "predicted": counts,
            "reference": previous,
            "match": counts == previous,
            "seconds": time.monotonic() - start,
        }
        rows.append(result)
        print(
            f"{len(rows)}/{len(baseline)} match={result['match']} {filename}: {counts}", flush=True
        )
    output = Path("data/outputs/onnx-parity")
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.json").write_text(json.dumps(rows, indent=2))
    summary = {
        "training": False,
        "policy_search": False,
        "images": len(rows),
        "all_category_counts_match_reference": sum(r["match"] for r in rows),
        "per_count_matches": {
            k: sum(r["predicted"][k] == r["reference"][k] for r in rows) for k in keys
        },
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "reference_predictions_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mac_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 1),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 3),
        "manifest": json.loads(Path("models/onnx-manifest.json").read_text()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
