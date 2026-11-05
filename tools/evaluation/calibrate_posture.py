"""Select a seated-probability cutoff on development image counts only.

Uses cached detector-tuning runs; never manufactures per-person training labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.evaluation.real_images import interval_error
from tools.evaluation.tune_detector import split_for, summarize


def seated_probability(posture: dict) -> float:
    return posture["confidence"] if posture["label"] == "Sitting" else 1 - posture["confidence"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--tuning", type=Path, default=Path("data/outputs/detector-tuning"))
    args = parser.parse_args()
    selected = json.loads((args.tuning / "selection.json").read_text())["winner"]
    labels = [
        r
        for r in json.loads(args.labels.read_text())["records"]
        if r.get("filename") and r.get("sitting") and r.get("standing")
    ]
    classifiers_hash = hashlib.sha256(Path("models/seated_classifier.pt").read_bytes()).hexdigest()
    weights_hash = hashlib.sha256((Path("models") / selected["weights"]).read_bytes()).hexdigest()
    source_rows = []
    for label in labels:
        identity = json.dumps(
            [
                hashlib.sha256((args.images / label["filename"]).read_bytes()).hexdigest(),
                classifiers_hash,
                weights_hash,
                selected["size"],
                selected["iou"],
                label.get("rotation_degrees_ccw", 0),
            ]
        )
        key = hashlib.sha256(identity.encode()).hexdigest()
        row = json.loads((args.tuning / "cache" / f"{key}.json").read_text())
        source_rows.append((label, row))
    trials = []
    for cutoff in (0.25, 0.35, 0.50, 0.65, 0.75):
        scored = []
        for label, row in source_rows:
            if label.get("exclude_from_model_selection", False):
                continue
            chosen = [
                i
                for i, d in enumerate(row["detections"])
                if d["confidence"] >= selected["confidence"]
            ]
            sitting = sum(seated_probability(row["postures"][i]) >= cutoff for i in chosen)
            counts = {
                "headcount": len(chosen),
                "sitting": sitting,
                "standing": len(chosen) - sitting,
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
                }
            )
        trial = {
            "seated_threshold": cutoff,
            "development": summarize(
                [r for r in scored if r["split"] == "development" and not r["needs_review"]]
            ),
            "validation": summarize([r for r in scored if r["split"] == "validation"]),
            "all": summarize(scored),
        }
        trials.append(trial)
        (args.tuning / f"posture-{cutoff:.2f}.json").write_text(json.dumps(scored, indent=2))
    winner = min(
        trials,
        key=lambda t: (
            t["development"]["mae"]["sitting"] + t["development"]["mae"]["standing"],
            abs(t["seated_threshold"] - 0.5),
        ),
    )
    result = {
        "detector": selected,
        "winner": winner,
        "trials": trials,
        "labels_sha256": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "selection": "Development sitting + standing MAE; tie prefers threshold nearest 0.5",
    }
    (args.tuning / "posture-selection.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
