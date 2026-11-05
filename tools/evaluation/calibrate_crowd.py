"""Choose a crowd guard using development counts, preserving a fixed sparse cutoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.evaluation.tune_detector import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning", type=Path, default=Path("data/outputs/detector-tuning"))
    args = parser.parse_args()
    root = args.tuning
    base = {
        r["filename"]: r
        for r in json.loads((root / "nano_baseline-0.25.json").read_text())
        if not r["exclude"]
    }
    low = {
        r["filename"]: r
        for r in json.loads((root / "nano_baseline-0.15.json").read_text())
        if not r["exclude"]
    }
    trials = []
    for guard in (15, 20, 25, 30, 40):
        rows = [
            low[name] if low[name]["predicted"]["headcount"] >= guard else row
            for name, row in base.items()
        ]
        result = {
            "crowd_min_people": guard,
            "development": summarize(
                [r for r in rows if r["split"] == "development" and not r["needs_review"]]
            ),
            "validation": summarize([r for r in rows if r["split"] == "validation"]),
            "all": summarize(rows),
        }
        trials.append(result)
        (root / f"adaptive-{guard}.json").write_text(json.dumps(rows, indent=2))
    winner = min(
        trials,
        key=lambda t: t["development"]["mae"]["sitting"] + t["development"]["mae"]["standing"],
    )
    result = {
        "weights": "yolov8n.pt",
        "size": 1280,
        "iou": 0.5,
        "confidence": 0.25,
        "crowd_confidence": 0.15,
        "seated_threshold": 0.5,
        "winner": winner,
        "trials": trials,
        "selection": "Min development sitting + standing MAE.",
        "limitations": "Exploratory follow-up after examining initial validation results; "
        "validation is not an untouched external test set.",
    }
    (root / "adaptive-selection.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
