"""Calibrate an empirical absolute-residual interval on validation predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from research.datasets.loader import load_split
from research.occlusion.uncertainty import residual_quantile


def calibrate(
    predictions_path: Path,
    output_path: Path,
    split_path: Path,
    calibration_version: str,
    quantile: float = 0.95,
) -> dict:
    raw_bytes = predictions_path.read_bytes()
    rows = list(csv.DictReader(raw_bytes.decode("utf-8").splitlines()))
    if not rows:
        raise ValueError("validation predictions cannot be empty")
    if {row.get("split") for row in rows} != {"val"}:
        raise ValueError("uncertainty must be calibrated only on the validation split")
    versions = {str(row.get("dataset_version")) for row in rows}
    if len(versions) != 1:
        raise ValueError("validation predictions must use one dataset version")
    split = load_split(split_path)
    if str(split.get("name", split_path.stem)) != "val":
        raise ValueError("uncertainty requires the frozen validation split file")
    if versions != {str(split.get("dataset_version"))}:
        raise ValueError("prediction dataset version does not match frozen validation split")
    allowed_images = {str(value) for value in split["image_ids"]}
    if any(str(row.get("image_id", "")) not in allowed_images for row in rows):
        raise ValueError("uncertainty predictions contain images outside the validation split")
    residuals = [int(row["queue_ground_truth"]) - int(row["queue_prediction"]) for row in rows]
    artifact = {
        "method": "absolute_validation_residual_quantile",
        "model_version": hashlib.sha256(raw_bytes + str(quantile).encode()).hexdigest()[:12],
        "dataset_version": versions.pop(),
        "split": "val",
        "calibration_version": calibration_version,
        "quantile": quantile,
        "delta": residual_quantile(residuals, quantile),
        "validation_frames": len(rows),
        "interval_claim": "empirical validation residual interval; not a formal coverage guarantee",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite uncertainty artifact: {output_path}")
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--calibration-version", required=True)
    parser.add_argument("--quantile", type=float, default=0.95)
    args = parser.parse_args()
    print(
        json.dumps(
            calibrate(
                args.predictions,
                args.output,
                args.split_file,
                args.calibration_version,
                args.quantile,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
