"""Fit expected queue spacing from low-occlusion training frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from research.occlusion.spacing import fit_expected_spacing

from .protocol import validate_rows_against_split


def fit(input_path: Path, output_path: Path, split_path: Path) -> dict:
    raw_bytes = input_path.read_bytes()
    raw = json.loads(raw_bytes)
    frames = raw.get("frames", [])
    validate_rows_against_split(raw, frames, split_path, expected_split="train")
    gaps: list[float] = []
    frames_used = 0
    for frame in frames:
        if frame.get("occlusion_band") != "none":
            continue
        positions = sorted(float(value) for value in frame.get("queue_path_positions", []))
        if len(positions) < 2:
            continue
        gaps.extend(b - a for a, b in zip(positions[:-1], positions[1:], strict=True))
        frames_used += 1
    expected = fit_expected_spacing(gaps)
    artifact = {
        "method": "median_low_occlusion_spacing",
        "model_version": hashlib.sha256(raw_bytes + b"spacing").hexdigest()[:12],
        "dataset_version": raw.get("dataset_version", "unversioned"),
        "split": raw.get("split"),
        "calibration_version": raw.get("calibration_version", "unversioned"),
        "coordinate_units": raw.get("coordinate_units", "arbitrary_units"),
        "expected_spacing": expected,
        "training_frames": frames_used,
        "training_gaps": len(gaps),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite spacing artifact: {output_path}")
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit(args.manifest, args.output, args.split_file), indent=2))


if __name__ == "__main__":
    main()
