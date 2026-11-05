"""Train a bounded count-residual regressor from frame-level training rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib

from .protocol import validate_rows_against_split

RESIDUAL_FEATURES = (
    "visible_count",
    "mean_local_density",
    "max_local_density",
    "mean_occlusion",
    "number_of_detections",
)


def train(input_path: Path, output_path: Path, split_path: Path, seed: int = 42) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor

    raw_bytes = input_path.read_bytes()
    raw = json.loads(raw_bytes)
    rows = raw.get("frames", [])
    if not rows:
        raise ValueError("occlusion training needs frame rows")
    validate_rows_against_split(raw, rows, split_path, expected_split="train")
    features = [[float(row[name]) for name in RESIDUAL_FEATURES] for row in rows]
    targets = [max(0, int(row["queue_ground_truth"]) - int(row["visible_count"])) for row in rows]
    model = HistGradientBoostingRegressor(loss="poisson", random_state=seed)
    model.fit(features, targets)
    version = hashlib.sha256(raw_bytes + b"residual").hexdigest()[:12]
    artifact = {
        "model": model,
        "method": "hist_gradient_boosting_poisson",
        "model_version": version,
        "feature_names": RESIDUAL_FEATURES,
        "dataset_version": raw.get("dataset_version", "unversioned"),
        "split": raw.get("split"),
        "random_seed": seed,
        "training_frames": len(rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite model artifact: {output_path}")
    joblib.dump(artifact, output_path)
    return {key: value for key, value in artifact.items() if key != "model"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args.manifest, args.output, args.split_file, args.seed), indent=2))


if __name__ == "__main__":
    main()
