"""Train Logistic Regression or GBDT membership from labelled feature rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib

from research import FEATURE_VERSION
from research.features.schemas import PRIMARY_FEATURE_NAMES, SUPPORTED_FEATURE_NAMES
from research.membership.boosted import create_model as create_gbdt
from research.membership.logistic import create_model as create_logistic

from .protocol import validate_rows_against_split


def train(
    input_path: Path,
    output_path: Path,
    split_path: Path,
    method: str,
    seed: int = 42,
    feature_names: tuple[str, ...] = PRIMARY_FEATURE_NAMES,
) -> dict:
    if method not in {"logistic", "gbdt"}:
        raise ValueError("method must be logistic or gbdt")
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_names must be non-empty and unique")
    unknown = sorted(set(feature_names) - set(SUPPORTED_FEATURE_NAMES))
    if unknown:
        raise ValueError(f"unknown membership features: {', '.join(unknown)}")
    raw_bytes = input_path.read_bytes()
    raw = json.loads(raw_bytes)
    if raw.get("feature_version") != FEATURE_VERSION:
        raise ValueError("training manifest feature version mismatch")
    rows = raw.get("people", [])
    validate_rows_against_split(raw, rows, split_path, expected_split="train")
    usable = [row for row in rows if row.get("role") != "uncertain"]
    uncertain_count = len(rows) - len(usable)
    labels = [int(row["role"] == "queue_member") for row in usable]
    if len(set(labels)) < 2:
        raise ValueError("training data must contain queue and non-queue examples")
    features = [[float(row[name]) for name in feature_names] for row in usable]
    model = create_logistic(seed) if method == "logistic" else create_gbdt(seed)
    model.fit(features, labels)
    version = hashlib.sha256(raw_bytes + method.encode()).hexdigest()[:12]
    artifact = {
        "model": model,
        "method": method,
        "model_version": version,
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names,
        "dataset_version": raw.get("dataset_version", "unversioned"),
        "split": raw.get("split"),
        "random_seed": seed,
        "training_examples": len(usable),
        "uncertain_excluded": uncertain_count,
        "detector": raw.get("detector"),
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
    parser.add_argument("--method", choices=["logistic", "gbdt"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--features",
        nargs="+",
        choices=SUPPORTED_FEATURE_NAMES,
        default=list(PRIMARY_FEATURE_NAMES),
        help="Explicit feature family for controlled ablations.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.manifest,
                args.output,
                args.split_file,
                args.method,
                args.seed,
                tuple(args.features),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
