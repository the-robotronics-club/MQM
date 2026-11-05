"""Deterministic session-grouped train/validation/test splitting."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .loader import load_manifest


def grouped_split(session_ids: list[str], seed: int = 42) -> dict[str, list[str]]:
    sessions = sorted(set(session_ids))
    if len(sessions) < 3:
        raise ValueError("at least three sessions are required for train/val/test splitting")
    random.Random(seed).shuffle(sessions)
    test_count = max(1, round(len(sessions) * 0.2))
    val_count = max(1, round(len(sessions) * 0.2))
    if test_count + val_count >= len(sessions):
        test_count = val_count = 1
    return {
        "train": sessions[: len(sessions) - val_count - test_count],
        "val": sessions[len(sessions) - val_count - test_count : -test_count],
        "test": sessions[-test_count:],
    }


def write_splits(manifest_path: Path, output: Path, seed: int = 42) -> None:
    manifest = load_manifest(manifest_path)
    groups = grouped_split([item.session_id for item in manifest.images], seed)
    output.mkdir(parents=True, exist_ok=True)
    for name, sessions in groups.items():
        path = output / f"{name}.json"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen split: {path}")
        image_ids = [item.image_id for item in manifest.images if item.session_id in sessions]
        payload = {
            "name": name,
            "dataset_version": manifest.version,
            "seed": seed,
            "session_ids": sessions,
            "image_ids": image_ids,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_splits(args.manifest, args.output, args.seed)


if __name__ == "__main__":
    main()
