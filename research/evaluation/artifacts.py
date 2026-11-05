"""Non-overwriting, reproducible experiment artifact writer."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import yaml


def file_hash(path: str | Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_experiment(
    output_root: Path,
    method: str,
    config: dict,
    metrics: dict,
    predictions: list[dict],
    per_session: list[dict],
    metadata_values: dict,
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = output_root / f"{stamp}-{method}-{digest}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _write_csv(output / "predictions.csv", predictions)
    _write_csv(output / "per_session.csv", per_session)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    package_names = ("numpy", "opencv-python", "scikit-learn", "torch", "ultralytics")
    dependencies: dict[str, str | None] = {}
    for name in package_names:
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    details = {
        "git_commit": commit,
        "timestamp": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "dependencies": dependencies,
        **metadata_values,
    }
    (output / "metadata.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
