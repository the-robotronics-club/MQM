"""JSON manifest and split loaders."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import DatasetManifest


def load_manifest(path: str | Path) -> DatasetManifest:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("images"), list):
        raise ValueError("dataset manifest must be an object containing an images list")
    return DatasetManifest.from_dict(raw)


def load_split(path: str | Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("image_ids"), list):
        raise ValueError("split must contain image_ids")
    return raw
