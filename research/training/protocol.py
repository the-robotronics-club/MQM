"""Frozen-split lineage checks shared by research training commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research.datasets.loader import load_split


def validate_rows_against_split(
    raw: dict[str, Any],
    rows: list[dict[str, Any]],
    split_path: Path,
    *,
    expected_split: str,
) -> dict[str, Any]:
    split = load_split(split_path)
    name = str(split.get("name", split_path.stem))
    if name != expected_split:
        raise ValueError(f"expected frozen {expected_split!r} split, got {name!r}")
    if raw.get("dataset_version") != split.get("dataset_version"):
        raise ValueError("derived manifest dataset version does not match frozen split")
    if raw.get("split") != expected_split:
        raise ValueError("derived manifest split label does not match frozen split")
    allowed_images = {str(value) for value in split["image_ids"]}
    allowed_sessions = {str(value) for value in split.get("session_ids", [])}
    for index, row in enumerate(rows):
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id not in allowed_images:
            raise ValueError(f"row {index} has image_id outside the frozen {expected_split} split")
        if allowed_sessions and row.get("session_id") not in (None, ""):
            if str(row["session_id"]) not in allowed_sessions:
                raise ValueError(
                    f"row {index} has session_id outside the frozen {expected_split} split"
                )
    return split
