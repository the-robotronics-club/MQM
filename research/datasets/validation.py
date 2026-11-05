"""Validate dataset files, labels, boxes, split leakage, and duplicate content."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image

from .loader import load_manifest, load_split
from .schema import (
    VALID_CROWD_BANDS,
    VALID_MEAL_PERIODS,
    VALID_OCCLUSIONS,
    VALID_ROLES,
    DatasetManifest,
)


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_manifest(
    manifest: DatasetManifest, root: Path, split_paths: list[Path] | None = None
) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    hashes: dict[str, str] = {}
    image_hashes: dict[str, str] = {}
    for record in manifest.images:
        if not record.image_id or record.image_id in seen_ids:
            errors.append(f"duplicate or empty image_id: {record.image_id!r}")
        seen_ids.add(record.image_id)
        if not record.session_id:
            errors.append(f"{record.image_id}: session_id is required")
        if not record.camera_id:
            errors.append(f"{record.image_id}: camera_id is required")
        try:
            timestamp = datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
            if timestamp.utcoffset() is None:
                raise ValueError("timezone is missing")
        except ValueError:
            errors.append(f"{record.image_id}: timestamp must be timezone-aware ISO 8601")
        if record.meal_period not in VALID_MEAL_PERIODS:
            errors.append(f"{record.image_id}: invalid meal_period {record.meal_period!r}")
        if record.crowd_band not in VALID_CROWD_BANDS:
            errors.append(f"{record.image_id}: invalid crowd_band {record.crowd_band!r}")
        if record.queue_ground_truth < 0:
            errors.append(f"{record.image_id}: queue_ground_truth must be non-negative")
        queued_labels = sum(item.role == "queue_member" for item in record.annotations)
        if queued_labels > record.queue_ground_truth:
            warnings.append(
                f"{record.image_id}: {queued_labels} visible queue labels exceed "
                f"queue ground truth {record.queue_ground_truth}"
            )
        image_path = root / record.path
        if not image_path.is_file():
            errors.append(f"{record.image_id}: image does not exist: {image_path}")
            continue
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        image_hashes[record.image_id] = digest
        if digest in hashes:
            warnings.append(f"duplicate image content: {record.image_id} and {hashes[digest]}")
        else:
            hashes[digest] = record.image_id
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError as exc:
            errors.append(f"{record.image_id}: unreadable image: {exc}")
            continue
        for index, annotation in enumerate(record.annotations):
            x1, y1, x2, y2 = annotation.bbox
            if annotation.role not in VALID_ROLES:
                errors.append(f"{record.image_id}[{index}]: invalid role {annotation.role!r}")
            if annotation.occlusion not in VALID_OCCLUSIONS:
                errors.append(
                    f"{record.image_id}[{index}]: invalid occlusion {annotation.occlusion!r}"
                )
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                errors.append(f"{record.image_id}[{index}]: bbox outside {width}x{height}")
    if split_paths:
        split_images: dict[str, str] = {}
        split_sessions: dict[str, str] = {}
        split_content: dict[str, tuple[str, str]] = {}
        split_clips: dict[tuple[str, str], str] = {}
        known = {item.image_id: item for item in manifest.images}
        for split_path in split_paths:
            split = load_split(split_path)
            name = str(split.get("name", split_path.stem))
            if split.get("dataset_version") != manifest.version:
                errors.append(f"{name}: dataset version does not match manifest")
            declared_sessions = {str(value) for value in split.get("session_ids", [])}
            for image_id in split["image_ids"]:
                if image_id not in known:
                    errors.append(f"{name}: unknown image_id {image_id}")
                    continue
                if image_id in split_images:
                    errors.append(
                        f"image {image_id} appears in {split_images[image_id]} and {name}"
                    )
                split_images[image_id] = name
                record = known[image_id]
                image_digest = image_hashes.get(image_id)
                if image_digest is not None:
                    previous = split_content.get(image_digest)
                    if previous is not None and previous[0] != name:
                        errors.append(
                            f"duplicate image content crosses splits: {previous[1]} "
                            f"in {previous[0]} and {image_id} in {name}"
                        )
                    else:
                        split_content[image_digest] = (name, image_id)
                if record.clip_id:
                    clip = (record.camera_id, record.clip_id)
                    if clip in split_clips and split_clips[clip] != name:
                        errors.append(
                            f"camera clip {clip} appears in {split_clips[clip]} and {name}"
                        )
                    split_clips[clip] = name
                session = record.session_id
                if declared_sessions and session not in declared_sessions:
                    errors.append(f"{name}: image {image_id} is outside its declared sessions")
                if session in split_sessions and split_sessions[session] != name:
                    errors.append(
                        f"session {session} appears in {split_sessions[session]} and {name}"
                    )
                split_sessions[session] = name
        if {path.stem for path in split_paths} >= {"train", "val", "test"}:
            missing = sorted(set(known) - set(split_images))
            if missing:
                errors.append(f"split files omit {len(missing)} manifest image(s)")
    return ValidationReport(tuple(errors), tuple(warnings))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--splits", type=Path)
    args = parser.parse_args()
    paths = sorted(args.splits.glob("*.json")) if args.splits else None
    report = validate_manifest(load_manifest(args.manifest), args.root, paths)
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    raise SystemExit(0 if report.valid else 1)


if __name__ == "__main__":
    main()
