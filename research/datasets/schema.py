"""Explicit annotation schema; labels never come from directory names."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

Role = Literal["queue_member", "passerby", "staff", "other_nonqueue", "uncertain"]
Occlusion = Literal["none", "partial", "heavy"]
MealPeriod = Literal["breakfast", "lunch", "dinner"]
CrowdBand = Literal["low", "medium", "high"]

VALID_ROLES = {"queue_member", "passerby", "staff", "other_nonqueue", "uncertain"}
VALID_OCCLUSIONS = {"none", "partial", "heavy"}
VALID_MEAL_PERIODS = {"breakfast", "lunch", "dinner"}
VALID_CROWD_BANDS = {"low", "medium", "high"}


@dataclass(frozen=True)
class PersonAnnotation:
    bbox: tuple[float, float, float, float]
    role: Role
    occlusion: Occlusion
    truncation: bool
    track_id: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PersonAnnotation:
        bbox = tuple(float(value) for value in raw["bbox"])
        if len(bbox) != 4:
            raise ValueError("annotation bbox must contain four coordinates")
        return cls(
            bbox=cast(tuple[float, float, float, float], bbox),
            role=raw["role"],
            occlusion=raw["occlusion"],
            truncation=bool(raw.get("truncation", False)),
            track_id=str(raw["track_id"]) if raw.get("track_id") is not None else None,
        )


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    path: str
    camera_id: str
    session_id: str
    clip_id: str | None
    timestamp: str
    meal_period: MealPeriod
    crowd_band: CrowdBand
    queue_ground_truth: int
    annotations: tuple[PersonAnnotation, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ImageRecord:
        return cls(
            image_id=str(raw["image_id"]),
            path=str(raw["path"]),
            camera_id=str(raw["camera_id"]),
            session_id=str(raw["session_id"]),
            clip_id=str(raw["clip_id"]) if raw.get("clip_id") is not None else None,
            timestamp=str(raw["timestamp"]),
            meal_period=raw["meal_period"],
            crowd_band=raw["crowd_band"],
            queue_ground_truth=int(raw["queue_ground_truth"]),
            annotations=tuple(
                PersonAnnotation.from_dict(item) for item in raw.get("annotations", [])
            ),
        )


@dataclass(frozen=True)
class DatasetManifest:
    version: str
    images: tuple[ImageRecord, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetManifest:
        return cls(
            str(raw.get("version", "unversioned")),
            tuple(ImageRecord.from_dict(item) for item in raw["images"]),
        )
