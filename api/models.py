"""Strict schemas for the inference, storage, API, and dashboard boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CameraMetrics(BaseModel):
    """One complete reading from one camera."""

    model_config = ConfigDict(extra="forbid")

    camera_id: str
    timestamp: datetime
    headcount: int = Field(ge=0)
    queue_count: int = Field(ge=0)
    seats_total: int | None = Field(ge=0)
    seats_occupied: int | None = Field(ge=0)
    seat_occupancy_pct: float | None = Field(ge=0, le=100)
    sitting: int | None = Field(default=None, ge=0)
    standing_other_count: int | None = Field(default=None, ge=0)
    scene_method: Literal["calibrated", "distance"] = "calibrated"
    crowd_level: Literal["green", "amber", "red"]
    zone_counts: dict[str, int]
    detections_raw: int = Field(ge=0)
    detections_counted: int = Field(ge=0)
    queue_visible: int | None = Field(default=None, ge=0)
    queue_hidden_estimate: int | None = Field(default=None, ge=0)
    queue_ci_low: int | None = Field(default=None, ge=0)
    queue_ci_high: int | None = Field(default=None, ge=0)
    queue_method: str | None = None
    queue_model_version: str | None = None

    @model_validator(mode="after")
    def validate_relationships(self) -> CameraMetrics:
        if self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must include a timezone")
        if self.queue_hidden_estimate is None and self.queue_count > self.headcount:
            raise ValueError("queue_count cannot exceed headcount")
        if self.queue_visible is not None and self.queue_visible > self.headcount:
            raise ValueError("queue_visible cannot exceed headcount")
        if self.queue_hidden_estimate is not None:
            if self.queue_visible is None:
                raise ValueError("queue_visible is required with hidden correction")
            if self.queue_count != self.queue_visible + self.queue_hidden_estimate:
                raise ValueError("queue_count must equal queue_visible + queue_hidden_estimate")
        if (self.queue_ci_low is None) != (self.queue_ci_high is None):
            raise ValueError("both queue confidence bounds must be present together")
        if self.queue_ci_low is not None and not (
            self.queue_ci_low <= self.queue_count <= self.queue_ci_high  # type: ignore[operator]
        ):
            raise ValueError("queue confidence bounds must contain queue_count")
        if self.scene_method == "distance":
            if self.sitting is None or self.standing_other_count is None:
                raise ValueError("distance counts require sitting and other standing")
            if self.queue_count + self.sitting + self.standing_other_count != self.headcount:
                raise ValueError("exclusive categories must sum to headcount")
        capacity = (self.seats_total, self.seats_occupied, self.seat_occupancy_pct)
        if any(v is None for v in capacity) and not all(v is None for v in capacity):
            raise ValueError("seat capacity metrics must be all present or all unavailable")
        if self.scene_method == "calibrated" and self.seats_total is None:
            raise ValueError("calibrated metrics require seat capacity")
        if self.detections_counted > self.detections_raw:
            raise ValueError("detections_counted cannot exceed detections_raw")
        if any(value < 0 for value in self.zone_counts.values()):
            raise ValueError("zone_counts cannot contain negative values")
        if self.seats_total is not None and self.seats_occupied is not None:
            if self.seats_occupied > self.seats_total:
                raise ValueError("seats_occupied cannot exceed seats_total")
            expected = (
                round(100 * self.seats_occupied / self.seats_total, 1) if self.seats_total else 0.0
            )
            if self.seat_occupancy_pct is None or abs(self.seat_occupancy_pct - expected) > 0.05:
                raise ValueError("seat_occupancy_pct does not match occupied/total seats")
        return self


class IngestResponse(BaseModel):
    status: Literal["ok"]
    metrics: CameraMetrics


class CameraStatus(BaseModel):
    camera_id: str
    online: bool
    metrics: CameraMetrics | None = None


class StatusResponse(BaseModel):
    cameras: dict[str, CameraStatus]


class HistoryPoint(BaseModel):
    timestamp: datetime
    headcount: int = Field(ge=0)
    queue_count: int = Field(ge=0)
    seats_occupied: int | None = Field(ge=0)
    seats_total: int | None = Field(ge=0)
    seat_occupancy_pct: float | None = Field(ge=0, le=100)
    sitting: int | None = Field(default=None, ge=0)
    standing_other_count: int | None = Field(default=None, ge=0)
    scene_method: Literal["calibrated", "distance"] = "calibrated"
    crowd_level: Literal["green", "amber", "red"]
    queue_visible: int | None = Field(default=None, ge=0)
    queue_hidden_estimate: int | None = Field(default=None, ge=0)
    queue_ci_low: int | None = Field(default=None, ge=0)
    queue_ci_high: int | None = Field(default=None, ge=0)
    queue_method: str | None = None
    queue_model_version: str | None = None


class HistoryResponse(BaseModel):
    camera_id: str
    minutes: int
    points: list[HistoryPoint]
