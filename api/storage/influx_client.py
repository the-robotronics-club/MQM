"""InfluxDB 2.x/Cloud adapter for timestamped metrics history."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from api.config import get_settings

MEASUREMENT = "mess_metrics"


@lru_cache(maxsize=1)
def get_influx_client() -> InfluxDBClient:
    settings = get_settings()
    return InfluxDBClient(
        url=settings.influx_url,
        token=settings.influx_token,
        org=settings.influx_org,
        timeout=10_000,
    )


def write_metrics(camera_id: str, metrics: dict, timestamp: datetime | None = None) -> None:
    """Synchronously persist one full reading, raising on any write failure."""
    settings = get_settings()
    if settings.storage_backend == "sqlite":
        from api.storage.sqlite_client import save

        save(camera_id, metrics)
        return
    timestamp = timestamp or datetime.now(UTC)
    point = (
        Point(MEASUREMENT)
        .tag("camera_id", camera_id)
        .field("headcount", int(metrics["headcount"]))
        .field("queue_count", int(metrics["queue_count"]))
        .field("crowd_level", str(metrics["crowd_level"]))
        .field("detections_raw", int(metrics["detections_raw"]))
        .field("detections_counted", int(metrics["detections_counted"]))
        .field("zone_counts", json.dumps(metrics["zone_counts"], separators=(",", ":")))
        .time(timestamp, WritePrecision.NS)
    )
    for name in (
        "seats_occupied",
        "seats_total",
        "seat_occupancy_pct",
        "sitting",
        "standing_other_count",
        "scene_method",
        "queue_visible",
        "queue_hidden_estimate",
        "queue_ci_low",
        "queue_ci_high",
        "queue_method",
        "queue_model_version",
    ):
        value = metrics.get(name)
        if value is not None:
            point = point.field(name, value)
    get_influx_client().write_api(write_options=SYNCHRONOUS).write(
        bucket=settings.influx_bucket,
        org=settings.influx_org,
        record=point,
    )


def query_history(camera_id: str, minutes: int = 60) -> list[dict]:
    """Return a camera's readings oldest-first over a bounded lookback."""
    settings = get_settings()
    if settings.storage_backend == "sqlite":
        from api.storage.sqlite_client import history

        return history(camera_id, minutes)
    quoted_camera_id = json.dumps(camera_id)
    query = f"""
from(bucket: {json.dumps(settings.influx_bucket)})
  |> range(start: -{int(minutes)}m)
  |> filter(fn: (r) => r._measurement == {json.dumps(MEASUREMENT)})
  |> filter(fn: (r) => r.camera_id == {quoted_camera_id})
  |> filter(fn: (r) => contains(value: r._field, set: [
      "headcount", "queue_count", "seats_occupied", "seats_total",
      "seat_occupancy_pct", "crowd_level", "queue_visible",
      "queue_hidden_estimate", "queue_ci_low", "queue_ci_high",
      "queue_method", "queue_model_version", "sitting", "standing_other_count", "scene_method"
  ]))
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
"""
    tables = get_influx_client().query_api().query(query=query, org=settings.influx_org)
    points: list[dict] = []
    for table in tables:
        for record in table.records:
            values = record.values
            point = {
                "timestamp": record.get_time(),
                "headcount": int(values["headcount"]),
                "queue_count": int(values["queue_count"]),
                "seats_occupied": values.get("seats_occupied"),
                "seats_total": values.get("seats_total"),
                "seat_occupancy_pct": values.get("seat_occupancy_pct"),
                "crowd_level": str(values["crowd_level"]),
            }
            for name in (
                "sitting",
                "standing_other_count",
                "scene_method",
                "queue_visible",
                "queue_hidden_estimate",
                "queue_ci_low",
                "queue_ci_high",
                "queue_method",
                "queue_model_version",
            ):
                if values.get(name) is not None:
                    point[name] = values[name]
            points.append(point)
    points.sort(key=lambda point: point["timestamp"])
    return points


def check_influx() -> bool:
    if get_settings().storage_backend == "sqlite":
        from api.storage.sqlite_client import check

        return check()
    return get_influx_client().health().status == "pass"


def reset_influx_client() -> None:
    client = get_influx_client.cache_info().currsize and get_influx_client()
    get_influx_client.cache_clear()
    if client:
        client.close()
