"""Validated environment configuration shared by the API and dashboard."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ZONE_CONFIG_PATH = REPO_ROOT / "config" / "zones.json"

load_dotenv(REPO_ROOT / ".env")


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is absent or invalid."""


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value or value.startswith("change-me"):
        raise ConfigurationError(f"{name} is required; copy .env.example to .env and set it")
    return value


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return value


def _influx_url() -> str:
    """Render supplies the private service hostname separately from its URL."""
    configured = os.environ.get("INFLUX_URL", "").strip()
    if configured:
        return _required("INFLUX_URL")
    host = os.environ.get("INFLUX_HOST", "").strip()
    return f"http://{host}:8086" if host else _required("INFLUX_URL")


def _choice(name: str, default: str, choices: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        raise ConfigurationError(f"{name} must be one of: {', '.join(sorted(choices))}")
    return value


def configured_camera_ids() -> tuple[str, ...]:
    """Return env-selected cameras, or every camera declared in zones.json."""
    try:
        cameras = json.loads(ZONE_CONFIG_PATH.read_text(encoding="utf-8"))["cameras"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read camera configuration: {exc}") from exc

    raw = os.environ.get("KNOWN_CAMERAS", "")
    selected = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    selected = selected or tuple(sorted(cameras))
    unknown = sorted(set(selected) - set(cameras))
    if unknown:
        raise ConfigurationError(
            f"KNOWN_CAMERAS contains cameras without zone configuration: {', '.join(unknown)}"
        )
    if not selected:
        raise ConfigurationError("no cameras are configured")
    return selected


def configured_cors_origins() -> tuple[str, ...]:
    return tuple(
        value.strip()
        for value in os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:8501").split(",")
        if value.strip()
    )


@dataclass(frozen=True)
class Settings:
    redis_url: str
    influx_url: str
    influx_token: str
    influx_org: str
    influx_bucket: str
    camera_api_key: str
    known_cameras: tuple[str, ...]
    latest_ttl_seconds: int
    read_cache_ttl_seconds: int
    max_upload_bytes: int
    cors_allow_origins: tuple[str, ...]
    queue_estimator: str
    frame_metrics_mode: str
    storage_backend: str
    inference_backend: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    backend = _choice("STORAGE_BACKEND", "external", {"external", "sqlite"})
    return Settings(
        redis_url=_required("REDIS_URL") if backend == "external" else "",
        influx_url=_influx_url() if backend == "external" else "",
        influx_token=_required("INFLUX_TOKEN") if backend == "external" else "",
        influx_org=_required("INFLUX_ORG") if backend == "external" else "",
        influx_bucket=_required("INFLUX_BUCKET") if backend == "external" else "",
        camera_api_key=_required("CAMERA_API_KEY"),
        known_cameras=configured_camera_ids(),
        latest_ttl_seconds=_positive_int("LATEST_TTL_SECONDS", 30),
        read_cache_ttl_seconds=_positive_int("READ_CACHE_TTL_SECONDS", 5),
        max_upload_bytes=_positive_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
        cors_allow_origins=configured_cors_origins(),
        queue_estimator=_choice(
            "QUEUE_ESTIMATOR",
            "production",
            {"production", "geometric", "membership", "occlusion"},
        ),
        frame_metrics_mode=_choice("FRAME_METRICS_MODE", "calibrated", {"calibrated", "distance"}),
        storage_backend=backend,
        inference_backend=_choice("INFERENCE_BACKEND", "torch", {"torch", "onnx"}),
    )


def reset_settings_cache() -> None:
    """Clear cached environment-derived settings (used by tests and reloads)."""
    get_settings.cache_clear()
