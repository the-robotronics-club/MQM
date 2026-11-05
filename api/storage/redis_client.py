"""Redis adapter for expiring latest-camera snapshots."""

from __future__ import annotations

import json
from functools import lru_cache

import redis

from api.config import get_settings

LATEST_KEY_PREFIX = "latest:"


@lru_cache(maxsize=1)
def get_redis_client() -> redis.Redis:
    return redis.Redis.from_url(get_settings().redis_url, decode_responses=True)


def latest_key(camera_id: str) -> str:
    return f"{LATEST_KEY_PREFIX}{camera_id}"


def set_latest(camera_id: str, metrics: dict) -> None:
    settings = get_settings()
    if settings.storage_backend == "sqlite":
        return  # The transactional history write also updates the derived latest view.
    get_redis_client().set(
        latest_key(camera_id),
        json.dumps(metrics, separators=(",", ":")),
        ex=settings.latest_ttl_seconds,
    )


def get_latest(camera_id: str) -> dict | None:
    if get_settings().storage_backend == "sqlite":
        return get_all_latest((camera_id,)).get(camera_id)
    raw = get_redis_client().get(latest_key(camera_id))
    return json.loads(raw) if raw is not None else None


def get_all_latest(camera_ids: tuple[str, ...]) -> dict[str, dict]:
    """Fetch every configured latest key in one Redis MGET call."""
    if not camera_ids:
        return {}
    if get_settings().storage_backend == "sqlite":
        from api.storage.sqlite_client import latest

        return latest(camera_ids)
    values = get_redis_client().mget([latest_key(camera_id) for camera_id in camera_ids])
    return {
        camera_id: json.loads(raw)
        for camera_id, raw in zip(camera_ids, values, strict=True)
        if raw is not None
    }


def check_redis() -> bool:
    if get_settings().storage_backend == "sqlite":
        from api.storage.sqlite_client import check

        return check()
    return bool(get_redis_client().ping())


def reset_redis_client() -> None:
    client = get_redis_client.cache_info().currsize and get_redis_client()
    get_redis_client.cache_clear()
    if client:
        client.close()
