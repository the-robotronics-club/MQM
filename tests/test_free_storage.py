from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from api.config import get_settings, reset_settings_cache
from api.main import app
from api.storage import sqlite_client as store


@pytest.fixture
def sqlite_env(configured_env, monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SNAPSHOT_SQLITE_PATH", str(tmp_path / "readings.db"))
    reset_settings_cache()


def reading(count=4):
    return {
        "camera_id": "mess_main",
        "timestamp": datetime.now(UTC).isoformat(),
        "headcount": count,
        "queue_count": 3,
        "sitting": count - 3,
        "standing_other_count": 0,
        "scene_method": "distance",
        "seats_total": None,
        "seats_occupied": None,
        "seat_occupancy_pct": None,
        "crowd_level": "green",
        "zone_counts": {},
        "detections_raw": count,
        "detections_counted": count,
    }


def test_sqlite_works_without_database_credentials(sqlite_env, monkeypatch):
    for key in ("REDIS_URL", "INFLUX_URL", "INFLUX_TOKEN", "INFLUX_ORG", "INFLUX_BUCKET"):
        monkeypatch.delenv(key, raising=False)
    reset_settings_cache()
    assert get_settings().storage_backend == "sqlite"
    assert store.check()


def test_latest_expiry_does_not_remove_history(sqlite_env, monkeypatch):
    clock = 1_000_000.0
    monkeypatch.setattr(store.time, "time", lambda: clock)
    metrics = reading()
    store.save("mess_main", metrics)
    assert store.latest(("mess_main",)) == {"mess_main": metrics}
    clock += get_settings().latest_ttl_seconds + 1
    assert store.latest(("mess_main",)) == {}
    assert store.history("mess_main", 60) == [metrics]


def test_sqlite_api_ingest_status_and_history(sqlite_env, monkeypatch):
    import api.ingestion as ingestion

    raw = reading()
    raw.pop("timestamp")
    monkeypatch.setattr(ingestion, "process_frame", lambda *_: raw)
    with TestClient(app) as client:
        assert client.get("/ready").json()["services"] == {"sqlite": True}
        response = client.post(
            "/ingest/mess_main",
            headers={"X-Camera-Key": "test-camera-key"},
            files={"file": ("frame.jpg", b"\xff\xd8\xfftest", "image/jpeg")},
        )
        assert response.status_code == 200, response.text
        current = client.get("/status/mess_main").json()
        history = client.get("/history/mess_main").json()["points"]
        assert current["metrics"]["headcount"] == 4
        assert history[0]["timestamp"] == current["metrics"]["timestamp"]
        assert history[0]["sitting"] == 1


def test_old_history_is_pruned(sqlite_env, monkeypatch):
    clock = 1_000_000.0
    monkeypatch.setattr(store.time, "time", lambda: clock)
    store.save("mess_main", reading())
    clock += 86401
    new = reading(5)
    store.save("mess_main", new)
    assert store.history("mess_main", 1440) == [new]
