from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.ingestion as ingestion
import api.main as api_main
import api.read as read_api
from api.main import app
from inference.metrics import compute_metrics
from inference.pipeline import InvalidImageError
from inference.zones import load_zones
from tests import fixtures as fx

ROOT = Path(__file__).resolve().parent.parent
JPEG = (ROOT / "data" / "samples" / "mess_hall_dense.jpg").read_bytes()


def sample_metrics() -> dict:
    return compute_metrics(fx.BUSY, load_zones("mess_main"), frame_size=(736, 490))


@pytest.fixture
def client(configured_env, monkeypatch):
    stored = {"latest": {}, "history": []}
    monkeypatch.setattr(ingestion, "process_frame", lambda *_: sample_metrics())

    def write(camera_id, metrics, timestamp=None):
        stored["history"].append({**metrics, "timestamp": timestamp})

    def cache(camera_id, metrics):
        stored["latest"][camera_id] = metrics

    monkeypatch.setattr(ingestion, "write_metrics", write)
    monkeypatch.setattr(ingestion, "set_latest", cache)
    monkeypatch.setattr(
        read_api,
        "get_all_latest",
        lambda camera_ids: {
            key: stored["latest"][key] for key in camera_ids if key in stored["latest"]
        },
    )
    monkeypatch.setattr(read_api, "get_latest", lambda camera_id: stored["latest"].get(camera_id))
    monkeypatch.setattr(read_api, "query_history", lambda *_: stored["history"])
    with TestClient(app) as test_client:
        yield test_client, stored


def ingest(client: TestClient, **kwargs):
    headers = kwargs.pop("headers", {"X-Camera-Key": "test-camera-key"})
    files = kwargs.pop("files", {"file": ("frame.jpg", JPEG, "image/jpeg")})
    return client.post("/ingest/mess_main", headers=headers, files=files, **kwargs)


def test_health_is_liveness_only(client):
    response = client[0].get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_checks_both_dependencies(client, monkeypatch):
    monkeypatch.setattr(api_main, "check_redis", lambda: True)
    monkeypatch.setattr(api_main, "check_influx", lambda: True)
    response = client[0].get("/ready")
    assert response.status_code == 200
    assert response.json()["services"] == {"redis": True, "influxdb": True}


def test_readiness_returns_503_for_dependency_failure(client, monkeypatch):
    monkeypatch.setattr(api_main, "check_redis", lambda: False)
    monkeypatch.setattr(api_main, "check_influx", lambda: True)
    response = client[0].get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_ingest_then_read_current_and_history(client):
    test_client, _ = client
    response = ingest(test_client)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["metrics"]["headcount"] == 12

    all_status = test_client.get("/status").json()["cameras"]["mess_main"]
    assert all_status["online"] is True
    assert all_status["metrics"]["queue_count"] == 3

    one_status = test_client.get("/status/mess_main").json()
    assert one_status["online"] is True
    history = test_client.get("/history/mess_main?minutes=60").json()
    assert len(history["points"]) == 1
    assert history["points"][0]["headcount"] == 12


def test_camera_is_explicitly_offline_without_latest_reading(client):
    response = client[0].get("/status")
    status = response.json()["cameras"]["mess_main"]
    assert status == {"camera_id": "mess_main", "online": False, "metrics": None}


def test_ingest_requires_camera_key(client):
    response = ingest(client[0], headers={})
    assert response.status_code == 401


def test_ingest_rejects_unknown_camera(client):
    response = client[0].post(
        "/ingest/unknown",
        headers={"X-Camera-Key": "test-camera-key"},
        files={"file": ("frame.jpg", JPEG, "image/jpeg")},
    )
    assert response.status_code == 404


def test_ingest_rejects_wrong_media_type(client):
    response = ingest(client[0], files={"file": ("frame.png", b"data", "image/png")})
    assert response.status_code == 415


def test_ingest_rejects_corrupt_frame(client, monkeypatch):
    def fail(*_):
        raise InvalidImageError("uploaded file is not a valid JPEG")

    monkeypatch.setattr(ingestion, "process_frame", fail)
    response = ingest(client[0])
    assert response.status_code == 422
    assert "valid JPEG" in response.json()["detail"]


def test_ingest_surfaces_inference_outage(client, monkeypatch):
    def fail(*_):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(ingestion, "process_frame", fail)
    response = ingest(client[0])
    assert response.status_code == 503
    assert response.json()["detail"] == "Inference is unavailable"


def test_ingest_surfaces_storage_failure(client, monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "write_metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("down")),
    )
    response = ingest(client[0])
    assert response.status_code == 503
    assert "could not be persisted" in response.json()["detail"]


def test_unknown_read_camera_and_invalid_history_window(client):
    assert client[0].get("/status/unknown").status_code == 404
    assert client[0].get("/history/mess_main?minutes=0").status_code == 422
    assert client[0].get("/history/mess_main?minutes=1441").status_code == 422


def test_read_store_failures_are_explicit_503(client, monkeypatch):
    monkeypatch.setattr(
        read_api, "get_all_latest", lambda *_: (_ for _ in ()).throw(RuntimeError("down"))
    )
    response = client[0].get("/status")
    assert response.status_code == 503


def test_corrupt_cached_schema_is_an_explicit_503(client, monkeypatch):
    bad = sample_metrics()
    bad.update(timestamp=datetime.now(UTC).isoformat(), queue_count=999)
    monkeypatch.setattr(read_api, "get_all_latest", lambda *_: {"mess_main": bad})
    response = client[0].get("/status")
    assert response.status_code == 503


def test_schema_rejects_impossible_metrics():
    from pydantic import ValidationError

    from api.models import CameraMetrics

    impossible = sample_metrics()
    impossible["seat_occupancy_pct"] = 101
    with pytest.raises(ValidationError):
        CameraMetrics(**impossible, timestamp=datetime.now(UTC))


def test_distance_snapshot_roundtrip_preserves_categories_and_unavailable_capacity(
    client, monkeypatch
):
    metrics = sample_metrics()
    metrics.update(
        scene_method="distance",
        sitting=8,
        standing_other_count=1,
        seats_total=None,
        seats_occupied=None,
        seat_occupancy_pct=None,
    )
    monkeypatch.setattr(ingestion, "process_frame", lambda *_: metrics)
    test_client, _ = client
    response = ingest(test_client)
    assert response.status_code == 200, response.text
    latest = test_client.get("/status/mess_main").json()["metrics"]
    history = test_client.get("/history/mess_main").json()["points"][0]
    for reading in (latest, history):
        assert reading["headcount"] == 12
        assert reading["sitting"] == 8
        assert reading["standing_other_count"] == 1
        assert reading["seats_total"] is None
        assert reading["scene_method"] == "distance"


def test_distance_schema_rejects_counts_that_do_not_sum():
    from pydantic import ValidationError

    from api.models import CameraMetrics

    metrics = sample_metrics()
    metrics.update(scene_method="distance", sitting=12, standing_other_count=12)
    with pytest.raises(ValidationError, match="sum to headcount"):
        CameraMetrics(**metrics, timestamp=datetime.now(UTC))


def test_render_private_history_host_is_expanded_to_http(configured_env, monkeypatch):
    from api.config import get_settings, reset_settings_cache

    monkeypatch.delenv("INFLUX_URL")
    monkeypatch.setenv("INFLUX_HOST", "history.internal")
    reset_settings_cache()
    assert get_settings().influx_url == "http://history.internal:8086"
