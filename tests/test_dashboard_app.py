from __future__ import annotations

from pathlib import Path

import requests
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def run_app(monkeypatch, base_url: str, get):
    monkeypatch.setenv("READ_API_BASE_URL", base_url)
    monkeypatch.setenv("DASHBOARD_DEFAULT_MODE", "Live cameras")
    monkeypatch.setenv("DASHBOARD_SNAPSHOT_ONLY", "false")
    monkeypatch.setattr(requests, "get", get)
    app = AppTest.from_file(APP).run(timeout=20)
    return app.segmented_control[0].set_value("Live cameras").run(timeout=20)


def test_upload_mode_does_not_require_api(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Photo upload must not request live camera status")

    monkeypatch.setattr(requests, "get", fail)
    monkeypatch.setenv("DASHBOARD_DEFAULT_MODE", "Upload image")
    monkeypatch.setenv("DASHBOARD_SNAPSHOT_ONLY", "false")
    app = AppTest.from_file(APP).run(timeout=20)
    assert not app.exception
    assert len(app.get("file_uploader")) == 1
    assert len(app.selectbox) == 1
    assert next(b for b in app.button if b.label == "Analyze image").disabled
    assert not app.error


def test_dashboard_renders_offline_camera(monkeypatch):
    def get(url, timeout):
        assert url == "http://offline-api/status"
        return Response(
            {"cameras": {"mess_main": {"camera_id": "mess_main", "online": False, "metrics": None}}}
        )

    app = run_app(monkeypatch, "http://offline-api", get)
    assert not app.exception
    assert "MQM" in app.title[0].value
    assert "North Dining Hall" in app.subheader[0].value
    assert not app.metric
    assert "Camera offline" in app.error[0].value


def test_dashboard_renders_metrics_and_history_chart(monkeypatch):
    metrics = {
        "camera_id": "mess_main",
        "timestamp": "2026-08-27T12:00:00Z",
        "headcount": 12,
        "queue_count": 3,
        "seats_total": 64,
        "seats_occupied": 8,
        "seat_occupancy_pct": 12.5,
        "crowd_level": "amber",
        "zone_counts": {},
        "detections_raw": 12,
        "detections_counted": 12,
    }

    def get(url, timeout):
        if url == "http://online-api/status":
            return Response(
                {
                    "cameras": {
                        "mess_main": {
                            "camera_id": "mess_main",
                            "online": True,
                            "metrics": metrics,
                        }
                    }
                }
            )
        assert url == "http://online-api/history/mess_main?minutes=60"
        return Response({"points": [metrics]})

    app = run_app(monkeypatch, "http://online-api", get)
    assert not app.exception
    assert [(element.label, element.value) for element in app.metric] == [
        ("Headcount", "12"),
        ("Queue", "3"),
        ("Seats", "8/64"),
        ("Occupied", "12.5%"),
    ]
    assert len(app.get("plotly_chart")) == 1
    assert any("Moderate Crowd" in element.value for element in app.markdown)


def test_dashboard_surfaces_read_api_failure(monkeypatch):
    def get(*_args, **_kwargs):
        raise requests.ConnectionError("offline")

    app = run_app(monkeypatch, "http://broken-api", get)
    assert not app.exception
    assert "Read API request failed" in app.error[0].value


def test_hosted_default_view_shows_snapshot_counts_without_occupancy(monkeypatch):
    monkeypatch.setenv("DASHBOARD_DEFAULT_MODE", "Live cameras")
    monkeypatch.setenv("READ_API_BASE_URL", "http://snapshot-api")
    monkeypatch.setenv("DASHBOARD_SNAPSHOT_ONLY", "true")
    metrics = {
        "camera_id": "mess_main",
        "timestamp": "2026-10-22T12:00:00Z",
        "scene_method": "distance",
        "headcount": 39,
        "queue_count": 25,
        "sitting": 13,
        "standing_other_count": 1,
        "seats_total": None,
        "seats_occupied": None,
        "seat_occupancy_pct": None,
    }

    def get(url, timeout):
        if url.endswith("/status"):
            return Response({"cameras": {"mess_main": {"online": True, "metrics": metrics}}})
        return Response({"points": []})

    monkeypatch.setattr(requests, "get", get)
    app = AppTest.from_file(APP).run(timeout=20)
    assert not app.exception
    assert [(m.label, m.value) for m in app.metric] == [
        ("People detected", "39"),
        ("Queue estimate", "25"),
        ("Sitting estimate", "13"),
        ("Other standing", "1"),
    ]


def test_render_dashboard_uses_private_api_host(monkeypatch):
    from dashboard.app import api_base_url

    monkeypatch.delenv("READ_API_BASE_URL", raising=False)
    monkeypatch.setenv("READ_API_HOST", "api.internal")
    monkeypatch.setenv("READ_API_PORT", "10000")
    assert api_base_url() == "http://api.internal:10000"


def test_photo_results_show_distance_queue_and_color_legend(monkeypatch):
    from io import BytesIO

    from PIL import Image

    monkeypatch.setenv("DASHBOARD_DEFAULT_MODE", "Upload image")
    monkeypatch.setenv("DASHBOARD_SNAPSHOT_ONLY", "false")
    output = BytesIO()
    Image.new("RGB", (100, 80)).save(output, format="PNG")
    app = AppTest.from_file(APP).run(timeout=20)
    app.session_state["photo_selection"] = (None, None, 0)
    app.session_state["photo_result"] = {
        "name": "test.png",
        "headcount": 6,
        "queue_count": 4,
        "sitting": 1,
        "standing_other_count": 1,
        "scene": {"method": "distance"},
        "posture_error": None,
        "width": 100,
        "height": 80,
        "metrics": None,
        "image": output.getvalue(),
    }
    app.run(timeout=20)
    assert not app.exception
    assert [(m.label, m.value) for m in app.metric] == [
        ("People detected", "6"),
        ("Queue estimate", "4"),
        ("Sitting estimate", "1"),
        ("Other standing", "1"),
    ]
    assert any("Blue Q" in caption.value for caption in app.caption)


def test_hosted_comparison_marks_offline_and_ignores_stale_counts(monkeypatch):
    monkeypatch.setenv("DASHBOARD_SNAPSHOT_ONLY", "true")
    monkeypatch.setenv("DASHBOARD_DEFAULT_MODE", "All messes comparison")
    monkeypatch.setenv("READ_API_BASE_URL", "http://comparison-api")
    monkeypatch.setattr(
        requests,
        "get",
        lambda *a, **kw: Response(
            {
                "cameras": {
                    "mess_main": {
                        "online": False,
                        "metrics": {"headcount": 999, "queue_count": 500},
                    }
                }
            }
        ),
    )
    app = AppTest.from_file(APP).run(timeout=20)
    assert not app.exception
    assert not app.metric
    assert len(app.warning) == 3
    assert list(app.segmented_control[0].options) == ["Live cameras", "All messes comparison"]
    next(b for b in app.button if b.key == "select_mess_btn_south_mess").click().run()
    assert not app.exception
    assert app.segmented_control[0].value == "Live cameras"
    assert app.segmented_control[1].value == "south_mess"
    assert "Camera offline" in app.error[0].value


def test_demo_profile_selects_assigned_mess(monkeypatch):
    monkeypatch.setenv("DASHBOARD_SNAPSHOT_ONLY", "true")
    monkeypatch.setenv("DASHBOARD_DEFAULT_MODE", "Live cameras")
    monkeypatch.setenv("READ_API_BASE_URL", "http://demo-profile-api")
    monkeypatch.setattr(requests, "get", lambda *a, **kw: Response({"cameras": {}}))
    app = AppTest.from_file(APP).run(timeout=20)
    next(b for b in app.button if b.key == "demo_btn_2024EE205").click().run()
    assert not app.exception
    assert app.segmented_control[1].value == "south_mess"
    assert "South Dining Hall" in app.subheader[0].value


def test_unknown_student_is_not_authenticated(monkeypatch):
    from dashboard.student_auth import login_student

    state = {}
    monkeypatch.setattr("dashboard.student_auth.st.session_state", state)
    ok, _ = login_student("UNKNOWN", "9999")
    assert not ok
    assert state["student_profile"] is None
    assert not state["is_authenticated"]
