import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np

from dashboard.live_camera import LiveDetector
from inference.posture import PosturePrediction
from inference.scene import summarize_scene


def frame():
    result = av.VideoFrame.from_ndarray(np.zeros((48, 64, 3), dtype=np.uint8), format="bgr24")
    result.pts = 1
    result.time_base = Fraction(1, 30)
    return result


def test_live_detection_throttles_and_preserves_video_timing(monkeypatch):
    calls = []

    def detect(image):
        calls.append(image.shape)
        return [{"bbox": [5, 5, 30, 40], "confidence": 0.9}]

    monkeypatch.setitem(sys.modules, "inference.model", SimpleNamespace(detect_people=detect))
    monkeypatch.setattr(
        "dashboard.live_camera.analyze_scene",
        lambda image, detections: summarize_scene(detections, [PosturePrediction("Sitting", 0.91)]),
    )
    monkeypatch.setattr("dashboard.live_camera.time.monotonic", lambda: 10.0)
    processor = LiveDetector()
    result = processor.recv(frame())
    processor.recv(frame())
    assert calls == [(48, 64, 3)]
    assert processor.snapshot()["headcount"] == 1
    assert processor.snapshot()["sitting"] == 1
    assert processor.snapshot()["standing"] == 0
    assert result.pts == 1 and result.time_base == Fraction(1, 30)
    assert result.to_ndarray(format="bgr24").any()
    processor.reset()
    assert processor.snapshot() == {}
    processor.recv(frame())
    assert len(calls) == 2


def test_failure_clears_previous_count(monkeypatch):
    def fail(image):
        raise RuntimeError("unavailable")

    monkeypatch.setitem(sys.modules, "inference.model", SimpleNamespace(detect_people=fail))
    processor = LiveDetector()
    processor.state = {"headcount": 5}
    original = frame()
    assert processor.recv(original) is original
    assert "error" in processor.snapshot()
    assert "headcount" not in processor.snapshot()


def test_zero_people_is_success(monkeypatch):
    monkeypatch.setitem(sys.modules, "inference.model", SimpleNamespace(detect_people=lambda _: []))
    processor = LiveDetector()
    processor.recv(frame())
    assert processor.snapshot()["headcount"] == 0
    assert "error" not in processor.snapshot()


def test_posture_failure_keeps_people_count(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "inference.model",
        SimpleNamespace(detect_people=lambda _: [{"bbox": [5, 5, 30, 40], "confidence": 0.9}]),
    )

    def fail_posture(image, detections):
        raise RuntimeError("unavailable")

    monkeypatch.setattr("dashboard.live_camera.analyze_scene", fail_posture)
    processor = LiveDetector()
    processor.recv(frame())
    state = processor.snapshot()
    assert state["headcount"] == 1
    assert state["sitting"] is None
    assert state["standing"] is None
    assert state["queue_count"] is None
    assert state["posture_error"]


def test_phone_page_requires_access_code(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        "dashboard.live_camera.webrtc_streamer",
        lambda **kwargs: SimpleNamespace(state=SimpleNamespace(playing=False)),
    )
    monkeypatch.setenv("PHONE_CAMERA_ACCESS_CODE", "local-test-code")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard/phone.py").run(
        timeout=20
    )
    assert not app.exception
    assert len(app.subheader) == 0
    app.text_input[0].set_value("wrong")
    app.button[0].click().run(timeout=20)
    assert app.error[0].value == "Incorrect access code."
    assert len(app.subheader) == 0
    app.text_input[0].set_value("local-test-code")
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert app.subheader[0].value == "Phone camera"


def test_phone_page_fails_closed_without_configuration(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.delenv("PHONE_CAMERA_ACCESS_CODE", raising=False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard/phone.py").run(
        timeout=20
    )
    assert not app.exception
    assert "Set PHONE_CAMERA_ACCESS_CODE" in app.error[0].value
    assert len(app.subheader) == 0
