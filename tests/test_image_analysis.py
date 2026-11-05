from __future__ import annotations

import sys
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from dashboard.image_analysis import MAX_IMAGE_BYTES, analyze_photo, decode_photo


@pytest.fixture(autouse=True)
def photo_classifier_configuration(monkeypatch):
    # Load the dependency before tests replace the detector module with a stub.
    from inference import classifiers

    monkeypatch.setenv("ATTRIBUTE_CLASSIFIERS_ENABLED", "false")
    assert classifiers.attribute_classifiers_enabled() is False


def photo_bytes(size=(80, 60), format="PNG"):
    output = BytesIO()
    Image.new("RGB", size, "red").save(output, format=format)
    return output.getvalue()


@pytest.mark.parametrize("data", [b"", b"not an image", b"x" * (MAX_IMAGE_BYTES + 1)])
def test_invalid_upload_is_rejected(data):
    with pytest.raises(ValueError):
        decode_photo(data)


@pytest.mark.parametrize("format", ["JPEG", "PNG"])
def test_photo_analysis_uses_original_resolution_and_bgr(monkeypatch, format):
    def detect(frame):
        assert frame.shape == (60, 80, 3)
        assert frame[0, 0, 2] > 250
        assert frame[0, 0, 0] < 5
        return [{"bbox": [5, 5, 30, 50], "confidence": 0.9}]

    monkeypatch.setitem(sys.modules, "inference.model", SimpleNamespace(detect_people=detect))
    result = analyze_photo(photo_bytes(format=format))
    assert result["headcount"] == 1
    assert result["metrics"] is None
    assert Image.open(BytesIO(result["image"])).size == (80, 60)


def test_wrong_layout_dimensions_rejected_before_detection(monkeypatch):
    def detect(frame):
        raise AssertionError("Detector must not run for incompatible geometry")

    monkeypatch.setitem(sys.modules, "inference.model", SimpleNamespace(detect_people=detect))
    with pytest.raises(ValueError):
        analyze_photo(photo_bytes(), "mess_main")


def test_no_people_is_a_successful_analysis(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "inference.model", SimpleNamespace(detect_people=lambda frame: [])
    )
    result = analyze_photo(photo_bytes())
    assert result["headcount"] == 0
    assert result["detections"] == []


def test_photo_posture_counts_share_live_classifier(monkeypatch):
    from inference.posture import PosturePrediction

    monkeypatch.setenv("ATTRIBUTE_CLASSIFIERS_ENABLED", "true")
    monkeypatch.setitem(
        sys.modules,
        "inference.model",
        SimpleNamespace(detect_people=lambda _: [{"bbox": [5, 5, 30, 50], "confidence": 0.9}]),
    )
    monkeypatch.setattr(
        "inference.scene.classify_postures",
        lambda *_: [PosturePrediction("Sitting", 0.9)],
    )
    result = analyze_photo(photo_bytes())
    assert result["headcount"] == result["sitting"] == 1
    assert result["standing"] == 0
    assert result["metrics"] is None


def test_photo_posture_failure_keeps_detection_count(monkeypatch):
    monkeypatch.setenv("ATTRIBUTE_CLASSIFIERS_ENABLED", "true")
    monkeypatch.setitem(
        sys.modules,
        "inference.model",
        SimpleNamespace(detect_people=lambda _: [{"bbox": [5, 5, 30, 50], "confidence": 0.9}]),
    )

    def fail(*_):
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr("inference.scene.classify_postures", fail)
    result = analyze_photo(photo_bytes())
    assert result["headcount"] == 1
    assert result["sitting"] is None and result["standing"] is None
    assert result["posture_error"]


def test_rotation_is_applied_before_detector_and_geometry(monkeypatch):
    shapes = []

    def detect(frame):
        shapes.append(frame.shape)
        return []

    monkeypatch.setitem(sys.modules, "inference.model", SimpleNamespace(detect_people=detect))
    result = analyze_photo(photo_bytes(size=(80, 60)), rotation_degrees_ccw=90)
    assert shapes == [(80, 60, 3)]
    assert (result["width"], result["height"]) == (60, 80)
    assert Image.open(BytesIO(result["image"])).size == (60, 80)
    with pytest.raises(ValueError, match="rotation"):
        decode_photo(photo_bytes(), rotation_degrees_ccw=45)
