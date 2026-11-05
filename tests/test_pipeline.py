from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from inference.classifiers import PersonAttributes
from inference.pipeline import InvalidImageError, decode_jpeg, process_frame
from inference.zones import ZoneConfigError
from tests import fixtures as fx

ROOT = Path(__file__).resolve().parent.parent
DENSE = ROOT / "data" / "samples" / "mess_hall_dense.jpg"
ALT = ROOT / "data" / "samples" / "mess_hall_alt.jpg"


def test_real_jpeg_to_metrics_pipeline_with_injected_detector():
    seen = {}

    def detector(image):
        seen["shape"] = image.shape
        return fx.BUSY

    metrics = process_frame(DENSE.read_bytes(), "mess_main", detector=detector)
    assert seen["shape"] == (490, 736, 3)
    assert metrics["camera_id"] == "mess_main"
    assert metrics["headcount"] == 12
    assert metrics["queue_count"] == 3
    assert metrics["seats_occupied"] == 8


def test_empty_detection_result_is_a_real_empty_reading():
    metrics = process_frame(DENSE.read_bytes(), "mess_main", detector=lambda _: [])
    assert metrics["headcount"] == 0
    assert metrics["detections_raw"] == 0


def test_attribute_classifiers_refine_only_the_relevant_spatial_zones():
    def classifier(_image, assignments):
        assert len(assignments) == len(fx.BUSY)
        return [
            PersonAttributes(
                is_queue=False,
                is_seated=False,
                queue_confidence=0.99,
                seated_confidence=0.99,
            )
            for _ in assignments
        ]

    metrics = process_frame(
        DENSE.read_bytes(),
        "mess_main",
        detector=lambda _: fx.BUSY,
        attribute_classifier=classifier,
    )
    assert metrics["headcount"] == 12
    assert metrics["queue_count"] == 0
    assert metrics["seats_occupied"] == 0


def test_corrupt_jpeg_is_rejected():
    with pytest.raises(InvalidImageError, match="JPEG"):
        decode_jpeg(b"not an image")


def test_non_jpeg_bytes_are_rejected_even_if_decodable():
    output = BytesIO()
    Image.new("RGB", (10, 10)).save(output, format="PNG")
    with pytest.raises(InvalidImageError, match="expected JPEG"):
        decode_jpeg(output.getvalue())


def test_jpeg_is_converted_to_opencv_bgr_order():
    output = BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(output, format="JPEG", quality=100)
    image, size = decode_jpeg(output.getvalue())
    assert size == (8, 8)
    assert image[0, 0, 0] < 5  # blue channel
    assert image[0, 0, 2] > 250  # red channel


def test_frame_coordinate_mismatch_is_rejected():
    with pytest.raises(ZoneConfigError, match="different coordinate spaces"):
        process_frame(ALT.read_bytes(), "mess_main", detector=lambda _: [])


def test_distance_snapshot_accepts_any_resolution_and_uses_shared_scene(monkeypatch):
    from inference.posture import PosturePrediction
    from inference.scene import summarize_scene

    monkeypatch.setenv("FRAME_METRICS_MODE", "distance")
    monkeypatch.setenv("QUEUE_ESTIMATOR", "production")
    output = BytesIO()
    Image.new("RGB", (1280, 720)).save(output, format="JPEG")
    detections = [{"bbox": [x, 100, x + 40, 300], "confidence": 0.8} for x in (100, 160, 220, 900)]
    postures = [PosturePrediction("Standing", 0.9)] * 3 + [PosturePrediction("Sitting", 0.95)]
    expected = summarize_scene(detections, postures)
    actual = process_frame(
        output.getvalue(),
        "mess_main",
        detector=lambda _: detections,
        posture_classifier=lambda *_: postures,
    )
    for key in ("headcount", "queue_count", "sitting", "standing_other_count"):
        assert actual[key] == expected[key]
    assert actual["headcount"] == 4
    assert actual["seats_total"] is None
    assert actual["seat_occupancy_pct"] is None
    assert actual["zone_counts"] == {}


def test_invalid_snapshot_mode_is_rejected_before_detection(monkeypatch):
    monkeypatch.setenv("FRAME_METRICS_MODE", "typo")
    with pytest.raises(ValueError, match="FRAME_METRICS_MODE"):
        process_frame(DENSE.read_bytes(), "mess_main", detector=lambda _: [])
