from types import SimpleNamespace

import numpy as np
import pytest
import torch

import inference.model as model
from inference.preprocessing import crop_black_padding


def test_padding_crop_preserves_content_and_original_frame():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    frame[20:100, 30:170] = 150
    original = frame.copy()
    crop, x, y = crop_black_padding(frame)
    assert (x, y) == (30, 20)
    assert crop.shape == (80, 140, 3)
    np.testing.assert_array_equal(crop, original[20:100, 30:170])
    crop[:] = 0
    np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize("kind", ["black", "tiny_light", "no_padding"])
def test_padding_crop_leaves_dark_or_unpadded_scenes_alone(kind):
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    if kind == "tiny_light":
        frame[50:52, 50:52] = 255
    if kind == "no_padding":
        frame[:] = 150
    crop, x, y = crop_black_padding(frame)
    assert crop is frame
    assert (x, y) == (0, 0)


@pytest.mark.parametrize("enabled", [True, False])
def test_detector_restores_coordinates_for_posture_and_camera_zones(monkeypatch, enabled):
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    frame[20:100, 30:170] = 150
    original = frame.copy()
    calls = []

    class FakeYOLO:
        def __init__(self, _):
            pass

        def predict(self, **kwargs):
            calls.append(kwargs)
            return [
                SimpleNamespace(
                    boxes=[
                        SimpleNamespace(
                            cls=torch.tensor(0),
                            conf=torch.tensor(0.8),
                            xyxy=torch.tensor([[5.0, 6.0, 15.0, 26.0]]),
                        )
                    ]
                )
            ]

    monkeypatch.setattr(model, "YOLO", FakeYOLO)
    detector = model.PersonDetector(trim_black_padding=enabled)
    result = detector.detect_people(frame)
    assert calls[0]["source"].shape == ((80, 140, 3) if enabled else frame.shape)
    assert result[0]["bbox"] == ([35.0, 26.0, 45.0, 46.0] if enabled else [5.0, 6.0, 15.0, 26.0])
    np.testing.assert_array_equal(frame, original)


def test_invalid_padding_setting_fails_before_loading_weights(monkeypatch):
    monkeypatch.setattr(model, "_default_detector", None)
    monkeypatch.setenv("MODEL_TRIM_BLACK_PADDING", "invalid")
    with pytest.raises(ValueError, match="MODEL_TRIM_BLACK_PADDING"):
        model.get_detector()
