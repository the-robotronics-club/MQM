from __future__ import annotations

import pytest

import inference.model as model


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confidence_threshold": -0.1}, "confidence_threshold"),
        ({"confidence_threshold": 1.1}, "confidence_threshold"),
        ({"iou_threshold": -0.1}, "iou_threshold"),
        ({"iou_threshold": 1.1}, "iou_threshold"),
        ({"image_size": 0}, "image_size"),
    ],
)
def test_detector_rejects_invalid_tuning_before_model_load(kwargs, message):
    with pytest.raises(ValueError, match=message):
        model.PersonDetector(**kwargs)


@pytest.mark.parametrize("minimum, expected", [(2, [0.3, 0.2]), (3, [0.3])])
def test_crowd_guard_preserves_sparse_precision(monkeypatch, minimum, expected):
    from types import SimpleNamespace

    import torch

    calls = []
    boxes = [
        SimpleNamespace(
            cls=torch.tensor(0),
            conf=torch.tensor(value),
            xyxy=torch.tensor([[1.0, 2.0, 10.0, 20.0]]),
        )
        for value in [0.3, 0.2, 0.1]
    ]

    class FakeYOLO:
        def __init__(self, _):
            pass

        def predict(self, **kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(boxes=boxes)]

    monkeypatch.setattr(model, "YOLO", FakeYOLO)
    detector = model.PersonDetector(adaptive_confidence=True, crowd_min_people=minimum)
    results = detector.detect_people("unused.jpg")
    assert [d["confidence"] for d in results] == pytest.approx(expected)
    assert calls[-1]["conf"] == 0.15
    assert results[0]["bbox"] == [1.0, 2.0, 10.0, 20.0]
    # An explicit override disables adaptation for a reproducible experiment.
    results = detector.detect_people("unused.jpg", confidence_threshold=0.25)
    assert len(results) == 1
    assert calls[-1]["conf"] == 0.25


@pytest.mark.parametrize("kwargs", [{"crowd_min_people": 0}, {"crowd_confidence_threshold": -1}])
def test_invalid_crowd_settings_fail_before_model_load(kwargs):
    with pytest.raises(ValueError):
        model.PersonDetector(**kwargs)
