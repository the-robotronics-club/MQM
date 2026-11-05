import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from inference.onnx_runtime import crop_tensor, decode_predictions, letterbox, suppress  # noqa: E402


def test_letterbox_preserves_rectangular_stride_and_rgb():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :, 2] = 255
    tensor, gain, left, top = letterbox(frame)
    assert tensor.shape == (1, 3, 736, 1280)
    assert (gain, left, top) == (1, 0, 8)
    assert tensor[0, 0, 20, 20] == 1
    assert tensor[0, 2, 20, 20] == 0


def test_numpy_nms_keeps_neighbor_but_removes_duplicate():
    boxes = np.array([[0, 0, 10, 30], [0, 0, 10, 30], [9, 0, 19, 30]], dtype=np.float32)
    assert suppress(boxes, np.array([0.9, 0.8, 0.7])) == [0, 2]


def test_yolo_decode_uses_top_class_and_restores_coordinates():
    raw = np.zeros((1, 84, 3), dtype=np.float32)
    raw[0, :4, :] = np.array([[30, 70, 100], [50, 50, 50], [20, 20, 20], [40, 40, 40]])
    raw[0, 4, :] = [0.9, 0.8, 0.2]
    raw[0, 5, 1] = 0.95  # A higher-scoring non-person class must not become a person.
    rows = decode_predictions(raw, (100, 200), gain=2, left=0, top=10)
    assert len(rows) == 1  # Sparse confidence guard also excludes 0.2.
    assert rows[0]["bbox"] == [10, 10, 20, 30]


def test_crop_matches_original_training_preprocessing():
    from inference.classifiers import _crop_tensor

    frame = np.random.default_rng(8).integers(0, 256, (180, 220, 3), dtype=np.uint8)
    bbox = [-5, 10.3, 180.8, 200]
    np.testing.assert_allclose(
        crop_tensor(frame, bbox), _crop_tensor(frame, bbox).numpy(), atol=1e-6
    )
