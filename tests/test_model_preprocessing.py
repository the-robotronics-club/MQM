"""Check production preprocessing against the training transform."""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from inference.classifiers import _crop_tensor
from tools.evaluation.real_images import interval_error


def test_classifier_matches_training_preprocessing():
    image = np.random.default_rng(7).integers(0, 256, (87, 61, 3), dtype=np.uint8)
    training = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    expected = training(Image.fromarray(image[:, :, ::-1]))
    actual = _crop_tensor(image, [0, 0, 61, 87])
    torch.testing.assert_close(actual, expected)


def test_count_interval_error_respects_annotation_uncertainty():
    assert interval_error(5, {"min": 4, "max": 6}) == 0
    assert interval_error(2, {"min": 4, "max": 6}) == 2
    assert interval_error(9, {"min": 4, "max": 6}) == 3
