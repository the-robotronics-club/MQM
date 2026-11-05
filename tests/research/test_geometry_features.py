from __future__ import annotations

import math

import pytest

from research.calibration.camera_calibration import CameraCalibration
from research.calibration.homography import (
    CalibrationError,
    compute_homography,
    validate_calibration,
)
from research.detection import Detection
from research.features import extract_features
from research.geometry import QueuePath, bottom_center, centroid, local_density, rank_along_path


def calibration() -> CameraCalibration:
    homography = compute_homography(
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [(0, 0), (10, 0), (10, 10), (0, 10)],
    )
    return CameraCalibration(
        "camera",
        (100, 100),
        homography,
        (0, 0),
        ((0, 0), (5, 0), (5, 10)),
        2.0,
        1.0,
        20.0,
        "synthetic-v1",
        "synthetic_units",
    )


def test_reference_points() -> None:
    assert bottom_center([10, 20, 30, 60]) == (20, 60)
    assert centroid([10, 20, 30, 60]) == (20, 40)


def test_identity_and_known_perspective_homographies() -> None:
    identity = compute_homography(
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        [(0, 0), (1, 0), (1, 1), (0, 1)],
    )
    assert identity.project_point((0.25, 0.75)) == pytest.approx((0.25, 0.75))
    perspective = compute_homography(
        [(0, 0), (2, 0), (2, 2), (0, 2)],
        [(0, 0), (4, 0), (4, 8), (0, 8)],
    )
    assert perspective.project_point((1, 1)) == pytest.approx((2, 4))
    projective = compute_homography(
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [(0, 0), (100, 0), (50, 50), (0, 50)],
    )
    assert projective.project_point((50, 50)) == pytest.approx((100 / 3, 100 / 3))


def test_invalid_calibration_and_frame_size() -> None:
    with pytest.raises(CalibrationError):
        compute_homography([(0, 0)] * 4, [(0, 0)] * 4)
    config = {
        "calibration_status": "validated",
        "frame_size": {"width": 100, "height": 100},
        "homography": {
            "image_points": [(0, 0), (100, 0), (100, 100), (0, 100)],
            "ground_points": [(0, 0), (10, 0), (10, 10), (0, 10)],
        },
        "counter": {"position": [0, 0]},
        "queue_path": [[0, 0], [10, 0]],
    }
    with pytest.raises(CalibrationError, match="does not match"):
        validate_calibration(config, (200, 100))


def test_curved_queue_projection_distance_and_rank() -> None:
    path = QueuePath([(0, 0), (5, 0), (5, 5)])
    projection = path.project((6, 3))
    assert projection.point == pytest.approx((5, 3))
    assert projection.distance == pytest.approx(1)
    assert projection.arc_length == pytest.approx(8)
    points = [(5, 4), (2, 0), (5, 1)]
    ranks, normalized = rank_along_path(points, path)
    assert ranks == [2, 0, 1]
    assert normalized == [1.0, 0.0, 0.5]


def test_ground_density_and_primary_features() -> None:
    assert local_density([(0, 0), (1, 0), (4, 0)], 1.1) == [1, 1, 0]
    detections = [
        Detection((0, 0, 20, 20), 0.9),
        Detection((10, 0, 30, 20), 0.8),
    ]
    features = extract_features(detections, calibration(), (100, 100))
    assert len(features) == 2
    assert features[0].image_point == (10, 20)
    assert features[0].ground_point == pytest.approx((1, 2))
    assert features[0].distance_to_counter == pytest.approx(math.sqrt(5))
    assert features[0].path_arc_length == pytest.approx(1)
    assert features[0].local_density == 1
    assert 0 <= features[0].occlusion_score <= 1
    assert len(features[0].vector()) == 8
