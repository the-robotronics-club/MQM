from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression

from research import FEATURE_VERSION
from research.features.schemas import PRIMARY_FEATURE_NAMES, PersonFeatures
from research.membership import GeometricRules, predict_membership
from research.occlusion import correct_queue_count
from research.occlusion.features import occlusion_scores
from research.occlusion.residual import predict_hidden
from research.occlusion.spacing import estimate_missing_from_spacing, fit_expected_spacing
from research.occlusion.uncertainty import interval, residual_quantile
from research.training.calibrate_uncertainty import calibrate
from research.training.fit_spacing import fit


def feature(identifier: str, distance: float, progress: float, density: int = 1) -> PersonFeatures:
    return PersonFeatures(
        identifier,
        (0, 0, 10, 20),
        (5, 20),
        (distance, progress),
        distance,
        distance,
        density,
        int(progress * 10),
        progress,
        0.9,
        0.2,
        progress,
        10,
        20,
        1,
        "middle",
        path_arc_length=progress,
    )


def test_occlusion_feature_is_bounded_and_responds_to_overlap() -> None:
    scores = occlusion_scores([(10, 10, 50, 80), (30, 10, 60, 80), (70, 10, 90, 80)], (100, 100))
    assert scores[0] > scores[2]
    assert all(0 <= score <= 1 for score in scores)


def test_geometric_membership_is_interpretable() -> None:
    rules = GeometricRules(max_distance_to_path=2, max_distance_to_counter=10)
    yes = rules.predict(feature("a", 1, 0.2))
    no = rules.predict(feature("b", 5, 0.2))
    assert yes.is_queue and "near_queue_path" in yes.reasons
    assert not no.is_queue and "not_near_queue_path" in no.reasons


def test_membership_model_serialization(tmp_path) -> None:
    rows = np.asarray([[0] * 8, [1] * 8, [0.1] * 8, [0.9] * 8])
    model = LogisticRegression().fit(rows, [0, 1, 0, 1])
    path = tmp_path / "membership.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": PRIMARY_FEATURE_NAMES,
            "feature_version": FEATURE_VERSION,
            "model_version": "synthetic-v1",
        },
        path,
    )
    predictions = predict_membership(
        [feature("a", 1, 0.1)],
        rules=GeometricRules(2, 10),
        method="logistic",
        model_path=path,
    )
    assert predictions[0].method == "P1_LOGISTIC"
    assert predictions[0].model_version == "synthetic-v1"
    assert predictions[0].to_dict()["detection_id"] == "a"


def test_spacing_correction_and_hidden_clamp() -> None:
    assert fit_expected_spacing([0.9, 1.0, 1.1]) == 1.0
    correction = estimate_missing_from_spacing([0, 1, 4, 10], 1, max_correction=3, max_path_gap=5)
    assert correction.hidden_estimate == 2
    rules = GeometricRules(2, 10)
    features = [feature("a", 1, 0), feature("b", 1, 5)]
    predictions = predict_membership(features, rules=rules)
    estimate = correct_queue_count(
        features,
        predictions,
        method="spacing",
        expected_spacing=1,
        max_correction=2,
        uncertainty_delta=1,
    )
    assert estimate.hidden_estimate == 2
    assert estimate.queue_count == estimate.visible_count + 2
    assert estimate.lower_bound <= estimate.queue_count <= estimate.upper_bound


def test_residual_model_prediction_is_clamped(tmp_path) -> None:
    model = DummyRegressor(strategy="constant", constant=100).fit([[0.0]], [100.0])
    path = tmp_path / "residual.joblib"
    joblib.dump({"model": model, "feature_names": ["visible_count"]}, path)
    assert predict_hidden({"visible_count": 2.0}, path, max_correction=3) == 3


def test_uncertainty_bounds_are_calibrated_only_from_supplied_residuals() -> None:
    delta = residual_quantile([0, 1, -2, 1], 0.75)
    low, high = interval(1, delta)
    assert low == 0
    assert high >= 2
    with pytest.raises(ValueError):
        residual_quantile([], 0.95)


def test_missing_occlusion_artifact_is_reported_as_fallback() -> None:
    features = [feature("a", 1, 0)]
    predictions = predict_membership(features, rules=GeometricRules(2, 10))
    spacing = correct_queue_count(features, predictions, method="spacing")
    residual = correct_queue_count(features, predictions, method="residual")
    assert spacing.hidden_estimate == 0
    assert spacing.lower_bound is None
    assert spacing.upper_bound is None
    assert spacing.method == "P2_NONE_FALLBACK_NO_EXPECTED_SPACING"
    assert residual.method == "P2_NONE_FALLBACK_NO_RESIDUAL_MODEL"


def test_empty_learned_membership_frame_is_supported() -> None:
    assert (
        predict_membership(
            [], rules=GeometricRules(2, 10), method="logistic", model_path="missing.joblib"
        )
        == []
    )


def test_spacing_fit_uses_only_low_occlusion_training_frames(tmp_path) -> None:
    source = tmp_path / "spacing-input.json"
    source.write_text(
        json.dumps(
            {
                "dataset_version": "v1",
                "split": "train",
                "coordinate_units": "metres",
                "calibration_version": "c1",
                "frames": [
                    {
                        "image_id": "a",
                        "occlusion_band": "none",
                        "queue_path_positions": [0, 1, 2.2],
                    },
                    {
                        "image_id": "b",
                        "occlusion_band": "heavy",
                        "queue_path_positions": [0, 10],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    split = tmp_path / "train.json"
    split.write_text(
        json.dumps(
            {
                "name": "train",
                "dataset_version": "v1",
                "image_ids": ["a", "b"],
                "session_ids": [],
            }
        ),
        encoding="utf-8",
    )
    artifact = fit(source, tmp_path / "spacing.json", split)
    assert artifact["expected_spacing"] == pytest.approx(1.1)
    assert artifact["training_frames"] == 1


def test_uncertainty_calibration_requires_validation_predictions(tmp_path) -> None:
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "image_id,session_id,queue_ground_truth,queue_prediction,dataset_version,split\n"
        "a,s1,5,3,v1,val\n"
        "b,s2,2,3,v1,val\n",
        encoding="utf-8",
    )
    split = tmp_path / "val.json"
    split.write_text(
        json.dumps(
            {
                "name": "val",
                "dataset_version": "v1",
                "image_ids": ["a", "b"],
                "session_ids": ["s1", "s2"],
            }
        ),
        encoding="utf-8",
    )
    artifact = calibrate(predictions, tmp_path / "uncertainty.json", split, "camera-v1", 0.95)
    assert artifact["split"] == "val"
    assert artifact["delta"] >= 1
