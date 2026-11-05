"""Research queue estimator composed from calibration, features, and correction."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from research.calibration import CameraCalibration, load_calibration
from research.config import REPO_ROOT, load_research_config
from research.detection import CallableDetectorAdapter
from research.features import extract_features
from research.membership import GeometricRules, predict_membership
from research.occlusion import correct_queue_count


def _load_spacing_artifact(
    path: str | Path | None, calibration: CameraCalibration
) -> tuple[float | None, str | None]:
    if path is None:
        return None, None
    source = Path(path)
    if not source.is_file():
        return None, None
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("calibration_version") != calibration.calibration_version:
        raise ValueError("spacing artifact calibration version does not match camera calibration")
    if raw.get("coordinate_units") != calibration.coordinate_units:
        raise ValueError("spacing artifact coordinate units do not match camera calibration")
    return float(raw["expected_spacing"]), str(raw.get("model_version", source.stem))


def _load_uncertainty_artifact(
    path: str | Path | None, calibration: CameraCalibration
) -> tuple[float | None, str | None]:
    if path is None:
        return None, None
    source = Path(path)
    if not source.is_file():
        return None, None
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("calibration_version") != calibration.calibration_version:
        raise ValueError(
            "uncertainty artifact calibration version does not match camera calibration"
        )
    if raw.get("split") != "val":
        raise ValueError("uncertainty artifact must be calibrated on the validation split")
    return float(raw["delta"]), str(raw.get("model_version", source.stem))


def estimate_queue(
    image: np.ndarray,
    raw_detections: list[dict],
    camera_id: str,
    frame_size: tuple[int, int],
    strategy: str,
) -> dict:
    runtime = load_research_config()
    calibration = load_calibration(camera_id)
    detections = CallableDetectorAdapter(lambda _: raw_detections).detect(image)
    reference = os.environ.get("RESEARCH_REFERENCE_POINT", "bottom_center")
    if reference not in {"bottom_center", "centroid"}:
        raise ValueError("RESEARCH_REFERENCE_POINT must be bottom_center or centroid")
    features = extract_features(detections, calibration, frame_size, reference=reference)
    rules = GeometricRules(
        max_distance_to_path=calibration.distance_to_path_threshold,
        max_distance_to_counter=calibration.max_counter_distance,
        min_distance_to_counter=calibration.min_counter_distance,
    )
    if strategy == "geometric":
        membership_method = "geometric"
        correction_method = "none"
    elif strategy == "membership":
        membership_method = os.environ.get("RESEARCH_MEMBERSHIP_METHOD", runtime.membership_method)
        correction_method = "none"
    elif strategy == "occlusion":
        membership_method = os.environ.get("RESEARCH_MEMBERSHIP_METHOD", runtime.membership_method)
        correction_method = os.environ.get("RESEARCH_OCCLUSION_METHOD", runtime.occlusion_method)
    else:
        raise ValueError(f"unsupported research queue estimator: {strategy}")
    configured_model = runtime.membership_models.get(membership_method)
    model_path = os.environ.get("RESEARCH_MEMBERSHIP_MODEL") or configured_model
    predictions = predict_membership(
        features,
        rules=rules,
        method=membership_method,
        model_path=model_path,
        threshold=runtime.membership_threshold,
        allow_geometric_fallback=os.environ.get("RESEARCH_ALLOW_GEOMETRIC_FALLBACK", "true").lower()
        in {"1", "true", "yes"},
    )
    expected_spacing_raw = os.environ.get("RESEARCH_EXPECTED_SPACING")
    delta_raw = os.environ.get("RESEARCH_UNCERTAINTY_DELTA")
    residual_override = os.environ.get("RESEARCH_RESIDUAL_MODEL")
    residual_model = (
        REPO_ROOT / residual_override
        if residual_override and not os.path.isabs(residual_override)
        else residual_override or runtime.residual_model
    )
    spacing_override = os.environ.get("RESEARCH_SPACING_MODEL")
    spacing_path = (
        REPO_ROOT / spacing_override
        if spacing_override and not os.path.isabs(spacing_override)
        else spacing_override or runtime.spacing_model
    )
    artifact_spacing, spacing_version = (
        _load_spacing_artifact(spacing_path, calibration)
        if not expected_spacing_raw and runtime.expected_spacing is None
        else (None, None)
    )
    uncertainty_override = os.environ.get("RESEARCH_UNCERTAINTY_MODEL")
    uncertainty_path = (
        REPO_ROOT / uncertainty_override
        if uncertainty_override and not os.path.isabs(uncertainty_override)
        else uncertainty_override or runtime.uncertainty_model
    )
    artifact_delta, uncertainty_version = (
        _load_uncertainty_artifact(uncertainty_path, calibration)
        if strategy == "occlusion" and not delta_raw and runtime.uncertainty_delta is None
        else (None, None)
    )
    estimate = correct_queue_count(
        features,
        predictions,
        method=correction_method,
        expected_spacing=(
            float(expected_spacing_raw)
            if expected_spacing_raw
            else runtime.expected_spacing or artifact_spacing
        ),
        residual_model_path=residual_model,
        max_correction=int(
            os.environ.get("RESEARCH_MAX_HIDDEN_CORRECTION", runtime.max_hidden_correction)
        ),
        uncertainty_delta=(
            float(delta_raw) if delta_raw else runtime.uncertainty_delta or artifact_delta
        ),
        spacing_min_mean_density=runtime.spacing_min_mean_density,
        spacing_gap_multiplier=runtime.spacing_gap_multiplier,
        spacing_max_gap_multiplier=runtime.spacing_max_gap_multiplier,
        correction_model_version=spacing_version,
    )
    return {
        "queue_count": estimate.queue_count,
        "queue_visible": estimate.visible_count,
        "queue_hidden_estimate": estimate.hidden_estimate,
        "queue_ci_low": estimate.lower_bound,
        "queue_ci_high": estimate.upper_bound,
        "queue_method": f"{predictions[0].method if predictions else 'P1_EMPTY'}+{estimate.method}",
        "queue_model_version": "+".join(
            filter(
                None,
                (
                    predictions[0].model_version if predictions else "none",
                    estimate.model_version,
                    uncertainty_version,
                ),
            )
        ),
        "research_diagnostics": {
            "interval_calibrated": estimate.interval_calibrated,
            "calibration_version": calibration.calibration_version,
            "coordinate_units": calibration.coordinate_units,
            "people": [
                {"features": feature.to_dict(), "membership": prediction.to_dict()}
                for feature, prediction in zip(features, predictions, strict=True)
            ],
        },
    }
