from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from PIL import Image

from research.datasets.build_features import (
    build_features,
    intersection_over_union,
    match_annotations,
)
from research.datasets.loader import load_manifest
from research.datasets.splits import grouped_split, write_splits
from research.datasets.validation import validate_manifest
from research.detection import Detection
from research.evaluation.ablations import parse_ablation
from research.evaluation.bootstrap import session_bootstrap
from research.evaluation.evaluate import evaluate
from research.evaluation.metrics import membership_metrics, queue_metrics
from research.evaluation.run_baseline import run


def make_manifest(tmp_path, sessions=("s1", "s2", "s3")):
    images = []
    for index, session in enumerate(sessions):
        filename = f"image-{index}.jpg"
        Image.new("RGB", (100, 80), (index * 19 % 256, 40, 10)).save(tmp_path / filename)
        images.append(
            {
                "image_id": f"image-{index}",
                "path": filename,
                "camera_id": "camera",
                "session_id": session,
                "clip_id": None,
                "timestamp": "2026-01-01T12:00:00Z",
                "meal_period": "lunch",
                "crowd_band": "low",
                "queue_ground_truth": 1,
                "annotations": [
                    {
                        "bbox": [10, 10, 30, 60],
                        "role": "queue_member",
                        "occlusion": "none",
                        "truncation": False,
                    }
                ],
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": "v1", "images": images}))
    return path


def test_dataset_validation_and_grouped_split_no_leakage(tmp_path) -> None:
    manifest_path = make_manifest(tmp_path, tuple(f"s{i}" for i in range(10)))
    manifest = load_manifest(manifest_path)
    assert validate_manifest(manifest, tmp_path).valid
    groups = grouped_split([item.session_id for item in manifest.images], seed=7)
    assert set(groups["train"]).isdisjoint(groups["val"])
    assert set(groups["train"]).isdisjoint(groups["test"])
    output = tmp_path / "splits"
    write_splits(manifest_path, output, seed=7)
    report = validate_manifest(manifest, tmp_path, list(output.glob("*.json")))
    assert report.valid


def test_dataset_validation_detects_bad_box_and_session_leakage(tmp_path) -> None:
    manifest_path = make_manifest(tmp_path)
    raw = json.loads(manifest_path.read_text())
    raw["images"][0]["annotations"][0]["bbox"] = [-1, 0, 20, 20]
    manifest_path.write_text(json.dumps(raw))
    manifest = load_manifest(manifest_path)
    assert not validate_manifest(manifest, tmp_path).valid
    split_a = tmp_path / "a.json"
    split_b = tmp_path / "b.json"
    base = {"dataset_version": "v1", "session_ids": ["s1"]}
    split_a.write_text(json.dumps({**base, "name": "train", "image_ids": ["image-0"]}))
    split_b.write_text(json.dumps({**base, "name": "test", "image_ids": ["image-0"]}))
    report = validate_manifest(manifest, tmp_path, [split_a, split_b])
    assert any("appears in" in error for error in report.errors)


def test_identical_images_cannot_cross_splits_under_different_ids_and_sessions(tmp_path) -> None:
    manifest_path = make_manifest(tmp_path)
    (tmp_path / "image-1.jpg").write_bytes((tmp_path / "image-0.jpg").read_bytes())
    splits = []
    for index, name in enumerate(("train", "test")):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps({"name": name, "dataset_version": "v1", "image_ids": [f"image-{index}"]})
        )
        splits.append(path)
    report = validate_manifest(load_manifest(manifest_path), tmp_path, splits)
    assert not report.valid
    assert any("duplicate image content crosses splits" in e for e in report.errors)


def test_duplicate_within_one_split_is_a_warning(tmp_path) -> None:
    manifest_path = make_manifest(tmp_path)
    (tmp_path / "image-1.jpg").write_bytes((tmp_path / "image-0.jpg").read_bytes())
    split = tmp_path / "train.json"
    split.write_text(
        json.dumps({"name": "train", "dataset_version": "v1", "image_ids": ["image-0", "image-1"]})
    )
    report = validate_manifest(load_manifest(manifest_path), tmp_path, [split])
    assert report.valid
    assert any("duplicate image content" in w for w in report.warnings)


def test_same_camera_clip_cannot_leak_by_changing_session_names(tmp_path) -> None:
    manifest_path = make_manifest(tmp_path)
    raw = json.loads(manifest_path.read_text())
    for row in raw["images"]:
        row["clip_id"] = "lunch-video"
    manifest_path.write_text(json.dumps(raw))
    splits = []
    for index, name in enumerate(("train", "test")):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps({"name": name, "dataset_version": "v1", "image_ids": [f"image-{index}"]})
        )
        splits.append(path)
    report = validate_manifest(load_manifest(manifest_path), tmp_path, splits)
    assert any("camera clip" in e for e in report.errors)
    # Independent cameras can reuse a clip filename without sharing frames.
    raw["images"][1]["camera_id"] = "second-camera"
    manifest_path.write_text(json.dumps(raw))
    assert validate_manifest(load_manifest(manifest_path), tmp_path, splits).valid


def test_queue_membership_metrics_and_session_bootstrap() -> None:
    assert queue_metrics([0, 2, 4], [0, 3, 2]) == {
        "mae": 1.0,
        "rmse": (5 / 3) ** 0.5,
        "exact_accuracy": 1 / 3,
        "within_1_accuracy": 2 / 3,
        "within_2_accuracy": 1.0,
        "within_3_accuracy": 1.0,
    }
    assert membership_metrics([True, True, False], [True, False, True]) == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    rows = [{"session_id": "a", "error": 1}, {"session_id": "b", "error": 3}]
    first = session_bootstrap(
        rows, lambda values: sum(row["error"] for row in values), iterations=20, seed=2
    )
    second = session_bootstrap(
        rows, lambda values: sum(row["error"] for row in values), iterations=20, seed=2
    )
    assert first == second


def test_ground_truth_box_baseline_is_independently_executable(tmp_path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (736, 490)).save(image_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": "v1",
                "images": [
                    {
                        "image_id": "frame",
                        "path": image_path.name,
                        "camera_id": "mess_main",
                        "session_id": "session",
                        "clip_id": None,
                        "timestamp": "2026-01-01T12:00:00Z",
                        "meal_period": "lunch",
                        "crowd_band": "low",
                        "queue_ground_truth": 1,
                        "annotations": [
                            {
                                "bbox": [320, 145, 360, 185],
                                "role": "queue_member",
                                "occlusion": "none",
                                "truncation": False,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    split_path = tmp_path / "test.json"
    split_path.write_text(
        json.dumps({"dataset_version": "v1", "image_ids": ["frame"]}), encoding="utf-8"
    )
    output = tmp_path / "predictions.csv"
    run(manifest_path, split_path, tmp_path, "b0", output)
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["queue_prediction"] == "1"
    assert rows[0]["method"] == "b0"


def test_evaluator_reports_breakdowns_membership_and_latency() -> None:
    rows = [
        {
            "image_id": "a",
            "session_id": "s1",
            "camera_id": "cam",
            "timestamp": "2026-01-01T12:00:00Z",
            "crowd_band": "low",
            "occlusion_band": "none",
            "meal_period": "lunch",
            "perspective_band": "near",
            "queue_ground_truth": "2",
            "queue_prediction": "1",
            "membership_truth": "[true, false]",
            "membership_prediction": "[true, true]",
            "latency_ms": "10",
        },
        {
            "image_id": "b",
            "session_id": "s2",
            "camera_id": "cam",
            "timestamp": "2026-01-02T12:00:00Z",
            "crowd_band": "high",
            "occlusion_band": "heavy",
            "meal_period": "dinner",
            "perspective_band": "far",
            "queue_ground_truth": "3",
            "queue_prediction": "3",
            "membership_truth": "[true]",
            "membership_prediction": "[true]",
            "latency_ms": "20",
        },
    ]
    metrics, per_session = evaluate(rows, bootstrap_iterations=20, seed=2)
    assert metrics["membership"]["recall"] == 1.0
    assert metrics["latency_ms"]["mean"] == 15.0
    assert set(metrics["breakdowns"]["crowd_band"]) == {"low", "high"}
    assert set(metrics["breakdowns"]["day"]) == {"2026-01-01", "2026-01-02"}
    assert len(per_session) == 2


def test_ablation_configuration_is_validated() -> None:
    assert parse_ablation(
        ["reference_point=bottom_center", "local_density=false", "inference_resolution=640"]
    ) == {
        "reference_point": "bottom_center",
        "local_density": False,
        "inference_resolution": 640,
    }


def test_detector_boxes_match_annotations_without_reusing_labels() -> None:
    detections = [
        Detection((0, 0, 10, 10), 0.9),
        Detection((20, 20, 30, 30), 0.8),
    ]
    annotations = (
        SimpleNamespace(bbox=(0, 0, 10, 10)),
        SimpleNamespace(bbox=(20, 20, 30, 30)),
    )
    assert intersection_over_union(detections[0].bbox, annotations[0].bbox) == 1
    assert match_annotations(detections, annotations, 0.5) == {0: 0, 1: 1}


def test_feature_manifest_is_built_from_frozen_split_and_annotations(tmp_path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 100)).save(image_path)
    manifest_path = make_manifest(tmp_path, ("s1",))
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["images"][0].update(
        {
            "camera_id": "camera",
            "path": image_path.name,
            "annotations": [
                {
                    "bbox": [10, 10, 30, 60],
                    "role": "queue_member",
                    "occlusion": "none",
                    "truncation": False,
                }
            ],
        }
    )
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    split_path = tmp_path / "train.json"
    split_path.write_text(
        json.dumps(
            {
                "name": "train",
                "dataset_version": "v1",
                "session_ids": ["s1"],
                "image_ids": ["image-0"],
            }
        ),
        encoding="utf-8",
    )
    calibration_dir = tmp_path / "calibrations"
    calibration_dir.mkdir()
    (calibration_dir / "camera.yaml").write_text(
        json.dumps(
            {
                "camera_id": "camera",
                "calibration_status": "validated",
                "coordinate_units": "synthetic",
                "frame_size": {"width": 100, "height": 100},
                "homography": {
                    "image_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                    "ground_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
                },
                "counter": {"position": [0, 0]},
                "queue_path": [[0, 0], [100, 100]],
                "density_radius": 10,
                "geometric": {"max_path_distance": 20, "max_counter_distance": 200},
                "calibration_version": "c1",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "features.json"
    payload = build_features(
        manifest_path,
        split_path,
        tmp_path,
        output,
        lambda _: [{"bbox": [10, 10, 30, 60], "confidence": 0.8}],
        calibration_dir=calibration_dir,
    )
    assert payload["matching_summary"]["matched_detections"] == 1
    assert payload["people"][0]["role"] == "queue_member"
    assert payload["people"][0]["detector_confidence"] == 0.8
