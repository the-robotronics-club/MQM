"""Evaluate precomputed held-out predictions and write reproducible artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from research import FEATURE_VERSION

from .ablations import parse_ablation
from .artifacts import file_hash, write_experiment
from .baselines import require_supported
from .bootstrap import session_bootstrap
from .metrics import membership_metrics, queue_metrics

BREAKDOWN_FIELDS = (
    "crowd_band",
    "occlusion_band",
    "meal_period",
    "perspective_band",
    "camera_id",
    "session_id",
    "day",
)


def load_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "image_id",
        "session_id",
        "queue_ground_truth",
        "queue_prediction",
        "dataset_version",
        "split",
    }
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"predictions CSV requires: {', '.join(sorted(required))}")
    return rows


def _queue_metrics(rows: Iterable[dict]) -> dict[str, float]:
    values = list(rows)
    return queue_metrics(
        [int(row["queue_ground_truth"]) for row in values],
        [int(row["queue_prediction"]) for row in values],
    )


def _day(row: dict) -> str | None:
    if row.get("day"):
        return str(row["day"])
    if not row.get("timestamp"):
        return None
    try:
        return (
            datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00")).date().isoformat()
        )
    except ValueError:
        return None


def _membership(rows: list[dict]) -> dict[str, float] | None:
    truth: list[bool] = []
    prediction: list[bool] = []
    for row in rows:
        if not row.get("membership_truth") or not row.get("membership_prediction"):
            continue
        expected = json.loads(row["membership_truth"])
        actual = json.loads(row["membership_prediction"])
        if not isinstance(expected, list) or not isinstance(actual, list):
            raise ValueError("membership columns must contain JSON arrays")
        truth.extend(bool(item) for item in expected)
        prediction.extend(bool(item) for item in actual)
    return membership_metrics(truth, prediction) if truth else None


def evaluate(
    rows: list[dict], *, bootstrap_iterations: int = 1000, seed: int = 42
) -> tuple[dict, list[dict]]:
    truth = [int(row["queue_ground_truth"]) for row in rows]
    prediction = [int(row["queue_prediction"]) for row in rows]
    metrics: dict = queue_metrics(truth, prediction)
    metrics["mae_session_bootstrap"] = session_bootstrap(
        rows,
        lambda sample: _queue_metrics(sample)["mae"],
        iterations=bootstrap_iterations,
        seed=seed,
    )
    metrics["rmse_session_bootstrap"] = session_bootstrap(
        rows,
        lambda sample: _queue_metrics(sample)["rmse"],
        iterations=bootstrap_iterations,
        seed=seed,
    )
    membership = _membership(rows)
    if membership is not None:
        metrics["membership"] = membership
    for source, target in (("detection_ap50", "ap50"), ("detection_map50_95", "map50_95")):
        values = [float(row[source]) for row in rows if row.get(source) not in (None, "")]
        if values:
            metrics.setdefault("detection", {})[target] = statistics.fmean(values)
    latency_values = [
        float(row["latency_ms"]) for row in rows if row.get("latency_ms") not in (None, "")
    ]
    if latency_values:
        ordered = sorted(latency_values)
        metrics["latency_ms"] = {
            "mean": statistics.fmean(latency_values),
            "median": statistics.median(latency_values),
            "p95": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        }

    breakdowns: dict[str, dict[str, dict[str, float]]] = {}
    for field in BREAKDOWN_FIELDS:
        grouped_values: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            value = _day(row) if field == "day" else row.get(field)
            if field == "occlusion_band" and not value:
                value = row.get("occlusion")
            if value not in (None, ""):
                grouped_values[str(value)].append(row)
        if grouped_values:
            breakdowns[field] = {
                value: _queue_metrics(group) for value, group in sorted(grouped_values.items())
            }
    metrics["breakdowns"] = breakdowns
    per_session = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["session_id"])].append(row)
    for session, session_rows in sorted(grouped.items()):
        result = _queue_metrics(session_rows)
        per_session.append({"session_id": session, **result})
    return metrics, per_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--calibration-version")
    parser.add_argument("--feature-version", default=FEATURE_VERSION)
    parser.add_argument(
        "--ablation",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Record a controlled ablation dimension; may be repeated.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    require_supported(args.method)
    rows = load_predictions(args.predictions)
    ablations = parse_ablation(args.ablation)
    if {str(row["split"]) for row in rows} != {args.split}:
        raise ValueError("prediction rows do not match the requested held-out split")
    if {str(row["dataset_version"]) for row in rows} != {args.dataset_version}:
        raise ValueError("prediction rows do not match --dataset-version")
    if args.method in {"p1", "p2"} and any(
        "FALLBACK" in str(row.get("inference_method", "")) for row in rows
    ):
        raise ValueError("learned-method evaluation cannot include geometric/artifact fallbacks")
    metrics, per_session = evaluate(
        rows, bootstrap_iterations=args.bootstrap_iterations, seed=args.seed
    )
    output = write_experiment(
        args.output,
        args.method,
        {
            "method": args.method,
            "split": args.split,
            "predictions": str(args.predictions),
            "bootstrap_iterations": args.bootstrap_iterations,
            "random_seed": args.seed,
            "ablations": ablations,
        },
        metrics,
        rows,
        per_session,
        {
            "dataset_split_version": args.dataset_version,
            "model_path": str(args.model_path) if args.model_path else None,
            "model_hash": file_hash(args.model_path),
            "feature_version": args.feature_version,
            "camera_calibration_version": args.calibration_version,
            "random_seed": args.seed,
        },
    )
    print(output)


if __name__ == "__main__":
    main()
