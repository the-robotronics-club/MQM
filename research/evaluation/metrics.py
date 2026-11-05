"""Queue-count and membership metrics without MAPE."""

from __future__ import annotations

import math

import numpy as np


def queue_metrics(truth: list[int], prediction: list[int]) -> dict[str, float]:
    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must have the same non-zero length")
    errors = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    absolute = np.abs(errors)
    return {
        "mae": float(np.mean(absolute)),
        "rmse": float(math.sqrt(np.mean(errors**2))),
        "exact_accuracy": float(np.mean(absolute == 0)),
        "within_1_accuracy": float(np.mean(absolute <= 1)),
        "within_2_accuracy": float(np.mean(absolute <= 2)),
        "within_3_accuracy": float(np.mean(absolute <= 3)),
    }


def membership_metrics(truth: list[bool], prediction: list[bool]) -> dict[str, float]:
    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must have the same non-zero length")
    tp = sum(expected and actual for expected, actual in zip(truth, prediction, strict=True))
    fp = sum(not expected and actual for expected, actual in zip(truth, prediction, strict=True))
    fn = sum(expected and not actual for expected, actual in zip(truth, prediction, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
