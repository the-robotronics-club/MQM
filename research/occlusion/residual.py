"""Serialized count-residual model helpers."""

from __future__ import annotations

from pathlib import Path

import joblib


def predict_hidden(features: dict[str, float], model_path: str | Path, max_correction: int) -> int:
    if max_correction < 0:
        raise ValueError("max correction cannot be negative")
    artifact = joblib.load(model_path)
    names = artifact["feature_names"]
    value = float(artifact["model"].predict([[features[name] for name in names]])[0])
    return min(max_correction, max(0, round(value)))
