"""One membership inference interface with per-person diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib

from research.features.schemas import PersonFeatures

from .rules import GeometricRules


@dataclass(frozen=True)
class MembershipPrediction:
    detection_id: str
    queue_probability: float
    is_queue: bool
    method: str
    feature_version: str
    model_version: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predict_membership(
    features: list[PersonFeatures],
    *,
    rules: GeometricRules,
    method: str = "geometric",
    model_path: str | Path | None = None,
    threshold: float = 0.5,
    allow_geometric_fallback: bool = True,
) -> list[MembershipPrediction]:
    if method == "geometric":
        return [
            MembershipPrediction(
                feature.detection_id,
                (result := rules.predict(feature)).score,
                result.is_queue,
                "B4_GEOMETRIC",
                feature.feature_version,
                "rules-v1",
                result.reasons,
            )
            for feature in features
        ]
    if method not in {"logistic", "gbdt"}:
        raise ValueError(f"unknown membership method: {method}")
    if not features:
        return []
    if model_path is None or not Path(model_path).is_file():
        if allow_geometric_fallback:
            fallback = predict_membership(features, rules=rules)
            return [
                MembershipPrediction(
                    item.detection_id,
                    item.queue_probability,
                    item.is_queue,
                    f"{item.method}_FALLBACK_NO_MODEL",
                    item.feature_version,
                    item.model_version,
                    item.reasons,
                )
                for item in fallback
            ]
        raise FileNotFoundError(f"membership model not found: {model_path}")
    artifact = joblib.load(model_path)
    names = tuple(artifact["feature_names"])
    if features and artifact.get("feature_version") != features[0].feature_version:
        raise ValueError("membership artifact feature version does not match extracted features")
    probabilities = artifact["model"].predict_proba([item.vector(names) for item in features])[:, 1]
    version = str(artifact.get("model_version", Path(model_path).stem))
    return [
        MembershipPrediction(
            feature.detection_id,
            float(probability),
            bool(probability >= threshold),
            f"P1_{method.upper()}",
            feature.feature_version,
            version,
        )
        for feature, probability in zip(features, probabilities, strict=True)
    ]
