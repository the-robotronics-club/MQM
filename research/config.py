"""Validated global configuration for research inference and experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from research import FEATURE_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "research.yaml"


class ResearchConfigurationError(ValueError):
    """Raised when research.yaml cannot define a safe, reproducible run."""


@dataclass(frozen=True)
class ResearchConfig:
    feature_version: str
    random_seed: int
    membership_method: str
    membership_threshold: float
    occlusion_method: str
    max_hidden_correction: int
    expected_spacing: float | None
    spacing_min_mean_density: float
    spacing_gap_multiplier: float
    spacing_max_gap_multiplier: float
    uncertainty_delta: float | None
    membership_models: dict[str, Path]
    spacing_model: Path | None
    uncertainty_model: Path | None
    residual_model: Path | None


def _optional_positive(raw: Any, name: str) -> float | None:
    if raw is None:
        return None
    value = float(raw)
    if value <= 0:
        raise ResearchConfigurationError(f"{name} must be positive when configured")
    return value


def _model_path(raw: Any) -> Path | None:
    if raw in (None, ""):
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else REPO_ROOT / path


def load_research_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ResearchConfig:
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ResearchConfigurationError(f"cannot read research configuration: {exc}") from exc
    if not isinstance(raw, dict):
        raise ResearchConfigurationError("research configuration must be a YAML object")

    membership = raw.get("membership", {})
    occlusion = raw.get("occlusion", {})
    models = raw.get("models", {})
    if not all(isinstance(item, dict) for item in (membership, occlusion, models)):
        raise ResearchConfigurationError("membership, occlusion, and models must be objects")
    method = str(membership.get("method", "logistic"))
    if method not in {"geometric", "logistic", "gbdt"}:
        raise ResearchConfigurationError("membership.method must be geometric, logistic, or gbdt")
    correction = str(occlusion.get("method", "spacing"))
    if correction not in {"none", "spacing", "residual"}:
        raise ResearchConfigurationError("occlusion.method must be none, spacing, or residual")
    threshold = float(membership.get("threshold", raw.get("membership_threshold", 0.5)))
    if not 0 <= threshold <= 1:
        raise ResearchConfigurationError("membership threshold must be between zero and one")
    max_hidden = int(occlusion.get("max_hidden_correction", 5))
    if max_hidden < 0:
        raise ResearchConfigurationError("max_hidden_correction cannot be negative")
    min_density = float(occlusion.get("spacing_min_mean_density", 0.5))
    gap_multiplier = float(occlusion.get("spacing_gap_multiplier", 1.5))
    max_gap_multiplier = float(occlusion.get("spacing_max_gap_multiplier", 6.0))
    if min_density < 0 or gap_multiplier <= 1 or max_gap_multiplier <= gap_multiplier:
        raise ResearchConfigurationError(
            "spacing density must be non-negative and gap multipliers must be ordered above one"
        )

    membership_models = {
        name: value
        for name in ("logistic", "gbdt")
        if (value := _model_path(models.get(f"membership_{name}"))) is not None
    }
    feature_version = str(raw.get("feature_version", ""))
    if feature_version != FEATURE_VERSION:
        raise ResearchConfigurationError(
            f"feature_version must be {FEATURE_VERSION!r}, got {feature_version!r}"
        )
    return ResearchConfig(
        feature_version=feature_version,
        random_seed=int(raw.get("random_seed", 42)),
        membership_method=method,
        membership_threshold=threshold,
        occlusion_method=correction,
        max_hidden_correction=max_hidden,
        expected_spacing=_optional_positive(occlusion.get("expected_spacing"), "expected_spacing"),
        spacing_min_mean_density=min_density,
        spacing_gap_multiplier=gap_multiplier,
        spacing_max_gap_multiplier=max_gap_multiplier,
        uncertainty_delta=_optional_positive(
            occlusion.get("uncertainty_delta"), "uncertainty_delta"
        ),
        membership_models=membership_models,
        spacing_model=_model_path(models.get("spacing")),
        uncertainty_model=_model_path(models.get("uncertainty")),
        residual_model=_model_path(models.get("occlusion_residual")),
    )
