"""Controlled paper ablation dimensions and CLI value validation."""

from __future__ import annotations

ABLATIONS = {
    "detector": ("pretrained", "fine_tuned"),
    "reference_point": ("centroid", "bottom_center"),
    "coordinate_space": ("image", "ground"),
    "membership_geometry": ("polygon", "queue_path"),
    "membership_model": ("geometric", "logistic", "gbdt"),
    "local_density": (False, True),
    "occlusion_feature": (False, True),
    "hidden_correction": (False, True),
    "source_resolution": (),
    "inference_resolution": (),
    "temporal": (False, True),
}


def parse_ablation(values: list[str]) -> dict[str, str | bool | int]:
    result: dict[str, str | bool | int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"ablation must use key=value syntax: {value!r}")
        key, raw = value.split("=", 1)
        if key not in ABLATIONS:
            raise ValueError(f"unknown ablation dimension: {key}")
        choices = ABLATIONS[key]
        parsed: str | bool | int
        if raw.lower() in {"true", "false"}:
            parsed = raw.lower() == "true"
        elif key in {"source_resolution", "inference_resolution"}:
            parsed = int(raw)
            if parsed <= 0:
                raise ValueError(f"{key} must be positive")
        else:
            parsed = raw
        if choices and parsed not in choices:
            allowed = ", ".join(str(item) for item in choices)
            raise ValueError(f"{key} must be one of: {allowed}")
        result[key] = parsed
    return result
