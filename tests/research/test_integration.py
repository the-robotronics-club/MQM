from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO

from PIL import Image

import research.estimator
from api.models import CameraMetrics
from inference.pipeline import process_frame
from tests import fixtures as fx


def jpeg() -> bytes:
    output = BytesIO()
    Image.new("RGB", (736, 490)).save(output, "JPEG")
    return output.getvalue()


def test_production_api_schema_remains_backward_compatible(monkeypatch) -> None:
    monkeypatch.setenv("QUEUE_ESTIMATOR", "production")
    metrics = process_frame(jpeg(), "mess_main", detector=lambda _: fx.BUSY)
    assert metrics["queue_count"] == 3
    assert "queue_visible" not in metrics
    parsed = CameraMetrics(**metrics, timestamp=datetime.now(UTC))
    assert parsed.queue_visible is None


def test_research_strategy_extends_existing_metrics(monkeypatch) -> None:
    monkeypatch.setenv("QUEUE_ESTIMATOR", "occlusion")
    monkeypatch.setattr(
        research.estimator,
        "estimate_queue",
        lambda *_: {
            ".optional": "removed",
            "queue_count": 5,
            "queue_visible": 3,
            "queue_hidden_estimate": 2,
            "queue_ci_low": 4,
            "queue_ci_high": 6,
            "queue_method": "P2_SPACING",
            "queue_model_version": "spacing-v1",
            "research_diagnostics": {"private": True},
        },
    )
    metrics = process_frame(jpeg(), "mess_main", detector=lambda _: fx.BUSY)
    metrics.pop(".optional")
    assert metrics["headcount"] == 12
    assert metrics["queue_count"] == 5
    assert "research_diagnostics" not in metrics
    CameraMetrics(**metrics, timestamp=datetime.now(UTC))
