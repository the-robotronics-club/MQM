"""FastAPI application combining isolated ingestion and read routers."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from api.config import configured_cors_origins, get_settings
from api.ingestion import router as ingestion_router
from api.read import router as read_router
from api.storage.influx_client import check_influx
from api.storage.redis_client import check_redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()  # fail fast on missing credentials or invalid camera IDs
    if settings.frame_metrics_mode == "distance" and settings.queue_estimator != "production":
        raise ValueError("FRAME_METRICS_MODE=distance requires QUEUE_ESTIMATOR=production")
    if settings.inference_backend == "onnx":
        from inference.onnx_runtime import get_onnx_models

        get_onnx_models()  # Fail startup if exports or runtime are unavailable.
    if settings.queue_estimator != "production":
        from research.calibration import load_calibration
        from research.config import load_research_config

        load_research_config()
        for camera_id in settings.known_cameras:
            load_calibration(camera_id)
    yield


app = FastAPI(
    title="Mess Congestion API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS origins are environment-controlled. The Streamlit server uses server-side
# requests, but this also supports a future static dashboard without opening writes
# to arbitrary browser origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(configured_cors_origins()),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Camera-Key"],
)


app.include_router(ingestion_router, tags=["ingestion"])
app.include_router(read_router, tags=["read"])


@app.get("/health")
def health() -> dict[str, str]:
    """Process liveness; does not pretend external dependencies are healthy."""
    return {"status": "ok"}


@app.get("/ready")
async def ready(response: Response) -> dict:
    """Dependency readiness with explicit per-service status."""
    redis_ok = influx_ok = False
    errors: dict[str, str] = {}
    try:
        redis_ok = await run_in_threadpool(check_redis)
    except Exception as exc:
        errors["redis"] = type(exc).__name__
    try:
        influx_ok = await run_in_threadpool(check_influx)
    except Exception as exc:
        errors["influxdb"] = type(exc).__name__
    if not (redis_ok and influx_ok):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if redis_ok and influx_ok else "not_ready",
        "inference_backend": get_settings().inference_backend,
        "storage_backend": get_settings().storage_backend,
        "services": (
            {"sqlite": redis_ok and influx_ok}
            if get_settings().storage_backend == "sqlite"
            else {"redis": redis_ok, "influxdb": influx_ok}
        ),
        "errors": errors,
    }
