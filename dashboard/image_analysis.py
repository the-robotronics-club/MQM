"""Analyze uploaded photos locally without publishing camera readings."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from inference.annotations import annotate_people

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def decode_photo(data: bytes, rotation_degrees_ccw: int = 0) -> Image.Image:
    if rotation_degrees_ccw not in (0, 90, 180, 270):
        raise ValueError("Choose a rotation of 0, 90, 180, or 270 degrees.")
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Choose a JPEG or PNG image smaller than 10 MB.")
    if not (data.startswith(b"\xff\xd8\xff") or data.startswith(b"\x89PNG\r\n\x1a\n")):
        raise ValueError("Only JPEG and PNG images are supported.")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG"}:
                raise ValueError("Only JPEG and PNG images are supported.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("Choose an image with no more than 20 million pixels.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            return (
                image.rotate(rotation_degrees_ccw, expand=True) if rotation_degrees_ccw else image
            )
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        ModuleNotFoundError,
    ) as exc:
        raise ValueError(
            "This file could not be opened as an image. Try another JPEG or PNG."
        ) from exc


def analyze_photo(data: bytes, camera_id: str | None = None, rotation_degrees_ccw: int = 0) -> dict:
    image = decode_photo(data, rotation_degrees_ccw)
    frame = np.asarray(image)[:, :, ::-1].copy()
    if camera_id:
        from inference.zones import assign_zones, load_zones

        config = load_zones(camera_id)
        # Validate geometry before running the expensive detector.
        assign_zones([], config, frame_size=image.size)

    from inference.model import detect_people

    detections = detect_people(frame)
    from inference.classifiers import attribute_classifiers_enabled

    scene = None
    posture_error = None
    if attribute_classifiers_enabled():
        from inference.scene import analyze_scene

        try:
            scene = analyze_scene(frame, detections)
        except Exception:
            import logging

            logging.getLogger(__name__).exception("Photo queue/posture analysis failed")
            posture_error = (
                "Queue and posture estimates are unavailable; person counting continues."
            )
    metrics = None
    if camera_id:
        from inference.classifiers import classify_people
        from inference.metrics import metrics_from_assignments

        assignments = assign_zones(detections, config, frame_size=image.size)
        attributes = (
            classify_people(frame, assignments) if attribute_classifiers_enabled() else None
        )
        metrics = metrics_from_assignments(
            assignments,
            config,
            detections_raw=len(detections),
            queue_flags=[item.is_queue for item in attributes] if attributes else None,
            seated_flags=[item.is_seated for item in attributes] if attributes else None,
        )

    image = annotate_people(image, scene["people"] if scene is not None else detections)
    output = BytesIO()
    image.save(output, format="PNG")
    return {
        "headcount": len(detections),
        "sitting": scene["sitting"] if scene is not None else None,
        "standing": scene["standing"] if scene is not None else None,
        "queue_count": scene["queue_count"] if scene is not None else None,
        "standing_other_count": scene["standing_other_count"] if scene is not None else None,
        "scene": scene,
        "posture_error": posture_error,
        "detections": detections,
        "metrics": metrics,
        "image": output.getvalue(),
        "width": image.width,
        "height": image.height,
    }
