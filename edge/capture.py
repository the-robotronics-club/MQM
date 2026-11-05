"""RPi JPEG capture/uploader; --image photo.jpg --once tests without a Pi."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests


def positive_number(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def capture_jpeg() -> bytes:
    command = shutil.which("rpicam-still") or shutil.which("libcamera-still")
    if not command:
        raise RuntimeError("Install rpicam-apps, or use --image photo.jpg --once")
    width = int(positive_number("IMAGE_WIDTH", 1280))
    height = int(positive_number("IMAGE_HEIGHT", 720))
    result = subprocess.run(
        [
            command,
            "-n",
            "--width",
            str(width),
            "--height",
            str(height),
            "--timeout",
            "1500",
            "--encoding",
            "jpg",
            "-o",
            "-",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    return result.stdout


def send_snapshot(image: bytes, server_url: str, camera_id: str, key: str) -> dict:
    if not image.startswith(b"\xff\xd8\xff"):
        raise ValueError("Snapshot must contain JPEG bytes")
    response = requests.post(
        f"{server_url.rstrip('/')}/ingest/{quote(camera_id, safe='')}",
        files={"file": ("snapshot.jpg", image, "image/jpeg")},
        headers={"X-Camera-Key": key},
        timeout=(10, positive_number("UPLOAD_TIMEOUT_SECONDS", 180)),
    )
    response.raise_for_status()
    body = response.json()
    if body.get("status") != "ok" or "headcount" not in body.get("metrics", {}):
        raise ValueError("Server did not acknowledge a stored reading")
    return body["metrics"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Send this JPEG instead of capturing a frame")
    parser.add_argument("--once", action="store_true", help="Send once and return success/failure")
    args = parser.parse_args()
    server = os.environ.get("SERVER_URL", "").strip().rstrip("/")
    key = os.environ.get("CAMERA_API_KEY", "").strip()
    camera = os.environ.get("CAMERA_ID", "mess_main").strip()
    parsed = urlsplit(server)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        parser.error("Set SERVER_URL to the API's HTTP(S) base URL")
    if not key or key.startswith("change-me") or not camera:
        parser.error("Set CAMERA_API_KEY and CAMERA_ID")
    interval = positive_number("CAPTURE_INTERVAL_SECONDS", 10)
    print(f"Sending snapshots for {camera}; interval={interval:g}s", flush=True)
    while True:
        started = time.monotonic()
        try:
            image = args.image.read_bytes() if args.image else capture_jpeg()
            metrics = send_snapshot(image, server, camera, key)
            print(f"Stored: people={metrics['headcount']} at {metrics['timestamp']}", flush=True)
        except (
            OSError,
            ValueError,
            RuntimeError,
            subprocess.SubprocessError,
            requests.RequestException,
        ) as exc:
            print(f"Snapshot failed ({type(exc).__name__}); next interval will retry", flush=True)
            if args.once:
                return 1
        else:
            if args.once:
                return 0
        time.sleep(max(0, interval - (time.monotonic() - started)))


if __name__ == "__main__":
    raise SystemExit(main())
