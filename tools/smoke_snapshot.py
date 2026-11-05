"""Real JPEG -> API -> Redis status -> Influx history smoke check, without an RPi."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from edge.capture import send_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--camera", default="mess_main")
    parser.add_argument("--output", type=Path, default=Path("data/outputs/snapshot-smoke.json"))
    args = parser.parse_args()
    load_dotenv()
    url = args.url.rstrip("/")
    ready = requests.get(f"{url}/ready", timeout=15)
    ready.raise_for_status()
    stored = send_snapshot(args.image.read_bytes(), url, args.camera, os.environ["CAMERA_API_KEY"])
    response = requests.get(f"{url}/status/{args.camera}", timeout=15)
    response.raise_for_status()
    current = response.json()
    if not current["online"] or current["metrics"]["timestamp"] != stored["timestamp"]:
        raise RuntimeError("Latest state does not contain the uploaded reading")
    for _ in range(15):
        response = requests.get(f"{url}/history/{args.camera}?minutes=60", timeout=15)
        response.raise_for_status()
        history = response.json()["points"]
        matched = next((p for p in history if p["timestamp"] == stored["timestamp"]), None)
        if matched is not None:
            break
        time.sleep(1)
    else:
        raise RuntimeError("Stored reading did not appear in history")
    for field in ("headcount", "queue_count", "sitting", "standing_other_count", "scene_method"):
        if matched[field] != stored[field] or current["metrics"][field] != stored[field]:
            raise RuntimeError(f"Readback mismatch for {field}")
    result = {
        "status": "passed",
        "api_url": url,
        "metrics": stored,
        "latest_readback": True,
        "history_readback": True,
        "physical_rpi": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
