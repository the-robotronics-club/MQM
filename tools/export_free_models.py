"""Export unchanged FP32 checkpoints for low-memory hosting; no training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from ultralytics import YOLO

from inference.classifiers import _load_mobilenet_v2


def main() -> None:
    torch.set_num_threads(2)
    root = Path("models")
    source = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("yolov8n.pt", "seated_classifier.pt")
    }
    # Dynamic spatial axes preserve the production stride-32 rectangular input.
    YOLO(str(root / "yolov8n.pt")).export(
        format="onnx",
        imgsz=1280,
        dynamic=True,
        simplify=False,
        opset=17,
        nms=False,
    )
    model = _load_mobilenet_v2(root / "seated_classifier.pt")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            torch.zeros(1, 3, 128, 128),
            str(root / "seated_classifier.onnx"),
            input_names=["images"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
            dynamo=False,
        )
    assert source == {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source
    }
    manifest = {
        "training": False,
        "precision": "FP32",
        "source_sha256": source,
        "export_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("yolov8n.onnx", "seated_classifier.onnx")
        },
    }
    (root / "onnx-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
