"""FP32 ONNX inference with production preprocessing; no torch runtime import."""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from inference.posture import PosturePrediction

ROOT = Path(__file__).resolve().parent.parent / "models"


def letterbox(frame: np.ndarray, size: int = 1280) -> tuple[np.ndarray, float, int, int]:
    h, w = frame.shape[:2]
    gain = min(size / h, size / w)
    resized = (round(w * gain), round(h * gain))
    dw, dh = (size - resized[0]) % 32, (size - resized[1]) % 32
    left, top = round(dw / 2 - 0.1), round(dh / 2 - 0.1)
    right, bottom = round(dw / 2 + 0.1), round(dh / 2 + 0.1)
    view = cv2.resize(frame, resized, interpolation=cv2.INTER_LINEAR)
    view = cv2.copyMakeBorder(
        view, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    tensor = np.ascontiguousarray(view[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor / 255.0, gain, left, top


def suppress(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.5) -> list[int]:
    """Greedy single-class NMS, same continuous-coordinate IoU as torchvision."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while len(order) and len(keep) < 300:
        i = int(order[0])
        keep.append(i)
        other = order[1:]
        if not len(other):
            break
        inter = np.maximum(
            0,
            np.minimum(boxes[i, 2:], boxes[other, 2:]) - np.maximum(boxes[i, :2], boxes[other, :2]),
        ).prod(axis=1)
        area = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
        overlap = inter / (area[i] + area[other] - inter)
        order = other[overlap <= iou]
    return keep


def decode_predictions(
    raw: np.ndarray, shape: tuple[int, int], gain: float, left: int, top: int
) -> list[dict]:
    candidates = raw[0].T
    classes = candidates[:, 4:].argmax(axis=1)
    confidence = candidates[:, 4:].max(axis=1)
    chosen = (classes == 0) & (confidence > 0.15)
    values, scores = candidates[chosen, :4], confidence[chosen]
    if not len(values):
        return []
    boxes = np.concatenate(
        (values[:, :2] - values[:, 2:] / 2, values[:, :2] + values[:, 2:] / 2), axis=1
    )
    keep = suppress(boxes, scores)
    boxes = boxes[keep]
    scores = scores[keep]
    cutoff = 0.15 if len(keep) >= 25 else 0.25
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) / gain, 0, shape[1])
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) / gain, 0, shape[0])
    return [
        {"bbox": box.tolist(), "confidence": float(score)}
        for box, score in zip(boxes, scores, strict=True)
        if score >= cutoff
    ]


def crop_tensor(frame: np.ndarray, box: list[float]) -> np.ndarray:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid person crop bounds")
    rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    resized = np.asarray(Image.fromarray(rgb).resize((128, 128), Image.Resampling.BILINEAR))
    tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (tensor - mean) / std


class OnnxSceneModels:
    def __init__(self) -> None:
        cv2.setNumThreads(1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.detector = ort.InferenceSession(
            str(ROOT / "yolov8n.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.seated = ort.InferenceSession(
            str(ROOT / "seated_classifier.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.lock = threading.Lock()

    def detect_people(self, frame: np.ndarray) -> list[dict]:
        tensor, gain, left, top = letterbox(frame)
        with self.lock:
            raw = self.detector.run(None, {self.detector.get_inputs()[0].name: tensor})[0]
        return decode_predictions(raw, frame.shape[:2], gain, left, top)

    def classify(self, frame: np.ndarray, detections: list[dict]) -> list[PosturePrediction]:
        output = []
        # Small batches bound peak memory in the 512 MB free service.
        for start in range(0, len(detections), 8):
            batch = np.stack([crop_tensor(frame, d["bbox"]) for d in detections[start : start + 8]])
            with self.lock:
                logits = self.seated.run(None, {self.seated.get_inputs()[0].name: batch})[0]
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            for p in probabilities[:, 0]:
                output.append(
                    PosturePrediction(
                        "Sitting" if p >= 0.5 else "Standing", float(p if p >= 0.5 else 1 - p)
                    )
                )
        return output


@lru_cache(maxsize=1)
def get_onnx_models() -> OnnxSceneModels:
    if os.environ.get("FRAME_METRICS_MODE", "calibrated") != "distance":
        raise ValueError("ONNX free hosting requires distance snapshots")
    expected = {
        "MODEL_CONFIDENCE": "0.25",
        "MODEL_IOU": "0.5",
        "MODEL_IMAGE_SIZE": "1280",
        "MODEL_CROWD_CONFIDENCE": "0.15",
        "MODEL_CROWD_MIN_PEOPLE": "25",
    }
    for key, value in expected.items():
        if float(os.environ.get(key, value)) != float(value):
            raise ValueError(f"Free ONNX profile requires {key}={value}")
    if os.environ.get("MODEL_ADAPTIVE_CONFIDENCE", "true").lower() not in {
        "true",
        "1",
        "yes",
        "on",
    }:
        raise ValueError("Free ONNX profile requires adaptive confidence")
    if os.environ.get("MODEL_TRIM_BLACK_PADDING", "false").lower() not in {
        "false",
        "0",
        "no",
        "off",
    }:
        raise ValueError("Free ONNX profile uses the untrimmed production frame")
    return OnnxSceneModels()
