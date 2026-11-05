# Retrained model integration and real-image evaluation

Upstream `4634cff` merged retrained MobileNetV2 classifiers from PR #8.
The new checkpoint hashes are recorded in
`../evaluation/2026-10-21-summary.json`. The September review describes the old
checkpoints and must not be used as verification of these new artifacts.

The supplied notebook trains and evaluates at 128×128 using PIL bilinear
resize, RGB, and ImageNet normalization. Production previously resized to
224×224 using OpenCV area interpolation. Production now matches the notebook's
inference transform, with an automated parity check against torchvision.
Label interpretation is unchanged: queue class 1; seated class 0.

Posture classification is shared by uploaded photos, live video, and the
batch evaluator. Arbitrary photos report sitting/standing estimates without
applying the saved camera polygons. Queue and seat occupancy still require
camera calibration. Classifier errors in photo mode retain person counts and
show a warning, rather than displaying a false zero posture count.

## Measured results

YOLOv8n, confidence 0.25, image size 1280; new seated checkpoint, 128×128 crops.
60 named images evaluated, including HEIC; one blank annotation row skipped.

| Mean absolute count error | All 60 | Exact, unflagged 49 |
| --- | ---: | ---: |
| Headcount | 8.700 | 5.816 |
| Sitting | 5.967 | 3.755 |
| Standing | 4.367 | 3.816 |

For count ranges, error is distance to the nearest interval endpoint, or zero
inside the range. The exact cohort uses ordinary absolute error. These are
count errors, not percentages or per-person classification accuracy. No
bounding boxes or queue labels were supplied, so detection mAP, classification
precision/recall, and queue accuracy cannot be established from these labels.

The detector undercounts many dense scenes. Training/test overlap is unknown;
this run is a diagnostic evaluation, not evidence of held-out generalization.
No thresholds were tuned on the evaluation images. Per-image predictions and
annotations remain local under `data/outputs/real-images/` and
`dataset_labels.json`; raw photos remain in the ignored download directory.

## Software checks

- Full suite in an isolated copy with tracked sample fixtures: 107 passed,
  1 live-stack test skipped. Existing local sample deletions were preserved.
- Photo-only suite: 9 passed, including posture counts and classifier failure handling.
- Ruff lint/format, MyPy (70 source files), diff whitespace checks, and deployment
  Compose configuration validation passed.
- Container build and live storage integration were not run: Docker daemon unavailable.
