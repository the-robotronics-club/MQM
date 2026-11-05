# First count-improvement pass

## Result

Keep Ishan's PR #8 MobileNetV2 checkpoints unchanged. YOLOv8n remains the
person detector. Production now runs a candidate pass at confidence 0.15,
retains that cutoff when at least 25 people candidates survive NMS, and otherwise
filters back to 0.25. IoU remains 0.5, image size 1280, seated cutoff 0.5.
This is an inference change, not classifier retraining.

| Mean absolute count error, same 59 images | Fixed 0.25 | Crowd guard |
| --- | ---: | ---: |
| Total people | 8.119 | 5.831 |
| Sitting | 5.746 | 4.593 |
| Standing | 3.966 | 3.441 |
| Images with all three counts correct | 4/59 | 5/59 |
| Images with total count within two | 24/59 | 27/59 |

Total-count error decreases 28%. This does not mean 72% person accuracy.
Individual count matches change from total 6→7, sitting 14→14, standing 13→17.
Sitting counts within two change 34→33; improvements are not universal.
The JSON aggregate includes all trials and both exact and near-match counts.

## Data review and fair comparison

- `IMG_20261005_200419.jpg`: user corrected standing count to 25–27; the
  original sitting count 13 is unconfirmed. Excluded from selection and comparison.
- `IMG_20260915_201649.jpg`: visually confirmed that stored pixels are sideways
  despite EXIF transpose. Apply 90° CCW in local label metadata. Both before and
  after runs use this correction. Upload UI now offers explicit photo rotation
  before preview, geometry validation, and inference.
- All 60 named files decode, including HEIC. One blank annotation row is skipped.
- Keep historical 60-image results separate: their labels/orientation differ.
- No exact duplicate labeled files were found by SHA-256.

## Experiments and limitations

Initial search: YOLOv8n at 1280/IoU 0.5, YOLOv8n at 1280/IoU 0.7,
YOLOv8n at 1600/IoU 0.7, YOLOv8s at 1280/IoU 0.7. For each, test
confidence cutoffs 0.05/0.10/0.15/0.20/0.25/0.35. Cache low-cutoff boxes and
posture probabilities; filter them for each cutoff. Fresh production inference
on all 60 named images exactly reproduced the selected cached counts.

Choose using sitting MAE + standing MAE on 34 unflagged development images.
The initial winner is fixed YOLOv8n 0.15. Testing seated cutoffs
0.25/0.35/0.50/0.65/0.75 selects the original 0.50 cutoff.
A follow-up tests sparse-scene guards at 15/20/25/30/40 candidates and selects 25
using the same development objective. Ten additional uncertain images contribute
only to the overall interval-error report, not selection.

Fifteen validation images are grouped by capture session: filenames containing
20261002, 20260903, or REMAN DEY. Validation headcount MAE changes 6.00→3.40,
sitting 5.33→3.93, standing 2.67→3.07. Thus standing gets worse on that subset.
The crowd guard was explored after viewing initial validation results; this
is exploratory validation, not a pristine held-out result. The dataset was
already inspected, and training overlap is unknown. New mounted-camera footage
is needed to assess generalization. Image-level counts cannot provide detector
mAP, crop-classification accuracy, or queue accuracy. Supervised retraining needs
reviewed person boxes and sitting/standing labels on individual crops.

## Runtime and reproducibility

`get_detector()` enables the crowd guard by default. `PersonDetector()` stays a
fixed-cutoff building block unless explicitly configured. An explicit per-call
confidence override disables adaptation so comparisons remain reproducible.
`MODEL_ADAPTIVE_CONFIDENCE=false` restores fixed behavior. Parameters are
`MODEL_CONFIDENCE=0.25`, `MODEL_CROWD_CONFIDENCE=0.15`, and
`MODEL_CROWD_MIN_PEOPLE=25`. The existing local environment is aligned.
Camera-zone confidence floors still apply independently (current sample 0.25).
These photo/live gains do not establish improvements to calibrated queue metrics.

```bash
python -m tools.evaluation.tune_detector --images /path/to/photos --labels /path/to/labels.json
python -m tools.evaluation.calibrate_posture --images /path/to/photos --labels /path/to/labels.json
python -m tools.evaluation.calibrate_crowd
python -m tools.evaluation.real_images --images /path/to/photos --labels /path/to/labels.json \
  --adaptive-confidence --output data/outputs/improved-real-images
```

Install `requirements-eval.txt`. Raw images, corrected labels, trial predictions,
and per-image comparisons remain local. Aggregate:
`docs/evaluation/2026-10-21-improvements.json`.
The [Ultralytics prediction reference](https://docs.ultralytics.com/modes/predict/)
explains confidence, image size, and NMS IoU. Incorrect local comments implying
that lower NMS IoU preserves more overlapping people have been corrected.

## Verification

112 tests passed, 1 opt-in live-storage test skipped, in an isolated source copy
with the tracked test fixtures restored. Existing local sample deletions remain
untouched. Tests cover sparse/crowd switching, explicit confidence overrides,
invalid settings, and photo orientation before inference. Ruff lint/format and
MyPy passed. Fresh real-image inference matches the cached search predictions.
No deployment was attempted.
