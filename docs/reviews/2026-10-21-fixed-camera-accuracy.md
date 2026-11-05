# Accuracy plan for the fixed cafeteria camera

The intended product counts **queued people, sitting people, and other standing
people** in a fixed cafeteria view. These categories must be exclusive and sum
to detected headcount. "Standing" in existing evaluation tables includes queued
people. A single photo cannot establish that a standing person is roaming;
movement needs video evidence.

The camera is not mounted yet. `config/zones.json` explicitly contains stock-photo
placeholder polygons. `research/configs/cameras/mess_main.example.yaml` is an
uncalibrated template. Neither supplies a validated map for this cafeteria.

## What currently limits accuracy

The shipped detector is YOLOv8n; Ishan's MobileNetV2 estimates posture. A shared
distance rule groups standing detections and conservatively rescues uncertain
seated predictions into a queue. It can confuse nearby standing groups with
queues because it does not know the serving counter or actual queue route.
The separate queue crop classifier is used by the calibrated ingestion API,
not by photo/live distance grouping. Replacing the shared grouping with that
classifier alone is not supported by current evaluation evidence.

The last production check on the reused 60 images has mean absolute count
errors of 5.80 total, 4.55 sitting, and 3.50 standing including queue. Total matches
on 7/60; all three counts match on 4/60. Ten images use label ranges. Count
agreement can hide missed people offset by duplicate or false detections; it
does not measure detector recall or correct per-person categorization.

Only image 419 has a confirmed queue label. Broad queue precision/recall cannot
be measured from the current image-level posture labels. Its confirmed counts
are 25 queued, 13 sitting, one other standing, 39 total.

## Proposed fixed-camera pipeline

1. Decode and orient frames; keep the mounted camera's field of view and
   resolution stable. Review the mounting view for distant people and table
   occlusion before calibrating geometry.
2. Detect visible people. Compare the current detector with the frozen sliced
   candidate below, including latency and false/duplicate boxes. Do not add an
   estimated number of invisible people merely to make the count match.
3. Classify posture from person crops. Use the same crop preprocessing and
   category definitions in evaluation and runtime.
4. Determine queue membership using the **actual serving corridor and queue
   path**, with perspective-aware geometry where floor correspondences can be
   measured. A position near a table or counter alone is insufficient evidence
   of posture or membership. Occluded feet make ground positions uncertain.
5. Once ordered clips exist, evaluate tracking and short temporal smoothing
   to reduce flicker and distinguish passersby from people waiting. Expire lost
   tracks promptly; persistence must not inflate current occupancy. This is a
   future experiment, not implemented or validated by this change.
6. Produce exclusive Q/S/O categories and colored boxes through one shared
   scene path. Measure accuracy before smoothing and after smoothing separately,
   including lag when people leave.

Existing `research/` modules provide homography, queue-path features, geometry
rules, session splits, and evaluation. Their software tests do not establish
real cafeteria accuracy. The calibrated API and photo/live distance path still
have different membership logic; integrating a validated camera-aware scene
policy remains work after an actual view and queue annotations exist.

## Frozen detector experiment

`tools/evaluation/sliced_detection.py` records a plan before inference and tests
exactly two policies: YOLO11m full frame, and that same full frame plus four
overlapping 60%-sized crops. Both use production black-padding removal, 1280
input, IoU 0.5, the existing 0.25/0.15 crowd confidence guard, and the current
bounded queue rule. Internal tile-edge fragments are rejected, boxes are
restored to source coordinates, and duplicate suppression precedes the global
crowd guard. No filename selects a policy. Orientation metadata is retained.

This tests the small-person hypothesis motivated by
[slicing-aided inference](https://arxiv.org/abs/2202.06934); it is a small custom
implementation, not the SAHI package or a reproduction of its paper results.
No training, confidence search, or automatic production promotion occurs.
Classifier and detector hashes are recorded and checked after inference.
All photos get local colored overlays and per-image predictions.

Reproduce from the repository root:

```bash
python -m tools.evaluation.sliced_detection \
  --images /path/to/photos --labels /path/to/dataset_labels.json \
  --output data/outputs/sliced-detection
```

The full-frame control isolates the crop effect from model/padding changes.
The previous YOLOv8n production results provide a second reference. Hardware
timings are measurements on this Mac, not Raspberry Pi or deployment benchmarks.

### Completed 60-image comparison

| Metric | Current production | YOLO11m full | YOLO11m sliced |
| --- | ---: | ---: | ---: |
| Total count MAE | 5.800 | 5.417 | 4.800 |
| Sitting count MAE | 4.550 | 3.883 | 2.917 |
| Standing count MAE, including queue | 3.500 | 3.717 | 4.267 |
| Total count matches | 7/60 | 12/60 | 5/60 |
| All three counts match | 4/60 | 7/60 | 4/60 |
| Mean inference seconds, this experiment | Not rerun | 1.572 | 5.979 |

The 30/60 all-count target remains unmet. The larger full-frame detector is a
candidate for new-footage testing, but worsens standing error. Slicing improves
mean total error by recovering some large crowd misses while producing more
overcounts elsewhere. Total error improves on 27 images and worsens on 25 versus
production; standing error improves on 20 and worsens on 30. Visual review found
overlapping head/torso and larger body boxes that can survive IoU suppression.
Per-person human boxes are needed to measure duplicate and missed detections.

On the 50 exact-label photos, total MAE is 3.700 production, 3.900 full-frame,
and 4.260 sliced. Thus the overall sliced total-MAE gain does **not** carry over
to the exact-label subset. On the 14 previously inspected unflagged validation
images, none of the three pipelines has an all-count match. These tradeoffs do
not support a default detector switch. The production model and settings remain
unchanged; both candidates stay experimental.

419 illustrates why a queue count alone cannot select the pipeline:

| 419 counts | Confirmed | Production | YOLO11m full | YOLO11m sliced |
| --- | ---: | ---: | ---: | ---: |
| Total | 39 | 35 | 38 | 41 |
| Queue | 25 | 20 | 23 | 25 |
| Sitting | 13 | 15 | 15 | 16 |
| Other standing | 1 | 0 | 0 | 0 |

The sliced queue count matches, but total and posture categories do not. It
also has no per-person queue annotation proving those are the correct 25 people.

Aggregates and frozen plan:
`docs/evaluation/2026-10-21-sliced-detection.json`.
Local results: `data/outputs/sliced-detection/comparison.md`, `full.json`,
`sliced.json`, and `overlays/{full,sliced}/`. All 60 images completed both policies;
detector/classifier checkpoint hashes are unchanged. No source photos, individual
annotations, or overlays are committed.

136 software tests passed; one opt-in live-stack test skipped. Ruff and MyPy
passed. Tests include duplicate-content and clip leakage, coordinate restoration,
tile-edge fragments, duplicate suppression, and applying the crowd guard after
merging. No deployment or physical-camera validation was performed.

## Generalization and data requirements

The 60 photos remain inference-only. Repeatedly inspecting them and selecting
settings on them can overfit the pipeline even when model weights never change.
They therefore remain a diagnostic development set, not an unseen test. The
existing filename-based development/validation grouping is also previously
inspected and does not establish generalization. The 30/60 target cannot justify
image-specific corrections or a large search over these labels.

After mounting, capture separate sessions with light/heavy queues, sitting
people, standing people outside the queue, staff, and people passing the line.
Annotate visible person boxes plus posture and queue membership, including
missed detections. Keep ambiguous and heavily occluded labels explicit.
Reserve entire new sessions before tuning and keep neighboring frames and
crops from the same source in one split. Do not add these 60 images to training.
Any future training needs separately sourced, reviewed data and authorization.

The dataset validator now rejects byte-identical images across splits even
under different image/session names, and rejects one camera clip crossing
splits. It still warns on duplicates within a split. Re-encoded/cropped images
need trustworthy source metadata; exact hashes cannot catch every near-duplicate.
Run validation with **all** split files to check cross-split leakage:

```bash
python -m research.datasets.validation --manifest /path/to/manifest.json \
  --root /path/to/images --splits /path/to/splits
```

Report detector precision/recall on human boxes; posture confusion on matched
people; end-to-end queue precision/recall; total/Q/S/O count MAE and exact/within-1
matches; per-session results; and latency. Count-only labels currently support
only the count portions of this evaluation. Original checkpoint training/test
overlap remains unverified.
