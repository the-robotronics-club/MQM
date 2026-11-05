# Final photo and phone-camera pipeline

The distance-based pipeline comes from the last `run_pipeline` definition in
Ishan's merged `training/mess_classifiers.ipynb`, with a later safeguard against
recursive posture reassignment. Runtime never executes notebook cells or training code.

1. Decode and orient the photo (or receive a BGR video frame).
2. YOLOv8n detects people in original frame coordinates, at image size 1280 and
   IoU 0.5. The confidence guard keeps 0.25 for sparse scenes and uses 0.15 when
   at least 25 crowd candidates exist.
3. Ishan's PR #8 seated MobileNetV2 classifies 128×128 RGB, ImageNet-normalized crops.
   The photo/live path loads only this classifier, so missing queue-crop weights
   cannot disable distance grouping.
4. Group standing people by bottom-center distance normalized by mean body height.
   Use distance threshold 1.6, minimum group size 3, and line elongation at least 2.
5. Reassign uncertain seated predictions to a nearby queue when seated probability
   is at most 0.9 and head-height difference is at most 0.2 of body height. Require
   two original standing members of the same queue as supporting neighbors.
   Reassigned seated predictions cannot act as supporting neighbors themselves.
6. Give each detected person exactly one output category: Queue, Sitting, or Other
   standing. These three counts sum to headcount. Blue Q, orange S, and green O
   boxes match the displayed counts in photos, downloads, and phone-camera video.

The queue-crop classifier is not used to decide distance-mode membership. The
calibrated ingestion API retains polygon-gated queue/seated classifiers and its
existing response schema; its zones still need mounted-camera calibration. Photo
and phone results do not write to camera history. Grouping is an estimate, and a
nearby group can be mistaken for a queue. Unavailable inference shows missing
estimates and a warning while retaining the detector count, not fake zero counts.

## Image 419 confirmation

Confirmed labels: queue 25, sitting 13, other standing 1, total 39.
The finalized photo path reproduces the preferred notebook distance output:
queue 20, sitting 15, other standing 0, total 35. Four raw seated predictions
are reassigned into the queue. This result is closer to the queue label than
the queue-crop classifier's 9, but still undercounts the queue by 5.

## Full-dataset diagnostic

All 60 named images decode; one blank annotation row is skipped. Image 419's
labels are now confirmed and included, and the sideways image is corrected.
These are new results, distinct from the earlier 59-image exploratory comparison.

| Mean absolute count error, same 60 images | Posture-only | Final distance categories |
| --- | ---: | ---: |
| Headcount | 5.800 | 5.800 |
| Sitting | 4.617 | 4.550 |
| Standing including queued people | 3.550 | 3.500 |

The bounded reassignment guard reduces errors compared with the previous chained
rule (sitting 5.167 and standing 3.817); differences relative to raw posture-only
counts are small. Only 419 has a confirmed queue label, so broader queue
accuracy is unknown. Do not describe these count errors as per-person accuracy.
Review person boxes and collect queue labels across new capture sessions before
using this as a validated operational queue counter. No new classifier training
was performed.

The latest [bounded-reassignment review](reviews/2026-10-21-bounded-rescue.md)
records the fresh 60-image check, session-group results, and remaining detection
bottleneck. All three counts still match on 4/60 images; the 30/60 target is unmet.

## Run and verify

Keep `ATTRIBUTE_CLASSIFIERS_ENABLED=true` for photo queue/posture estimates.
Run the app with `python -m streamlit run dashboard/app.py`. For evaluation:

```bash
python -m pip install -r requirements-eval.txt
python -m tools.evaluation.real_images --images /path/to/photos --labels /path/to/labels.json \
  --adaptive-confidence --queue-mode distance --output data/outputs/final-distance-evaluation
```

Use `--queue-mode posture` to reproduce raw posture counts. Queue error is computed
only for records carrying `queue_count`; other standing error requires
`roaming_count`. Per-image boxes and model posture probabilities are retained in
local evaluation JSON. No raw photos or source annotations are committed.

136 tests passed and 1 opt-in live-storage test skipped in the isolated validation
copy. Ruff lint/format and MyPy cover the runtime and UI. Tests check grouping,
line versus blob rejection, cautious posture reassignment, exclusive counts,
shared annotation colors, live timing/failure behavior, and UI queue metrics.
The real 419 production result matches the notebook result. Docker/live storage
and physical phone capture were not validated in this pass. No deployment run.

## Frozen-weight accuracy experiments

The later [accuracy study](reviews/2026-10-21-accuracy-trials.md) compares
preprocessing and detector candidates without training on the supplied photos.
Do not treat its reused 60-image benchmark as an unseen test or promote its
highest-scoring settings solely to reach a count-match target.

`MODEL_TRIM_BLACK_PADDING=true` optionally crops outer near-black screenshot
padding before detection and restores original box coordinates for posture and
camera geometry. It skips nearly black frames. This option is off by default:
an earlier comparison with the original chained rule reduced total-count error from 5.80 to 5.05,
but standing error worsens from 3.82 to 4.08 and all-count matches fall from
4/60 to 3/60. Add `--trim-black-padding` to the evaluation command to test it.
Independent footage is required before enabling it for a camera.

The [fixed-camera accuracy review](reviews/2026-10-21-fixed-camera-accuracy.md)
adds a frozen YOLO11m full-frame versus overlapping-crop comparison. Both remain
experimental: full-frame improves some metrics but worsens standing error;
slicing reduces overall total/sitting MAE while worsening standing error and
exact-label total MAE. No default detector switch was made. The camera is not
mounted yet; saved stock-photo zones require replacement with its real view.
