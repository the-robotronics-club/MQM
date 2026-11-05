# Bounded queue reassignment with frozen models

The default photo/video scene pipeline now requires support from **two original
standing detections in the same queue** before reclassifying an uncertain seated
prediction. Reclassified people cannot recursively pull more seated neighbors
into a queue. Geometry thresholds, classifier weights, detector weights, and
image size are unchanged. The helper's legacy behavior remains available through
explicit overrides for comparison.

## Fresh 60-image production result

| Mean absolute count error | Previous rule | Bounded rule |
| --- | ---: | ---: |
| Total | 5.800 | 5.800 |
| Sitting | 5.167 | 4.550 |
| Standing including queue | 3.817 | 3.500 |

All three counts match on **4/60 images**, unchanged. Total matches on 7/60, also
unchanged. **The 30/60 target remains unmet.** With these fixed detection boxes,
changing posture or queue logic alone cannot produce 30 all-count matches.
Ten images use count ranges rather than exact labels; matches respect those
supplied intervals. Count matches do not establish per-person detection accuracy.

The rule improves sitting error on 21 images and worsens it on 3; standing error
improves on 15 and worsens on 9. Total reassigned seated predictions fall from
117 to 62. All category counts remain exclusive and sum to detected headcount.

The combined sitting/standing error improves in both diagnostic session groups:

| Session group | Previous rule | Bounded rule |
| --- | ---: | ---: |
| Development, 35 unflagged images | 6.915 | 6.000 |
| Validation, 14 unflagged images | 7.500 | 6.643 |

The 11 flagged images contribute only to the overall comparison. These sessions
have already been inspected in earlier experiments, so the grouped comparison
is diagnostic and does not establish unseen generalization.

419 is unchanged: 35 total, 20 queued, 15 sitting, 0 other standing, compared with
confirmed 39 total, 25 queued, 13 sitting, and 1 other standing. Queue accuracy
across scenes is still unknown because only this image has a confirmed queue label.

## Fixed experiments

Before running, `tools/evaluation/frozen_pipeline.py` records its plan locally.
It compares original versus flip-averaged posture probabilities, three fixed
detection sources, and four queue policies: 24 variants, without training or a
confidence-threshold sweep. Sources are production YOLOv8n, the prior YOLO11m
candidate, and a conservative two-detector agreement merge. Source prediction
and classifier hashes are recorded. No image count labels enter inference, and
there are no filename-specific corrections.

The development leader is flip averaging with reassignment disabled. That
combination is not promoted: queue membership has only one labeled example, and
its whole-image count score alone is insufficient to choose the operational
queue policy. The promoted change is the minimal structural guard against
recursive reassignment; it adds no classifier forward pass and improves errors
in both session groups. Flip averaging and detector agreement remain experiments.
The best all-count result among these 24 variants is still 8/60, from YOLO11m
with different error tradeoffs; it is not made the default.

A separate [official YOLO26m](https://docs.ultralytics.com/models/yolo26) checkpoint
was tested with the existing confidence guard and unchanged scene settings at
the time of the comparison. Its native end-to-end detector does not use the
standard NMS step. Total-count error is 6.033, sitting 4.900, and standing 3.883,
with 4/60 all-count matches. It is not selected.

## Data audit and avoiding overfitting

No training occurred. The supplied 60 photos remain outside any new training
set. Original classifier files are unchanged, and inference uses evaluation
mode without gradient updates. Do not choose further rules merely to hit 30/60;
collect separate capture sessions for developing and checking the next detector
change. Keep neighboring frames and crops from one source image in one split.

A filename-only audit of the local training copy found 394 `not_queue` filenames
also present in `seated`, and all 229 `queue` filenames also present in `standing`.
The notebook's earlier labeler copies each selection into these paired folders;
a later labeler supports a separate standing-outside-queue choice. This is a
reason to review class coverage on independent training images, not proof of the
exact data used by the pushed checkpoints. Byte-level corpus auditing was not
completed, and the original training/test overlap remains unverified.

## Verification and artifacts

129 tests passed, one opt-in live-stack test skipped. Ruff and MyPy passed.
New tests check recursive seated-row expansion, two-neighbor support, rejection
of evidence split across different queues, invalid settings, and isolated
per-call overrides. A fresh production run on all 60 images reproduces the
bounded-rule experiment's counts. No deployment was run.

Aggregates: `docs/evaluation/2026-10-21-bounded-rescue.json`,
`docs/evaluation/2026-10-21-frozen-pipeline.json`, and
`docs/evaluation/2026-10-21-yolo26m.json`.
Local per-image predictions: `data/outputs/bounded-rescue-evaluation/`.
Raw photos, crop caches, and downloaded detector weights are not committed.
