# Accuracy experiments and benchmark policy

The target of 30 correct images out of 60 has **not** been reached. No model
weights were trained or changed, and no experimental policy was promoted into
production. The supplied 60 images remain a diagnostic benchmark, not a new
training set.

## Results

For this comparison, an image matches when total, sitting, and standing all
match its supplied labels. Ten images have label ranges, so a value inside a
range counts as a match; 50 images have exact labels. These are count matches,
not per-person detection accuracy. Queue labels are too sparse to score queue
accuracy in this experiment.

| Pipeline or exploratory policy | All three counts match | Mean total error | Mean sitting error | Mean standing error |
| --- | ---: | ---: | ---: | ---: |
| Production YOLOv8n + distance reassignment | 4/60 | 5.80 | 5.17 | 3.82 |
| Production detector + raw posture | 5/60 | 5.80 | 4.62 | 3.55 |
| Overlapping crops + raw posture, descriptive best-match policy | 5/60 | 4.15 | 4.00 | 3.07 |
| YOLO11m + raw posture, descriptive best-match policy | 7/60 | 5.65 | 4.18 | 3.72 |

The best tested raw-posture policy gives 7 matches, all on exact-label images.
The best total-only policy gives 12/60 total matches, using a different policy.
These are descriptive results after looking at this benchmark, not evidence
that the policies will perform better on new footage.

For 419, YOLO11m at confidence 0.15 gives 38 total, 17 sitting, and 21 standing
versus confirmed 39 total, 13 sitting, and 26 standing. It improves headcount
while worsening sitting error relative to the production distance output.
Its queue membership was not evaluated here.

## Complete-pipeline follow-up

The larger detector was also passed through the existing distance queue logic.
At fixed confidence 0.15 it gives 8/60 all-count matches, total error 5.65,
sitting error 4.82, and standing error 4.55. For 419 it gives 38 total, 15 sitting,
and 23 queued people. This improves that image but increases overall standing
error. The diagnostic validation group still has zero all-count matches. No
detector or threshold winner is promoted on the strength of this benchmark.
Fresh inference on all 60 images reproduced the cached complete-pipeline counts
for every image. Results are in
`docs/evaluation/2026-10-21-yolo11m-distance.json`; the new local 419 annotation is
`data/outputs/yolo11m-distance-evaluation/image-419.png`.

A generic padding-removal option was added with coordinate restoration and
conservative dark-frame guards. A fresh full-pipeline run gives total error
5.05, sitting error 4.85, standing error 4.08, and 3/60 all-count matches.
It therefore stays **off by default**, configurable through
`MODEL_TRIM_BLACK_PADDING=true` or the evaluator's `--trim-black-padding` flag.
Its aggregate results are in `docs/evaluation/2026-10-21-padding.json`.

## What was tested

Eight detection strategies, each evaluated with 135 global policies, make
1,080 count-policy comparisons. The first four reuse cached detections from
the previous study; the latter four ran new inference on all 60 images:

- YOLOv8n, 1280 input, IoU 0.5.
- YOLOv8n, 1280 input, IoU 0.7.
- YOLOv8n, 1600 input, IoU 0.7.
- YOLOv8s, 1280 input, IoU 0.7.
- YOLOv8n after trimming outer near-black screenshot padding.
- The same trimming with detector test-time augmentation.
- Full frame plus four overlapping crops, merged at IoU 0.5, rejecting fragments
  cut by internal crop edges.
- [Official YOLO11m](https://docs.ultralytics.com/models/yolo11/), 1280 input,
  IoU 0.5, after padding removal.

Policies compare eight detector confidence cutoffs, five seated-probability
cutoffs, optional crowd adaptation, and additional NMS at 0.3/0.4. They apply
the same rules to every image; there are no filename-based count corrections.
Image count labels are used for scoring, not passed into detection or posture
inference. New experiments report raw posture counts without distance rescue.

## Preventing overfitting

Repeatedly trying settings against these images can overfit their results even
without training model weights. All 60 have now been inspected, so none can be
described as a pristine unseen test set. The grouped diagnostic comparison uses
35 unflagged development images and 14 unflagged validation images; 11 flagged
images contribute only to overall results. The YOLO11m development leader gets
zero all-count matches on those 14 validation images. Do not promote it because
its overall benchmark count is higher.

Do not train on these 60 to force the 30/60 result. Keep the current weights
frozen while improving general preprocessing and pipeline settings. For the next
pipeline change:

1. Collect separate development and test capture sessions to check whether the
   pipeline changes help beyond these familiar photos. If training is later
   required, use a separate training set with reviewed person boxes and posture
   labels; do not train on these 60.
2. Keep development and final test sessions separate from any training. Keep
   neighboring video frames and crops of one source image in the same split.
3. Choose preprocessing, detector, posture, and queue settings on development
   data. Freeze them before evaluating the final test sessions.
4. Report the final test result and this 60-image regression result separately,
   including exact matches, errors, and processing time.

The original checkpoints' training-data overlap with the supplied photos is
unknown; these experiments establish only that no new training occurred.

## Reproduction and artifacts

`tools/evaluation/accuracy_trials.py` caches model outputs and produces grouped
metrics and per-image predictions. It never trains or changes the runtime
configuration. Aggregate figures and checkpoint hashes are in
`docs/evaluation/2026-10-21-accuracy-trials.json`.

```bash
python -m tools.evaluation.accuracy_trials \
  --images /path/to/photos --labels /path/to/dataset_labels.json
python -m tools.evaluation.accuracy_trials \
  --images /path/to/photos --labels /path/to/dataset_labels.json \
  --extra-weights yolo11m.pt --only-extra \
  --output data/outputs/accuracy-yolo11m
```

Install `requirements-eval.txt`; the original four strategies require the prior
detector-tuning caches. Local per-image results are in
`data/outputs/accuracy-yolo11m/comparison.md`. Raw photos, caches, annotations,
and downloaded detector weights stay local. Cached timing is indicative only;
new inference runs overlapped and do not constitute a controlled latency test.

126 tests pass; one live-stack test is skipped. Ruff and MyPy pass. Direct checks
verified padding-coordinate restoration, duplicate
suppression and posture alignment, unchanged source data, interval scoring, and
empty crops. All 60 images decoded and completed inference for each new strategy.
