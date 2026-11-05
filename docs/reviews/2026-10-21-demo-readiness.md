# Demo readiness review

The distance-based photo and phone-camera pipeline is ready for a controlled
demo. Treat the counts as estimates; the current labeled set does not support a
claim of reliable queue counting across scenes.

## What to show

Use `data/outputs/image-419-queue/distance-boxes.png` to show the preferred
notebook output and `data/outputs/image-419-final/distance-boxes.png` to show
the current shared runtime output. Both use blue for Queue, orange for Sitting,
and green for Other standing. The current runtime reports 20 queue, 15 sitting,
0 other standing, 35 total. The confirmed image count is 25 queue, 13 sitting,
1 roaming, 39 total. State both numbers during the demo.

For an example that matched all three labels exactly, use
`Screenshot from 2026-09-07 18-43-16 - REMAN DEY.png` (empty scene) or
`WhatsApp Image 2026-10-05 at 21.20.02.jpeg` (4 standing people). Avoid presenting
these as proof of performance; they are individual examples from the evaluation
set.

The referenced images and contact sheet are generated locally beside the ignored
raw photo set; only this review and aggregate figures belong in GitHub.

## Known hard cases

The contact sheet `data/outputs/final-distance-evaluation/worst-cases.png`
shows detections on the eight largest headcount misses. Several crowded scenes
have small, partly hidden people that the detector misses. Others produce many
low-confidence boxes or poor posture crops.

| Image | Predicted total | Label total | Absolute error |
| --- | ---: | ---: | ---: |
| `IMG_20261005_130554.jpg` | 43 | 78–80 | 35 |
| `Screenshot_2026-10-05-20-55-31-46_99c04817c0de5652397fc8b56c3b3817.jpg` | 12 | 44–46 | 32 |
| `Screenshot_2026-10-05-20-55-34-96_99c04817c0de5652397fc8b56c3b3817.jpg` | 12 | 43–45 | 31 |
| `IMG_20261005_200408.jpg` | 39 | 67–69 | 28 |
| `IMG_20261005_200427.jpg` | 5 | 22–30 | 17 |

The current distance pipeline's mean absolute errors across 60 images are 5.8
for total headcount, 5.17 for sitting, and 3.82 for standing (including queue).
Only one image has a confirmed queue label: 419, where the queue estimate is
20 versus 25. That is not enough evidence to report queue accuracy across
different scenes.

## Why the confidence threshold stays at 0.15 for dense scenes

On the 59-image tuning set, a fixed 0.05 detector threshold improved some of
the hardest totals: `IMG_20261005_130554.jpg` moved from 43 to 69 against 78–80,
and `IMG_20261005_200408.jpg` moved from 39 to 64 against 67–69. But across all
59 images, average error worsened:

| Detector threshold | Total error | Sitting error | Standing error |
| --- | ---: | ---: | ---: |
| 0.15 | 5.88 | 4.41 | 3.58 |
| 0.05 | 9.54 | 5.59 | 5.80 |

The 0.05 setting creates too many false detections on other images. Keep the
current adaptive guard and collect more labeled crowded scenes before trying a
scene-specific threshold.

## Two-day completion priorities

1. Run a short phone-camera or recorded-clip demo in the actual cafeteria view;
   confirm that people, counts, and colors refresh without stalls.
2. Record a small set of new frames with human-checked total, sitting, and queue
   counts. Include several scenes and keep captures from one session together
   when reporting results.
3. Review the missed boxes on those frames before changing confidence settings
   or training. Keep the raw photos local and add only labels and aggregate
   results to the repo.

No deployment was run. The live camera, mounted geometry, and physical phone
capture remain to be checked in the demonstration environment.
