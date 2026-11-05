# Comprehensive Model Evaluation & Benchmarking Report
**Project:** Mess Congestion Live Monitoring & Queue Quality Analysis (MQM)  
**Date:** November 2026  
**Status:** Submission-Ready Benchmark Synthesis

---

## 1. Executive Summary

This report aggregates all measured computer vision models, input resolution sweeps, and inference pipeline configurations tested for the **Mess Queue and Congestion Monitoring (MQM)** system. The objective is real-time, low-compute estimation of mess dining hall occupancy, including **total headcount**, **queue length**, **seat occupancy (sitting count)**, and **standing patrons** from camera snapshots under a 10-second Raspberry Pi capture budget.

### Key Results Snapshot
* **Architecture Dominance:** Input resolution dominates model architecture. Upscaling input from 640px to 2560px on YOLOv8n yielded a **5.4× increase in person detections (8 → 43)** on dense hall imagery, vastly exceeding gains from switching to heavier architectures at 640px.
* **Production Deployment Choice:** **YOLOv8n (1280px, Adaptive Crowd Guard + Bounded Queue Rescue)** remains the promoted production model. It delivers a 28% reduction in headcount error over fixed-cutoff baselines with an average CPU inference latency of **~0.65s**, safely within the 10-second capture cadence.
* **Accuracy Tradeoffs on 60 Real Mess Benchmark Images:**
  * **Lowest Headcount & Sitting Error:** **Sliced YOLO11m** achieved Headcount MAE **4.80** and Sitting MAE **2.92**, but increased standing error (MAE 4.27) and quad-folded CPU latency to **~5.56s**.
  * **Highest All-Count Exact Matches:** **YOLO11m Distance Pipeline** achieved **8 / 60** all-count matches and 11 headcount matches (MAE: Headcount 5.65, Sitting 4.82, Standing 4.55).
  * **Best Balanced Exploration Candidate:** **YOLO11m Full-Frame Adaptive** (Headcount MAE **5.42**, Sitting MAE **3.88**, Standing MAE **3.72**, 16 scenes within $\pm 1$ across all counts).
  * **Obsolete Approaches:** RT-DETR-L degraded to complete failure at high resolutions due to fixed positional embeddings; SAHI slicing was outperformed by native upscaling in both accuracy and latency.

---

## 2. Phase 1: Architecture & Resolution Sweep (Dense Hall Stock Photo)

Initial architectural exploration evaluated person detector backbones across input resolutions on `data/samples/mess_hall_dense.jpg` ($736 \times 490$ source, estimated 80–120 patrons) on local CPU (`conf=0.25`, `iou=0.5`, steady-state median of three timed runs).

### Detections by Model & Input Resolution

| Model | 640px | 960px | 1280px | 1600px | 1920px | 2560px | 3200px | 4096px | Peak Detection |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | 8 | 19 | 27 | 34 | 41 | **43** | 40 | 30 | 43 @ 2560px |
| **YOLOv8s** | 16 | 22 | 24 | 34 | 36 | **49** | 42 | 31 | 49 @ 2560px |
| **YOLO11x** | 5 | 12 | 31 | 43 | 55 | **72** | 71 | 59 | 72 @ 2560px |
| **RT-DETR-L** | **27** | 29 | 29 | 16 | 5 | **0** | — | — | 29 @ 960/1280px |

### Inference Latency (Seconds per Frame on CPU)

| Model | 640px | 1280px | 1920px | 2560px | 4096px | Detections / Sec (at 2560px) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | 0.06s | 0.22s | 0.57s | 1.12s | 2.78s | **38.4** det/sec |
| **YOLOv8s** | 0.13s | 0.52s | 1.22s | 2.66s | 6.66s | **18.4** det/sec |
| **YOLO11x** | 0.84s | 3.38s | 7.37s | 14.55s | 37.23s | **4.9** det/sec |
| **RT-DETR-L** | 0.79s | 3.22s | 7.74s | 11.19s | — | — |

### Key Findings & Disqualifications
1. **Resolution Dominance:** Upscaling from 640 to 2560 gave a 5.4× gain for YOLOv8n. Tuning confidence and resolution had a far higher impact than swapping architectures.
2. **Saturation at 2560px:** All YOLO models peaked at 2560px and degraded above 3200px due to interpolation blur exceeding detector anchor receptive fields.
3. **RT-DETR Failure:** RT-DETR-L was strongest at native 640px (27 detections vs YOLOv8n's 8), but its fixed transformer positional embeddings collapsed at high resolution (dropping to 0 detections at 2560px).
4. **SAHI Tiling Obsolete:** SAHI with 320px slices produced 30 detections at 1.24s. Direct YOLOv8n upscaling to 2560px produced 43 detections in 1.12s. Native upscaling beat SAHI on both speed and accuracy.
5. **YOLO11x Compute Infeasibility:** YOLO11x achieved 72 detections but took **14.55s per frame** on CPU, violating the 10s capture cadence.

---

## 3. Phase 2: Attribute Classifiers (MobileNetV2)

Fine-grained posture and queue separation uses lightweight MobileNetV2 classifiers operating on cropped person bounding boxes:
* **Seated Classifier (`seated_classifier.pt`):** Binary classification (Class 0: Seated, Class 1: Standing).
  * Checkpoint Hash: `03edbbb715d08b6a8d847e4c9c46c51bb6536b9fa8b4618095b69fc15ac85ceb`
* **Queue Classifier (`queue_classifier.pt`):** Binary classification (Class 0: Non-Queue, Class 1: Queue).
  * Checkpoint Hash: `e2ee02dd66f367e00b333941a9e6397474bf097360482a8382ef959c48d4e5cd`
* **Inference Pipeline:** Cropped RGB bounding boxes are resized to $128 \times 128$ via PIL bilinear interpolation with standard ImageNet normalization.

---

## 4. Phase 3: Comprehensive Benchmark on 60 Real Mess Images

The primary diagnostic evaluation consists of 60 real mess hall photos with human annotations (50 exact count annotations, 10 interval-labeled annotations, split into 35 development, 14 validation, and 11 flagged diagnostic scenes).

### Master Evaluation Table (All 60 Images)

| Model & Strategy | Confidence & Cutoff | Headcount MAE | Sitting MAE | Standing MAE | Exact All Matches (/60) | Exact HC Matches (/60) | Mean Latency (s) | Promotion Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLOv8n Baseline** | conf 0.25, iou 0.5 | 8.70 | 5.97 | 4.37 | 4 / 60 | 6 / 60 | 0.65s | Legacy baseline |
| **YOLOv8n Adaptive Crowd** | conf 0.25 / 0.15 ($\ge$25) | 5.83 | 4.59 | 3.44 | 5 / 59 | 7 / 59 | 0.65s | 28% error reduction |
| **YOLOv8n Bounded Rescue** | conf 0.25 / 0.15, 2 seeds | **5.80** | **4.55** | **3.50** | **4 / 60** | **7 / 60** | **0.65s** | **PROMOTED (Production)** |
| **YOLOv8n + Padding Trim** | black crop guard, conf 0.25 | 5.05 | 4.85 | 4.08 | 3 / 60 | 8 / 60 | 0.75s | Optional flag |
| **YOLOv8n Overlap Crops** | conf 0.20, seated 0.40 | 6.50 | 4.63 | 3.65 | 6 / 60 | 9 / 60 | 1.20s | Exploratory |
| **YOLOv8n Large Input** | 1600px, conf 0.25, NMS 0.3 | 5.70 | 5.05 | 3.20 | 5 / 60 | 9 / 60 | 1.52s | Higher sitting error |
| **YOLOv8s Overlap** | 1280px, conf 0.25, seated 0.7 | 5.93 | 5.42 | 3.73 | 7 / 60 | 9 / 60 | 1.46s | High sitting error |
| **YOLO11m Raw Posture** | conf 0.15, seated 0.50 | 5.65 | 4.18 | 3.72 | 7 / 60 | 11 / 60 | 3.24s | Experimental |
| **YOLO11m Distance Queue** | conf 0.15, padding trim | 5.65 | 4.82 | 4.55 | **8 / 60** | **11 / 60** | 3.24s | Highest all-match score |
| **YOLO11m Full Frame** | conf 0.25 / 0.15 adaptive | 5.42 | 3.88 | 3.72 | 7 / 60 | 12 / 60 | 1.47s | Best overall balance |
| **YOLO11m Sliced (Tiled)** | 4x 60% tiles + full, 1280px | **4.80** | **2.92** | 4.27 | 4 / 60 | 5 / 60 | 5.56s | Lowest HC & Sit MAE; high standing error |
| **YOLO26m (End-to-End)** | conf 0.25 / 0.15 adaptive | 6.03 | 4.90 | 3.88 | 4 / 60 | 6 / 60 | 1.80s | Moderate accuracy |
| **Consensus Ensemble** | YOLOv8n + YOLO11m merge | 6.72 | 4.63 | 3.58 | 3 / 60 | 4 / 60 | 4.50s | Over-conservative |
| **Posture Flip-Avg (TTA)** | horizontal flip avg, no rescue | 5.80 | 4.47 | 3.43 | 5 / 60 | 7 / 60 | 0.85s | Dev leader; minor gain |

---

### Unflagged Cohort Breakdown (50 Exact Labels, 35 Development, 14 Validation)

| Model & Pipeline Variant | Exact 50 HC MAE | Exact 50 Sit MAE | Exact 50 Stand MAE | Dev 35 HC MAE | Dev 35 Sit MAE | Val 14 HC MAE | Val 14 Sit MAE | Val 14 Matches |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n Baseline** | 5.82 | 3.76 | 3.82 | 6.91 | 4.51 | 7.50 | 7.79 | 0 / 14 |
| **YOLOv8n Bounded Rescue (Prod)** | **3.78** | **2.86** | **3.33** | **3.83** | **2.46** | **3.64** | **3.86** | 0 / 14 |
| **YOLOv8n Overlap** | 4.26 | 2.84 | 3.38 | 4.46 | 2.46 | 4.07 | 4.00 | 0 / 14 |
| **YOLOv8s Overlap** | 3.98 | 3.66 | 3.72 | 3.97 | 3.03 | 4.29 | 5.50 | 1 / 14 |
| **YOLO11m Distance Pipeline** | 4.31 | 3.63 | 4.51 | 4.71 | 3.26 | 3.29 | 4.57 | 0 / 14 |
| **YOLO11m Full Frame Adaptive** | 3.90 | 2.46 | 3.60 | 4.49 | 2.23 | 2.71 | 3.21 | 0 / 14 |
| **YOLO11m Sliced** | 4.26 | 2.06 | 4.16 | 5.03 | 1.83 | 2.64 | 2.79 | 0 / 14 |
| **YOLO26m** | 4.18 | 3.18 | 3.78 | 4.40 | 2.71 | 3.57 | 4.07 | 0 / 14 |

---

## 5. Phase 4: Extreme Benchmark Case: Image 419

Image `IMG_20261005_200419.jpg` is the primary high-density reference image with fully verified individual ground truth counts:
* **Ground Truth:** **39 Total Headcount**, **25 In Queue**, **13 Sitting**, **1 Standing Outside Queue**.

### Model Predictions on Image 419

| Pipeline / Model | Total Detected | Queue Count | Sitting Count | Other Standing | Queue Error | Total Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ground Truth Reference** | **39** | **25** | **13** | **1** | — | — |
| **Production YOLOv8n Bounded** | 35 | 20 | 15 | 0 | -5 | -4 |
| **YOLO11m Distance (conf 0.15)** | 38 | 23 | 15 | 0 | -2 | -1 |
| **YOLO11m Raw Posture** | 38 | — | 17 | 21 | — | -1 |
| **YOLO11m Sliced Tiling** | 38 | **25** | 13 | 0 | **0** | -1 |
| **YOLO26m** | 36 | 23 | 13 | 0 | -2 | -3 |

*Observation:* While Sliced YOLO11m achieves perfect queue segmentation on Image 419, its 5.56s CPU latency and higher standing error on wider halls prevent universal promotion. YOLOv8n bounded rescue maintains balanced performance without runtime instability.

---

## 6. Structural Innovation: Bounded Queue Rescue

To resolve seated-vs-queue classification errors in dense queues where patrons lean or sit partially on counters:
1. **Unconstrained Distance Rescue (Former):** If a person was classified as seated with $P(\text{seated}) \le 0.9$ and within 1.6m of a queue line, they were reclassified as queued. This led to **runaway cascading**, erroneously pulling seated diners at adjacent tables into the queue.
2. **Bounded Rescue Rule (Promoted):** Requires corroboration from **at least two original standing queue members** before reclassifying an uncertain seated prediction. Recursive neighbor reclassification is strictly barred.
3. **Outcome:** Reduced spurious reassignments from 117 to 62, reduced Sitting MAE from 5.17 to 4.55, and Standing MAE from 3.82 to 3.50 across all 60 images without adding any classifier inference overhead.

---

## 7. Submission Checklist & File References

* Master CSV Export: `docs/evaluation/MODEL_METRICS_SUBMISSION.csv`
* Raw Benchmark JSONs:
  * `docs/evaluation/2026-10-21-accuracy-trials.json` (1,080 exploratory search trials)
  * `docs/evaluation/2026-10-21-frozen-pipeline.json` (24 fixed pipeline variants)
  * `docs/evaluation/2026-10-21-sliced-detection.json` (Sliced vs Full-frame evaluation)
  * `docs/evaluation/2026-10-21-bounded-rescue.json` (Bounded rescue verification)
  * `docs/evaluation/2026-10-21-yolo11m-distance.json` (YOLO11m complete pipeline)
  * `docs/evaluation/2026-10-21-yolo26m.json` (YOLO26m evaluation)
* Ground Truth Annotation Source: `dataset_labels.json`

