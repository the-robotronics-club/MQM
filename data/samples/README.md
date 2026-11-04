# Mess Quality Monitoring (MQM) - Benchmark Samples & Model Outputs

This directory contains representative raw real-world mess hall photos selected from the Google Drive benchmark dataset along with their corresponding model detection and posture classification overlays.

---

## Benchmark Image Pairs

### 1. Primary Diagnostic Case: Image 419 (Queue + Seated Diners)
* **Raw Image:** [`benchmark_raw/img_419_queue_and_seated.jpg`](benchmark_raw/img_419_queue_and_seated.jpg)
* **Model Overlay:** [`model_outputs/img_419_queue_and_seated_model_overlay.png`](model_outputs/img_419_queue_and_seated_model_overlay.png)
* **Ground Truth:** 39 Headcount | 25 Queued | 13 Sitting | 1 Other Standing
* **Model Prediction (YOLO11m/Bounded):** 38 Headcount | 23 Queued | 15 Sitting | 0 Other Standing
* **Notes:** Primary diagnostic reference image with human-verified individual ground-truth counts.

### 2. High-Density Rush Hour: Perfect Match (27 People)
* **Raw Image:** [`benchmark_raw/img_phone_perfect_match_27_people.jpg`](benchmark_raw/img_phone_perfect_match_27_people.jpg)
* **Model Overlay:** [`model_outputs/img_phone_perfect_match_27_people_model_overlay.png`](model_outputs/img_phone_perfect_match_27_people_model_overlay.png)
* **Ground Truth:** 27 Headcount | 19 Standing/Queue | 8 Sitting
* **Model Prediction:** 27 Headcount | 19 Standing/Queue | 8 Sitting
* **Notes:** Zero error across all three categories (Headcount error: 0, Sitting error: 0, Standing error: 0).

### 3. High Seated Density Dining Hall (28 People)
* **Raw Image:** [`benchmark_raw/img_seated_hall_28_people.jpg`](benchmark_raw/img_seated_hall_28_people.jpg)
* **Model Overlay:** [`model_outputs/img_seated_hall_28_people_model_overlay.png`](model_outputs/img_seated_hall_28_people_model_overlay.png)
* **Ground Truth:** 27 Headcount | 26 Sitting | 1 Standing
* **Model Prediction:** 28 Headcount | 27 Sitting | 1 Standing
* **Notes:** Tests seated posture classifier under table occlusion and close dining proximity.

### 4. Rush Hour Counter Line & Tables
* **Raw Image:** [`benchmark_raw/img_rush_hour_queue_and_dining.jpg`](benchmark_raw/img_rush_hour_queue_and_dining.jpg)
* **Model Overlay:** [`model_outputs/img_rush_hour_queue_and_dining_model_overlay.png`](model_outputs/img_rush_hour_queue_and_dining_model_overlay.png)
* **Ground Truth:** 51 Headcount | 19 Sitting | 32 Standing
* **Model Prediction:** 40 Headcount | 19 Sitting | 21 Standing (18 Queued, 3 Other Standing)
* **Notes:** Exact match on sitting count (19); under-detects distant queue tail at 1280px resolution.

### 5. Serving Counter Line Formation
* **Raw Image:** [`benchmark_raw/img_counter_queue_alignment.jpg`](benchmark_raw/img_counter_queue_alignment.jpg)
* **Model Overlay:** [`model_outputs/img_counter_queue_alignment_model_overlay.png`](model_outputs/img_counter_queue_alignment_model_overlay.png)
* **Ground Truth:** 28 Headcount | 0 Sitting | 28 Standing
* **Model Prediction:** 30 Headcount | 1 Sitting | 29 Standing (29 Queued)
* **Notes:** Evaluates geometric queue path grouping along the food counter.

### 6. Small Group Entrance Queue: Perfect Match
* **Raw Image:** [`benchmark_raw/img_small_queue_exact_match.jpeg`](benchmark_raw/img_small_queue_exact_match.jpeg)
* **Model Overlay:** [`model_outputs/img_small_queue_exact_match_model_overlay.png`](model_outputs/img_small_queue_exact_match_model_overlay.png)
* **Ground Truth:** 4 Headcount | 0 Sitting | 4 Standing (4 Queue)
* **Model Prediction:** 4 Headcount | 0 Sitting | 4 Standing (4 Queue)
* **Notes:** Clean small group entrance verification.
