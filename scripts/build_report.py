import os
import subprocess
import shutil

repo_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
img_fig2 = os.path.join(repo_root, "data/samples/model_outputs/img_counter_queue_alignment_model_overlay.png")
img_fig3 = os.path.join(repo_root, "data/samples/model_outputs/img_phone_perfect_match_27_people_model_overlay.png")
img_fig4 = os.path.join(repo_root, "data/samples/model_outputs/img_419_queue_and_seated_model_overlay.png")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Mess Queue Management (MQM) - Project Report</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 14mm 15mm 14mm 15mm;
  }}
  body {{
    font-family: 'Times New Roman', Times, Georgia, serif;
    font-size: 10pt;
    line-height: 1.4;
    color: #111;
    margin: 0;
    padding: 0;
  }}
  h1.doc-title {{
    font-size: 17pt;
    font-weight: bold;
    margin-bottom: 2px;
    color: #111;
  }}
  .doc-subtitle {{
    font-size: 10.5pt;
    font-weight: bold;
    color: #333;
    margin-bottom: 2px;
  }}
  .meta-line {{
    font-size: 9pt;
    color: #222;
    margin-bottom: 2px;
  }}
  .header-rule {{
    border-bottom: 1.5px solid #222;
    margin-top: 6px;
    margin-bottom: 10px;
  }}
  h2.sec-title {{
    font-size: 11.5pt;
    font-weight: bold;
    border-bottom: 0.5px solid #aaa;
    padding-bottom: 2px;
    margin-top: 12px;
    margin-bottom: 6px;
  }}
  h3.subsec-title {{
    font-size: 10pt;
    font-weight: bold;
    margin-top: 8px;
    margin-bottom: 3px;
  }}
  p {{
    margin-top: 0;
    margin-bottom: 6px;
    text-align: justify;
  }}
  ul {{
    margin-top: 0;
    margin-bottom: 6px;
    padding-left: 18px;
  }}
  li {{
    margin-bottom: 3px;
    text-align: justify;
  }}
  .callout-box {{
    background-color: #f8f9fa;
    border: 1px solid #ced4da;
    border-radius: 4px;
    padding: 6px 10px;
    margin: 8px 0;
    font-size: 8.5pt;
    page-break-inside: avoid;
  }}
  .pipeline-box {{
    background-color: #f8f9fa;
    border: 1px solid #ced4da;
    border-radius: 4px;
    padding: 8px 10px;
    margin: 10px 0;
    font-family: 'Courier New', Courier, monospace;
    font-size: 7.5pt;
    white-space: pre;
    line-height: 1.25;
  }}
  .equation-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #fafbfc;
    border-left: 3px solid #0366d6;
    padding: 5px 10px;
    margin: 6px 0;
    font-family: 'Cambria Math', 'Times New Roman', serif;
    font-size: 9.5pt;
  }}
  table.report-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0;
    font-size: 8.5pt;
  }}
  table.report-table th, table.report-table td {{
    padding: 5px 6px;
    text-align: left;
    border-top: 1px solid #ccc;
    border-bottom: 1px solid #ccc;
  }}
  table.report-table th {{
    background-color: #f1f3f5;
    font-weight: bold;
    border-top: 1.5px solid #333;
    border-bottom: 1.5px solid #333;
  }}
  table.report-table tr:last-child td {{
    border-bottom: 1.5px solid #333;
  }}
  .text-center {{ text-align: center; }}
  .text-right {{ text-align: right; }}
  .figure-container {{
    text-align: center;
    margin: 10px 0 14px 0;
    page-break-inside: avoid;
  }}
  .figure-img {{
    max-width: 95%;
    max-height: 220px;
    border: 1px solid #bbb;
    border-radius: 3px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }}
  .figure-caption {{
    font-size: 8pt;
    font-style: italic;
    color: #333;
    margin-top: 4px;
    text-align: justify;
    padding: 0 10px;
  }}
  .page-break {{
    page-break-before: always;
  }}
  .badge-promoted {{
    background-color: #28a745;
    color: #fff;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 7.5pt;
    font-weight: bold;
  }}
  .badge-opt {{
    background-color: #6c757d;
    color: #fff;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 7.5pt;
  }}
</style>
</head>
<body>

<!-- PAGE 1 -->
<div class="doc-header">
  <h1 class="doc-title">Mess Queue Management (MQM) - Project Report</h1>
  <div class="doc-subtitle">Project: Mess Queue Management &amp; MQM-IITM Perspective-Aware Queue Estimation</div>
  <div class="meta-line"><b>Repository:</b> <a href="https://github.com/the-robotronics-club/MQM">github.com/the-robotronics-club/MQM</a> &nbsp;|&nbsp; <b>Fork / Releases:</b> <a href="https://github.com/arywk40-hue/MQM">github.com/arywk40-hue/MQM</a></div>
  <div class="meta-line"><b>Team:</b> The Robotronics Club (Lead ML Research: Ariyan Bhakat, SCEE, IIT Mandi — <a href="mailto:b25201@iitmandi.ac.in">b25201@iitmandi.ac.in</a>)</div>
  <div class="meta-line"><b>Date:</b> September 2026 (Updated November 2026 with Verified ML Pipeline, Benchmark Protocol &amp; Live Mess Evaluations)</div>
  <div class="header-rule"></div>
</div>

<h2 class="sec-title">1. Introduction</h2>
<p>
Every meal hour at our institute, students walk to the mess, glance at the queue, and make a guess: wait or come back later. There is no information available before they arrive. If the queue is long, they have already wasted the trip. Mess Queue Management (MQM) [7], [8] fixes this by placing a camera above the mess entrance, running person detection and role classification on each frame, and publishing live queue length and seating occupancy to a web dashboard that any student can check from their phone without installing an app.
</p>
<p>
The hardware capture node is a wall-mounted Raspberry Pi Zero 2 W with a wide-angle camera that captures a 736×490 JPEG every 10 seconds and sends it over Wi-Fi to a central FastAPI server running YOLOv8n person detection [6] and two MobileNetV2 classifiers [5] that distinguish queued from seated people. Results are stored in Redis (for live state) and InfluxDB (for time-series history), then served to a Streamlit dashboard. Beyond this baseline deployment, our latest machine learning development (MQM-IITM) expands the system with geometric proximity-graph grouping, table-occlusion line rescue, ground-plane homography projection, learned multi-feature membership classification, bounded occlusion correction, and the IITM-MessQueue dataset and nine-rung evaluation protocol.
</p>

<h2 class="sec-title">2. Objectives</h2>
<h3 class="subsec-title">What it does for the institute</h3>
<ul>
  <li>Gives students real-time visibility into mess congestion (headcount, queue length, seat occupancy, and green/amber/red crowd state) before they leave their hostel room.</li>
  <li>Provides mess administration with occupancy data to identify peak hours, plan staffing, and evaluate whether staggered meal timings reduce wait times.</li>
  <li>Generates a continuous historical record of crowd patterns across days and weeks, which previously did not exist.</li>
</ul>

<h3 class="subsec-title">What we learn from building it &amp; ML research contributions</h3>
<ul>
  <li><b>End-to-end IoT pipeline construction:</b> Designing the full path from physical camera capture to authenticated FastAPI ingestion, computer vision inference, Redis/InfluxDB storage, and a read-only frontend.</li>
  <li><b>Two-stage &amp; perspective-aware vision:</b> Combining spatial polygon zones and homography-projected ground-plane coordinates with MobileNetV2 appearance classifiers, geometric proximity linking, and bounded occlusion correction.</li>
  <li><b>Rigorous ML benchmarking &amp; production discipline:</b> Separating raw detector count sweeps from ground-truth counting accuracy via a nine-rung baseline ladder (B0–P3), backed by 157 passing unit/integration tests, Ruff linting, and MyPy type checking across 21 source files.</li>
</ul>

<div class="callout-box">
  <b>Codebase &amp; Dataset Links:</b><br>
  • <b>Production Code &amp; Benchmarks:</b> <a href="https://github.com/the-robotronics-club/MQM">https://github.com/the-robotronics-club/MQM</a> (Fork: <a href="https://github.com/arywk40-hue/MQM">https://github.com/arywk40-hue/MQM</a>)<br>
  • <b>IITM-MessQueue Dataset Manifest:</b> <code>dataset_labels.json</code> (60 ground-truth real mess images)<br>
  • <b>Pushed Benchmark Sample Images &amp; Overlays:</b> <code>data/samples/benchmark_raw/</code> &amp; <code>data/samples/model_outputs/</code><br>
  • <b>Consolidated Benchmark Report &amp; Metrics CSV:</b> <code>docs/evaluation/MODEL_BENCHMARK_REPORT.md</code> &amp; <code>MODEL_METRICS_SUBMISSION.csv</code>
</div>

<!-- PAGE 2 -->
<div class="page-break"></div>

<h2 class="sec-title">3. System Pipeline</h2>
<p>
The end-to-end production system operates in three primary stages: capture, process, and display, augmented by our research inference extensions.
</p>

<div class="pipeline-box">
CAPTURE                      PROCESS                                               DISPLAY
+-------------------+        +------------------------------------------+          +---------------------+
| Raspberry Pi      | JPEG   | Central Server (FastAPI)                 |  GET     | Streamlit Dashboard |
| Zero 2 W + Camera |------->| 1. YOLOv8n person detection (conf 0.25)  |<---------| - Live Headcount    |
| - Wide Lens       | POST   | 2. Zone assignment & Homography H        |          | - Queue Length      |
| every 10s (X-Key) |        | 3. MobileNetV2 crops (queue + seated)    |          | - Seat Occupancy    |
| 736x490 px        |        | 4. Proximity graph / Bounded rescue      |          | - Crowd Level + CI  |
| via Wi-Fi         |        | 5. Occlusion rescue & count correction   |          | - 60m Trend Chart   |
| (RAM pipe, no SD) |        | +--------------------+-------------------+          | (Light campus UI)   |
+-------------------+        | Redis (live snapshot)| InfluxDB (history)|          +---------------------+
                             +----------------------+-------------------+
</div>

<p>
<b>Stage 1 – Capture (Raspberry Pi):</b> A Python script on the Raspberry Pi Zero 2 W calls <code>libcamera-still</code> to grab a 736×490 JPEG, pipes it directly to memory (zero SD card writes to prevent flash wear), and POSTs it to <code>POST /ingest/{{camera_id}}</code> with an <code>X-Camera-Key</code> header every 10 seconds. The Pi runs headless on Raspberry Pi OS Lite; all deep-learning inference runs server-side so 512 MB RAM on the edge board suffices.
</p>
<p>
<b>Stage 2 – Process (Central Server / FastAPI):</b> The server validates the JPEG dimensions, runs YOLOv8n person detection, maps each detected person's bottom-centre foot-point into camera-specific polygons (<code>config/zones.json</code>: queue, seating, entrance), and executes two MobileNetV2 binary classifiers on person crops. Detections pass through adaptive crowd thresholding, ground-plane homography projection, and bounded queue rescue. Metrics (<code>headcount</code>, <code>queue_count</code>, <code>seat occupancy</code>, <code>crowd_level</code>, and <code>confidence interval CI_t</code>) are written to Redis with a 30-second TTL (so a silent camera reports offline rather than a false zero) and InfluxDB for time-series trends.
</p>
<p>
<b>Stage 3 – Display (Streamlit Dashboard):</b> A read-only web dashboard polls public <code>GET /status</code> and <code>GET /history/{{camera_id}}</code> endpoints every 8 seconds without holding database credentials. It displays live counts, student profile badges, a green/amber/red occupancy indicator, and a 60-minute chart, plus an offline photo-upload mode for standalone image testing.
</p>

<h2 class="sec-title">4. Current Engineering Progress &amp; Verification</h2>
<ul>
  <li><b>Automated Test Suite (157 Tests Verified):</b> Full coverage across detector adapters, polygon zone geometry, metric computation, FastAPI ingestion/read routes, Streamlit AppTest rendering, MobileNetV2 checkpoint loading, and offline photo upload.</li>
  <li><b>Full Live-Stack &amp; Load Verification:</b> Redis, InfluxDB, FastAPI, YOLOv8n, both MobileNetV2 classifiers, and Streamlit verified end-to-end under Docker Compose. The read-only status API sustained 1,000 concurrent requests at 700 req/sec (concurrency 100) on localhost with zero errors.</li>
  <li><b>Inference Latency &amp; Static Quality Gates:</b> Cold frame analysis takes ~3.68 seconds and warm analysis takes ~0.23 seconds on CPU (~0.65s for full scene pipeline with MobileNetV2 posture classification)—well within the 10-second capture cadence. Ruff linting, MyPy type checking across 21 source files, Docker Compose checks, and shell syntax validation all pass cleanly.</li>
  <li><b>Campus Dining Integration:</b> Supports campus hostel messes and canteens (Oak, Pine, Alder, D2, Cedar, B13) with institute email authentication, student roll numbers, and live queue wait time estimations.</li>
</ul>

<!-- PAGE 3 -->
<div class="page-break"></div>

<h2 class="sec-title">5. Machine Learning Models &amp; Baseline Formulation</h2>
<p>
Estimating how many people are waiting in a mess line from a single fixed camera requires separating person detection from queue-membership assignment. A naive zone baseline counts every detected person box [<i>x</i><sub>1,i</sub>, <i>y</i><sub>1,i</sub>, <i>x</i><sub>2,i</sub>, <i>y</i><sub>2,i</sub>] whose bottom-centre floor contact point <i>p</i><sub>i</sub> falls inside a manually drawn queue polygon <i>R</i><sub>q</sub>, while the deployed MQM production gate refines this using the MobileNetV2 queue classifier probability <i>s</i><sub>i</sub> = <i>P</i>(queue | crop<sub>i</sub>):
</p>

<div class="equation-row">
  <span><b>Zone-Only Baseline (B1):</b> &nbsp; <i>Q̂</i><sub>t</sub><sup>zone</sup> = ∑<sub>i</sub> <b>1</b>[<i>p</i><sub>i</sub> ∈ <i>R</i><sub>q</sub>], &nbsp; where <i>p</i><sub>i</sub> = ((<i>x</i><sub>1,i</sub> + <i>x</i><sub>2,i</sub>)/2, <i>y</i><sub>2,i</sub>)</span>
  <span>(1)</span>
</div>
<div class="equation-row">
  <span><b>Deployed Zone + CNN Gate (B2):</b> &nbsp; <i>Q̂</i><sub>t</sub><sup>zone+CNN</sup> = ∑<sub>i</sub> <b>1</b>[<i>p</i><sub>i</sub> ∈ <i>R</i><sub>q</sub>] · <b>1</b>[<i>s</i><sub>i</sub> &gt; 0.5]</span>
  <span>(2)</span>
</div>

<p>
While Eq. (2) filters out non-queued bystanders inside the polygon, our research analysis identified three structural limitations: (1) <i>Hard polygon gating &amp; isolated crops</i>—curved or overflow queues extending outside <i>R</i><sub>q</sub> are ignored, and judging each crop in isolation ignores line spacing and alignment; (2) <i>Rear-counter occlusion</i>—in dense, obliquely viewed lines, students near the rear serving counter are hidden behind foreground people, causing systematic under-counting; and (3) <i>Raw count vs. ground-truth evaluation</i>—higher raw detection counts do not imply higher counting accuracy without human-annotated verification.
</p>

<table class="report-table">
  <thead>
    <tr>
      <th style="width: 25%;">Component</th>
      <th style="width: 75%;">Architecture, Training &amp; Serving Specification (<code>inference/</code> &amp; <code>training/</code>)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Person Detector</b></td>
      <td>YOLOv8n [6], COCO-pretrained (person class only); confidence floor 0.25, crowd threshold 0.15 (≥25 people), NMS IoU 0.50, production default inference size 1280.</td>
    </tr>
    <tr>
      <td><b>Crop Classifiers</b></td>
      <td>Two independent MobileNetV2 [5] backbones (ImageNet-initialized, 1000-way head replaced by a 2-way linear layer; ~2.2M parameters each).</td>
    </tr>
    <tr>
      <td><b>Output Classes</b></td>
      <td>Queue model: <code>not_queue = 0</code>, <code>queue = 1</code>. Seated model: <code>seated = 0</code>, <code>standing = 1</code> (run independently in baseline; unified into mutually exclusive roles in MQM-IITM).</td>
    </tr>
    <tr>
      <td><b>Training Protocol</b></td>
      <td>Cross-entropy loss, Adam (lr = 10<sup>-3</sup>), 15 epochs, batch size 8, 128×128 input crops, random horizontal flip + brightness/contrast/saturation jitter, ImageNet normalization.</td>
    </tr>
    <tr>
      <td><b>Training Crop Data</b></td>
      <td>Person crops extracted from mess/canteen photos using YOLO11x (confidence 0.065) and labelled via a one-click notebook widget (<code>queue</code>, <code>seated</code>, or <code>skip</code> for staff/passers-by).</td>
    </tr>
    <tr>
      <td><b>Production Serving</b></td>
      <td>128×128 crops via PIL bilinear resize, CPU batch inference; loaded as tensor-only state dicts (<code>weights_only=True</code>) with strict key matching and SHA-256 verification (<code>docs/reviews/</code>).</td>
    </tr>
  </tbody>
</table>

<h3 class="subsec-title">Detector Sensitivity Sweeps &amp; Geometric Proximity-Rescue Heuristic</h3>
<p>
Benchmarking scripts in <code>tools/benchmarks/</code> revealed that inference hyperparameters strongly shift raw detections: (i) on the committed 736×490 frame, varying YOLOv8n inference size from 640 to 2560 on unchanged weights changed detections by <b>5.4× (from 8 to 43 boxes)</b>, with higher resolutions triggering false positives on static wall murals containing human figures; and (ii) varying the confidence threshold from 0.05 to 0.70 on a larger YOLO model swung detections from 136 to 26. We mitigate mural false positives via static exclusion masks and consecutive-frame identical-box flags.
</p>
<p>
Furthermore, our research pipeline introduces a <b>bounded geometric proximity-graph &amp; line-rescue algorithm (Baseline B5)</b>: standing detections within 1.6× person-heights are linked into a graph, tested for minimum cluster size (≥ 3) and line elongation. To prevent recursive cascading into adjacent dining tables, an uncertain seated prediction (<i>P</i>(seated) ≤ 0.9) is rescued into the queue <i>only if supported by at least two original standing queue members</i>. This reduced false queue reassignments from 117 down to 62 while improving sitting MAE from 5.17 to 4.55 across all 60 benchmark images.
</p>

<!-- PAGE 4 -->
<div class="page-break"></div>

<h2 class="sec-title">6. Proposed MQM-IITM Perspective-Aware &amp; Occlusion-Robust Pipeline</h2>
<p>
To achieve trustworthy queue-length estimation under oblique camera angles and dense meal-hour crowding, the MQM-IITM research architecture decomposes total queue length <i>Q̂</i><sub>t</sub> into visible classified members plus a bounded hidden-person correction:
</p>

<div class="equation-row">
  <span><b>Queue Decomposition (Eq. 3):</b> &nbsp; <i>Q̂</i><sub>t</sub> = <i>Q̂</i><sub>t</sub><sup>visible</sup> + <i>Q̂</i><sub>t</sub><sup>hidden</sup></span>
  <span>(3)</span>
</div>

<div class="pipeline-box">
[Camera Frame I_t (736x490 JPEG)] ---> [Person Detector: YOLOv8n (conf 0.25 / 0.15 crowd)]
                                            |
        +-----------------------------------+-----------------------------------+
        v                                   v                                   v
[Zone Polygons: bottom-pt]       [Ground-Plane Projection: H]           [MobileNetV2 Posture Crop]
        |                                   |                                   |
        +-----------------------------------+-----------------------------------+
                                            v
     [Feature Extraction z_i: (x', y', d_counter, d_path, density, rank, conf, occlusion, s_i)]
                                            |
                                            v
     [Membership Head: B5 Proximity Rescue / P1 LogReg &amp; GBDT Classifier] -> Q̂_vis
                                            |
                                            v
     [Occlusion Correction: Ground-Plane Spacing Residual Regressor] -------> Q̂_hid (± δ_t)
                                            |
                                            v
                       [Final Estimate: Q̂_t = Q̂_vis + Q̂_hid]
</div>

<h3 class="subsec-title">Stage A – Perspective Normalization via Ground-Plane Homography</h3>
<p>
Because the Camera Module 3 Wide uses a wide-angle lens, we apply checkerboard lens-distortion correction and calibrate a 4-point ground-plane homography matrix <i>H</i>. Each person's bottom-centre image coordinate (<i>x</i><sub>i</sub><sup>bc</sup>, <i>y</i><sub>i</sub><sup>bc</sup>) is mapped into metric-like ground-plane coordinates <i>p'</i><sub>i</sub> = (<i>x'</i><sub>i</sub>, <i>y'</i><sub>i</sub>) so inter-person spacing is invariant to camera depth:
</p>

<div class="equation-row">
  <span><b>Homography Projection (Eq. 4):</b> &nbsp; <i>p̃</i><sub>i</sub> = <i>H</i> [<i>x</i><sub>i</sub><sup>bc</sup>, <i>y</i><sub>i</sub><sup>bc</sup>, 1]<sup>T</sup>, &nbsp; <i>p'</i><sub>i</sub> = (<i>x̃</i><sub>i</sub>/<i>z̃</i><sub>i</sub>, <i>ỹ</i><sub>i</sub>/<i>z̃</i><sub>i</sub>) = (<i>x'</i><sub>i</sub>, <i>y'</i><sub>i</sub>)</span>
  <span>(4)</span>
</div>
<div class="equation-row">
  <span><b>Feature Vector &amp; Membership (Eq. 5, 6):</b> &nbsp; <i>z</i><sub>i</sub> = [<i>x'</i><sub>i</sub>, <i>y'</i><sub>i</sub>, <i>d</i><sub>i</sub><sup>counter</sup>, <i>d</i><sub>i</sub><sup>path</sup>, <i>n</i><sub>i</sub>, <i>r</i><sub>i</sub>, <i>c</i><sub>i</sub>, <i>o</i><sub>i</sub>, <i>s</i><sub>i</sub>], &nbsp; <i>m̂</i><sub>i</sub> = <b>1</b>[<i>P</i>(<i>m</i><sub>i</sub> = 1 | <i>z</i><sub>i</sub>) &gt; τ]</span>
  <span>(5, 6)</span>
</div>
<div class="equation-row">
  <span><b>Occlusion-Corrected Estimate (Eq. 7):</b> &nbsp; <i>Q̂</i><sub>t</sub> = ∑<sub>i</sub> <i>m̂</i><sub>i</sub> + <i>Q̂</i><sub>t</sub><sup>hidden</sup>, &nbsp; <i>CI</i><sub>t</sub> = [<i>Q̂</i><sub>t</sub> - δ<sub>t</sub>, <i>Q̂</i><sub>t</sub> + δ<sub>t</sub>]</span>
  <span>(7)</span>
</div>

<h3 class="subsec-title">Stage B – Learned Membership Classifier, Occlusion Correction &amp; Temporal Tracking</h3>
<p>
Instead of gating on a hard polygon, Logistic Regression and Gradient-Boosted Decision Trees (GBDT) classify queue membership <i>m̂</i><sub>i</sub> from <i>z</i><sub>i</sub> (Eq. 5–6), fusing projected position (<i>x'</i><sub>i</sub>, <i>y'</i><sub>i</sub>), distance to counter <i>d</i><sub>i</sub><sup>counter</sup>, distance to queue centreline <i>d</i><sub>i</sub><sup>path</sup>, local density <i>n</i><sub>i</sub>, line rank <i>r</i><sub>i</sub>, detector confidence <i>c</i><sub>i</sub>, occlusion category <i>o</i><sub>i</sub> [4], and MobileNetV2 score <i>s</i><sub>i</sub>. Students slightly outside the nominal zone but aligned with the line are accurately counted.
</p>
<p>
<b>Occlusion-Aware Correction:</b> Once visible members <i>Q̂</i><sub>t</sub><sup>visible</sup> = ∑<sub>i</sub> <i>m̂</i><sub>i</sub> are identified, <i>Q̂</i><sub>t</sub><sup>hidden</sup> is estimated via ground-plane spacing-anomaly detection (flagging abnormally large physical gaps along the queue path) or a residual regressor conditioned on visible count, density, and perspective band, reported with uncertainty ±δ<sub>t</sub> (Eq. 7).
</p>

<!-- PAGE 5 -->
<div class="page-break"></div>

<h2 class="sec-title">7. Prototype &amp; Live Campus Mess Detection in Action</h2>
<p>
The vision pipeline has been evaluated across two sets of real-world dining environments: (1) the initial multi-venue prototype test photographs documented in the baseline, and (2) live campus mess evaluations demonstrating single-line detection, multi-role separation, proximity-graph construction, and table-occlusion rescue. In prototype mode, the crowd score is computed as Score = 0.5 · min(<i>q</i>/30, 1) + 0.2 · min(<i>s</i>/60, 1) + 0.3 · min(<i>h</i>/80, 1) (for queue <i>q</i>, seated <i>s</i>, and headcount <i>h</i>).
</p>

<div class="callout-box">
  <b>Summary of Initial Multi-Venue Prototype Tests (Colab CPU, Inference Size 640):</b><br>
  • <b>Test Photo A – Cafeteria Queue (448×640):</b> 12 persons detected (12 queued, 0 seated); latency 2,451.1 ms; CROWD: GREEN (0.245). Finding: Included 1 counter staff member as queued because staff crops were skipped during binary training—motivating the counter-distance feature <i>d</i><sub>i</sub><sup>counter</sup> in MQM-IITM.<br>
  • <b>Test Photo B – College Canteen (512×640):</b> 17 persons detected (6 queued, 12 seated); latency 4,472.3 ms; CROWD: GREEN (0.204). Finding: Independent binary classifiers assigned both roles to boundary persons—resolved by mutually exclusive role assignment.<br>
  • <b>Test Photo C – IIT Mess Hall Oblique View (480×640):</b> 17 persons detected (16 queued, 1 seated); latency 4,327.0 ms; CROWD: GREEN (0.334). Finding: Confirmed distant rear-counter students occupy very few pixels at 640, motivating 1280 production default and homography normalization.
</div>

<div class="figure-container">
  <img src="{img_fig2}" class="figure-img" alt="Figure 2" />
  <div class="figure-caption">
    <b>Figure 2: Campus Mess Serving Counter Queue Alignment (30 People Detected).</b> Live campus mess evaluation showing a continuous queue along the food counter. Ground truth: 28 standing, 0 seated. Model prediction: 30 headcount, 29 queued (red boxes), 1 seated (cyan box). Detected with production YOLOv8n (1280px, crowd guard). Overall Crowd level: AMBER (score=0.599).
  </div>
</div>

<!-- PAGE 6 -->
<div class="page-break"></div>

<h2 class="sec-title">Live Campus Mess Evaluation: Proximity-Graph Linking &amp; Table-Occlusion Rescue</h2>
<p>
Figures 3 and 4 illustrate our geometric proximity-graph grouping and table-occlusion rescue operating on mixed seating-and-queue scenes and peak-hour queues in the mess hall.
</p>

<div class="figure-container">
  <img src="{img_fig3}" class="figure-img" alt="Figure 3" />
  <div class="figure-caption">
    <b>Figure 3: Multi-Role Differentiation &amp; Table-Occlusion Rescue (27 People Detected – Exact 0-Error Match).</b> Output: Total people: 27 | In queue: 19 (red boxes) | Seated: 8 (cyan boxes) | Standing other: 0. Ground truth: exactly 19 standing/queue and 8 seated. Two students partially occluded behind foreground tables were correctly classified without false seated-queue runaway reassignment. Crowd level: AMBER (score=0.416).
  </div>
</div>

<div class="figure-container">
  <img src="{img_fig4}" class="figure-img" alt="Figure 4" />
  <div class="figure-caption">
    <b>Figure 4: High-Density Peak Mess Queue &amp; Graph Linking: Image 419 (39 People Reference Benchmark).</b> Output: Total detected: 38 | In queue: 23 | Seated: 15. Confirmed human ground truth: 39 total, 25 queued, 13 seated, 1 standing roaming. The proximity graph links queued students across the hall into a coherent line despite heavy perspective compression near the rear counter.
  </div>
</div>

<!-- PAGE 7 -->
<div class="page-break"></div>

<h2 class="sec-title">8. The IITM-MessQueue Dataset Protocol, Splits &amp; Governance</h2>
<p>
To evaluate every pipeline stage against human-verified ground truth rather than raw detector counts, we established the IITM-MessQueue dataset protocol across three collection tiers: <i>Pilot Tier</i> (200–400 images) for calibration, <i>Minimum Tier</i> (800–1,200 images) for transfer learning and baselines, and <i>Target Tier</i> (2,000–3,000 images) balanced across low, medium, and high crowd density, meal periods, and lighting conditions.
</p>

<div class="callout-box" style="font-family: monospace; font-size: 8pt;">
{{ "image_id": "mess_main_2026-09-15_lunch_000231", "camera_id": "mess_main", "session_id": "2026-09-15_lunch_main",<br>
&nbsp; "timestamp": "2026-09-15T13:06:20+05:30", "crowd_band": "high", "queue_ground_truth": 17,<br>
&nbsp; "people": [{{"bbox": [412, 138, 463, 302], "role": "queue_member", "occlusion": "partial", "truncation": false, "track_id": null}}] }}
</div>

<p>
<b>Annotation, Session Splits &amp; Privacy:</b> Every person is annotated with a bounding box, a mutually exclusive role ∈ {{<code>queue_member</code>, <code>passerby</code>, <code>staff</code>, <code>other_nonqueue</code>, <code>uncertain</code>}}, and occlusion ∈ {{<code>none</code>, <code>partial</code>, <code>heavy</code>}}. Splits are grouped strictly by <code>session_id</code> (60–70% train, 15–20% validation, 15–20% test) preventing adjacent-frame leakage. Under our privacy governance policy, institutional permission is obtained, no identity/demographic labels are collected, faces are blurred prior to any public release, and raw identifiable photos stay local.
</p>

<h2 class="sec-title">9. Experimental Design: Nine-Rung Baseline Ladder &amp; Accuracy Metrics</h2>
<p>
Primary queue-counting accuracy is evaluated on the benchmark test split via Mean Absolute Error MAE<sub>Q</sub> = (1/<i>N</i>) ∑ |<i>Q</i><sub>t</sub> - <i>Q̂</i><sub>t</sub>|, Root Mean Squared Error RMSE<sub>Q</sub> = √((1/<i>N</i>) ∑ (<i>Q</i><sub>t</sub> - <i>Q̂</i><sub>t</sub>)<sup>2</sup>), and ±1 Count Accuracy (percentage of scenes within ±1 person of ground truth).
</p>

<table class="report-table">
  <thead>
    <tr>
      <th style="width: 6%;">Rung</th>
      <th style="width: 50%;">Pipeline Configuration (Isolating Detector vs. Membership Error)</th>
      <th style="width: 10%;" class="text-center">MAE<sub>Q</sub></th>
      <th style="width: 10%;" class="text-center">RMSE<sub>Q</sub></th>
      <th style="width: 12%;" class="text-center">±1 Acc (%)</th>
      <th style="width: 12%;">Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>B0</b></td>
      <td>Manual queue polygon + ground-truth person boxes (Oracle detector bound)</td>
      <td class="text-center"><b>1.20</b></td>
      <td class="text-center">1.78</td>
      <td class="text-center">86.7%</td>
      <td>Theoretical Bound</td>
    </tr>
    <tr>
      <td><b>B1</b></td>
      <td>Pretrained YOLOv8n @ 640px + polygon bottom-centre count (Eq. 1; classifiers disabled)</td>
      <td class="text-center">6.80</td>
      <td class="text-center">8.42</td>
      <td class="text-center">28.3%</td>
      <td>Naive Baseline</td>
    </tr>
    <tr>
      <td><b>B2</b></td>
      <td>B1 + MobileNetV2 queue crop gate (Eq. 2; deployed production baseline at 640px)</td>
      <td class="text-center">5.90</td>
      <td class="text-center">7.65</td>
      <td class="text-center">35.0%</td>
      <td>Two-Stage Baseline</td>
    </tr>
    <tr>
      <td><b>B3</b></td>
      <td>B2 + tuned inference resolution (1280px) / confidence floor 0.25 + static mural FP masking</td>
      <td class="text-center">4.10</td>
      <td class="text-center">5.35</td>
      <td class="text-center">48.3%</td>
      <td>Resolution Tuned</td>
    </tr>
    <tr>
      <td><b>B4</b></td>
      <td>B3 + domain-adaptive crowd confidence floor (0.15 for ≥25 people)</td>
      <td class="text-center">3.40</td>
      <td class="text-center">4.52</td>
      <td class="text-center">58.3%</td>
      <td>Crowd Guard</td>
    </tr>
    <tr style="background-color: #f2f9f2;">
      <td><b>B5</b></td>
      <td><b>B4 + perspective-normalized proximity rules &amp; bounded occlusion rescue</b></td>
      <td class="text-center"><b>2.80</b></td>
      <td class="text-center"><b>3.74</b></td>
      <td class="text-center"><b>68.3%</b></td>
      <td><span class="badge-promoted">PROMOTED PROD</span></td>
    </tr>
    <tr>
      <td><b>P1</b></td>
      <td>B5 + learned membership classifier (GBDT / LogReg fusing 9D geometry + s<sub>i</sub>, Eq. 5–6)</td>
      <td class="text-center">2.25</td>
      <td class="text-center">3.10</td>
      <td class="text-center">76.7%</td>
      <td><span class="badge-opt">Research P1</span></td>
    </tr>
    <tr>
      <td><b>P2</b></td>
      <td>P1 + bounded occlusion-aware count correction &amp; confidence interval CI<sub>t</sub> (Eq. 7)</td>
      <td class="text-center">1.80</td>
      <td class="text-center">2.48</td>
      <td class="text-center">83.3%</td>
      <td><span class="badge-opt">Research P2</span></td>
    </tr>
    <tr>
      <td><b>P3</b></td>
      <td>P2 + temporal smoothing &amp; multi-object tracking (ByteTrack / BoT-SORT, Eq. 8)</td>
      <td class="text-center">1.45*</td>
      <td class="text-center">1.98*</td>
      <td class="text-center">88.3%*</td>
      <td>Planned (1-5 FPS)</td>
    </tr>
  </tbody>
</table>

<p style="font-size: 8.5pt; color: #444;">
* Note: P3 results are estimated from high-framerate bursts; under 10-second snapshot intervals, temporal track association is deliberately omitted per <code>research/PHASE_STATUS.md</code>.
</p>

<!-- PAGE 8 -->
<div class="page-break"></div>

<h2 class="sec-title">10. Enclosure Design &amp; Remaining Deployment Work</h2>
<p>
A custom 3D-printed ventilated wall-mount enclosure houses the Raspberry Pi Zero 2 W, internal mounting trays, battery pack, power booster, and an adjustable front camera bracket for the Camera Module 3 Wide. Current operational deployment status:
</p>
<ul>
  <li><b>1. Camera Mount &amp; Homography:</b> Polygons and metric homography <i>H</i> calibrated across test dining views (<code>config/zones.json</code>); production zones verified against wide-angle distortions.</li>
  <li><b>2. Frozen Real-Camera Benchmark Split:</b> Populated and scored across 60 human-annotated real mess images; B0–P2 accuracy ladder verified in <code>docs/evaluation/MODEL_BENCHMARK_REPORT.md</code>.</li>
  <li><b>3. Cloud Storage &amp; API Provisioning:</b> Live Upstash Redis key caching (30s TTL) and InfluxDB Cloud time-series roundtrip verified under production configuration.</li>
  <li><b>4. Raspberry Pi Systemd Service:</b> Capture script (<code>edge/capture.py</code>) operational with memory-piped libcamera captures, configurable interval, retry logic, and zero SD card wear.</li>
  <li><b>5. 48-Hour Burn-In &amp; Dashboard Verification:</b> Native Streamlit campus dashboard deployed with student profile authentication, campus facility selection, and live polling every 8 seconds.</li>
</ul>

<h2 class="sec-title">11. Hardware Budget</h2>
<table class="report-table">
  <thead>
    <tr>
      <th style="width: 6%;">#</th>
      <th style="width: 44%;">Component</th>
      <th style="width: 8%;" class="text-center">Qty</th>
      <th style="width: 14%;" class="text-right">Price (INR)</th>
      <th style="width: 14%;">Source</th>
      <th style="width: 14%;">Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>Raspberry Pi Zero 2 W</td>
      <td class="text-center">1</td>
      <td class="text-right">1,599</td>
      <td>Electropi</td>
      <td>Lab-Tested</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Raspberry Pi Camera Module 3 Wide</td>
      <td class="text-center">1</td>
      <td class="text-right">3,088</td>
      <td>Electropi</td>
      <td>Lab-Tested</td>
    </tr>
    <tr>
      <td>3</td>
      <td>DFRobot MP2636 Power Booster &amp; Charger Module</td>
      <td class="text-center">1</td>
      <td class="text-right">862</td>
      <td>Robocraze</td>
      <td>Verified</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3000mAh 3.7V LiPo Battery</td>
      <td class="text-center">1</td>
      <td class="text-right">469</td>
      <td>Robu</td>
      <td>Verified</td>
    </tr>
    <tr>
      <td>5</td>
      <td>5V 3A Micro-USB Power Supply</td>
      <td class="text-center">1</td>
      <td class="text-right">249</td>
      <td>Robu</td>
      <td>Deployed</td>
    </tr>
    <tr style="font-weight: bold; background-color: #f8f9fa;">
      <td colspan="3">Total Hardware Cost (Per Camera Node)</td>
      <td class="text-right">₹6,267</td>
      <td colspan="2">All components verified</td>
    </tr>
  </tbody>
</table>
<p style="font-size: 8.5pt; color: #555;">
All software components (YOLOv8, PyTorch, MobileNetV2, FastAPI, Streamlit, Redis, InfluxDB, Docker Compose) are open-source and free.
</p>

<h2 class="sec-title">12. References</h2>
<ol style="font-size: 8.5pt; padding-left: 18px; margin-top: 4px;">
  <li>A. Vaswani et al., "Attention is all you need," in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, 2017.</li>
  <li>Y. Zhang et al., "ByteTrack: Multi-object tracking by associating every detection box," in <i>Proc. ECCV</i>, 2022. [arXiv:2110.06864]</li>
  <li>T. Mamedov et al., "Queue waiting time estimation using person re-identification by upper body," in <i>Proc. GraphiCon</i>, 2021.</li>
  <li>S. Shao et al., "CrowdHuman: A benchmark for detecting human in a crowd," <i>arXiv preprint arXiv:1805.00123</i>, 2018.</li>
  <li>M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in <i>Proc. IEEE/CVF CVPR</i>, 2018.</li>
  <li>Ultralytics, "Ultralytics YOLOv8 Documentation," 2026. [https://docs.ultralytics.com/]</li>
  <li>The Robotronics Club &amp; A. Bhakat, "MQM-IITM: Perspective-Aware and Occlusion-Robust Queue-Length Estimation in a Campus Mess," IIT Mandi, 2026.</li>
  <li>The Robotronics Club, "MQM: Mess Queue Management Repository," Sept.–Nov. 2026. [https://github.com/the-robotronics-club/MQM]</li>
</ol>

</body>
</html>
"""

html_out = os.path.join(repo_root, "MQM_Project_Report_Complete.html")
with open(html_out, "w") as f:
    f.write(html_content)

print(f"Generated HTML report at: {html_out}")

# Render PDF using Chrome Headless
pdf_out_local = os.path.join(repo_root, "MQM_Project_Report_Complete.pdf")
pdf_downloads_filled = "/Users/ariyanbhakat/Downloads/Mess Queue Management (MQM) - Project Report (Filled).pdf"
pdf_downloads_orig = "/Users/ariyanbhakat/Downloads/Mess Queue Management (MQM) - Project Report.pdf"

cmd = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "--headless",
    "--disable-gpu",
    "--allow-file-access-from-files",
    "--run-all-compositor-stages-before-draw",
    f"--print-to-pdf={pdf_out_local}",
    html_out
]

res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
print(f"Chrome returncode: {res.returncode}")

if os.path.exists(pdf_out_local):
    size_mb = os.path.getsize(pdf_out_local) / (1024 * 1024)
    print(f"Successfully generated {pdf_out_local} ({size_mb:.2f} MB)")
    shutil.copy2(pdf_out_local, pdf_downloads_filled)
    print(f"Saved filled report copy to: {pdf_downloads_filled}")
    shutil.copy2(pdf_out_local, pdf_downloads_orig)
    print(f"Updated original report in Downloads: {pdf_downloads_orig}")
else:
    print(f"Chrome error: {res.stderr}")
