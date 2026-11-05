# Mess Queue Management

Mess Queue Management monitors congestion in a college mess. Test it without a camera by uploading a JPEG/PNG in the Streamlit dashboard, or connect Raspberry Pi cameras for live readings and history.

## Test without a camera

Open the dashboard and select **Upload image** (the default view). Choose a JPEG
or PNG up to 10 MB, or select a file that you placed in `data/samples/local/`,
then click **Analyze image**. The result includes a person count and a
downloadable image with numbered detection boxes. Results remain in the browser
session without the camera TTL; uploads do not write to Redis or InfluxDB.

`data/samples/local/` is intentionally ignored by Git so that personal, licensed,
or otherwise non-redistributable test photos are never committed. Distance-based queue and posture counts are estimates for each photo. Seat occupancy
and calibrated camera metrics require the saved viewpoint and resolution.

For arbitrary photos, the default pipeline is **YOLOv8n → Ishan's seated classifier
→ distance-based queue grouping**. It reports total people, queued people, sitting
people, and other standing people. These last three categories sum to the total.
Blue Q boxes mark queue estimates, orange S marks sitting, and green O marks other
standing people. The grouping can confuse nearby groups with queues. It does not
use the queue crop classifier to decide membership.

Enable **Use saved mess layout** only for the configured camera's viewpoint and
resolution to add zone-based queue, seat occupancy, and crowd metrics. Uploaded
photos are not written to camera history. The dashboard has no public built-in
photo samples; put permitted photos in the ignored local-samples folder.

Photo analysis runs the repository's YOLO and optional attribute classifiers in
the dashboard process. Install the full `requirements.txt` and local model weights
on that host; it does not need a running API or database. **Live cameras** retains
the API-backed, read-only monitoring view.

This README documents the implemented software, not only the original project proposal. Current photo/live distance pipeline finalized on 21 October 2026.

## Live phone camera on local Wi-Fi

Choose **Phone camera** in the dashboard for continuous browser video and person
detection. On a Mac, `http://localhost:8501` can request webcam permission. A phone
needs a trusted HTTPS address; opening the Mac's plain HTTP IP address is not enough.

For an iPhone and Mac on the same Wi-Fi, launch the camera-only HTTPS page:

```bash
# Use the Python environment where requirements-dev.txt is installed.
/path/to/venv/bin/python tools/serve_phone.py --address YOUR_MAC_WIFI_IP
```

On macOS, `ipconfig getifaddr en0` usually prints the Wi-Fi address. The launcher
prints a certificate download URL, an HTTPS URL, and a random access code.
It binds only to that local address, with HTTPS on port 8502 and certificate
download on port 8765. No public tunnel, STUN service, or TURN relay is configured.

1. In iPhone Safari, open the printed certificate URL and allow the profile download.
2. Install **MQM local camera test** under Settings → General → VPN & Device Management.
3. Under Settings → General → About → Certificate Trust Settings, enable full trust
   for **MQM local camera test**. See [Apple's certificate instructions](https://support.apple.com/en-us/102390).
4. Open the printed HTTPS URL, enter the access code, tap **START**, and allow the camera.
5. Use **STOP** when finished. Remove the test certificate profile after testing.

The rear camera is preferred and audio is disabled. Frames are analyzed on the
Mac up to twice per second; the returned video shows the latest analyzed frame
with boxes. Frames and counts are not persisted to Redis/InfluxDB. This handheld
mode estimates queued, sitting, and other standing people using the same distance
pipeline as photo uploads. Calibrated seat occupancy and camera history still
use the existing fixed-camera ingestion system. Keep Safari open and the Mac awake.

The test certificate expires in seven days and is stored outside the repository;
each launcher run creates a new certificate and access code. Stop the launcher
with Ctrl-C. If the Wi-Fi isolates devices, local video connections may fail even
though both devices use the same network. Physical iPhone capture and permission
handling must be verified on the phone; automated tests cover frame processing
and the access-code gate, not Safari's camera hardware.

## What is working

- Authenticated camera-frame ingestion with FastAPI.
- Real YOLOv8 person detection in original frame coordinates.
- Camera-specific polygon zones for queue, seating, and entrance regions.
- Reviewed MobileNetV2 classifiers that refine queue and seated counts from detected person crops.
- Redis latest-reading cache with camera TTL/offline state.
- InfluxDB time-series history for charts.
- Read-only Streamlit dashboard with current metrics and a 60-minute trend.
- Docker Compose local storage, reproducible commands, API tests, classifier tests, dashboard tests, and an opt-in real-stack test.

## Research queue estimator

The deployed polygon + MobileNetV2 estimator remains the default `BCURRENT` baseline.
The new `research/` package adds detector-neutral records, homography projection, a curved
ground-plane queue path, typed features, deterministic and learned membership, bounded
occlusion correction, session-grouped dataset tools, baseline/ablation runners, session
bootstrap intervals, and reproducible result artifacts. Set `QUEUE_ESTIMATOR` to
`geometric`, `membership`, or `occlusion` only after creating a validated per-camera
calibration. See `research/README.md` and `research/PHASE_STATUS.md` for exact commands and
the data-dependent limitations.

## Hardware components

| Component | Quantity | Purchased | Price (INR) | Link |
| --- | --- | --- | --- | --- |
| DFRobot MP2636 Power booster and charger module | 1 | YES | 900 | [Robocraze](https://robocraze.com/products/dfrobot-mp2636-power-booster-charger-module?variant=47362616459488&country=IN&currency=INR&utm_medium=product_sync&utm_source=google&utm_content=sag_organic&utm_campaign=sag_organic&utm_source=google&utm_medium=cpc&utm_campaign=BL+%7C+Pmax+%7C+Feed+Only+%7C+RoboCraze+%7C+Electronic+Components+%7C+31%2F05&utm_source=googleads&utm_medium=ppc&utm_campaign=21337209786&utm_content=_&utm_term=&campaignid=21337209786&adgroupid=&campaign=21337209786&gad_source=1&gad_campaignid=21343423652&gbraid=0AAAAADgHQvaLVEJeEAKHwMU3YTgQoi5BT&gclid=Cj0KCQjw--7UBhCpARIsAGJBptjohsvz7vmGAU_d2penL6hIC7rWc6sJIatZNQsmgroT2qs71N9GuFcaAhuzEALw_wcB) |
| Raspberry Pi Camera Module 3 Wide | 1 | YES | 3088 | [Electropi](https://www.electropi.in/raspberry-pi-camera-module-3-wide) |
| 3000mAh 3.7V Micro Lipo battery | 1 | YES | 469 | [Robu](https://robu.in/product/nova-105050-3000mah-3-7v-micro-lipo-battery-pack/?gad_source=1&gad_campaignid=17427802559&gbraid=0AAAAADvLFWfBOAxwtKoc_rB5XuRQbOHss&gclid=Cj0KCQjw--7UBhCpARIsAGJBptjysFMZOT-CO76wAV8GTk03JG791q1cWhqdjimmNx-tbqhBeZpl-JQaAqo8EALw_wcB) |
| Raspberry Pi 3 Model A+ | 1 | YES | 3211 | [Robu](https://robu.in/product/raspberry-pi-3-model-a/) |
| Standard 5V 3A Power Supply with Micro USB Plug | 1 | YES | 249 | [Robu](https://robu.in/product/orange-5v-3a-power-supply-adapter-charger-with-micro-usb-plug/?gad_source=1&gad_campaignid=17427802559&gbraid=0AAAAADvLFWdjgjAWQUpBgB_tcSoG4QZNW&gclid=CjwKCAjwkaXUBhASEiwAZI3ds7AxVBPzPY9SKF68SyXc_KrhFeh9InSrM8iIQR74nv2zibZH_YFcJBoCRKUQAvD_BwE) |

## System architecture

```text
Raspberry Pi camera
  │ authenticated JPEG POST
  ▼
FastAPI ingestion API
  ├─ OpenCV JPEG decoding and frame-size validation
  ├─ YOLOv8 person detection
  ├─ queue/seating/entrance zone assignment
  ├─ MobileNetV2 queue + seated crop classifiers
  ├─ InfluxDB: durable timestamped history
  └─ Redis: latest reading with expiry

Streamlit dashboard
  ├─ GET /status       → current online/offline state
  └─ GET /history/{id} → recent time-series trend
```

The dashboard never receives database credentials or connects directly to Redis/InfluxDB. Ingestion succeeds only after both storage writes succeed; inference and storage failures are returned explicitly.

## Software delivered

### Computer vision and metrics

- inference/model.py loads YOLOv8 and returns COCO person detections only.
- inference/pipeline.py decodes JPEG bytes, validates camera coordinate space, and runs the full inference-to-metrics path.
- inference/zones.py maps person floor points to configured polygons and rejects incompatible frame dimensions instead of producing false zeroes.
- inference/metrics.py produces headcount, queue count, occupied seats, occupancy percentage, crowd level, per-zone counts, and detection totals.
- inference/scene.py shares the finalized photo/live distance queue estimates.
- inference/distance_queue.py promotes the notebook grouping and posture correction into runtime code.
- inference/annotations.py draws shared blue/orange/green category boxes.
- inference/classifiers.py safely loads two reviewed tensor-only MobileNetV2 state dictionaries and classifies YOLO person crops in a batch.

In the calibrated API, queue classification applies only to people already in a queue zone. Seated classification applies only to people in a seating zone. This combines visual posture/context with the physical layout rather than allowing a classifier to invent a queue or seat anywhere in the frame.

### Backend and storage

- POST /ingest/{camera_id} accepts authenticated multipart JPEG frames.
- X-Camera-Key authentication, upload-size checks, JPEG validation, unknown-camera rejection, and strict Pydantic metric invariants are enforced.
- Redis stores expiring current snapshots; offline cameras return online: false rather than fake zero values.
- InfluxDB stores complete readings and returns sorted historical metrics.
- GET /status, GET /status/{camera_id}, and GET /history/{camera_id} are read-only endpoints.
- /health is process liveness; /ready checks Redis and InfluxDB and returns HTTP 503 until both are available.

### Dashboard

- dashboard/app.py provides a read-only Streamlit dashboard.
- It refreshes current state every eight seconds and shows headcount, queue, seats, occupancy, crowd level, timestamp, and a trend chart.
- It renders explicit offline, configuration, API-error, and no-history states instead of using mock readings.

### Operations and quality tooling

- docker-compose.yml runs Redis and InfluxDB locally.
- scripts/dev.sh starts storage, FastAPI, and Streamlit together.
- Makefile supplies install, test, lint, type-check, format-check, and storage lifecycle commands.
- .env.example and .streamlit/secrets.toml.example document required variables without committing real credentials.
- tools/load_test.py provides concurrent endpoint load testing.

## Pull-request history

| PR | Status | Delivered work |
| --- | --- | --- |
| [#1](https://github.com/the-robotronics-club/MQM/pull/1) | Closed | Initial infrastructure proposal; superseded by the integrated implementation. |
| [#2](https://github.com/the-robotronics-club/MQM/pull/2) | Merged | Real JPEG-to-YOLO-to-zone metrics pipeline and regression tests. |
| [#3](https://github.com/the-robotronics-club/MQM/pull/3) | Merged | Authenticated API, Redis current state, Influx history, readiness checks, and API tests. |
| [#4](https://github.com/the-robotronics-club/MQM/pull/4) | Merged | Read-only Streamlit dashboard, API client, current metrics, trend chart, and dashboard tests. |
| [#5](https://github.com/the-robotronics-club/MQM/pull/5) | Merged | Developer workflow, quality tooling, load test, Compose validation, and implementation documentation. |
| [#6](https://github.com/the-robotronics-club/MQM/pull/6) | Merged | Queue/seated MobileNetV2 classifier integration, reviewed weights, configuration, provenance note, and focused tests. |

## Local setup

Requirements: Python 3.11+, Docker Desktop/Engine with Compose, and curl.

```bash
make install
cp .env.example .env
```

Edit .env and replace every change-me value. At minimum, set:

- CAMERA_API_KEY — secret used by camera uploads.
- INFLUX_TOKEN — local InfluxDB or production Cloud token.
- INFLUX_INIT_PASSWORD — local InfluxDB bootstrap password.

Keep the reviewed classifiers enabled unless diagnosing or retraining:

```dotenv
ATTRIBUTE_CLASSIFIERS_ENABLED=true
QUEUE_CLASSIFIER_WEIGHTS=queue_classifier.pt
SEATED_CLASSIFIER_WEIGHTS=seated_classifier.pt
```

Start the complete application:

```bash
make dev
```

If the repository is in a cloud-synced Desktop folder and Python hangs while
reading dependencies, use an environment stored on a fully local filesystem:

```bash
MQM_VENV_DIR=/absolute/path/to/local/venv make dev
```

Install `requirements-dev.txt` into that environment first. The override applies
to the API and dashboard launched by `make dev`; other Makefile checks still use
the repository's `.venv`. Environments under `/tmp` are temporary and may need to
be recreated after cleanup or a restart.

- Dashboard: <http://127.0.0.1:8501>
- API docs: <http://127.0.0.1:8000/docs>
- Liveness: <http://127.0.0.1:8000/health>
- Readiness: <http://127.0.0.1:8000/ready>

Press Ctrl-C to stop FastAPI and Streamlit. Stop local storage when finished:

```bash
make services-down
```

## Exercise the real flow

With make dev running, upload the committed frame that matches the active 736×490 zone geometry:

```bash
set -a; source .env; set +a

curl --fail --request POST http://127.0.0.1:8000/ingest/mess_main \
  --header "X-Camera-Key: $CAMERA_API_KEY" \
  --form "file=@data/samples/mess_hall_dense.jpg;type=image/jpeg"

curl --fail http://127.0.0.1:8000/status
curl --fail http://127.0.0.1:8000/status/mess_main
curl --fail "http://127.0.0.1:8000/history/mess_main?minutes=60"
```

Expected failure behavior:

- Unknown cameras return HTTP 404.
- Missing/invalid camera keys return HTTP 401.
- Non-JPEG, corrupt, oversized, or wrong-resolution frames are rejected.
- Unavailable model or storage dependencies return HTTP 503.
- A camera with no fresh Redis reading reports online: false.

## API contract

| Endpoint | Purpose |
| --- | --- |
| POST /ingest/{camera_id} | Authenticated multipart JPEG ingestion. |
| GET /status | Current state for every configured camera. |
| GET /status/{camera_id} | Current state for one camera, including offline status. |
| GET /history/{camera_id}?minutes=60 | One to 1,440 minutes of time-series history. |
| GET /health | Process liveness. |
| GET /ready | Redis and InfluxDB dependency readiness. |

Each reading contains:

```text
camera_id, timestamp, headcount, queue_count,
seats_total, seats_occupied, seat_occupancy_pct,
crowd_level, zone_counts, detections_raw, detections_counted
```

## Configuration reference

| Variable | Purpose |
| --- | --- |
| CAMERA_API_KEY | Shared secret required for camera ingestion. |
| KNOWN_CAMERAS | Comma-separated IDs defined in config/zones.json. |
| REDIS_URL | Local Redis or managed Upstash URL. |
| LATEST_TTL_SECONDS | Freshness window before a camera becomes offline. |
| INFLUX_URL, INFLUX_TOKEN | InfluxDB 2.x/Cloud connection settings. |
| INFLUX_ORG, INFLUX_BUCKET | InfluxDB write/query destination. |
| READ_API_BASE_URL | API URL used by Streamlit. |
| CORS_ALLOW_ORIGINS | Allowed browser origins; no wildcard default. |
| MODEL_* | YOLO weights, confidence, IoU, inference size, and adaptive crowd confidence. |
| ATTRIBUTE_CLASSIFIERS_ENABLED | Enables queue/seated crop classification. |
| QUEUE_CLASSIFIER_WEIGHTS, SEATED_CLASSIFIER_WEIGHTS | Reviewed paths under models/. |
| QUEUE_ESTIMATOR | `production` (default), `geometric`, `membership`, or `occlusion`. |
| RESEARCH_* | Optional model, spacing, uncertainty, reference-point, and correction overrides. |

For production, put backend storage credentials in hosting environment configuration. Put only READ_API_BASE_URL in Streamlit secrets; never put Redis or InfluxDB credentials in the dashboard.

## Models and training handover

- models/yolov8n.pt is the standard YOLO cache. It is ignored and can be downloaded automatically by Ultralytics.
- models/queue_classifier.pt and models/seated_classifier.pt are the retrained checkpoints merged in Ishan’s PR #8. The seated model drives photo/live distance grouping; the queue model remains available to the calibrated API.
- The raw content/ training handover is ignored. It contains source photos and crop datasets not required at runtime and needs separate privacy/provenance review before publication.
- Architecture, label mapping, hashes, and safe-load verification are in [docs/reviews/2026-09-04-attribute-classifiers.md](docs/reviews/2026-09-04-attribute-classifiers.md).

## Verification commands

```bash
make test
make lint
make typecheck
make format-check
python tools/load_test.py --url http://127.0.0.1:8000/status --requests 1000 --concurrency 100
```

To run the real storage/API test after starting the application, run `make test-live`
in a second terminal. It loads credentials from `.env` and verifies that the
newly uploaded sample reaches both current status and timestamp-matched history.
This writes one sample reading to the configured databases.

For an API running at a different URL:

```bash
BASE_URL=http://127.0.0.1:8000 make test-live
```

Latest verification for PR #6:

- Full suite: **63 passed, 1 opt-in test skipped**.
- Focused classifier/pipeline/metrics suite: **37 passed**.
- Ruff, MyPy, formatting, compilation, shell syntax, and Compose validation: **passed**.
- Isolated real stack with Redis, InfluxDB, FastAPI, YOLO, both classifiers, and Streamlit dashboard: **passed**.

## Repository layout

```text
api/                    FastAPI routes, auth, schemas, cache, storage adapters
dashboard/              Read-only Streamlit UI and HTTP client
inference/              YOLO detection, crop classifiers, zones, metrics
models/                 Reviewed classifiers and local YOLO cache
config/zones.json       Camera geometry, capacities, crowd thresholds
data/samples/           Committed test frames
tests/                  Unit, API, dashboard, pipeline, and live-stack tests
tools/                  Load test, zone drawing, evaluation utilities
docs/                   Handover records, model reviews, implementation notes
```

## Deployment and current model evaluation

See [the finalized photo/live pipeline](docs/PIPELINE.md) for the selected distance
queue method, image 419 results, and full-dataset limitations.

See [deployment instructions](docs/DEPLOYMENT.md) for the container setup and
[October model evaluation](docs/reviews/2026-10-21-retrained-models.md) for measured
count errors on 60 real images. The updated classifiers use 128×128 crops.
The [follow-up improvement study](docs/reviews/2026-10-21-count-improvements.md)
compares fixed thresholds and a crowd guard. Photo/live detection now keeps the
0.25 cutoff for sparse scenes and uses 0.15 when at least 25 crowd candidates exist.
Set `MODEL_ADAPTIVE_CONFIDENCE=false` to reproduce fixed-cutoff behavior. Camera-zone
confidence floors still apply independently. Raw evaluation photos stay local.

## Remaining deployment work

- Retrace config/zones.json using frames from final mounted cameras. Current zones are calibrated to a 736×490 sample and deliberately reject other sizes.
- Confirm Pi capture resolution, lens crop, mounting angle, and upload cadence.
- Validate queue/seated classifier accuracy on held-out footage from the real mess before using estimates operationally.
- Provision production Redis/InfluxDB credentials only in the backend host.
- Configure the Raspberry Pi uploader to send authenticated JPEG frames at the desired interval.

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for handover guidance and docs/reviews/ for model/integration evidence.
