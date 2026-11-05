# Deployment

## Free-only Render demo

The default `render.yaml` and explicit `render.free.yaml` both define the free-only demo. The earlier paid alternative is archived as `render.paid.yaml` and is not deployed.
It defines two Free web services: a low-memory ONNX API and the existing live
Streamlit dashboard. There is no paid database or disk. The API stores recent
readings in local SQLite and expires latest state after 120 seconds; history
retains up to 24 hours while that instance is running. **History resets on
restart, redeploy, or sleep.** This is a demo deployment, not archival storage.

The API uses FP32 exports of the original YOLOv8n and seated MobileNetV2
checkpoints, with the existing 1280 rectangular preprocessing, confidence guard,
posture preprocessing, and distance queue rule. No training or label-based
threshold search is part of export. The runtime excludes PyTorch; the separate
build stage needs PyTorch to export the models. `Dockerfile.free-dashboard`
contains only dashboard dependencies and starts in snapshot-only Live cameras.

Create a Render Blueprint from branch `codex/04-integration-quality`, setting
**Blueprint Path to `render.free.yaml`**. Verify both plans show Free. The
`READ_API_BASE_URL` prompt takes the new API's public HTTPS URL; Free API services
cannot receive private-network traffic. Correct the dashboard environment if
Render assigns a different service URL. Use the API's generated camera secret
only on the RPi; the dashboard does not receive it.

Free services sleep on idle and share the workspace's monthly free instance
hours with other free services. Wake the dashboard/API before the demo and allow
for a cold start. The RPi uploader's default read timeout is 180 seconds. See
[Render's free-tier limits](https://render.com/docs/free) for the current quota
and sleep behavior. If free resources are exhausted, stop and resolve within
the free tier; do not select a paid upgrade under this deployment authorization.

## RPi snapshot to live dashboard

The target flow is `RPi JPEG -> POST /ingest/mess_main -> YOLO/posture/distance
counts -> SQLite recent readings -> dashboard Live cameras` on the free setup.
The optional external-storage deployment uses InfluxDB history and Redis latest.
The RPi only captures/uploads; inference runs on the server. The dashboard
refreshes every eight seconds while open; a stopped camera goes offline after
the latest-reading TTL instead of continuing to show a stale live count.

Use `FRAME_METRICS_MODE=distance` until a mounted view is calibrated. This mode
accepts arbitrary JPEG dimensions and runs the same scene logic as photo
evaluation. It reports headcount, queue, sitting, and other standing. Seat
capacity/occupancy are JSON `null` and are not displayed as measured occupancy.
`FRAME_METRICS_MODE=calibrated` retains the original zone-based behavior and
resolution checks. Keep `QUEUE_ESTIMATOR=production` with distance snapshots.

For one server, the self-contained deployment is:

```bash
# Set strong credentials in .env using .env.example first.
docker compose -f compose.snapshot.yml up -d --build
docker compose -f compose.snapshot.yml ps
```

Redis and InfluxDB are private. API 8000 and dashboard 8501 bind to server
loopback; route them through an HTTPS reverse proxy before using a remote Pi.
InfluxDB history uses a persistent Docker volume. Raw JPEGs are processed in
memory, not added to the history database.

### Render

`render.paid.yaml` is the archived paid Blueprint: a 2 GB API, free live dashboard,
internal-only free Key Value cache, and private 512 MB InfluxDB with a 1 GB disk.
**This includes recurring paid resources; review the current Render cost before
deploying.** The free dashboard can sleep, so opening it can require a cold start.
The API and dashboard use separate HTTPS `onrender.com` URLs. API and history
service memory must be monitored under actual capture load.

In Render, create a new Blueprint using this repository and branch
`codex/04-integration-quality`; the archived paid path is `render.paid.yaml`. The current authorization is free-only: do not deploy that paid alternative.
Generated camera/database secrets are stored in Render environment variables.
The dashboard receives the API's private hostname only. No database secret is
passed to the dashboard. Auto-deploy is disabled for the API/dashboard so future
research pushes do not silently replace the running model service.

After deployment, obtain `CAMERA_API_KEY` from the API's Render environment
panel and configure it on the Pi without committing it or posting it publicly.
The same camera ID must be used on both ends:

```bash
export SERVER_URL=https://YOUR-API-SERVICE.onrender.com
export CAMERA_ID=mess_main
# Set CAMERA_API_KEY securely in the Pi's environment.
python3 edge/capture.py
```

The sender prefers `rpicam-still`, falls back to `libcamera-still`, and captures
1280x720 by default. `IMAGE_WIDTH`, `IMAGE_HEIGHT`, `CAPTURE_INTERVAL_SECONDS`
(default 10), and `UPLOAD_TIMEOUT_SECONDS` (default 180) are configurable.
For calibrated mode the resolution must match its saved camera zones.

Test the exact upload path from a laptop before connecting the Pi:

```bash
python -m tools.smoke_snapshot --url https://YOUR-API-SERVICE.onrender.com \
  --image /path/to/snapshot.jpg
# Or send a single frame using the actual Pi uploader:
python3 edge/capture.py --image /path/to/snapshot.jpg --once
```

The smoke check requires a fresh acknowledged reading in current status and
history with matching counts/timestamp. Merely seeing `/health` return 200 does
not prove inference or persistence. Open the dashboard's Live cameras view and
verify that the new timestamp/headcount appears. Physical RPi capture is a
separate final hardware check.

The free Render API and dashboard use separate images; only the API contains
model exports. The optional `compose.deploy.yml` deployment uses a shared
image with separate API/dashboard processes, restart policies, and health
checks. No raw training/evaluation photos are copied into these images.

## Single host with Docker Compose

1. Install Docker Engine/Compose on the server.
2. Set backend credentials in `.env` using `.env.example`. Use managed Redis
   and InfluxDB, or a separately secured storage stack. Container `localhost`
   refers to the container itself, not the host; use reachable service URLs.
3. Build and start:

   ```bash
   docker compose -f compose.deploy.yml up -d --build
   docker compose -f compose.deploy.yml ps
   curl --fail http://127.0.0.1:8000/ready
   curl --fail http://127.0.0.1:8501/_stcore/health
   ```

4. Route an HTTPS reverse proxy to the loopback dashboard port 8501, with
   WebSocket support. Route authenticated camera ingestion to API port 8000.
   Keep database ports private. The dashboard receives only the read API URL;
   backend credentials are passed only to the API.
5. Calibrate `config/zones.json` for the mounted camera. Current sample geometry
   does not apply to arbitrary real images. Photo/live distance queue and posture estimates do not need zones.

The phone video mode currently uses local peer connectivity without STUN/TURN.
Remote internet video requires separately configured WebRTC connectivity and
HTTPS; uploaded photos and API ingestion do not depend on that mode.

The container uses CPU PyTorch and runs as a non-root user. First model loading
can be slow. `/ready` checks storage availability; it does not measure model
accuracy. Start with at least 4 GB RAM and measure latency on the actual host.

## Evaluation before rollout

```bash
python -m pip install -r requirements-eval.txt
python -m tools.evaluation.real_images \
  --images /path/to/local/photos --labels /path/to/dataset_labels.json \
  --adaptive-confidence
```

Reports are written under ignored `data/outputs/real-images/`. HEIC input is
supported by the evaluation extra. JPEG/PNG are supported in dashboard uploads.
The checked-in aggregate result is in `docs/evaluation/2026-10-21-summary.json`.
The current count errors are significant in dense scenes; treat output as an
estimate and validate on a separate held-out capture session before operational use.

The snapshot flow has been tested locally with the real detector, Redis,
InfluxDB, and dashboard readback; see the dated snapshot deployment review for
the exact scope. A public deployment additionally needs the hosting account and
approval of its selected recurring plan.

For fixed-cutoff comparisons, omit `--adaptive-confidence`. Runtime uses the crowd
guard by default; `MODEL_ADAPTIVE_CONFIDENCE=false` disables it. The default 0.25
camera-zone floor continues to apply to calibrated API metrics until camera-specific
validation supports changing it.
