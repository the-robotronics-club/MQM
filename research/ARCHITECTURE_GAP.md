# MQM research architecture gap

## Current architecture

The deployed path is `JPEG -> OpenCV decode -> YOLOv8 person boxes -> bottom-centre
image point -> configured image-space polygons -> optional MobileNetV2 queue and seated
crop classifiers -> metrics`. FastAPI authenticates and accepts JPEGs, Redis stores the
latest complete reading with a TTL, InfluxDB stores history, and Streamlit reads only the
API. The production estimator is retained as baseline **BCURRENT**.

## Target architecture

The research path adds a detector-neutral record, camera homography, ground-plane
projection, an ordered queue centreline, interpretable geometric/context features,
geometric or learned membership, bounded missing-count correction, and explicitly
qualified uncertainty. It produces the existing `queue_count` plus optional diagnostics.

## Reusable components

- `inference/model.py`: cached YOLOv8 detector and replaceable weight path.
- `inference/pipeline.py`: validated JPEG decode and production orchestration.
- `inference/zones.py`: bottom-centre/centroid functions and production polygons.
- `inference/classifiers.py`: reviewed production queue/seated classifiers.
- `inference/metrics.py`: stable occupancy and crowd metrics.
- `api/`, Redis, InfluxDB, and `dashboard/`: operational transport and presentation.

## Modified components

- `inference/pipeline.py` dispatches queue estimation by `QUEUE_ESTIMATOR` while keeping
  `production` as the default.
- API/storage schemas accept optional research fields without changing existing fields.
- Streamlit shows compact research diagnostics only when those fields are present.
- Tooling and dependencies include reproducible research training/evaluation commands.

## New components

The `research/` package separates detection adapters, calibration, geometry, features,
membership, occlusion correction, datasets, training, and evaluation. Research code never
depends on an Ultralytics result object. Experiment writers reject an existing run folder.

## Backward compatibility

`QUEUE_ESTIMATOR=production` executes the prior polygon + MobileNetV2 path. `queue_count`
remains required. Research fields are optional in API responses and are omitted for the
production path. Existing dashboard consumers can ignore them. No database migration is
required; InfluxDB writes optional fields only when present.

## Dataset and training dependencies

Real camera correspondences, session-labelled images, per-person roles/occlusion labels,
and frozen session-grouped manifests are not in this repository. Learned membership and
residual correction therefore require trained artifacts created from validated manifests.
Uncertain people are excluded from supervised membership training and reported separately.
Missing artifacts never trigger an untrained learned model: membership can explicitly fall
back to geometric rules, and missing-count correction returns zero with an uncalibrated
interval marker.

## Experimental dependencies

Valid comparisons require the same frozen test sessions, real per-camera calibration,
detector checkpoints, annotation protocol, and timing environment. Detection AP requires
detection ground truth and is recorded only when supplied. Temporal P3 remains disabled
until ordered 1–5 FPS clips exist. No result in this package is a paper claim until an
experiment artifact records its inputs and held-out evaluation.
