# Research implementation phase status

This status separates implemented software from evidence that still requires the IIT Mandi
mess dataset. No row below is an experimental performance claim.

| Phase | Implemented software | Evidence still required |
|---|---|---|
| 0 | Repository audit, architecture gap, file plan | None |
| 1 | Package boundaries, typed detections/features, synthetic tests | Real annotations |
| 2 | Homography computation, projection, validation, camera loader | Surveyed correspondences and approved camera YAML |
| 3 | Curved queue path, arc position, rank, density, feature extraction | Queue centreline calibration |
| 4 | Deterministic B4 rule model with reasons and scores | Validation-set threshold selection |
| 5 | Manifest schema, validation, deterministic session splits, frozen-file guard | Session-labelled image manifest |
| 6 | Logistic/GBDT trainers, uncertain-label exclusion, serialized inference | Train/validation feature manifests |
| 7 | Per-person membership diagnostics and visible count | Trained membership artifact |
| 8 | Low-occlusion spacing fitter and bounded gap correction | Low-occlusion training gaps |
| 9 | Bounded residual regressor and inference | Frame-level residual training rows |
| 10 | Validation-residual quantile intervals and explicit calibration flag | Validation residuals |
| 11 | Executable B0–B4/P1/P2 runners, BCURRENT runner, metrics, breakdowns, session bootstrap, ablation records | Frozen held-out predictions and optional detection AP |
| 12 | `QUEUE_ESTIMATOR` switch, startup validation, optional API/Influx fields, dashboard expander | A validated camera config before enabling a research mode |
| 13 | P3 is deliberately rejected for sparse observations | Ordered 1–5 FPS clips are required before temporal work |

Run `python -m pytest -q`, `python -m ruff check ...`, and `python -m mypy ...` for software
verification. Run the commands in `research/README.md` only after the corresponding real
calibration and data artifacts exist.
