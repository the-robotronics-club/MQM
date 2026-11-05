# Experiment output contract

Each run creates a new `results/<timestamp>-<method>-<config-hash>/` directory containing
`config.yaml`, `metrics.json`, `predictions.csv`, `per_session.csv`, and `metadata.json`.
Metadata records the commit, UTC timestamp, split version, model path and SHA-256, feature
and calibration versions, random seed, configuration, and Python/package versions. Existing
run directories are never overwritten. Metrics are computed only from supplied predictions;
the writer never inserts placeholder accuracy values.

Prediction CSV files must identify `image_id`, `session_id`, `queue_ground_truth`,
`queue_prediction`, `dataset_version`, and `split`. Optional columns enable additional
reports: crowd/occlusion/meal/perspective/camera/day breakdowns, JSON arrays named
`membership_truth` and `membership_prediction`, `detection_ap50`,
`detection_map50_95`, and `latency_ms`. Bootstrap intervals resample complete sessions.

Use repeated `--ablation KEY=VALUE` arguments on the evaluator to record the controlled
dimension in `config.yaml`. Train membership artifacts with `--features` to remove local
density or occlusion, or to select `x_image y_image` for the raw-coordinate comparison.
The evaluator records results but never treats a configuration label as evidence that an
ablation was actually run; the prediction-generating command remains authoritative.
