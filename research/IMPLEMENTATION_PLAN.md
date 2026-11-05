# File-by-file implementation plan

1. `research/detection.py`: normalize current or alternative detectors behind one protocol.
2. `research/calibration/`: load, validate, compute, and apply per-camera homographies.
3. `research/geometry/`: reference points, curved centreline projection, distances, density,
   and ranks.
4. `research/features/`: typed primary feature vector and deterministic extraction.
5. `research/membership/`: interpretable B4 rules, serialized Logistic Regression/GBDT,
   and one diagnostic prediction interface.
6. `research/occlusion/`: visible-box occlusion, spacing and residual correction, clamps,
   and validation-residual intervals.
7. `research/datasets/`: annotation schema, loaders, deterministic grouped splits, and a
   validation CLI with leakage/hash checks.
8. `research/evaluation/`: named baseline registry, count/membership metrics, session
   bootstrap, ablation definitions, artifact metadata, and evaluator CLI.
9. `research/training/`: manifest-only membership and residual-model training commands.
10. `research/estimator.py` and `inference/pipeline.py`: strategy dispatch that defaults to
    BCURRENT and preserves the API contract.
11. `api/models.py`, storage, and dashboard: optional research diagnostics end to end.
12. `tests/research/`: synthetic geometry, datasets, correction, metrics, serialization,
    strategy, and API compatibility coverage; existing tests remain unchanged where possible.
