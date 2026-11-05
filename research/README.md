# MQM research extension

This package implements the paper-oriented path alongside the deployed estimator. It does
not contain paper results or claim that untrained methods improve production.

The default application remains `QUEUE_ESTIMATOR=production` (**BCURRENT**). Research modes
are `geometric`, `membership`, and `occlusion`. They require a validated camera file under
`research/configs/cameras/`; learned modes additionally require model artifacts.

No additional pretrained queue model is required. B1 uses the existing pretrained YOLOv8
person detector. B2 onward can use locally fine-tuned YOLO weights supplied with
`--detector-weights`. Membership and residual correction artifacts are trained from the
mess dataset. If a membership artifact is absent, the configured geometric fallback is
named in diagnostics. If spacing or residual artifacts are absent, hidden correction stays
at zero and the method is explicitly labelled as a fallback.

## Commands

First copy the camera template, replace every correspondence/path value with surveyed data,
set `calibration_status: validated`, and give it a real calibration version:

```bash
cp research/configs/cameras/mess_main.example.yaml \
  research/configs/cameras/mess_main.yaml
python -c "from research.calibration import load_calibration; print(load_calibration('mess_main'))"
```

Do not enable a research estimator with the template geometry.

```bash
python -m research.datasets.validation --manifest data/research/manifest.json \
  --root data/research \
  --splits data/research/splits
python -m research.datasets.splits --manifest data/research/manifest.json \
  --output data/research/splits --seed 42
python -m research.datasets.build_features --manifest data/research/manifest.json \
  --split-file data/research/splits/train.json --root data/research \
  --detector-weights models/yolo-mess-best.pt \
  --output data/research/train-features.json
python -m research.training.train_membership --manifest data/research/train-features.json \
  --split-file data/research/splits/train.json \
  --method logistic --output models/research/membership-logistic.joblib
python -m research.training.train_occlusion --manifest data/research/train-frames.json \
  --split-file data/research/splits/train.json \
  --output models/research/occlusion-residual.joblib
python -m research.training.fit_spacing --manifest data/research/train-spacing.json \
  --split-file data/research/splits/train.json \
  --output models/research/spacing.json
python -m research.training.calibrate_uncertainty \
  --predictions results/input/p2-validation.csv \
  --split-file data/research/splits/val.json \
  --calibration-version mess-main-v1 --output models/research/uncertainty.json
python -m research.evaluation.run_baseline --manifest data/research/manifest.json \
  --split-file data/research/splits/test.json --root data/research --method b0 \
  --output results/input/b0-predictions.csv
python -m research.evaluation.evaluate --predictions results/input/predictions.csv \
  --method p1 --split test --dataset-version v1 --output results
```

Fine-tuned and ablation examples:

```bash
python -m research.evaluation.run_baseline --manifest data/research/manifest.json \
  --split-file data/research/splits/test.json --root data/research --method b4 \
  --detector-weights models/yolo-mess-best.pt --inference-resolution 1280 \
  --output results/input/b4-predictions.csv

python -m research.training.train_membership \
  --manifest data/research/train-features.json --method logistic \
  --split-file data/research/splits/train.json \
  --features x_ground y_ground distance_to_counter distance_to_path \
    rank_normalized detector_confidence \
  --output models/research/membership-logistic-no-density-occlusion.joblib

python -m research.evaluation.evaluate \
  --predictions results/input/p1-no-density.csv --method p1 --split test \
  --dataset-version v1 --model-path models/research/membership-logistic-no-density-occlusion.joblib \
  --calibration-version mess-main-v1 --ablation local_density=false \
  --ablation occlusion_feature=false --output results
```

Dataset and artifact formats are documented in `datasets/README.md` and
`experiments/README.md`. Camera ground units are arbitrary unless the correspondence survey
explicitly records metric units. The checked-in camera file is a schema template and is not
enabled calibration.
