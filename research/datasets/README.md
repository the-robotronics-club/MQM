# Dataset schema and protocol

The JSON manifest contains `version` and `images`. Each image record requires `image_id`,
`path`, `camera_id`, `session_id`, optional `clip_id`, ISO timestamp, `meal_period`,
`crowd_band`, non-negative `queue_ground_truth`, and `annotations`. Each annotation requires
`bbox`, `role`, `occlusion`, `truncation`, and optional `track_id`.

Roles are `queue_member`, `passerby`, `staff`, `other_nonqueue`, or `uncertain`. Occlusion is
`none`, `partial`, or `heavy`. Uncertain annotations are excluded from primary supervised
training. Splits group complete sessions. Generated test manifests are frozen: the split
writer refuses to overwrite any existing split file.

The validator checks files, image readability, box bounds, labels, non-negative ground
truth, duplicate IDs, cross-split image/session leakage, version mismatch, duplicate image
hashes, and visible-label/ground-truth inconsistencies.
Byte-identical content across splits is an error even when image IDs and session
names differ. Frames with the same camera ID and clip ID must stay in one split.
Duplicates within one split remain warnings. Re-encoded, cropped, or neighboring
images still require accurate source/session metadata; byte hashes alone cannot
detect those relationships.

Membership training consumes a derived JSON feature manifest. It must declare
`dataset_version`, `split`, `feature_version`, and `people`. Every person row contains an
explicit `role` plus the named features in `research.features.schemas`. The training CLI
rejects `split: test`, excludes `uncertain`, and records the selected feature family in the
model artifact. This also permits controlled raw-image/ground-coordinate and
density/occlusion ablations without changing model code.

Every derived person or frame row also carries `image_id` and, when available,
`session_id`. Training commands require the frozen train split file and verify this lineage;
uncertainty calibration requires the frozen validation split. A split label inside a
derived file is never accepted as the only leakage control.

Spacing fitting consumes a derived frame manifest with `dataset_version`, `split`,
`calibration_version`, `coordinate_units`, and `frames`. Each frame has an
`occlusion_band` and ordered or unordered `queue_path_positions`. Only `none` occlusion
frames contribute consecutive gaps. Residual-model manifests contain frame-level
`queue_ground_truth`, `visible_count`, and the feature fields declared by
`research.training.train_occlusion.RESIDUAL_FEATURES`. Both trainers reject the test split.

Derived manifests must be generated from the same validated source manifest and frozen
split. They are data artifacts and are not synthesized by this repository because matching
detector boxes to person annotations requires the real annotated frames.
