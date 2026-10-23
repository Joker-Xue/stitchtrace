# v1 formal experiment protocol (fixed before running)

Written 2026-10-23 before any formal run. Results from the pilot (`investigations/`) motivated the
design but are not reused. Any deviation is recorded in `DEVIATIONS.md` with its reason.

## Data
BBBC039 v1, all 50 images of the official `test.txt` split (no selection, no replacement).
Reference annotation: `skimage.measure.label(mask[..., 0])` (default full connectivity), as in the
dataset's colour-coded masks. Images are used as stored (uint16).

## Models and preprocessing
- StarDist 0.9.2, `2D_versatile_fluo` weights (local copy), percentile normalisation 1–99.8 over the
  whole image, default thresholds.
- Cellpose 4.2.1.1, `cpsam_v2` weights, `normalize={'percentile': [1, 99.8]}`, `diameter=None`, MPS.
  The distributed driver receives `diameter=30` only to set its overlap (2 × 30 = 60 px).

## Runs per image
1. Untiled inference with each model (the stitching-free reference). Cellpose untiled is run twice
   on the first 10 images to measure run-to-run nondeterminism.
2. StarDist capture for each configuration (block_size, min_overlap), context 96:
   (320,48), (352,32), (352,48), (352,64), (384,48), (384,64). A configuration that StarDist refuses
   (assertion or object-size RuntimeError) is recorded as a refusal, not retried.
   For every successful capture also run plain `predict_instances_big` and require pixel equality
   with the captured output (instrumentation fidelity).
3. Cellpose capture for block_size 128, 192, 256 (square). For the first 10 images at block 256 also
   run plain `distributed_eval` and require pixel equality with the captured output.

## Policies replayed (no inference)
- ownership: native, retain_all, crop, iou_merge (threshold 0.5), center
- graph: native, disable_edges, relink (threshold 1.0)
Each replay is timed 5 times; the median is reported.

## Measurements
For every policy output: `stitchtrace.compare` against (a) the untiled prediction of the same model
and (b) the reference annotation; IoU 0.5; split/merge cover 0.2. `stitchtrace.attribute` against the
untiled prediction with the bundle's seams (half-width 1 px). Bundle size on disk. Wall-clock times
of untiled inference, capture and replay.

## Primary outcomes (stated in advance)
- O1: fraction of StarDist captures whose native output equals untiled inference (object level).
- O2: Cellpose native vs relink F1 and split count against untiled inference, per block size.
- O3: share of Cellpose native discrepancies classified `seam_split`, and how many relink removes.
- O4: replay time per policy vs capture/inference time.
Ground-truth metrics are secondary: they describe model quality, not stitching.

## Addendum A1 (2026-10-23, written after seeing the Cellpose arm, exploratory)
Observation: after `relink`, 20–28 discrepant objects per block size remain and are classified
`interior` (away from seams); they are equally present under `native`. Hypothesis: they come from
per-crop percentile normalisation in Cellpose, not from assembly. Test, labelled post hoc in any
report: for all 50 test images at block 256, normalise the whole image once (percentiles 1–99.8),
pass the result with `normalize=False` to both the untiled and the distributed predictor, capture,
and replay `native` and `relink`. Prediction: `interior` discrepancies under `relink` fall to near
zero while `seam_split` under `native` stays of the same order.

## Addendum A2 (2026-10-23, written after seeing the 3D arm, exploratory)
Observation: on 3D block grids the native (and relink) assembly leaves dozens of unmatched
candidate objects per volume; nearly all touch a seam and their median size is 9–14 voxels
(nuclei ~900 voxels). They are fragments cut off at block faces whose contact is removed by the face
shrink. Rule tested, labelled post hoc in any report: policy `relink_absorb` = `relink`, then every
linked piece smaller than `min_size` voxels that touches a seam is merged into the piece across the
seam with which it shares the largest raw face contact, or removed if it has none. `min_size` is
fixed to 15, Cellpose's own default minimum object size, and is not tuned. It is evaluated on every
saved Cellpose bundle (2D: 3 block sizes × 50 images; 3D: 7 chain + 2 grid volumes) without
inference. Prediction: `seam_extra` falls sharply on 3D grids, F1 rises, and merges do not increase.
