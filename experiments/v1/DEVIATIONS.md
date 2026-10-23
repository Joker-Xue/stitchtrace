# Deviations from PROTOCOL.md

## D1 (2026-10-23, before the formal run): crops are passed to predictors as copies
A one-image smoke run gave unstable Cellpose results: the same full image produced 153 objects or
109 objects depending on what had been predicted before, and a plain run of the distributed driver
did not reproduce the capture. Diagnosis (both MPS and CPU, isolated in fresh processes):
`CellposeModel.eval` 4.2.1.1 normalises float32 input arrays in place. The harness passed views of
one shared image, so every crop prediction rewrote that image region and the overlapping regions of
neighbouring crops. An initial hypothesis of thread-unsafe MPS inference was wrong and its
workaround was removed.

Changes: `capture_cellpose` now passes a private copy of each crop to the predictor (regression
test `tests/test_cellpose_capture.py` fails without it); the experiment's predictor copies its input.
Serving the recorded tile predictions of a capture to the unmodified driver reproduced the captured
output pixel for pixel, so the instrumentation itself was not at fault. All smoke-run outputs were
deleted; no formal result predates this change.

## D2 (2026-10-23, during the Cellpose arm): concurrent test run
The unit test suite (~1 min, CPU) was run once while image 3 of the Cellpose arm was being
processed. Wall-clock times recorded for that image may be inflated; segmentation outputs are
unaffected. Timing summaries are reported with and without that image.

## D3 (2026-10-23, StarDist arm): empty-block relabeling bug, arm rerun
The StarDist arm stopped on its second image: `retain_all` relabeled a block that contained no
object, and `_relabel` indexed an empty lookup table. Fixed with a regression test
(`test_policies_with_an_empty_tile`). The whole StarDist arm (including its first image) was
deleted and rerun with the fixed code. The Cellpose arm is kept: its graph policies never call the
ownership relabeling path, and all 50 images completed (log tracebacks are Dask
`CommClosedError`/`StreamClosedError` messages at cluster shutdown). Code versions:
`runs/v1/CODE_COMMIT` (Cellpose arm) and `runs/v1/CODE_COMMIT_STARDIST` (StarDist arm).

## S0 (2026-10-23, before PROTOCOL-3D.md was fixed): timing probe on a test volume
Before the 3D protocol was written, a timing probe ran untiled Cellpose-SAM 3D inference on test
volume `mir61L1_1228062` with `diameter=None` and `diameter=20`, and printed F1 against its
annotation. Only the wall-clock time of that probe is used (to size the experiment). The
`diameter` choice is made on the validation split alone, as PROTOCOL-3D.md specifies, and the
probe's accuracy numbers are not used for any decision.

## S1 (2026-10-23, before the formal 3D run): scope and diameter
The timing probe took 1,361 s for one untiled volume (while two unrelated jobs from another
project used ~260% CPU), implying ~19 h for the full protocol. Changes made before any formal 3D
result: (1) `diameter=None` (the Cellpose-SAM default) instead of a validation-tuned value, so no
data are used to choose it and the validation smoke run is dropped; (2) the chain blocking runs on
all 7 test volumes, the grid blocking on the first 2 volumes in file-name order only. A load log
(`runs/v1/cellpose3d/load.log`, `uptime` every 60 s) is kept because the workstation was shared;
3D timings are reported as measured under that load.

## D4 (2026-10-23, found by the 3D smoke test): relink shrinks each block face separately
The oracle smoke test of the 3D arm showed that `relink` (then shrinking whole seam planes) linked
differently from the native driver on 3D block grids (129 vs 139 edges), whereas both agree on a
single chain of blocks. The native driver shrinks each block's own face patch, so pixels at patch
borders are treated differently. `face_edges` now shrinks per block face; a regression test checks
that relink merges exactly the pieces the upstream face graph merges on random 3D grids (it fails
for the old implementation). Edges are compared as the groups they induce, because the upstream
grouping records each face component as a chain of labels.

Consequence for the 2D Cellpose arm and A1: their `relink` numbers were computed with the old
implementation. All policies were re-evaluated from the saved bundles and untiled references with
the corrected code, without any new inference (`reevaluate.py`); inference and capture timings
are unchanged, replay timings are re-measured. Code version: `runs/v1/CODE_COMMIT_REEVAL`.
For A1 the untiled references had not been saved; they were regenerated (deterministic inference,
same code path) and all 50 matched the originally recorded untiled-vs-annotation metrics exactly
before re-evaluation. `run_cellpose_globalnorm.py` now saves them.

## S2 (2026-10-23, during the 3D run): short concurrent test runs
While the 3D arm ran, the unit tests (about 10 s and 65 s of CPU, at ~07:45 and ~08:10) and small
NumPy-only environment checks were run on the same machine, on top of the unrelated external load
recorded in `load.log`. 3D wall-clock times are therefore reported only as measured under shared
load; segmentation outputs are unaffected.

## D5 (2026-10-23, after the 3D arm): attribution of extra objects; all arms re-evaluated
The 3D grid results showed many unmatched *candidate* objects (fragments at seams), which the
reference-side attribution did not count. `attribute` now also classifies unmatched candidate
objects (`seam_extra`, `interior_extra`); a test covers it. The exploratory policy
`relink_absorb` (Addendum A2) was added. Every saved bundle of every arm was re-evaluated with this
code, without inference (`reevaluate.py D5`); inference and capture timings are unchanged.
Previous result files are kept as `results_before_D5`. Code: `runs/v1/CODE_COMMIT_D5`.
