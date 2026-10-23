# v1 scale experiment protocol: 3D volumes with long block chains (fixed before the formal run)

Written 2026-10-23 after the 2D arms and before any formal 3D run. A smoke run on one *validation*
volume (timing and the `diameter` choice) precedes it; no test volume is used to choose anything.
Deviations go to `DEVIATIONS.md` with prefix S.

## Data
C. elegans L1 nuclei (Long et al. 2009; masks curated by Kainmueller; Hirsch & Kainmueller 2020),
Zenodo record 5942575, CC-BY-4.0, archive SHA-256
1def07491cdad89e381cbe4437ef03da3af8f78d127e8152cd9b32bdab152c4e. All 7 volumes of the published
`test` split (shape 140 × 140 × 994–1240, Z Y X; 524–547 annotated nuclei each). The `val` split
is used only for the smoke run.

## Model
Cellpose 4.2.1.1, `cpsam_v2`, `do_3D=True`, `z_axis=0`, `normalize={'percentile': [1, 99.8]}`,
`diameter` fixed from the validation smoke run, MPS. Every predictor call receives a copy.

## Runs per volume
1. Untiled 3D inference (reference).
2. Cellpose distributed capture for two block shapes (Z, Y, X):
   - chain: (140, 140, 128) — one row of 8–10 blocks along X, seams on X only;
   - grid: (70, 70, 128) — 2 × 2 × 8–10 blocks, seams on all three axes.
   Distributed overlap = 2 × 20 px (`diameter=20` passed to the driver only).

## Policies replayed
native, disable_edges, relink (threshold 1.0).

## Measurements
As in PROTOCOL.md (compare and attribute vs untiled and vs annotation; capture/replay times;
bundle size), plus **edge-set agreement**: the set of edges recomputed by `relink` versus the edges
recorded from the native driver. In 3D the upstream face shrink is per pixel, so the prediction is
identical edge sets and identical outputs for `native` and `relink`.

## Primary outcomes (stated in advance)
- S1: edge-set agreement and native/relink output identity in 3D.
- S2: F1 and split/merge counts of native assembly vs untiled inference for chain and grid blocks;
  attribution of the differences.
- S3: replay time vs capture and untiled inference time at volume scale; bundle size.
