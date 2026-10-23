# Addendum A1 result (exploratory, post hoc)

Run: all 50 test images, block 256, one global percentile normalisation (1–99.8) with
`normalize=False` for both the untiled and the distributed predictor. Code: `runs/v1/CODE_COMMIT_A1`.

| normalisation | policy | F1 vs untiled | split | seam_split | seam_other | interior |
|---|---|---|---|---|---|---|
| per crop (main arm) | native | 0.9019 | 546 | 533 | 2 | 28 |
| per crop (main arm) | relink | 0.9891 | 15 | 2 | 2 | 28 |
| global | native | 0.9013 | 548 | 533 | 2 | 30 |
| global | relink | 0.9888 | 17 | 2 | 2 | 30 |

**The hypothesis is rejected.** Interior discrepancies did not fall (28 → 30); the first five
images happened to show none, which is why the early impression was misleading. Per-crop
normalisation is therefore not their source. The seam result is unchanged under both
normalisations (533 seam splits under `native`, 2 after `relink`).

Characterisation of the 28 interior objects of the main arm (block 256; 0.5% of the 5,474 objects
in the 50 untiled references): 16 lie within 60 px (the driver's context margin) of a seam; 13 are
matched but split into two pieces, 9 are covered but differ in outline enough to be unmatched, and
6 (15–189 px) are essentially absent from the block-wise result. They are prediction-level differences between block and whole-image
inference; their exact mechanism is not established here. No assembly rule can remove them, and
the attribution keeps them apart from seam-linking errors.
