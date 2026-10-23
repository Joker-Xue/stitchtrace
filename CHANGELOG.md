# Changelog

## 1.0.0 (2026-10-24)
First public release, accompanying the SoftwareX submission. The experiment bundles archived at
https://doi.org/10.5281/zenodo.23599583 were captured with the pre-release `1.0.0.dev0`, whose code
differs from 1.0.0 only in the version string; their evaluations were recomputed with this code.

- Policy interface: `register_policy`, `available_policies`, `run_policy`; `replay` accepts any
  registered name or callable plus numeric parameters.
- New ownership policies `crop`, `iou_merge`, `center`; new graph policy `relink` with a
  dimension-correct face shrink (repairs unlinked 2D seams of Cellpose 4.2.1.1, cf.
  MouseLand/cellpose#1364 and #1480).
- `compare` (object matching, splits, merges) and `attribute` (seam attribution), NumPy only.
- CLI: `policies`, `compare`, `replay --param`.
- Cellpose capture passes a copy of every crop to the predictor; Cellpose normalises float32 input
  in place, which previously could rewrite the caller's image and neighbouring crops. Temporary
  Zarr and Dask products are no longer committed into the bundle.
- StarDist capture reports any corrupt cache archive as `CACHE_CORRUPT`.
- Vectorised `measurements` and `canonical`; streamed hashing; parent-directory fsync after commit;
  `replay` no longer re-verifies an already opened bundle.
- Compatibility checks can record instead of refuse an unreviewed upstream (`allow_unverified`).
- Historical 0.1.x bundles open and replay unchanged.

## 0.1.2 and earlier
Private research candidates; see [docs/history](docs/history).
