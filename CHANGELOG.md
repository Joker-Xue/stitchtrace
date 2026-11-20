# Changelog

## 1.0.1 (2026-11-20)
Bug fix and documentation corrections; bundles, schema, policies and replay are unchanged.

- `compare`: matching is now strictly one to one. In 1.0.0 every pair with IoU >= `iou` was accepted,
  so at an exact tie at IoU 0.5 (one object formed by two equal halves of an object on the other side)
  both pairs were counted: e.g. tp=2, fp=-1, F1=1.33 for one candidate covering two equal reference
  objects, and `mean_matched_iou` averaged over both pairs when two candidates halve one reference object.
  Pairs are now accepted in the order IoU descending, reference id, candidate id (new helper
  `stitchtrace.compare.match`); this keeps one pair per tie and is a maximum-cardinality matching.
  Without exact ties the results are identical to 1.0.0. `attribute` inherits the fix.
  A re-evaluation of all 55,837 comparisons of the archived and of the external experiments found 60 with an
  exact tie conflict (two candidates halving one reference object); in all of them only `mean_matched_iou`
  changed (by at most 0.025); counts, F1, split/merge and attribution were unchanged.
- Regression tests for ties (`tests/test_matching_ties.py`).
- Documentation: the 2D Cellpose face-shrink defect is referenced to MouseLand/cellpose#1480 only;
  #1364 (a report on a slurm branch that could not be reproduced) was cited in error in 1.0.0
  (README, CHANGELOG, `policies.shrink_face` docstring). README lists `relink_absorb`. API.md states the
  tie rule. The relink F1 values of `experiments/v1/ADDENDUM-A1-RESULT.md` are corrected to the final
  re-evaluation (0.9895 and 0.9893 instead of 0.9891 and 0.9888).
- The 1.0.0 entry below mentions a SoftwareX submission; that submission was withdrawn. The 1.0.0 release
  date is 2026-10-24 in UTC+8 (2026-10-23 UTC on Zenodo).

## 1.0.0 (2026-10-24)
First public release, accompanying the SoftwareX submission. The experiment bundles archived at
https://doi.org/10.5281/zenodo.23599583 were captured with the pre-release `1.0.0.dev0`, whose code
differs from 1.0.0 only in the version string; their evaluations were recomputed with this code.

- Policy interface: `register_policy`, `available_policies`, `run_policy`; `replay` accepts any
  registered name or callable plus numeric parameters.
- New ownership policies `crop`, `iou_merge`, `center`; new graph policy `relink` with a
  dimension-correct face shrink (repairs unlinked 2D seams of Cellpose 4.2.1.1, cf.
  MouseLand/cellpose#1480; 1.0.0 also cited #1364 in error).
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
