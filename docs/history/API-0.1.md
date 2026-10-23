# Python API / CLI contract (0.1)

`Bundle.open(path) -> Bundle`: verify schema1, complete declared file set, SHA256/size, supported backend, numeric labels, coordinates, spacing, graph endpoint/full-node dependency consistency and tile selection consistency. No inference imports. Failures use `StitchTraceError.code`; malformed bundles are normalized to coded errors. Hashes detect accidental damage, not a malicious rewrite of both data and manifest.

`replay(bundle_or_path, policy="native") -> (ndarray, dict)`: re-open/reverify before use; native checks canonical partition against captured expected output. Graph IDs may be renumbered; use partition/object measurements rather than raw label values across backends. Ownership preserves recorded writes in original order. `disable_edges` only for graph, `retain_all` only for ownership. Unknown policies fail. Alternative order can matter: no claim arbitrary overlapping last-writer operations commute.

`measurements(labels, spacing=None)`: nonnegative integer 2D/3D input, zero background. Returns pixels, physical_size, centroid and integer coordinate_sum by instance ID. Coordinates follow array axes, pixel centres at integer indices, zero origin, no affine transform. `physical_size` is merely spacing product times pixel count; using unit spacing with uncalibrated images does not establish physical calibration.

`capture_stardist(model, image, destination, *, model_id, block_size=320, min_overlap=64, context=96, spacing=None)`: caller-normalized image; internal model method unchanged, weights/config/thresholds fingerprinted. Supports tested official 2D model; 3D evidence is labeloracle only. Valid geometric inequalities do not guarantee native Block.cover accepts a shape: original BBBC039 block320 case demonstrates this. No silent parameter adjustment. Use512 for the documented supplemental configuration. Caller's custom normalization/predictor modifications outside recorded inputs are not automatically understood.

`capture_cellpose(image, predict_tile, destination, *, predictor_id, block_size, diameter=2, spacing=None)`: explicit callable takes crop ndarray and crop slices, returns local instance labels; native distributed driver processes/stitches. Caller must identify external predictor state in predictor_id. Actual experiments use label oracles; no tested full neural Cellpose inference claim. Driver uses real Dask threads and Zarr2, not fake futures.

Both capture calls serialize internal hooks with a shared process lock, restore patched symbols after exceptions and hold a per-destination OS lock. Different destinations can run in separate processes; same-process calls serialize. This is not general thread isolation from unrelated external code that monkeypatches the same library or changes cwd. Cellpose temporarily changes cwd under the shared lock; arbitrary host-application threads remain outside this guarantee. No NFS/multi-host transaction guarantee.

Completed same-contract destination returns its verified bundle without inference; changed contract is refused. Interrupted staging is retained as `.NAME.work`; exact-key complete tile cache is reused. OS file lock releases after process death. Data writes use same-filesystem atomic replacement and fsync; directory commit uses rename. This is tested process-failure recovery, not certified crash/power-loss durability across filesystems. Corrupt cache is rejected instead of automatically regenerated; preserve/quarantine it and use a new destination after review.

CLI: `verify`, `inspect`, `replay --policy --output`, `measure --spacing`. Exit0 success, exit2 coded invalid input/errors. `verify` checks integrity/structural semantics; `replay` additionally checks native partition. Use distinct output file paths for concurrent CLI calls. Capture is Python API, not a CLI model-download wrapper.

## Supported native versions

- Cellpose4.2.1.1, package distributed_segmentation source/signatures fixed.
- StarDist0.9.2, relevant big/matching/models.base source/signatures fixed.
- TensorFlow2.21.0 for recorded model experiments; Zarr2.18.7 for native Cellpose.
- Python3.11.15/macOS ARM64 tested. Core dependency NumPy2.4.6 tested; declared supported range is provisional pending broader CI.

Compatibility definitions are shipped in `backends/compatibility.json`; no automatic regenerate-on-failure command. A maintainer must audit changed upstream code and rerun equivalence tests before adding support. Capture records the implementation file hashes used for that run.
