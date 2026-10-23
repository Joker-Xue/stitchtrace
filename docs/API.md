# API and CLI reference (1.0)

All core functions need NumPy only and never import an inference backend. Errors raised at the
public boundary are `StitchTraceError` with a stable `code` attribute.

## Bundles

`Bundle.open(path) -> Bundle`
Verifies a schema-1 bundle: complete declared file set (no undeclared, linked or special files),
sizes and streamed SHA-256 digests, metadata types, label arrays (non-negative integers, 2D/3D, at
most 64 Mi elements), bounds, graph closure (every packed label is a node, every edge endpoint is a
node) and, for ownership records, the raw→local crop, whole-object selection and one-to-one writes
of every block. Hashes detect damage relative to the manifest; they do not authenticate a bundle
whose data and manifest were rewritten together.

`bundle.meta` (metadata dict), `bundle.array(name)` (memory-mapped declared array),
`bundle.cache_labels(name)` (raw block prediction labels).

## Replay and policies

`replay(bundle_or_path, policy="native", **params) -> (labels, measurements)`
Runs a policy on the recorded predictions and returns canonical labels (IDs numbered by first
appearance in C order) and per-object measurements. `native` must reproduce the recorded output
partition, otherwise `REPLAY_MISMATCH`. `policy` may be a registered name or a callable.

`register_policy(backend, name, fn=None)` registers `fn(bundle, **params) -> labels` for
`"ownership"` or `"graph"`; usable as a decorator. `available_policies(backend=None)` lists names.
`run_policy(bundle, policy, **params)` applies a policy without the native check.

Helpers in `stitchtrace.policies`: `block_seams(bundle)` (per axis, the first index of each next
block core), `ownership_boxes(tiles)`, `shrink_face(face, threshold)`, `face_edges(packed, seams,
threshold)`.

## Comparison and attribution

`compare(reference, candidate, iou=0.5, cover=0.2) -> dict`
Object matching at IoU ≥ `iou` (one-to-one because `iou` ≥ 0.5): `tp`, `fp`, `fn`, precision,
recall, F1, mean matched IoU. A reference object is *split* when ≥ 2 candidate objects each cover
≥ `cover` of it; a candidate object is a *merge* when it covers ≥ `cover` of each of ≥ 2 reference
objects. Lists of split, merge and unmatched IDs are included.

`attribute(reference, candidate, seams, iou=0.5, width=1) -> dict`
Locates every discrepancy relative to the block seams. Reference side (`objects`): each unmatched or
split reference object, with whether it touches a seam band of half-width `width`, how many
candidate pieces cover it and a category — `seam_split` (touches a seam and is covered by ≥ 2
pieces), `seam_other` (touches a seam, otherwise different) or `interior`. Candidate side
(`candidates`): each unmatched candidate object, as `seam_extra` (touches a seam, e.g. a fragment
cut off at a block face) or `interior_extra`. Returns the comparison, per-category counts and both
record lists.

`measurements(labels, spacing=None)` — pixel count, physical size (pixel count × product of
spacing), centroid (array-axis order, zero origin, scaled by spacing) and integer coordinate sum
per object. `canonical(labels)` — partition-preserving renumbering.

## Capture (optional dependencies)

`capture_stardist(model, image, destination, *, model_id, block_size=320, min_overlap=64,
context=96, spacing=None, allow_unverified=False)`
`image` must already be normalised by the caller. Wraps the read, predict, filter and write steps
of `predict_instances_big`; records raw block predictions (labels and polygons), the context crop,
the native selection and the written labels. Weights, configuration and thresholds are hashed into
the run contract. StarDist itself refuses geometries violating its block constraints or objects
larger than the overlap; such refusals propagate unchanged.

`capture_cellpose(image, predict_tile, destination, *, predictor_id, block_size, diameter=2,
spacing=None, allow_unverified=False)`
Runs the native `distributed_eval` (Dask threads, Zarr 2) with `predict_tile(crop, slices)` as the
block predictor. Each call receives a private copy of the crop (Cellpose normalises float32 input in
place). `diameter` sets the driver's overlap (2 × diameter). Records local and packed block labels,
the full packed field, the face-adjacency edges and the native relabeling. `predictor_id` must
change whenever the predictor's state changes.

Both captures hold a per-destination OS lock and a process-wide hook lock, write files atomically,
and commit by renaming a staging directory after the manifest is written. An interrupted capture
keeps `.NAME.work` and resumes from cached block predictions; a completed destination with the
same contract is returned without inference; a different contract raises `CONTRACT_CONFLICT`.

## CLI

```
stitchtrace verify BUNDLE
stitchtrace inspect BUNDLE
stitchtrace policies
stitchtrace replay BUNDLE [--policy NAME] [--param KEY=VALUE ...] --output FILE.npy
stitchtrace measure LABELS.npy [--spacing S ...]
stitchtrace compare REFERENCE.npy CANDIDATE.npy [--iou 0.5] [--bundle BUNDLE]
```
JSON on stdout; exit code 0 on success, 2 with a JSON error on stderr otherwise. `replay` refuses an
existing output file. `compare --bundle` adds seam attribution.
