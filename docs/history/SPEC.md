# Engineering specification 0.1 — frozen before implementation

Private research candidate: stitchtrace (PyPI404 and public GitHub name search0, 2026-10-19; not trademark clearance). No public license granted. Preserve pilots.

Core Python>=3.11, NumPy only; APIs Bundle.open, replay(policy), measurements(spacing), verify, capture_stardist(model,image,destination,...), capture_cellpose(image,predict_tile,destination,...). CLI verify/replay/measure/inspect. Native capture optional extras fixed to StarDist0.9.2 / Cellpose4.2.1.1 with checked source signatures/hash. Internal hook serializes same-process calls, restores on failure; separate processes isolated. Unsupported versions fail closed.

Bundle schema1: backend discriminant ownership/graph, run contract hash, axes/shape/spacing, complete dependency inventory, SHA256+size per file, runtime source identity. NPY/JSON only, no pickle/executable artifacts/weights. Ownership stores actual context-cropped prediction, actual filter result and actual write label mapping, read/write bounds, kept/rejected local IDs, order. Graph stores full packed core, full observed graph, used nodes and original output; closure not cropped. Capture expected original output; replay native partition/measurements validate. Alternative policies retain_all (ownership, last nonzero write), disable_edges (graph), never accuracy claims.

Writes: destination-parent staging folder and single-writer file lock; atomic file replacements+fsync; manifest commit last then directory rename. Same contract completed run idempotent, different contract conflict. Interrupted staging persists; exact per-block cache keys include run contract and crop; invalid bytes reject, no silent reuse. No stale process lock deletion; OS lock release on process exit. Within-process hooks protected by a backend-wide lock. Local POSIX filesystem supported; no distributed/NFS guarantee.

Safety: no symlink paths/files, absolute/traversal filenames, duplicate manifest keys, unknown schema/backend, out-of-bound crops, noninteger/negative labels, invalid coordinates, missing required dependencies, content mismatch. Fail closed. Content hashes integrity not authenticity; malicious rewrites of all evidence not prevented. Limits validated before loading; no extraction of untrusted archives.

Tests: 2D/3D/core anisotropic measurement, partition relabel/order (only independent writes; overlapping intervention order explicitly semantic), clean/no-seam controls, invalid/corrupt/path input, transitive graph closure, failure resume/cache invalidation/idempotency/concurrent locking, real installed adapter equivalence, new-env inference/network-denied replay.

No segmentation algorithm innovation, automatic error truth, human-user benefit assertion. Report actual supported behavior and unmet items, never equate scaffolding/tests to all requirements completed.
