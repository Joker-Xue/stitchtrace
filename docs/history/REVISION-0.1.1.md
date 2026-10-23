# 0.1.1 audit revision

This separate working copy addresses independent audit M1–M4. The 0.1.0 archive and independent audit directory were not edited. Historical experimental claims in copied documents remain 0.1.0 evidence; this revision adds no scientific data, models or novelty claims.

- M1: validate the root manifest path before opening it; share a no-symlink/no-special-file inventory between writer and reader; exclude only the exact root manifest, including nested manifest.json files in hashes.
- M2: require in-image read/write bounds, write containment, raw-label extent and exact crop equality, and whole-local-object keep/drop decisions. NPZ caches use the labels member; NPY label-only fixtures are accepted. This validates recorded internal consistency, not the scientific correctness of the native ownership predicate.
- M3: validate manifest, inventory records, metadata and tile records before field access. Invalid documents raise coded StitchTraceError; CLI emits JSON with exit 2.
- M4: reproduction requires an explicit cache directory. A temporary environment installs the complete lock offline before any reproduction directory is created. A separate preparation command requires --download to access official PyPI. No inherited developer-cache path is silently selected.

## Evidence
results/revision/red-tests.log records 15 failures and one pass against unchanged 0.1.0 source before fixes.
results/revision/green-tests.log records the expanded suite, including special files, nested JSON types and empty-cache failure before output creation.
results/revision/build.log records the 0.1.1 wheel build.
results/revision/full-preflight.log records full-lock offline installation into a temporary fresh environment using an explicitly named existing cache; this is not evidence of a cold full dependency download.
results/revision/wheel-compatibility.json, when present, records 15 previous bundles replayed from the installed wheel in a minimal environment without inference packages.
The previous-release replay is read-only and compares canonical labels against recorded expected output. It does not rerun model inference or independently reconfirm every alternative-policy metric.

## Cold setup / migration
Use macOS ARM64 with Python 3.11 for the pinned full reproduction. Install Python 3.11 and uv explicitly before these commands; Python auto-download is disabled by preparation. Run from this project directory:

```sh
python3.11 experiments/prepare_dependencies.py --cache-dir "$PWD/dependency-cache" --download
python3.11 experiments/reproduce.py --cache-dir "$PWD/dependency-cache" --preflight-only
python3.11 experiments/reproduce.py --cache-dir "$PWD/dependency-cache" --name audit-rerun
```

For only wheel-based reading/replay, prepare the NumPy lock with --core:
```sh
python3.11 experiments/prepare_dependencies.py --cache-dir "$PWD/core-cache" --core --download
python3.11 experiments/prepare_dependencies.py --cache-dir "$PWD/core-cache" --core
UV_CACHE_DIR="$PWD/core-cache" uv venv --python python3.11 .reader-venv
UV_CACHE_DIR="$PWD/core-cache" uv pip install --offline --python .reader-venv/bin/python dist/stitchtrace-0.1.1-py3-none-any.whl
```

A project archive alone does not include dependency wheels or Python. To move offline, explicitly transfer the prepared uv cache and compatible Python/uv runtimes to the same OS/architecture/Python version, then rerun offline preflight at the new cache path. Changing platform requires preparing compatible dependencies again. Full inference reproduction additionally requires the shipped selected fixtures and model files; no new data or model downloads are performed by these commands.

## Validation outcome (2026-10-19)
- Full suite: 55 passed (11.79 seconds).
- Installed wheel in a fresh NumPy-only environment: all 15 previous-release bundles passed.
- Full-lock explicit existing-cache preflight: passed.
- Empty explicit cache: coded DEPENDENCY_PREFLIGHT failure, no reproduction output created.
- Cold core download: not completed. Default sandbox DNS blocked PyPI; two controlled-network approval attempts timed out before execution. No successful cold-network download is claimed. The preparation commands and offline failure behavior are implemented, but a connected host must still execute the documented cold-download check.
- New wheel filename: stitchtrace-0.1.1-py3-none-any.whl.
- No new inference experiments, scientific data or model downloads, publication, license decisions or submission.
