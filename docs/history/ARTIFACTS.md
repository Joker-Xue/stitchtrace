# Artifact map

- `src/`, `pyproject.toml`, `dist/`: installable implementation/wheel.
- `tests/`: automated contract, transaction, failure and compatibility tests.
- `experiments/PLAN.md`, `AMENDMENT-01.md`: frozen original plan and explicit post-failure engineering regression.
- `experiments/handwritten.py`: independent full baseline; never consumes tool bundles.
- `results/finite/`: all12 originally planned failures, no replacements.
- `results/engineering512/*/tool-final/`: final real-image bundles; baseline/native and earlier tool runs retained beside them.
- `results/oracles/`:8 Cellpose labeloracle pairs plus order tests; `results/sd3d-oracle/`:3 StarDist3D oracle permutations.
- `results/recovery/`: actual model interruption/resume/idempotency/concurrent-call evidence.
- `results/*.log`, `*-summary.json`, `offline-audit.json`: raw tests/measurements/audits.
- `research/rerun-evidence/`: compact copied logs and JSON from complete reruns; not runnable bundles. Full duplicate rerun trees remain in execution workspace and are excluded from the delivery copy to avoid duplicating environments/data/bundles.
- `data/`: official downloaded archives and only4 selected/decoded pairs, plus the earlier official small development fixture for engineering recovery tests.
- `models/`: previously downloaded official small model, separately attributed; never read by offline audit.
- `research/source-snapshots/capture-v1`: exact code used for recorded final captures before the last two reader-only validation additions.

Delivery excludes virtual environments, uv/package caches, Python bytecode, font/Numba caches, build intermediates and duplicate full reproduction trees. These are not counted as research evidence. Existing pilots and full workspace reruns are retained, not deleted.
