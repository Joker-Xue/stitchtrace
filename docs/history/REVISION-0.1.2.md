# 0.1.2 reliability revision

2026-10-19. This separate revision addresses R1/R2 and README R3 from the independent 0.1.1 review. Version 0.1.1 and the independent audit tree are preserved. No scientific inputs, models, intervention policies or real-inference experiments were added.

## Changes

- **R1, inaccessible inventory subtrees:** directory traversal now supplies an error callback. Permission errors and traversal I/O errors produce INVENTORY_UNREADABLE rather than silently treating an inaccessible subtree as empty. Links and special files remain rejected. Valid empty directories remain accepted.
- **R2, invalid NPZ/ZIP archives:** array parsing boundaries explicitly convert ZIP CRC, truncation, encryption, unsupported compression and related NumPy/decompression errors into coded errors. Ownership raw-cache failures return INVALID_CACHE; reusable capture-cache failures return CACHE_CORRUPT. CLI verification returns JSON on stderr and exit 2. Hash checks remain unchanged; tests reseal the outer manifest to exercise internal archive parsing.
- **R3, current documentation:** README installs the 0.1.2 wheel, takes an explicit previous-release bundle path and explicit dependency cache, and creates a fresh environment/output directory. Old instructions were moved to HISTORICAL-README-0.1.1.md. The entire current quickstart was extracted from README and executed verbatim.
- **R4 semantic scope:** no new native-label mapping rule was introduced. Current checks establish raw/local/whole-object internal consistency and replay the recorded writes; they do not independently prove all native ownership/relabeling rules. README explicitly narrows this claim.

## Verification

All evidence below is in results/revision-0.1.2/.

| Evidence | Observed result |
| --- | --- |
| red-tests.log | Against unchanged 0.1.1 source: 8 failures, 3 passes |
| green-tests.log | 0.1.2 source suite: 66 passed, 13.14 s |
| wheel-tests.log | Same full suite importing installed 0.1.2 wheel, with existing capture/test dependencies: 66 passed, 11.83 s |
| build.log; wheel-source-match.json | Wheel built; all 9 packaged Python files exactly match source |
| quickstart-executed.sh; quickstart.log; quickstart-result.json | Every current README command succeeded from a newly created working directory and virtual environment; README/script SHA recorded |
| wheel-compatibility.json | Fresh NumPy-only wheel environment: all 15 previous bundles match recorded expected labels |
| isolation.sb; isolation-checks.log; isolated-wheel-replays.json | OS denial of network and real model-weight reads asserted; all 15 wheel replays passed |
| protected-before.json; protected-integrity.json | Recorded pre-existing 0.1.1/audit files checked for unchanged bytes |

The 11 new cases cover real permission denial (permissions restored afterward), injected traversal PermissionError and I/O failure, a legal empty directory, stored/deflated CRC errors, truncated ZIP, encrypted members, unsupported compression, malformed NumPy member header, and valid compressed NPZ. The full suite also preserves the original M1–M4 regressions. On hosts whose privileges bypass chmod restrictions, the real-permission test explicitly skips; injected traversal tests remain. On this host the real-permission test ran and passed; there were no skips.

The minimal wheel replay environment contains NumPy and StitchTrace, without TensorFlow, Torch, StarDist or Cellpose. The full 66-test wheel run is a different validation context that reuses capture/test dependencies; it must not be described as running all optional-backend tests with NumPy alone.

## Full experiment rerun (not rerun during this revision)

With Python 3.11, uv on PATH, the supplied inputs/models, and an explicitly prepared full dependency cache, run from this project:

~~~sh
python3.11 experiments/reproduce.py --cache-dir "$STITCHTRACE_CACHE" --name audit-012
~~~

The name must be new. This command creates a reproduction tree and runs inference, unlike the README's preflight-only command. The required cache parameter and current wheel filename are present. This round verified the complete README quickstart and preflight; it did not rerun the full inference experiment. The real inference evidence remains 0.1.0 evidence.

Cold network preparation remains unverified and was not requested or retried during 0.1.2. The earlier default-sandbox attempt failed DNS resolution; two subsequent escalated invocations were not started because automatic approval timed out. This is distinct from a policy refusal or a successful cold download. Explicit existing-cache offline core/full-lock installations passed; they do not establish cold-network reproducibility. See the historical 0.1.1 revision note for preparation/migration mechanics.

No public repository, Zenodo upload, license commitment, payment, publication or submission was performed. Passing these targeted checks is not a claim that all possible malformed-input boundaries or scientific claims have been independently audited.
