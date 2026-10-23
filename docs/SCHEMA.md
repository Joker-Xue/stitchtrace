# Bundle schema1

`manifest.json`: `schema:1`, `files:{relative_path:{sha256,bytes}}`. Inventory contains all files except manifest itself. Relative POSIX members cannot be absolute, have empty/dot/parent components, backslashes or symlink components. No pickle or executable loader. `bundle.json` must be in inventory.

Common metadata: `schema`, `backend`, `contract`, `shape`, `axes`, `spacing`; capture adds parameters, source version/module/signature/implementation identity and predictor provenance. Expected output is `expected.npy`. Core reader imposes64M-pixel arrays/shape and per-file1GiB bound; not designed for hostile concurrent filesystem mutation or unbounded large-volume streaming.

`backend="graph"`: `packed.npy`, positive `nodes` equal all used packed IDs, complete `edges` with both endpoints in nodes. Full graph retained, never a spatial crop missing transitive dependencies. Capture also stores local/packed tile labels and original native mapping. Node connectivity encodes actual native union decisions; replay computes connected components and compact IDs. Zero is background. No proposed edge implies measured biological identity.

`backend="ownership"`: explicit `order`, ordered `tiles`. Each tile has ID, actual read/write bounds, raw cache reference, local/selected/written array paths, kept/rejected IDs. Selected labels are either zero or the same local ID; one selected local ID maps to one written label. Write domains cover output shape. Recorded raw cache includes actual prediction labels and polygons; local is after context crop. Rejected IDs describe this window's responsibility decision, not disappearance of a biological object. Native replay applies recorded writes; it does not need model weights or re-run the native responsibility predicate. The pilot separately tested predicate recomputation; this minimal core replays decisions.

`retain_all` replaces ownership selection with all context-cropped labels and uses original nonzero last-write semantics. It cannot change model context or tile geometry; those require different predictions. `disable_edges` removes all graph edges without changing local predictions. Neither intervention is an accuracy recommendation.

Schema1 fields will not be silently reinterpreted. Unsupported versions fail. Adding a compatible optional provenance field is allowed; changing decisions/coordinates requires a new schema and explicit migration. The reader is authoritative for semantic validation; no claim of schema authenticity/signature validation.
