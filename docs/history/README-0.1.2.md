# StitchTrace 0.1.2 — private research candidate

Capture recorded instance-assembly decisions, replay without inference or weights,
and measure changes under explicit assembly interventions. Ownership and graph
assembly retain their distinct semantics. Validation checks internal consistency;
it does not prove that every record originated from the native algorithm.

No public license has been granted. Nothing here has been published or submitted.
See [license options](docs/LICENSE-OPTIONS.md).

## Current offline quickstart

Prerequisites: macOS/POSIX, uv on PATH, Python 3.11, a prepared compatible uv cache,
and the separately preserved 0.1.0 evidence directory. This revision contains code,
selected inputs and models, but does not duplicate the 0.1.0 result bundles.

Before running, export these variables with existing absolute paths:
STITCHTRACE_PYTHON (Python 3.11 executable), STITCHTRACE_CACHE (prepared uv cache),
and STITCHTRACE_PREVIOUS_RELEASE (the preserved stitchtrace 0.1.0 directory).
Start in the 0.1.2 project directory. The commands create a new temporary working
directory and install the current wheel plus pinned NumPy offline.

<!-- verified-quickstart -->
~~~sh
set -eu
PROJECT="$(pwd)"
test -x "$STITCHTRACE_PYTHON"
test -d "$STITCHTRACE_CACHE"
test -d "$STITCHTRACE_PREVIOUS_RELEASE"
export UV_CACHE_DIR="$STITCHTRACE_CACHE"
export UV_PYTHON_DOWNLOADS=never
WORK="$(mktemp -d /tmp/stitchtrace-quickstart.XXXXXX)"
cd "$WORK"
uv venv --python "$STITCHTRACE_PYTHON" .venv
uv pip install --offline --python .venv/bin/python -r "$PROJECT/requirements-core.lock" "$PROJECT/dist/stitchtrace-0.1.2-py3-none-any.whl"
export BUNDLE="$STITCHTRACE_PREVIOUS_RELEASE/results/engineering512/validation-0/tool-final"
.venv/bin/stitchtrace verify "$BUNDLE"
.venv/bin/stitchtrace replay "$BUNDLE" --output replayed.npy
.venv/bin/stitchtrace replay "$BUNDLE" --policy retain_all --output alternative.npy
.venv/bin/stitchtrace measure replayed.npy --spacing 1 1
.venv/bin/python - <<'PYTHON'
import os
import numpy as np
from stitchtrace import Bundle, replay, measurements
bundle = Bundle.open(os.environ["BUNDLE"])
labels, objects = replay(bundle)
assert np.array_equal(labels, np.load("replayed.npy", allow_pickle=False))
print({"api_objects": len(measurements(labels, [1, 1]))})
PYTHON
.venv/bin/python "$PROJECT/experiments/prepare_dependencies.py" --cache-dir "$STITCHTRACE_CACHE" --core
.venv/bin/python "$PROJECT/experiments/reproduce.py" --cache-dir "$STITCHTRACE_CACHE" --preflight-only
printf 'Quickstart outputs: %s\n' "$WORK"
~~~

The final preflight installs the full capture lock into a disposable environment;
it requires a cache prepared for that lock, not only NumPy. It does not run model
inference or create a reproduction results tree. Each invocation uses a new output
directory. Unit spacing above denotes pixels, not micrometres.

## Reproduction and evidence

See [0.1.2 changes and validation](docs/REVISION-0.1.2.md), [API](docs/API.md),
[schema](docs/SCHEMA.md), and [historical experiment audit](docs/AUDIT-REPORT.md).
Full experiment reruns require both --cache-dir and --name on
experiments/reproduce.py; the current complete command is in the revision guide.

A project copy alone is not an offline dependency distribution. Cold network
preparation remains unverified and was not retried in this revision. The preparation
mechanism and explicit warm-cache offline installs have been tested. Historical
commands are isolated in [the old README](docs/HISTORICAL-README-0.1.1.md), not current
installation instructions.

Scientific scope remains four real StarDist 2D images, each with four tiles in the
post-freeze block512 engineering regression. The frozen block320 configuration
failed geometry checks on all four images. Cellpose and StarDist 3D evidence uses
label oracles. This revision adds no real inference evidence, accuracy claim,
human-efficiency evidence or public-release commitment.
