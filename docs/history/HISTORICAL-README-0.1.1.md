# StitchTrace 0.1.1 audit revision

See [revision evidence and explicit cold preparation](docs/REVISION-0.1.1.md). Older experiment descriptions below are preserved as historical 0.1.0 context. Reproduction now requires --cache-dir; prepare dependencies explicitly before offline use.

# StitchTrace 0.1.0 — private research candidate

Capture and preserve **actual instance-assembly decisions**, replay without inference or weights, and measure changes under explicit assembly interventions. Cellpose graph connectivity and StarDist responsibility ownership remain distinct. This is not a segmentation model or automatic proof that a result is wrong.

No public license has been granted for this project. See [license options](docs/LICENSE-OPTIONS.md). Nothing here has been published or submitted.

## Install / quickstart

Python3.11+, local POSIX filesystem. The core needs NumPy only:

```sh
python -m pip install ./dist/stitchtrace-0.1.0-py3-none-any.whl
stitchtrace verify results/engineering512/validation-0/tool-final
stitchtrace replay results/engineering512/validation-0/tool-final --output replayed.npy
stitchtrace replay results/engineering512/validation-0/tool-final --policy retain_all --output alternative.npy
stitchtrace measure replayed.npy --spacing 1 1
```

Use a new output path; CLI refuses existing output. Spacing follows array axes. Real BBBC039 physical pixel sizes were not supplied, so unit spacing denotes pixel units, not micrometres.

```python
from stitchtrace import Bundle, replay, measurements
bundle = Bundle.open("results/engineering512/validation-0/tool-final")
labels, objects = replay(bundle)
```

Optional native capture (fixed versions, tested Python3.11/macOS ARM64):

```sh
python -m pip install '.[stardist]'       # TensorFlow, StarDist
python -m pip install '.[cellpose]'       # Cellpose, Dask, Zarr2
```

```python
from stitchtrace.backends.stardist import capture_stardist
# model: already constructed official StarDist model; image: already normalized ndarray
capture_stardist(model, image, "new-run", model_id="my-weight-provenance",
                 block_size=512, min_overlap=64, context=96)
```

Weights/config/thresholds are also fingerprinted automatically; a user label alone cannot make changed StarDist weights reuse a cache. Cellpose capture accepts an explicit `predict_tile(array, crop)` callable and required `predictor_id`; callers must change its ID when external predictor state changes. No Cellpose model is downloaded or silently instantiated. Current actual Cellpose experiments use **label oracles**, not neural inference.

## Evidence and limitations

Read [audit report](docs/AUDIT-REPORT.md), [API](docs/API.md), [schema](docs/SCHEMA.md), [frozen experiment](experiments/PLAN.md), and [post-freeze geometry amendment](experiments/AMENDMENT-01.md).

The original frozen block320 configuration failed native geometry checks on all4 selected BBBC039 images. A separately labelled block512 engineering regression succeeded. Do not present the latter as preregistered success. No clinical, training-disjoint accuracy or human-efficiency claim is supported.

## Reproduce without overwriting evidence

```sh
python3.11 experiments/reproduce.py --name audit-01
```

This creates a new project-local reproduction tree, isolated environments, runs tests and both original/engineering configurations, and performs OS-isolated replay on macOS. Requires `uv`, cached packages from the recorded environment, and supplied local selected data/model files. It uses offline package installation; missing wheels fail visibly. It never publishes or changes system security settings. Alternatively follow the individual scripts/logs in `experiments/` and `results/`. Full source download archives are retained as provenance; only4 selected image/mask pairs enter the experiment.
