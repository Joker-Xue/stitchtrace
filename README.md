# StitchTrace

**Record, replay and attribute the tile-assembly decisions of tiled instance segmentation — without re-running inference.**

Large microscopy images are segmented block by block and the per-block labels are then stitched
into one label field. The stitching step decides which block owns an object, which pieces are the
same object and which identifiers are rewritten. Those decisions are normally lost: only the final
mask survives, so a cut or duplicated nucleus cannot be traced to the rule that produced it, and
testing a different stitching rule means running the model again.

StitchTrace wraps the *native* assembly of two widely used tools and stores what they actually did:

| backend | native mechanism recorded | adapter |
|---|---|---|
| StarDist 0.9.2 `predict_instances_big` | per-block raw predictions, context crop, whole-object ownership selection, relabeled writes | `capture_stardist` |
| Cellpose 4.2.1.1 `distributed_eval` | per-block predictions, packed block labels, face-adjacency graph, native relabeling | `capture_cellpose` |

A recorded *bundle* (JSON + NPY files with a SHA-256 manifest) can then be opened anywhere with
NumPy alone to

- **replay** the native result exactly (checked against the recorded output);
- **replay alternative stitching policies** on the same predictions — built-in or your own function;
- **compare** two label fields object by object (matching, splits, merges) and
- **attribute** each difference to the block seams it crosses.

Typical use: diagnose a stitching artefact, evaluate a candidate fix of an assembly rule on
recorded data, or compare assembly rules for a pipeline, all without a GPU, model weights or the
inference stack.

## Installation

```bash
pip install git+https://github.com/Joker-Xue/stitchtrace                          # core: NumPy only (replay, compare, attribute, CLI)
pip install "stitchtrace[stardist] @ git+https://github.com/Joker-Xue/stitchtrace"  # + capture from StarDist
pip install "stitchtrace[cellpose] @ git+https://github.com/Joker-Xue/stitchtrace"  # + capture from Cellpose distributed segmentation
```

StitchTrace is not yet published on PyPI.

Python ≥ 3.11 on Linux or macOS (capture uses POSIX file locks; Windows is not supported).
Capture adapters check the installed upstream version, source hashes and hooked signatures against
`backends/compatibility.json`; pass `allow_unverified=True` to record a mismatch instead of refusing.

## Quickstart (NumPy only, ~1 minute)

The example writes a synthetic bundle whose recorded graph has no cross-block edges — the state
Cellpose 4.2.1.1 produces for 2D images (MouseLand/cellpose#1364, #1480).

```bash
python examples/make_demo_bundle.py demo          # 12 discs cut into 22 recorded pieces
stitchtrace verify demo
stitchtrace policies
stitchtrace replay demo --output native.npy                     # exactly the recorded result
stitchtrace replay demo --policy relink --output relinked.npy   # dimension-correct face linking
python -c "import numpy as np; np.save('ref.npy', np.load('demo/reference.npy'))"
stitchtrace compare ref.npy native.npy --bundle demo            # 4 objects split at seams
stitchtrace compare ref.npy relinked.npy --bundle demo          # seam splits removed
```

## Python API

```python
from stitchtrace import Bundle, replay, compare, attribute, register_policy
from stitchtrace.policies import block_seams

bundle = Bundle.open("demo")                      # verifies hashes, schema and decisions
native, objects = replay(bundle)                  # labels + per-object measurements
relinked, _ = replay(bundle, "relink", threshold=1.0)

@register_policy("graph", "no_links_on_x_seams")  # any function bundle -> labels
def my_rule(bundle):
    ...

report = attribute(native, relinked, block_seams(bundle))
print(report["summary"])                          # seam_split / seam_other / interior
```

Capturing from a real model (requires the optional dependencies):

```python
from stitchtrace.backends.cellpose import capture_cellpose
from cellpose import models

model = models.CellposeModel(gpu=True)
predict = lambda crop, slices: model.eval(crop, normalize={"percentile": [1, 99.8]})[0]
bundle = capture_cellpose(image, predict, "run-001", predictor_id="cpsam_v2;p1-99.8",
                          block_size=[256, 256], diameter=30)
```

`capture_stardist(model, normalized_image, "run-002", model_id=..., block_size=..., min_overlap=..., context=...)`
records StarDist; weights, configuration and thresholds are fingerprinted automatically.
Captures are transactional: an interrupted run resumes from cached block predictions, a completed
run with the same contract is reused, and a conflicting contract is refused.

## Built-in policies

| backend | policy | meaning |
|---|---|---|
| ownership | `native` | recorded StarDist writes in original order |
| ownership | `retain_all` | all context-cropped candidates, original write order (diagnostic baseline) |
| ownership | `crop` | raw block labels cut at the midline of each overlap (generic crop-and-paste) |
| ownership | `iou_merge` | `crop` plus linking of pieces whose raw objects overlap with IoU ≥ `threshold` |
| ownership | `center` | each object pasted whole by the block whose midline box holds its centroid |
| graph | `native` | recorded face-adjacency edges |
| graph | `disable_edges` | no cross-block linking |
| graph | `relink` | edges recomputed from the packed labels with a per-pixel face shrink (`threshold`) |

`relink` reproduces the upstream face shrink exactly on 3D faces and differs only where the
upstream code collapses a 1-pixel 2D face to a scalar gradient. Like upstream, it does not link
contacts narrower than the shrink margin.

## Documentation

- [API and CLI reference](docs/API.md) · [bundle schema](docs/SCHEMA.md)
- [Experiments](experiments/v1/README.md) and [results](runs/v1/PROVENANCE.md) of the accompanying paper;
  all recorded bundles of these experiments are archived on Zenodo (https://doi.org/10.5281/zenodo.23599583)
- [Change history](CHANGELOG.md); pre-1.0 audit material is kept in [docs/history](docs/history)

## Limitations

- Records the assembly of the two supported upstream versions; other versions need a compatibility
  review (or `allow_unverified=True`, which is recorded in the bundle).
- Validation checks internal consistency of a bundle; hashes detect damage, not a deliberate rewrite
  of data and manifest together.
- Arrays are held in memory (limit 64 Mi elements per array); not a streaming engine for very large
  volumes.
- Transactions assume a local POSIX filesystem.

## Citation and license

The recorded bundles and result tables of the accompanying paper are archived as:

> Lin, X. (2026). *StitchTrace v1 experiment data: recorded tile-assembly bundles and results*
> (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23599583

Citation details of the software and paper will be added on publication. StitchTrace is released under the [MIT License](Licence.txt).
Third-party packages, models and datasets keep their own licenses; copies of the relevant notices are in [docs/third-party](docs/third-party).
