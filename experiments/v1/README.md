# Experiments of StitchTrace 1.0

Protocols were fixed before the formal runs: [PROTOCOL.md](PROTOCOL.md) (2D, BBBC039) and
[PROTOCOL-3D.md](PROTOCOL-3D.md) (3D, C. elegans). Every deviation is recorded with its reason in
[DEVIATIONS.md](DEVIATIONS.md); exploratory analyses are reported in
[ADDENDUM-A1-RESULT.md](ADDENDUM-A1-RESULT.md) and [ADDENDUM-A2-RESULT.md](ADDENDUM-A2-RESULT.md).
Results are in [`runs/v1`](../../runs/v1).

## Environment

Python 3.11 with the pinned capture dependencies:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-capture.lock -e . scikit-image
```

The exact versions of the experiment environment are in `runs/v1/environment.txt`.

## Data and models (not redistributed here)

| item | source | place it at |
|---|---|---|
| BBBC039 images, masks, metadata | https://bbbc.broadinstitute.org/BBBC039 (`images.zip`, `masks.zip`, `metadata.zip`) | `data/bbbc039/{images,masks,metadata}` |
| C. elegans nuclei | https://zenodo.org/records/5942575 (`c_elegans_nuclei.zip`, SHA-256 `1def0749…152c4e`) | `data/celegans/c_elegans_nuclei` |
| StarDist `2D_versatile_fluo` | stardist-models v0.1 (`StarDist2D.from_pretrained`) | `models/2D_versatile_fluo/` (config.json, thresholds.json, weights_best.h5) |
| Cellpose `cpsam_v2` | downloaded automatically by Cellpose 4.2.1.1 on first use | `~/.cellpose/models` |

## Running

From `experiments/v1`:

```bash
python run_cellpose.py                 # 2D Cellpose arm, 50 images x 3 block sizes
python run_stardist.py                 # 2D StarDist arm, 50 images x 6 settings
python run_cellpose_globalnorm.py      # Addendum A1 (exploratory)
python run_cellpose_3d.py test         # 3D arm, 7 volumes (about 10 h on an Apple M5)
python reevaluate.py D5                # re-evaluate all saved bundles with the current code, no inference
python summarize.py && python figures.py && python figure_architecture.py
```

Every run is resumable: completed images are skipped and captures resume from cached block
predictions. Replay, comparison and attribution of saved bundles need NumPy only.
