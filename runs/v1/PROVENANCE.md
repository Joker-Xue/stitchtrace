# Provenance of the v1 results

These files are the outputs of the experiments in `experiments/v1` (protocols `PROTOCOL.md`,
`PROTOCOL-3D.md`; deviations `DEVIATIONS.md`; addenda results `ADDENDUM-A1-RESULT.md`,
`ADDENDUM-A2-RESULT.md`).

- `*/results/*.json` — one record per image or volume: inference and capture times, bundle sizes,
  and for every assembly policy the comparison and seam attribution against untiled inference and
  against the annotation. All policy evaluations are from the final re-evaluation (D5) with the code
  of this release; inference outputs were not recomputed.
- `summary.json`, `SUMMARY.md` — aggregated tables (`experiments/v1/summarize.py`).
- `figures/` — paper figures (`experiments/v1/figures.py`, `figure_architecture.py`).
- `environment.txt` — exact package versions of the experiment environment.

The experiments were run from a private development history before this repository was created.
The deviation records refer to that history; the code in this release is the code that produced
the final evaluations. Captured bundles and untiled predictions (about 8.4 GB) are archived
separately; their location will be given in the README and the paper.

Datasets: BBBC039 (https://bbbc.broadinstitute.org/BBBC039, CC0) and the C. elegans nuclei
dataset (Zenodo record 5942575, CC-BY-4.0). Models: StarDist `2D_versatile_fluo`
(stardist-models, BSD-3-Clause) and Cellpose `cpsam_v2` (downloaded by Cellpose 4.2.1.1).
