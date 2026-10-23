# Pilot investigations (superseded)

Scripts from the direction check that preceded the v1 protocols (October 2026): the BBBC039
`test-1` under-segmentation check (`test1_untiled.py`, model limitation, not stitching), a first
policy replay on StarDist records (`policy_probe.py`) and the first Cellpose distributed probes
(`cellpose_probe.py`, `cellpose_relink.py`). They motivated the design of `experiments/v1` but
their numbers are not used: the pilot Cellpose probes passed views of a shared image to the
predictor (see `experiments/v1/DEVIATIONS.md`, D1), and `relink` was later corrected (D4).
