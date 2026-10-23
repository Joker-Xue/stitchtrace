# Addendum A2 result (exploratory, post hoc)

Policy `relink_absorb` (min_size 15 = Cellpose default, not tuned), evaluated on every saved
Cellpose bundle without inference (re-evaluation D5). Totals over images/volumes, against untiled
inference of the same model.

| arm | relink: F1 / seam_extra / fp / merges / fn | relink_absorb: F1 / seam_extra / fp / merges / fn |
|---|---|---|
| 2D, 128 px (50 images) | 0.984 / 71 / 128 / 10 / 16 | 0.990 / 6 / 63 / 10 / 16 |
| 2D, 192 px | 0.989 / 42 / 94 / 8 / 18 | 0.992 / 5 / 57 / 8 / 18 |
| 2D, 256 px | 0.990 / 34 / 84 / 10 / 17 | 0.993 / 2 / 52 / 10 / 17 |
| 3D chain (7 volumes) | 0.982 / 56 / 96 / 31 / 44 | 0.986 / 27 / 67 / 31 / 44 |
| 3D grid (2 volumes) | 0.944 / 108 / 113 / 9 / 17 | 0.973 / 39 / 44 / 9 / 17 |

The prediction holds: seam fragments fall sharply, F1 rises, merges and missed objects are
unchanged. Against the annotation, 3D grid F1 rises from 0.769 to 0.793. Because the rule was
designed after seeing these data, the numbers show what the tool allows one to test in seconds;
they are not a validated improvement of Cellpose and are reported as exploratory.
