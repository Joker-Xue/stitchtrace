# Post-freeze engineering regression, not preregistered validation

Observed original12/12 native/tool/baseline attempts fail before inference at native Block.cover overlap assertion for520x696 images, block320/min_overlap64/context96. Preserve all original outcomes. No original attempt counted successful or replaced.

One global supplemental configuration: block512 (largest power of2 not exceeding shortest original image dimension520), min_overlap64/context96 unchanged, same4 selectedimages, same normalization/weights/thresholds. This value chosen by geometry only, before supplemental prediction; no parameter sweep, image replacement or object inspection. Separate results/engineering512. This is post-observation engineering regression, not an independent/preregistered performance claim. Failure remains final for a case, no further tuning in this round.
