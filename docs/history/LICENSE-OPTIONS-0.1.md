# License decision pending owner approval

No public license is granted for newly authored StitchTrace code in this private candidate. Pyproject deliberately omits a license declaration; no public repository, package publication or license acceptance has occurred.

Possible owner decisions: BSD-3-Clause (consistent with principal scientific backends), MIT (short permissive terms), or Apache-2.0 (explicit patent provisions). This document is not a selection or legal opinion. Before public distribution, actual authors should approve attribution, contributor ownership and chosen terms, and review redistribution of bundled dependencies/data/models. Dependencies are installed separately; their terms remain applicable.

BBBC039v1 official page states CC0. Cite Caicedo et al.2018 and BBBC/Ljosa et al.2012 as recommended: https://bbbc.broadinstitute.org/BBBC039 . Only4 fixed validation/test image/mask pairs used; full official source archives retained privately as provenance. Mask conversion follows first channel + connected components from the official linked example.

StarDist pretrained model comes from official stardist-models v0.1, BSD-3-Clause license retained in models/LICENSE-StarDist-models.txt. Model ZIP and weights hashes remain recorded. No additional training or different model download. Dataset/model training overlap is not ruled out: BBBC039 partly overlaps BBBC038/DSB2018 according to official data page, so this experiment is not claimed to be training-disjoint accuracy validation.

Installed dependency declarations/licenses are copied in docs/third-party for audit, including NumPy, Cellpose, StarDist, TensorFlow and CSBDeep. This is provenance, not a blanket relicensing of third-party material or exhaustive legal certification of every transitive binary dependency. No upstream source code is vendored into the installable wheel; compatibility metadata records identifiers/hashes/signatures.
