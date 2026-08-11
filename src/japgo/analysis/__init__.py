"""Phase 3 — road analysis.

Answers the question the roadmap sets for this phase: **which environmental features actually
predict road structure, and which do not?** The exit criterion is a ranked, quantified list with
the null results stated, so this package is built to report an absence of relationship as
confidently as a presence of one.

The shape is deliberately two-sided:

* :mod:`japgo.analysis.features` reduces a tile's raster stack to environmental **predictors**.
* :mod:`japgo.analysis.structure` reduces a tile's road graph to structural **responses**.
* :mod:`japgo.analysis.correlate` associates the two across a corpus.

Keeping predictors and responses in separate modules is not tidiness. It is what stops a measure
from quietly appearing on both sides of the correlation, which would manufacture a relationship
out of nothing.

Nothing here consumes a model. Phase 3 exists to establish what a model would have to beat, and it
must be runnable before Phase 4 exists at all.
"""

from __future__ import annotations

from .correlate import Association, Study, correlate, spearman
from .features import ENVIRONMENTAL_FEATURES, environmental_features
from .structure import ROAD_STRUCTURE_METRICS, road_structure

__all__ = [
    "ENVIRONMENTAL_FEATURES",
    "ROAD_STRUCTURE_METRICS",
    "Association",
    "Study",
    "correlate",
    "environmental_features",
    "road_structure",
    "spearman",
]
