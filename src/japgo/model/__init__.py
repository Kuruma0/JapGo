"""Phase 4 — the baseline model and the floor it has to clear.

The roadmap's exit criterion for this phase is not "the model trains". It is *beats a non-learned
prior on APLS/TOPO on a held-out city*, so everything here is arranged around a comparison rather
than a loss curve:

* :mod:`japgo.model.dataset` — patch indexing and the geographic folds, with the leakage check.
* :mod:`japgo.model.nets` — the dull U-Net of spec §51, sized against the 16 GB budget.
* :mod:`japgo.model.baseline` — the priors to beat, and the metric that decides it.
* :mod:`japgo.model.train` — one run per fold, config pinned beside the checkpoint.

Torch is in the ``ml`` extra, so every import of it is deferred: the provenance and pipeline layers
must stay installable on a machine with no GPU stack.
"""

from __future__ import annotations

from .baseline import Score, best_threshold, score
from .dataset import Fold, assert_no_overlap, configured_fold, index_patches, leave_one_site_out

__all__ = [
    "Fold",
    "Score",
    "assert_no_overlap",
    "best_threshold",
    "configured_fold",
    "index_patches",
    "leave_one_site_out",
    "score",
]
