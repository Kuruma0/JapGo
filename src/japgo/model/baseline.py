"""The non-learned priors a model has to beat, and the metrics that decide it.

Spec §51 and the Phase 4 exit criterion both insist the baseline is dull and that anything fancier
must beat it. That only means something if the floor is honest, so there are two priors here and
the harder one is the default comparison:

* **constant** — predict the training set's road pixel frequency everywhere. Trivial, but it is
  the score a model gets for learning nothing, and on a sparse target that score is not zero.
* **built proximity** — predict road wherever buildings are. Not learned, but genuinely
  informative: roads and buildings co-occur, which is most of what a weak model discovers. A
  U-Net that cannot beat this has learned "roads are where the town is" and nothing else.

Metrics are pixel-level here on purpose. Research doc §16.2 rejects pixel similarity as the
*primary* measure of a generated network, and rightly — but Phase 4's question is narrower: does
the model recover the road layer of a place it has never seen? For that, F1 against the known mask
is the right instrument, and the structural measures in :mod:`japgo.analysis` are what turn it
into a Phase 5 argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROAD_TARGET = "road_mask"


@dataclass(frozen=True)
class Score:
    """Pixel agreement between a probability field and the known road mask."""

    f1: float
    precision: float
    recall: float
    iou: float
    threshold: float
    positives: int

    def describe(self) -> str:
        return (
            f"F1 {self.f1:.3f}  P {self.precision:.3f}  R {self.recall:.3f}  "
            f"IoU {self.iou:.3f}  @thr {self.threshold:.2f}"
        )


def score(
    probability: np.ndarray,
    truth: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    threshold: float = 0.5,
) -> Score:
    """F1 and friends over the valid pixels only.

    Void is excluded rather than counted as background: a tile that is 40% unobserved would
    otherwise score well for predicting nothing there, and the sea would flatter every model.
    """
    p = np.asarray(probability, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64) > 0.5
    mask = np.ones(t.shape, dtype=bool) if valid is None else np.asarray(valid) > 0.5

    predicted = (p >= threshold) & mask
    actual = t & mask

    tp = float(np.count_nonzero(predicted & actual))
    fp = float(np.count_nonzero(predicted & ~actual))
    fn = float(np.count_nonzero(~predicted & actual))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return Score(f1, precision, recall, iou, threshold, int(np.count_nonzero(actual)))


def best_threshold(
    probability: np.ndarray,
    truth: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    steps: int = 19,
) -> Score:
    """The score at its most favourable threshold.

    Quoted for every model *and* every prior, so the comparison cannot be won by tuning one side's
    cutoff. Road pixels are a small minority of a tile, and 0.5 is arbitrary against that skew.
    """
    candidates = np.linspace(0.05, 0.95, steps)
    return max(
        (score(probability, truth, valid=valid, threshold=float(t)) for t in candidates),
        key=lambda s: s.f1,
    )


def constant_prior(shape: tuple[int, int], rate: float) -> np.ndarray:
    """The frequency prior: the same probability everywhere."""
    return np.full(shape, float(rate), dtype=np.float32)


def road_rate(loader, patches, *, target_index: int, crop: int) -> float:
    """Fraction of valid pixels that are road, measured on the training patches only.

    Measured on training data because a prior fitted to the evaluation set is not a prior, it is a
    leak — the same reason the model does not get to see the held-out site.
    """
    positives = total = 0
    for patch in patches:
        stack, targets = loader.read(patch, crop)
        valid = stack[-1] > 0.5
        positives += int(np.count_nonzero((targets[target_index] > 0.5) & valid))
        total += int(np.count_nonzero(valid))
    return positives / total if total else 0.0


def built_proximity_prior(
    stack: np.ndarray,
    *,
    built_index: int,
    radius: int = 12,
) -> np.ndarray:
    """Predict road near buildings, by dilating the building mask.

    The harder floor. Implemented as a summed-area table rather than a sliding window for the
    reason AGENTS.md gives: ``sliding_window_view`` plus a reduction is O(n·w²) and materialises a
    view w² times the array.
    """
    mask = (stack[built_index] > 0.5).astype(np.float64)
    integral = np.pad(mask, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    rows, cols = mask.shape
    r0 = np.clip(np.arange(rows) - radius, 0, rows)
    r1 = np.clip(np.arange(rows) + radius + 1, 0, rows)
    c0 = np.clip(np.arange(cols) - radius, 0, cols)
    c1 = np.clip(np.arange(cols) + radius + 1, 0, cols)

    total = (
        integral[np.ix_(r1, c1)]
        - integral[np.ix_(r0, c1)]
        - integral[np.ix_(r1, c0)]
        + integral[np.ix_(r0, c0)]
    )
    area = np.outer(r1 - r0, c1 - c0)
    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(area > 0, total / area, 0.0)
    return np.clip(density * 4.0, 0.0, 1.0).astype(np.float32)
