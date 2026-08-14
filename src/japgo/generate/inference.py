"""The frozen model, behind an interface that knows nothing about training.

Everything upstream of this module — folds, priors, APLS, the sampler, the corpus — exists to
answer research questions. A game-world generator should not have to import any of it, or know
that leave-one-site-out folds were ever a thing. This is the boundary.

The contract is deliberately narrow: terrain and world channels in, a road probability field out,
plus the metadata needed to reproduce the call. What the probability *means* structurally is the
procedural layer's problem, which is the §13 hand-off and invariant 5 — **ML proposes, procedural
disposes.**

A frozen model is a checkpoint plus everything needed to feed it the same way twice. That is more
than the weights: it is the channel order, the stack version, the resolution, the CRS convention
and the registry hash the corpus was built under. All of it travels in
:class:`FrozenModel.describe`, because a checkpoint whose preprocessing cannot be reconstructed is
not frozen, it is merely saved.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..geo.tiling import Bounds
from ..pipeline.channels import StackSpec, load_stack_spec

DEFAULT_THRESHOLD = 0.45
"""Where to cut the probability field by default.

Not 0.5. The baseline's operating point sits below it on every fold — the class is a few percent
of pixels and the loss is weighted, so the calibrated cutoff is lower than the naive one. Callers
should override per model; this is a starting value, not a claim.
"""


@dataclass(frozen=True)
class ModelCard:
    """What a caller needs to know to use, reproduce or replace the frozen model."""

    checkpoint: str
    trained_on: str
    """Corpus description — tile count and the sites it spans."""

    channels: list[str]
    stack_version: int
    resolution_m: float
    crs: str
    registry_hash: str | None
    width: int
    threshold: float
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    held_out: list[str] = field(default_factory=list)
    """Sites this checkpoint never saw. Named rather than described, so a caller can *check*.

    A leave-one-site-out checkpoint is only honest evidence on the site it held out. Every other
    site in the corpus was training data, and output over one of those tiles is partly recall. The
    demonstration page reads this field to label each panel, because a page that shows a training
    tile without saying so overstates the system exactly where it is easiest to be fooled.
    """

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: Path) -> ModelCard:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def unseen(self, site: str | None) -> bool | None:
        """Whether ``site`` was held out. ``None`` when the card does not say."""
        if not self.held_out or site is None:
            return None
        return site in self.held_out


@dataclass(frozen=True)
class RoadPrediction:
    """What the model proposes, and enough context to act on it."""

    probability: np.ndarray
    """``(rows, cols)`` in [0, 1]. The proposal, not the answer."""

    bounds: Bounds
    crs: str
    resolution_m: float
    threshold: float

    @property
    def candidate_mask(self) -> np.ndarray:
        """The probability field cut at ``threshold``.

        Exposed as a property rather than stored, so a caller can re-cut the same prediction at a
        different threshold without re-running the model. Thresholding is cheap; inference is not.
        """
        return self.probability >= self.threshold

    @property
    def coverage(self) -> float:
        return float(self.candidate_mask.mean())


class FrozenModel:
    """A trained checkpoint, loaded once and callable many times.

    Holds no reference to the training package beyond the network definition itself. Deliberately
    accepts a raw channel stack rather than a :class:`~japgo.pipeline.assemble.TileBundle`: the
    generator will eventually feed it synthetic terrain that was never a tile, and a signature
    that demands a manifest would make that awkward for no benefit.
    """

    def __init__(self, card: ModelCard, *, device: str | None = None) -> None:
        import torch

        from ..model.nets import build_unet

        self.card = card
        self.spec: StackSpec = load_stack_spec()
        if self.spec.stack_version != card.stack_version:
            raise ValueError(
                f"model expects stack v{card.stack_version}, this checkout is "
                f"v{self.spec.stack_version}. The channels do not mean the same thing."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = build_unet(len(card.channels), width=card.width)
        self._model.load_state_dict(torch.load(card.checkpoint, map_location="cpu"))
        self._model = self._model.to(self.device).eval()

    @classmethod
    def load(cls, card_path: Path, *, device: str | None = None) -> FrozenModel:
        return cls(ModelCard.read(card_path), device=device)

    def predict(
        self,
        stack: np.ndarray,
        bounds: Bounds,
        *,
        threshold: float | None = None,
        crs: str | None = None,
    ) -> RoadPrediction:
        """Run the model over one channel stack.

        ``stack`` is ``(channels, rows, cols)`` in the order named by the card. The order is
        checked by length only — names cannot be recovered from an array — so a caller assembling
        channels by hand must follow :attr:`ModelCard.channels`. Getting it wrong produces a
        confident, plausible, entirely wrong prediction, which is the failure this project has
        learned to fear most.
        """
        import torch

        if stack.shape[0] != len(self.card.channels):
            raise ValueError(
                f"expected {len(self.card.channels)} channels in the order "
                f"{self.card.channels}, got {stack.shape[0]}"
            )

        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(stack, dtype=np.float32)[None])
            x = x.to(self.device)
            with torch.autocast(
                device_type=self.device, dtype=torch.float16, enabled=self.device == "cuda"
            ):
                logits = self._model(x)
            probability = torch.sigmoid(logits.float())[0, 0].cpu().numpy()

        return RoadPrediction(
            probability=probability,
            bounds=bounds,
            crs=crs or self.card.crs,
            resolution_m=self.card.resolution_m,
            threshold=self.card.threshold if threshold is None else threshold,
        )

    def describe(self) -> str:
        c = self.card
        return "\n".join([
            f"checkpoint   {c.checkpoint}",
            f"trained on   {c.trained_on}",
            f"channels     {len(c.channels)}: {', '.join(c.channels)}",
            f"stack        v{c.stack_version} @ {c.resolution_m:g} m/px, {c.crs}",
            f"registry     {c.registry_hash}",
            f"threshold    {c.threshold}",
            f"metrics      " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(c.metrics.items())),
            f"device       {self.device}",
        ])
