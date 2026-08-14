"""Running the frozen model over a heightfield that is larger than a tile.

The model was trained on 512 m crops inside 1.512 km read extents and has no opinion about size —
it is fully convolutional. What it does have an opinion about is *scale*, in two ways that a naive
whole-world forward pass gets wrong:

**Memory.** A 4 km world at 1 m/px is 16 M cells; fifteen float32 channels of it is 960 MB before
a single activation. Invariant 9 says patch-based, and this is where that applies outside training.

**Normalisation.** The ``elevation`` channel is tile-relative — each training sample had its own
mean subtracted, over about 1.5 km of ground. Subtracting a 4 km world's mean instead hands the
model a channel whose spread is several times anything it saw, which is precisely the departure
the Phase 5 sweep showed it degrades under. So each window is normalised over its own read extent,
reproducing the training convention at the training scale.

Windows are predicted with a halo and written back core-only, the same arrangement as the tile
grid (invariant 6). Nothing is blended: a cell's prediction comes from exactly one window, the one
that had 256 m of context around it.

Everything not derived from terrain is left at the stack's nodata fill. That is a deliberate,
declared handicap rather than an oversight — see :func:`terrain_stack`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geo.raster import Raster
from ..geo.tiling import Bounds
from ..pipeline.assemble import terrain_planes
from ..pipeline.channels import StackSpec
from .inference import FrozenModel, RoadPrediction

TERRAIN_CHANNELS = ("elevation", "slope", "aspect_sin", "aspect_cos", "roughness")


@dataclass(frozen=True)
class WindowPlan:
    """How a world is cut up for inference. Recorded because it changes the answer.

    The defaults are not arbitrary: 1000 m of core inside a 256 m halo is 1512 px, which is
    exactly the read extent of a corpus tile. Every window the model sees during generation is
    therefore the same size and shape as every sample it saw during training.

    Uniform shape is also what makes this fast. MIOpen tunes convolution kernels per input shape
    and the first call at a new one costs tens of seconds; ragged edge windows turned a 4 km world
    into sixteen separate tuning runs and 285 s of wall clock for 1 s of arithmetic.
    """

    window_px: int = 1000
    halo_px: int = 256

    @property
    def read_px(self) -> int:
        return self.window_px + 2 * self.halo_px

    def windows(self, rows: int, cols: int):
        """Core slices, in row-major order. The read extent around each is always ``read_px``."""
        for r0 in range(0, rows, self.window_px):
            for c0 in range(0, cols, self.window_px):
                yield (r0, min(r0 + self.window_px, rows), c0, min(c0 + self.window_px, cols))

    def read_window(self, elevation: np.ndarray, r0: int, c0: int) -> tuple[np.ndarray, int, int]:
        """The padded read extent for a core window, and the core's offset within it.

        Where the world does not extend far enough, the terrain is mirrored. Reflection rather
        than zero fill because a wall of zero elevation at the boundary is a cliff, and a cliff is
        a feature the model would answer.
        """
        rows, cols = elevation.shape
        rr0, cc0 = r0 - self.halo_px, c0 - self.halo_px
        rr1, cc1 = rr0 + self.read_px, cc0 + self.read_px

        block = elevation[max(rr0, 0) : min(rr1, rows), max(cc0, 0) : min(cc1, cols)]
        pad = (
            (max(-rr0, 0), max(rr1 - rows, 0)),
            (max(-cc0, 0), max(cc1 - cols, 0)),
        )
        if any(any(side) for side in pad):
            block = np.pad(block, pad, mode="reflect")
        return np.ascontiguousarray(block), rr0, cc0


def terrain_stack(
    dem: Raster, spec: StackSpec, *, valid: np.ndarray | None = None
) -> np.ndarray:
    """A model-ready channel stack from terrain alone.

    Buildings and land use are left at ``nodata_fill``. For a game handing over its own heightfield
    that is simply the truth — there are no buildings yet, which is why it is asking. For an
    experiment it is a declared handicap: five of the fifteen channels carry signal and ten are
    constant, and the model has never seen a sample like that. Both cases want the same code and
    the same honesty about what the model is being given.
    """
    planes = terrain_planes(dem)
    rows, cols = dem.data.shape
    out = np.full((spec.depth, rows, cols), np.float32(spec.nodata_fill), np.float32)

    for i, channel in enumerate(spec.channels):
        if channel.name == "valid":
            out[i] = 1.0 if valid is None else valid.astype(np.float32)
            continue
        plane = planes.get(channel.name)
        if plane is None:
            continue
        out[i] = np.nan_to_num(
            channel.apply_normalisation(np.asarray(plane, np.float32)),
            nan=spec.nodata_fill, posinf=spec.nodata_fill, neginf=spec.nodata_fill,
        )
    return out


def predict_world(
    model: FrozenModel,
    elevation: np.ndarray,
    bounds: Bounds,
    *,
    plan: WindowPlan | None = None,
    threshold: float | None = None,
    crs: str | None = None,
    progress=None,
) -> RoadPrediction:
    """Predict road probability across a whole world, window by window.

    ``elevation`` is absolute metres. Each window's channels are built from that window's read
    extent, so tile-relative normalisation happens at the scale the model was trained at.
    """
    plan = plan or WindowPlan()
    crs = crs or model.card.crs
    resolution = model.card.resolution_m
    rows, cols = elevation.shape

    probability = np.zeros((rows, cols), np.float32)
    for r0, r1, c0, c1 in plan.windows(rows, cols):
        block, rr0, cc0 = plan.read_window(elevation, r0, c0)
        read = Bounds(
            bounds.minx + cc0 * resolution,
            bounds.maxy - (rr0 + plan.read_px) * resolution,
            bounds.minx + (cc0 + plan.read_px) * resolution,
            bounds.maxy - rr0 * resolution,
        )
        stack = terrain_stack(Raster(block, read, crs), model.spec)
        predicted = model.predict(stack, read, threshold=threshold, crs=crs)
        probability[r0:r1, c0:c1] = predicted.probability[
            r0 - rr0 : r1 - rr0, c0 - cc0 : c1 - cc0
        ]
        if progress is not None:
            progress((r0, c0))

    return RoadPrediction(
        probability=probability,
        bounds=bounds,
        crs=crs,
        resolution_m=resolution,
        threshold=model.card.threshold if threshold is None else threshold,
    )
