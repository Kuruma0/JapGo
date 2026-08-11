"""Environmental predictors: a tile's raster stack reduced to scalars.

These are the *left* half of the Phase 3 study — the environment, as the model will see it. Every
measure here is computed over the **core only** and masked by the ``valid`` channel, for two
reasons that both bite silently if ignored:

* **The halo overlaps the neighbours.** Averaging over core+halo counts the same ground once per
  adjacent tile, which inflates the apparent sample size of exactly the correlation this package
  is trying to measure honestly (invariant 6, research doc §16.1).
* **Fill is not data.** ``nodata_fill`` is 0.0, and zero is a plausible slope. Treating filled
  cells as observations biases every terrain measure toward flat.

Values are reported in **real units**, not the stack's normalised ones. The stack stores slope
divided by 50 and building height divided by 60 because that is what a network wants; a findings
table that says "slope 0.28" instead of "slope 14.1%" is unreadable by the geographer whose
judgement the §1.3 success criterion depends on.
"""

from __future__ import annotations

import numpy as np

from ..pipeline.assemble import TileBundle
from ..pipeline.channels import Normalise

DEFAULT_GRADE_LIMIT_PCT = 12.0
"""Grade above which most of the hierarchy cannot go — see ``config/road_hierarchy.yaml``.

12% is the limit for local streets and the classes just above them, so "fraction of the tile above
12%" is a fair proxy for "fraction a road network cannot easily cross". The Atami tile in
docs/decision-log.md sits at 90.8% by this measure.
"""

FLAT_SLOPE_PCT = 2.0
"""Below this, aspect is numerically defined but physically meaningless — direction of a flat
field is noise, and averaging it in dilutes the real signal from the slopes that matter."""

ENVIRONMENTAL_FEATURES: tuple[str, ...] = (
    "relief_m",
    "slope_median_pct",
    "slope_p90_pct",
    "slope_above_limit_frac",
    "roughness_mean_m",
    "aspect_alignment",
    "built_frac",
    "building_height_mean_m",
    "landuse_built_frac",
    "landuse_agricultural_frac",
    "landuse_forest_frac",
    "landuse_water_frac",
)
"""The predictor vocabulary, fixed and ordered.

Fixed because a study whose feature set drifts between runs cannot be compared against its own
earlier results, and Phase 3's output is meant to be a durable ranking.
"""


def core_view(bundle: TileBundle, data: np.ndarray) -> np.ndarray:
    """Crop a ``(rows, cols)`` array from the read extent down to the tile core.

    Returns ``data`` unchanged for a bundle that was assembled without a halo — read from
    :attr:`~japgo.pipeline.assemble.TileBundle.with_halo` rather than guessed at from the shape.
    Guessing is not possible: a 1000² stack is a 1 km core at 1 m *or* a read extent at 1.512 m,
    and choosing wrong silently crops a core-only tile to its middle 44%.
    """
    if not bundle.with_halo:
        return data

    rows, cols = data.shape[-2:]
    resolution = bundle.tile.read.width / cols
    inset = int(round(bundle.tile.halo_m / resolution))

    if inset <= 0 or rows - 2 * inset <= 0 or cols - 2 * inset <= 0:
        return data
    return data[inset : rows - inset, inset : cols - inset]


def _real_units(bundle: TileBundle, name: str) -> np.ndarray:
    """A channel over the core, with the stack's normalisation undone."""
    channel = bundle.spec.get(name)
    data = core_view(bundle, bundle.channel(name)).astype(np.float64)

    if channel.normalise is Normalise.SCALE:
        if not channel.scale:
            raise ValueError(f"channel {name!r} is scale-normalised but declares no scale")
        return data * channel.scale
    # TILE_RELATIVE leaves differences intact, which is all any measure here reads.
    return data


def _valid_mask(bundle: TileBundle) -> np.ndarray:
    return core_view(bundle, bundle.channel("valid")) > 0.5


def _masked_stat(values: np.ndarray, mask: np.ndarray, fn) -> float:
    """Apply ``fn`` to the masked values, or return NaN when nothing is observed.

    NaN rather than 0.0 on purpose: "no data here" and "measured zero" must stay distinguishable
    all the way into the findings table, because a correlation computed over silently-zeroed voids
    is worse than one computed over fewer tiles.
    """
    selected = values[mask]
    if selected.size == 0:
        return float("nan")
    return float(fn(selected))


def environmental_features(bundle: TileBundle) -> dict[str, float]:
    """Reduce one tile's environment to the :data:`ENVIRONMENTAL_FEATURES` vector.

    Missing measures come back as NaN rather than being omitted, so every tile yields the same
    keys and the study can count how much of each feature it actually has.
    """
    valid = _valid_mask(bundle)

    elevation = _real_units(bundle, "elevation")
    slope = _real_units(bundle, "slope")
    roughness = _real_units(bundle, "roughness")

    observed_elevation = elevation[valid]
    relief = (
        float(observed_elevation.max() - observed_elevation.min())
        if observed_elevation.size
        else float("nan")
    )

    features: dict[str, float] = {
        "relief_m": relief,
        "slope_median_pct": _masked_stat(slope, valid, np.median),
        "slope_p90_pct": _masked_stat(slope, valid, lambda v: np.percentile(v, 90)),
        "slope_above_limit_frac": _masked_stat(
            slope, valid, lambda v: np.mean(v > DEFAULT_GRADE_LIMIT_PCT)
        ),
        "roughness_mean_m": _masked_stat(roughness, valid, np.mean),
        "aspect_alignment": _aspect_alignment(bundle, slope, valid),
    }

    building_mask = core_view(bundle, bundle.channel("building_mask"))
    features["built_frac"] = _masked_stat(building_mask, valid, np.mean)

    height = _real_units(bundle, "building_height")
    built = valid & (building_mask > 0.5)
    features["building_height_mean_m"] = _masked_stat(height, built, np.mean)

    for landuse in ("built", "agricultural", "forest", "water"):
        name = f"landuse_{landuse}"
        features[f"{name}_frac"] = _masked_stat(
            core_view(bundle, bundle.channel(name)), valid, np.mean
        )

    return {name: features[name] for name in ENVIRONMENTAL_FEATURES}


def _aspect_alignment(bundle: TileBundle, slope: np.ndarray, valid: np.ndarray) -> float:
    """How consistently the terrain faces one way, in [0, 1].

    The resultant length of the aspect unit vectors over sloped ground. A planar hillside scores
    near 1; a conical peak or a dissected plateau scores near 0. This is the measure that should
    separate the Kawanehon valley — where terrain faces two ways across a corridor — from the Izu
    coast, where it faces mostly seaward. Both are "steep"; slope alone cannot tell them apart,
    and the site-selection document predicts different road responses for each.
    """
    sloped = valid & (slope > FLAT_SLOPE_PCT)
    if not sloped.any():
        return float("nan")

    sin = core_view(bundle, bundle.channel("aspect_sin"))[sloped]
    cos = core_view(bundle, bundle.channel("aspect_cos"))[sloped]
    return float(np.hypot(sin.mean(), cos.mean()))


def coverage(bundle: TileBundle) -> float:
    """Fraction of the core carrying real observations.

    Not a predictor — a quality gate. A tile that is mostly void produces feature values that are
    technically defined and practically meaningless, so the study filters on this rather than
    letting thin tiles vote.
    """
    return float(_valid_mask(bundle).mean())
