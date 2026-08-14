"""Synthetic terrain: worlds the model has provably never seen.

Everything the frozen model has been shown so far is Shizuoka, and every evaluation of it has been
a reconstruction test — predict the roads that are actually there, score against OSM. That leaves
the generation question unanswered. A game does not hand the system a real place; it hands it a
heightfield that has never existed, and asks what roads belong on it.

So this module builds heightfields from parameters and a seed, with no reference to any tile,
raster or road network in the corpus. Four archetypes, each assembled from landform primitives
rather than sampled from data:

* **mountain_valley** — a trunk valley and a tributary cut into a ridged massif.
* **coastal** — sea, terrace, cliff line and an inland corridor.
* **plain** — low gradient with gentle hills and broad flats.
* **basin** — a flat floor ringed by mountains, breached by passes at different heights.

The construction that does most of the work is a weighted replacement: a valley is a floor
elevation that descends downstream, plus a Gaussian weight saying how completely it replaces the
land around it. Close in, the ground *is* the floor; far out, the massif is untouched; in between
the wall curves into the ridge above it. Confluences fall out of two weights summing past one.

**Only the terrain channels are populated.** The stack also carries buildings and land use, and
those are precisely the settlement information the experiment must not supply — a model told where
the houses are is not being asked what the terrain implies. They are filled with the stack's
nodata value, uniformly, for every world. That is a real distribution shift and the reason
:func:`japgo.generate.synthetic` ships alongside a control: the same zeroing applied to a real
tile, where the full-channel answer is known.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from ..geo.raster import Raster
from ..geo.tiling import Bounds, Tile

ARCHETYPES = ("mountain_valley", "coastal", "plain", "basin")


@dataclass(frozen=True)
class TerrainParams:
    """Everything needed to rebuild a world, including what its archetype ignored.

    Fields an archetype does not read are still recorded. A parameter file that omitted them could
    not be diffed against another archetype's, and the point of writing parameters down is to be
    able to say exactly what differed between two worlds.
    """

    archetype: str
    seed: int
    size_m: float = 4000.0
    resolution_m: float = 1.0

    # --- fractal detail, shared by every archetype ---------------------------------------------
    noise_relief_m: float = 60.0
    noise_octaves: int = 6
    noise_base_cells: int = 4
    noise_gain: float = 0.5

    detail_relief_m: float = 2.5
    """High-frequency undulation added last, over everything.

    Without it the valley floors are glass: the massif's detail is blended away exactly where the
    valley weight approaches one, and a hillshade shows a hard edge between textured mountain and
    mirror-smooth floor. Real floors are not smooth, and the model reads terrain at metre scale.
    """

    # --- the dominant landform -----------------------------------------------------------------
    base_elevation_m: float = 200.0
    """Lowest ground in the world, before noise."""

    relief_m: float = 700.0
    """Height of the dominant landform above ``base_elevation_m``."""

    # --- valleys (mountain_valley, coastal corridor) --------------------------------------------
    valley_wall_width_m: float = 480.0
    """How far a valley's influence reaches. With the floor 500 m below the ridge line this puts
    the steepest wall grade near 80% — which is what the real Kawanehon tiles measure."""

    valley_floor_width_m: float = 60.0
    valley_drop_m: float = 250.0
    """How much the trunk valley floor descends from head to mouth."""

    tributary_drop_m: float = 200.0

    # --- coastal ---------------------------------------------------------------------------------
    shoreline_x_m: float = 900.0
    shore_wobble_m: float = 320.0
    sea_depth_m: float = 40.0
    terrace_width_m: float = 350.0
    terrace_height_m: float = 18.0
    cliff_width_m: float = 220.0

    # --- basin -------------------------------------------------------------------------------------
    basin_radius_m: float = 1150.0
    basin_wall_width_m: float = 900.0
    pass_bearings_deg: tuple[float, ...] = (20.0, 130.0, 250.0, 315.0)
    pass_depths_m: tuple[float, ...] = (0.62, 0.45, 0.78, 0.30)
    """Fraction of the wall height removed at each pass, so the passes sit at different heights."""

    pass_width_deg: float = 11.0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def size_px(self) -> int:
        return int(round(self.size_m / self.resolution_m))


#: Per-archetype overrides. The defaults above describe the mountain valley; everything else is a
#: departure from it, written here rather than in code so a world's parameters are one lookup.
ARCHETYPE_DEFAULTS: dict[str, dict] = {
    "mountain_valley": {
        "base_elevation_m": 220.0, "relief_m": 780.0, "noise_relief_m": 110.0,
        "valley_wall_width_m": 460.0, "valley_drop_m": 260.0, "detail_relief_m": 3.0,
    },
    "coastal": {
        "base_elevation_m": 0.0, "relief_m": 320.0, "noise_relief_m": 45.0,
        "valley_wall_width_m": 380.0, "valley_drop_m": 150.0, "noise_base_cells": 5,
        "detail_relief_m": 2.0,
    },
    "plain": {
        "base_elevation_m": 15.0, "relief_m": 45.0, "noise_relief_m": 9.0,
        "noise_base_cells": 3, "noise_octaves": 5, "detail_relief_m": 0.6,
    },
    "basin": {
        "base_elevation_m": 300.0, "relief_m": 720.0, "noise_relief_m": 80.0,
        "detail_relief_m": 2.5,
    },
}


#: Parameters measured in metres of ground, which therefore have to scale with the world.
#: The defaults describe a 4 km world; a 1 km one with a shoreline 900 m in and a 1150 m basin
#: radius is not a smaller version of the same place, it is all sea and no basin.
SCALES_WITH_SIZE = (
    "valley_wall_width_m", "valley_floor_width_m", "shoreline_x_m", "shore_wobble_m",
    "terrace_width_m", "cliff_width_m", "basin_radius_m", "basin_wall_width_m",
)

REFERENCE_SIZE_M = 4000.0


def params_for(archetype: str, seed: int, size_m: float = REFERENCE_SIZE_M, **overrides):
    """The standard parameters for an archetype at a seed.

    Seeds change the noise field and the landform placement; they do not change the archetype's
    character. That separation is what makes three seeds an actual replicate rather than three
    different experiments.

    Heights are left alone when the world is resized. Only horizontal extents scale — a 1 km world
    should be a 1 km piece of the same country, not a scale model of it with 1000 m mountains
    compressed into it.
    """
    if archetype not in ARCHETYPES:
        raise ValueError(f"unknown archetype {archetype!r}; expected one of {ARCHETYPES}")

    values = {**ARCHETYPE_DEFAULTS[archetype]}
    factor = size_m / REFERENCE_SIZE_M
    if factor != 1.0:
        defaults = TerrainParams(archetype=archetype, seed=seed)
        for name in SCALES_WITH_SIZE:
            values[name] = values.get(name, getattr(defaults, name)) * factor
    return TerrainParams(
        archetype=archetype, seed=seed, size_m=size_m, **{**values, **overrides}
    )


@dataclass
class SyntheticWorld:
    """A heightfield and the parameters that produced it."""

    params: TerrainParams
    elevation: np.ndarray
    """``(rows, cols)`` float32, **absolute** metres above the world datum."""

    name: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def bounds(self) -> Bounds:
        return Bounds(0.0, 0.0, self.params.size_m, self.params.size_m)

    @property
    def tile(self) -> Tile:
        """The whole world as a single haloless tile, so the project's structural measures apply.

        A synthetic world has no neighbours to borrow context from, so core and read extent
        coincide. Reusing :class:`Tile` rather than inventing a second geometry keeps
        ``road_structure`` comparable between these worlds and the real corpus.
        """
        return Tile(zone=8, ix=0, iy=0, core_size_m=self.params.size_m, halo_m=0.0)

    def raster(self, crs: str = "EPSG:6676") -> Raster:
        return Raster(self.elevation, self.bounds, crs)

    def describe(self) -> str:
        z = self.elevation
        return (
            f"{self.name or self.params.archetype}  seed {self.params.seed}  "
            f"{self.params.size_m / 1000:.0f} km at {self.params.resolution_m:g} m/px  "
            f"elevation {z.min():.0f}-{z.max():.0f} m"
        )


# ---------------------------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------------------------


def _value_noise(rng: np.random.Generator, shape: tuple[int, int], cells: int) -> np.ndarray:
    """Smoothstep-interpolated value noise on a ``cells``-square lattice.

    Value noise rather than Perlin because the lattice artefacts that make Perlin worth the extra
    machinery do not matter under six octaves of summation, and this stays readable.
    """
    rows, cols = shape
    lattice = rng.random((cells + 1, cells + 1), dtype=np.float32)

    def axis(n: int) -> tuple[np.ndarray, np.ndarray]:
        f = np.linspace(0.0, cells, n, dtype=np.float32)
        i = np.clip(np.floor(f), 0, cells - 1).astype(np.intp)
        t = f - i
        return i, t * t * (3.0 - 2.0 * t)          # smoothstep: C1, so slope stays continuous

    iy, sy = axis(rows)
    ix, sx = axis(cols)

    g00 = lattice[np.ix_(iy, ix)]
    g01 = lattice[np.ix_(iy, ix + 1)]
    g10 = lattice[np.ix_(iy + 1, ix)]
    g11 = lattice[np.ix_(iy + 1, ix + 1)]

    top = g00 + (g01 - g00) * sx
    bottom = g10 + (g11 - g10) * sx
    return top + (bottom - top) * sy[:, None]


def _fbm(rng: np.random.Generator, shape: tuple[int, int], p: TerrainParams) -> np.ndarray:
    """Fractal sum in [-1, 1]. Deterministic given the generator's state."""
    total = np.zeros(shape, np.float32)
    amplitude, cells, weight = 1.0, p.noise_base_cells, 0.0
    for _ in range(p.noise_octaves):
        total += np.float32(amplitude) * _value_noise(rng, shape, cells)
        weight += amplitude
        amplitude *= p.noise_gain
        cells *= 2
    return (total / np.float32(weight)) * np.float32(2.0) - np.float32(1.0)


def _ridged(rng: np.random.Generator, shape: tuple[int, int], p: TerrainParams) -> np.ndarray:
    """Ridged fractal in [0, 1] — the sharp crests a mountain massif needs.

    ``1 - |noise|`` folds the field at zero, which turns smooth maxima into creases. Without it a
    summed fBm reads as dunes, and a road network laid over dunes is not testing anything about
    mountains.
    """
    return 1.0 - np.abs(_fbm(rng, shape, p))


# ---------------------------------------------------------------------------------------------
# landform primitives
# ---------------------------------------------------------------------------------------------


def _grid(p: TerrainParams) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates in world metres, ``(x, y)`` broadcastable to the raster."""
    n = p.size_px
    axis = (np.arange(n, dtype=np.float32) + 0.5) * np.float32(p.resolution_m)
    x = axis[None, :]
    y = (np.float32(p.size_m) - axis)[:, None]      # row 0 is the top, i.e. maximum y
    return x, y


def _polyline_fields(
    x: np.ndarray, y: np.ndarray, points: list[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Distance to a polyline, and normalised position along it at the nearest point.

    Both fields come from the same pass because a valley needs them together: distance sets the
    V-profile, position along sets how far the floor has descended.
    """
    shape = np.broadcast_shapes(x.shape, y.shape)
    best_d = np.full(shape, np.inf, np.float32)
    best_t = np.zeros(shape, np.float32)

    spans = [math.dist(a, b) for a, b in zip(points, points[1:], strict=False)]
    total = sum(spans) or 1.0
    travelled = 0.0

    for (ax, ay), (bx, by), span in zip(points, points[1:], spans, strict=False):
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy or 1.0
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / denom, 0.0, 1.0).astype(np.float32)
        d = np.hypot(x - (ax + t * dx), y - (ay + t * dy)).astype(np.float32)

        closer = d < best_d
        best_t = np.where(closer, (travelled + t * span) / total, best_t).astype(np.float32)
        best_d = np.where(closer, d, best_d)
        travelled += span

    return best_d, best_t


def _valley(
    x: np.ndarray,
    y: np.ndarray,
    points: list[tuple[float, float]],
    *,
    head_m: float,
    mouth_m: float,
    wall_width_m: float,
    floor_width_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """A valley as a floor elevation and the weight with which it replaces the surrounding land.

    Not a surface to take the minimum of. That was the first version and it produced planes: a
    floor plus a wall rising linearly with distance is *exactly* a plane, so ``min(massif, wall)``
    faceted the whole world into flat triangles meeting at straight creases, with the mountains
    surviving only in the gaps. It looked like cut glass.

    A weight instead. Close to the line the weight is one and the ground is the valley floor —
    flat, and descending from head to mouth. Far away the weight is zero and the massif is
    untouched. The transition is Gaussian, so the walls curve into the ridges above them and the
    steepest grade is set by how far the valley's influence reaches rather than by a slope
    constant that never stops climbing.
    """
    d, t = _polyline_fields(x, y, points)
    floor = (np.float32(head_m) + np.float32(mouth_m - head_m) * t).astype(np.float32)
    reach = np.maximum(d - np.float32(floor_width_m * 0.5), 0.0) / np.float32(wall_width_m)
    return floor, np.exp(-reach * reach).astype(np.float32)


def _carve(surface: np.ndarray, valleys: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Replace the surface with valley floors, weighted.

    Where two valleys overlap their weights sum past one and the floor is their weighted mean,
    which is what makes a confluence a single continuous floor rather than a step between two.
    """
    if not valleys:
        return surface
    total = np.zeros_like(surface)
    weighted = np.zeros_like(surface)
    for floor, weight in valleys:
        total += weight
        weighted += weight * floor
    share = np.clip(total, 0.0, 1.0)
    floor_mix = weighted / np.maximum(total, np.float32(1e-6))
    return (share * floor_mix + (1.0 - share) * surface).astype(np.float32)


def _meander(
    rng: np.random.Generator,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    segments: int,
    amplitude: float,
) -> list[tuple[float, float]]:
    """A wandering path between two points. Valleys do not run straight, and a straight one would
    let the model off the geometric hook the experiment exists to set."""
    sx, sy = start
    ex, ey = end
    length = math.hypot(ex - sx, ey - sy) or 1.0
    nx, ny = -(ey - sy) / length, (ex - sx) / length          # unit normal

    offsets = rng.normal(0.0, amplitude, segments + 1)
    offsets[0] = offsets[-1] = 0.0
    # A rolling mean turns independent draws into a path that curves rather than zigzags.
    smooth = np.convolve(offsets, np.ones(3) / 3.0, mode="same")

    out = []
    for i, off in enumerate(smooth):
        f = i / segments
        out.append((sx + (ex - sx) * f + nx * off, sy + (ey - sy) * f + ny * off))
    return out


# ---------------------------------------------------------------------------------------------
# archetypes
# ---------------------------------------------------------------------------------------------


def _mountain_valley(p: TerrainParams, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    x, y = _grid(p)
    shape = (p.size_px, p.size_px)

    massif = (
        np.float32(p.base_elevation_m + p.relief_m * 0.55)
        + np.float32(p.relief_m * 0.45) * _ridged(rng, shape, p)
        + np.float32(p.noise_relief_m) * _fbm(rng, shape, p)
    )

    s = p.size_m
    trunk = _meander(rng, (s * 0.18, s * 0.02), (s * 0.72, s * 0.98),
                     segments=7, amplitude=s * 0.09)
    # The tributary meets the trunk at a point on it, so the confluence is a real junction of two
    # floors rather than two valleys that happen to overlap.
    join = trunk[len(trunk) // 2 + 1]
    tributary = _meander(rng, (s * 0.95, s * 0.24), join, segments=5, amplitude=s * 0.06)

    head = p.base_elevation_m + p.valley_drop_m
    surface = _carve(massif, [
        _valley(x, y, trunk, head_m=head, mouth_m=p.base_elevation_m,
                wall_width_m=p.valley_wall_width_m, floor_width_m=p.valley_floor_width_m),
        _valley(x, y, tributary,
                head_m=p.base_elevation_m + p.valley_drop_m + p.tributary_drop_m,
                mouth_m=head - 15.0,
                wall_width_m=p.valley_wall_width_m * 0.7,
                floor_width_m=p.valley_floor_width_m * 0.6),
    ])
    return surface, [
        "trunk valley " + " ".join(f"({a:.0f},{b:.0f})" for a, b in trunk),
        "tributary joins the trunk at "
        f"({join[0]:.0f},{join[1]:.0f}) from ({tributary[0][0]:.0f},{tributary[0][1]:.0f})",
    ]


def _coastal(p: TerrainParams, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    x, y = _grid(p)
    shape = (p.size_px, p.size_px)

    # The shoreline wanders in y, which is what makes headlands and bays rather than a ruled edge.
    wobble = _value_noise(rng, (p.size_px, 1), 6) * 2.0 - 1.0
    shore = np.float32(p.shoreline_x_m) + np.float32(p.shore_wobble_m) * wobble

    inland = x - shore                                   # <0 sea, >0 land
    terrace = np.clip(inland / np.float32(p.terrace_width_m), 0.0, 1.0)
    cliff = np.clip(
        (inland - np.float32(p.terrace_width_m)) / np.float32(p.cliff_width_m), 0.0, 1.0
    )
    cliff = cliff * cliff * (3.0 - 2.0 * cliff)          # smoothstep, so the cliff top rolls over

    # The upland is hills, not a plateau: without the ridged term the whole inland half is a
    # featureless shelf and the archetype stops testing anything except the cliff.
    upland = 0.45 + 0.55 * _ridged(rng, shape, p)
    land = (
        np.float32(p.terrace_height_m) * terrace
        + np.float32(p.relief_m) * cliff * upland
        + np.float32(p.noise_relief_m) * _fbm(rng, shape, p) * np.clip(cliff + 0.15, 0.0, 1.0)
    )
    sea = -np.float32(p.sea_depth_m) * np.clip(-inland / (np.float32(p.size_m) * 0.25), 0.0, 1.0)
    surface = np.where(inland >= 0.0, land, sea).astype(np.float32)

    # One corridor inland, so there is a route off the coast that is not a cliff climb.
    s = p.size_m
    corridor = _meander(rng, (float(p.shoreline_x_m), s * 0.55), (s * 0.98, s * 0.72),
                        segments=5, amplitude=s * 0.07)
    # Two gullies through the cliff line, so the terrace is not sealed off by a continuous wall.
    gullies = []
    for fy in (0.22, 0.83):
        mouth = (float(p.shoreline_x_m) + p.terrace_width_m * 0.4, s * fy)
        head = (s * 0.45, s * (fy + float(rng.uniform(-0.06, 0.06))))
        gullies.append(_meander(rng, mouth, head, segments=4, amplitude=s * 0.03))

    surface = _carve(surface, [
        _valley(x, y, corridor, head_m=p.relief_m * 0.55, mouth_m=p.terrace_height_m,
                wall_width_m=p.valley_wall_width_m, floor_width_m=p.valley_floor_width_m * 1.4),
        *[
            _valley(x, y, g, head_m=p.relief_m * 0.38, mouth_m=p.terrace_height_m * 0.6,
                    wall_width_m=p.valley_wall_width_m * 0.5,
                    floor_width_m=p.valley_floor_width_m * 0.7)
            for g in gullies
        ],
    ])
    return surface, [
        f"shoreline near x={p.shoreline_x_m:.0f} m, wandering +/-{p.shore_wobble_m:.0f} m",
        "inland corridor " + " ".join(f"({a:.0f},{b:.0f})" for a, b in corridor),
        f"{len(gullies)} gullies breaching the cliff line",
        "sea is terrain only: the land-use water channel is zero, as for every world here",
    ]


def _plain(p: TerrainParams, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    x, y = _grid(p)
    shape = (p.size_px, p.size_px)

    surface = (
        np.float32(p.base_elevation_m)
        + np.float32(p.noise_relief_m) * _fbm(rng, shape, p)
    )

    hills = []
    for _ in range(int(rng.integers(3, 6))):
        hx = float(rng.uniform(0.15, 0.85)) * p.size_m
        hy = float(rng.uniform(0.15, 0.85)) * p.size_m
        radius = float(rng.uniform(0.09, 0.18)) * p.size_m
        height = float(rng.uniform(0.4, 1.0)) * p.relief_m
        d2 = (x - hx) ** 2 + (y - hy) ** 2
        surface = surface + np.float32(height) * np.exp(-d2 / np.float32(2.0 * radius**2))
        hills.append(f"({hx:.0f},{hy:.0f}) r={radius:.0f} h={height:.0f}")

    # A shallow drainage line, because a plain with no low ground has no reason for a bridge or a
    # crossing and would flatter the model.
    creek = _meander(rng, (0.0, p.size_m * 0.35), (p.size_m, p.size_m * 0.62),
                     segments=6, amplitude=p.size_m * 0.05)
    surface = _carve(surface, [
        _valley(x, y, creek, head_m=p.base_elevation_m - 2.0, mouth_m=p.base_elevation_m - 6.0,
                wall_width_m=140.0, floor_width_m=90.0),
    ])
    return surface.astype(np.float32), ["hills: " + "; ".join(hills), "one shallow drainage line"]


def _basin(p: TerrainParams, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    x, y = _grid(p)
    shape = (p.size_px, p.size_px)

    cx = cy = p.size_m / 2.0
    dx, dy = x - np.float32(cx), y - np.float32(cy)
    r = np.hypot(dx, dy).astype(np.float32)
    bearing = np.degrees(np.arctan2(dx, dy)).astype(np.float32)

    # A circular rim is a crater, not a basin. Perturbing the radius at low frequency gives the
    # rim spurs and re-entrants, which is what makes one pass easier to reach than another.
    wobble = _fbm(rng, shape, TerrainParams(
        archetype=p.archetype, seed=p.seed, size_m=p.size_m, resolution_m=p.resolution_m,
        noise_octaves=3, noise_base_cells=3,
    ))
    r = r * (1.0 + np.float32(0.22) * wobble)

    wall = np.clip((r - np.float32(p.basin_radius_m)) / np.float32(p.basin_wall_width_m), 0.0, 1.0)
    wall = wall * wall * (3.0 - 2.0 * wall)

    # Each pass removes a fraction of the wall height over an angular window, so the passes sit at
    # different elevations and the choice between them is a real one.
    relief = np.full(shape, np.float32(p.relief_m))
    notes = []
    for deg, depth in zip(p.pass_bearings_deg, p.pass_depths_m, strict=False):
        delta = np.abs((bearing - np.float32(deg) + 180.0) % 360.0 - 180.0)
        gate = np.exp(-(delta**2) / np.float32(2.0 * p.pass_width_deg**2)).astype(np.float32)
        relief = relief * (1.0 - np.float32(depth) * gate)
        notes.append(f"pass at {deg:.0f} deg, wall cut to {(1 - depth) * 100:.0f}%"
                     f" (~{p.base_elevation_m + p.relief_m * (1 - depth):.0f} m)")

    # Ridges on the rim and beyond it, so the surrounding country is mountains rather than a
    # raised plain with a hole in it.
    crests = 0.55 + 0.45 * _ridged(rng, shape, p)
    surface = (
        np.float32(p.base_elevation_m)
        + relief * wall * crests
        + np.float32(p.noise_relief_m) * _fbm(rng, shape, p) * (0.15 + 0.85 * wall)
    )
    return surface.astype(np.float32), [
        f"basin floor radius {p.basin_radius_m:.0f} m, rim perturbed +/-22%", *notes,
    ]


_BUILDERS: dict[str, Callable[[TerrainParams, np.random.Generator], tuple[np.ndarray, list[str]]]] = {
    "mountain_valley": _mountain_valley,
    "coastal": _coastal,
    "plain": _plain,
    "basin": _basin,
}


def _detail(rng: np.random.Generator, shape: tuple[int, int], p: TerrainParams) -> np.ndarray:
    """Metre-scale undulation, applied over the finished landform."""
    fine = TerrainParams(
        archetype=p.archetype, seed=p.seed, size_m=p.size_m, resolution_m=p.resolution_m,
        noise_octaves=3, noise_base_cells=48, noise_gain=0.55,
    )
    return np.float32(p.detail_relief_m) * _fbm(rng, shape, fine)


def synthesise(params: TerrainParams, *, name: str = "") -> SyntheticWorld:
    """Build a world. The same parameters and seed always give the same heightfield."""
    builder = _BUILDERS[params.archetype]
    rng = np.random.default_rng(params.seed)
    elevation, notes = builder(params, rng)
    elevation = elevation + _detail(rng, elevation.shape, params)
    return SyntheticWorld(
        params=params,
        elevation=np.ascontiguousarray(elevation, dtype=np.float32),
        name=name or f"{params.archetype}_seed_{params.seed:03d}",
        notes=notes,
    )
