"""Tile assembly.

Where the provenance gate, the tile grid and the source adapters meet. The assembler takes a
:class:`~japgo.geo.tiling.Tile` and the layers read for it, and produces a
:class:`TileBundle`: the raster stack, the vector layers, and a manifest that was *computed* from
what actually went in rather than declared alongside it.

The whole point of computing the manifest here is that ``redistribution_class`` then cannot drift
from reality. If a training-only source contributes a single channel, the bundle knows it, and the
exporter will refuse to ship it as attribution-only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..core.buildings import Building
from ..core.manifest import SourceRecord, TileManifest
from ..core.roads import RoadGraph
from ..geo import terrain as terrain_ops
from ..geo.raster import Raster
from ..geo.tiling import Tile
from ..provenance import SourceGate, registry_hash
from .channels import Channel, StackSpec, load_stack_spec
from . import rasterize

log = logging.getLogger(__name__)

PREPROCESSING_VERSION = "1"


@dataclass
class TileInputs:
    """The raw layers read for one tile, keyed by the source that produced them."""

    elevation: Raster | None = None
    buildings: list[Building] = field(default_factory=list)
    landuse: dict[str, Raster] = field(default_factory=dict)
    """Channel-group name -> coverage raster, as produced by the NLNI adapter."""

    roads: RoadGraph | None = None
    """The prediction target. OSM-derived and therefore training-only."""

    records: list[SourceRecord] = field(default_factory=list)

    def add_record(self, record: SourceRecord | None) -> None:
        if record is not None:
            self.records.append(record)


@dataclass
class TileBundle:
    """An assembled tile: stack, vectors, manifest."""

    tile: Tile
    stack: np.ndarray
    """``(channels, rows, cols)``, float32, normalised, NaN-free."""

    spec: StackSpec
    manifest: TileManifest
    buildings: list[Building] = field(default_factory=list)
    roads: RoadGraph | None = None
    targets: np.ndarray | None = None
    """``(targets, rows, cols)``, or ``None`` when the tile carries inputs only."""

    warnings: list[str] = field(default_factory=list)

    with_halo: bool = True
    """Whether ``stack`` covers the read extent (core plus halo) or the core alone.

    Recorded rather than inferred. The two cases are indistinguishable from the array shape alone
    — a 1000² stack is either a 1 km core at 1 m or a read extent at 1.512 m — and a consumer that
    guesses wrong crops a core-only tile down to its middle and reports statistics over 44% of the
    ground without failing. Same reasoning as ``core_size_m``/``halo_m`` on the manifest: geometry
    the reader needs is carried, not reconstructed.
    """

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.stack.shape)  # type: ignore[return-value]

    def channel(self, name: str) -> np.ndarray:
        return self.stack[self.spec.index_of(name)]

    def target(self, name: str) -> np.ndarray:
        if self.targets is None:
            raise ValueError(
                f"tile {self.tile.id} carries no targets. Pass a road graph in TileInputs to "
                "build a training pair."
            )
        return self.targets[self.spec.target_index_of(name)]

    @property
    def has_targets(self) -> bool:
        return self.targets is not None

    @property
    def is_trainable(self) -> bool:
        """A tile is trainable only if it has both inputs and targets."""
        return self.has_targets and self.coverage > 0.0

    @property
    def coverage(self) -> float:
        """Fraction of the tile with real observations."""
        return float(self.channel("valid").mean())

    def redistribution_class(self, gate: SourceGate, *, role=None) -> str:
        return self.manifest.redistribution_class(gate.registry, role=role)

    def input_redistribution_class(self, gate: SourceGate) -> str:
        """What may be shipped from this tile's *inputs* — the question that governs export."""
        from ..core.manifest import SourceRole

        return self.manifest.redistribution_class(gate.registry, role=SourceRole.INPUT)

    def attribution(self, gate: SourceGate) -> list[str]:
        return self.manifest.attribution(gate)


class TileAssembler:
    """Composes source layers into a model-ready tile."""

    def __init__(
        self,
        gate: SourceGate,
        *,
        spec: StackSpec | None = None,
        resolution: float = 1.0,
    ) -> None:
        self.gate = gate
        self.spec = spec or load_stack_spec()
        self.resolution = resolution

    # -----------------------------------------------------------------------------------------

    def assemble(self, tile: Tile, inputs: TileInputs, *, with_halo: bool = True) -> TileBundle:
        """Build one tile.

        The stack is rasterised over the tile's **read** extent (core plus halo) by default. The
        halo is read but never predicted into — it is what stops roads dead-ending at tile seams,
        and it must be present from the first ingest because adding it later invalidates every
        cached sample.
        """
        bounds = tile.read if with_halo else tile.core
        crs = self._crs_from(inputs)
        warnings: list[str] = []

        planes: dict[str, np.ndarray] = {}
        valid = np.ones(self._shape(bounds), dtype=bool)

        # --- terrain ---------------------------------------------------------------------
        if inputs.elevation is not None:
            dem = self._align(inputs.elevation, bounds, crs)
            planes.update(self._terrain_planes(dem))
            valid &= ~np.isnan(dem.data)
        else:
            warnings.append("no elevation input; terrain channels are zero and valid is unset")
            valid[:] = False

        # --- built form ------------------------------------------------------------------
        if inputs.buildings:
            planes.update(self._building_planes(inputs.buildings, bounds, crs))
        elif "plateau" in self.spec.required_sources:
            # An empty tile is legitimate — forest, water, farmland. Recorded, not warned about,
            # because treating "no buildings here" as an error would reject exactly the rural
            # tiles the Kawanehon site depends on.
            planes.update(self._empty_building_planes(bounds))

        # --- land use --------------------------------------------------------------------
        if inputs.landuse:
            for name, raster in inputs.landuse.items():
                if name in self.spec.names:
                    planes[name] = self._align(raster, bounds, crs).data

        # --- targets ---------------------------------------------------------------------
        targets = None
        if inputs.roads is not None:
            targets = self._target_planes(inputs.roads, bounds, crs)

        # --- assemble --------------------------------------------------------------------
        stack = self._stack_planes(planes, valid, bounds)
        manifest = self._manifest(tile, inputs, crs)

        for w in warnings:
            log.warning("%s: %s", tile.id, w)

        return TileBundle(
            tile=tile,
            stack=stack,
            spec=self.spec,
            manifest=manifest,
            buildings=list(inputs.buildings),
            roads=inputs.roads,
            targets=targets,
            warnings=warnings,
            with_halo=with_halo,
        )

    # -----------------------------------------------------------------------------------------

    def _shape(self, bounds) -> tuple[int, int]:
        return (
            int(round(bounds.height / self.resolution)),
            int(round(bounds.width / self.resolution)),
        )

    def _crs_from(self, inputs: TileInputs):
        if inputs.elevation is not None:
            return inputs.elevation.crs
        raise ValueError(
            "cannot determine the tile CRS: no elevation input. Pass at least one raster layer, "
            "or set the CRS explicitly."
        )

    def _align(self, raster: Raster, bounds, crs) -> Raster:
        """Bring a source raster onto the tile grid."""
        rows, cols = self._shape(bounds)

        if raster.data.shape == (rows, cols) and np.allclose(
            raster.bounds.as_tuple(), bounds.as_tuple()
        ):
            return raster

        out = Raster.empty(bounds, self.resolution, crs)
        target = out.data

        # Nearest-neighbour resample. Adequate because sources are already at or finer than the
        # working resolution; a tile needing genuine upsampling is a data gap, not a smoothing
        # problem, and `valid` will say so.
        rr = np.arange(rows)
        cc = np.arange(cols)
        ys = bounds.maxy - (rr + 0.5) * self.resolution
        xs = bounds.minx + (cc + 0.5) * self.resolution

        src_rows = np.floor((raster.bounds.maxy - ys) / raster.res_y).astype(np.int64)
        src_cols = np.floor((xs - raster.bounds.minx) / raster.res_x).astype(np.int64)

        row_ok = (src_rows >= 0) & (src_rows < raster.height)
        col_ok = (src_cols >= 0) & (src_cols < raster.width)

        rows_idx = np.where(row_ok, src_rows, 0)
        cols_idx = np.where(col_ok, src_cols, 0)

        sampled = raster.data[np.ix_(rows_idx, cols_idx)]
        target[:] = np.where(np.outer(row_ok, col_ok), sampled, np.nan)
        return out

    def _terrain_planes(self, dem: Raster) -> dict[str, np.ndarray]:
        aspect_deg = terrain_ops.aspect(dem).data
        aspect_rad = np.radians(aspect_deg)

        # Flat cells have no aspect. sin/cos of NaN would poison the stack, so flats resolve to
        # (0, 0) — a vector of zero length, which is the honest encoding of "no direction".
        flat = np.isnan(aspect_rad)

        return {
            "elevation": dem.data,
            "slope": terrain_ops.slope(dem, as_percent=True).data,
            "aspect_sin": np.where(flat, 0.0, np.sin(aspect_rad)),
            "aspect_cos": np.where(flat, 0.0, np.cos(aspect_rad)),
            "roughness": terrain_ops.roughness(dem).data,
        }

    def _building_planes(self, buildings, bounds, crs) -> dict[str, np.ndarray]:
        planes = {
            "building_mask": rasterize.building_mask(buildings, bounds, self.resolution, crs).data,
            "building_height": rasterize.building_height(
                buildings, bounds, self.resolution, crs
            ).data,
        }
        for coarse in ("residential", "commercial", "industrial"):
            name = f"building_coarse_{coarse}"
            if name in self.spec.names:
                planes[name] = rasterize.building_class_mask(
                    buildings, coarse, bounds, self.resolution, crs
                ).data
        return planes

    def _target_planes(self, graph: RoadGraph, bounds, crs) -> np.ndarray:
        """Rasterise the road graph into the target stack."""
        rows, cols = self._shape(bounds)
        out = np.zeros((self.spec.target_depth, rows, cols), np.float32)

        mask = rasterize.road_mask(graph, bounds, self.resolution, crs)
        centreline = rasterize.road_mask(graph, bounds, self.resolution, crs, use_width=False)
        classes = rasterize.road_class_raster(graph, bounds, self.resolution, crs)
        sin_r, cos_r = rasterize.road_orientation(graph, bounds, self.resolution, crs)

        planes = {
            "road_mask": mask.data,
            "road_centreline": centreline.data,
            "road_class": classes.data,
            "road_orientation_sin": sin_r.data,
            "road_orientation_cos": cos_r.data,
        }

        for i, channel in enumerate(self.spec.targets):
            plane = planes.get(channel.name)
            if plane is None:
                continue
            out[i] = np.nan_to_num(
                channel.apply_normalisation(np.asarray(plane, np.float32)),
                nan=self.spec.nodata_fill,
                posinf=self.spec.nodata_fill,
                neginf=self.spec.nodata_fill,
            )
        return out

    def _empty_building_planes(self, bounds) -> dict[str, np.ndarray]:
        zeros = np.zeros(self._shape(bounds), np.float32)
        return {
            c.name: zeros.copy() for c in self.spec.channels if c.source == "plateau"
        }

    def _stack_planes(self, planes: dict[str, np.ndarray], valid: np.ndarray, bounds) -> np.ndarray:
        rows, cols = self._shape(bounds)
        out = np.zeros((self.spec.depth, rows, cols), np.float32)

        for i, channel in enumerate(self.spec.channels):
            if channel.name == "valid":
                out[i] = valid.astype(np.float32)
                continue

            plane = planes.get(channel.name)
            if plane is None:
                continue  # left at nodata_fill

            normalised = channel.apply_normalisation(np.asarray(plane, np.float32))
            out[i] = np.nan_to_num(
                normalised,
                nan=self.spec.nodata_fill,
                posinf=self.spec.nodata_fill,
                neginf=self.spec.nodata_fill,
            )

        return out

    def _manifest(self, tile: Tile, inputs: TileInputs, crs) -> TileManifest:
        manifest = TileManifest(
            tile_id=tile.id,
            zone=tile.zone,
            crs=crs.to_string(),
            core_size_m=tile.core_size_m,
            halo_m=tile.halo_m,
            registry_hash=registry_hash(),
            preprocessing_version=PREPROCESSING_VERSION,
        )
        for record in inputs.records:
            manifest.add(record)

        # Re-verify against current policy. A tile legitimate when built can become illegitimate
        # if a licence is later reclassified, and that must surface here rather than never.
        manifest.verify(self.gate)
        return manifest


def channel_summary(bundle: TileBundle) -> list[tuple[str, float, float, float]]:
    """(name, min, mean, max) per channel. For the Phase 2 visualiser and for eyeballing a build."""
    return [
        (
            c.name,
            float(bundle.stack[i].min()),
            float(bundle.stack[i].mean()),
            float(bundle.stack[i].max()),
        )
        for i, c in enumerate(bundle.spec.channels)
    ]
