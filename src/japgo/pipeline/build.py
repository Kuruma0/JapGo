"""Region build orchestration.

Drives adapters across a region's tiles and writes a manifest-carrying tile set. This is the layer
Phase 1's exit criterion names: *one command produces a valid tile set for the MVP area*.

Source files are located by convention under a data root, so a build is reproducible from a site
name plus a directory rather than from a list of paths someone typed once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..core.manifest import SourceRole
from ..geo.crs import zone as plane_zone
from ..geo.tiling import Bounds, Tile, TileGrid
from ..geo.crs import from_wgs84
from ..provenance import SourceGate
from .assemble import TileAssembler, TileBundle, TileInputs
from .splits import Site, Split, make_split
from .store import write_attribution, write_index, write_tile

log = logging.getLogger(__name__)

DEFAULT_SITES_PATH = Path("config/sites.yaml")


class SiteSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    archetype: str
    bbox: tuple[float, float, float, float]
    municipality: str | None = None
    note: str | None = None

    plateau_url: str | None = None
    """CityGML package for this site's municipality, read remotely by ``--remote``.

    Recorded per site because the URL is per municipality and per vintage, and the vintage is part
    of what makes a build reproducible (invariant 8). Resolve a new one from the G-Spatial CKAN
    API: ``package_show?id=plateau-<code>-<name>-shi-<year>``.
    """


class SitesConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    sites_version: int
    zone: int
    sites: dict[str, SiteSpec]
    default_split: dict[str, str] = Field(default_factory=dict)

    def bounds_of(self, name: str) -> Bounds:
        """Site bounds projected into the configured zone."""
        spec = self.sites[name]
        crs = plane_zone(self.zone).crs
        minlon, minlat, maxlon, maxlat = spec.bbox
        x0, y0 = from_wgs84(minlon, minlat, crs)
        x1, y1 = from_wgs84(maxlon, maxlat, crs)
        return Bounds(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _find_sites(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        path = candidate / DEFAULT_SITES_PATH
        if path.is_file():
            return path
    raise FileNotFoundError(f"could not find {DEFAULT_SITES_PATH} walking up from {here}")


@lru_cache(maxsize=4)
def load_sites(path: Path | None = None) -> SitesConfig:
    path = path or _find_sites()
    return SitesConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# ------------------------------------------------------------------------------------------------


@dataclass
class SourceFiles:
    """Where a build finds its inputs.

    Paths are optional: a build with only terrain still produces valid tiles, just thinner ones.
    Refusing to build without every source would make the pipeline untestable until every download
    finished, which is how a pipeline stays unrun for a month.
    """

    terrain: list[Path] = field(default_factory=list)
    plateau: list[Path] = field(default_factory=list)
    landuse: list[Path] = field(default_factory=list)
    roads: list[Path] = field(default_factory=list)

    @classmethod
    def discover(cls, root: Path, site: str) -> SourceFiles:
        """Locate source files by convention under ``root/<site>/``.

        ``terrain/*.las``, ``plateau/*.gml``, ``landuse/*.geojson|*.shp``, ``roads/*.osm``.
        """
        base = Path(root) / site
        return cls(
            terrain=sorted(base.glob("terrain/*.la[sz]")),
            plateau=sorted(base.glob("plateau/**/*.gml")),
            landuse=sorted([*base.glob("landuse/*.geojson"), *base.glob("landuse/*.shp")]),
            roads=sorted(base.glob("roads/*.osm")),
        )

    @property
    def is_empty(self) -> bool:
        return not (self.terrain or self.plateau or self.landuse or self.roads)

    def describe(self) -> str:
        return (
            f"terrain={len(self.terrain)} plateau={len(self.plateau)} "
            f"landuse={len(self.landuse)} roads={len(self.roads)}"
        )


@dataclass
class BuildReport:
    site: str
    tiles_written: list[str] = field(default_factory=list)
    tiles_skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attribution: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.tiles_written)


class TileInputSource(Protocol):
    """Where a build's per-tile inputs come from.

    Two implementations: :class:`_FileSource` over staged local files, and
    :class:`~japgo.pipeline.remote.RemoteSources` over the published endpoints. An optional
    ``prepare(tiles)`` lets a provider fetch region-wide inputs once before the loop starts.
    """

    def inputs_for(self, tile: Tile) -> TileInputs | None: ...


class _FileSource:
    """Adapts the staged-files path to :class:`TileInputSource`."""

    def __init__(self, builder: RegionBuilder, files: SourceFiles) -> None:
        self._builder = builder
        self._shared = builder._read_shared(files)

    def inputs_for(self, tile: Tile) -> TileInputs | None:
        return self._builder._inputs_for(tile, self._shared)


class RegionBuilder:
    """Builds a tile set for one site."""

    def __init__(
        self,
        gate: SourceGate,
        *,
        sites: SitesConfig | None = None,
        resolution: float = 1.0,
        min_coverage: float = 0.5,
    ) -> None:
        self.gate = gate
        self.sites = sites or load_sites()
        self.zone = plane_zone(self.sites.zone)
        self.grid = TileGrid(self.zone)
        self.assembler = TileAssembler(gate, resolution=resolution)
        self.min_coverage = min_coverage
        """Fraction of a tile that must carry real observations before it is written.

        A tile that is mostly void is not a hard failure — LiDAR occlusion and water absorption
        leave genuine gaps — but it is useless for training, and writing it anyway pollutes the
        dataset with samples the loss will mostly mask out. Rejecting at build time is far cheaper
        than discovering it as an unexplained metric during Phase 4.
        """

    # -----------------------------------------------------------------------------------------

    def tiles_for(self, site: str, *, bounds: Bounds | None = None) -> list[Tile]:
        """Tiles covering a site, or covering ``bounds`` if given.

        The override exists for smoke runs over a known-interesting sub-extent. Site bboxes in
        ``config/sites.yaml`` are deliberately approximate pending first ingest, and a ten-tile
        run starting at the corner of an approximate box is usually ten tiles of sea.
        """
        if site not in self.sites.sites:
            raise KeyError(f"unknown site {site!r}; have {sorted(self.sites.sites)}")
        return list(self.grid.tiles_covering(bounds or self.sites.bounds_of(site)))

    def site_record(self, site: str, tile_ids: list[str]) -> Site:
        return Site(
            name=site,
            archetype=self.sites.sites[site].archetype,
            tiles=frozenset(tile_ids),
        )

    def build(
        self,
        site: str,
        files: SourceFiles,
        out_root: Path,
        *,
        limit: int | None = None,
    ) -> BuildReport:
        """Build and write every tile for a site."""
        if files.is_empty:
            report = BuildReport(site=site)
            report.warnings.append(
                f"no source files found for {site}. Expected them under "
                f"<root>/{site}/{{terrain,plateau,landuse,roads}}/"
            )
            return report

        return self.build_from(site, _FileSource(self, files), out_root, limit=limit)

    def build_from(
        self,
        site: str,
        source: TileInputSource,
        out_root: Path,
        *,
        limit: int | None = None,
        bounds: Bounds | None = None,
    ) -> BuildReport:
        """Build a site from any input provider.

        Split out from :meth:`build` so that where the bytes come from — staged files or the
        published sources over the network — is the only thing that varies. Coverage gating,
        manifest writing, attribution and the one-bad-tile-does-not-abort-a-region behaviour are
        identical either way, and duplicating them would be how the two paths drift apart.
        """
        report = BuildReport(site=site)
        tiles = self.tiles_for(site, bounds=bounds)
        if limit is not None:
            tiles = tiles[:limit]

        prepare = getattr(source, "prepare", None)
        if prepare is not None:
            prepare(tiles)

        for tile in tiles:
            try:
                inputs = source.inputs_for(tile)
                if inputs is None:
                    report.tiles_skipped.append(tile.id)
                    continue
                bundle = self.assembler.assemble(tile, inputs)

                if bundle.coverage < self.min_coverage:
                    report.tiles_skipped.append(tile.id)
                    report.warnings.append(
                        f"{tile.id}: coverage {bundle.coverage:.1%} is below the "
                        f"{self.min_coverage:.0%} minimum; not written"
                    )
                    continue

                write_tile(out_root, bundle)
                report.tiles_written.append(tile.id)
                report.warnings.extend(f"{tile.id}: {w}" for w in bundle.warnings)
            except Exception as exc:  # noqa: BLE001 - one bad tile must not abort a region
                log.exception("tile %s failed", tile.id)
                report.tiles_skipped.append(tile.id)
                report.warnings.append(f"{tile.id}: {type(exc).__name__}: {exc}")

        if report.tiles_written:
            report.attribution = self._attribution(out_root, report.tiles_written)
            write_index(out_root, report.tiles_written)

        return report

    # -----------------------------------------------------------------------------------------

    def _read_shared(self, files: SourceFiles) -> dict:
        """Read whole-region sources once rather than per tile.

        PLATEAU and OSM files cover many tiles; re-parsing them per tile would dominate runtime.
        Terrain is the exception — LAS files are large and read per tile with a window.
        """
        from ..sources import OsmAdapter, PlateauAdapter

        shared: dict = {"buildings": [], "roads": None, "records": []}

        if files.plateau:
            adapter = PlateauAdapter(self.gate, target_crs=self.zone.crs)
            for path in files.plateau:
                result = adapter.read(path)
                shared["buildings"].extend(result.layers.get("buildings", []))
            shared["records"].append(
                adapter.make_record(layers=["buildings"], note=f"{len(files.plateau)} file(s)")
            )

        if files.roads:
            adapter = OsmAdapter(self.gate, target_crs=self.zone.crs)
            graphs = [
                adapter.read(path, purpose="training").layers["roads"][0] for path in files.roads
            ]
            shared["roads"] = _merge_graphs(graphs)
            shared["records"].append(
                adapter.make_record(layers=["roads"], note=f"{len(files.roads)} file(s)").model_copy(
                    update={"role": SourceRole.TARGET}
                )
            )

        shared["terrain_files"] = files.terrain
        shared["terrain_extents"] = {
            path: _las_extent(path) for path in files.terrain
        }
        # One adapter for the whole region, not one per tile: its point cache is what turns a
        # per-tile full parse of a ~300 MB file into a single parse per file.
        if files.terrain:
            from ..sources import VirtualShizuokaAdapter

            shared["terrain_adapter"] = VirtualShizuokaAdapter(
                self.gate, target_crs=self.zone.crs
            )

        shared["landuse_files"] = files.landuse
        return shared

    def _inputs_for(self, tile: Tile, shared: dict) -> TileInputs | None:
        from ..sources import NlniLanduseAdapter, VirtualShizuokaAdapter

        bounds = tile.read
        inputs = TileInputs()

        # Terrain: first file that actually covers the tile wins.
        #
        # The header bbox is checked before opening the point records. Without this the builder
        # re-parses every LAS file for every tile — with ~300 MB files and a few hundred tiles per
        # site that is hours of pure waste, and the read is discarded when it does not overlap.
        terrain_adapter = shared.get("terrain_adapter")
        for path in shared["terrain_files"]:
            extent = shared["terrain_extents"].get(path)
            if extent is not None and not extent.intersects(bounds):
                continue
            try:
                adapter = terrain_adapter or VirtualShizuokaAdapter(
                    self.gate, target_crs=self.zone.crs
                )
                result = adapter.read(path, bounds=bounds, resolution=self.assembler.resolution)
                inputs.elevation = result.layers["elevation"][0]
                inputs.add_record(result.record)
                break
            except ValueError:
                continue  # overlaps the header bbox but holds no points here

        if inputs.elevation is None:
            return None  # the assembler needs a CRS reference; a tile with no terrain is skipped

        inputs.buildings = [
            b for b in shared["buildings"] if _footprint_intersects(b.footprint, bounds)
        ]

        for path in shared["landuse_files"]:
            adapter = NlniLanduseAdapter(self.gate, target_crs=self.zone.crs)
            result = adapter.read(path, bounds=bounds, resolution=self.assembler.resolution)
            inputs.landuse.update({k: v[0] for k, v in result.layers.items()})
            inputs.add_record(result.record)
            break

        if shared["roads"] is not None:
            inputs.roads = _clip_graph(shared["roads"], bounds)

        for record in shared["records"]:
            if record.source_id not in {r.source_id for r in inputs.records}:
                inputs.records.append(record)

        return inputs

    def _attribution(self, out_root: Path, tile_ids: list[str]) -> list[str]:
        from .store import read_tile

        lines: list[str] = []
        for tile_id in tile_ids[:1]:  # every tile in a site shares its sources
            bundle = read_tile(out_root, tile_id)
            lines = bundle.attribution(self.gate)
        if lines:
            write_attribution(out_root, lines)
        return lines


# ------------------------------------------------------------------------------------------------


def _las_extent(path: Path) -> Bounds | None:
    """Read a LAS/LAZ bounding box from its header, without touching the point records.

    Cheap enough to call for every file at the start of a build, and it turns a per-tile full
    parse into a per-tile integer comparison.
    """
    try:
        import laspy

        with laspy.open(path) as reader:
            header = reader.header
            return Bounds(
                float(header.mins[0]),
                float(header.mins[1]),
                float(header.maxs[0]),
                float(header.maxs[1]),
            )
    except Exception as exc:  # noqa: BLE001 - a bad header must not abort the build
        log.warning("could not read header extent from %s: %s", path, exc)
        return None


def _footprint_intersects(ring, bounds: Bounds) -> bool:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return not (
        max(xs) < bounds.minx
        or min(xs) > bounds.maxx
        or max(ys) < bounds.miny
        or min(ys) > bounds.maxy
    )


def _merge_graphs(graphs):
    from ..core.roads import RoadGraph

    if not graphs:
        return None
    merged = RoadGraph(crs=graphs[0].crs)
    for graph in graphs:
        for node in graph.nodes.values():
            merged.nodes.setdefault(node.id, node)
        for edge in graph.edges.values():
            merged.edges.setdefault(edge.id, edge)
    return merged


def _clip_graph(graph, bounds: Bounds):
    """Keep edges whose geometry touches ``bounds``.

    Clipping by bounding box rather than by exact intersection: an edge crossing the halo should be
    kept whole, because the halo exists precisely so that features continue past the tile edge.
    """
    from ..core.roads import RoadGraph

    out = RoadGraph(crs=graph.crs, tile_id=None, lod_level=graph.lod_level)
    for edge in graph.edges.values():
        if not _footprint_intersects(edge.geometry, bounds):
            continue
        for node_id in (edge.u, edge.v):
            if node_id in graph.nodes and node_id not in out.nodes:
                out.add_node(graph.nodes[node_id])
        out.add_edge(edge)
    return out


def build_default_split(builder: RegionBuilder, written: dict[str, list[str]]):
    """Build the split defined in ``config/sites.yaml`` from what was actually written."""
    sites = {
        name: builder.site_record(name, tile_ids)
        for name, tile_ids in written.items()
    }
    assignment = {
        name: Split(builder.sites.default_split[name])
        for name in sites
        if name in builder.sites.default_split
    }
    return make_split(sites, assignment)
