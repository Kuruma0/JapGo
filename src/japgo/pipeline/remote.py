"""Building tiles from the published sources directly, without staging them on disk.

The counterpart to :class:`~japgo.pipeline.build.SourceFiles`. That path expects every input
already downloaded and unpacked locally, which for this project's two major sources is the
expensive way round: a PLATEAU municipality ZIP is ~15 GB for ~200 MB of building GML, and VIRTUAL
SHIZUOKA's Grid text is ~58× the raster it becomes. AGENTS.md names staging them as a trap costing
43 GB per 100-tile site against 0.75 GB.

The fetchers that avoid it already existed and were tested; nothing composed them. This module is
that composition:

* **terrain** — :class:`~japgo.sources.meshindex.TerrainFetcher` streams each 0.5 m mesh, grids it
  in memory and caches only the raster.
* **buildings** — :class:`~japgo.sources.fetch.ArchiveFetcher` range-reads the CityGML members the
  tiles actually need, selected by the JIS mesh code in each member's name.
* **roads** — :class:`~japgo.sources.overpass.OverpassClient`, one query per region, cached.

Region-wide inputs are fetched once in :meth:`RemoteSources.prepare`; only terrain is per tile,
because terrain genuinely is. Everything persisted is a cache keyed by content, so a rebuild makes
no network requests at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..core.manifest import SourceRecord, SourceRole
from ..geo.crs import to_wgs84
from ..geo.tiling import Bounds, Tile
from ..provenance import SourceGate
from .assemble import TileInputs

log = logging.getLogger(__name__)

BLDG_MEMBER = r"udx/bldg/.*\.gml$"
"""Building models only. A PLATEAU package also carries terrain, disaster and land-use models,
and pulling those is most of why the archive is 15 GB."""

LANDUSE_STEM = "L03-b-16"
LANDUSE_URL = (
    "https://nlftp.mlit.go.jp/ksj/gml/data/L03-b/{stem}/{stem}_{mesh}-jgd_GML.zip"
)
"""FY2016 土地利用細分メッシュ, one archive per primary mesh.

``-jgd`` not ``-tky``: the same data is published in both JGD2000 and the old Tokyo datum, and the
two differ by a few hundred metres. Taking the wrong one misaligns land use against everything
else by more than a tile's halo — visible in the Phase 2 report, but only if someone looks.
"""


@dataclass
class RemoteReport:
    """What the network actually cost, so the claim in AGENTS.md stays checkable."""

    archive_bytes: int = 0
    archive_total: int = 0
    roads_bytes: int = 0
    landuse_bytes: int = 0
    members: list[str] = field(default_factory=list)
    meshes: set[str] = field(default_factory=set)
    cache_hits: int = 0

    def describe(self) -> str:
        share = (
            f" ({100.0 * self.archive_bytes / self.archive_total:.4f}% of archive)"
            if self.archive_total
            else ""
        )
        return (
            f"plateau {self.archive_bytes / 1e6:.1f} MB{share} over {len(self.members)} member(s); "
            f"roads {self.roads_bytes / 1e6:.1f} MB; landuse {self.landuse_bytes / 1e6:.1f} MB; "
            f"terrain meshes {len(self.meshes)}"
        )


class RemoteSources:
    """A :class:`~japgo.pipeline.build.RegionBuilder` input provider that fetches on demand."""

    def __init__(
        self,
        gate: SourceGate,
        crs,
        *,
        cache_dir: Path,
        plateau_url: str | None = None,
        resolution: float = 1.0,
        overpass_endpoint: str | None = None,
    ) -> None:
        self.gate = gate
        self.crs = crs
        self.cache_dir = Path(cache_dir)
        self.plateau_url = plateau_url
        self.resolution = resolution
        self.overpass_endpoint = overpass_endpoint

        self.report = RemoteReport()
        self._buildings: list = []
        self._landuse: list = []
        """``(primary mesh square, shapefile path)``, chosen per tile by which square holds it."""
        self._roads = None
        self._records: list[SourceRecord] = []
        self._prepared = False
        self._terrain = None
        """One fetcher for the whole run, not one per tile.

        Its :class:`~japgo.sources.meshindex.MeshIndex` holds the vector-tile index of which mesh
        lives where; rebuilding it per tile re-fetches that index for every tile in the site.
        """

    # -- region-wide inputs -----------------------------------------------------------------

    def prepare(self, tiles: list[Tile]) -> RemoteReport:
        """Fetch everything that covers more than one tile, once."""
        if self._prepared:
            return self.report
        if not tiles:
            raise ValueError("prepare() needs at least one tile to know what to fetch")

        extent = _union(t.read for t in tiles)
        self._fetch_buildings(extent)
        self._fetch_landuse(extent)
        self._fetch_roads(extent)
        self._prepared = True
        return self.report

    def _fetch_landuse(self, extent: Bounds) -> None:
        """Download the NLNI land use mesh for each primary mesh the tiles touch.

        Pinned to the FY2016 vintage: it is the newest published *and* the only one the datalist
        marks as open data — see the ``nlni_landuse`` registry entry. The URL carries the vintage,
        so a registry change and a silent data change cannot come apart.
        """
        from ..sources.jismesh import decode, primary_meshes_for

        self.gate.assert_ingestible("nlni_landuse")
        west, south, east, north = _wgs84_box(extent, self.crs)
        destination = self.cache_dir / "landuse"

        for code in primary_meshes_for(west, south, east, north):
            shapefile = destination / f"{LANDUSE_STEM}_{code}.shp"
            if shapefile.is_file():
                self.report.cache_hits += 1
            else:
                fetched = _fetch_landuse_mesh(code, destination)
                if fetched is None:
                    continue
                self.report.landuse_bytes += fetched
            self._landuse.append((decode(code), shapefile))

        log.info("landuse: %d primary mesh file(s)", len(self._landuse))

    def _fetch_buildings(self, extent: Bounds) -> None:
        if not self.plateau_url:
            log.info("no PLATEAU url given; building channels will be empty")
            return

        from ..sources import ArchiveFetcher, PlateauAdapter
        from ..sources.jismesh import code_in, decode

        destination = self.cache_dir / "plateau"
        west, south, east, north = _wgs84_box(extent, self.crs)

        fetcher = ArchiveFetcher(self.gate, "plateau")
        cached = sorted(destination.glob("*.gml"))
        if cached:
            log.info("plateau: %d member(s) already cached", len(cached))
            self.report.cache_hits += len(cached)
            paths = cached
        else:
            members = fetcher.list_members(self.plateau_url, pattern=BLDG_MEMBER)
            wanted = [
                m.name
                for m in members
                if (code := code_in(Path(m.name).name)) is None
                or decode(code).intersects(west, south, east, north)
            ]
            log.info(
                "plateau: %d of %d building members cover the tiles", len(wanted), len(members)
            )
            if not wanted:
                # Loud, because the alternative is a corpus that builds cleanly with every
                # building channel zero. That looks like "this area has no buildings" rather than
                # "the tiles and the package do not overlap", and the two are indistinguishable
                # downstream — the study would report a real-looking null for built form.
                log.warning(
                    "plateau: none of %d building members intersect the tiles "
                    "(tiles span lon %.4f..%.4f lat %.4f..%.4f). Building channels will be "
                    "empty. Check the site extent against the municipality's coverage.",
                    len(members), west, east, south, north,
                )
                self.report.members = []
                return

            # An anchored alternation of the exact member names: the fetcher takes a pattern, and
            # a pattern is how we avoid pulling the members we just decided we do not need.
            pattern = "(?:%s)$" % "|".join(_escape(n) for n in wanted)
            fetched = fetcher.extract(self.plateau_url, destination, pattern=pattern)
            self.report.archive_bytes += fetched.bytes_fetched
            self.report.archive_total = fetched.archive_size
            self.report.members = list(fetched.members)
            paths = sorted(destination.glob("*.gml"))

        adapter = PlateauAdapter(self.gate, target_crs=self.crs)
        for path in paths:
            self._buildings.extend(adapter.read(path).layers.get("buildings", []))
        if paths:
            self._records.append(
                adapter.make_record(layers=["buildings"], note=f"{len(paths)} member(s), remote")
            )
        log.info("plateau: %d buildings", len(self._buildings))

    def _fetch_roads(self, extent: Bounds) -> None:
        from ..sources import OsmAdapter
        from ..sources.overpass import OverpassClient

        client = OverpassClient(
            self.gate,
            cache_dir=self.cache_dir / "osm",
            **({"endpoint": self.overpass_endpoint} if self.overpass_endpoint else {}),
        )
        fetched = client.fetch(extent, self.crs, purpose="training", key="region")
        self.report.roads_bytes += fetched.bytes_fetched
        if fetched.from_cache:
            self.report.cache_hits += 1

        adapter = OsmAdapter(self.gate, target_crs=self.crs)
        self._roads = adapter.read(fetched.path, purpose="training").layers["roads"][0]
        self._records.append(
            adapter.make_record(layers=["roads"], note="overpass").model_copy(
                update={"role": SourceRole.TARGET}
            )
        )
        log.info("overpass: %d edges", len(self._roads.edges))

    # -- per-tile ------------------------------------------------------------------------------

    def describe(self) -> str:
        return self.report.describe()

    def inputs_for(self, tile: Tile) -> TileInputs | None:
        """Assemble one tile's inputs, fetching only its terrain."""
        if not self._prepared:
            self.prepare([tile])

        if self._terrain is None:
            from ..sources import TerrainFetcher

            self._terrain = TerrainFetcher(self.gate, cache_dir=self.cache_dir / "terrain")

        dem, meshes = self._terrain.dem_for(
            tile.read, self.crs, resolution=self.resolution, key=tile.id
        )
        if not meshes:
            log.warning("%s: no terrain meshes cover this tile", tile.id)
            return None
        self.report.meshes.update(meshes)

        records = [
            SourceRecord(
                source_id="virtual_shizuoka",
                layers=["elevation"],
                note=f"{len(meshes)} mesh(es)",
            ),
            *self._records,
        ]

        # Clip to the read extent exactly as the staged-files path does. Handing every tile the
        # whole region's graph would write a region-sized roads.json into each tile directory and
        # leave `bundle.roads` meaning something different depending on which provider built it.
        from .build import _clip_graph

        roads = _clip_graph(self._roads, tile.read) if self._roads is not None else None

        inputs = TileInputs(
            elevation=dem,
            buildings=_within(self._buildings, tile.read),
            roads=roads,
            records=records,
        )
        self._add_landuse(inputs, tile)
        return inputs

    def _add_landuse(self, inputs: TileInputs, tile: Tile) -> None:
        """Rasterise land use for this tile from whichever primary mesh actually contains it."""
        if not self._landuse:
            return

        from ..sources import NlniLanduseAdapter
        from ..sources.nlni import SHAPEFILE_CODE_FIELD, SHAPEFILE_ENCODING

        centre_lon, centre_lat = to_wgs84(*tile.core.centre, self.crs)
        for square, path in self._landuse:
            if not square.intersects(centre_lon, centre_lat, centre_lon, centre_lat):
                continue
            adapter = NlniLanduseAdapter(self.gate, target_crs=self.crs)
            result = adapter.read(
                path,
                bounds=tile.read,
                resolution=self.resolution,
                code_field=SHAPEFILE_CODE_FIELD,
                encoding=SHAPEFILE_ENCODING,
            )
            inputs.landuse.update({k: v[0] for k, v in result.layers.items()})
            inputs.add_record(result.record)
            return

        log.warning("%s: no land use primary mesh contains this tile", tile.id)


def _fetch_landuse_mesh(code: str, destination: Path) -> int | None:
    """Download one primary mesh's land use archive and keep only the shapefile parts.

    The archive also carries a ~9 MB GML of the same content and a metadata XML; neither is read,
    so neither is written. Returns bytes fetched, or ``None`` if the mesh is not published.
    """
    import urllib.error
    import urllib.request
    import zipfile
    from io import BytesIO

    from ..sources.fetch import USER_AGENT

    url = LANDUSE_URL.format(stem=LANDUSE_STEM, mesh=code)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=300) as response:
            blob = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # Primary meshes are large; one covering a site can be mostly sea and unpublished.
            log.info("landuse: no L03-b file for primary mesh %s", code)
            return None
        raise

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(blob)) as archive:
        for info in archive.infolist():
            if info.filename.lower().endswith((".shp", ".shx", ".dbf", ".prj")):
                (destination / Path(info.filename).name).write_bytes(archive.read(info))

    log.info("landuse: primary mesh %s, %.1f MB fetched", code, len(blob) / 1e6)
    return len(blob)


def _escape(name: str) -> str:
    import re

    return re.escape(name)


def _union(boxes) -> Bounds:
    boxes = list(boxes)
    return Bounds(
        min(b.minx for b in boxes),
        min(b.miny for b in boxes),
        max(b.maxx for b in boxes),
        max(b.maxy for b in boxes),
    )


def _wgs84_box(bounds: Bounds, crs) -> tuple[float, float, float, float]:
    west, south = to_wgs84(bounds.minx, bounds.miny, crs)
    east, north = to_wgs84(bounds.maxx, bounds.maxy, crs)
    return west, south, east, north


def _within(buildings: list, bounds: Bounds) -> list:
    """Buildings whose first footprint vertex falls in the tile's read extent.

    Coarse on purpose: the rasteriser clips properly, and this only exists to keep a municipality's
    worth of footprints from being handed to every tile.
    """
    return [
        b
        for b in buildings
        if b.footprint and bounds.contains_point(b.footprint[0][0], b.footprint[0][1])
    ]
