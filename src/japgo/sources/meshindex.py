"""VIRTUAL SHIZUOKA mesh discovery.

The prefecture does not publish one file per region. It publishes a **vector-tile index**: an XYZ
endpoint whose features are mesh polygons carrying a ``MESH_NO`` and the ``URL`` of that mesh's
data. Terrain for a specific place is therefore a two-step fetch — read the index tile covering
the area, then download only the meshes that intersect it.

That is why terrain here costs megabytes rather than gigabytes. A Grid mesh is roughly 400 x 300 m
and about 2 MB zipped, against the multi-gigabyte per-region archives the Phase 0 research
expected.

The index is Mapbox Vector Tile format. Rather than take a protobuf dependency for two field
types, the wire format is read directly — MVT is a small, stable spec and only the parts used here
are implemented, with anything unrecognised skipped rather than guessed at.
"""

from __future__ import annotations

import logging
import math
import urllib.request
import zipfile

import numpy as np
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from ..geo.crs import to_wgs84
from ..geo.tiling import Bounds
from ..provenance import SourceGate
from .fetch import USER_AGENT

log = logging.getLogger(__name__)

#: Published index endpoints for the 2019 LP (aerial LiDAR) survey.
GRID_INDEX = "https://gic-shizuoka.s3.ap-northeast-1.amazonaws.com/2020/Vectortile2025/LPGRD/{z}/{x}/{y}.pbf"
ORTHO_INDEX = "https://gic-shizuoka.s3.ap-northeast-1.amazonaws.com/2020/Vectortile2025/LPORTHO/{z}/{x}/{y}.pbf"
CONTOUR_INDEX = "https://gic-shizuoka.s3.ap-northeast-1.amazonaws.com/2020/Vectortile2025/LPCONT/{z}/{x}/{y}.pbf"

INDEX_ZOOM = 14
"""Zoom at which one index tile covers a useful area without returning thousands of features."""


# ------------------------------------------------------------------------------------------------
# Minimal MVT reading
# ------------------------------------------------------------------------------------------------


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _fields(buf: bytes):
    """Yield ``(field_number, payload)`` for length-delimited and varint fields."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 2:
            length, i = _varint(buf, i)
            yield number, buf[i : i + length]
            i += length
        elif wire == 0:
            value, i = _varint(buf, i)
            yield number, value
        elif wire == 5:
            yield number, buf[i : i + 4]
            i += 4
        elif wire == 1:
            yield number, buf[i : i + 8]
            i += 8
        else:  # group types, removed from proto3
            return


def _packed(buf: bytes) -> list[int]:
    out, i = [], 0
    while i < len(buf):
        value, i = _varint(buf, i)
        out.append(value)
    return out


def _rings(commands: list[int]) -> list[list[tuple[int, int]]]:
    """Decode MVT geometry commands into rings of tile-space coordinates."""
    rings: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    x = y = i = 0

    while i < len(commands):
        header = commands[i]
        i += 1
        command, count = header & 0x7, header >> 3

        if command in (1, 2):  # MoveTo, LineTo
            for _ in range(count):
                dx, dy = commands[i], commands[i + 1]
                i += 2
                x += (dx >> 1) ^ -(dx & 1)  # zigzag
                y += (dy >> 1) ^ -(dy & 1)
                if command == 1:
                    if current:
                        rings.append(current)
                    current = [(x, y)]
                else:
                    current.append((x, y))
        elif command == 7:  # ClosePath
            if current:
                rings.append(current)
                current = []
        else:
            break

    if current:
        rings.append(current)
    return rings


@dataclass(frozen=True)
class Mesh:
    """One indexed mesh: its identifier, download URL and geographic extent."""

    mesh_no: str
    url: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def intersects_lonlat(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> bool:
        return not (
            self.max_lon < min_lon
            or self.min_lon > max_lon
            or self.max_lat < min_lat
            or self.min_lat > max_lat
        )


def _tile_of(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def _to_lonlat(px: float, py: float, extent: int, zoom: int, tx: int, ty: int) -> tuple[float, float]:
    n = 2**zoom
    lon = (tx + px / extent) / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + py / extent) / n))))
    return lon, lat


class MeshIndex:
    """Finds VIRTUAL SHIZUOKA mesh downloads for an area, under the provenance gate."""

    source_id = "virtual_shizuoka"

    def __init__(self, gate: SourceGate, *, template: str = GRID_INDEX, zoom: int = INDEX_ZOOM) -> None:
        self.gate = gate
        self.template = template
        self.zoom = zoom

    def meshes_for(self, bounds: Bounds, crs) -> list[Mesh]:
        """Every indexed mesh intersecting ``bounds`` (given in ``crs``)."""
        self.gate.assert_ingestible(self.source_id)

        corners = [
            to_wgs84(bounds.minx, bounds.miny, crs),
            to_wgs84(bounds.minx, bounds.maxy, crs),
            to_wgs84(bounds.maxx, bounds.miny, crs),
            to_wgs84(bounds.maxx, bounds.maxy, crs),
        ]
        lons = [c[0] for c in corners]
        lats = [c[1] for c in corners]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        x0, y1 = _tile_of(min_lon, min_lat, self.zoom)
        x1, y0 = _tile_of(max_lon, max_lat, self.zoom)

        found: dict[str, Mesh] = {}
        for tx in range(min(x0, x1), max(x0, x1) + 1):
            for ty in range(min(y0, y1), max(y0, y1) + 1):
                for mesh in self._read_index_tile(tx, ty):
                    if mesh.intersects_lonlat(min_lon, min_lat, max_lon, max_lat):
                        found[mesh.mesh_no] = mesh
        return sorted(found.values(), key=lambda m: m.mesh_no)

    def _read_index_tile(self, tx: int, ty: int) -> list[Mesh]:
        url = self.template.format(z=self.zoom, x=tx, y=ty)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
        except Exception as exc:  # noqa: BLE001 - a missing index tile is normal off-coverage
            log.debug("index tile %s unavailable: %s", url, exc)
            return []

        out: list[Mesh] = []
        for number, layer in _fields(data):
            if number != 3 or not isinstance(layer, bytes):
                continue
            keys: list[str] = []
            values: list[str] = []
            features: list[bytes] = []
            extent = 4096

            for lnum, payload in _fields(layer):
                if lnum == 3 and isinstance(payload, bytes):
                    keys.append(payload.decode("utf-8", "replace"))
                elif lnum == 4 and isinstance(payload, bytes):
                    for vnum, vpayload in _fields(payload):
                        if vnum == 1 and isinstance(vpayload, bytes):
                            values.append(vpayload.decode("utf-8", "replace"))
                elif lnum == 5 and isinstance(payload, int):
                    extent = payload
                elif lnum == 2 and isinstance(payload, bytes):
                    features.append(payload)

            for feature in features:
                tags: list[int] = []
                geometry: list[int] = []
                for fnum, payload in _fields(feature):
                    if fnum == 2 and isinstance(payload, bytes):
                        tags = _packed(payload)
                    elif fnum == 4 and isinstance(payload, bytes):
                        geometry = _packed(payload)

                attributes = {
                    keys[tags[i]]: values[tags[i + 1]]
                    for i in range(0, len(tags) - 1, 2)
                    if tags[i] < len(keys) and tags[i + 1] < len(values)
                }
                mesh_no, mesh_url = attributes.get("MESH_NO"), attributes.get("URL")
                if not mesh_no or not mesh_url:
                    continue

                points = [
                    _to_lonlat(px, py, extent, self.zoom, tx, ty)
                    for ring in _rings(geometry)
                    for px, py in ring
                ]
                if not points:
                    continue
                lons = [p[0] for p in points]
                lats = [p[1] for p in points]
                out.append(Mesh(mesh_no, mesh_url, min(lons), min(lats), max(lons), max(lats)))
        return out

    # -------------------------------------------------------------------------------------

    def read_mesh(self, mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Download one mesh and parse it to ``(x, y, z)`` **without touching disk**.

        The Grid product is plain text, and text is roughly seven times its zipped size: one tile's
        terrain is ~30 meshes and 411 MB extracted, against ~9 MB for the raster it becomes. At
        three sites that difference is tens of gigabytes of intermediate nobody reads twice.
        """
        self.gate.assert_ingestible(self.source_id)

        request = urllib.request.Request(mesh.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=180) as response:
            blob = response.read()

        columns: list[np.ndarray] = []
        with zipfile.ZipFile(BytesIO(blob)) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".txt"):
                    continue
                text = archive.read(info).decode("ascii", "replace")
                values = np.fromstring(text, sep=" ", dtype=np.float64)
                if values.size % 3:
                    log.warning("%s: %d numbers, not a multiple of 3", info.filename, values.size)
                    continue
                columns.append(values.reshape(-1, 3))

        if not columns:
            return (np.empty(0), np.empty(0), np.empty(0))
        points = np.vstack(columns)
        return points[:, 0], points[:, 1], points[:, 2]

    def download(self, mesh: Mesh, destination: Path) -> list[Path]:
        """Download and unpack one mesh. Returns the extracted files."""
        self.gate.assert_ingestible(self.source_id)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        request = urllib.request.Request(mesh.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=180) as response:
            blob = response.read()

        written: list[Path] = []
        with zipfile.ZipFile(BytesIO(blob)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                target = destination / Path(info.filename).name
                target.write_bytes(archive.read(info))
                written.append(target)
        return written


class TerrainFetcher:
    """Builds a tile's DEM directly from the published meshes, caching the raster.

    This is the shape terrain ingest should take: stream each mesh, grid it in memory, keep the
    raster, discard the text. Nothing intermediate reaches disk, and a rebuilt tile is a cache hit
    rather than a re-download.
    """

    def __init__(
        self,
        gate: SourceGate,
        *,
        index: MeshIndex | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.gate = gate
        self.index = index or MeshIndex(gate)
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_path(self, key: str, resolution: float) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{key}_{resolution:g}m.npz"

    def dem_for(
        self,
        bounds: Bounds,
        crs,
        *,
        resolution: float = 1.0,
        key: str | None = None,
        progress: bool = False,
    ) -> tuple["Raster", list[str]]:
        """Return the DEM over ``bounds`` and the mesh ids that contributed to it."""
        from ..geo.raster import Raster

        cache = self._cache_path(key, resolution) if key else None
        if cache is not None and cache.is_file():
            payload = np.load(cache, allow_pickle=False)
            raster = Raster(payload["data"].astype(np.float32), bounds, crs)
            return raster, [str(m) for m in payload["meshes"]]

        meshes = self.index.meshes_for(bounds, crs)
        accumulator = Raster.empty(bounds, resolution, crs)
        used: list[str] = []

        for position, mesh in enumerate(meshes, 1):
            x, y, z = self.read_mesh_points(mesh)
            if x.size == 0:
                continue
            _accumulate(accumulator, x, y, z, resolution)
            used.append(mesh.mesh_no)
            if progress:
                log.info("terrain %d/%d %s", position, len(meshes), mesh.mesh_no)

        if cache is not None and used:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, data=accumulator.data, meshes=np.array(used))

        return accumulator, used

    def read_mesh_points(self, mesh: Mesh):
        return self.index.read_mesh(mesh)


def _accumulate(target, x: np.ndarray, y: np.ndarray, z: np.ndarray, resolution: float) -> None:
    """Bin one mesh's posts into an existing raster, in place."""
    bounds = target.bounds
    rows, cols = target.data.shape

    inside = (
        (x >= bounds.minx) & (x < bounds.maxx) & (y >= bounds.miny) & (y < bounds.maxy)
    )
    if not inside.any():
        return
    x, y, z = x[inside], y[inside], z[inside]

    col = np.floor((x - bounds.minx) / resolution).astype(np.int64)
    row = np.floor((bounds.maxy - y) / resolution).astype(np.int64)
    np.clip(col, 0, cols - 1, out=col)
    np.clip(row, 0, rows - 1, out=row)

    flat = row * cols + col
    counts = np.bincount(flat, minlength=rows * cols)
    sums = np.bincount(flat, weights=z, minlength=rows * cols)

    with np.errstate(invalid="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan).reshape(rows, cols)

    fill = ~np.isnan(mean)
    target.data[fill] = mean[fill].astype(np.float32)
