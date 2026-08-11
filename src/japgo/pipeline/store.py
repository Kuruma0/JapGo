"""Tile persistence.

Zarr for the raster stack (chunked, compressed, one chunk per tile), JSON for the manifest,
GeoParquet for vectors. Every tile written is accompanied by its manifest — a stack on disk
without provenance is not a tile, it is an orphan.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..core.buildings import Building
from ..core.manifest import TileManifest
from ..geo.tiling import Tile, parse_tile_id
from .assemble import TileBundle
from .channels import StackSpec, load_stack_spec

STACK_FILE = "stack.zarr"
MANIFEST_FILE = "manifest.json"
BUILDINGS_FILE = "buildings.parquet"
TARGETS_FILE = "targets.zarr"
ROADS_FILE = "roads.json"


def tile_dir(root: Path, tile: Tile | str) -> Path:
    tile_id = tile if isinstance(tile, str) else tile.id
    return Path(root) / tile_id


def write_tile(root: Path, bundle: TileBundle) -> Path:
    """Write a bundle to ``root/<tile_id>/``. Returns the directory."""
    import zarr

    out = tile_dir(root, bundle.tile)
    out.mkdir(parents=True, exist_ok=True)

    store_path = out / STACK_FILE
    if store_path.exists():
        _remove_tree(store_path)

    array = zarr.open_array(
        store=str(store_path),
        mode="w",
        shape=bundle.stack.shape,
        chunks=bundle.stack.shape,  # one chunk per tile: the tile is the unit of everything
        dtype="float32",
    )
    array[:] = bundle.stack
    array.attrs["channels"] = bundle.spec.names
    array.attrs["stack_version"] = bundle.spec.stack_version
    array.attrs["tile_id"] = bundle.tile.id

    if bundle.targets is not None:
        target_path = out / TARGETS_FILE
        if target_path.exists():
            _remove_tree(target_path)
        target_array = zarr.open_array(
            store=str(target_path),
            mode="w",
            shape=bundle.targets.shape,
            chunks=bundle.targets.shape,
            dtype="float32",
        )
        target_array[:] = bundle.targets
        target_array.attrs["channels"] = bundle.spec.target_names
        target_array.attrs["tile_id"] = bundle.tile.id

    bundle.manifest.write(out / MANIFEST_FILE)

    if bundle.buildings:
        _write_buildings(out / BUILDINGS_FILE, bundle.buildings)

    if bundle.roads is not None:
        # The road graph is a first-class output, not a rasterisation intermediate: the Phase 2
        # alignment check needs the vectors, and Phase 3's metrics are computed on the graph.
        (out / ROADS_FILE).write_text(
            bundle.roads.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    return out


def read_tile(root: Path, tile_id: str, *, spec: StackSpec | None = None) -> TileBundle:
    """Read a bundle back. Raises if the manifest is missing."""
    import zarr

    out = tile_dir(root, tile_id)
    manifest_path = out / MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{out} has no {MANIFEST_FILE}. A stack without provenance is not a tile — refusing "
            "to load it."
        )

    manifest = TileManifest.read(manifest_path)
    array = zarr.open_array(store=str(out / STACK_FILE), mode="r")
    stack = np.asarray(array[:], dtype=np.float32)

    stored = list(array.attrs.get("channels", []))
    spec = spec or load_stack_spec()
    if stored and stored != spec.names:
        raise ValueError(
            f"{tile_id}: stored channels {stored} do not match the current stack spec "
            f"{spec.names}. Rebuild the tile rather than reinterpreting it."
        )

    buildings: list[Building] = []
    parquet = out / BUILDINGS_FILE
    if parquet.is_file():
        buildings = _read_buildings(parquet)

    targets = None
    target_path = out / TARGETS_FILE
    if target_path.exists():
        target_array = zarr.open_array(store=str(target_path), mode="r")
        targets = np.asarray(target_array[:], dtype=np.float32)
        stored_targets = list(target_array.attrs.get("channels", []))
        if stored_targets and stored_targets != spec.target_names:
            raise ValueError(
                f"{tile_id}: stored targets {stored_targets} do not match the current spec "
                f"{spec.target_names}. Rebuild the tile rather than reinterpreting it."
            )

    roads = None
    roads_path = out / ROADS_FILE
    if roads_path.is_file():
        from ..core.roads import RoadGraph

        roads = RoadGraph.model_validate_json(roads_path.read_text(encoding="utf-8"))

    return TileBundle(
        tile=parse_tile_id(tile_id, core_size_m=manifest.core_size_m, halo_m=manifest.halo_m),
        stack=stack,
        spec=spec,
        manifest=manifest,
        buildings=buildings,
        roads=roads,
        targets=targets,
    )


def list_tiles(root: Path) -> list[str]:
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / MANIFEST_FILE).is_file())


def write_attribution(root: Path, lines: list[str]) -> Path:
    """Write the assembled attribution block beside a tile set.

    Attribution assembled by hand is attribution that will eventually be wrong, so it is written
    from the manifests rather than typed.
    """
    path = Path(root) / "ATTRIBUTION.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------------


def _write_buildings(path: Path, buildings: list[Building]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "id": [b.id for b in buildings],
            "source_id": [b.source_id for b in buildings],
            "fine_type": [b.fine_type for b in buildings],
            "coarse_type": [b.coarse_type for b in buildings],
            "type_source": [str(b.type_source) for b in buildings],
            "height_m": [b.height_m for b in buildings],
            "storeys_above_ground": [b.storeys_above_ground for b in buildings],
            "year_of_construction": [b.year_of_construction for b in buildings],
            "raw_usage_code": [b.raw_usage_code for b in buildings],
            "lod": [b.lod for b in buildings],
            # WKT rather than WKB: readable in a text diff, and the volumes here do not justify
            # the compactness. Revisit if a tile ever holds enough buildings for it to matter.
            "footprint_wkt": [_to_wkt(b.footprint) for b in buildings],
        }
    )
    pq.write_table(table, path)


def _read_buildings(path: Path) -> list[Building]:
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    out = []
    for i in range(len(table["id"])):
        out.append(
            Building(
                id=table["id"][i],
                source_id=table["source_id"][i],
                footprint=_from_wkt(table["footprint_wkt"][i]),
                height_m=table["height_m"][i],
                storeys_above_ground=table["storeys_above_ground"][i],
                year_of_construction=table["year_of_construction"][i],
                fine_type=table["fine_type"][i],
                coarse_type=table["coarse_type"][i],
                type_source=table["type_source"][i],
                raw_usage_code=table["raw_usage_code"][i],
                lod=table["lod"][i],
            )
        )
    return out


def _to_wkt(ring: list[tuple[float, float]]) -> str:
    coords = ", ".join(f"{x:.4f} {y:.4f}" for x, y in ring)
    return f"POLYGON (({coords}))"


def _from_wkt(wkt: str) -> list[tuple[float, float]]:
    inner = wkt[wkt.index("((") + 2 : wkt.rindex("))")]
    return [tuple(float(v) for v in pair.split()) for pair in inner.split(",")]  # type: ignore[misc]


def _remove_tree(path: Path) -> None:
    import shutil

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def write_index(root: Path, tile_ids: list[str]) -> Path:
    """A flat index of the tile set, for dataset loaders."""
    path = Path(root) / "tiles.json"
    path.write_text(json.dumps({"tiles": sorted(tile_ids)}, indent=2) + "\n", encoding="utf-8")
    return path
