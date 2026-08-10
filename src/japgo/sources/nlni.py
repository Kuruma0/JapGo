"""国土数値情報 (National Land Numerical Information) adapter — land use.

MLIT publishes land use as a mesh grid, ``L03-b 土地利用細分メッシュ``, with the land use type in
attribute ``L03b_002`` as a four-digit code. Part of the redistributable core: CC BY 4.0-compatible
government terms, so its geometry may reach shipped output.

One caution carried from research doc §6.5 and enforced by the registry: **licence terms differ
between vintages**. The registry records the terms per product per vintage, and the download site
warns about this explicitly. An unconfirmed vintage is quarantined rather than assumed current.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..geo.crs import assert_metric
from ..geo.raster import Raster
from ..geo.tiling import Bounds
from .base import ReadResult, SourceAdapter

log = logging.getLogger(__name__)

DEFAULT_LANDUSE_PATH = Path("config/landuse.yaml")
LANDUSE_CODE_FIELD = "L03b_002"


class LanduseSpec(BaseModel):
    """The land use vocabulary and its grouping into raster channels."""

    model_config = ConfigDict(frozen=True)

    landuse_version: int
    nlni_codes: dict[str, dict]
    classes: list[str]
    channel_groups: dict[str, list[str]] = Field(default_factory=dict)

    def class_for(self, code: str) -> tuple[str, bool]:
        """Map an NLNI code to a semantic class. Returns ``(class, was_mapped)``."""
        entry = self.nlni_codes.get(str(code).strip())
        if entry is None:
            return "unknown", False
        return str(entry["class"]), True

    def group_for(self, semantic_class: str) -> str | None:
        for group, members in self.channel_groups.items():
            if semantic_class in members:
                return group
        return None

    @property
    def channel_names(self) -> list[str]:
        return list(self.channel_groups)


def _find_landuse(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        path = candidate / DEFAULT_LANDUSE_PATH
        if path.is_file():
            return path
    raise FileNotFoundError(f"could not find {DEFAULT_LANDUSE_PATH} walking up from {here}")


@lru_cache(maxsize=4)
def load_landuse_spec(path: Path | None = None) -> LanduseSpec:
    path = path or _find_landuse()
    return LanduseSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class NlniLanduseAdapter(SourceAdapter):
    """Reads NLNI land use mesh into grouped coverage rasters."""

    source_id = "nlni_landuse"
    provides = ("landuse",)

    def __init__(self, gate, *, target_crs, spec: LanduseSpec | None = None) -> None:
        super().__init__(gate)
        self.target_crs = assert_metric(target_crs)
        self.spec = spec or load_landuse_spec()

    def read(
        self,
        path: Path,
        *,
        bounds: Bounds,
        resolution: float = 1.0,
        code_field: str = LANDUSE_CODE_FIELD,
        **kwargs,
    ) -> ReadResult:
        """Rasterise land use mesh polygons into one binary coverage raster per channel group."""
        self.open()  # provenance gate

        import geopandas as gpd
        from rasterio import features

        from ..pipeline.rasterize import transform_for

        path = Path(path)
        frame = gpd.read_file(path)

        if code_field not in frame.columns:
            raise ValueError(
                f"{path.name}: no {code_field!r} column. Columns present: {list(frame.columns)}. "
                "NLNI land use mesh carries the 土地利用種別 code in L03b_002; pass code_field= "
                "if this product uses a different attribute."
            )

        if frame.crs is None:
            raise ValueError(
                f"{path.name}: no CRS on the source. NLNI ships JGD2011 geographic; refusing to "
                "guess, because a silent CRS assumption misaligns every downstream layer."
            )
        frame = frame.to_crs(self.target_crs)

        shape = (
            int(round(bounds.height / resolution)),
            int(round(bounds.width / resolution)),
        )
        transform = transform_for(bounds, resolution)

        by_group: dict[str, list] = {name: [] for name in self.spec.channel_names}
        unmapped: dict[str, int] = {}

        for code, geometry in zip(frame[code_field], frame.geometry, strict=False):
            if geometry is None or geometry.is_empty:
                continue
            semantic, mapped = self.spec.class_for(str(code))
            if not mapped:
                unmapped[str(code)] = unmapped.get(str(code), 0) + 1
                continue
            group = self.spec.group_for(semantic)
            if group is not None:
                by_group[group].append((geometry.__geo_interface__, 1))

        layers: dict[str, list] = {}
        for group, shapes in by_group.items():
            if shapes:
                burned = features.rasterize(
                    shapes,
                    out_shape=shape,
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=False,
                ).astype(np.float32)
            else:
                burned = np.zeros(shape, np.float32)
            layers[group] = [Raster(burned, bounds, self.target_crs)]

        warnings = [
            f"NLNI land use code {code!r} ({count} feature(s)) is not in config/landuse.yaml"
            for code, count in sorted(unmapped.items(), key=lambda kv: -kv[1])
        ]
        for w in warnings:
            log.warning("%s: %s", path.name, w)

        return ReadResult(
            layers=layers,
            record=self.make_record(
                layers=sorted(layers),
                note=f"{path.name}; {len(frame)} mesh cells; field={code_field}",
            ),
            warnings=warnings,
        )
