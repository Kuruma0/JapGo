"""PLATEAU CityGML reader.

PLATEAU is the primary source of shipped reconstruction geometry (research doc §6.1c) and the
reason the MVP is legally clean: it carries buildings, roads and land use under attribution-only
terms, plus the semantic attributes that would otherwise need a classifier to invent —
**usage**, **construction year** and **height**.

Structure notes, verified against the PLATEAU data specification:

* Buildings live at ``core:cityObjectMember/bldg:Building``.
* Footprints: ``bldg:lod0RoofEdge`` is the true from-above perimeter and is preferred.
  ``bldg:lod0FootPrint`` and the base ring of ``bldg:lod1Solid`` are fallbacks.
* ``bldg:measuredHeight``, ``bldg:storeysAboveGround``, ``bldg:yearOfConstruction`` and
  ``bldg:usage`` sit directly on the building.
* ``bldg:usage`` carries a ``codeSpace`` attribute pointing at one of the ~55 codelist XML files
  shipped inside the package. The codelist is authoritative; the taxonomy config only seeds a
  mapping. Codes absent from the config are reported rather than silently bucketed.

Coordinates in PLATEAU are geographic (JGD2011 lat/lon with ellipsoidal height) in ``gml:posList``
as ``lat lon height`` triples. They are projected to the working metric CRS on read, because every
morphological measure downstream is in metres.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

from pyproj import CRS

from ..core.buildings import Building, LabelSource, Taxonomy, load_taxonomy
from ..geo.crs import assert_metric, from_wgs84
from .base import ReadResult, SourceAdapter

log = logging.getLogger(__name__)

NS = {
    "core": "http://www.opengis.net/citygml/2.0",
    "bldg": "http://www.opengis.net/citygml/building/2.0",
    "gml": "http://www.opengis.net/gml",
    "uro": "https://www.geospatial.jp/iur/uro/3.0",
    "tran": "http://www.opengis.net/citygml/transportation/2.0",
    "luse": "http://www.opengis.net/citygml/landuse/2.0",
    "xlink": "http://www.w3.org/1999/xlink",
}

# PLATEAU has shipped several uro namespace versions. Match on local name instead of pinning one.
_URO_PATTERN = re.compile(r"\{https://www\.geospatial\.jp/iur/uro/[\d.]+\}(.+)")


class CodelistResolver:
    """Resolves ``codeSpace`` references against the codelist XML shipped in the package.

    The codelist is the authority. Reading it means an ingest survives PLATEAU revising or
    extending its own code definitions without a code change here.
    """

    def __init__(self, gml_path: Path) -> None:
        self.base = gml_path.parent
        self._cache: dict[str, dict[str, str]] = {}

    def label(self, code_space: str | None, code: str) -> str | None:
        """Human-readable label for a code, or ``None`` if the codelist is unavailable."""
        if not code_space:
            return None
        table = self._load(code_space)
        return table.get(code)

    def _load(self, code_space: str) -> dict[str, str]:
        if code_space in self._cache:
            return self._cache[code_space]

        path = (self.base / code_space).resolve()
        table: dict[str, str] = {}
        if path.is_file():
            try:
                root = ET.parse(path).getroot()
                for entry in root.iter():
                    if not entry.tag.endswith("Definition"):
                        continue
                    name = _text(entry, "gml:name")
                    value = _text(entry, "gml:description")
                    if name is not None and value is not None:
                        table[name] = value
            except ET.ParseError as exc:
                log.warning("could not parse codelist %s: %s", path, exc)
        else:
            log.debug("codelist not found: %s", path)

        self._cache[code_space] = table
        return table


def _text(element: ET.Element, path: str) -> str | None:
    found = element.find(path, NS)
    if found is None or found.text is None:
        return None
    text = found.text.strip()
    return text or None


def _uro_find(element: ET.Element, local_name: str) -> ET.Element | None:
    """Find a ``uro:`` descendant by local name, tolerating namespace version drift."""
    for node in element.iter():
        match = _URO_PATTERN.match(node.tag)
        if match and match.group(1) == local_name:
            return node
    return None


def _parse_pos_list(text: str) -> list[tuple[float, float, float]]:
    """Parse a ``gml:posList`` of ``lat lon height`` triples."""
    values = [float(v) for v in text.split()]
    if len(values) % 3 != 0:
        raise ValueError(f"posList length {len(values)} is not a multiple of 3")
    return [(values[i], values[i + 1], values[i + 2]) for i in range(0, len(values), 3)]


class PlateauAdapter(SourceAdapter):
    """Reads PLATEAU CityGML building files into :class:`~japgo.core.buildings.Building`."""

    source_id = "plateau"
    provides = ("buildings",)

    #: Footprint sources in preference order. lod0RoofEdge is the true from-above perimeter.
    FOOTPRINT_PATHS = (
        "bldg:lod0RoofEdge",
        "bldg:lod0FootPrint",
    )

    def __init__(self, gate, *, target_crs: CRS | int, taxonomy: Taxonomy | None = None) -> None:
        super().__init__(gate)
        self.target_crs = assert_metric(target_crs)
        self.taxonomy = taxonomy or load_taxonomy()

    # -----------------------------------------------------------------------------------------

    def read(self, path: Path, **kwargs) -> ReadResult:
        """Read one CityGML file."""
        self.open()  # provenance gate

        path = Path(path)
        codelists = CodelistResolver(path)
        root = ET.parse(path).getroot()

        buildings: list[Building] = []
        unmapped: Counter[str] = Counter()
        skipped = 0

        for node in root.iter(f"{{{NS['bldg']}}}Building"):
            building = self._parse_building(node, codelists, unmapped)
            if building is None:
                skipped += 1
                continue
            buildings.append(building)

        warnings = []
        if skipped:
            warnings.append(f"{skipped} building(s) skipped: no usable footprint geometry")
        for code, count in unmapped.most_common():
            warnings.append(
                f"PLATEAU usage code {code!r} ({count} building(s)) is not in the taxonomy config; "
                "add it to config/building_taxonomy.yaml"
            )
        for w in warnings:
            log.warning("%s: %s", path.name, w)

        return ReadResult(
            layers={"buildings": buildings},
            record=self.make_record(layers=["buildings"], note=path.name),
            warnings=warnings,
        )

    # -----------------------------------------------------------------------------------------

    def _parse_building(
        self,
        node: ET.Element,
        codelists: CodelistResolver,
        unmapped: Counter[str],
    ) -> Building | None:
        footprint, lod = self._extract_footprint(node)
        if footprint is None:
            return None

        usage_node = node.find("bldg:usage", NS)
        usage_code = (usage_node.text or "").strip() if usage_node is not None else None

        fine, mapped = ("unknown", False)
        type_source = LabelSource.UNKNOWN
        if usage_code:
            fine, mapped = self.taxonomy.from_plateau_usage(usage_code)
            type_source = LabelSource.PLATEAU
            if not mapped:
                unmapped[usage_code] += 1

        attributes: dict[str, str] = {}
        if usage_code:
            label = codelists.label(
                usage_node.get("codeSpace") if usage_node is not None else None, usage_code
            )
            if label:
                attributes["usage_label"] = label

        return Building(
            id=node.get(f"{{{NS['gml']}}}id") or f"anon-{id(node)}",
            source_id=self.source_id,
            footprint=footprint,
            height_m=_float(_text(node, "bldg:measuredHeight")),
            storeys_above_ground=_int(_text(node, "bldg:storeysAboveGround")),
            storeys_below_ground=_int(_text(node, "bldg:storeysBelowGround")),
            fine_type=fine,
            coarse_type=self.taxonomy.coarse_for(fine),
            type_source=type_source,
            year_of_construction=self._year_of_construction(node),
            raw_usage_code=usage_code,
            lod=lod,
            attributes=attributes,
        )

    def _extract_footprint(self, node: ET.Element) -> tuple[list[tuple[float, float]] | None, int | None]:
        """Return the footprint ring in the target CRS, and the LOD it came from."""
        for lod, path in enumerate(self.FOOTPRINT_PATHS):
            geom = node.find(path, NS)
            if geom is None:
                continue
            ring = self._first_ring(geom)
            if ring:
                return ring, 0

        # Fallback: the base ring of the lod1Solid extrusion.
        solid = node.find("bldg:lod1Solid", NS)
        if solid is not None:
            ring = self._first_ring(solid)
            if ring:
                return ring, 1

        return None, None

    def _first_ring(self, geom: ET.Element) -> list[tuple[float, float]] | None:
        pos_list = geom.find(".//gml:posList", NS)
        if pos_list is None or not pos_list.text:
            return None
        try:
            coords = _parse_pos_list(pos_list.text)
        except ValueError as exc:
            log.warning("malformed posList: %s", exc)
            return None

        # PLATEAU posLists are lat lon height; from_wgs84 takes (lon, lat).
        ring = [from_wgs84(lon, lat, self.target_crs) for lat, lon, _ in coords]
        if len(ring) < 4:
            return None
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        return ring

    def _year_of_construction(self, node: ET.Element) -> int | None:
        """Construction year, from the core attribute or the uro detail extension.

        Implausible values are discarded rather than propagated — see :func:`_plausible_year`.
        """
        direct = _plausible_year(_int(_text(node, "bldg:yearOfConstruction")))
        if direct:
            return direct

        detail = _uro_find(node, "yearOfConstruction")
        if detail is not None and detail.text:
            return _plausible_year(_int(detail.text.strip()))
        return None


#: Real PLATEAU data uses ``0001`` as a sentinel for "construction year unknown". Taken at face
#: value it becomes the year 1, which is not merely wrong but actively corrupting: the §16
#: development-age thread reads this field, and a cohort of year-1 buildings would drag every
#: median and skew any age-vs-morphology relationship the project is trying to measure.
#: Observed in Atami 2023, where 0001 is the most common single value in a mesh tile.
EARLIEST_PLAUSIBLE_YEAR = 1500


def _plausible_year(year: int | None) -> int | None:
    """Reject sentinel and impossible construction years."""
    if year is None:
        return None
    if year < EARLIEST_PLAUSIBLE_YEAR:
        return None
    if year > date.today().year + 1:  # a building completing next year is plausible; 2999 is not
        return None
    return year


def _float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        # PLATEAU sometimes carries a full date where a year is expected.
        match = re.match(r"(\d{4})", value)
        return int(match.group(1)) if match else None
