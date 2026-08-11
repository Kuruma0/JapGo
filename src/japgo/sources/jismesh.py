"""JIS X 0410 standard grid square codes.

PLATEAU names its CityGML members by mesh code — ``53392546_bldg_6697_op.gml`` covers one 3rd-mesh
square. Decoding that from the filename is what lets a build fetch the three members covering its
tiles instead of all 69 covering the municipality, which is the difference between a few megabytes
and a few hundred.

The scheme is nested and fixed:

============  ==========================  ==================
Digits        Square                      Size
============  ==========================  ==================
4 (``5339``)  1st mesh, "primary"         40′ lat × 1° lon
6 (``533925``) 2nd mesh                   5′ lat × 7′30″ lon
8 (``53392546``) 3rd mesh, "standard"     30″ lat × 45″ lon
============  ==========================  ==================

Codes are in Tokyo-datum-era geographic coordinates that are treated as WGS84 by every modern
publisher, PLATEAU included. The residual shift is far below a mesh square, and this is used to
*select* files rather than to place geometry — the GML carries its own coordinates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MESH_CODE = re.compile(r"(?<!\d)(\d{4}|\d{6}|\d{8})(?!\d)")

_LAT_UNIT = 2.0 / 3.0
"""Degrees of latitude per 1st mesh: 40 minutes."""


class MeshCodeError(ValueError):
    """The code is not 4, 6 or 8 digits, or its components are out of range."""


@dataclass(frozen=True)
class MeshSquare:
    """A mesh square as a WGS84 bounding box."""

    code: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def intersects(self, west: float, south: float, east: float, north: float) -> bool:
        """Whether this square overlaps a WGS84 box. Touching edges do not count as overlap."""
        return not (
            self.max_lon <= west
            or self.min_lon >= east
            or self.max_lat <= south
            or self.min_lat >= north
        )


def decode(code: str) -> MeshSquare:
    """Decode a 4, 6 or 8 digit mesh code into its bounding box."""
    code = str(code).strip()
    if len(code) not in (4, 6, 8) or not code.isdigit():
        raise MeshCodeError(f"{code!r} is not a 4, 6 or 8 digit mesh code")

    p, q = int(code[0:2]), int(code[2:4])
    lat, lon = p * _LAT_UNIT, q + 100.0
    lat_size, lon_size = _LAT_UNIT, 1.0

    if len(code) >= 6:
        r, s = int(code[4]), int(code[5])
        if r > 7 or s > 7:
            raise MeshCodeError(f"{code!r}: 2nd mesh digits must be 0-7, got {r}{s}")
        lat_size, lon_size = lat_size / 8.0, lon_size / 8.0
        lat, lon = lat + r * lat_size, lon + s * lon_size

    if len(code) == 8:
        t, u = int(code[6]), int(code[7])
        lat_size, lon_size = lat_size / 10.0, lon_size / 10.0
        lat, lon = lat + t * lat_size, lon + u * lon_size

    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        # Reached by a four-digit number that is not a mesh code at all — PLATEAU filenames also
        # carry an EPSG code (``..._bldg_6697_op.gml``), and 6697 decodes to longitude 197. Raising
        # lets `code_in` fall back to "might be relevant" and keep the file, which is the safe
        # direction: a spurious download costs bandwidth, a spurious exclusion costs data.
        raise MeshCodeError(f"{code!r} decodes outside the globe (lon {lon}, lat {lat})")

    return MeshSquare(
        code=code,
        min_lon=lon,
        min_lat=lat,
        max_lon=lon + lon_size,
        max_lat=lat + lat_size,
    )


def code_in(name: str) -> str | None:
    """The mesh code in a filename, or ``None`` if it carries none.

    Returns ``None`` rather than raising: a member whose name cannot be decoded must still be
    selectable, because the alternative is silently dropping data on a naming-convention change.
    Callers treat an undecodable name as "might be relevant".
    """
    match = MESH_CODE.search(name)
    if not match:
        return None
    try:
        decode(match.group(1))
    except MeshCodeError:
        return None
    return match.group(1)


def primary_meshes_for(west: float, south: float, east: float, north: float) -> list[str]:
    """The 4-digit primary mesh codes covering a WGS84 box.

    NLNI land use is distributed one file per primary mesh, so this is how a site extent turns
    into a download list. A primary mesh is ~90 x 74 km, so a site normally needs one or two.
    """
    codes = []
    p0, p1 = int(south / _LAT_UNIT), int(north / _LAT_UNIT)
    q0, q1 = int(west) - 100, int(east) - 100
    for p in range(min(p0, p1), max(p0, p1) + 1):
        for q in range(min(q0, q1), max(q0, q1) + 1):
            code = f"{p:02d}{q:02d}"
            if decode(code).intersects(west, south, east, north):
                codes.append(code)
    return codes


def covering(names: list[str], west: float, south: float, east: float, north: float) -> list[str]:
    """Filter filenames to those whose mesh code overlaps the box, keeping undecodable names."""
    kept = []
    for name in names:
        code = code_in(name)
        if code is None or decode(code).intersects(west, south, east, north):
            kept.append(name)
    return kept
