"""Shared fixtures.

The CityGML fixture is written by hand rather than downloaded. A real PLATEAU package is hundreds
of megabytes, and a synthetic file lets a test assert on a *specific* known input — including the
malformed cases that real data will eventually contain.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from japgo.provenance import SourceGate, find_registry, load_registry

# Shizuoka, roughly. Zone 8.
LAT, LON = 34.976, 138.383


@pytest.fixture(scope="session")
def registry():
    return load_registry(find_registry())


@pytest.fixture(scope="session")
def gate(registry):
    return SourceGate(registry)


def _ring(lat: float, lon: float, size_deg: float = 0.0001) -> str:
    """A closed square ring as a PLATEAU-style `lat lon height` posList."""
    pts = [
        (lat, lon),
        (lat, lon + size_deg),
        (lat + size_deg, lon + size_deg),
        (lat + size_deg, lon),
        (lat, lon),
    ]
    return " ".join(f"{a} {b} 10.0" for a, b in pts)


@pytest.fixture
def citygml_path(tmp_path: Path) -> Path:
    """A small but structurally faithful PLATEAU CityGML file.

    Deliberately includes: a detached house with full attributes, an apartment whose footprint only
    exists as an lod1Solid, a building with an unmapped usage code, and a building with no usable
    geometry at all.
    """
    content = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <core:CityModel
            xmlns:core="http://www.opengis.net/citygml/2.0"
            xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
            xmlns:gml="http://www.opengis.net/gml"
            xmlns:uro="https://www.geospatial.jp/iur/uro/3.0">

          <core:cityObjectMember>
            <bldg:Building gml:id="bldg-house-1">
              <bldg:usage codeSpace="../../codelists/Building_usage.xml">411</bldg:usage>
              <bldg:measuredHeight uom="m">6.5</bldg:measuredHeight>
              <bldg:storeysAboveGround>2</bldg:storeysAboveGround>
              <bldg:yearOfConstruction>1998</bldg:yearOfConstruction>
              <bldg:lod0RoofEdge>
                <gml:MultiSurface><gml:surfaceMember><gml:Polygon>
                  <gml:exterior><gml:LinearRing>
                    <gml:posList>{_ring(LAT, LON)}</gml:posList>
                  </gml:LinearRing></gml:exterior>
                </gml:Polygon></gml:surfaceMember></gml:MultiSurface>
              </bldg:lod0RoofEdge>
            </bldg:Building>
          </core:cityObjectMember>

          <core:cityObjectMember>
            <bldg:Building gml:id="bldg-apartment-1">
              <bldg:usage codeSpace="../../codelists/Building_usage.xml">412</bldg:usage>
              <bldg:measuredHeight uom="m">24.0</bldg:measuredHeight>
              <bldg:storeysAboveGround>8</bldg:storeysAboveGround>
              <bldg:lod1Solid>
                <gml:Solid><gml:exterior><gml:CompositeSurface>
                  <gml:surfaceMember><gml:Polygon>
                    <gml:exterior><gml:LinearRing>
                      <gml:posList>{_ring(LAT + 0.001, LON, 0.0002)}</gml:posList>
                    </gml:LinearRing></gml:exterior>
                  </gml:Polygon></gml:surfaceMember>
                </gml:CompositeSurface></gml:exterior></gml:Solid>
              </bldg:lod1Solid>
            </bldg:Building>
          </core:cityObjectMember>

          <core:cityObjectMember>
            <bldg:Building gml:id="bldg-unmapped-1">
              <bldg:usage codeSpace="../../codelists/Building_usage.xml">999</bldg:usage>
              <bldg:lod0RoofEdge>
                <gml:MultiSurface><gml:surfaceMember><gml:Polygon>
                  <gml:exterior><gml:LinearRing>
                    <gml:posList>{_ring(LAT + 0.002, LON)}</gml:posList>
                  </gml:LinearRing></gml:exterior>
                </gml:Polygon></gml:surfaceMember></gml:MultiSurface>
              </bldg:lod0RoofEdge>
            </bldg:Building>
          </core:cityObjectMember>

          <core:cityObjectMember>
            <bldg:Building gml:id="bldg-nogeom-1">
              <bldg:usage codeSpace="../../codelists/Building_usage.xml">441</bldg:usage>
            </bldg:Building>
          </core:cityObjectMember>

          <core:cityObjectMember>
            <bldg:Building gml:id="bldg-uro-year-1">
              <bldg:usage codeSpace="../../codelists/Building_usage.xml">413</bldg:usage>
              <bldg:lod0RoofEdge>
                <gml:MultiSurface><gml:surfaceMember><gml:Polygon>
                  <gml:exterior><gml:LinearRing>
                    <gml:posList>{_ring(LAT + 0.003, LON)}</gml:posList>
                  </gml:LinearRing></gml:exterior>
                </gml:Polygon></gml:surfaceMember></gml:MultiSurface>
              </bldg:lod0RoofEdge>
              <uro:buildingDetailAttribute>
                <uro:BuildingDetailAttribute>
                  <uro:yearOfConstruction>1975</uro:yearOfConstruction>
                </uro:BuildingDetailAttribute>
              </uro:buildingDetailAttribute>
            </bldg:Building>
          </core:cityObjectMember>

        </core:CityModel>
        """)
    path = tmp_path / "udx" / "bldg" / "53392633_bldg_6697_op.gml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def codelist_path(citygml_path: Path) -> Path:
    """The Building_usage codelist the CityGML `codeSpace` points at."""
    content = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <gml:Dictionary xmlns:gml="http://www.opengis.net/gml">
          <gml:dictionaryEntry>
            <gml:Definition>
              <gml:description>住宅</gml:description>
              <gml:name>411</gml:name>
            </gml:Definition>
          </gml:dictionaryEntry>
          <gml:dictionaryEntry>
            <gml:Definition>
              <gml:description>共同住宅</gml:description>
              <gml:name>412</gml:name>
            </gml:Definition>
          </gml:dictionaryEntry>
        </gml:Dictionary>
        """)
    path = citygml_path.parents[2] / "codelists" / "Building_usage.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
