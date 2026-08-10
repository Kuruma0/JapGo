"""Tests for the PLATEAU CityGML adapter."""

from __future__ import annotations

import pytest

from japgo.core import LabelSource, load_taxonomy
from japgo.geo import SHIZUOKA
from japgo.provenance import ProvenanceViolation, SourceGate, load_registry
from japgo.sources import PlateauAdapter


@pytest.fixture
def adapter(gate):
    return PlateauAdapter(gate, target_crs=SHIZUOKA.crs)


# ---------------------------------------------------------------------------------------------
# The gate is applied before anything is read
# ---------------------------------------------------------------------------------------------


def test_adapter_refuses_to_read_before_open(gate):
    a = PlateauAdapter(gate, target_crs=SHIZUOKA.crs)
    with pytest.raises(RuntimeError, match="provenance gate"):
        _ = a.source


def test_adapter_requires_a_metric_crs(gate):
    with pytest.raises(ValueError, match="geographic CRS"):
        PlateauAdapter(gate, target_crs=4326)


def test_quarantined_source_cannot_be_read_through_an_adapter(gate):
    """The gate is not bypassable by writing a new adapter."""

    class GsiAdapter(PlateauAdapter):
        source_id = "gsi_tiles"

    a = GsiAdapter(gate, target_crs=SHIZUOKA.crs)
    with pytest.raises(ProvenanceViolation, match="quarantined"):
        a.open()


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------


def test_reads_buildings(adapter, citygml_path):
    result = adapter.read(citygml_path)
    ids = {b.id for b in result.layers["buildings"]}
    assert ids == {"bldg-house-1", "bldg-apartment-1", "bldg-unmapped-1", "bldg-uro-year-1"}


def test_building_without_geometry_is_skipped_and_reported(adapter, citygml_path):
    result = adapter.read(citygml_path)
    assert "bldg-nogeom-1" not in {b.id for b in result.layers["buildings"]}
    assert any("no usable footprint" in w for w in result.warnings)


def test_attributes_are_read(adapter, citygml_path):
    house = _by_id(adapter.read(citygml_path), "bldg-house-1")
    assert house.height_m == pytest.approx(6.5)
    assert house.storeys_above_ground == 2
    assert house.year_of_construction == 1998
    assert house.raw_usage_code == "411"


def test_usage_code_maps_to_taxonomy(adapter, citygml_path):
    result = adapter.read(citygml_path)
    house = _by_id(result, "bldg-house-1")
    assert house.fine_type == "detached_house"
    assert house.coarse_type == "residential"
    assert house.type_source is LabelSource.PLATEAU

    apartment = _by_id(result, "bldg-apartment-1")
    assert apartment.fine_type == "apartment"
    assert apartment.coarse_type == "residential"


def test_unmapped_usage_code_is_surfaced_not_swallowed(adapter, citygml_path):
    """An unmapped code is a taxonomy gap to close, not a data error to hide."""
    result = adapter.read(citygml_path)
    assert _by_id(result, "bldg-unmapped-1").fine_type == "unknown"
    assert any("'999'" in w and "taxonomy config" in w for w in result.warnings)


def test_lod1_solid_is_used_when_lod0_is_absent(adapter, citygml_path):
    apartment = _by_id(adapter.read(citygml_path), "bldg-apartment-1")
    assert apartment.lod == 1
    assert apartment.is_closed


def test_year_falls_back_to_uro_extension(adapter, citygml_path):
    """PLATEAU carries construction year in the uro detail block as well as on the building."""
    assert _by_id(adapter.read(citygml_path), "bldg-uro-year-1").year_of_construction == 1975


# ---------------------------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------------------------


def test_footprints_are_projected_to_metres(adapter, citygml_path):
    """A ~0.0001 degree square is roughly 9-11 m per side at this latitude."""
    house = _by_id(adapter.read(citygml_path), "bldg-house-1")
    assert 50 < house.area_m2 < 200
    assert 30 < house.perimeter_m < 60


def test_footprint_rings_are_closed(adapter, citygml_path):
    for b in adapter.read(citygml_path).layers["buildings"]:
        assert b.is_closed, f"{b.id} ring is not closed"


def test_estimated_height_falls_back_to_storeys(adapter, citygml_path):
    house = _by_id(adapter.read(citygml_path), "bldg-house-1")
    assert house.estimated_height_m == pytest.approx(6.5)  # measured wins

    unmapped = _by_id(adapter.read(citygml_path), "bldg-unmapped-1")
    assert unmapped.estimated_height_m is None  # no measurement, no storeys, no guess


# ---------------------------------------------------------------------------------------------
# Codelist resolution
# ---------------------------------------------------------------------------------------------


def test_codelist_label_is_resolved_when_present(adapter, citygml_path, codelist_path):
    house = _by_id(adapter.read(citygml_path), "bldg-house-1")
    assert house.attributes.get("usage_label") == "住宅"


def test_missing_codelist_is_not_fatal(adapter, citygml_path):
    """Codelists are shipped per package; a missing one degrades the label, not the ingest."""
    house = _by_id(adapter.read(citygml_path), "bldg-house-1")
    assert "usage_label" not in house.attributes
    assert house.fine_type == "detached_house"  # taxonomy seed still works


# ---------------------------------------------------------------------------------------------
# Provenance record
# ---------------------------------------------------------------------------------------------


def test_read_produces_a_provenance_record(adapter, citygml_path):
    record = adapter.read(citygml_path).record
    assert record is not None
    assert record.source_id == "plateau"
    assert record.layers == ["buildings"]
    assert record.retrieved_at


def test_every_building_carries_its_source_id(adapter, citygml_path):
    """Invariant 3: source_id on every vector feature, to preserve Collective Database status."""
    for b in adapter.read(citygml_path).layers["buildings"]:
        assert b.source_id == "plateau"


# ---------------------------------------------------------------------------------------------
# Taxonomy config
# ---------------------------------------------------------------------------------------------


def test_every_fine_type_has_a_valid_coarse_type():
    tax = load_taxonomy()
    for fine, entry in tax.fine.items():
        assert entry["coarse"] in tax.coarse, f"{fine} maps to unknown coarse type"


def test_every_plateau_code_maps_to_a_declared_fine_type():
    tax = load_taxonomy()
    for code, entry in tax.plateau_usage.items():
        assert entry["fine"] in tax.fine, f"code {code} maps to undeclared fine type"


def test_osm_building_yes_maps_correctly():
    """Regression: YAML 1.1 coerces a bare `yes` key to the boolean true.

    `building=yes` is the most common building tag in OSM, so an unquoted key silently breaks the
    single most frequent case rather than an obscure one.
    """
    tax = load_taxonomy()
    assert "yes" in tax.osm_building
    assert tax.from_osm_building("yes") == ("unknown", True)


def test_osm_tags_map_to_taxonomy():
    tax = load_taxonomy()
    assert tax.from_osm_building("apartments") == ("apartment", True)
    assert tax.from_osm_building("HOUSE") == ("detached_house", True)  # case-insensitive
    assert tax.from_osm_building("nonsense") == ("unknown", False)


def test_low_confidence_codes_are_flagged_for_reconciliation():
    """These were inferred rather than verified; reconcile on first real ingest."""
    assert set(load_taxonomy().low_confidence_codes) == {
        "403", "404", "451", "452", "453", "454",
    }


def _by_id(result, building_id):
    return next(b for b in result.layers["buildings"] if b.id == building_id)
