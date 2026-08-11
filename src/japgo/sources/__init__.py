"""Per-dataset source adapters.

Every adapter passes through the provenance gate in :meth:`SourceAdapter.open` before reading a
byte. Adapters return core representation objects, never raw source structures — the point of the
layer is that swapping a source does not ripple downstream.
"""

from .base import ReadResult, SourceAdapter
from .fetch import ArchiveFetcher, ArchiveMember, HttpRangeFile, RangeNotSupported
from .meshindex import GRID_INDEX, Mesh, MeshIndex
from .nlni import NlniLanduseAdapter, load_landuse_spec
from .osm import OsmAdapter, assert_training_only_use, split_at_intersections
from .plateau import PlateauAdapter
from .virtual_shizuoka import VirtualShizuokaAdapter

__all__ = [
    "ArchiveFetcher",
    "ArchiveMember",
    "GRID_INDEX",
    "HttpRangeFile",
    "Mesh",
    "MeshIndex",
    "NlniLanduseAdapter",
    "OsmAdapter",
    "PlateauAdapter",
    "RangeNotSupported",
    "ReadResult",
    "SourceAdapter",
    "VirtualShizuokaAdapter",
    "assert_training_only_use",
    "load_landuse_spec",
    "split_at_intersections",
]
