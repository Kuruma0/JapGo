"""The engine-independent core representation.

Nothing in this package may import or reference a game-engine type. Engines are exporters that
consume a versioned interchange bundle; if logic would otherwise live in two exporters, it belongs
here instead.
"""

from .buildings import Building, LabelSource, Taxonomy, load_taxonomy
from .manifest import SCHEMA_VERSION, SourceRecord, TileManifest
from .roads import (
    Control,
    Edge,
    Node,
    NodeKind,
    RoadClassSpec,
    RoadGraph,
    RoadHierarchy,
    Structure,
    load_hierarchy,
)

__all__ = [
    "SCHEMA_VERSION",
    "Building",
    "Control",
    "Edge",
    "LabelSource",
    "Node",
    "NodeKind",
    "RoadClassSpec",
    "RoadGraph",
    "RoadHierarchy",
    "SourceRecord",
    "Structure",
    "Taxonomy",
    "TileManifest",
    "load_hierarchy",
    "load_taxonomy",
]
