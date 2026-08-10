"""The engine-independent core representation.

Nothing in this package may import or reference a game-engine type. Engines are exporters that
consume a versioned interchange bundle; if logic would otherwise live in two exporters, it belongs
here instead.
"""

from .manifest import SCHEMA_VERSION, SourceRecord, TileManifest

__all__ = ["SCHEMA_VERSION", "SourceRecord", "TileManifest"]
