"""Per-dataset source adapters.

Every adapter passes through the provenance gate in :meth:`SourceAdapter.open` before reading a
byte. Adapters return core representation objects, never raw source structures — the point of the
layer is that swapping a source does not ripple downstream.
"""

from .base import ReadResult, SourceAdapter
from .plateau import PlateauAdapter
from .virtual_shizuoka import VirtualShizuokaAdapter

__all__ = ["PlateauAdapter", "ReadResult", "SourceAdapter", "VirtualShizuokaAdapter"]
