"""Phase 2 visualisation — the alignment gate.

No modelling begins until a human has confirmed every layer is spatially aligned. Vector/raster
misalignment corrupts training without producing an error, which is why this is a phase gate
rather than a convenience.
"""

from .image import COLOURMAPS, colourise, png_bytes, shade
from .report import render_tile, summarise, write_index_page, write_report

__all__ = [
    "COLOURMAPS",
    "colourise",
    "png_bytes",
    "render_tile",
    "shade",
    "summarise",
    "write_index_page",
    "write_report",
]
