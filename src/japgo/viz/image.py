"""Array to PNG, without a plotting dependency.

The visualiser must run on any machine that can run the pipeline, including a headless Linux
training box (invariant: Linux is a likely target). Pulling in matplotlib or Pillow for what is
ultimately "write an RGB array as a PNG" would add a heavyweight dependency to the one tool whose
job is to be available when something looks wrong.

A PNG is a zlib stream plus four chunks. That is cheap to write directly.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np


def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` or ``(H, W, 4)`` uint8 array as a PNG."""
    if rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {rgb.dtype}")
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError(f"expected (H, W, 3) or (H, W, 4), got {rgb.shape}")

    height, width, channels = rgb.shape
    colour_type = 2 if channels == 3 else 6

    # Each scanline is prefixed with a filter byte; 0 means "no filtering".
    raw = np.hstack(
        [np.zeros((height, 1), np.uint8), rgb.reshape(height, width * channels)]
    ).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


# ------------------------------------------------------------------------------------------------
# Colour maps
#
# Perceptually ordered and colour-blind safe where it matters. Kept as short control-point lists
# and interpolated, rather than 256-entry tables, so they stay readable and editable.
# ------------------------------------------------------------------------------------------------

_VIRIDIS = [
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
    (180, 222, 44), (253, 231, 37),
]

_TERRAIN = [
    (60, 90, 130), (90, 140, 90), (140, 175, 100), (200, 200, 130),
    (190, 160, 110), (160, 120, 90), (170, 160, 155), (250, 250, 250),
]

#: Diverging, for signed fields such as tile-relative elevation. Blue-white-red.
_DIVERGING = [
    (33, 102, 172), (103, 169, 207), (209, 229, 240), (247, 247, 247),
    (253, 219, 199), (239, 138, 98), (178, 24, 43),
]

_MAGMA = [
    (0, 0, 4), (28, 16, 68), (79, 18, 123), (129, 37, 129),
    (181, 54, 122), (229, 80, 100), (251, 135, 97), (254, 194, 135),
    (252, 253, 191),
]

COLOURMAPS = {
    "viridis": _VIRIDIS,
    "terrain": _TERRAIN,
    "diverging": _DIVERGING,
    "magma": _MAGMA,
    "grey": [(0, 0, 0), (255, 255, 255)],
}


def _ramp(control: list[tuple[int, int, int]], size: int = 256) -> np.ndarray:
    points = np.asarray(control, dtype=np.float64)
    positions = np.linspace(0.0, 1.0, len(points))
    target = np.linspace(0.0, 1.0, size)
    return np.stack(
        [np.interp(target, positions, points[:, i]) for i in range(3)], axis=1
    ).astype(np.uint8)


def colourise(
    data: np.ndarray,
    *,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Map a 2-D array to RGBA.

    Cells excluded by ``mask`` — and any NaN — become fully transparent rather than a colour.
    Painting nodata as a colour is how a void starts looking like a measurement.
    """
    values = np.asarray(data, dtype=np.float64)
    finite = np.isfinite(values)
    if mask is not None:
        finite &= mask.astype(bool)

    if vmin is None:
        vmin = float(np.nanmin(values[finite])) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(values[finite])) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9

    normalised = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)
    normalised = np.nan_to_num(normalised, nan=0.0)

    ramp = _ramp(COLOURMAPS.get(cmap, _VIRIDIS))
    indices = (normalised * (len(ramp) - 1)).astype(np.int32)

    rgba = np.zeros((*values.shape, 4), np.uint8)
    rgba[..., :3] = ramp[indices]
    rgba[..., 3] = np.where(finite, 255, 0)
    return rgba


def shade(hillshade: np.ndarray) -> np.ndarray:
    """Grey RGBA from a 0-255 hillshade, used as the base layer everything else sits on."""
    values = np.nan_to_num(np.asarray(hillshade, dtype=np.float64), nan=128.0)
    grey = np.clip(values, 0, 255).astype(np.uint8)
    rgba = np.zeros((*grey.shape, 4), np.uint8)
    for i in range(3):
        rgba[..., i] = grey
    rgba[..., 3] = 255
    return rgba


def downsample_rgba(rgba: np.ndarray, factor: int) -> np.ndarray:
    """Decimate for display. A 1512² PNG per layer is a slow page for no extra insight."""
    if factor <= 1:
        return rgba
    return rgba[::factor, ::factor]
