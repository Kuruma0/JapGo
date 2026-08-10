"""The raster stack specification.

The ordered channel list the model consumes, loaded from ``config/raster_stack.yaml``.

Channels declare the registry source they derive from. That single fact is what lets the assembler
compute a tile's contributing sources — and therefore its ``redistribution_class`` and attribution
block — automatically. A hand-maintained list of "what went into this tile" is a list that will
eventually be wrong, and being wrong about it is a licensing failure rather than a bug.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_STACK_PATH = Path("config/raster_stack.yaml")


class Normalise(StrEnum):
    NONE = "none"
    SCALE = "scale"
    TILE_RELATIVE = "tile_relative"


class Channel(BaseModel):
    """One channel of the raster stack."""

    model_config = ConfigDict(frozen=True, extra="allow")

    name: str
    source: str | None = None
    """Registry source id, or ``None`` for computed channels such as ``valid``."""

    derived_from: str | None = None
    units: str = "unit"
    normalise: Normalise = Normalise.NONE
    scale: float | None = None
    note: str | None = None

    def apply_normalisation(self, data: np.ndarray) -> np.ndarray:
        """Normalise a channel's values for model consumption."""
        out = data.astype(np.float32)

        if self.normalise is Normalise.SCALE:
            if not self.scale:
                raise ValueError(f"channel {self.name!r} uses scale normalisation but has no scale")
            return out / np.float32(self.scale)

        if self.normalise is Normalise.TILE_RELATIVE:
            valid = ~np.isnan(out)
            if not valid.any():
                return np.zeros_like(out)
            return out - np.float32(out[valid].mean())

        return out


class StackSpec(BaseModel):
    """The full stack specification."""

    model_config = ConfigDict(frozen=True)

    stack_version: int
    nodata_fill: float = 0.0
    channels: list[Channel] = Field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.channels]

    @property
    def depth(self) -> int:
        """Model input width. Derivable from config alone, without reading code."""
        return len(self.channels)

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"no channel {name!r}; have {self.names}") from exc

    def get(self, name: str) -> Channel:
        return self.channels[self.index_of(name)]

    @property
    def required_sources(self) -> list[str]:
        """Registry source ids this stack depends on, in declaration order."""
        seen: list[str] = []
        for c in self.channels:
            if c.source and c.source not in seen:
                seen.append(c.source)
        return seen

    def channels_for(self, source_id: str) -> list[Channel]:
        return [c for c in self.channels if c.source == source_id]


def _find_stack(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        path = candidate / DEFAULT_STACK_PATH
        if path.is_file():
            return path
    raise FileNotFoundError(f"could not find {DEFAULT_STACK_PATH} walking up from {here}")


@lru_cache(maxsize=4)
def load_stack_spec(path: Path | None = None) -> StackSpec:
    path = path or _find_stack()
    return StackSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
