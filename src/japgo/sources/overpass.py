"""Fetching OSM road geometry from the Overpass API.

The road layer is the Phase 3 response variable and the Phase 4 target, and it was the one input
with no automated path: :class:`~japgo.sources.osm.OsmAdapter` reads ``.osm`` XML from disk, and
nothing produced that XML. This closes the gap.

Two properties matter more than convenience here.

**It stays training-only.** OSM geometry is ODbL and must never reach shipped reconstruction
output (invariant 3b, research doc §6.1c). Every fetch declares a purpose and is checked by
:func:`~japgo.sources.osm.assert_training_only_use` before a request is made, so an attempted
``export`` fails at the point of the mistake rather than at the end of a pipeline run.

**It caches, and it fetches per region rather than per tile.** Overpass is a free shared service
with real usage limits. One query per 1 km tile over a corpus would be both slow and rude; one
query per site, cached to disk and reused on every rebuild, is neither. A rebuilt corpus makes no
requests at all.
"""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..geo.crs import to_wgs84
from ..geo.tiling import Bounds
from ..provenance import SourceGate
from .osm import assert_training_only_use

log = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://overpass-api.de/api/interpreter"
SOURCE_ID = "osm"
USER_AGENT = "japgo/0.1 (research data pipeline; contact via repository)"
"""Overpass asks clients to identify themselves. An anonymous bulk fetcher gets blocked, and
deservedly."""

HIGHWAY_QUERY = """[out:xml][timeout:{timeout}];
way["highway"]({south:.6f},{west:.6f},{north:.6f},{east:.6f});
(._;>;);
out body;
"""


class OverpassError(RuntimeError):
    """The API refused, timed out, or returned something that is not an OSM document."""


@dataclass(frozen=True)
class FetchedRoads:
    path: Path
    bytes_fetched: int
    from_cache: bool


class OverpassClient:
    """Fetches highway ways over a bounding box, caching the raw XML."""

    def __init__(
        self,
        gate: SourceGate,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        cache_dir: Path | None = None,
        timeout: int = 180,
        retries: int = 3,
        backoff_s: float = 5.0,
    ) -> None:
        self.gate = gate
        self.endpoint = endpoint
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.timeout = timeout
        self.retries = retries
        self.backoff_s = backoff_s

    def open(self):
        return self.gate.assert_ingestible(SOURCE_ID)

    def query_for(self, bounds: Bounds, crs) -> str:
        """The Overpass QL for a metric-CRS bounding box, converted to WGS84 at the boundary."""
        west, south = to_wgs84(bounds.minx, bounds.miny, crs)
        east, north = to_wgs84(bounds.maxx, bounds.maxy, crs)
        return HIGHWAY_QUERY.format(
            timeout=self.timeout, south=south, west=west, north=north, east=east
        )

    def fetch(
        self,
        bounds: Bounds,
        crs,
        *,
        purpose: str = "training",
        key: str | None = None,
    ) -> FetchedRoads:
        """Fetch roads over ``bounds`` and return the path to the cached ``.osm`` document."""
        assert_training_only_use(purpose)
        self.open()

        query = self.query_for(bounds, crs)
        cache = self._cache_path(key, query)
        if cache is not None and cache.is_file() and cache.stat().st_size > 0:
            log.info("overpass: cache hit %s", cache.name)
            return FetchedRoads(path=cache, bytes_fetched=0, from_cache=True)

        payload = self._request(query)
        if b"<osm" not in payload[:2000]:
            raise OverpassError(
                f"response from {self.endpoint} is not an OSM document; first bytes: "
                f"{payload[:200]!r}"
            )

        destination = cache or Path(f"overpass_{_digest(query)}.osm")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return FetchedRoads(path=destination, bytes_fetched=len(payload), from_cache=False)

    # -- internals ------------------------------------------------------------------------------

    def _cache_path(self, key: str | None, query: str) -> Path | None:
        if self.cache_dir is None:
            return None
        name = f"{key}_" if key else ""
        return self.cache_dir / f"{name}{_digest(query)}.osm"

    def _request(self, query: str) -> bytes:
        """POST the query, retrying on the rate-limit and gateway responses Overpass uses.

        429 and 504 are the documented "you are asking too fast" and "the query took too long"
        answers. Both are worth retrying with a pause; anything else is a real error and is raised
        immediately rather than hammered at.
        """
        data = urllib.parse.urlencode({"data": query}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=data, headers={"User-Agent": USER_AGENT}
        )

        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout + 30) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in {429, 502, 503, 504}:
                    raise OverpassError(f"{self.endpoint} returned HTTP {exc.code}") from exc
                pause = self.backoff_s * attempt
                log.warning(
                    "overpass HTTP %d (attempt %d/%d); waiting %.0fs",
                    exc.code, attempt, self.retries, pause,
                )
                time.sleep(pause)
            except urllib.error.URLError as exc:
                last = exc
                pause = self.backoff_s * attempt
                log.warning(
                    "overpass unreachable (%s), attempt %d/%d; waiting %.0fs",
                    exc.reason, attempt, self.retries, pause,
                )
                time.sleep(pause)

        raise OverpassError(f"{self.endpoint} failed after {self.retries} attempts: {last}")


def _digest(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
