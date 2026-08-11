"""Remote archive access, so a build does not require downloading everything first.

PLATEAU ships one ZIP per municipality per year, and they are enormous: **Atami — one of the
smaller MVP cities — is 15 GB**, because a package bundles LOD2 textures, disaster models, terrain
and 3D Tiles alongside the buildings we actually want. Three MVP sites would be 45 GB before a
single byte of terrain, and Shizuoka City is larger still.

Downloading a 15 GB archive to extract 40 MB of building GML is the wrong shape. The CDN answers
HTTP range requests, and a ZIP keeps its directory at the *end* of the file, so the archive can be
read remotely: fetch the central directory, then fetch only the members wanted.

:class:`HttpRangeFile` is a seekable file-like object over HTTP ranges. Handing it to the stdlib
``zipfile`` module means ZIP parsing — including Zip64, which a 15 GB archive requires — is handled
by code that already works, rather than by a hand-rolled parser that would need to be trusted.

Nothing here bypasses the provenance gate: :class:`ArchiveFetcher` takes a
:class:`~japgo.provenance.SourceGate` and asks permission before opening anything.
"""

from __future__ import annotations

import logging
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ..provenance import Source, SourceGate

log = logging.getLogger(__name__)

DEFAULT_CHUNK = 1 << 20  # 1 MiB
USER_AGENT = "japgo/0.1 (research data pipeline)"


class RangeNotSupported(RuntimeError):
    """The server will not serve byte ranges, so remote reading is impossible."""


class HttpRangeFile:
    """A seekable, read-only file-like object backed by HTTP range requests.

    Reads are served from a small block cache so that ``zipfile``'s many small seeks near the
    central directory do not become one request each.
    """

    def __init__(self, url: str, *, chunk_size: int = DEFAULT_CHUNK, timeout: float = 60.0) -> None:
        self.url = url
        self.chunk_size = chunk_size
        self.timeout = timeout
        self._pos = 0
        self._blocks: dict[int, bytes] = {}
        self._size = self._probe_size()
        self.bytes_fetched = 0

    # -- probing ------------------------------------------------------------------------------

    def _probe_size(self) -> int:
        """Determine total length, and confirm the server honours ranges.

        Uses a one-byte GET rather than HEAD: the PLATEAU asset CDN refuses HEAD outright.
        """
        request = urllib.request.Request(
            self.url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            content_range = response.headers.get("Content-Range")
            if not content_range:
                raise RangeNotSupported(
                    f"{self.url} did not answer with Content-Range; the whole file would have to "
                    "be downloaded."
                )
            match = re.search(r"/(\d+)$", content_range)
            if not match:
                raise RangeNotSupported(f"unparsable Content-Range: {content_range!r}")
            return int(match.group(1))

    # -- file protocol ------------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def read(self, amount: int = -1) -> bytes:
        if amount is None or amount < 0:
            amount = self._size - self._pos
        amount = min(amount, self._size - self._pos)
        if amount <= 0:
            return b""

        out = bytearray()
        remaining = amount
        while remaining > 0:
            index = self._pos // self.chunk_size
            block = self._block(index)
            offset = self._pos - index * self.chunk_size
            piece = block[offset : offset + remaining]
            if not piece:
                break
            out += piece
            self._pos += len(piece)
            remaining -= len(piece)
        return bytes(out)

    def close(self) -> None:  # pragma: no cover - nothing to release
        self._blocks.clear()

    # -- fetching -----------------------------------------------------------------------------

    def _block(self, index: int) -> bytes:
        cached = self._blocks.get(index)
        if cached is not None:
            return cached

        start = index * self.chunk_size
        end = min(start + self.chunk_size, self._size) - 1
        if end < start:
            return b""

        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = response.read()

        self.bytes_fetched += len(data)
        # Bounded cache. The access pattern is "central directory, then one member at a time", so
        # a small cache is enough and an unbounded one would defeat the point of streaming.
        if len(self._blocks) > 64:
            self._blocks.clear()
        self._blocks[index] = data
        return data


@dataclass
class ArchiveMember:
    name: str
    compressed_size: int
    uncompressed_size: int


@dataclass
class FetchReport:
    url: str
    members: list[str] = field(default_factory=list)
    bytes_fetched: int = 0
    archive_size: int = 0

    @property
    def saved_fraction(self) -> float:
        if not self.archive_size:
            return 0.0
        return 1.0 - (self.bytes_fetched / self.archive_size)


class ArchiveFetcher:
    """Selectively extracts members from a remote ZIP, under the provenance gate."""

    def __init__(
        self, gate: SourceGate, source_id: str, *, chunk_size: int = DEFAULT_CHUNK
    ) -> None:
        self.gate = gate
        self.source_id = source_id
        self.chunk_size = chunk_size
        """Range-request block size. The 1 MiB default suits multi-gigabyte archives, where the
        round-trip dominates; drop it for small archives, where it would fetch more than needed."""

    def open(self) -> Source:
        return self.gate.assert_ingestible(self.source_id)

    def list_members(self, url: str, *, pattern: str | None = None) -> list[ArchiveMember]:
        """List archive contents without downloading the payload."""
        self.open()
        handle = HttpRangeFile(url, chunk_size=self.chunk_size)
        with zipfile.ZipFile(handle) as archive:
            infos = archive.infolist()
        matcher = re.compile(pattern) if pattern else None
        return [
            ArchiveMember(i.filename, i.compress_size, i.file_size)
            for i in infos
            if not i.is_dir() and (matcher is None or matcher.search(i.filename))
        ]

    def extract(
        self,
        url: str,
        destination: Path,
        *,
        pattern: str,
        limit: int | None = None,
        max_bytes: int | None = None,
    ) -> FetchReport:
        """Extract members matching ``pattern`` into ``destination``.

        ``max_bytes`` caps the uncompressed volume written, so a mistaken pattern cannot quietly
        pull down the whole archive.
        """
        self.open()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        handle = HttpRangeFile(url, chunk_size=self.chunk_size)
        report = FetchReport(url=url, archive_size=handle.size)
        matcher = re.compile(pattern)
        written = 0

        with zipfile.ZipFile(handle) as archive:
            selected = [
                i for i in archive.infolist() if not i.is_dir() and matcher.search(i.filename)
            ]
            selected.sort(key=lambda i: i.header_offset)  # sequential reads beat random ones
            if limit is not None:
                selected = selected[:limit]

            for info in selected:
                if max_bytes is not None and written + info.file_size > max_bytes:
                    log.warning(
                        "stopping extraction at %s: would exceed max_bytes=%d", info.filename, max_bytes
                    )
                    break
                target = destination / Path(info.filename).name
                with archive.open(info) as src, target.open("wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                written += info.file_size
                report.members.append(info.filename)

        report.bytes_fetched = handle.bytes_fetched
        return report
