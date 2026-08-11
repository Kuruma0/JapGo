"""Tests for remote archive access.

Served from a local HTTP server rather than the real CDN: the behaviour under test is range
handling and selective extraction, and a test that depends on a 15 GB asset staying online is a
test that will fail for reasons unrelated to the code.
"""

from __future__ import annotations

import http.server
import threading
import zipfile
from pathlib import Path

import pytest

from japgo.provenance import ProvenanceViolation
from japgo.sources.fetch import ArchiveFetcher, HttpRangeFile, RangeNotSupported


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """A minimal range-capable static server.

    ``SimpleHTTPRequestHandler`` ignores Range entirely, so it cannot exercise the code under
    test. Real origins — including the PLATEAU asset CDN — do honour ranges.
    """

    directory: Path = Path()

    def _resolve(self) -> Path:
        return self.directory / self.path.lstrip("/")

    def do_GET(self):  # noqa: N802
        path = self._resolve()
        if not path.is_file():
            self.send_error(404)
            return

        data = path.read_bytes()
        header = self.headers.get("Range")

        if header and header.startswith("bytes="):
            spec = header.split("=", 1)[1]
            start_s, _, end_s = spec.partition("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else len(data) - 1
            end = min(end, len(data) - 1)
            body = data[start : end + 1]

            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class _NoRangeHandler(http.server.BaseHTTPRequestHandler):
    """A server that ignores Range headers, as some origins do."""

    payload = b"x" * 5000

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()
        self.wfile.write(self.payload)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def archive_server(tmp_path_factory):
    """Serve a directory containing a multi-member ZIP over HTTP with range support."""
    root = tmp_path_factory.mktemp("served")

    with zipfile.ZipFile(root / "package.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(6):
            zf.writestr(f"udx/bldg/5239400{i}_bldg_6697_op.gml", f"<gml>{'A' * (2000 * (i + 1))}</gml>")
        zf.writestr("codelists/Building_usage.xml", "<Dictionary/>")
        zf.writestr("udx/tran/road.gml", "<gml>roads</gml>")
        zf.writestr("metadata.xml", "<meta/>")

    handler = type("_Bound", (_RangeHandler,), {"directory": root})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/package.zip"
    server.shutdown()


@pytest.fixture(scope="module")
def no_range_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NoRangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/whatever"
    server.shutdown()


# ---------------------------------------------------------------------------------------------
# HttpRangeFile
# ---------------------------------------------------------------------------------------------


def test_reports_total_size(archive_server):
    assert HttpRangeFile(archive_server).size > 0


def test_reads_match_a_local_read(archive_server, tmp_path):
    handle = HttpRangeFile(archive_server, chunk_size=1024)
    handle.seek(0)
    remote = handle.read()

    import urllib.request

    with urllib.request.urlopen(archive_server) as response:
        local = response.read()
    assert remote == local


def test_seek_and_tell(archive_server):
    handle = HttpRangeFile(archive_server, chunk_size=512)
    assert handle.seek(100) == 100
    assert handle.tell() == 100
    assert handle.seek(-10, 2) == handle.size - 10
    assert len(handle.read()) == 10


def test_seek_is_clamped_to_the_file(archive_server):
    handle = HttpRangeFile(archive_server)
    assert handle.seek(10**12) == handle.size
    assert handle.read() == b""


def test_blocks_are_cached(archive_server):
    """zipfile makes many small seeks near the directory; each must not be its own request."""
    handle = HttpRangeFile(archive_server, chunk_size=1 << 20)
    handle.seek(0)
    handle.read(10)
    first = handle.bytes_fetched
    handle.seek(20)
    handle.read(10)
    assert handle.bytes_fetched == first


def test_server_without_range_support_is_refused(no_range_server):
    """Better to fail clearly than to silently download 15 GB."""
    with pytest.raises(RangeNotSupported):
        HttpRangeFile(no_range_server)


# ---------------------------------------------------------------------------------------------
# ZIP over ranges
# ---------------------------------------------------------------------------------------------


def test_lists_members_without_downloading_the_payload(archive_server, gate):
    fetcher = ArchiveFetcher(gate, "plateau")
    members = fetcher.list_members(archive_server)
    assert any(m.name.endswith("road.gml") for m in members)
    assert len(members) == 9


def test_member_pattern_filters(archive_server, gate):
    fetcher = ArchiveFetcher(gate, "plateau")
    members = fetcher.list_members(archive_server, pattern=r"udx/bldg/.*\.gml$")
    assert len(members) == 6
    assert all("bldg" in m.name for m in members)


def test_extracts_only_the_requested_members(archive_server, gate, tmp_path):
    """The saving is the whole point: a small chunk makes it observable on a tiny fixture.

    With the 1 MiB default this archive fits in one block, so partial fetching only shows up once
    the archive is much larger than the chunk — which is exactly the real case (15 GB).
    """
    fetcher = ArchiveFetcher(gate, "plateau", chunk_size=256)
    report = fetcher.extract(archive_server, tmp_path, pattern=r"udx/bldg/.*\.gml$", limit=2)

    written = sorted(p.name for p in tmp_path.glob("*.gml"))
    assert len(written) == 2
    assert len(report.members) == 2
    assert report.bytes_fetched < report.archive_size
    assert report.saved_fraction > 0


def test_extracted_content_is_correct(archive_server, gate, tmp_path):
    fetcher = ArchiveFetcher(gate, "plateau")
    fetcher.extract(archive_server, tmp_path, pattern=r"52394000_bldg", limit=1)
    text = (tmp_path / "52394000_bldg_6697_op.gml").read_text(encoding="utf-8")
    assert text.startswith("<gml>")
    assert text.count("A") == 2000


def test_max_bytes_caps_a_runaway_pattern(archive_server, gate, tmp_path):
    """A mistaken pattern must not quietly pull the whole archive."""
    fetcher = ArchiveFetcher(gate, "plateau")
    report = fetcher.extract(archive_server, tmp_path, pattern=r".*", max_bytes=3000)
    assert len(report.members) < 9


def test_fetching_is_gated_by_provenance(archive_server, gate, tmp_path):
    """A quarantined source cannot be fetched, however convenient the URL."""
    fetcher = ArchiveFetcher(gate, "gsi_tiles")
    with pytest.raises(ProvenanceViolation, match="quarantined"):
        fetcher.list_members(archive_server)


def test_unregistered_source_cannot_be_fetched(archive_server, gate):
    fetcher = ArchiveFetcher(gate, "some_random_site")
    with pytest.raises(LookupError, match="no registry entry"):
        fetcher.extract(archive_server, Path("."), pattern=r".*")
