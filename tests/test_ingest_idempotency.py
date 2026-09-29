"""Ingestion worker logic: content-hash based idempotency (mocked pipeline)."""

from pathlib import Path

import pytest

from config import Settings
from ingestion import ingest


@pytest.fixture
def fake_settings(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    s = Settings.__new__(Settings)
    object.__setattr__(s, "raw_data_dir", raw)
    object.__setattr__(s, "processed_dir", processed)
    # Settings is frozen; give the partial instance a hashable identity so it
    # can be used as an lru_cache key in shared-retriever tests.
    object.__setattr__(s, "_test_id", id(s))
    # NOTE: monkeypatching __hash__ on the class is fine here because
    # no test relies on Settings being an lru_cache key.
    type(s).__hash__ = lambda self: id(self)
    type(s).__eq__ = lambda self, other: self is other
    return s


def test_ingest_moves_file_and_skips_duplicate(monkeypatch, fake_settings, tmp_path):
    calls = []
    monkeypatch.setattr(ingest, "load_file", lambda p, settings=None: calls.append(p) or 3)

    pdf = fake_settings.raw_data_dir / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    assert ingest.ingest_file(pdf, fake_settings) is True
    assert not pdf.exists()
    moved = list(fake_settings.processed_dir.iterdir())
    assert len(moved) == 1 and moved[0].name.endswith("__a.pdf")

    # Re-uploading identical content is detected and skipped (no re-index).
    pdf2 = fake_settings.raw_data_dir / "a.pdf"
    pdf2.write_bytes(b"%PDF-1.4 fake")
    assert ingest.ingest_file(pdf2, fake_settings) is True
    assert calls == [Path(pdf)], "duplicate content must not be re-ingested"
    assert not pdf2.exists()


def test_failed_ingest_keeps_file_for_retry(monkeypatch, fake_settings):
    def boom(path, settings=None):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(ingest, "load_file", boom)
    pdf = fake_settings.raw_data_dir / "bad.pdf"
    pdf.write_bytes(b"data")

    assert ingest.ingest_file(pdf, fake_settings) is False
    assert pdf.exists(), "failed files must remain for retry"
    assert not any(p.name.endswith("__bad.pdf") for p in fake_settings.processed_dir.iterdir())


def test_updated_file_is_reindexed(monkeypatch, fake_settings):
    """Same name + new content must trigger a fresh load_file call."""
    calls = []
    monkeypatch.setattr(ingest, "load_file", lambda p, settings=None: calls.append(p) or 1)

    pdf = fake_settings.raw_data_dir / "report.pdf"
    pdf.write_bytes(b"version one")
    assert ingest.ingest_file(pdf, fake_settings) is True

    # User uploads an updated version of the same file.
    pdf_new = fake_settings.raw_data_dir / "report.pdf"
    pdf_new.write_bytes(b"version two - different bytes")
    assert ingest.ingest_file(pdf_new, fake_settings) is True

    assert len(calls) == 2, "updated content must be re-indexed, not skipped"
    archived = {p.name for p in fake_settings.processed_dir.iterdir()}
    assert sum(n.endswith("__report.pdf") for n in archived) == 2


def test_scan_once_shares_one_retriever_across_batch(monkeypatch, fake_settings):
    """Performance guard: the heavy retriever is built once per scan, not per file."""
    builds = []

    class FakeRetriever:
        pass

    def fake_build(settings=None):
        builds.append(settings)
        return FakeRetriever()

    from ingestion import helper

    monkeypatch.setattr(helper, "build_parent_retriever", fake_build)
    helper.get_ingest_retriever.cache_clear()

    docs = []

    def fake_load(path, settings=None):
        # simulate indexing through the shared retriever
        helper.get_ingest_retriever(settings)
        docs.append(path)
        return 1

    monkeypatch.setattr(ingest, "load_file", fake_load)

    for name in ("a.pdf", "b.pdf", "c.pdf"):
        (fake_settings.raw_data_dir / name).write_bytes(name.encode())

    ok = ingest.scan_once(fake_settings)
    assert ok == 3
    assert len(builds) == 1, "retriever/embedding model must be built once per batch"
    # cache cleared after the scan so long-lived watch loops release stores
    assert helper.get_ingest_retriever.cache_info().currsize == 0
