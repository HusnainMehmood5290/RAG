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
    return s


def test_ingest_moves_file_and_skips_duplicate(monkeypatch, fake_settings, tmp_path):
    calls = []
    monkeypatch.setattr(ingest, "load_file", lambda p: calls.append(p) or 3)

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
    def boom(path):
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(ingest, "load_file", boom)
    pdf = fake_settings.raw_data_dir / "bad.pdf"
    pdf.write_bytes(b"data")

    assert ingest.ingest_file(pdf, fake_settings) is False
    assert pdf.exists(), "failed files must remain for retry"
    assert not any(p.name.endswith("__bad.pdf") for p in fake_settings.processed_dir.iterdir())
