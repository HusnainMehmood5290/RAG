"""Ingestion worker: watches ``raw_data/`` for new PDFs and indexes them.

Idempotency is content-based: successfully ingested files are moved to
``processed/`` under a ``<sha256-prefix>__<name>`` filename, so identical
re-uploads are detected and skipped instead of duplicated in the store.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from config import Settings, get_settings
from ingestion.helper import file_fingerprint, load_file
from logging_config import configure_logging

logger = logging.getLogger(__name__)


def _already_processed(target_dir: Path, fingerprint: str, name: str) -> bool:
    return (target_dir / f"{fingerprint[:16]}__{name}").exists()


def ingest_file(path: Path, settings: Settings) -> bool:
    """Ingest a single PDF. Returns True on success."""
    fingerprint = file_fingerprint(path)
    if _already_processed(settings.processed_dir, fingerprint, path.name):
        logger.info("Skipping (content already ingested): %s", path.name)
        path.unlink(missing_ok=True)
        return True

    try:
        count = load_file(path)
    except Exception:
        logger.exception("Failed to ingest %s - leaving it in raw_data for retry.", path)
        return False

    dest = settings.processed_dir / f"{fingerprint[:16]}__{path.name}"
    path.replace(dest)
    logger.info("Ingested %s (%d documents) -> %s", path.name, count, dest.name)
    return True


def scan_once(settings: Settings) -> int:
    """Process every pending PDF in the raw-data folder. Returns success count."""
    pdfs = sorted(p for p in settings.raw_data_dir.glob("*.pdf") if p.is_file())
    if not pdfs:
        logger.debug("No new PDFs found in %s", settings.raw_data_dir)
    return sum(1 for pdf in pdfs if ingest_file(pdf, settings))


def watch(settings: Settings) -> None:
    logger.info(
        "Watching %s for new PDFs every %ds. Press Ctrl+C to stop.",
        settings.raw_data_dir,
        settings.watch_interval_seconds,
    )
    try:
        while True:
            scan_once(settings)
            time.sleep(settings.watch_interval_seconds)
    except KeyboardInterrupt:
        logger.info("Interrupted. Exiting gracefully.")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="PDF ingestion worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Scan and ingest pending files, then exit (for cron/CI use).",
    )
    args = parser.parse_args()
    settings = get_settings()
    if args.once:
        scan_once(settings)
    else:
        watch(settings)


if __name__ == "__main__":
    main()
