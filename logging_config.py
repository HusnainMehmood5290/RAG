"""Project-wide logging setup."""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger exactly once, idempotently.

    Level can be set via the ``LOG_LEVEL`` env var (DEBUG/INFO/WARNING/...).
    """
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    if root.handlers:  # already configured (e.g. by another entrypoint)
        root.setLevel(resolved)
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.setLevel(resolved)
    # Third-party loggers are extremely chatty at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
