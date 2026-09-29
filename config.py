"""Centralized, validated application configuration.

All environment variables are read exactly once, through ``get_settings()``.
Missing required variables fail fast with a clear error message instead of
crashing deep inside the application with an obscure ``TypeError``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent


class ConfigurationError(RuntimeError):
    """Raised when required environment configuration is missing or invalid."""


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(
            f"Required environment variable '{name}' is not set. "
            "Copy '.env.example' to '.env' and fill in the values."
        )
    return value


def _optional_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Environment variable '{name}' must be an integer, got: {raw!r}"
        ) from exc


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the application configuration."""

    # --- Required secrets / identifiers -------------------------------------
    google_api_key: str = field(default_factory=lambda: _require("GOOGLE_API_KEY"))

    # --- Model selection ------------------------------------------------------
    llm_name: str = field(default_factory=lambda: os.getenv("LLM_NAME", "gemini-1.5-flash"))
    embedding_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_NAME", "sentence-transformers/all-mpnet-base-v2"
        )
    )

    # --- Chunking ---------------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _optional_int("CHUNK_SIZE", 400))
    chunk_overlap: int = field(default_factory=lambda: _optional_int("CHUNK_OVERLAP", 50))
    parent_chunk_size: int = field(default_factory=lambda: _optional_int("PARENT_CHUNK_SIZE", 4000))

    # --- Storage paths (resolved relative to the project root) ----------------
    local_store_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("LOCAL_STORE", "store/local_store")
    )
    vector_store_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("VECTOR_STORE", "store/vector_store")
    )
    collection_name: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "documents"))

    # --- Ingestion -----------------------------------------------------------
    raw_data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "ingestion/raw_data")
    )
    processed_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("PROCESSED_DIR", "ingestion/processed")
    )
    watch_interval_seconds: int = field(
        default_factory=lambda: _optional_int("WATCH_INTERVAL_SECONDS", 10)
    )

    # --- Retrieval / generation ----------------------------------------------
    retrieval_k: int = field(default_factory=lambda: _optional_int("RETRIEVAL_K", 4))

    def validate(self) -> None:
        """Cross-field validation with human-readable errors."""
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})."
            )
        if self.retrieval_k < 1:
            raise ConfigurationError(f"RETRIEVAL_K must be >= 1, got {self.retrieval_k}.")

    def ensure_directories(self) -> None:
        for path in (
            self.local_store_path,
            self.vector_store_path,
            self.raw_data_dir,
            self.processed_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load ``.env`` (once), build and validate settings."""
    load_dotenv(PROJECT_ROOT / ".env")
    settings = Settings()
    settings.validate()
    settings.ensure_directories()
    return settings
