"""Tests for centralized configuration validation."""

import dataclasses

import pytest

import config
from config import ConfigurationError, Settings, _optional_int


def _clear_cache():
    config.get_settings.cache_clear()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Isolate each test from ambient env + cached settings.

    Also disable the real .env file so tests only see what they set.
    """
    for key in ("GOOGLE_API_KEY", "CHUNK_SIZE", "CHUNK_OVERLAP", "RETRIEVAL_K"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **kw: False)
    _clear_cache()
    yield
    _clear_cache()


def test_missing_api_key_fails_fast(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ConfigurationError, match="GOOGLE_API_KEY"):
        config.get_settings()


def test_non_integer_chunk_size_fails_with_clear_message(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CHUNK_SIZE", "abc")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ConfigurationError, match="'CHUNK_SIZE' must be an integer"):
        config.get_settings()


def test_overlap_ge_size_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "150")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ConfigurationError, match="smaller than"):
        config.get_settings()


def test_defaults_applied(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    settings = config.get_settings()
    assert settings.chunk_size == 400
    assert settings.chunk_overlap == 50
    assert settings.retrieval_k == 4
    assert settings.collection_name == "documents"
    assert settings.max_context_chars == 8000
    # ensure_directories created the store folders under tmp project root
    assert (tmp_path / "store/local_store").is_dir()
    assert (tmp_path / "store/vector_store").is_dir()


def test_negative_chunk_size_rejected_early(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("CHUNK_SIZE", "-10")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ConfigurationError, match="CHUNK_SIZE.*>= 1"):
        config.get_settings()


def test_zero_retrieval_k_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    monkeypatch.setenv("RETRIEVAL_K", "0")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ConfigurationError, match="'RETRIEVAL_K' must be >= 1"):
        config.get_settings()


def test_api_key_hidden_from_repr(monkeypatch, tmp_path):
    """The raw secret must never appear in repr()/logs (SecretStr guard)."""
    monkeypatch.setenv("GOOGLE_API_KEY", "super-secret-value")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    settings = config.get_settings()
    assert "super-secret-value" not in repr(settings)
    assert settings.google_api_key.get_secret_value() == "super-secret-value"


def test_optional_int_parses_and_defaults():
    import os

    os.environ["_TEST_INT"] = "42"
    assert _optional_int("_TEST_INT", 7) == 42
    del os.environ["_TEST_INT"]
    assert _optional_int("_TEST_INT", 7) == 7


def test_settings_is_frozen():
    assert Settings.__dataclass_params__.frozen is True
    frozen = Settings.__new__(Settings)
    object.__setattr__(frozen, "google_api_key", "k")
    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.google_api_key = "other"
