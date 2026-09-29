"""Lazy, cached model singletons (LLM + embeddings).

Models are heavy to initialize, so they are constructed on first use and
reused process-wide. Errors propagate to the caller instead of being
swallowed and returning ``None``.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import httpx

from config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm():
    """Return a cached ChatGoogleGenerativeAI instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    logger.info("Initializing LLM: %s", settings.llm_name)
    return ChatGoogleGenerativeAI(
        model=settings.llm_name,
        google_api_key=settings.google_api_key.get_secret_value(),
        temperature=0.2,
        # Explicit request timeout so a hung API call fails fast and visibly
        # instead of blocking the CLI/UI indefinitely. (google-genai already
        # retries transient 429/5xx internally with exponential backoff.)
        additional_args={
            "timeout": httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
        },
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Return a cached HuggingFace embedding model.

    ``normalize_embeddings=True`` produces unit-length vectors so cosine and
    dot-product ranking agree; sentence-transformers batches encode calls
    internally during ingestion. Set ``HF_EMBEDDINGS_DEVICE=cuda`` to run the
    encoder on the GPU (large speedup for bulk ingestion); defaults to CPU.
    """
    import os

    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    device = os.getenv("HF_EMBEDDINGS_DEVICE", "cpu")
    logger.info("Initializing embeddings: %s (device=%s)", settings.embedding_name, device)
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
