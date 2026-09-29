"""Lazy, cached model singletons (LLM + embeddings).

Models are heavy to initialize, so they are constructed on first use and
reused process-wide. Errors propagate to the caller instead of being
swallowed and returning ``None``.
"""

from __future__ import annotations

import logging
from functools import lru_cache

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
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Return a cached HuggingFace embedding model."""
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    logger.info("Initializing embeddings: %s", settings.embedding_name)
    return HuggingFaceEmbeddings(model_name=settings.embedding_name)
