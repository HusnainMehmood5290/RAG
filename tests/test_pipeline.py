"""Unit tests for the RAG pipeline (LLM and retriever are mocked).

These tests verify the retrieval -> prompt -> generation wiring without
requiring API keys, model downloads, or vector stores.
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import rag.pipeline as pipeline


@pytest.fixture(autouse=True)
def reset_caches():
    """Ensure lru_cache state never leaks between tests."""
    pipeline.get_retriever.cache_clear()
    yield
    pipeline.get_retriever.cache_clear()


DOCS = [
    Document(
        page_content="Alpha corp revenue grew 10%.",
        metadata={"source": "a.pdf", "Header 1": "Alpha"},
    ),
    Document(page_content="Beta corp lost money.", metadata={"source": "b.pdf"}),
]


def test_format_docs_includes_titles_and_content():
    text = pipeline.format_docs(DOCS)
    assert "Alpha corp revenue" in text
    assert "(Alpha)" in text  # header title used when present
    assert "(b.pdf)" in text  # falls back to source
    assert "---" in text  # separator between blocks


def test_format_docs_empty_returns_placeholder():
    assert "no documents" in pipeline.format_docs([]).lower()


def test_answer_question_grounds_llm_in_single_retrieval(monkeypatch):
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = DOCS
    monkeypatch.setattr(pipeline, "get_retriever", lambda: fake_retriever)

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="Grounded answer.")
    import models.model as model_mod

    monkeypatch.setattr(model_mod, "get_llm", lambda: fake_llm)

    result = pipeline.answer_question("How did Alpha perform?")

    assert result["answer"] == "Grounded answer."
    assert result["context"] == DOCS
    # retriever invoked exactly once with the question
    fake_retriever.invoke.assert_called_once_with("How did Alpha perform?")
    # LLM received a [system, human] message list containing the retrieved context
    (messages,), _ = fake_llm.invoke.call_args
    rendered = "\n".join(m.content for m in messages)
    assert "Alpha corp revenue" in rendered
    assert "How did Alpha perform?" in rendered


def test_answer_question_strips_whitespace(monkeypatch):
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = DOCS
    monkeypatch.setattr(pipeline, "get_retriever", lambda: fake_retriever)
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="ok")
    import models.model as model_mod

    monkeypatch.setattr(model_mod, "get_llm", lambda: fake_llm)

    pipeline.answer_question("   padded query   ")
    fake_retriever.invoke.assert_called_once_with("padded query")


@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_answer_question_rejects_empty(bad):
    with pytest.raises(ValueError, match="must not be empty"):
        pipeline.answer_question(bad)


def test_non_string_llm_content_is_coerced(monkeypatch):
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = DOCS
    monkeypatch.setattr(pipeline, "get_retriever", lambda: fake_retriever)
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content=[{"text": "part"}])
    import models.model as model_mod

    monkeypatch.setattr(model_mod, "get_llm", lambda: fake_llm)

    result = pipeline.answer_question("q")
    assert isinstance(result["answer"], str) and "part" in result["answer"]
