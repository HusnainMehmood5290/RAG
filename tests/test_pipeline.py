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


def _fake_settings(**overrides):
    s = MagicMock()
    s.max_context_chars = overrides.get("max_context_chars", 10_000)
    return s


def test_format_docs_includes_titles_and_content():
    text, truncated = pipeline.format_docs(DOCS)
    assert "Alpha corp revenue" in text
    assert "(Alpha)" in text  # header title used when present
    assert "(b.pdf)" in text  # falls back to source
    assert "---" in text  # separator between blocks
    assert truncated is False


def test_format_docs_empty_returns_placeholder():
    text, truncated = pipeline.format_docs([])
    assert "no documents" in text.lower()
    assert truncated is False


def test_format_docs_truncates_to_budget():
    big = [Document(page_content="X" * 100, metadata={"source": f"{i}.pdf"}) for i in range(5)]
    text, truncated = pipeline.format_docs(big, max_chars=250)
    assert truncated is True
    assert "omitted" in text
    # First doc always kept; later docs dropped once budget is exceeded.
    assert text.count("[") >= 1
    assert len([ln for ln in text.split("---") if "XXX" in ln]) < 5


def test_prompt_wraps_context_in_tags():
    messages = pipeline.PROMPT.invoke({"context": "CTX", "input": "Q?"}).to_messages()
    human = messages[-1].content
    assert "<context>" in human and "</context>" in human
    assert "untrusted DATA" in human  # injection guard instruction present


def test_answer_question_grounds_llm_in_single_retrieval(monkeypatch):
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = DOCS
    monkeypatch.setattr(pipeline, "get_retriever", lambda: fake_retriever)

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="Grounded answer.")
    import models.model as model_mod

    monkeypatch.setattr(model_mod, "get_llm", lambda: fake_llm)

    result = pipeline.answer_question("How did Alpha perform?", settings=_fake_settings())

    assert result.answer == "Grounded answer."
    assert result.context == DOCS
    assert result.truncated is False
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

    pipeline.answer_question("   padded query   ", settings=_fake_settings())
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

    result = pipeline.answer_question("q", settings=_fake_settings())
    assert isinstance(result.answer, str) and "part" in result.answer


def test_history_condenses_search_query(monkeypatch):
    """With conversation history, retrieval uses the condensed standalone query
    while the LLM still sees the original user wording."""
    fake_retriever = MagicMock()
    fake_retriever.invoke.return_value = DOCS
    monkeypatch.setattr(pipeline, "get_retriever", lambda: fake_retriever)

    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [
        MagicMock(content="What is the revenue and profit of Alpha corp?"),  # condensation
        MagicMock(content="Final answer."),  # generation
    ]
    import models.model as model_mod

    monkeypatch.setattr(model_mod, "get_llm", lambda: fake_llm)

    result = pipeline.answer_question(
        "and profit?",
        history=[("user", "How did Alpha perform?"), ("assistant", "Revenue grew 10%.")],
        settings=_fake_settings(),
    )
    assert result.answer == "Final answer."
    # retrieval used the condensed standalone question, not the raw follow-up
    search_query = fake_retriever.invoke.call_args[0][0]
    assert search_query == "What is the revenue and profit of Alpha corp?"


def test_condense_question_without_history_is_identity():
    assert pipeline.condense_question("plain q", []) == "plain q"


def test_condense_question_falls_back_on_error(monkeypatch):
    import models.model as model_mod

    boom = MagicMock()
    boom.invoke.side_effect = RuntimeError("network down")
    monkeypatch.setattr(model_mod, "get_llm", lambda: boom)
    # Should swallow the error and return the original question.
    out = pipeline.condense_question("original", [("user", "hi"), ("assistant", "hello")])
    assert out == "original"
