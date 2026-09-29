"""Retrieval + generation pipeline built on LangChain's ParentDocumentRetriever.

Small "child" chunks are embedded for precise vector search; the large
"parent" chunk containing each match is returned to the LLM for answering,
which maximizes exactness of the retrieved context.

The chain is assembled with LangChain Expression Language (LCEL). Heavy
third-party imports live inside functions so unit tests stay fast and never
require API keys or model downloads.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from config import Settings, get_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "context documents. If the answer is not present in the context, say so "
    "clearly instead of guessing. Cite the source document titles when relevant."
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {input}"),
    ]
)


def format_docs(docs: list[Document]) -> str:
    """Render retrieved parent documents into prompt context."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("Header 1") or doc.metadata.get("source") or f"Document {i}"
        blocks.append(f"[{i}] ({title})\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks) if blocks else "(no documents retrieved)"


def build_parent_retriever(settings: Settings | None = None):
    """Create a ParentDocumentRetriever backed by Chroma + a local docstore."""
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.storage import LocalFileStore
    from langchain.storage._lc_store import create_kv_docstore
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from models.model import get_embeddings

    settings = settings or get_settings()

    docstore = LocalFileStore(str(settings.local_store_path))
    vector_store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.vector_store_path),
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.parent_chunk_size)

    return ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=create_kv_docstore(docstore),
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )


@lru_cache(maxsize=1)
def get_retriever():
    """Return a cached retriever (avoids rebuilding stores per query)."""
    settings = get_settings()
    retriever = build_parent_retriever(settings)
    retriever.search_kwargs = {"k": settings.retrieval_k}
    logger.info("Parent-document retriever initialized.")
    return retriever


def answer_question(question: str) -> dict:
    """Answer a single question via the RAG chain.

    Returns a dict with keys ``answer`` (str) and ``context`` (list of Documents).
    The context is retrieved exactly once and reused for grounding, so the
    displayed sources always match what the LLM actually saw.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")
    question = question.strip()

    docs = get_retriever().invoke(question)
    context = format_docs(docs)
    # to_messages() -> [system, human]; ChatGoogleGenerativeAI accepts both a
    # ChatPromptValue and a raw message list, so pass the messages explicitly.
    messages = PROMPT.invoke({"context": context, "input": question}).to_messages()

    from models.model import get_llm

    response = get_llm().invoke(messages)
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return {"answer": answer, "context": docs}
