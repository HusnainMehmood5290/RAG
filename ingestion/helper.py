"""Document loading and chunking helpers.

PDFs are converted to Markdown (pymupdf4llm), split on Markdown headers, and
indexed into the ParentDocumentRetriever's stores. All failures raise; callers
decide how to handle them.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from config import get_settings
from rag.pipeline import build_parent_retriever

logger = logging.getLogger(__name__)

# Map each Markdown header level to its own metadata key (levels 4-6 were
# previously mislabeled "Header 3").
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),
]


def file_fingerprint(path: Path) -> str:
    """Stable sha256 hash of file contents (for idempotent ingestion)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_to_documents(md_content: str, source: str) -> list[Document]:
    """Split Markdown text into header-scoped LangChain documents."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)
    docs = splitter.split_text(md_content)
    if not docs:
        raise ValueError(f"Markdown splitting produced no documents for {source!r}.")
    for doc in docs:
        doc.metadata.setdefault("source", source)
    return docs


def convert_pdf_to_markdown(pdf_path: Path) -> str:
    """Convert a PDF file to Markdown text."""
    md_content = pymupdf4llm.to_markdown(str(pdf_path))
    if not md_content.strip():
        raise ValueError(f"PDF conversion produced empty Markdown for {pdf_path}.")
    return md_content


def load_file(file_path: str | Path) -> int:
    """Process one PDF: convert -> split -> index into the parent retriever.

    Returns the number of parent documents added. Raises on any failure so
    the caller can log/skip without corrupting state.
    """
    path = Path(file_path)
    settings = get_settings()

    logger.info("Converting PDF to Markdown: %s", path.name)
    md_content = convert_pdf_to_markdown(path)

    documents = markdown_to_documents(md_content, source=path.name)
    logger.info("Split %s into %d header-scoped documents.", path.name, len(documents))

    retriever = build_parent_retriever(settings)
    fingerprint = file_fingerprint(path)
    for doc in documents:
        doc.metadata["file_hash"] = fingerprint
    ids = [f"{fingerprint}:{i}" for i in range(len(documents))]

    retriever.add_documents(documents, ids=ids)
    logger.info(
        "Indexed %d documents from %s into collection '%s'.",
        len(documents),
        path.name,
        settings.collection_name,
    )
    return len(documents)
