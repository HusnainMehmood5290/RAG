"""Document loading and chunking helpers.

PDFs are converted to Markdown (pymupdf4llm), split on Markdown headers, and
indexed into the ParentDocumentRetriever's stores. All failures raise; callers
decide how to handle them.

Re-indexing an updated file is safe: all vectors and docstore entries from the
file's previous fingerprint are deleted *before* the new ones are added, so
stores never contain stale or duplicated content for the same source name.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from config import Settings, get_settings
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
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep header text in page_content for context quality
    )
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


@lru_cache(maxsize=1)
def get_ingest_retriever(settings: Settings | None = None):
    """Build the retriever once per process.

    Creating Chroma + the embedding model per file is expensive and can leave
    stale SQLite handles behind; sharing one instance across a batch avoids
    both problems.
    """
    return build_parent_retriever(settings or get_settings())


def _purge_previous_index(retriever, source_name: str, current_hash: str) -> int:
    """Delete vectors/docstore entries belonging to older ingests of this file.

    Matches documents whose ``source`` equals this filename but whose
    ``file_hash`` differs from the one about to be indexed (i.e., the file was
    updated since the last run). Returns the number of parent docs removed.

    SECURITY: if the query or the delete fails, this raises instead of
    returning silently -- otherwise stale content from a previous version of
    the document would stay retrievable alongside (or instead of) the new
    content, leaking superseded data into answers. Callers abort ingestion of
    the file and leave it in ``raw_data`` for retry.
    """
    try:
        results = retriever.vectorstore.get(where={"source": source_name})
    except Exception as exc:
        raise RuntimeError(
            f"Cannot verify prior index state for {source_name!r}; "
            "refusing to re-index over potentially stale vectors."
        ) from exc
    ids = results.get("ids") or []
    metadatas = results.get("metadatas") or []
    # Single-pass scan; skip the network round-trip entirely when nothing is stale.
    stale_ids = [
        cid
        for cid, meta in zip(ids, metadatas, strict=False)
        if (meta or {}).get("file_hash", "") not in ("", current_hash)
    ]
    if not stale_ids:
        return 0
    # Child vector ids look like "<parent_id>:<child_idx>" (or ":<n>" suffix);
    # map them back to unique parent ids before touching the docstore.
    parent_ids = {cid.rsplit(":", 1)[0] for cid in stale_ids}
    try:
        retriever.vectorstore.delete(ids=stale_ids)
        retriever.docstore.delete(list(parent_ids))
    except Exception as exc:
        raise RuntimeError(
            f"Partial purge failure for {source_name!r}: stale vectors may "
            "still be retrievable. Re-run ingestion to retry."
        ) from exc
    logger.info(
        "Purged %d stale child vectors (%d parents) from previous index of %s.",
        len(stale_ids),
        len(parent_ids),
        source_name,
    )
    return len(parent_ids)


def load_file(file_path: str | Path, settings: Settings | None = None) -> int:
    """Process one PDF: convert -> split -> index into the parent retriever.

    Returns the number of parent documents added. Raises on any failure so
    the caller can log/skip without corrupting state.
    """
    path = Path(file_path)
    settings = settings or get_settings()

    logger.info("Converting PDF to Markdown: %s", path.name)
    md_content = convert_pdf_to_markdown(path)

    documents = markdown_to_documents(md_content, source=path.name)
    logger.info("Split %s into %d header-scoped documents.", path.name, len(documents))

    retriever = get_ingest_retriever(settings)
    fingerprint = file_fingerprint(path)
    for doc in documents:
        doc.metadata["file_hash"] = fingerprint
    ids = [f"{fingerprint}:{i}" for i in range(len(documents))]

    # Delete-before-insert keeps re-indexing an updated file atomic-ish and
    # prevents duplicate/stale chunks from lingering in either store.
    _purge_previous_index(retriever, path.name, fingerprint)

    retriever.add_documents(documents, ids=ids)
    logger.info(
        "Indexed %d documents from %s into collection '%s'.",
        len(documents),
        path.name,
        settings.collection_name,
    )
    return len(documents)
