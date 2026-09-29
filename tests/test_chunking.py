"""Unit tests for Markdown header splitting (no external services needed)."""

import pytest

from ingestion.helper import HEADERS_TO_SPLIT_ON, markdown_to_documents

MD = """# Title
Intro text.
## Section A
Details A.
### Sub A.1
Deep detail.
#### Subsub A.1.1
Even deeper.
## Section B
Details B.
"""


def test_header_metadata_levels_are_distinct():
    docs = markdown_to_documents(MD, source="doc.md")
    keys = set()
    for d in docs:
        keys.update(d.metadata.keys())
    # levels 4-6 must NOT be collapsed into "Header 3"
    assert "Header 4" in keys
    assert "Header 3" in keys
    assert "Header 2" in keys


def test_source_metadata_attached():
    docs = markdown_to_documents(MD, source="doc.md")
    assert all(d.metadata["source"] == "doc.md" for d in docs)


def test_empty_markdown_raises():
    with pytest.raises(ValueError, match="no documents"):
        markdown_to_documents("", source="empty.md")


def test_headers_map_defined():
    assert len(HEADERS_TO_SPLIT_ON) == 6
    labels = [label for _, label in HEADERS_TO_SPLIT_ON]
    assert labels == [f"Header {i}" for i in range(1, 7)]
