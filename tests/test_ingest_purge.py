"""Delete-before-insert purge of stale vectors when a file is re-indexed."""

from unittest.mock import MagicMock

from ingestion.helper import _purge_previous_index


def _fake_retriever(ids, metadatas):
    r = MagicMock()
    r.vectorstore.get.return_value = {"ids": ids, "metadatas": metadatas}
    return r


def test_purge_removes_only_stale_hashes():
    old, new = "a" * 32, "b" * 32
    retriever = _fake_retriever(
        ids=[f"{old}:0:1", f"{old}:0:2", f"{new}:0:1"],
        metadatas=[
            {"source": "x.pdf", "file_hash": old},
            {"source": "x.pdf", "file_hash": old},
            {"source": "x.pdf", "file_hash": new},
        ],
    )
    removed = _purge_previous_index(retriever, "x.pdf", new)
    assert removed == 1  # one unique parent id ({old}:0)
    deleted_ids = retriever.vectorstore.delete.call_args.kwargs["ids"]
    assert set(deleted_ids) == {f"{old}:0:1", f"{old}:0:2"}
    deleted_parents = retriever.docstore.delete.call_args[0][0]
    assert set(deleted_parents) == {f"{old}:0"}


def test_purge_noop_when_all_current():
    cur = "c" * 32
    retriever = _fake_retriever([f"{cur}:0:1"], [{"file_hash": cur}])
    assert _purge_previous_index(retriever, "x.pdf", cur) == 0
    retriever.vectorstore.delete.assert_not_called()
    retriever.docstore.delete.assert_not_called()


def test_purge_aborts_on_vectorstore_errors():
    retriever = MagicMock()
    retriever.vectorstore.get.side_effect = RuntimeError("chroma locked")
    # SECURITY: must raise. Silently skipping the purge would leave stale
    # vectors from a previous version of the document retrievable forever.
    with pytest.raises(RuntimeError, match="refusing to re-index"):
        _purge_previous_index(retriever, "x.pdf", "h")


def test_purge_aborts_on_partial_delete_failure():
    old, new = "a" * 32, "b" * 32
    retriever = _fake_retriever(
        ids=[f"{old}:0:1"], metadatas=[{"source": "x.pdf", "file_hash": old}]
    )
    retriever.docstore.delete.side_effect = OSError("disk full")
    with pytest.raises(RuntimeError, match="Partial purge failure"):
        _purge_previous_index(retriever, "x.pdf", new)
