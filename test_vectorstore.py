"""
Offline tests for vectorstore.py, using the deterministic fake embedder
(tests/fake_embedder.py) instead of downloading real model weights.

Run:
    pytest support_assistant/tests/test_vectorstore.py -v
"""
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vectorstore import VectorStore, load_and_chunk_documents, DOCS_DIR
from fake_embedder import fake_embed_fn

@pytest.fixture
def store(tmp_path):
    # A fresh, uniquely-named directory per test avoids ChromaDB's
    # PersistentClient re-using a stale in-process connection to a path
    # that was just deleted and recreated by a previous test.
    vs = VectorStore(persist_dir=tmp_path / "chroma", collection_name="test_zepto", embed_fn=fake_embed_fn)
    chunks = load_and_chunk_documents(DOCS_DIR)
    vs.ingest(chunks)
    return vs


def test_load_and_chunk_documents_returns_all_8():
    chunks = load_and_chunk_documents(DOCS_DIR)
    ids = {c["id"] for c in chunks}
    assert ids == {f"doc_{i:02d}" for i in range(1, 9)}
    for c in chunks:
        assert len(c["text"]) > 0


def test_ingest_and_count(store):
    assert store.count() == 8


@pytest.mark.parametrize("query,expected_doc", [
    ("how much is the delivery fee", "doc_01"),
    ("what is the return window for unopened packaged items", "doc_02"),
    ("what are the membership tiers and pricing", "doc_03"),
    ("how do I track my rider on a map", "doc_04"),
    ("can I cancel my order after it is packed", "doc_05"),
    ("my order arrived with missing items", "doc_06"),
    ("what denominations do gift cards come in", "doc_07"),
    ("what are your customer support hours", "doc_08"),
])
def test_retrieval_returns_correct_source_document(store, query, expected_doc):
    results = store.query(query, top_k=3)
    top_ids = [r["id"] for r in results]
    assert expected_doc in top_ids, f"Expected {expected_doc} in top-3 for {query!r}, got {top_ids}"
