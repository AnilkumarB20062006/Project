"""
vectorstore.py — Task 1: chunk the 8 policy docs, embed them locally with
sentence-transformers/all-MiniLM-L6-v2, and store/query them in ChromaDB.

No API key and no paid service: both sentence-transformers and ChromaDB run
entirely on-machine.

The embedding function is dependency-injected (`embed_fn`) so this module
can be unit-tested with a deterministic stand-in embedder, without needing
to download model weights — see tests/test_vectorstore.py.
"""
from pathlib import Path
from typing import Callable, List, Optional

import chromadb

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "docs"
CHROMA_DIR = HERE / "chroma_store"
COLLECTION_NAME = "zepto_policies"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None  # lazy-loaded singleton so the model is only ever loaded once


def default_embed_fn(texts: List[str]) -> List[List[float]]:
    """Real embedder: sentence-transformers/all-MiniLM-L6-v2, run locally."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model.encode(list(texts), normalize_embeddings=True).tolist()


def load_and_chunk_documents(docs_dir: Path = DOCS_DIR, max_chars: int = 700) -> List[dict]:
    """
    Per-document chunking: each of the 8 policy docs (all comfortably under
    `max_chars`, the longest is ~580 chars) becomes exactly one chunk, which
    is the "simple per-document chunk" scheme the task explicitly allows
    given how short these documents are. A fixed-size fallback split still
    kicks in for any document that exceeds `max_chars`, so the function
    degrades sensibly on longer input too.
    Returns: [{"id": "doc_01", "text": "..."}, ...]
    """
    chunks = []
    for path in sorted(docs_dir.glob("doc_*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        doc_id = path.stem
        if len(text) <= max_chars:
            chunks.append({"id": doc_id, "text": text})
        else:
            for i, start in enumerate(range(0, len(text), max_chars)):
                chunks.append({
                    "id": f"{doc_id}_chunk{i}",
                    "text": text[start:start + max_chars],
                })
    return chunks


class VectorStore:
    def __init__(
        self,
        persist_dir: Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embed_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
    ):
        self.embed_fn = embed_fn or default_embed_fn
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(collection_name)

    def ingest(self, chunks: List[dict]):
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        embeddings = self.embed_fn(texts)
        self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings)

    def query(self, text: str, top_k: int = 3) -> List[dict]:
        """Cosine-similarity top-k retrieval. Returns [{"id","text","score"}, ...]."""
        query_embedding = self.embed_fn([text])[0]
        res = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        out = []
        for doc_id, doc_text, distance in zip(res["ids"][0], res["documents"][0], res["distances"][0]):
            # Chroma's default space is cosine distance for this client config;
            # similarity = 1 - distance.
            out.append({"id": doc_id, "text": doc_text, "score": 1 - distance})
        return out

    def count(self) -> int:
        return self.collection.count()


def build_default_store(force_rebuild: bool = False) -> VectorStore:
    store = VectorStore()
    if force_rebuild or store.count() == 0:
        chunks = load_and_chunk_documents()
        store.ingest(chunks)
    return store


if __name__ == "__main__":
    store = build_default_store(force_rebuild=True)
    print(f"Ingested {store.count()} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
    sample = store.query("How much is the delivery fee?", top_k=3)
    for r in sample:
        print(f"  {r['id']} (score={r['score']:.3f}): {r['text'][:80]}...")
