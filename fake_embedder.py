"""
fake_embedder.py — a deterministic, dependency-free stand-in for
sentence-transformers, used ONLY by the test suite.

It is NOT semantically as good as a real embedding model, but it is
similarity-sensitive to shared keywords (bag-of-words + hashing, L2-normalized),
which is enough to verify that ingestion/retrieval/graph wiring behave
correctly end-to-end without downloading ~90MB of model weights or requiring
network access — useful in this sandbox and in any offline CI runner.
"""
import hashlib
import re
from typing import List

DIM = 256


def _hash_index(token: str) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM


def _embed_one(text: str) -> List[float]:
    vec = [0.0] * DIM
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for tok in tokens:
        vec[_hash_index(tok)] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def fake_embed_fn(texts: List[str]) -> List[List[float]]:
    return [_embed_one(t) for t in texts]
