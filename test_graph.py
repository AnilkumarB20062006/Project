"""
Offline end-to-end test of the LangGraph graph (mock mode) and the FastAPI
/ask endpoint, using the fake embedder so no model download / network
access is required.

Run:
    pytest support_assistant/tests/test_graph.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import vectorstore
from fake_embedder import fake_embed_fn


@pytest.fixture(autouse=True)
def isolated_mock_store(tmp_path, monkeypatch):
    """
    Force the whole module to use an isolated ChromaDB directory + the fake
    embedder for this test session, and make sure MOCK_LLM defaults to the
    graded baseline (mock mode).
    """
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setattr(vectorstore, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(vectorstore, "default_embed_fn", fake_embed_fn)

    # graph.py caches a singleton store/graph at module level -> reset both
    # so this test doesn't reuse a real (or another test's) store.
    import graph
    graph._store = None
    graph._compiled_graph = None
    yield
    graph._store = None
    graph._compiled_graph = None


def test_classify_intent_policy_question():
    from graph import classify_intent
    state = {"query": "What is your delivery fee?", "intent": None,
              "retrieved_chunks": None, "answer": None, "sources": None, "confidence": None}
    result = classify_intent(state)
    assert result["intent"] == "policy_question"


def test_classify_intent_general_question():
    from graph import classify_intent
    state = {"query": "What is the capital of France?", "intent": None,
              "retrieved_chunks": None, "answer": None, "sources": None, "confidence": None}
    result = classify_intent(state)
    assert result["intent"] == "general_question"


def test_graph_routes_policy_question_to_retrieval():
    from graph import ask
    result = ask("How much does standard delivery cost?")
    assert result["intent"] == "policy_question"
    assert result["answer"].startswith("Based on the retrieved context:")
    assert len(result["sources"]) == 3  # top-3 retrieved chunk ids
    assert result["confidence"] == 1.0


def test_graph_routes_general_question_to_direct_answer():
    from graph import ask
    result = ask("Tell me a joke about the weather.")
    assert result["intent"] == "general_question"
    assert result["answer"] == "I can only answer questions about Zepto policies right now."
    assert result["sources"] == []
    assert result["confidence"] == 1.0


def test_fastapi_ask_endpoint_policy_question():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.post("/ask", json={"query": "What are your customer support hours?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body and "sources" in body and "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert isinstance(body["sources"], list)


def test_fastapi_ask_endpoint_general_question():
    from fastapi.testclient import TestClient
    from main import app
    client = TestClient(app)
    resp = client.post("/ask", json={"query": "Who won the last World Cup?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == []
