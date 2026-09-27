"""
graph.py — Task 3: LangGraph StateGraph with a TypedDict state and 3 nodes.

    classify_intent -----> (conditional edge) -----> retrieve_and_answer -> END
                                                 \\--> direct_answer      -> END

MOCK_LLM toggle (see module README):
  - unset or "1" (default, GRADED BASELINE): every generation step is a
    deterministic, rule-based / templated mock — no LLM call, no network call.
  - "0" (optional, ungraded extension): generation steps call a real LLM.

Retrieval itself (embedding + ChromaDB lookup) always runs for real in BOTH
modes — only the final *answer generation* branches on MOCK_LLM.
"""
import os
from typing import List

from langgraph.graph import StateGraph, END

from schema import GraphState
from vectorstore import build_default_store
from prompt_template import build_prompt

POLICY_KEYWORDS = [
    "delivery", "return", "refund", "membership", "tracking",
    "cancel", "gift card", "support hours",
]

_store = None  # lazy singleton so ChromaDB/embeddings load once per process


def _get_store():
    global _store
    if _store is None:
        _store = build_default_store()
    return _store


def is_mock_mode() -> bool:
    return os.environ.get("MOCK_LLM", "1") != "0"


# ---------------------------------------------------------------------------
# Node 1 — classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query_lower = state["query"].lower()

    if is_mock_mode():
        # Mock mode (graded baseline): keyword heuristic, no LLM call.
        intent = "policy_question" if any(k in query_lower for k in POLICY_KEYWORDS) else "general_question"
    else:
        # Optional MOCK_LLM=0 extension: call a real LLM to classify instead.
        intent = _llm_classify_intent(state["query"])

    return {**state, "intent": intent}


def _llm_classify_intent(query: str) -> str:
    """Optional real-LLM classification path (MOCK_LLM=0). Requires an LLM
    client to be configured (see README) — left as an integration point."""
    raise NotImplementedError(
        "Set MOCK_LLM=0 and wire up a real LLM client (e.g. Groq) here to use "
        "the optional extension. The graded baseline never calls this."
    )


# ---------------------------------------------------------------------------
# Node 2 — retrieve_and_answer (for policy_question)
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    # Retrieval always runs for real in both modes (no API key/network needed).
    store = _get_store()
    retrieved = store.query(state["query"], top_k=3)

    if is_mock_mode():
        top_chunk = retrieved[0]
        snippet = top_chunk["text"][:200]
        answer = f"Based on the retrieved context: {snippet}"
        sources = [r["id"] for r in retrieved]
        confidence = 1.0
    else:
        prompt = build_prompt(state["query"], retrieved)
        answer, sources, confidence = _llm_generate_grounded_answer(prompt, retrieved)

    return {
        **state,
        "retrieved_chunks": retrieved,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


def _llm_generate_grounded_answer(prompt: str, retrieved: List[dict]):
    """Optional real-LLM path (MOCK_LLM=0): call the LLM with the structured
    prompt, then validate/retry against the AskResponse schema (see README
    for the retry-on-validation-failure description)."""
    raise NotImplementedError(
        "Set MOCK_LLM=0 and wire up a real LLM client here. The graded "
        "baseline never calls this."
    )


# ---------------------------------------------------------------------------
# Node 3 — direct_answer (for general_question)
# ---------------------------------------------------------------------------
FALLBACK_ANSWER = "I can only answer questions about Zepto policies right now."


def direct_answer(state: GraphState) -> GraphState:
    if is_mock_mode():
        answer = FALLBACK_ANSWER
        confidence = 1.0
    else:
        answer = _llm_direct_answer(state["query"])
        confidence = 0.7

    return {
        **state,
        "retrieved_chunks": [],
        "answer": answer,
        "sources": [],
        "confidence": confidence,
    }


def _llm_direct_answer(query: str) -> str:
    """Optional real-LLM path (MOCK_LLM=0): answer directly, no retrieval."""
    raise NotImplementedError(
        "Set MOCK_LLM=0 and wire up a real LLM client here. The graded "
        "baseline never calls this."
    )


# ---------------------------------------------------------------------------
# Conditional routing (does NOT depend on MOCK_LLM)
# ---------------------------------------------------------------------------
def route_from_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_from_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(query: str) -> GraphState:
    graph = get_graph()
    initial_state: GraphState = {
        "query": query, "intent": None, "retrieved_chunks": None,
        "answer": None, "sources": None, "confidence": None,
    }
    return graph.invoke(initial_state)


if __name__ == "__main__":
    for q in ["What's your delivery fee?", "What's the meaning of life?"]:
        result = ask(q)
        print(f"Q: {q}\n  intent={result['intent']}  answer={result['answer']!r}\n"
              f"  sources={result['sources']}  confidence={result['confidence']}\n")
