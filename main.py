"""
main.py — Task 6: FastAPI wrapper around the LangGraph support-assistant
pipeline.

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 7860

Then:
    curl -X POST http://localhost:7860/ask \\
         -H "Content-Type: application/json" \\
         -d '{"query": "How much does delivery cost?"}'
"""
from fastapi import FastAPI, HTTPException

from schema import AskRequest, AskResponse
from graph import ask as run_graph

app = FastAPI(
    title="Zepto Support Assistant",
    description="A small RAG service answering Zepto policy questions, "
                 "grounded in Zepto's own policy documents.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    try:
        result = run_graph(request.query)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc))

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"] or [],
        confidence=result["confidence"],
    )
