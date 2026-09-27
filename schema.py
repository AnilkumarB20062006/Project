"""
schema.py — Pydantic request/response models + the LangGraph state TypedDict.
"""
from typing import List, Literal, TypedDict, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The customer's question.")


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list, description="Chunk/document IDs used.")
    confidence: float = Field(..., ge=0.0, le=1.0)


class GraphState(TypedDict):
    """State threaded through every LangGraph node."""
    query: str
    intent: Optional[Literal["policy_question", "general_question"]]
    retrieved_chunks: Optional[List[dict]]   # [{"id": ..., "text": ..., "score": ...}, ...]
    answer: Optional[str]
    sources: Optional[List[str]]
    confidence: Optional[float]
