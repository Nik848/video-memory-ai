"""
Query Routes
Semantic search and LLM-powered question answering.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.query_engine import query

router = APIRouter(prefix="/query", tags=["Query"])


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 10
    use_reranker: Optional[bool] = True
    use_llm: Optional[bool] = True


@router.post("/")
def query_videos(request: QueryRequest):
    """
    Ask a question across all ingested videos.
    Returns LLM-generated answer with source attribution.
    """
    try:
        result = query(
            user_query=request.question,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
            use_llm=request.use_llm
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
def search_videos(q: str, top_k: int = 10):
    """
    Raw semantic search (no LLM) — returns matching chunks.
    """
    try:
        result = query(
            user_query=q,
            top_k=top_k,
            use_reranker=False,
            use_llm=False
        )
        return {
            "query": q,
            "results": result["sources"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
