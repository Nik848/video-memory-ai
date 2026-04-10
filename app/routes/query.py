"""
Query Routes
Semantic search and LLM-powered question answering.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import time
from app.services.query_engine import query
from app.services.auth import require_api_key, get_user_id
from app.services.metrics import log_query_metric, aggregate_metrics

router = APIRouter(prefix="/query", tags=["Query"])


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 10
    use_reranker: Optional[bool] = True
    use_llm: Optional[bool] = True


class QueryFeedbackRequest(BaseModel):
    question: str
    query_intent: Optional[str] = "general_search"
    precision_at_k: Optional[float] = None
    recall_at_k: Optional[float] = None
    answer_relevance: Optional[float] = None
    feedback: Optional[str] = None


@router.post("/", dependencies=[Depends(require_api_key)])
def query_videos(
    request: QueryRequest,
    page: int = 1,
    page_size: int = 5,
    user_id: str = Depends(get_user_id),
):
    """
    Ask a question across all ingested videos.
    Returns LLM-generated answer with source attribution.
    """
    try:
        start = time.perf_counter()
        result = query(
            user_query=request.question,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
            use_llm=request.use_llm,
            user_id=user_id,
            page=page,
            page_size=page_size,
        )
        log_query_metric(
            user_id=user_id,
            query_text=request.question,
            query_intent=result.get("query_intent", "general_search"),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", dependencies=[Depends(require_api_key)])
def search_videos(
    q: str,
    top_k: int = 10,
    page: int = 1,
    page_size: int = 10,
    user_id: str = Depends(get_user_id),
):
    """
    Raw semantic search (no LLM) — returns matching chunks.
    """
    try:
        result = query(
            user_query=q,
            top_k=top_k,
            use_reranker=False,
            use_llm=False,
            user_id=user_id,
            page=page,
            page_size=page_size,
        )
        return {
            "query": q,
            "query_intent": result.get("query_intent"),
            "normalized_query": result.get("normalized_query"),
            "results": result["sources"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", dependencies=[Depends(require_api_key)])
def submit_feedback(request: QueryFeedbackRequest, user_id: str = Depends(get_user_id)):
    log_query_metric(
        user_id=user_id,
        query_text=request.question,
        query_intent=request.query_intent or "general_search",
        latency_ms=0.0,
        precision_at_k=request.precision_at_k,
        recall_at_k=request.recall_at_k,
        answer_relevance=request.answer_relevance,
        feedback=request.feedback,
    )
    return {"status": "ok"}


@router.get("/metrics", dependencies=[Depends(require_api_key)])
def get_query_metrics(user_id: str = Depends(get_user_id)):
    return aggregate_metrics(user_id)
