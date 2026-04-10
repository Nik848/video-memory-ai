"""
Query evaluation metric logging and aggregation.
"""
import uuid
from datetime import datetime, timezone

from app.models.database import SessionLocal
from app.models.schemas import QueryMetric


def log_query_metric(
    *,
    user_id: str,
    query_text: str,
    query_intent: str,
    latency_ms: float,
    precision_at_k: float = None,
    recall_at_k: float = None,
    answer_relevance: float = None,
    feedback: str = None,
):
    db = SessionLocal()
    try:
        row = QueryMetric(
            id=str(uuid.uuid4()),
            user_id=user_id,
            query_text=query_text,
            query_intent=query_intent,
            latency_ms=latency_ms,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            answer_relevance=answer_relevance,
            feedback=feedback,
            created_at=datetime.now(timezone.utc),
        )
        db.add(row)
        db.commit()
    finally:
        db.close()


def aggregate_metrics(user_id: str) -> dict:
    db = SessionLocal()
    try:
        rows = db.query(QueryMetric).filter(QueryMetric.user_id == user_id).all()
        if not rows:
            return {
                "total_queries": 0,
                "avg_latency_ms": None,
                "avg_precision_at_k": None,
                "avg_recall_at_k": None,
                "avg_answer_relevance": None,
                "positive_feedback_rate": None,
            }

        def _avg(values):
            nums = [v for v in values if v is not None]
            return float(sum(nums) / len(nums)) if nums else None

        positive = [r for r in rows if (r.feedback or "").lower() == "positive"]
        return {
            "total_queries": len(rows),
            "avg_latency_ms": _avg([r.latency_ms for r in rows]),
            "avg_precision_at_k": _avg([r.precision_at_k for r in rows]),
            "avg_recall_at_k": _avg([r.recall_at_k for r in rows]),
            "avg_answer_relevance": _avg([r.answer_relevance for r in rows]),
            "positive_feedback_rate": float(len(positive) / len(rows)),
        }
    finally:
        db.close()
