"""
Query Engine
Semantic search → Reranking → LLM Answer Generation → Source Attribution
"""
import requests
import logging
from typing import Optional

from app.services.embedder import embed_query
from app.services.vector_store import search as faiss_search
from app.models.database import SessionLocal
from app.models.schemas import Chunk, Video
from app.config import (
    TOP_K_RESULTS, TOP_N_RERANKED,
    OLLAMA_BASE_URL, OLLAMA_MODEL
)

logger = logging.getLogger(__name__)

# Optional: cross-encoder reranker
_reranker = None


def _get_reranker():
    """Lazy-load cross-encoder reranker."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            from app.config import RERANKER_MODEL
            logger.info(f"Loading reranker: {RERANKER_MODEL}")
            _reranker = CrossEncoder(RERANKER_MODEL)
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}. "
                           "Falling back to FAISS scores only.")
            _reranker = False  # marker that we tried and failed
    return _reranker if _reranker is not False else None


def query(user_query: str, top_k: int = None,
          use_reranker: bool = True,
          use_llm: bool = True) -> dict:
    """
    Full query pipeline: embed → search → rerank → LLM.

    Args:
        user_query: Natural language question
        top_k: Number of results (default from config)
        use_reranker: Whether to apply reranking
        use_llm: Whether to generate LLM answer

    Returns:
        dict with: answer, sources, raw_results
    """
    if top_k is None:
        top_k = TOP_K_RESULTS

    # Step 1: Embed the query
    query_embedding = embed_query(user_query)

    # Step 2: FAISS search
    faiss_results = faiss_search(query_embedding, top_k=top_k)

    if not faiss_results:
        return {
            "answer": "No relevant content found. Try ingesting some videos first.",
            "sources": [],
            "raw_results": []
        }

    # Step 3: Fetch chunk details from DB
    db = SessionLocal()
    try:
        enriched_results = []
        for result in faiss_results:
            chunk = db.query(Chunk).filter(
                Chunk.id == result["chunk_id"]
            ).first()

            if chunk:
                video = db.query(Video).filter(
                    Video.id == chunk.video_id
                ).first()

                enriched_results.append({
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "start": chunk.start_time,
                    "end": chunk.end_time,
                    "video_id": chunk.video_id,
                    "video_title": video.title if video else "Unknown",
                    "video_url": video.url if video else "",
                    "chunk_type": chunk.chunk_type,
                    "score": result["score"]
                })
    finally:
        db.close()

    if not enriched_results:
        return {
            "answer": "Could not find matching chunks in database.",
            "sources": [],
            "raw_results": faiss_results
        }

    # Step 4: Reranking (optional)
    if use_reranker:
        reranker = _get_reranker()
        if reranker:
            enriched_results = _rerank(
                reranker, user_query, enriched_results
            )

    # Trim to top N
    top_results = enriched_results[:TOP_N_RERANKED]

    # Step 5: LLM answer generation (optional)
    answer = None
    if use_llm:
        answer = _generate_answer(user_query, top_results)

    # Step 6: Build source attribution
    sources = []
    for r in top_results:
        sources.append({
            "video_id": r["video_id"],
            "video_title": r["video_title"],
            "video_url": r["video_url"],
            "start": r["start"],
            "end": r["end"],
            "text": r["text"],
            "score": r["score"],
            "type": r["chunk_type"]
        })

    return {
        "answer": answer or "LLM unavailable. See sources below.",
        "sources": sources,
        "query": user_query,
        "total_results": len(faiss_results)
    }


def _rerank(reranker, query_text: str, results: list) -> list:
    """Apply cross-encoder reranking to results."""
    try:
        pairs = [(query_text, r["text"]) for r in results]
        scores = reranker.predict(pairs)

        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)

        # Sort by rerank score descending
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        logger.info("Reranking applied successfully")
    except Exception as e:
        logger.warning(f"Reranking failed: {e}")

    return results


def _generate_answer(query_text: str, sources: list) -> Optional[str]:
    """
    Use Ollama (Mistral) to generate an answer grounded in sources.
    """
    context = "\n\n".join([
        f"[Source: {s['video_title']} | {s['start']:.1f}s-{s['end']:.1f}s]\n{s['text']}"
        for s in sources
    ])

    prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided video content.
Do NOT make up information. If the context doesn't contain enough information, say so.

=== VIDEO CONTENT ===
{context}

=== QUESTION ===
{query_text}

=== INSTRUCTIONS ===
- Answer the question using ONLY the video content above
- Be concise and specific
- Reference which video the information comes from
- If unsure, say "Based on the available content, I cannot fully answer this"

Answer:"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 512
                }
            },
            timeout=30
        )

        if response.status_code == 200:
            answer = response.json().get("response", "").strip()
            logger.info("LLM generated answer successfully")
            return answer
        else:
            logger.warning(f"Ollama returned status {response.status_code}")
            return None

    except requests.ConnectionError:
        logger.warning("Cannot connect to Ollama. Is it running?")
        return None
    except Exception as e:
        logger.warning(f"LLM generation failed: {e}")
        return None
