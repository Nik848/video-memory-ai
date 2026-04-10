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

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            from app.config import RERANKER_MODEL
            logger.info(f"Loading reranker: {RERANKER_MODEL}")
            _reranker = CrossEncoder(RERANKER_MODEL)
        except Exception as e:
            logger.warning(f"Reranker failed: {e}")
            _reranker = False
    return _reranker if _reranker is not False else None


def query(user_query: str, top_k: int = None,
          use_reranker: bool = True,
          use_llm: bool = True) -> dict:

    if top_k is None:
        top_k = TOP_K_RESULTS

    # 1. Embed
    query_embedding = embed_query(user_query)

    # 2. Search
    faiss_results = faiss_search(query_embedding, top_k=top_k)

    if not faiss_results:
        return {
            "answer": "No relevant content found.",
            "sources": [],
            "raw_results": []
        }

    # 3. Fetch from DB
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
            "answer": "No usable chunks found.",
            "sources": [],
            "raw_results": faiss_results
        }

    # 4. Rerank
    if use_reranker:
        reranker = _get_reranker()
        if reranker:
            enriched_results = _rerank(reranker, user_query, enriched_results)

    # limit results (IMPORTANT)
    top_results = enriched_results[:TOP_N_RERANKED]

    # 5. LLM
    answer = None
    if use_llm:
        answer = _generate_answer(user_query, top_results)

    # 6. Sources
    sources = [{
        "video_id": r["video_id"],
        "video_title": r["video_title"],
        "video_url": r["video_url"],
        "start": r["start"],
        "end": r["end"],
        "text": r["text"],
        "score": r["score"],
        "type": r["chunk_type"]
    } for r in top_results]

    return {
        "answer": answer if answer else "LLM failed (check logs).",
        "sources": sources,
        "query": user_query,
        "total_results": len(faiss_results)
    }


def _rerank(reranker, query_text: str, results: list) -> list:
    try:
        pairs = [(query_text, r["text"]) for r in results]
        scores = reranker.predict(pairs)

        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)

        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    except Exception as e:
        logger.warning(f"Rerank failed: {e}")

    return results


def _generate_answer(query_text: str, sources: list) -> str:
    """
    Proper Ollama call (chat API, no silent failure)
    """

    # 🔴 limit context (critical)
    sources = sources[:3]

    context = "\n\n".join([
        f"[{s['video_title']} | {s['start']:.1f}-{s['end']:.1f}s]\n{s['text']}"
        for s in sources
    ])

    prompt = f"""
Answer ONLY using the provided video content.

CONTENT:
{context}

QUESTION:
{query_text}

RULES:
- Do not hallucinate
- If unsure, say you don’t know
- Be concise

ANSWER:
"""

    try:
        url = f"{OLLAMA_BASE_URL}/api/chat"

        print("➡️ Calling Ollama:", url)
        print("➡️ Model:", OLLAMA_MODEL)

        response = requests.post(
            url,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            },
            timeout=360
        )

        print("STATUS:", response.status_code)
        print("RAW:", response.text)

        if response.status_code != 200:
            raise Exception(response.text)

        data = response.json()

        return data["message"]["content"].strip()

    except Exception as e:
        print("🔥 LLM FAILURE:", str(e))
        raise e  # DO NOT SILENCE