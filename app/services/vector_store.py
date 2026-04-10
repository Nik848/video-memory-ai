"""
FAISS Vector Store
Handles indexing and searching embeddings.
Persists index to disk for durability.
"""
import faiss
import numpy as np
import json
import os
import logging
from threading import Lock

from app.config import FAISS_DIR, EMBEDDING_DIM

logger = logging.getLogger(__name__)

INDEX_PATH = os.path.join(FAISS_DIR, "index.faiss")
MAPPING_PATH = os.path.join(FAISS_DIR, "chunk_mapping.json")

# Thread-safe lock for index operations
_lock = Lock()

# Module-level state
_index = None
_chunk_mapping = []  # Maps FAISS index position -> chunk_id


def _load_or_create_index():
    """Load existing FAISS index or create a new one."""
    global _index, _chunk_mapping

    if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH):
        _index = faiss.read_index(INDEX_PATH)
        with open(MAPPING_PATH, "r") as f:
            _chunk_mapping = json.load(f)
        logger.info(f"Loaded FAISS index: {_index.ntotal} vectors")
    else:
        _index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product (cosine after L2 norm)
        _chunk_mapping = []
        logger.info("Created new FAISS index")


def _get_index():
    """Get or initialize the FAISS index."""
    global _index
    if _index is None:
        _load_or_create_index()
    return _index


def _save_index():
    """Persist FAISS index and mapping to disk."""
    with _lock:
        faiss.write_index(_index, INDEX_PATH)
        with open(MAPPING_PATH, "w") as f:
            json.dump(_chunk_mapping, f)
    logger.info(f"Saved FAISS index: {_index.ntotal} vectors")


def add_embeddings(embeddings: np.ndarray, chunk_ids: list) -> list:
    """
    Add embeddings to the FAISS index.

    Args:
        embeddings: numpy array of shape (N, 384)
        chunk_ids: list of chunk ID strings

    Returns:
        list of FAISS index positions assigned
    """
    global _chunk_mapping

    index = _get_index()

    if len(embeddings) == 0:
        return []

    # L2 normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    with _lock:
        start_pos = index.ntotal
        index.add(embeddings)

        positions = list(range(start_pos, start_pos + len(chunk_ids)))
        _chunk_mapping.extend(chunk_ids)

    _save_index()

    logger.info(f"Added {len(chunk_ids)} embeddings to FAISS "
                f"(total: {index.ntotal})")

    return positions


def search(query_embedding: np.ndarray, top_k: int = 10) -> list:
    """
    Search FAISS index for similar embeddings.

    Args:
        query_embedding: numpy array of shape (384,)
        top_k: number of results to return

    Returns:
        list of dicts: [{"chunk_id": ..., "score": ...}, ...]
    """
    index = _get_index()

    if index.ntotal == 0:
        return []

    # Reshape and normalize
    query = query_embedding.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)

    # Search
    scores, indices = index.search(query, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_chunk_mapping):
            continue
        results.append({
            "chunk_id": _chunk_mapping[idx],
            "score": float(score),
            "faiss_index": int(idx)
        })

    return results


def get_total_vectors() -> int:
    """Return total number of vectors in the index."""
    index = _get_index()
    return index.ntotal
