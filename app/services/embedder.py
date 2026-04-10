"""
Embedding Service
Generates 384-dim embeddings using all-MiniLM-L6-v2.
"""
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Lazy-loaded model
_model = None


def _get_model():
    """Lazy-load embedding model."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def generate_embeddings(texts: list) -> np.ndarray:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text strings

    Returns:
        numpy array of shape (len(texts), 384)
    """
    if not texts:
        return np.array([])

    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False)

    logger.info(f"Generated {len(texts)} embeddings, "
                f"shape: {embeddings.shape}")

    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Generate embedding for a single query string.

    Args:
        query: User query text

    Returns:
        numpy array of shape (384,)
    """
    model = _get_model()
    embedding = model.encode([query], show_progress_bar=False)
    return np.array(embedding[0], dtype=np.float32)