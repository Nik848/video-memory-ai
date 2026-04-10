"""
Clustering engine for grouping related chunks by embedding similarity.
Uses HDBSCAN when available, falls back to KMeans.
"""
import logging
import math
from collections import Counter
from collections import defaultdict
import re

from app.models.schemas import Chunk
from app.services.embedder import generate_embeddings

logger = logging.getLogger(__name__)

KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10


def _cluster_embeddings(
    embeddings,
    min_cluster_size: int = 2,
    kmeans_clusters: int = None,
):
    """Cluster embeddings with HDBSCAN-first strategy."""
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, min_cluster_size),
            metric="euclidean",
            prediction_data=False,
        )
        labels = clusterer.fit_predict(embeddings)
        logger.info("Clustering done with HDBSCAN")
        return labels.tolist()
    except Exception as hdbscan_error:
        logger.warning(f"HDBSCAN unavailable/failed: {hdbscan_error}")

    try:
        from sklearn.cluster import KMeans

        sample_count = len(embeddings)
        if kmeans_clusters and kmeans_clusters > 0:
            n_clusters = min(sample_count, kmeans_clusters)
        else:
            # Sqrt(N) is a lightweight heuristic for unknown cluster counts.
            n_clusters = max(1, min(sample_count, int(math.sqrt(sample_count))))
        labels = KMeans(
            n_clusters=n_clusters,
            random_state=KMEANS_RANDOM_STATE,
            n_init=KMEANS_N_INIT,
        ).fit_predict(embeddings)
        logger.info("Clustering done with KMeans fallback")
        return labels.tolist()
    except Exception as kmeans_error:
        logger.warning(f"KMeans unavailable/failed: {kmeans_error}")

    logger.warning("All clustering backends unavailable. Assigning default cluster 0.")
    return [0] * len(embeddings)


def assign_clusters(
    db,
    min_cluster_size: int = 2,
    kmeans_clusters: int = None,
    user_id: str = "public",
) -> dict:
    """
    Assign cluster IDs to all chunks in the database.
    Returns clustering summary.
    """
    chunks = db.query(Chunk).filter(
        Chunk.user_id == user_id
    ).order_by(Chunk.created_at.asc()).all()
    if not chunks:
        return {
            "total_chunks": 0,
            "total_clusters": 0,
            "noise_chunks": 0,
            "distribution": {},
        }

    texts = [c.text or "" for c in chunks]
    embeddings = generate_embeddings(texts)
    labels = _cluster_embeddings(
        embeddings,
        min_cluster_size=min_cluster_size,
        kmeans_clusters=kmeans_clusters,
    )

    for chunk, label in zip(chunks, labels):
        chunk.cluster_id = int(label)

    label_map = _derive_cluster_labels(chunks)
    for chunk in chunks:
        if chunk.cluster_id is None:
            chunk.cluster_label = None
            continue
        chunk.cluster_label = label_map.get(chunk.cluster_id)

    db.commit()

    distribution = Counter(labels)
    valid_cluster_ids = [
        cluster_id for cluster_id in distribution.keys() if cluster_id >= 0
    ]

    return {
        "total_chunks": len(chunks),
        "total_clusters": len(valid_cluster_ids),
        "noise_chunks": distribution.get(-1, 0),
        "distribution": {str(k): v for k, v in distribution.items()},
        "cluster_labels": {str(k): v for k, v in label_map.items()},
        "quality": _quality_metrics(embeddings, labels),
    }


def _derive_cluster_labels(chunks) -> dict:
    stop_words = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is",
        "are", "be", "with", "that", "this", "it", "as", "at", "by", "from",
    }
    grouped_texts = defaultdict(list)
    for c in chunks:
        if c.cluster_id is None or c.cluster_id < 0:
            continue
        grouped_texts[c.cluster_id].append(c.text or "")

    labels = {}
    for cid, texts in grouped_texts.items():
        words = re.findall(r"[a-zA-Z]{3,}", " ".join(texts).lower())
        counts = Counter(w for w in words if w not in stop_words)
        top_terms = [w for w, _ in counts.most_common(3)]
        labels[cid] = " / ".join(top_terms) if top_terms else f"cluster-{cid}"
    return labels


def _quality_metrics(embeddings, labels) -> dict:
    try:
        from sklearn.metrics import silhouette_score
        valid_labels = [l for l in labels if l >= 0]
        unique = set(valid_labels)
        if len(unique) < 2:
            return {"silhouette": None, "reason": "not_enough_clusters"}
        return {"silhouette": float(silhouette_score(embeddings, labels))}
    except Exception:
        return {"silhouette": None, "reason": "metric_unavailable"}
