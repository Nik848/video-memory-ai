"""
Video Management Routes
List videos, get status, view chunks, explore clusters.
"""
from collections import defaultdict

from fastapi import APIRouter, HTTPException
from app.models.database import SessionLocal
from app.models.schemas import Video, Chunk, Job
from app.services.vector_store import get_total_vectors
from app.services.clustering import assign_clusters

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.get("/")
def list_videos():
    """List all ingested videos with their status."""
    db = SessionLocal()
    try:
        videos = db.query(Video).order_by(Video.created_at.desc()).all()
        return {
            "total": len(videos),
            "videos": [
                {
                    "id": v.id,
                    "url": v.url,
                    "title": v.title,
                    "duration": v.duration,
                    "status": v.status,
                    "created_at": str(v.created_at),
                    "chunk_count": db.query(Chunk).filter(
                        Chunk.video_id == v.id
                    ).count()
                }
                for v in videos
            ]
        }
    finally:
        db.close()


@router.get("/{video_id}")
def get_video(video_id: str):
    """Get details of a specific video including its chunks."""
    db = SessionLocal()
    try:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        chunks = db.query(Chunk).filter(
            Chunk.video_id == video_id
        ).order_by(Chunk.start_time).all()

        return {
            "id": video.id,
            "url": video.url,
            "title": video.title,
            "duration": video.duration,
            "status": video.status,
            "error_message": video.error_message,
            "created_at": str(video.created_at),
            "chunks": [
                {
                    "id": c.id,
                    "text": c.text,
                    "start": c.start_time,
                    "end": c.end_time,
                    "type": c.chunk_type,
                    "cluster_id": c.cluster_id
                }
                for c in chunks
            ]
        }
    finally:
        db.close()


@router.get("/{video_id}/chunks")
def get_video_chunks(video_id: str):
    """Get all chunks for a specific video."""
    db = SessionLocal()
    try:
        chunks = db.query(Chunk).filter(
            Chunk.video_id == video_id
        ).order_by(Chunk.start_time).all()

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No chunks found for this video"
            )

        return {
            "video_id": video_id,
            "total": len(chunks),
            "chunks": [
                {
                    "id": c.id,
                    "text": c.text,
                    "start": c.start_time,
                    "end": c.end_time,
                    "type": c.chunk_type,
                    "cluster_id": c.cluster_id
                }
                for c in chunks
            ]
        }
    finally:
        db.close()


@router.get("/stats")
def get_stats():
    """Get system statistics."""
    db = SessionLocal()
    try:
        total_videos = db.query(Video).count()
        completed_videos = db.query(Video).filter(
            Video.status == "completed"
        ).count()
        total_chunks = db.query(Chunk).count()
        total_vectors = get_total_vectors()

        return {
            "total_videos": total_videos,
            "completed_videos": completed_videos,
            "total_chunks": total_chunks,
            "total_vectors": total_vectors
        }
    finally:
        db.close()


@router.get("/clusters")
def get_clusters(
    recompute: bool = False,
    min_cluster_size: int = 2,
    kmeans_clusters: int = 0,
):
    """Explore chunk clusters with optional recomputation."""
    db = SessionLocal()
    try:
        clustering_summary = None
        if recompute:
            clustering_summary = assign_clusters(
                db,
                min_cluster_size=min_cluster_size,
                kmeans_clusters=kmeans_clusters or None,
            )

        chunks = db.query(Chunk).filter(Chunk.cluster_id.isnot(None)).all()
        if not chunks:
            return {
                "total_clusters": 0,
                "clusters": [],
                "recomputed": recompute,
                "summary": clustering_summary
            }

        grouped = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.cluster_id].append(chunk)

        clusters = []
        for cluster_id, cluster_chunks in sorted(grouped.items(), key=lambda x: x[0]):
            videos = {c.video_id for c in cluster_chunks}
            clusters.append({
                "cluster_id": cluster_id,
                "chunk_count": len(cluster_chunks),
                "video_count": len(videos),
                "samples": [
                    {
                        "chunk_id": c.id,
                        "video_id": c.video_id,
                        "start": c.start_time,
                        "end": c.end_time,
                        "text": c.text,
                        "type": c.chunk_type
                    }
                    for c in cluster_chunks[:3]
                ]
            })

        return {
            "total_clusters": len(clusters),
            "clusters": clusters,
            "recomputed": recompute,
            "summary": clustering_summary
        }
    finally:
        db.close()
