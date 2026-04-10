"""
Ingestion Routes
Handles video submission and async processing.
"""
import uuid
import hashlib
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.services.pipeline import process_video
from app.models.database import SessionLocal
from app.models.schemas import Job, Video
from app.models.schemas import VideoStatus
from app.services.auth import require_api_key, get_user_id
from app.services.job_queue import start_worker
from app.config import JOB_MAX_ATTEMPTS

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class IngestRequest(BaseModel):
    url: str
    force_reingest: bool = False


class IngestResponse(BaseModel):
    video_id: str
    job_id: str
    status: str
    message: str


def _is_valid_url(url: str) -> bool:
    """Basic URL validation with scheme + netloc checks."""
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _source_fingerprint(url: str, user_id: str) -> str:
    normalized = (url or "").strip().lower()
    return hashlib.sha256(f"{user_id}:{normalized}".encode("utf-8")).hexdigest()


@router.post("/", response_model=None, dependencies=[Depends(require_api_key)])
def ingest_video(request: IngestRequest, user_id: str = Depends(get_user_id)):
    """
    Submit a video URL for ingestion.
    Processing runs in the background.
    Returns immediately with a job_id to track progress.
    """
    try:
        if not _is_valid_url(request.url):
            raise HTTPException(status_code=400, detail="Invalid URL format")

        # Synchronous processing path for small/manual use.
        result = process_video(request.url, user_id=user_id)
        return result

    except Exception:
        raise HTTPException(status_code=500, detail="Ingestion failed")


@router.post("/async", dependencies=[Depends(require_api_key)])
def ingest_video_async(request: IngestRequest, user_id: str = Depends(get_user_id)):
    """
    Submit a video URL for async ingestion.
    Returns immediately, use /status/{job_id} to track.
    """
    db = SessionLocal()
    try:
        if not _is_valid_url(request.url):
            raise HTTPException(status_code=400, detail="Invalid URL format")

        fingerprint = _source_fingerprint(request.url, user_id)
        existing_video = db.query(Video).filter(
            Video.user_id == user_id,
            Video.source_fingerprint == fingerprint,
            Video.status.in_([VideoStatus.QUEUED, VideoStatus.PROCESSING, VideoStatus.COMPLETED])
        ).order_by(Video.created_at.desc()).first()
        if existing_video and not request.force_reingest:
            existing_job = db.query(Job).filter(
                Job.user_id == user_id,
                Job.video_id == existing_video.id
            ).order_by(Job.created_at.desc()).first()
            return {
                "status": "deduplicated",
                "message": "Duplicate URL detected for this user. Returning existing job/video.",
                "video_id": existing_video.id,
                "job_id": existing_job.id if existing_job else None
            }

        video_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        db_video = Video(
            id=video_id,
            user_id=user_id,
            url=request.url,
            source_fingerprint=fingerprint,
            status=VideoStatus.QUEUED
        )
        db_job = Job(
            id=job_id,
            user_id=user_id,
            video_id=video_id,
            status=VideoStatus.QUEUED,
            current_step="queued",
            max_attempts=JOB_MAX_ATTEMPTS,
        )
        db.add(db_video)
        db.add(db_job)
        db.commit()
    finally:
        db.close()

    start_worker()
    return {
        "status": "queued",
        "message": "Video submitted for processing.",
        "video_id": video_id,
        "job_id": job_id
    }


@router.get("/status/{job_id}", dependencies=[Depends(require_api_key)])
def get_job_status(job_id: str, user_id: str = Depends(get_user_id)):
    """Return detailed status for a processing job."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id, Job.user_id == user_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        video = db.query(Video).filter(Video.id == job.video_id).first()
        return {
            "job_id": job.id,
            "video_id": job.video_id,
            "video_status": video.status if video else None,
            "job_status": job.status,
            "current_step": job.current_step,
            "attempt_count": job.attempt_count,
            "max_attempts": job.max_attempts,
            "next_retry_at": str(job.next_retry_at) if job.next_retry_at else None,
            "dead_lettered_at": str(job.dead_lettered_at) if job.dead_lettered_at else None,
            "error_message": job.error_message,
            "created_at": str(job.created_at),
            "updated_at": str(job.updated_at),
        }
    finally:
        db.close()


@router.post("/retry/{job_id}", dependencies=[Depends(require_api_key)])
def retry_failed_job(job_id: str, user_id: str = Depends(get_user_id)):
    """Retry a failed ingestion job."""
    db = SessionLocal()
    try:
        existing_job = db.query(Job).filter(
            Job.id == job_id,
            Job.user_id == user_id
        ).first()
        if not existing_job:
            raise HTTPException(status_code=404, detail="Job not found")
        if existing_job.status != VideoStatus.FAILED:
            raise HTTPException(
                status_code=400,
                detail="Only failed jobs can be retried"
            )

        video = db.query(Video).filter(Video.id == existing_job.video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="Related video not found")

        new_job_id = str(uuid.uuid4())
        new_job = Job(
            id=new_job_id,
            user_id=user_id,
            video_id=video.id,
            status=VideoStatus.QUEUED,
            current_step="queued",
            max_attempts=existing_job.max_attempts or JOB_MAX_ATTEMPTS,
        )
        video.status = VideoStatus.QUEUED
        video.error_message = None
        db.add(new_job)
        db.commit()

        start_worker()
        return {
            "status": "queued",
            "video_id": video.id,
            "job_id": new_job_id,
            "message": "Retry job queued successfully."
        }
    finally:
        db.close()
