"""
Ingestion Routes
Handles video submission and async processing.
"""
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from app.services.pipeline import process_video
from app.models.database import SessionLocal
from app.models.schemas import Job, Video
from app.models.schemas import VideoStatus

router = APIRouter(prefix="/ingest", tags=["Ingestion"])


class IngestRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    video_id: str
    job_id: str
    status: str
    message: str


# Store for background task results (in production, use Redis)
_task_results = {}


def _run_pipeline(url: str, video_id: str = None, job_id: str = None):
    """Background task wrapper for the pipeline."""
    try:
        result = process_video(url, video_id=video_id, job_id=job_id)
        _task_results[result["job_id"]] = result
    except Exception as e:
        _task_results[job_id or url] = {"status": "failed", "error": str(e)}


@router.post("/", response_model=None)
def ingest_video(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Submit a video URL for ingestion.
    Processing runs in the background.
    Returns immediately with a job_id to track progress.
    """
    try:
        if not request.url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="Invalid URL format")

        # Run synchronously for now (can switch to background)
        result = process_video(request.url)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/async")
def ingest_video_async(request: IngestRequest,
                       background_tasks: BackgroundTasks):
    """
    Submit a video URL for async ingestion.
    Returns immediately, use /status/{job_id} to track.
    """
    db = SessionLocal()
    try:
        if not request.url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="Invalid URL format")

        video_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())

        db_video = Video(id=video_id, url=request.url, status=VideoStatus.QUEUED)
        db_job = Job(
            id=job_id,
            video_id=video_id,
            status=VideoStatus.QUEUED,
            current_step="queued"
        )
        db.add(db_video)
        db.add(db_job)
        db.commit()
    finally:
        db.close()

    background_tasks.add_task(_run_pipeline, request.url, video_id, job_id)
    return {
        "status": "queued",
        "message": "Video submitted for processing.",
        "video_id": video_id,
        "job_id": job_id
    }


@router.get("/status/{job_id}")
def get_job_status(job_id: str):
    """Return detailed status for a processing job."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        video = db.query(Video).filter(Video.id == job.video_id).first()
        return {
            "job_id": job.id,
            "video_id": job.video_id,
            "video_status": video.status if video else None,
            "job_status": job.status,
            "current_step": job.current_step,
            "error_message": job.error_message,
            "created_at": str(job.created_at),
            "updated_at": str(job.updated_at),
        }
    finally:
        db.close()


@router.post("/retry/{job_id}")
def retry_failed_job(job_id: str, background_tasks: BackgroundTasks):
    """Retry a failed ingestion job."""
    db = SessionLocal()
    try:
        existing_job = db.query(Job).filter(Job.id == job_id).first()
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
            video_id=video.id,
            status=VideoStatus.QUEUED,
            current_step="queued",
        )
        video.status = VideoStatus.QUEUED
        video.error_message = None
        db.add(new_job)
        db.commit()

        background_tasks.add_task(_run_pipeline, video.url, video.id, new_job_id)
        return {
            "status": "queued",
            "video_id": video.id,
            "job_id": new_job_id,
            "message": "Retry job queued successfully."
        }
    finally:
        db.close()
