"""
Ingestion Routes
Handles video submission and async processing.
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, HttpUrl
from app.services.pipeline import process_video
from app.models.database import SessionLocal
from app.models.schemas import Job, Video

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


def _run_pipeline(url: str):
    """Background task wrapper for the pipeline."""
    try:
        result = process_video(url)
        _task_results[result["video_id"]] = result
    except Exception as e:
        _task_results[url] = {"status": "failed", "error": str(e)}


@router.post("/", response_model=None)
def ingest_video(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Submit a video URL for ingestion.
    Processing runs in the background.
    Returns immediately with a job_id to track progress.
    """
    try:
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
    background_tasks.add_task(_run_pipeline, request.url)
    return {
        "status": "queued",
        "message": "Video submitted for processing. "
                   "Use /videos endpoint to check status.",
        "url": request.url
    }