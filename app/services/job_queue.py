"""
Database-backed ingestion queue worker with retry/backoff/dead-letter handling.
"""
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from app.config import (
    JOB_BASE_BACKOFF_SECONDS,
    JOB_MAX_ATTEMPTS,
    JOB_MAX_BACKOFF_SECONDS,
    QUEUE_WORKER_POLL_SECONDS,
)
from app.models.database import SessionLocal
from app.models.schemas import Job, Video, VideoStatus
from app.services.pipeline import process_video

logger = logging.getLogger(__name__)

_worker_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def start_worker() -> None:
    """Start singleton background queue worker."""
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return
    _stop_event.clear()
    _worker_thread = threading.Thread(target=_worker_loop, daemon=True, name="job-queue-worker")
    _worker_thread.start()
    logger.info("Queue worker started")


def stop_worker() -> None:
    """Stop queue worker gracefully."""
    _stop_event.set()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _next_backoff_seconds(attempt_number: int) -> int:
    backoff = JOB_BASE_BACKOFF_SECONDS * (2 ** max(0, attempt_number - 1))
    return min(backoff, JOB_MAX_BACKOFF_SECONDS)


def _worker_loop() -> None:
    while not _stop_event.is_set():
        did_work = _process_next_job()
        if not did_work:
            time.sleep(QUEUE_WORKER_POLL_SECONDS)


def _process_next_job() -> bool:
    db = SessionLocal()
    try:
        now = _utcnow()
        job = (
            db.query(Job)
            .filter(
                Job.status.in_([VideoStatus.QUEUED, VideoStatus.FAILED]),
                ((Job.next_retry_at.is_(None)) | (Job.next_retry_at <= now)),
                Job.dead_lettered_at.is_(None),
            )
            .order_by(Job.created_at.asc())
            .first()
        )
        if not job:
            return False

        video = db.query(Video).filter(Video.id == job.video_id).first()
        if not video:
            job.status = VideoStatus.FAILED
            job.error_message = "Related video not found"
            db.commit()
            return True

        if (job.attempt_count or 0) >= (job.max_attempts or JOB_MAX_ATTEMPTS):
            _mark_dead_letter(db, job, video, "Exceeded max attempts")
            return True

        job.status = VideoStatus.PROCESSING
        job.current_step = "processing"
        job.attempt_count = (job.attempt_count or 0) + 1
        job.updated_at = now
        video.status = VideoStatus.PROCESSING
        db.commit()
        job_id = job.id
        video_id = video.id
        video_url = video.url
        user_id = job.user_id or "public"
    finally:
        db.close()

    try:
        process_video(video_url, video_id=video_id, job_id=job_id, user_id=user_id)
    except Exception as exc:
        _handle_job_failure(job_id, str(exc))
    return True


def _handle_job_failure(job_id: str, error_message: str) -> None:
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return
        video = db.query(Video).filter(Video.id == job.video_id).first()
        attempts = job.attempt_count or 0
        max_attempts = job.max_attempts or JOB_MAX_ATTEMPTS

        if attempts >= max_attempts:
            _mark_dead_letter(db, job, video, error_message)
            return

        backoff = _next_backoff_seconds(attempts)
        job.status = VideoStatus.FAILED
        job.error_message = error_message
        job.current_step = "retry_scheduled"
        job.next_retry_at = _utcnow() + timedelta(seconds=backoff)
        if video:
            video.status = VideoStatus.FAILED
            video.error_message = error_message
        db.commit()
    finally:
        db.close()


def _mark_dead_letter(db, job: Job, video: Optional[Video], error_message: str) -> None:
    job.status = VideoStatus.FAILED
    job.error_message = error_message
    job.current_step = "dead_lettered"
    job.dead_lettered_at = _utcnow()
    if video:
        video.status = VideoStatus.FAILED
        video.error_message = error_message
    db.commit()
