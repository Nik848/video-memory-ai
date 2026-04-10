"""
Pipeline Orchestrator
Coordinates the full video ingestion pipeline:
  Download → Audio → Transcribe → OCR → Chunk → Embed → Store

Tracks progress in the database for each step.
"""
import uuid
import logging
from datetime import datetime, timezone

from app.models.database import SessionLocal
from app.models.schemas import Video, Chunk, Job, VideoStatus
from app.services.downloader import download_video
from app.services.audio_extractor import extract_audio
from app.services.transcriber import transcribe_audio
from app.services.chunker import chunk_transcript, chunk_ocr_texts
from app.services.embedder import generate_embeddings
from app.services.vector_store import add_embeddings

logger = logging.getLogger(__name__)


def process_video(url: str) -> dict:
    """
    Full synchronous pipeline for a single video.

    Args:
        url: Video URL

    Returns:
        dict with video_id, status, and processing results
    """
    video_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    db = SessionLocal()

    try:
        # ── Create DB records ──────────────────────────────
        video = Video(id=video_id, url=url, status=VideoStatus.PROCESSING)
        job = Job(id=job_id, video_id=video_id,
                  status=VideoStatus.PROCESSING, current_step="download")
        db.add(video)
        db.add(job)
        db.commit()

        # ── Step 1: Download ──────────────────────────────
        _update_job(db, job, "download")
        download_result = download_video(url, video_id)
        video.video_path = download_result["video_path"]
        video.title = download_result["title"]
        video.duration = download_result["duration"]
        db.commit()

        # ── Step 2: Extract Audio ─────────────────────────
        _update_job(db, job, "audio_extraction")
        audio_path = extract_audio(download_result["video_path"], video_id)
        video.audio_path = audio_path
        db.commit()

        # ── Step 3: Transcribe (ASR) ──────────────────────
        _update_job(db, job, "transcription")
        transcript_result = transcribe_audio(audio_path, video_id)
        video.transcript_path = transcript_result["transcript_path"]
        db.commit()

        # ── Step 4: OCR (optional, non-blocking) ──────────
        ocr_chunks = []
        try:
            _update_job(db, job, "ocr")
            from app.services.ocr_engine import extract_text_from_video
            ocr_results = extract_text_from_video(
                download_result["video_path"], video_id
            )
            if ocr_results:
                ocr_chunks = chunk_ocr_texts(ocr_results, video_id)
                logger.info(f"OCR produced {len(ocr_chunks)} chunks")
        except Exception as e:
            logger.warning(f"OCR step failed (non-fatal): {e}")

        # ── Step 5: Chunk transcript ──────────────────────
        _update_job(db, job, "chunking")
        speech_chunks = chunk_transcript(
            transcript_result["segments"], video_id, chunk_type="speech"
        )

        # Merge speech + OCR chunks
        all_chunks = speech_chunks + ocr_chunks

        if not all_chunks:
            logger.warning(f"No chunks produced for video {video_id}")
            video.status = VideoStatus.COMPLETED
            job.status = VideoStatus.COMPLETED
            job.current_step = "completed"
            db.commit()
            return {
                "video_id": video_id,
                "job_id": job_id,
                "status": "completed",
                "message": "No text content found in video"
            }

        # ── Step 6: Generate Embeddings ───────────────────
        _update_job(db, job, "embedding")
        texts = [c["text"] for c in all_chunks]
        embeddings = generate_embeddings(texts)

        # ── Step 7: Store in FAISS ────────────────────────
        _update_job(db, job, "indexing")
        chunk_ids = [c["id"] for c in all_chunks]
        faiss_positions = add_embeddings(embeddings, chunk_ids)

        # ── Step 8: Save chunks to DB ─────────────────────
        _update_job(db, job, "saving")
        for i, chunk_data in enumerate(all_chunks):
            db_chunk = Chunk(
                id=chunk_data["id"],
                video_id=video_id,
                text=chunk_data["text"],
                start_time=chunk_data["start"],
                end_time=chunk_data["end"],
                chunk_type=chunk_data["type"],
                faiss_index=faiss_positions[i] if i < len(faiss_positions) else None
            )
            db.add(db_chunk)

        # ── Mark completed ────────────────────────────────
        video.status = VideoStatus.COMPLETED
        job.status = VideoStatus.COMPLETED
        job.current_step = "completed"
        db.commit()

        logger.info(f"Pipeline completed for video {video_id}: "
                     f"{len(all_chunks)} chunks indexed")

        return {
            "video_id": video_id,
            "job_id": job_id,
            "status": "completed",
            "title": video.title,
            "duration": video.duration,
            "speech_chunks": len(speech_chunks),
            "ocr_chunks": len(ocr_chunks),
            "total_chunks": len(all_chunks),
            "language": transcript_result.get("language", "unknown")
        }

    except Exception as e:
        logger.error(f"Pipeline failed for video {video_id}: {e}")

        # Update DB with failure
        try:
            video.status = VideoStatus.FAILED
            video.error_message = str(e)
            job.status = VideoStatus.FAILED
            job.error_message = str(e)
            db.commit()
        except Exception:
            db.rollback()

        raise

    finally:
        db.close()


def _update_job(db, job: Job, step: str):
    """Update the current processing step for a job."""
    job.current_step = step
    job.updated_at = datetime.now(timezone.utc)
    db.commit()
    logger.info(f"Job {job.id}: step={step}")
