"""
Pipeline Orchestrator
Coordinates the full video ingestion pipeline:
  Download → Audio → Transcribe → OCR → Chunk → Embed → Store

Tracks progress in the database for each step.
"""
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from app.models.database import SessionLocal
from app.models.schemas import Video, Chunk, Job, VideoStatus
from app.services.downloader import download_video
from app.services.audio_extractor import extract_audio
from app.services.transcriber import transcribe_audio
from app.services.chunker import chunk_transcript, chunk_ocr_texts
from app.services.normalization import normalize_chunks_to_english
from app.services.embedder import generate_embeddings
from app.services.vector_store import add_embeddings
from app.services.clustering import assign_clusters
from app.services.categorizer import categorize_video

logger = logging.getLogger(__name__)


def process_video(
    url: str,
    video_id: Optional[str] = None,
    job_id: Optional[str] = None,
    user_id: str = "public",
) -> dict:
    """
    Full synchronous pipeline for a single video.

    Args:
        url: Video URL

    Returns:
        dict with video_id, status, and processing results
    """
    video_id = video_id or str(uuid.uuid4())
    job_id = job_id or str(uuid.uuid4())

    db = SessionLocal()
    video = None
    job = None

    try:
        # ── Create DB records ──────────────────────────────
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            video = Video(
                id=video_id,
                user_id=user_id,
                url=url,
                status=VideoStatus.PROCESSING,
            )
            db.add(video)
        else:
            video.user_id = user_id or video.user_id
            video.url = url
            video.status = VideoStatus.PROCESSING
            video.error_message = None

        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            job = Job(
                id=job_id,
                user_id=user_id,
                video_id=video_id,
                status=VideoStatus.PROCESSING,
                current_step="download"
            )
            db.add(job)
        else:
            job.status = VideoStatus.PROCESSING
            job.current_step = "download"
            job.error_message = None

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

        # ── Step 6: Normalize multilingual text ───────────
        _update_job(db, job, "normalization")
        all_chunks = normalize_chunks_to_english(all_chunks)

        # ── Step 7: Categorize video (non-fatal) ──────────
        try:
            _update_job(db, job, "categorization")
            video.category = categorize_video(
                title=video.title,
                chunk_texts=[c["text"] for c in all_chunks]
            )
            db.commit()
        except Exception as e:
            logger.warning(f"Categorization step failed (non-fatal): {e}")

        # ── Step 8: Generate Embeddings ───────────────────
        _update_job(db, job, "embedding")
        texts = [c["text"] for c in all_chunks]
        embeddings = generate_embeddings(texts)

        # ── Step 9: Store in FAISS ────────────────────────
        _update_job(db, job, "indexing")
        chunk_ids = [c["id"] for c in all_chunks]
        faiss_positions = add_embeddings(embeddings, chunk_ids)

        # ── Step 10: Save chunks to DB ────────────────────
        _update_job(db, job, "saving")
        for i, chunk_data in enumerate(all_chunks):
            db_chunk = Chunk(
                id=chunk_data["id"],
                user_id=user_id,
                video_id=video_id,
                text=chunk_data["text"],
                start_time=chunk_data["start"],
                end_time=chunk_data["end"],
                chunk_type=chunk_data["type"],
                faiss_index=faiss_positions[i] if i < len(faiss_positions) else None
            )
            db.add(db_chunk)
        db.commit()

        # ── Step 11: Clustering (non-fatal) ───────────────
        cluster_summary = None
        try:
            _update_job(db, job, "clustering")
            cluster_summary = assign_clusters(db)
        except Exception as e:
            logger.warning(f"Clustering step failed (non-fatal): {e}")

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
            "language": transcript_result.get("language", "unknown"),
            "clusters": cluster_summary
        }

    except Exception as e:
        logger.error(f"Pipeline failed for video {video_id}: {e}")

        # Update DB with failure
        try:
            if video:
                video.status = VideoStatus.FAILED
                video.error_message = str(e)
            if job:
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
