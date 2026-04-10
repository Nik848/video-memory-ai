"""
Chunking Engine
Splits transcript segments into manageable chunks with metadata.
Supports both speech and OCR chunks.
"""
import uuid
import json
import os
import logging

from app.config import MAX_CHUNK_WORDS, CHUNK_DIR

logger = logging.getLogger(__name__)


def chunk_transcript(segments: list, video_id: str,
                     chunk_type: str = "speech") -> list:
    """
    Split transcript segments into chunks with metadata.

    Args:
        segments: List of Whisper segments [{"text": ..., "start": ..., "end": ...}]
        video_id: UUID of the source video
        chunk_type: "speech", "ocr", or "mixed"

    Returns:
        List of chunk dicts with id, text, start, end, video_id, type
    """
    chunks = []
    current_text = ""
    start_time = None
    end_time = None

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        words = text.split()

        if start_time is None:
            start_time = seg["start"]

        if len(current_text.split()) + len(words) <= MAX_CHUNK_WORDS:
            current_text += " " + text
            end_time = seg["end"]
        else:
            # Flush current chunk
            if current_text.strip():
                chunks.append(_make_chunk(
                    current_text.strip(), start_time, end_time,
                    video_id, chunk_type
                ))

            # Start new chunk
            current_text = text
            start_time = seg["start"]
            end_time = seg["end"]

    # Final chunk
    if current_text.strip():
        chunks.append(_make_chunk(
            current_text.strip(), start_time, end_time,
            video_id, chunk_type
        ))

    # Save to disk
    chunk_path = os.path.join(CHUNK_DIR, f"{video_id}.json")
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Chunked: {len(chunks)} chunks for video {video_id}")

    return chunks


def chunk_ocr_texts(ocr_results: list, video_id: str) -> list:
    """
    Convert OCR results into chunks.

    Args:
        ocr_results: List of {"text": ..., "timestamp": ...}
        video_id: UUID of the source video

    Returns:
        List of chunk dicts
    """
    # Convert OCR results to segment-like format
    segments = []
    for result in ocr_results:
        if result.get("text", "").strip():
            ts = result.get("timestamp", 0)
            segments.append({
                "text": result["text"],
                "start": ts,
                "end": ts + 1  # OCR doesn't have exact end time
            })

    return chunk_transcript(segments, video_id, chunk_type="ocr")


def _make_chunk(text: str, start: float, end: float,
                video_id: str, chunk_type: str) -> dict:
    """Create a chunk dict with all required metadata."""
    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "start": round(start, 2),
        "end": round(end, 2),
        "video_id": video_id,
        "type": chunk_type,
        "cluster_id": None
    }