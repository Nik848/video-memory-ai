"""
Transcription Service (ASR)
Uses OpenAI Whisper to transcribe audio with timestamps.
"""
import whisper
import json
import os
import logging

from app.config import WHISPER_MODEL, TRANSCRIPT_DIR

logger = logging.getLogger(__name__)

# Load model once at module level (expensive to reload)
_model = None


def _get_model():
    """Lazy-load whisper model to avoid loading at import time."""
    global _model
    if _model is None:
        logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def transcribe_audio(audio_path: str, video_id: str) -> dict:
    """
    Transcribe audio to text with timestamps.

    Args:
        audio_path: Path to the WAV audio file
        video_id: UUID of the video

    Returns:
        dict with keys: transcript_path, segments, full_text
    """
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{video_id}.json")

    try:
        model = _get_model()
        result = model.transcribe(audio_path)

        # Save full result
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        segments = result.get("segments", [])
        full_text = result.get("text", "").strip()

        logger.info(f"Transcribed: {len(segments)} segments, "
                     f"{len(full_text)} chars -> {transcript_path}")

        return {
            "transcript_path": transcript_path,
            "segments": segments,
            "full_text": full_text,
            "language": result.get("language", "unknown")
        }

    except Exception as e:
        logger.error(f"Transcription failed for {audio_path}: {e}")
        raise Exception(f"Transcription failed: {str(e)}")