"""
Audio Extraction Service
Extracts audio from video, converts to mono 16kHz WAV for Whisper compatibility.
"""
from pydub import AudioSegment
import os
import logging

from app.config import AUDIO_DIR, AUDIO_SAMPLE_RATE, AUDIO_CHANNELS

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, video_id: str) -> str:
    """
    Extract audio from video file and convert to Whisper-compatible format.

    Args:
        video_path: Path to the downloaded video
        video_id: UUID of the video

    Returns:
        Path to the extracted WAV audio file
    """
    audio_path = os.path.join(AUDIO_DIR, f"{video_id}.wav")

    try:
        audio = AudioSegment.from_file(video_path)

        # Convert to mono, 16kHz for Whisper compatibility
        audio = audio.set_channels(AUDIO_CHANNELS)
        audio = audio.set_frame_rate(AUDIO_SAMPLE_RATE)

        audio.export(audio_path, format="wav")

        logger.info(f"Audio extracted: {audio_path} "
                     f"(mono, {AUDIO_SAMPLE_RATE}Hz, {len(audio)/1000:.1f}s)")

        return audio_path

    except Exception as e:
        logger.error(f"Audio extraction failed for {video_path}: {e}")
        raise Exception(f"Audio extraction failed: {str(e)}")