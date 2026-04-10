"""
OCR Module
Extracts on-screen text from video frames.
Pipeline: Video → Frame Sampling → OCR → Text Aggregation
"""
import cv2
import os
import logging
from typing import List, Dict

from app.config import FRAMES_DIR, OCR_FRAME_INTERVAL, OCR_LANGUAGES

logger = logging.getLogger(__name__)

# Lazy-loaded OCR reader
_reader = None
_ocr_backend = None


def _get_reader():
    """Lazy-load OCR reader with PaddleOCR preferred, EasyOCR fallback."""
    global _reader, _ocr_backend
    if _reader is None:
        try:
            from paddleocr import PaddleOCR
            logger.info("Loading PaddleOCR backend")
            _reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            _ocr_backend = "paddleocr"
        except Exception as paddle_error:
            logger.warning(f"PaddleOCR unavailable, falling back to EasyOCR: {paddle_error}")
            import easyocr
            logger.info(f"Loading EasyOCR with languages: {OCR_LANGUAGES}")
            _reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
            _ocr_backend = "easyocr"
    return _reader, _ocr_backend


def extract_frames(video_path: str, video_id: str,
                   interval: int = None) -> List[Dict]:
    """
    Extract frames from video at regular intervals.

    Args:
        video_path: Path to video file
        video_id: UUID of the video
        interval: Seconds between frame captures (default from config)

    Returns:
        List of {"frame_path": ..., "timestamp": ...}
    """
    if interval is None:
        interval = OCR_FRAME_INTERVAL

    frame_dir = os.path.join(FRAMES_DIR, video_id)
    os.makedirs(frame_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval)

    frames = []
    frame_num = 0

    while frame_num < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_num / fps
        frame_path = os.path.join(frame_dir, f"frame_{frame_num:06d}.jpg")
        cv2.imwrite(frame_path, frame)

        frames.append({
            "frame_path": frame_path,
            "timestamp": round(timestamp, 2)
        })

        frame_num += frame_interval

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from video {video_id}")

    return frames


def ocr_frame(frame_path: str) -> str:
    """
    Run OCR on a single frame.

    Args:
        frame_path: Path to frame image

    Returns:
        Extracted text string
    """
    reader, backend = _get_reader()

    try:
        if backend == "paddleocr":
            results = reader.ocr(frame_path, cls=True) or []
            lines = results[0] if results else []
            texts = [line[1][0] for line in lines if line[1][1] > 0.3]
        else:
            results = reader.readtext(frame_path)
            # results is list of (bbox, text, confidence)
            texts = [text for _, text, conf in results if conf > 0.3]
        return " ".join(texts).strip()
    except Exception as e:
        logger.warning(f"OCR failed on {frame_path}: {e}")
        return ""


def extract_text_from_video(video_path: str, video_id: str) -> List[Dict]:
    """
    Full OCR pipeline: extract frames → OCR → aggregate text.

    Args:
        video_path: Path to video file
        video_id: UUID of the video

    Returns:
        List of {"text": ..., "timestamp": ...} with deduplicated text
    """
    frames = extract_frames(video_path, video_id)

    if not frames:
        return []

    ocr_results = []
    seen_texts = set()

    for frame_info in frames:
        text = ocr_frame(frame_info["frame_path"])

        if text and text.lower() not in seen_texts:
            seen_texts.add(text.lower())
            ocr_results.append({
                "text": text,
                "timestamp": frame_info["timestamp"]
            })

    logger.info(f"OCR extracted {len(ocr_results)} unique text segments "
                f"from video {video_id}")

    return ocr_results
