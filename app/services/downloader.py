"""
Video Downloader Service
Downloads video from URL using yt-dlp, returns path and metadata.
"""
import yt_dlp
import os
import logging

from app.config import DOWNLOAD_DIR, COOKIES_PATH

logger = logging.getLogger(__name__)


def download_video(url: str, video_id: str) -> dict:
    """
    Download video and return metadata.

    Args:
        url: Video URL (Instagram Reel, YouTube Short, etc.)
        video_id: UUID assigned to this video

    Returns:
        dict with keys: video_path, title, duration
    """
    output_path = os.path.join(DOWNLOAD_DIR, f"{video_id}.mp4")

    ydl_opts = {
        'format': 'mp4/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
    }

    # Only use cookies if the file exists and is non-empty
    if os.path.exists(COOKIES_PATH) and os.path.getsize(COOKIES_PATH) > 0:
        ydl_opts['cookiefile'] = COOKIES_PATH

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        title = info.get("title", "Untitled")
        duration = info.get("duration", 0)

        logger.info(f"Downloaded: {title} ({duration}s) -> {output_path}")

        return {
            "video_path": output_path,
            "title": title,
            "duration": duration
        }

    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        raise Exception(f"Download failed: {str(e)}")