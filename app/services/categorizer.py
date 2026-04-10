"""
Video Categorization Service
Classifies a video into a single category using Ollama LLM.
"""
import logging
import re
from typing import List, Optional

import requests

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, VIDEO_CATEGORIES

logger = logging.getLogger(__name__)


def categorize_video(title: Optional[str], chunk_texts: List[str]) -> Optional[str]:
    """
    Classify video content into one category label.
    Returns None if categorization is unavailable.
    """
    if not chunk_texts:
        return None

    sampled_chunks = chunk_texts[:20]
    context = "\n".join(f"- {text[:300]}" for text in sampled_chunks if text)
    if not context.strip():
        return None

    categories_csv = ", ".join(VIDEO_CATEGORIES)
    prompt = f"""Classify this short video into exactly one category from this list:
{categories_csv}

Video title: {title or "Unknown"}
Video content:
{context}

Rules:
- Return only one category from the list.
- Output just the category text with no explanation.
"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=30,
        )
        if response.status_code != 200:
            logger.warning(f"Ollama categorization returned {response.status_code}")
            return None

        raw_output = response.json().get("response", "")
        category = _normalize_category(raw_output)
        if category:
            logger.info(f"Video categorized as '{category}'")
            return category

        logger.warning(f"Unrecognized category output: {raw_output!r}")
        return "other"
    except requests.ConnectionError:
        logger.warning("Cannot connect to Ollama for categorization")
        return None
    except Exception as e:
        logger.warning(f"Video categorization failed: {e}")
        return None


def _normalize_category(raw_output: str) -> Optional[str]:
    if not raw_output:
        return None

    cleaned = raw_output.strip().lower()
    cleaned = re.sub(r"[^a-z\s-]", "", cleaned)
    cleaned = cleaned.replace("-", " ").strip()

    categories_normalized = {c.replace("-", " ").lower(): c for c in VIDEO_CATEGORIES}
    if cleaned in categories_normalized:
        return categories_normalized[cleaned]

    for normalized, canonical in categories_normalized.items():
        if normalized in cleaned:
            return canonical

    return None
